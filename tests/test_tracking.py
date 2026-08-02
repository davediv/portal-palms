"""One-Euro filtering and the lock-free handoff between the two threads."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.tracking import HandState, OneEuroFilter, TripleBuffer

F32 = np.float32
SHAPE = (C.NUM_HANDS, C.NUM_LANDMARKS, 3)


def valid(both: bool = True) -> NDArray[np.bool_]:
    v = np.zeros(C.NUM_HANDS, dtype=bool)
    v[0] = True
    v[1] = both
    return v


# --------------------------------------------------------------------------- one euro


def test_filter_snaps_to_the_first_sample() -> None:
    f = OneEuroFilter(SHAPE)
    x = np.full(SHAPE, 0.7, dtype=F32)
    out = np.zeros(SHAPE, dtype=F32)
    f.filter(x, 1 / 30, out, valid())
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_filter_removes_jitter_from_a_still_hand() -> None:
    """Raw landmarks jitter badly; a still hand must come out quiet."""
    rng = np.random.default_rng(3)
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    base = np.full(SHAPE, 0.5, dtype=F32)
    noisy_spread = 0.0
    filtered_spread = 0.0
    for i in range(200):
        noise = rng.normal(0.0, 0.004, SHAPE).astype(F32)
        x = base + noise
        f.filter(x, 1 / 30, out, valid())
        if i > 60:
            noisy_spread += float(np.abs(x - 0.5).mean())
            filtered_spread += float(np.abs(out - 0.5).mean())
    assert filtered_spread < noisy_spread * 0.45


def test_filter_keeps_up_with_fast_motion() -> None:
    """The beta term exists so a fast hand does not lag. Check it actually does."""
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    x = np.zeros(SHAPE, dtype=F32)
    for i in range(40):
        x[:] = i * 0.02
        f.filter(x, 1 / 30, out, valid())
    # 0.03 of frame width is ~30px at 1080: about one frame of travel at this
    # speed. Without the pixel-space beta conversion this sits above 0.09.
    lag = float(np.abs(out - x).max())
    assert lag < 0.03, f"too much lag on a fast ramp: {lag}"


def test_filter_snaps_again_after_a_hand_is_absent() -> None:
    """A hand re-entering frame must not ease in from where it used to be."""
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    here = np.full(SHAPE, 0.2, dtype=F32)
    for _ in range(30):
        f.filter(here, 1 / 30, out, valid())
    np.testing.assert_allclose(out, 0.2, atol=1e-3)

    absent = valid(both=False)
    for _ in range(10):
        f.filter(np.zeros(SHAPE, dtype=F32), 1 / 30, out, absent)

    back = np.full(SHAPE, 0.8, dtype=F32)
    f.filter(back, 1 / 30, out, valid())
    assert out[1].mean() == pytest.approx(0.8, abs=1e-4), "hand 1 should have snapped"


def test_filter_never_allocates_after_init() -> None:
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    x = np.full(SHAPE, 0.3, dtype=F32)
    f.filter(x, 1 / 30, out, valid())
    before = [id(getattr(f, s)) for s in OneEuroFilter.__slots__]
    for i in range(100):
        x[:] = 0.3 + i * 0.001
        f.filter(x, 1 / 30, out, valid())
    after = [id(getattr(f, s)) for s in OneEuroFilter.__slots__]
    assert before == after, "filter state arrays were replaced, not written in place"


def test_filter_output_stays_float32() -> None:
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    f.filter(np.full(SHAPE, 0.4, dtype=F32), 1 / 30, out, valid())
    assert out.dtype == np.float32


def test_filter_tolerates_a_zero_timestep() -> None:
    f = OneEuroFilter(SHAPE)
    out = np.zeros(SHAPE, dtype=F32)
    f.filter(np.full(SHAPE, 0.4, dtype=F32), 0.0, out, valid())
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- handoff


def test_triple_buffer_never_hands_out_the_slot_being_written() -> None:
    b = TripleBuffer(8, 8)
    for _ in range(50):
        w = b.write_slot
        b.publish()
        r = b.read_slot
        b.acquire()
        assert b.write_slot is not b.read_slot
        assert w is not b.write_slot or r is not b.read_slot


def test_triple_buffer_sequence_advances_only_on_publish() -> None:
    b = TripleBuffer(8, 8)
    first = b.acquire()
    assert b.acquire() == first
    b.publish()
    assert b.acquire() == first + 1


def test_triple_buffer_survives_concurrent_writer_and_reader() -> None:
    """The real invariant: no torn reads and no exceptions under contention."""
    b = TripleBuffer(4, 4)
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            n = 0
            while not stop.is_set():
                slot = b.write_slot
                slot.landmarks[:] = n % 100
                slot.t = float(n)
                b.publish()
                n += 1
        except BaseException as exc:
            errors.append(exc)

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    try:
        for _ in range(4000):
            b.acquire()
            slot = b.read_slot
            v = slot.landmarks
            # Every element of a published slot must carry the same value.
            assert v.min() == v.max(), "torn read: the writer touched a slot we hold"
    finally:
        stop.set()
        th.join(timeout=2.0)
    assert not errors


# --------------------------------------------------------------------------- read side


class _FakeTracker:
    def __init__(self) -> None:
        self.buffers: TripleBuffer | None = TripleBuffer(4, 4)

    def publish(self, pos: float, vel: float, t: float, present: bool = True) -> None:
        assert self.buffers is not None
        slot = self.buffers.write_slot
        slot.landmarks[:] = pos
        slot.velocity[:] = vel
        slot.present[:] = present
        slot.world_scale[:] = 0.1
        slot.t = t
        self.buffers.publish()


def test_hand_state_extrapolates_between_detections() -> None:
    """Tracking runs at 30Hz and the display at 60. The gap has to be filled."""
    tr = _FakeTracker()
    s = HandState()
    tr.publish(0.5, 1.0, 100.0)
    s.update(tr, 100.0)
    np.testing.assert_allclose(s.landmarks, 0.5, atol=1e-5)
    s.update(tr, 100.02)  # no new detection, 20ms later
    np.testing.assert_allclose(s.landmarks, 0.52, atol=1e-5)


def test_hand_state_clamps_runaway_extrapolation() -> None:
    """A dropped detection must not fling a landmark across the frame."""
    tr = _FakeTracker()
    s = HandState()
    tr.publish(0.5, 50.0, 100.0)
    s.update(tr, 100.0)
    s.update(tr, 101.0)  # a whole second with no detection
    assert float(np.abs(s.landmarks - 0.5).max()) <= C.EXTRAPOLATE_MAX_DELTA + 1e-6


def test_hand_state_marks_a_hand_lost_after_the_timeout() -> None:
    tr = _FakeTracker()
    s = HandState()
    tr.publish(0.5, 0.0, 100.0)
    s.update(tr, 100.0)
    assert s.both
    s.update(tr, 100.0 + C.HAND_LOST_SECONDS + 0.01)
    both, any_hand = s.both, s.any_hand
    assert not both
    assert not any_hand


def test_hand_state_flags_a_new_camera_frame_exactly_once() -> None:
    tr = _FakeTracker()
    s = HandState()
    tr.publish(0.5, 0.0, 100.0)
    s.update(tr, 100.0)
    assert s.frame_dirty
    s.update(tr, 100.01)
    assert not s.frame_dirty, "the texture would be re-uploaded for nothing"


def test_hand_state_is_safe_before_any_detection() -> None:
    class Empty:
        buffers = None

    s = HandState()
    s.update(Empty(), 1.0)  # type: ignore[arg-type]
    assert not s.any_hand


# --------------------------------------------------------------------------- slimming


def test_stub_modules_satisfy_an_import_and_are_marked_as_stubs() -> None:
    """The stub has to survive the import patterns mediapipe actually uses.

    Notably `import a.b.c` followed by a bare reference to `c`, which only works
    because the import system sets the child as an attribute of the parent --
    the reason this goes through a meta-path finder rather than a sys.modules
    entry.
    """
    from portal_palms import mediapipe_slim

    finder = mediapipe_slim._StubFinder()
    spec = finder.find_spec("matplotlib.pyplot")
    assert spec is not None
    assert finder.find_spec("numpy") is None, "must not intercept anything else"

    loader = spec.loader
    assert loader is not None
    module = loader.create_module(spec)
    assert module is not None
    loader.exec_module(module)
    assert mediapipe_slim.is_stub(module)
    assert not mediapipe_slim.is_stub(np)
    # Attribute reads must work: drawing_styles pulls constants off matplotlib
    # while it is being imported.
    assert module.rcParams is not None
    assert module.colors.LinearSegmentedColormap is not None


def test_calling_into_a_stub_explains_itself() -> None:
    from portal_palms import mediapipe_slim

    inert = mediapipe_slim._Inert("matplotlib.pyplot.figure")
    with pytest.raises(RuntimeError, match="mediapipe_slim"):
        inert()
