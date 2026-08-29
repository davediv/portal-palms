"""The zero-allocation claim, as an executable assertion.

The spec's hard requirement is that the steady-state loop does not allocate,
because with the cyclic collector switched off a growing heap of tracked
containers is the one thing that could still produce a pause -- and a pause is
a dropped frame that shows up in a 60fps recording.

The measure is the number of *live* GC-tracked objects, which is what a
collection would have to walk. `gc.get_count()` looks like the obvious choice
and is not: CPython's freelists recycle small tuples and lists without touching
that counter, so 100k tuple churns move it by 1 and a thousand retained tuples
move it by 0. It cannot distinguish churn from growth. `len(gc.get_objects())`
can.

Scalars are not tracked at all and are freed immediately by refcounting, so
they can never cause a pause; containers are the only thing that can.
"""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest

from portal_palms import config as C
from portal_palms import gestures as G
from portal_palms.content.metaphors import SCENES
from portal_palms.render.gl import buffer_prefix_views
from portal_palms.render.primitives import INSTANCE_FLOATS, MAX_PER_PRIM, PRIM_COUNT, Painter
from portal_palms.render.text import NumBuf
from portal_palms.tracking import HandState, OneEuroFilter, TripleBuffer

F32 = np.float32
FRAMES = 1500


class FakePool:
    def __init__(self) -> None:
        self.instances = [
            np.zeros((MAX_PER_PRIM, INSTANCE_FLOATS), dtype=np.float32) for _ in range(PRIM_COUNT)
        ]


class _Source:
    def __init__(self) -> None:
        self.buffers: TripleBuffer | None = TripleBuffer(8, 8)

    def publish(self, spread: float, t: float) -> None:
        assert self.buffers is not None
        slot = self.buffers.write_slot
        lm = slot.landmarks
        for h in range(C.NUM_HANDS):
            side = -1.0 if h == 0 else 1.0
            lm[h, C.WRIST, 0] = 0.5 + side * spread * 0.5
            lm[h, C.WRIST, 1] = 0.5
            lm[h, C.MIDDLE_MCP, 0] = 0.5 + side * spread * 0.5
            lm[h, C.MIDDLE_MCP, 1] = 0.5 - 0.19
        slot.present[:] = True
        slot.velocity[:] = 0.0
        slot.world_scale[:] = 0.145
        slot.t = t
        self.buffers.publish()


def test_buffer_prefix_views_cover_only_live_rows_without_copying() -> None:
    data = np.zeros((4, 3), dtype=F32)
    prefixes = buffer_prefix_views(data)
    assert [view.nbytes for view in prefixes] == [0, 12, 24, 36, 48]

    data[0, 1] = 7.0
    first_row = np.frombuffer(prefixes[1], dtype=F32)
    assert first_row.tolist() == [0.0, 7.0, 0.0]


def test_buffer_prefix_views_reject_noncontiguous_arrays() -> None:
    data = np.zeros((4, 6), dtype=F32)[:, ::2]
    with pytest.raises(ValueError, match="contiguous 2D"):
        buffer_prefix_views(data)


def _steady_state(body: object, warmup: int = 300) -> int:
    """Tracked-object growth across a second identical window of FRAMES.

    Two windows, and only the second counts. The first absorbs anything created
    lazily on a code path's first execution -- a numpy ufunc cache, a branch
    taken for the first time. Those are bounded and happen once; what would
    actually hurt is per-frame growth, and that shows up in both windows
    equally. This asserts the second window is exactly flat.
    """
    fn = body  # bound to a local, exactly as the render loop does
    for i in range(warmup):
        fn(i)  # type: ignore[operator]
    gc.collect()
    gc.disable()
    try:
        for i in range(warmup, warmup + FRAMES):
            fn(i)  # type: ignore[operator]
        before = len(gc.get_objects())
        for i in range(warmup + FRAMES, warmup + FRAMES * 2):
            fn(i)  # type: ignore[operator]
        return len(gc.get_objects()) - before
    finally:
        gc.enable()


def test_gesture_update_allocates_nothing() -> None:
    src = _Source()
    hands = HandState()
    portal = G.PortalGesture()
    swipe = G.SwipeGesture()
    dt = 1.0 / 60.0

    def frame(i: int) -> None:
        t = i * dt
        # Republish at 30Hz, exactly as the camera thread would.
        if i % 2 == 0:
            src.publish(0.20 + 0.45 * (0.5 + 0.5 * math.sin(t * 0.7)), t)
        hands.update(src, t)
        portal.update(hands, t, dt)
        swipe.update(hands, t)

    assert _steady_state(frame) == 0


def test_one_euro_allocates_nothing() -> None:
    shape = (C.NUM_HANDS, C.NUM_LANDMARKS, 3)
    flt = OneEuroFilter(shape)
    x = np.zeros(shape, dtype=F32)
    out = np.zeros(shape, dtype=F32)
    valid = np.ones(C.NUM_HANDS, dtype=bool)

    def frame(i: int) -> None:
        x[:] = 0.4 + 0.1 * math.sin(i * 0.05)
        flt.filter(x, 1.0 / 30.0, out, valid)

    assert _steady_state(frame) == 0


@pytest.mark.parametrize("name", sorted(SCENES))
def test_painting_a_scene_allocates_nothing(name: str) -> None:
    """The metaphors run every frame the portal is open, so they count too."""
    painter = Painter(FakePool())  # type: ignore[arg-type]
    sc = SCENES[name]

    def frame(i: int) -> None:
        painter.reset()
        sc.paint(painter, (i / 60.0) % sc.loop)

    assert _steady_state(frame, warmup=120) == 0


def test_number_formatting_allocates_nothing_when_stable() -> None:
    """The HUD formats a dozen numbers a frame; f-strings would allocate every one."""
    num = NumBuf(4)

    def frame(_i: int) -> None:
        num.fixed(0, 16.62, 2)
        num.fixed(1, 60.0, 0)
        num.integer(2, 4096)

    assert _steady_state(frame) == 0


def test_number_formatting_still_tracks_a_changing_value() -> None:
    num = NumBuf(2)
    assert num.fixed(0, 1.25, 2) == "1.25"
    assert num.fixed(0, 1.25, 2) == "1.25"
    assert num.fixed(0, -3.5, 1) == "-3.5"
    assert num.fixed(0, 0.0, 0) == "0"
    assert num.integer(1, 4096) == "4096"
