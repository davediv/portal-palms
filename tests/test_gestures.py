"""The gesture recognisers.

The properties that matter are the ones a bare threshold would fail: no
oscillation on the boundary, a real dwell time before anything latches, and
identical behaviour regardless of how far the user is from the camera.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms import gestures as G
from portal_palms.tracking import HandState

Arr = NDArray[np.float32]

F32 = np.float32


# --------------------------------------------------------------------------- helpers


def pose(spread_units: float, hand_px: float = 0.145, cy: float = 0.5, roll: float = 0.0) -> Arr:
    """Two hands `spread_units` hand-sizes apart, at a given apparent hand size.

    `hand_px` is the apparent wrist->middle-MCP length as a fraction of frame
    width, so shrinking it simulates the user standing further away.
    """
    lm = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
    half = spread_units * hand_px * 0.5
    y_scale = G.frame_aspect()
    for h in range(C.NUM_HANDS):
        side = -1.0 if h == 0 else 1.0
        wx = 0.5 + side * half
        lm[h, C.WRIST, 0] = wx
        lm[h, C.WRIST, 1] = cy
        # Hand axis points "up" the image, rotated by roll.
        lm[h, C.MIDDLE_MCP, 0] = wx + math.sin(roll) * hand_px
        lm[h, C.MIDDLE_MCP, 1] = cy - math.cos(roll) * hand_px / y_scale
        for tip in (C.INDEX_TIP, C.MIDDLE_TIP, C.RING_TIP, C.PINKY_TIP, C.THUMB_TIP):
            lm[h, tip, 0] = lm[h, C.MIDDLE_MCP, 0]
            lm[h, tip, 1] = lm[h, C.MIDDLE_MCP, 1] - hand_px * 0.8 / y_scale
    return lm


def phase_of(p: G.PortalGesture) -> G.PortalPhase:
    """Read the current phase through a call.

    `assert p.phase is X` narrows the attribute for the rest of the function,
    and mypy does not invalidate that narrowing when the next `p.update()`
    changes it -- so a later `assert p.phase is Y` looks statically impossible.
    Returning it from a function hands back a plain PortalPhase each time.
    """
    return p.phase


def state(lm: Arr, both: bool = True) -> HandState:
    s = HandState()
    np.copyto(s.landmarks, lm)
    s.present[0] = True
    s.present[1] = both
    return s


# --------------------------------------------------------------------------- hysteresis


def test_hysteresis_rejects_an_inverted_band() -> None:
    with pytest.raises(ValueError):
        G.Hysteresis(enter=1.0, exit_=2.0)


def test_hysteresis_needs_the_hold_time_before_it_latches() -> None:
    h = G.Hysteresis(enter=1.0, exit_=0.5, hold_ms=200.0)
    assert not h.update(2.0, 0.000)
    assert not h.update(2.0, 0.100)
    assert h.progress == pytest.approx(0.5, abs=0.01)
    assert h.update(2.0, 0.200)


def test_hysteresis_does_not_oscillate_between_the_thresholds() -> None:
    """The whole reason for two thresholds. Sitting between them must be stable."""
    h = G.Hysteresis(enter=1.0, exit_=0.5, hold_ms=0.0)
    assert h.update(1.2, 0.0)
    flips = 0
    was = h.active
    for i in range(400):
        # A signal parked at 0.75 with noise: above exit, below enter.
        value = 0.75 + math.sin(i * 1.7) * 0.2
        h.update(value, 1.0 + i * 0.01)
        flips += h.active != was
        was = h.active
    assert flips == 0


def test_hysteresis_hold_restarts_if_the_signal_dips() -> None:
    h = G.Hysteresis(enter=1.0, exit_=0.5, hold_ms=200.0)
    h.update(1.5, 0.00)
    h.update(1.5, 0.15)
    h.update(0.9, 0.16)  # below enter but above exit: the countdown restarts
    assert h.progress == 0.0
    # The clock restarts from here, so the full 200ms is required again --
    # the 150ms already banked before the dip does not count.
    assert not h.update(1.5, 0.30)
    assert not h.update(1.5, 0.45)
    assert h.update(1.5, 0.51)


def test_hysteresis_releases_only_below_the_exit_threshold() -> None:
    h = G.Hysteresis(enter=1.0, exit_=0.5, hold_ms=0.0)
    assert h.update(1.1, 0.0)
    assert h.update(0.6, 0.1), "still above exit, must stay latched"
    assert not h.update(0.4, 0.2)


# --------------------------------------------------------------------------- measures


def test_hand_size_is_isotropic_under_rotation() -> None:
    """A hand held sideways must measure the same as one held upright.

    MediaPipe normalises x by width and y by height, so without the aspect
    correction this is off by 33% on a 4:3 sensor and every threshold drifts
    as the user rotates their wrist.
    """
    upright = G.hand_size(pose(4.0, roll=0.0), 0)
    sideways = G.hand_size(pose(4.0, roll=math.pi / 2), 0)
    assert sideways == pytest.approx(upright, rel=0.02)


@pytest.mark.parametrize("hand_px", [0.30, 0.145, 0.07, 0.04])
def test_spread_is_distance_invariant(hand_px: float) -> None:
    """0.5m or 2m from the camera, the same gesture must read the same."""
    assert G.wrist_spread(pose(4.0, hand_px=hand_px)) == pytest.approx(4.0, rel=0.02)


def test_spread_grows_with_separation() -> None:
    assert G.wrist_spread(pose(1.0)) < G.wrist_spread(pose(3.0)) < G.wrist_spread(pose(6.0))


def test_hand_roll_tracks_the_wrist_axis() -> None:
    assert G.hand_roll(pose(4.0, roll=0.0), 0) == pytest.approx(-math.pi / 2, abs=0.02)
    assert G.hand_roll(pose(4.0, roll=0.4), 0) == pytest.approx(-math.pi / 2 + 0.4, abs=0.03)


def test_angle_lerp_takes_the_short_way_round() -> None:
    out = G.angle_lerp(3.0, -3.0, 0.5)
    assert abs(((out + math.pi) % (2 * math.pi)) - math.pi) > 3.0


# --------------------------------------------------------------------------- portal


def test_portal_needs_both_hands() -> None:
    p = G.PortalGesture()
    p.update(state(pose(5.0), both=False), 0.0, 0.016)
    assert phase_of(p) is not G.PortalPhase.LOCKED
    assert p.openness == 0.0


def test_portal_radius_tracks_separation_without_popping() -> None:
    """Radius maps directly to wrist distance, so it must be smooth and monotone."""
    p = G.PortalGesture()
    radii = []
    for i in range(60):
        spread = 1.0 + i * 0.08
        p.update(state(pose(spread)), i * 0.016, 0.016)
        radii.append(p.radius)
    deltas = np.diff(radii)
    assert (deltas > 0).all(), "radius must be monotone as the hands separate"
    assert deltas.max() < deltas.mean() * 2.0, "no step changes"


def test_portal_locks_only_after_the_hold_time() -> None:
    p = G.PortalGesture()
    s = state(pose(5.5))
    t = 0.0
    p.update(s, t, 0.016)
    for _ in range(int(C.PORTAL_HOLD_MS / 1000.0 * 60) - 4):
        t += 1 / 60
        p.update(s, t, 1 / 60)
    assert phase_of(p) is not G.PortalPhase.LOCKED
    for _ in range(8):
        t += 1 / 60
        p.update(s, t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.LOCKED


def test_portal_stays_locked_through_small_wobble() -> None:
    p = G.PortalGesture()
    t = 0.0
    for _ in range(150):
        t += 1 / 60
        p.update(state(pose(5.5)), t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.LOCKED
    for i in range(200):
        t += 1 / 60
        # Wobble across the enter threshold but never below exit.
        p.update(state(pose(4.3 + math.sin(i * 0.9) * 0.6)), t, 1 / 60)
        assert phase_of(p) is G.PortalPhase.LOCKED


def test_portal_unlocks_and_asks_to_close_when_hands_come_together() -> None:
    p = G.PortalGesture()
    t = 0.0
    for _ in range(150):
        t += 1 / 60
        p.update(state(pose(5.5)), t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.LOCKED
    t += 1 / 60
    p.update(state(pose(1.2)), t, 1 / 60)
    assert p.close_requested
    assert phase_of(p) is G.PortalPhase.CLOSING


def test_portal_collapses_gracefully_when_a_hand_leaves() -> None:
    """One hand leaving frame must shrink the portal, not freeze or crash it."""
    p = G.PortalGesture()
    t = 0.0
    for _ in range(20):
        t += 1 / 60
        p.update(state(pose(4.6)), t, 1 / 60)
    assert p.openness > 0.5

    gone = state(pose(4.6), both=False)
    for _ in range(120):
        t += 1 / 60
        p.update(gone, t, 1 / 60)
    assert p.openness == 0.0
    assert phase_of(p) is G.PortalPhase.IDLE


def test_locked_portal_survives_a_brief_hand_dropout() -> None:
    """Losing a word because someone scratched their nose is worse than a beat of lag."""
    p = G.PortalGesture()
    t = 0.0
    for _ in range(150):
        t += 1 / 60
        p.update(state(pose(5.5)), t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.LOCKED
    one = state(pose(5.5), both=False)
    for _ in range(30):
        t += 1 / 60
        p.update(one, t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.LOCKED


def test_locked_portal_closes_when_both_hands_go() -> None:
    p = G.PortalGesture()
    t = 0.0
    for _ in range(150):
        t += 1 / 60
        p.update(state(pose(5.5)), t, 1 / 60)
    none = HandState()
    t += 1 / 60
    p.update(none, t, 1 / 60)
    assert phase_of(p) is G.PortalPhase.CLOSING
    assert p.close_requested


def test_portal_seam_below_the_minimum_spread() -> None:
    p = G.PortalGesture()
    p.update(state(pose(0.9)), 0.0, 0.016)
    assert phase_of(p) is G.PortalPhase.SEAM
    assert p.openness == pytest.approx(0.0, abs=1e-4)


def test_portal_openness_is_full_at_the_open_threshold() -> None:
    p = G.PortalGesture()
    p.update(state(pose(C.PORTAL_OPEN_SPREAD + 0.5)), 0.0, 0.016)
    assert p.openness == pytest.approx(1.0)


def test_portal_rotation_follows_both_hands() -> None:
    p = G.PortalGesture()
    t = 0.0
    for _ in range(120):
        t += 1 / 60
        p.update(state(pose(4.6, roll=0.0)), t, 1 / 60)
    flat = p.rotation
    for _ in range(240):
        t += 1 / 60
        p.update(state(pose(4.6, roll=0.5)), t, 1 / 60)
    assert abs(G.angle_lerp(flat, p.rotation, 1.0) - flat) > 0.3


def test_portal_never_raises_on_an_empty_hand_state() -> None:
    p = G.PortalGesture()
    empty = HandState()
    for i in range(30):
        p.update(empty, i * 0.016, 0.016)
    assert phase_of(p) is G.PortalPhase.IDLE


# --------------------------------------------------------------------------- swipe


def _drive_swipe(sw: G.SwipeGesture, dx_per_frame: float, frames: int, t0: float = 0.0) -> int:
    """Move one hand horizontally and return the first non-zero swipe result."""
    fired = 0
    x = 0.5
    for i in range(frames):
        x += dx_per_frame
        lm = pose(4.0)
        lm[:, :, 0] += x - 0.5
        s = state(lm, both=False)
        got = sw.update(s, t0 + i / 60.0)
        if got and not fired:
            fired = got
    return fired


def test_swipe_fires_on_a_real_flick() -> None:
    sw = G.SwipeGesture()
    assert _drive_swipe(sw, 0.016, 20) == 1


def test_swipe_direction_follows_the_hand() -> None:
    sw = G.SwipeGesture()
    assert _drive_swipe(sw, -0.016, 20) == -1


def test_swipe_ignores_a_slow_drift() -> None:
    sw = G.SwipeGesture()
    assert _drive_swipe(sw, 0.0015, 120) == 0


def test_swipe_ignores_a_single_frame_spike() -> None:
    """One jittery frame must never skip the learner's word."""
    sw = G.SwipeGesture()
    for i in range(10):
        sw.update(state(pose(4.0), both=False), i / 60.0)
    lm = pose(4.0)
    lm[:, :, 0] += 0.06  # one big frame, then nothing
    assert sw.update(state(lm, both=False), 10 / 60.0) == 0


def test_swipe_ignores_a_tracking_discontinuity() -> None:
    sw = G.SwipeGesture()
    for i in range(10):
        sw.update(state(pose(4.0), both=False), i / 60.0)
    lm = pose(4.0)
    lm[:, :, 0] += 0.4  # hand teleports: reacquisition, not a gesture
    assert sw.update(state(lm, both=False), 10 / 60.0) == 0


def test_swipe_has_a_refractory_period() -> None:
    sw = G.SwipeGesture()
    assert _drive_swipe(sw, 0.016, 20) == 1
    assert _drive_swipe(sw, 0.016, 20, t0=20 / 60.0) == 0


def test_swipe_resets_when_the_hands_leave() -> None:
    sw = G.SwipeGesture()
    _drive_swipe(sw, 0.016, 10)
    sw.update(HandState(), 1.0)
    assert sw.speed == 0.0


@pytest.mark.parametrize("hand_px", [0.30, 0.145, 0.07])
def test_swipe_threshold_is_distance_invariant(hand_px: float) -> None:
    """A flick that works up close must work from across the room, and vice versa."""
    sw = G.SwipeGesture()
    x = 0.5
    fired = 0
    for i in range(20):
        x += 0.11 * hand_px  # same motion measured in hand-sizes
        lm = pose(4.0, hand_px=hand_px)
        lm[:, :, 0] += x - 0.5
        got = sw.update(state(lm, both=False), i / 60.0)
        fired = fired or got
    assert fired == 1


# --------------------------------------------------------------------------- misc


def test_smoothstep_clamps_and_is_monotone() -> None:
    assert G.smoothstep(0.0, 1.0, -5.0) == 0.0
    assert G.smoothstep(0.0, 1.0, 5.0) == 1.0
    prev = -1.0
    for i in range(21):
        v = G.smoothstep(0.0, 1.0, i / 20.0)
        assert v >= prev
        prev = v


def test_smoothstep_handles_a_degenerate_band() -> None:
    assert G.smoothstep(1.0, 1.0, 0.5) == 0.0
    assert G.smoothstep(1.0, 1.0, 1.5) == 1.0
