"""Gesture recognisers.

Two rules hold everywhere in this file:

1. Every distance is divided by hand size — distance(wrist, middle MCP) — so a
   gesture behaves identically at 0.5m and at 2m from the camera.
2. Nothing is a bare threshold. Every recogniser has a separate enter level, a
   lower exit level, and a minimum hold time. A bare threshold oscillates while
   the signal sits on the boundary, and users read that flicker as a broken app.

One correction underpins both: MediaPipe normalises x by frame width and y by
frame height, so on a 4:3 camera a vertical centimetre and a horizontal
centimetre are *not* the same number. Every distance below therefore scales y by
height/width first. Without it a hand held upright measures 33% longer than the
same hand held sideways, and every threshold drifts with wrist rotation.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.tracking import HandState

F32 = np.float32
Arr = NDArray[np.float32]


# --------------------------------------------------------------------------- primitives


class Hysteresis:
    """A latching threshold with a dwell time.

    `value` rises above `enter` for `hold_ms` -> active. It stays active until
    `value` falls below `exit_`, which is deliberately lower than `enter`.
    """

    __slots__ = ("_above_since", "_enter", "_exit", "_hold_s", "active", "progress")

    def __init__(self, enter: float, exit_: float, hold_ms: float = 0.0) -> None:
        if exit_ >= enter:
            raise ValueError("exit threshold must sit below enter threshold")
        self._enter = enter
        self._exit = exit_
        self._hold_s = hold_ms * 1e-3
        self._above_since = -1.0
        self.active = False
        self.progress = 0.0

    def reset(self) -> None:
        self._above_since = -1.0
        self.active = False
        self.progress = 0.0

    def update(self, value: float, now: float) -> bool:
        if self.active:
            if value < self._exit:
                self.active = False
                self._above_since = -1.0
                self.progress = 0.0
            else:
                self.progress = 1.0
            return self.active

        if value >= self._enter:
            if self._above_since < 0.0:
                self._above_since = now
            if self._hold_s <= 0.0:
                self.active = True
                self.progress = 1.0
            else:
                elapsed = now - self._above_since
                self.progress = elapsed / self._hold_s
                if self.progress >= 1.0:
                    self.progress = 1.0
                    self.active = True
        else:
            self._above_since = -1.0
            self.progress = 0.0
        return self.active


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- measures

# height/width of the capture device. Set once from the real camera dimensions;
# 4:3 is the default because that is what 640x480 gives us.
_Y_SCALE = C.CAM_H / C.CAM_W


def set_frame_aspect(width: int, height: int) -> None:
    """Tell the gesture layer the true camera dimensions, once, at startup."""
    global _Y_SCALE
    if width > 0 and height > 0:
        _Y_SCALE = height / width


def frame_aspect() -> float:
    return _Y_SCALE


def _dist(landmarks: Arr, ha: int, la: int, hb: int, lb: int) -> float:
    """Isotropic distance between two landmarks, in units of frame *width*."""
    dx = float(landmarks[ha, la, 0] - landmarks[hb, lb, 0])
    dy = float(landmarks[ha, la, 1] - landmarks[hb, lb, 1]) * _Y_SCALE
    return math.sqrt(dx * dx + dy * dy)


def hand_size(landmarks: Arr, hand: int) -> float:
    """distance(wrist, middle MCP) — the unit every other distance is divided by."""
    d = _dist(landmarks, hand, C.MIDDLE_MCP, hand, C.WRIST)
    return d if d > 1e-3 else 1e-3


def wrist_distance(landmarks: Arr) -> float:
    """Raw distance between the two wrists, in units of frame width."""
    return _dist(landmarks, 0, C.WRIST, 1, C.WRIST)


def wrist_spread(landmarks: Arr) -> float:
    """Distance between the two wrists, in hand-sizes.

    Normalised by the mean of both hand sizes so one hand turning edge-on --
    which shrinks its apparent length -- does not make the portal jump.
    """
    unit = 0.5 * (hand_size(landmarks, 0) + hand_size(landmarks, 1))
    return wrist_distance(landmarks) / unit


def hand_roll(landmarks: Arr, hand: int) -> float:
    """Roll of one hand in radians, from the wrist -> middle MCP axis."""
    dx = float(landmarks[hand, C.MIDDLE_MCP, 0] - landmarks[hand, C.WRIST, 0])
    dy = float(landmarks[hand, C.MIDDLE_MCP, 1] - landmarks[hand, C.WRIST, 1]) * _Y_SCALE
    return math.atan2(dy, dx)


def pinch_distance(landmarks: Arr, hand: int) -> float:
    """Thumb tip to index tip, in hand-sizes."""
    return _dist(landmarks, hand, C.THUMB_TIP, hand, C.INDEX_TIP) / hand_size(landmarks, hand)


def openness(landmarks: Arr, hand: int) -> float:
    """Mean fingertip-to-wrist distance in hand-sizes: ~1.4 curled, ~2.3 splayed."""
    unit = hand_size(landmarks, hand)
    total = 0.0
    for tip in (C.INDEX_TIP, C.MIDDLE_TIP, C.RING_TIP, C.PINKY_TIP):
        total += _dist(landmarks, hand, tip, hand, C.WRIST)
    return (total * 0.25) / unit


def angle_lerp(a: float, b: float, t: float) -> float:
    """Interpolate between two angles the short way around."""
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * t


# --------------------------------------------------------------------------- portal


class PortalPhase(IntEnum):
    IDLE = 0  # no usable hands
    SEAM = 1  # hands together: a thin bright line
    OPENING = 2  # pulling apart, below the lock threshold
    ARMED = 3  # held past the threshold, counting down to lock
    LOCKED = 4  # scene is playing
    CLOSING = 5  # imploding shut


class PortalGesture:
    """Two hands pulling apart to tear a portal open.

    Radius maps directly and continuously to wrist spread, so there is no
    popping. The lock is the only latched decision, and it is a full hysteresis
    state machine with a dwell time.
    """

    __slots__ = (
        "_lock",
        "_prev_center_x",
        "_prev_center_y",
        "_rot_init",
        "center_x",
        "center_y",
        "close_requested",
        "hold_progress",
        "implode",
        "openness",
        "phase",
        "radius",
        "rotation",
        "spread",
        "vel_x",
        "vel_y",
    )

    phase: PortalPhase

    def __init__(self) -> None:
        self._lock = Hysteresis(
            enter=C.PORTAL_ENTER_SPREAD,
            exit_=C.PORTAL_EXIT_SPREAD,
            hold_ms=C.PORTAL_HOLD_MS,
        )
        self.phase = PortalPhase.IDLE
        self.spread = 0.0
        # 0 at a closed seam, 1 fully open. Drives the rim shader directly.
        self.openness = 0.0
        self.radius = 0.0
        self.rotation = 0.0
        self.hold_progress = 0.0
        self.center_x = 0.5
        self.center_y = 0.5
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.implode = 0.0
        self.close_requested = False
        self._prev_center_x = 0.5
        self._prev_center_y = 0.5
        self._rot_init = False

    def reset(self) -> None:
        self._lock.reset()
        self.phase = PortalPhase.IDLE
        self.openness = 0.0
        self.hold_progress = 0.0
        self.implode = 0.0
        self.close_requested = False

    def unlock(self) -> None:
        """Leave the locked state without requiring the hands to close first."""
        self._lock.reset()
        self.phase = PortalPhase.CLOSING
        self.implode = 1.0

    # ------------------------------------------------------------------ update

    def update(self, hands: HandState, now: float, dt: float) -> None:
        self.close_requested = False
        lm = hands.landmarks

        if not hands.both:
            self._collapse(hands, dt)
            return

        self.spread = wrist_spread(lm)

        cx = 0.5 * float(lm[0, C.WRIST, 0] + lm[1, C.WRIST, 0])
        cy = 0.5 * float(lm[0, C.WRIST, 1] + lm[1, C.WRIST, 1])
        if dt > 1e-5:
            inv = 1.0 / dt
            self.vel_x = (cx - self._prev_center_x) * inv
            self.vel_y = (cy - self._prev_center_y) * inv
        self._prev_center_x = cx
        self._prev_center_y = cy
        self.center_x = cx
        self.center_y = cy

        # Both hands roll together -> the ring rolls with them.
        r0 = hand_roll(lm, 0)
        r1 = hand_roll(lm, 1)
        # Hand 0 points right-ish and hand 1 left-ish when framing a portal, so
        # average their rolls after folding hand 1 onto the same half-turn.
        target = 0.5 * (r0 + (r1 + math.pi))
        if not self._rot_init:
            self.rotation = target
            self._rot_init = True
        else:
            self.rotation = angle_lerp(self.rotation, target, C.PORTAL_ROTATION_SMOOTH)

        was_locked = self.phase is PortalPhase.LOCKED
        locked = self._lock.update(self.spread, now)
        self.hold_progress = self._lock.progress

        if locked:
            self.phase = PortalPhase.LOCKED
            self.openness = 1.0
        elif was_locked:
            # Fell out of the latch: the user pulled their hands back together.
            self.phase = PortalPhase.CLOSING
            self.implode = 1.0
            self.close_requested = True
        else:
            self.openness = smoothstep(C.PORTAL_SEAM_SPREAD, C.PORTAL_OPEN_SPREAD, self.spread)
            if self.implode > 0.0:
                self.implode -= dt * (1000.0 / C.PORTAL_IMPLODE_MS)
                if self.implode < 0.0:
                    self.implode = 0.0
                # The implosion overrides hand position: once it starts the
                # portal collapses on its own clock, so it snaps shut crisply
                # instead of easing out as slowly as the hands happen to move.
                self.openness *= self.implode * self.implode
                self.phase = PortalPhase.CLOSING
            elif self.spread < C.PORTAL_SEAM_SPREAD:
                self.phase = PortalPhase.SEAM
            else:
                self.phase = PortalPhase.ARMED if self.hold_progress > 0.0 else PortalPhase.OPENING

        self._update_radius(lm)

    # ------------------------------------------------------------------ helpers

    def _update_radius(self, lm: Arr) -> None:
        """Radius is the literal half-distance between the wrists, in screen widths.

        Mapping it directly rather than through a filter is what makes the
        portal feel physically attached to the hands. The One-Euro filter
        upstream has already taken the jitter out, so there is nothing left to
        smooth here and smoothing anyway would only add lag.
        """
        # Leave a margin so the rim sits inside the palms rather than under them.
        self.radius = 0.5 * wrist_distance(lm) * 0.86

    def _collapse(self, hands: HandState, dt: float) -> None:
        """One or both hands left frame. Shrink to nothing; never freeze, never crash."""
        if self.phase is PortalPhase.LOCKED:
            # A locked scene keeps playing while a hand is momentarily lost --
            # dropping the user's word because they scratched their nose is worse
            # than holding the portal open a beat too long.
            if not hands.any_hand:
                self._lock.reset()
                self.phase = PortalPhase.CLOSING
                self.implode = 1.0
                self.close_requested = True
            return

        self._lock.reset()
        self.hold_progress = 0.0
        decay = 1.0 - min(dt * 7.0, 1.0)
        self.openness *= decay
        self.radius *= decay
        self.spread *= decay
        if self.implode > 0.0:
            self.implode -= dt * (1000.0 / C.PORTAL_IMPLODE_MS)
            if self.implode < 0.0:
                self.implode = 0.0
        if self.openness < 0.004:
            self.openness = 0.0
            self.radius = 0.0
            self.phase = PortalPhase.IDLE
        else:
            self.phase = PortalPhase.CLOSING
        self._rot_init = False

    @property
    def is_open(self) -> bool:
        """True while the interior scene needs to be re-rendered."""
        return self.openness > 0.01 or self.implode > 0.0


# --------------------------------------------------------------------------- swipe


class SwipeGesture:
    """A horizontal flick with one hand, used to advance to the next word.

    Speed alone is not enough to define a swipe. A single fast frame -- one
    jittery landmark, one re-acquired hand -- clears any speed threshold you
    care to set, and skipping the learner's word because of one frame is a bad
    failure. So this requires both: sustained speed *and* real distance
    travelled in one direction, with the usual enter/exit hysteresis on top.
    """

    __slots__ = (
        "_armed",
        "_last_fire",
        "_last_x",
        "_prev_t",
        "_travel",
        "direction",
        "speed",
    )

    ENTER_SPEED = 3.4  # hand-sizes / second, must be sustained
    EXIT_SPEED = 1.2  # below this the travel accumulator resets
    MIN_TRAVEL = 1.10  # hand-sizes of continuous motion before it counts
    REFRACTORY_S = 0.55
    # A real flick moves well under this in one frame. Anything larger is the
    # tracker losing and reacquiring a hand, or the two hands trading slots.
    MAX_JUMP = 0.55

    def __init__(self) -> None:
        self._last_x = 0.0
        self._prev_t = 0.0
        self._travel = 0.0
        self._armed = True
        self._last_fire = -10.0
        self.speed = 0.0
        self.direction = 0

    def reset(self) -> None:
        self._prev_t = 0.0
        self._travel = 0.0
        self._armed = True
        self.speed = 0.0
        self.direction = 0

    def update(self, hands: HandState, now: float) -> int:
        """Returns -1 (swipe left), +1 (swipe right) on the frame it fires, else 0."""
        hand = 0 if hands.present[0] else (1 if hands.present[1] else -1)
        if hand < 0:
            self.reset()
            return 0

        lm = hands.landmarks
        x = float(lm[hand, C.WRIST, 0]) / hand_size(lm, hand)
        if self._prev_t <= 0.0:
            self._last_x = x
            self._prev_t = now
            return 0

        dt = now - self._prev_t
        self._prev_t = now
        if dt < 1e-4 or dt > 0.5:
            self._last_x = x
            return 0

        jump = x - self._last_x
        self._last_x = x
        if abs(jump) > self.MAX_JUMP:
            # A discontinuity, not a movement. Drop the run and start over.
            self._travel = 0.0
            self.speed = 0.0
            return 0

        vx = jump / dt
        self.speed = abs(vx)

        if self.speed < self.EXIT_SPEED:
            self._travel = 0.0
            self._armed = True
            return 0

        # Reversing direction restarts the run rather than cancelling it out.
        if self._travel * jump < 0.0:
            self._travel = 0.0
        self._travel += jump

        if not self._armed:
            return 0
        if (
            self.speed >= self.ENTER_SPEED
            and abs(self._travel) >= self.MIN_TRAVEL
            and (now - self._last_fire) > self.REFRACTORY_S
        ):
            self._armed = False
            self._travel = 0.0
            self._last_fire = now
            self.direction = 1 if jump > 0.0 else -1
            return self.direction
        return 0
