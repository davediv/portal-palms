"""A fake camera and a fake pair of hands, for demoing and for verification.

Two uses, both real:

* `--demo` lets a tutor page through the whole deck on a laptop with no camera,
  or with the camera already claimed by a video call.
* `--offscreen` drives the entire app -- gesture machine, portal, scene, UI,
  bloom, composite -- and writes PNGs, which is how the shipped frames get
  looked at without a human holding their hands up for every check.

It presents exactly the interface HandTracker does, so the app cannot tell the
difference and no production code grows a branch for it.
"""

from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from portal_palms import config as C
from portal_palms.tracking import TripleBuffer, mirror_landmarks

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

F32 = np.float32

# A hand, in wrist-relative units, roughly as MediaPipe reports one held up
# palm-forward. Only the landmarks the gestures actually read need to be right;
# the rest keep the shape plausible.
_HAND = np.array(
    [
        [0.00, 0.00], [-0.16, -0.10], [-0.26, -0.26], [-0.30, -0.40], [-0.32, -0.53],
        [-0.10, -0.44], [-0.13, -0.66], [-0.15, -0.79], [-0.16, -0.90],
        [0.02, -0.47], [0.02, -0.71], [0.02, -0.86], [0.02, -0.98],
        [0.14, -0.44], [0.17, -0.65], [0.19, -0.79], [0.20, -0.90],
        [0.25, -0.36], [0.31, -0.52], [0.34, -0.63], [0.36, -0.73],
    ],
    dtype=F32,
)  # fmt: skip


class SyntheticTracker:
    """Drop-in stand-in for HandTracker."""

    def __init__(
        self, mirror: bool = True, script: Callable[[float], tuple[float, ...]] | None = None
    ) -> None:
        self.mirror = mirror
        self.cam_w = C.CAM_W
        self.cam_h = C.CAM_H
        self.buffers: TripleBuffer | None = TripleBuffer(self.cam_h, self.cam_w)
        self.error: str | None = None
        self.camera_ready = True
        self.model_ready = True
        self.delegate_name = "FAKE"
        self.detect_ms = 0.0
        self.capture_ms = 0.0
        self.detections = 0
        self.script = script or default_script
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Settable so an offscreen run can align it with the app's clock.
        self.epoch = time.perf_counter()
        self._prev = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._prev_t = 0.0
        self._paint_backdrop()

    # ------------------------------------------------------------------ frames

    def _paint_backdrop(self) -> None:
        """A neutral gradient so the composed frame is not sitting on pure black."""
        assert self.buffers is not None
        h, w = self.cam_h, self.cam_w
        ramp_y = np.linspace(0.0, 1.0, h, dtype=F32)[:, None]
        ramp_x = np.linspace(0.0, 1.0, w, dtype=F32)[None, :]
        v = 26.0 + 34.0 * (1.0 - ramp_y) + 10.0 * ramp_x
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[..., 0] = np.clip(v * 0.55, 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(v * 0.72, 0, 255).astype(np.uint8)
        img[..., 2] = np.clip(v * 1.00, 0, 255).astype(np.uint8)
        for slot in self.buffers.slots:
            np.copyto(slot.rgb, img)

    # ------------------------------------------------------------------ lifecycle

    def start(self, on_ready: Callable[[], None] | None = None) -> None:
        if on_ready is not None:
            on_ready()
        self._thread = threading.Thread(target=self._run, name="portal-synthetic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        period = 1.0 / C.CAM_FPS
        while not self._stop.is_set():
            self.step(time.perf_counter())
            time.sleep(period)

    # ------------------------------------------------------------------ posing

    def step(self, now: float) -> None:
        """Write one synthetic detection, stamped with an absolute timestamp.

        `now` must come from the same clock the app passes to HandState.update.
        Mixing clocks makes the extrapolation window nonsense and shows up as
        landmarks teleporting.
        """
        buffers = self.buffers
        assert buffers is not None
        slot = buffers.write_slot
        spread, cx, cy, roll, both = self.script(now - self.epoch)

        lm = slot.landmarks
        lm[:] = 0.0
        # Scaled so wrist->middle-MCP measures about 0.145 of frame width, which
        # is what a real hand at arm's length reads on a 640x480 sensor. Without
        # that the synthetic spread ratios do not match the real thresholds and
        # the demo exercises the wrong part of the hysteresis.
        size = 0.30
        for h in range(C.NUM_HANDS):
            side = -1.0 if h == 0 else 1.0
            wx = cx + side * spread * 0.5
            wy = cy
            # Hands face each other, so the second one is mirrored in x.
            for i in range(C.NUM_LANDMARKS):
                px = _HAND[i, 0] * size * -side
                py = _HAND[i, 1] * size / (C.CAM_H / C.CAM_W)
                ca, sa = math.cos(roll * side), math.sin(roll * side)
                lm[h, i, 0] = wx + px * ca - py * sa
                lm[h, i, 1] = wy + px * sa + py * ca
                lm[h, i, 2] = 0.0
        if self.mirror:
            mirror_landmarks(lm)

        slot.present[0] = True
        slot.present[1] = bool(both)
        dt = max(now - self._prev_t, 1e-3) if self._prev_t > 0.0 else 1.0 / C.CAM_FPS
        np.subtract(lm, self._prev, out=slot.velocity)
        slot.velocity *= 1.0 / dt
        if self._prev_t <= 0.0:
            slot.velocity[:] = 0.0
        np.copyto(self._prev, lm)
        self._prev_t = now

        for h in range(C.NUM_HANDS):
            dx = lm[h, C.MIDDLE_MCP, 0] - lm[h, C.WRIST, 0]
            dy = lm[h, C.MIDDLE_MCP, 1] - lm[h, C.WRIST, 1]
            slot.world_scale[h] = max(float(np.hypot(dx, dy)), 1e-3)
        slot.t = now
        buffers.publish()
        self.detections += 1


def ease01(x: float) -> float:
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * (3.0 - 2.0 * x)


def default_script(t: float) -> tuple[float, ...]:
    """Open, hold long enough to lock, look at it, close. Then again.

    Timed against the real thresholds in config so the demo actually exercises
    the hysteresis and the 1.5s hold rather than skipping past them.
    """
    period = 16.0
    u = (t % period) / period
    # Continuous at every phase boundary on purpose: a step in the script is a
    # step in wrist velocity, and that reads to the gesture layer as a flick.
    if u < 0.10:
        spread = 0.14
    elif u < 0.26:
        spread = 0.14 + ease01((u - 0.10) / 0.16) * 0.56
    elif u < 0.80:
        spread = 0.70 + (math.sin((u - 0.26) / 0.54 * math.tau * 2.0) - 0.0) * 0.02
    elif u < 0.90:
        spread = 0.70 - ease01((u - 0.80) / 0.10) * 0.56
    else:
        spread = 0.14
    cy = 0.46 + math.sin(t * 0.6) * 0.05
    cx = 0.5 + math.sin(t * 0.4) * 0.04
    roll = math.sin(t * 0.5) * 0.22

    # Then a real swipe: drop one hand and flick, which is how anyone actually
    # advances. The portal stays locked on one hand, so the scene keeps playing
    # through the gesture.
    both = 1.0
    if 0.70 <= u < 0.80:
        both = 0.0
        cx += ease01((u - 0.70) / 0.07) * 0.42
    elif u >= 0.95:
        both = 0.0
    return spread, cx, cy, roll, both


def still(spread: float, roll: float = 0.0) -> Callable[[float], tuple[float, ...]]:
    def script(_t: float) -> tuple[float, ...]:
        return spread, 0.5, 0.46, roll, 1.0

    return script


def make(kind: str) -> Any:
    return SyntheticTracker(script=default_script) if kind == "demo" else SyntheticTracker()
