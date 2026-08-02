"""Camera capture and hand landmark detection on a worker thread.

Architecture, because it is the whole point of this file:

    worker thread (camera rate, ~30Hz)      main thread (display rate, 60Hz+)
    -----------------------------------     ---------------------------------
    VideoCapture.read   (releases GIL)      HandState.update(now)
    cvtColor -> preallocated RGB buffer       - index swap under a lock
    detect_for_video    (releases GIL)        - extrapolate from last velocity
    mirror + One-Euro into a free slot      render
    publish: one index swap under a lock

Nothing is ever allocated to hand a result across. Three preallocated slots
rotate between the writer and the reader, so both critical sections are two
integer assignments and the reader can touch its slot with no lock held at all.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Final, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms import mediapipe_slim

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from collections.abc import Callable

F32 = np.float32
Arr = NDArray[np.float32]

_TWO_PI: Final[float] = 6.283185307179586


# --------------------------------------------------------------------------- mirror


def mirror_landmarks(landmarks: Arr) -> Arr:
    """Flip landmark x in place to (1 - x) and return the same array.

    The preview is displayed mirrored, so every consumer downstream of detection
    must see mirrored coordinates. Doing it anywhere later than here produces an
    app that feels subtly wrong in a way that is very hard to trace.

    Only x is touched: y already points down in both the image and our screen
    space, and z is a depth in hand-relative units that mirroring does not
    affect.
    """
    x = landmarks[..., 0]
    np.subtract(1.0, x, out=x)
    return landmarks


# --------------------------------------------------------------------------- One-Euro


class OneEuroFilter:
    """One-Euro filter vectorised across the whole (hands, landmarks, axes) array.

    Every temporary is preallocated. There is no Python-level loop over the 126
    scalars, and no array is allocated after __init__.
    """

    __slots__ = (
        "_a",
        "_alpha",
        "_beta",
        "_cutoff",
        "_dcutoff",
        "_dx",
        "_dx_prev",
        "_initialised",
        "_mincutoff",
        "_scratch",
        "_x_prev",
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        mincutoff: float = C.OE_MIN_CUTOFF,
        beta: float = C.OE_BETA,
        dcutoff: float = C.OE_D_CUTOFF,
    ) -> None:
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._x_prev = np.zeros(shape, dtype=F32)
        self._dx_prev = np.zeros(shape, dtype=F32)
        self._dx = np.zeros(shape, dtype=F32)
        self._cutoff = np.zeros(shape, dtype=F32)
        self._alpha = np.zeros(shape, dtype=F32)
        self._a = np.zeros(shape, dtype=F32)
        self._scratch = np.zeros(shape, dtype=F32)
        # One flag per leading-axis entry (one per hand), so a hand that leaves
        # and comes back restarts clean instead of lerping from a stale pose.
        self._initialised = np.zeros(shape[0], dtype=bool)

    def reset_all(self) -> None:
        self._initialised[:] = False

    def filter(self, x: Arr, dt: float, out: Arr, valid: NDArray[np.bool_]) -> None:
        """Filter `x` into `out`. `out` may alias `x`.

        `valid` marks which leading-axis entries carry real data this tick. An
        entry that is invalid is left uninitialised, so when it comes back it
        snaps to its true position instead of easing in from wherever it was
        last seen — a hand re-entering frame must not fly in from the corner.
        """
        if dt <= 0.0:
            dt = 1.0 / C.CAM_FPS

        fresh = valid & ~self._initialised
        if fresh.any():
            self._x_prev[fresh] = x[fresh]
            self._dx_prev[fresh] = 0.0

        inv_dt = 1.0 / dt

        # dx = (x - x_prev) / dt
        np.subtract(x, self._x_prev, out=self._dx)
        self._dx *= inv_dt

        # dx_hat = lerp(dx_prev, dx, alpha(dcutoff))
        a_d = dt / ((1.0 / (_TWO_PI * self._dcutoff)) + dt)
        self._dx -= self._dx_prev
        self._dx *= a_d
        self._dx_prev += self._dx  # _dx_prev is now dx_hat

        # cutoff = mincutoff + beta * |dx_hat|, with the speed put back into the
        # pixel units beta was tuned in. See OE_SPEED_SCALE in config.
        np.abs(self._dx_prev, out=self._cutoff)
        self._cutoff *= self._beta * C.OE_SPEED_SCALE
        self._cutoff += self._mincutoff

        # alpha = dt / (1/(2*pi*cutoff) + dt)
        np.multiply(self._cutoff, _TWO_PI, out=self._alpha)
        np.reciprocal(self._alpha, out=self._alpha)
        self._alpha += dt
        np.divide(dt, self._alpha, out=self._alpha)

        # x_hat = x_prev + alpha * (x - x_prev)
        np.subtract(x, self._x_prev, out=self._scratch)
        self._scratch *= self._alpha
        self._x_prev += self._scratch
        np.copyto(out, self._x_prev)
        self._initialised[:] = valid


# --------------------------------------------------------------------------- slots


class _Slot:
    """One preallocated detection result. Never handed across a queue, only swapped."""

    __slots__ = ("landmarks", "present", "rgb", "t", "velocity", "world_scale")

    def __init__(self, cam_h: int, cam_w: int) -> None:
        self.landmarks: Arr = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self.velocity: Arr = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self.present: NDArray[np.bool_] = np.zeros(C.NUM_HANDS, dtype=bool)
        # Hand size in normalised units: distance(wrist, middle MCP). Every
        # gesture threshold is expressed in multiples of this.
        self.world_scale: Arr = np.zeros(C.NUM_HANDS, dtype=F32)
        self.rgb: NDArray[np.uint8] = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        self.t: float = 0.0


class TripleBuffer:
    """Single-writer / single-reader handoff with no allocation and no copying.

    Three slots rotate: the writer always owns one, the reader always owns one,
    and the third is either the newest published result or a spare. Both
    critical sections are index assignments.
    """

    __slots__ = ("_lock", "_read", "_ready", "_seq", "_slots", "_write")

    def __init__(self, cam_h: int, cam_w: int) -> None:
        self._slots = (_Slot(cam_h, cam_w), _Slot(cam_h, cam_w), _Slot(cam_h, cam_w))
        self._write = 0
        self._ready = 1
        self._read = 2
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def slots(self) -> tuple[_Slot, _Slot, _Slot]:
        """All three slots. Init-time seeding only -- never touch during a run."""
        return self._slots

    @property
    def write_slot(self) -> _Slot:
        return self._slots[self._write]

    @property
    def read_slot(self) -> _Slot:
        return self._slots[self._read]

    def publish(self) -> None:
        """Writer side: hand off the slot just written and take the spare."""
        with self._lock:
            self._write, self._ready = self._ready, self._write
            self._seq += 1

    def acquire(self) -> int:
        """Reader side: take the newest published slot. Returns the sequence number."""
        with self._lock:
            self._read, self._ready = self._ready, self._read
            return self._seq


# --------------------------------------------------------------------------- camera


# Keyed rather than branched: mypy narrows sys.platform to whichever host it
# is running on and reports the other branches as unreachable, which would hide
# real errors in the code that actually ships to those platforms.
_CAPTURE_BACKENDS: Final[dict[str, int]] = {
    "darwin": int(cv2.CAP_AVFOUNDATION),
    "win32": int(cv2.CAP_MSMF),
}


def _backend_for_platform() -> int:
    return _CAPTURE_BACKENDS.get(sys.platform, int(cv2.CAP_V4L2))


def open_camera(index: int = C.CAM_INDEX) -> Any:
    """Open the capture device with a shallow buffer and MJPG at 640x480/30.

    A deep driver buffer is the single largest source of hand-to-screen latency
    and users read that lag as broken tracking, so BUFFERSIZE 1 matters more
    than it looks.
    """
    cap = cv2.VideoCapture(index, _backend_for_platform())
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
    cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# --------------------------------------------------------------------------- tracker


class HandTracker:
    """Owns the camera and the landmarker on a worker thread."""

    def __init__(self, model_path: str, mirror: bool = True) -> None:
        self.model_path = model_path
        self.mirror = mirror

        self.cam_w = C.CAM_W
        self.cam_h = C.CAM_H
        self.buffers: TripleBuffer | None = None

        self.error: str | None = None
        self.camera_ready = False
        self.model_ready = False
        self.delegate_name = "CPU"

        self.detect_ms = 0.0
        self.capture_ms = 0.0
        self.detections = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cap: Any = None
        self._landmarker: Any = None
        self._filter: OneEuroFilter | None = None
        self._on_ready: Callable[[], None] | None = None

        # Worker-local scratch, allocated once.
        self._raw = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._smoothed = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._prev_smoothed = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._swap_tmp = np.zeros((C.NUM_LANDMARKS, 3), dtype=F32)
        self._had = np.zeros(C.NUM_HANDS, dtype=bool)
        self._prev_t = 0.0
        self._t0_ns = 0
        self._last_ts_ms = -1

    # ------------------------------------------------------------------ lifecycle

    def start(self, on_ready: Callable[[], None] | None = None) -> None:
        self._on_ready = on_ready
        self._thread = threading.Thread(target=self._run, name="portal-tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            self._landmarker.close()

    # ------------------------------------------------------------------ worker

    def _build_landmarker(self) -> Any:
        # Keep matplotlib and sounddevice out; see mediapipe_slim for why this
        # is worth 12 seconds of packaged cold start.
        mediapipe_slim.install()
        # Imported here, not at module scope: `import mediapipe` costs ~0.5s and
        # must happen behind the camera explainer, never before the window opens.
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        def make(delegate: Any) -> Any:
            return HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path, delegate=delegate),
                    running_mode=RunningMode.VIDEO,
                    num_hands=C.NUM_HANDS,
                    min_hand_detection_confidence=C.MIN_DETECTION_CONFIDENCE,
                    min_hand_presence_confidence=C.MIN_PRESENCE_CONFIDENCE,
                    min_tracking_confidence=C.MIN_TRACKING_CONFIDENCE,
                )
            )

        # The GPU delegate is only genuinely usable on Linux in the published
        # wheels. On macOS it *constructs* without complaint and then aborts the
        # entire process partway through the first inference, inside native
        # code, where there is no Python exception to catch -- so wrapping
        # construction in try/except is not protection, it is a crash with extra
        # steps. Only Linux gets to try. Everywhere else goes straight to CPU,
        # which is what the 30Hz detection budget is sized around anyway.
        if sys.platform.startswith("linux"):
            try:
                lm = make(BaseOptions.Delegate.GPU)
                self.delegate_name = "GPU"
                return lm
            except Exception:  # fall back to CPU rather than fail to start
                pass
        lm = make(BaseOptions.Delegate.CPU)
        self.delegate_name = "CPU"
        return lm

    def _run(self) -> None:
        try:
            self._cap = open_camera()
            if self._cap is None:
                self.error = "camera"
                return
            ok, probe = self._cap.read()
            if not ok or probe is None:
                self.error = "camera"
                return
            self.cam_h, self.cam_w = int(probe.shape[0]), int(probe.shape[1])
            self.buffers = TripleBuffer(self.cam_h, self.cam_w)
            self.camera_ready = True

            self._landmarker = self._build_landmarker()
            self._filter = OneEuroFilter((C.NUM_HANDS, C.NUM_LANDMARKS, 3))
            self.model_ready = True
            if self._on_ready is not None:
                self._on_ready()
        except Exception as exc:
            self.error = str(exc)
            return

        self._t0_ns = time.perf_counter_ns()
        self._loop()

    def _loop(self) -> None:
        import mediapipe as mp

        cap = self._cap
        landmarker = self._landmarker
        buffers = self.buffers
        assert buffers is not None
        fmt = mp.ImageFormat.SRGB
        cvt = cv2.cvtColor
        bgr2rgb = cv2.COLOR_BGR2RGB
        perf_ns = time.perf_counter_ns
        stop = self._stop

        while not stop.is_set():
            t_cap = perf_ns()
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue

            slot = buffers.write_slot
            if bgr.shape[0] != slot.rgb.shape[0] or bgr.shape[1] != slot.rgb.shape[1]:
                continue
            cvt(bgr, bgr2rgb, dst=slot.rgb)
            t_det = perf_ns()
            self.capture_ms = (t_det - t_cap) * 1e-6

            # detect_for_video wants a strictly increasing millisecond stamp off a
            # single monotonic origin. Wall clock would go backwards on an NTP step.
            ts_ms = (t_det - self._t0_ns) // 1_000_000
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = int(ts_ms)

            # The only per-frame allocation in the app, and it is off the render
            # critical path: mp.Image is a thin wrapper the C++ graph requires.
            image = mp.Image(image_format=fmt, data=slot.rgb)
            try:
                result = landmarker.detect_for_video(image, int(ts_ms))
            except Exception:
                continue
            self.detect_ms = (perf_ns() - t_det) * 1e-6

            self._ingest(result, slot, t_det * 1e-9)
            buffers.publish()
            self.detections += 1

    # ------------------------------------------------------------------ ingest

    def _assign_slots(self, raw: Arr, present: NDArray[np.bool_], n: int) -> None:
        """Pin each detected hand to the slot it occupied last frame.

        MediaPipe's output order is not stable and its handedness label has
        changed mirroring convention between releases, so neither is safe to key
        on. Nearest-previous-wrist assignment is: it never flips mid-gesture,
        and unlike a bare left/right split it does not swap slots when a lone
        hand crosses the centre of the frame.
        """
        prev = self._prev_smoothed
        have_history = self._prev_t > 0.0

        if n == C.NUM_HANDS:
            if not have_history:
                # Cold start: left-to-right, so slot 0 is the visually left hand.
                if raw[0, C.WRIST, 0] > raw[1, C.WRIST, 0]:
                    self._swap_rows(raw)
                return
            keep = self._wrist_dist(raw, 0, prev, 0) + self._wrist_dist(raw, 1, prev, 1)
            swap = self._wrist_dist(raw, 0, prev, 1) + self._wrist_dist(raw, 1, prev, 0)
            if swap < keep:
                self._swap_rows(raw)
            return

        if n == 1:
            # Route the single hand to whichever slot it was in, so the portal
            # collapses around a stable anchor instead of teleporting.
            to_slot = 0
            if have_history and self._had[1]:
                if not self._had[0] or self._wrist_dist(raw, 0, prev, 1) < self._wrist_dist(
                    raw, 0, prev, 0
                ):
                    to_slot = 1
            elif not have_history and raw[0, C.WRIST, 0] > 0.5:
                to_slot = 1
            if to_slot == 1:
                self._swap_rows(raw)
                present[0] = False
                present[1] = True

    def _swap_rows(self, raw: Arr) -> None:
        tmp = self._swap_tmp
        np.copyto(tmp, raw[0])
        raw[0] = raw[1]
        raw[1] = tmp

    @staticmethod
    def _wrist_dist(a: Arr, ai: int, b: Arr, bi: int) -> float:
        dx = float(a[ai, C.WRIST, 0] - b[bi, C.WRIST, 0])
        dy = float(a[ai, C.WRIST, 1] - b[bi, C.WRIST, 1])
        return dx * dx + dy * dy

    def _ingest(self, result: Any, slot: _Slot, now: float) -> None:
        raw = self._raw
        raw[:] = 0.0
        present = slot.present
        present[:] = False

        hands = result.hand_landmarks
        n = len(hands)
        if n > C.NUM_HANDS:
            n = C.NUM_HANDS
        for h in range(n):
            marks = hands[h]
            row = raw[h]
            for i in range(C.NUM_LANDMARKS):
                m = marks[i]
                row[i, 0] = m.x
                row[i, 1] = m.y
                row[i, 2] = m.z
            present[h] = True

        if self.mirror:
            mirror_landmarks(raw)

        self._assign_slots(raw, present, n)

        flt = self._filter
        assert flt is not None
        dt = now - self._prev_t if self._prev_t > 0.0 else 1.0 / C.CAM_FPS
        flt.filter(raw, dt, self._smoothed, present)

        # Velocity in normalised units per second, for dead reckoning between
        # detections on the render thread.
        vel = slot.velocity
        np.subtract(self._smoothed, self._prev_smoothed, out=vel)
        vel *= 1.0 / max(dt, 1e-4)
        for h in range(C.NUM_HANDS):
            # A hand that just appeared has no previous position, so the
            # difference above is meaningless. Extrapolating it would fling the
            # portal across the screen on the first frame of every re-entry.
            if not (present[h] and self._had[h]):
                vel[h] = 0.0

        np.copyto(slot.landmarks, self._smoothed)
        np.copyto(self._prev_smoothed, self._smoothed)
        np.copyto(self._had, present)
        self._prev_t = now
        slot.t = now

        for h in range(C.NUM_HANDS):
            if present[h]:
                dx = self._smoothed[h, C.MIDDLE_MCP, 0] - self._smoothed[h, C.WRIST, 0]
                dy = self._smoothed[h, C.MIDDLE_MCP, 1] - self._smoothed[h, C.WRIST, 1]
                slot.world_scale[h] = max(float(np.hypot(dx, dy)), 1e-3)
            else:
                slot.world_scale[h] = 1e-3


# --------------------------------------------------------------------------- read side


class TrackerSource(Protocol):
    """What the render thread needs from whatever is producing detections.

    Narrow on purpose: it lets the synthetic tracker and the test fakes stand in
    for the real one without either side knowing about the other.
    """

    buffers: TripleBuffer | None


class HandState:
    """The render thread's view of the hands. Updated once per frame, never allocates."""

    __slots__ = (
        "_anchor",
        "_seen_seq",
        "_vel",
        "anchor_t",
        "frame_dirty",
        "landmarks",
        "last_seen",
        "present",
        "rgb",
        "world_scale",
    )

    def __init__(self) -> None:
        self.landmarks: Arr = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self.present: NDArray[np.bool_] = np.zeros(C.NUM_HANDS, dtype=bool)
        self.world_scale: Arr = np.full(C.NUM_HANDS, 0.1, dtype=F32)
        self.last_seen: Arr = np.zeros(C.NUM_HANDS, dtype=F32)
        self.rgb: NDArray[np.uint8] | None = None
        self.frame_dirty = False
        self.anchor_t = 0.0
        self._anchor: Arr = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._vel: Arr = np.zeros((C.NUM_HANDS, C.NUM_LANDMARKS, 3), dtype=F32)
        self._seen_seq = -1

    def update(self, tracker: TrackerSource, now: float) -> None:
        buffers = tracker.buffers
        if buffers is None:
            return
        seq = buffers.acquire()
        if seq != self._seen_seq:
            self._seen_seq = seq
            slot = buffers.read_slot
            np.copyto(self._anchor, slot.landmarks)
            np.copyto(self._vel, slot.velocity)
            np.copyto(self.world_scale, slot.world_scale)
            self.anchor_t = slot.t
            self.rgb = slot.rgb
            self.frame_dirty = True
            for h in range(C.NUM_HANDS):
                if slot.present[h]:
                    self.last_seen[h] = now
        else:
            self.frame_dirty = False

        # Dead reckon forward so 30Hz tracking still looks like 60Hz motion.
        dt = now - self.anchor_t
        if dt > C.EXTRAPOLATE_MAX_SECONDS:
            dt = C.EXTRAPOLATE_MAX_SECONDS
        elif dt < 0.0:
            dt = 0.0
        np.multiply(self._vel, dt, out=self.landmarks)
        np.clip(
            self.landmarks,
            -C.EXTRAPOLATE_MAX_DELTA,
            C.EXTRAPOLATE_MAX_DELTA,
            out=self.landmarks,
        )
        self.landmarks += self._anchor

        for h in range(C.NUM_HANDS):
            self.present[h] = (now - self.last_seen[h]) < C.HAND_LOST_SECONDS

    @property
    def both(self) -> bool:
        return bool(self.present[0] and self.present[1])

    @property
    def any_hand(self) -> bool:
        return bool(self.present[0] or self.present[1])
