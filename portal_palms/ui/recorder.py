"""Clip recording: asynchronous PBO readback piped to ffmpeg.

The frame is already composed into an offscreen 1080x1920 target for the
letterboxed blit, so recording costs one extra readback and no second render
pass.

That readback is the part that has to be careful. A plain `fbo.read()` is a
synchronous glReadPixels: it stalls the pipeline until the GPU catches up, and
at 1080x1920 that is tens of milliseconds -- a dropped frame every time.
Reading into a GL buffer instead makes the transfer asynchronous, and reading
*last* frame's buffer while this frame's is still in flight means the CPU never
waits on the GPU at all.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Final

import moderngl
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.ui.progress import clips_dir

# Deliberately not a module constant: --size rewrites the composition before
# the App is built, and an import-time value would be the stale default.
POOL: Final = 6  # in-flight frames between the loop and the encoder thread


def find_ffmpeg() -> str | None:
    """Prefer the bundled binary, fall back to whatever is on PATH."""
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return shutil.which("ffmpeg")


class Recorder:
    """Records the composed frame to an mp4. Never stalls the render loop."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.available = find_ffmpeg() is not None
        self.recording = False
        self.frames = 0
        self.seconds = 0.0
        self.error: str | None = None
        self.path: Path | None = None

        # Two GL buffers ping-ponged: glReadPixels targets one while the other
        # (issued last frame, now complete) is copied out.
        frame_bytes = C.RENDER_W * C.RENDER_H * 3
        self.pbo = [ctx.buffer(reserve=frame_bytes), ctx.buffer(reserve=frame_bytes)]
        self.slot = 0
        self.primed = False

        self._pool: list[NDArray[np.uint8]] = [
            np.zeros(frame_bytes, dtype=np.uint8) for _ in range(POOL)
        ]
        self._free: deque[int] = deque(range(POOL))
        self._queued: deque[int] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self.dropped = 0

    # ------------------------------------------------------------------ control

    def start(self) -> bool:
        if self.recording:
            return False
        exe = find_ffmpeg()
        if exe is None:
            self.error = "ffmpeg not found"
            return False

        out_dir = clips_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        # Sequence numbers rather than a timestamp: this file deliberately has
        # no clock dependency, and the preview screen names the clip anyway.
        n = 1
        while (out_dir / f"portal-palms-{n:03d}.mp4").exists():
            n += 1
        self.path = out_dir / f"portal-palms-{n:03d}.mp4"

        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{C.RENDER_W}x{C.RENDER_H}", "-r", str(C.RECORD_FPS),
            "-i", "-",
            # The framebuffer comes back bottom-up; let ffmpeg flip it rather
            # than spending CPU on np.flipud every frame.
            "-vf", "vflip",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.path),
        ]  # fmt: skip
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as exc:
            self.error = str(exc)
            return False

        self._stop.clear()
        self._free.clear()
        self._free.extend(range(POOL))
        self._queued.clear()
        self.frames = 0
        self.seconds = 0.0
        self.dropped = 0
        self.primed = False
        self.error = None
        self._thread = threading.Thread(target=self._pump, name="portal-encode", daemon=True)
        self._thread.start()
        self.recording = True
        return True

    def stop(self) -> Path | None:
        if not self.recording:
            return None
        self.recording = False
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.returncode not in (0, None):
                self.error = "ffmpeg failed"
        return self.path

    def discard(self) -> None:
        p = self.path
        self.path = None
        if p is not None:
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------------ per frame

    def capture(self, fbo: moderngl.Framebuffer, dt: float) -> None:
        """Issue this frame's readback and copy out the previous one."""
        if not self.recording:
            return
        self.seconds += dt
        if self.seconds >= C.RECORD_MAX_SECONDS:
            self.recording = False
            self._stop.set()
            self._wake.set()
            return

        nxt = 1 - self.slot
        fbo.read_into(self.pbo[nxt], components=3)  # asynchronous into a GL buffer

        if self.primed:
            with self._lock:
                idx = self._free.popleft() if self._free else -1
            if idx < 0:
                # The encoder is behind. Skipping a frame keeps the render loop
                # at rate; a stalled loop would be far more visible than a
                # dropped frame in the clip.
                self.dropped += 1
            else:
                self.pbo[self.slot].read_into(self._pool[idx])
                with self._lock:
                    self._queued.append(idx)
                self._wake.set()
                self.frames += 1
        self.slot = nxt
        self.primed = True

    # ------------------------------------------------------------------ encoder

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        while True:
            with self._lock:
                idx = self._queued.popleft() if self._queued else -1
            if idx < 0:
                if self._stop.is_set():
                    break
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            try:
                proc.stdin.write(self._pool[idx])
            except (BrokenPipeError, OSError):
                self.error = "encoder closed"
                break
            finally:
                with self._lock:
                    self._free.append(idx)

    # ------------------------------------------------------------------ misc

    def release(self) -> None:
        if self.recording:
            self.stop()
        for b in self.pbo:
            b.release()


def reveal(path: Path) -> None:
    """Open the platform file browser with the clip selected."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError:
        pass
