"""Frame metrics and the debug HUD, drawn with the app's own GL text path.

Shipped in the release build, toggled with `d`. Everything it needs is measured
off time.perf_counter_ns rather than assumed, and the sparkline is what actually
proves the "no allocation spikes over a five minute session" claim -- a GC pause
shows up as an isolated spike in an otherwise flat trace, and you cannot see
that in an average.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Final

import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.render.text import ALIGN_LEFT, ALIGN_RIGHT, NumBuf, ShapeRenderer, TextRenderer

F32 = np.float32
KB: Final = 1024.0
# Render-frame pixels. The frame is letterboxed down by about 0.52, so this
# lands near 14 display px -- small, but this is the one overlay allowed to be.
HUD_PX: Final = 27.0


def _rss_mb() -> float:
    """Peak resident set size in MB.

    Peak rather than current on purpose: the question this number answers is
    "is the steady state leaking?", and a peak that stops climbing answers it
    definitively. Getting current RSS portably would mean a dependency, and the
    dependency budget is spent.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in kilobytes; the BSDs and macOS report bytes.
    if sys.platform == "linux":
        return raw / KB
    return raw / (KB * KB)


class Metrics:
    """Rolling frame timings. Fixed-size arrays; nothing here allocates."""

    __slots__ = (
        "_rss_at",
        "cpu_ms",
        "draw_calls",
        "dropped",
        "frame",
        "frame_ms",
        "gpu_wait_ms",
        "head",
        "history",
        "particles",
        "rolling_ms",
        "rss_mb",
        "since_trip",
    )

    def __init__(self) -> None:
        self.history: NDArray[np.float32] = np.zeros(C.SPARKLINE_LEN, dtype=F32)
        self.head = 0
        self.frame = 0
        self.frame_ms = 0.0
        self.cpu_ms = 0.0
        self.gpu_wait_ms = 0.0
        self.rolling_ms = 16.6
        self.draw_calls = 0
        self.particles = 0
        self.dropped = 0
        self.rss_mb = 0.0
        self._rss_at = 0.0
        self.since_trip = 0.0

    def push(self, frame_ms: float, cpu_ms: float, now: float) -> None:
        self.frame_ms = frame_ms
        self.cpu_ms = cpu_ms
        self.history[self.head] = frame_ms
        self.head = (self.head + 1) % C.SPARKLINE_LEN
        self.frame += 1
        # Exponential rolling average with a ~1s time constant at 60fps.
        self.rolling_ms += (frame_ms - self.rolling_ms) * 0.016
        if frame_ms > C.FRAME_BUDGET_MS * 1.5:
            self.dropped += 1
        if now - self._rss_at > 1.0:
            self._rss_at = now
            self.rss_mb = _rss_mb()

    @property
    def fps(self) -> float:
        return 1000.0 / self.frame_ms if self.frame_ms > 0.01 else 0.0


class LowPowerGovernor:
    """Trips into low-power mode when the frame time will not hold, and says so."""

    __slots__ = ("_over_since", "active", "announced_at", "forced")

    def __init__(self) -> None:
        self.active = False
        self.forced = False
        self._over_since = -1.0
        self.announced_at = -100.0

    def update(self, rolling_ms: float, now: float) -> bool:
        """Returns True on the frame the mode changes."""
        if self.forced:
            return False
        if self.active:
            return False
        if rolling_ms > C.LOWPOWER_TRIP_MS:
            if self._over_since < 0.0:
                self._over_since = now
            elif now - self._over_since >= C.LOWPOWER_TRIP_SECONDS:
                self.active = True
                self.announced_at = now
                return True
        else:
            self._over_since = -1.0
        return False

    def toggle(self, now: float) -> None:
        self.active = not self.active
        self.forced = True
        self.announced_at = now
        self._over_since = -1.0


class Hud:
    """The debug overlay. Drawn from the same SDF atlas as everything else."""

    X: Final = 0.035
    W: Final = 0.50

    def __init__(self, text: TextRenderer, shapes: ShapeRenderer) -> None:
        self.text = text
        self.shapes = shapes
        self.num = NumBuf(24)
        self.visible = False
        self._spark = np.zeros(C.SPARKLINE_LEN, dtype=F32)

    def draw(
        self,
        m: Metrics,
        detect_ms: float,
        delegate: str,
        spread: float,
        low_power: bool,
    ) -> None:
        if not self.visible:
            return
        t = self.text
        s = self.shapes
        mono = t.mono
        num = self.num
        col = (0.949, 0.957, 0.941)
        dim = (0.45, 0.52, 0.60)
        good = (0.247, 0.839, 0.549)
        bad = (1.0, 0.239, 0.498)

        rows = 9
        line = 0.0225
        top = 0.028
        h = line * rows + 0.055
        s.rect(self.X + self.W * 0.5, top + h * 0.5, self.W * 0.5, h * 0.5,
               (0.02, 0.03, 0.06), 0.86, 0.012)  # fmt: skip

        y = top + 0.020
        left = self.X + 0.018
        right = self.X + self.W - 0.018

        def row(label: str, value: str, c: tuple[float, float, float]) -> None:
            nonlocal y
            t.add(label, mono, left, y, HUD_PX, dim, 1.0, ALIGN_LEFT)
            t.add(value, mono, right, y, HUD_PX, c, 1.0, ALIGN_RIGHT)
            y += line

        over = m.frame_ms > C.FRAME_BUDGET_MS
        row("frame ms", num.fixed(0, m.frame_ms, 2), bad if over else good)
        row("fps", num.fixed(1, m.fps, 0), bad if over else good)
        row("cpu ms", num.fixed(2, m.cpu_ms, 2), col)
        row("detect ms", num.fixed(3, detect_ms, 1), col)
        row("draw calls", num.fixed(4, m.draw_calls, 0), bad if m.draw_calls > 30 else col)
        row("particles", num.fixed(5, m.particles, 0), col)
        row("rss mb max", num.fixed(6, m.rss_mb, 1), col)
        row("dropped", num.fixed(7, m.dropped, 0), bad if m.dropped else col)
        row("spread", num.fixed(8, spread, 2), col)

        t.add(delegate, mono, left, top + 0.004, HUD_PX * 0.82, good, 1.0, ALIGN_LEFT)
        if low_power:
            t.add("LOW POWER", mono, right, top + 0.004, HUD_PX * 0.82, (1.0, 0.773, 0.192), 1.0,
                  ALIGN_RIGHT)  # fmt: skip

        self._sparkline(m, top + h + 0.012)

    def _sparkline(self, m: Metrics, y: float) -> None:
        """120 frames of frame time. Bars over budget are drawn hot."""
        s = self.shapes
        hist = m.history
        np.copyto(self._spark, hist)
        peak = float(self._spark.max())
        if peak < C.FRAME_BUDGET_MS * 1.4:
            peak = C.FRAME_BUDGET_MS * 1.4

        h = 0.038
        s.rect(self.X + self.W * 0.5, y + h * 0.5, self.W * 0.5, h * 0.5,
               (0.02, 0.03, 0.06), 0.86, 0.008)  # fmt: skip
        # The 16.6ms budget line, so a glance tells you pass or fail.
        budget_y = y + h - (C.FRAME_BUDGET_MS / peak) * h
        s.rect(self.X + self.W * 0.5, budget_y, self.W * 0.5, 0.0006,
               (0.247, 0.839, 0.549), 0.5)  # fmt: skip

        n = C.SPARKLINE_LEN
        bw = self.W / n
        head = m.head
        for i in range(n):
            v = float(self._spark[(head + i) % n])
            if v <= 0.0:
                continue
            bh = min(v / peak, 1.0) * h
            hot = v > C.FRAME_BUDGET_MS
            s.rect(
                self.X + (i + 0.5) * bw,
                y + h - bh * 0.5,
                bw * 0.34,
                bh * 0.5,
                (1.0, 0.239, 0.498) if hot else (0.45, 0.60, 0.75),
                0.95 if hot else 0.7,
            )


def now_ms() -> float:
    return time.perf_counter_ns() * 1e-6
