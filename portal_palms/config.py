"""Every tunable in one place.

Nothing in here allocates at runtime: colours are module-level float32 arrays
built once at import, and the scalars are plain Python floats that get bound to
locals inside hot loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

F32 = np.float32
Vec = NDArray[np.float32]

# --------------------------------------------------------------------------- paths

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = PACKAGE_ROOT.parent
ASSETS: Final[Path] = PROJECT_ROOT / "assets"
MODEL_PATH: Final[Path] = ASSETS / "models" / "hand_landmarker.task"
FONT_ATLAS_PATH: Final[Path] = ASSETS / "fonts" / "atlas.bin"
FONT_METRICS_PATH: Final[Path] = ASSETS / "fonts" / "atlas.json"
WORDS_PATH: Final[Path] = PACKAGE_ROOT / "content" / "words.json"


def resolve_asset_root() -> Path:
    """Assets live next to the package in a checkout and next to the exe in a bundle."""
    import sys

    bundle = getattr(sys, "_MEIPASS", None)
    if bundle is not None:
        return Path(bundle) / "assets"
    return ASSETS


# --------------------------------------------------------------------------- render target

# The whole app is composed into one offscreen frame and then letterboxed into
# the window, so a recording is just a readback of the framebuffer we already
# drew. The default is vertical because that is the shape the clips need to be.
#
# Not Final: --size / --landscape rewrite these once, before anything reads
# them. Everything downstream reads at construction or per frame, so the whole
# app reshapes from this one call. The only rule is that set_render_size() must
# run before App is built.
RENDER_W: int = 1080
RENDER_H: int = 1920
RENDER_ASPECT: float = RENDER_W / RENDER_H

# The type scale was designed against a 1920-tall frame. Sizes below are quoted
# at that reference and multiplied by this, so a shorter frame gets
# proportionally smaller text rather than text that swallows the layout.
UI_SCALE: float = 1.0

REFERENCE_H: Final[int] = 1920


def set_render_size(width: int, height: int) -> None:
    """Set the composition size. Must be called before the App is constructed."""
    global RENDER_W, RENDER_H, RENDER_ASPECT, UI_SCALE
    RENDER_W = max(int(width), 320)
    RENDER_H = max(int(height), 320)
    RENDER_ASPECT = RENDER_W / RENDER_H
    # Geometric mean of "scale with height" and "scale with the short side".
    # Pure height scaling makes a landscape frame's type look timid; pure
    # short-side scaling makes it swallow the layout. This lands between.
    UI_SCALE = ((RENDER_H / REFERENCE_H) * (min(RENDER_W, RENDER_H) / 1080.0)) ** 0.5


def is_portrait() -> bool:
    return RENDER_ASPECT < 1.0


# The interior scene is viewed through a wide ellipse, so it is rendered
# landscape and at a fraction of the main target: 345k pixels against 2.07M.
# Composing it into a 9:16 buffer would spend most of the fill rate on pixels
# the ellipse mask never shows.
SCENE_W: Final[int] = 720
SCENE_H: Final[int] = 480
SCENE_ASPECT: Final[float] = SCENE_H / SCENE_W
# How much of the scene buffer the fully open ellipse covers.
SCENE_ZOOM: Final[float] = 2.2

BLOOM_DIV: Final[int] = 4
BLOOM_DIV_LOW: Final[int] = 8

WINDOW_W: Final[int] = 600
WINDOW_H: Final[int] = 1000
WINDOW_TITLE: Final[str] = "Portal Palms"

# --------------------------------------------------------------------------- camera

CAM_W: Final[int] = 640
CAM_H: Final[int] = 480
CAM_FPS: Final[int] = 30
CAM_INDEX: Final[int] = 0

# --------------------------------------------------------------------------- tracking

NUM_HANDS: Final[int] = 2
NUM_LANDMARKS: Final[int] = 21

MIN_DETECTION_CONFIDENCE: Final[float] = 0.5
MIN_PRESENCE_CONFIDENCE: Final[float] = 0.5
MIN_TRACKING_CONFIDENCE: Final[float] = 0.5

# One-Euro. Tuned so a still hand is dead quiet and a fast hand does not lag.
OE_MIN_CUTOFF: Final[float] = 1.0
OE_BETA: Final[float] = 0.007
OE_D_CUTOFF: Final[float] = 1.0
# beta multiplies the speed estimate, so its value is only meaningful together
# with the units that speed is in. The canonical 0.007 comes from the One-Euro
# paper, where the signal is in *pixels*. MediaPipe hands us [0,1] normalised
# coordinates, which makes every speed ~640x smaller and beta ~640x too weak:
# the adaptive term stops doing anything and the filter degenerates into a flat
# 1Hz low-pass, which costs about 100px of lag on a fast hand. Converting the
# derivative back into pixels restores the behaviour the constant was tuned for.
OE_SPEED_SCALE: Final[float] = float(CAM_W)

# A hand is considered gone this long after its last sighting.
HAND_LOST_SECONDS: Final[float] = 0.35
# Dead-reckoning between 30Hz detections, clamped so a dropped detection cannot
# fling a landmark across the frame.
EXTRAPOLATE_MAX_SECONDS: Final[float] = 0.08
EXTRAPOLATE_MAX_DELTA: Final[float] = 0.06

# Landmark indices.
WRIST: Final[int] = 0
THUMB_TIP: Final[int] = 4
INDEX_MCP: Final[int] = 5
INDEX_TIP: Final[int] = 8
MIDDLE_MCP: Final[int] = 9
MIDDLE_TIP: Final[int] = 12
RING_MCP: Final[int] = 13
RING_TIP: Final[int] = 16
PINKY_MCP: Final[int] = 17
PINKY_TIP: Final[int] = 20

# --------------------------------------------------------------------------- portal gesture

# All distances below are in hand-sizes: distance(wrist, middle_mcp). That makes
# the gesture behave the same at 0.5m and at 2m from the camera.
#
# For calibration: wrists touching reads about 0.9 hand-sizes, a relaxed frame
# about 3.0, and a comfortable full spread about 4.6. Press `d` for the HUD to
# see the live value if a particular camera or room needs these nudged.
PORTAL_SEAM_SPREAD: Final[float] = 1.4  # below this the portal is a thin seam
PORTAL_OPEN_SPREAD: Final[float] = 4.4  # at this spread the portal is fully open
PORTAL_ENTER_SPREAD: Final[float] = 4.1  # hysteresis: cross up to arm the hold
PORTAL_EXIT_SPREAD: Final[float] = 3.2  # hysteresis: fall below to disarm
PORTAL_HOLD_MS: Final[float] = 1500.0  # hold fully open this long to lock

PORTAL_IMPLODE_MS: Final[float] = 420.0

# The ring follows the average roll of the two hands.
PORTAL_ROTATION_SMOOTH: Final[float] = 0.18

# --------------------------------------------------------------------------- palette


def _rgb(h: int) -> Vec:
    return np.array(
        ((h >> 16 & 0xFF) / 255.0, (h >> 8 & 0xFF) / 255.0, (h & 0xFF) / 255.0),
        dtype=F32,
    )


BASE: Final[Vec] = _rgb(0x0B0F1A)
GREEN: Final[Vec] = _rgb(0x3FD68C)
YELLOW: Final[Vec] = _rgb(0xFFC531)
MAGENTA: Final[Vec] = _rgb(0xFF3D7F)
OFFWHITE: Final[Vec] = _rgb(0xF2F4F0)
DIM: Final[Vec] = _rgb(0x5A6472)

# --------------------------------------------------------------------------- performance

TARGET_FPS: Final[int] = 60
FRAME_BUDGET_MS: Final[float] = 1000.0 / TARGET_FPS
SPARKLINE_LEN: Final[int] = 120

PARTICLE_COUNT: Final[int] = 4096
PARTICLE_COUNT_LOW: Final[int] = 2048

# Auto low-power trip: rolling average over this window above this threshold.
LOWPOWER_TRIP_MS: Final[float] = 20.0
LOWPOWER_TRIP_SECONDS: Final[float] = 2.0
LOWPOWER_FPS: Final[int] = 30

MAX_SCENE_INSTANCES: Final[int] = 512
MAX_TEXT_GLYPHS: Final[int] = 512

# --------------------------------------------------------------------------- recording

RECORD_MAX_SECONDS: Final[float] = 30.0
RECORD_COUNTDOWN: Final[int] = 3
RECORD_FPS: Final[int] = 30
