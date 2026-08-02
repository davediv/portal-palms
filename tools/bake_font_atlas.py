"""Bake the three app faces into one signed-distance-field atlas.

Run once, offline:

    uv run python tools/bake_font_atlas.py

Output is assets/fonts/atlas.bin (raw single-channel bytes) plus atlas.json.
Neither Pillow nor this script is imported by the app -- shipping a pre-baked
atlas keeps Pillow out of the runtime dependency list entirely and takes glyph
rasterisation off the cold-start path.

A signed distance field is used rather than a plain coverage bitmap because the
target word is drawn very large and the HUD very small from the same texture.
An SDF stays crisp across that whole range from a single 1024x1024 page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"

ATLAS_SIZE = 1024

ASCII = "".join(chr(c) for c in range(32, 127))
# The display face is only ever set in caps, so it does not need lowercase and
# the space it saves buys a much higher bake resolution for the big word.
DISPLAY_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;'!?-+%/()"


class Face(NamedTuple):
    name: str
    file: str
    size: int
    spread: int
    chars: str


# The display face carries the target word at ~185 render px, so it is baked
# far larger than the others: at 72 the SDF quantisation showed as visible
# faceting on the curves once magnified 2.6x, and this thing gets filmed off a
# screen. The body and mono faces are never magnified anything like as hard, so
# they give the page space back.
FACES = (
    Face("display", "BarlowCondensed-Bold.ttf", 104, 13, DISPLAY_CHARS),
    Face("body", "Barlow-Regular.ttf", 40, 6, ASCII),
    Face("bodybold", "Barlow-SemiBold.ttf", 40, 6, ASCII),
    Face("mono", "JetBrainsMono-Variable.ttf", 28, 5, ASCII),
)


def signed_distance(mask: NDArray[np.uint8], spread: int) -> NDArray[np.uint8]:
    """Euclidean SDF from a coverage bitmap, encoded to 0..255 with 0.5 at the edge."""
    inside = (mask > 127).astype(np.uint8)
    outside = 1 - inside
    # distanceTransform measures from every non-zero pixel to the nearest zero.
    d_out = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    sd = d_out - d_in
    sd = np.clip(sd / (2.0 * spread) + 0.5, 0.0, 1.0)
    return (sd * 255.0).astype(np.uint8)


def bake() -> int:
    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE), dtype=np.uint8)
    metrics: dict[str, object] = {"size": ATLAS_SIZE, "faces": {}}

    shelf_y = 0
    for face in FACES:
        path = FONT_DIR / face.file
        if not path.exists():
            print(f"missing font: {path}", file=sys.stderr)
            return 1
        font = ImageFont.truetype(str(path), face.size)
        ascent, descent = font.getmetrics()

        glyphs: dict[str, list[float]] = {}
        pad = face.spread + 2
        cursor_x = 0
        row_h = 0

        for ch in face.chars:
            bbox = font.getbbox(ch)
            advance = float(font.getlength(ch))
            gw: int = int(max(bbox[2] - bbox[0], 0))
            gh: int = int(max(bbox[3] - bbox[1], 0))

            if gw == 0 or gh == 0:  # space and friends carry advance only
                glyphs[ch] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, advance / face.size]
                continue

            cw, chh = int(gw) + pad * 2, int(gh) + pad * 2
            tile = Image.new("L", (cw, chh), 0)
            ImageDraw.Draw(tile).text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=255)
            sdf = signed_distance(np.asarray(tile), face.spread)

            if cursor_x + cw > ATLAS_SIZE:
                cursor_x = 0
                shelf_y += row_h
                row_h = 0
            if shelf_y + chh > ATLAS_SIZE:
                print("atlas overflow -- lower a face size", file=sys.stderr)
                return 1

            atlas[shelf_y : shelf_y + chh, cursor_x : cursor_x + cw] = sdf

            # Everything below is stored in em units (divided by the bake size)
            # so the renderer can scale to any pixel size with one multiply.
            glyphs[ch] = [
                cursor_x / ATLAS_SIZE,
                shelf_y / ATLAS_SIZE,
                cw / ATLAS_SIZE,
                chh / ATLAS_SIZE,
                (bbox[0] - pad) / face.size,  # left bearing, em
                (bbox[1] - pad) / face.size,  # top bearing from baseline-ascent, em
                advance / face.size,
            ]
            cursor_x += cw
            row_h = max(row_h, chh)

        shelf_y += row_h
        metrics["faces"][face.name] = {  # type: ignore[index]
            "glyphs": glyphs,
            "spread": face.spread / face.size,
            "ascent": ascent / face.size,
            "descent": descent / face.size,
            "line": (ascent + descent) / face.size,
            "size": face.size,
        }
        print(f"{face.name:9s} {len(glyphs):3d} glyphs, shelf now y={shelf_y}")

    (FONT_DIR / "atlas.bin").write_bytes(atlas.tobytes())
    (FONT_DIR / "atlas.json").write_text(json.dumps(metrics, separators=(",", ":")))
    used = shelf_y / ATLAS_SIZE
    print(f"wrote atlas.bin ({atlas.nbytes} bytes), {used:.0%} of the page used")

    cv2.imwrite(str(FONT_DIR / "atlas_preview.png"), atlas)
    return 0


if __name__ == "__main__":
    raise SystemExit(bake())
