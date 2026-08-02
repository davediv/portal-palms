"""SDF text and 2D shapes: two instanced draw calls for the entire UI.

The atlas is baked offline by tools/bake_font_atlas.py, so nothing here
rasterises a glyph, loads a font file, or imports Pillow. Laying out a string
is a walk over cached per-glyph metrics in a numpy array and a few scalar
writes into a preallocated instance buffer.

All UI coordinates are y-down from the top-left of the 1080x1920 render frame,
normalised to 0..1. Sizes are in render-frame pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, TypedDict

import moderngl
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.render import shaders as S
from portal_palms.render.gl import set_uniform

F32 = np.float32
Arr = NDArray[np.float32]

FIRST_CHAR: Final = 32
LAST_CHAR: Final = 126
CHAR_SLOTS: Final = LAST_CHAR - FIRST_CHAR + 1

# Per glyph: u, v, uw, vh, bearing_x, bearing_y, w, h, advance. The last five
# are in em, so one multiply by the pixel size gives layout.
G_U, G_V, G_UW, G_VH, G_BX, G_BY, G_W, G_H, G_ADV = range(9)
GLYPH_FLOATS: Final = 9

TEXT_FLOATS: Final = 14  # rect(4) uv(4) color(4) params(2)
SHAPE_FLOATS: Final = 12  # rect(4) color(4) params(4)

ALIGN_LEFT: Final = 0
ALIGN_CENTER: Final = 1
ALIGN_RIGHT: Final = 2


class FaceBlob(TypedDict):
    """One face's entry in atlas.json, as written by tools/bake_font_atlas.py."""

    glyphs: dict[str, list[float]]
    spread: float
    ascent: float
    descent: float
    line: float
    size: float


class Face:
    """One baked typeface."""

    __slots__ = ("ascent", "descent", "glyphs", "line", "name", "valid")

    def __init__(self, name: str, blob: FaceBlob) -> None:
        self.name = name
        self.ascent = float(blob["ascent"])
        self.descent = float(blob["descent"])
        self.line = float(blob["line"])
        self.glyphs = np.zeros((CHAR_SLOTS, GLYPH_FLOATS), dtype=F32)
        self.valid = np.zeros(CHAR_SLOTS, dtype=bool)

        atlas_size = 1024.0
        bake = float(blob["size"])
        for ch, g in blob["glyphs"].items():
            code = ord(ch)
            if code < FIRST_CHAR or code > LAST_CHAR:
                continue
            i = code - FIRST_CHAR
            row = self.glyphs[i]
            row[G_U] = g[0]
            row[G_V] = g[1]
            row[G_UW] = g[2]
            row[G_VH] = g[3]
            row[G_BX] = g[4]
            row[G_BY] = g[5]
            row[G_W] = g[2] * atlas_size / bake
            row[G_H] = g[3] * atlas_size / bake
            row[G_ADV] = g[6]
            self.valid[i] = True

    def measure(self, text: str, px: float) -> float:
        """Advance width of a string in render-frame pixels."""
        total = 0.0
        glyphs = self.glyphs
        first = FIRST_CHAR
        last = LAST_CHAR
        for ch in text:
            code = ord(ch)
            if code < first or code > last:
                continue
            total += float(glyphs[code - first, G_ADV])
        return total * px


class TextRenderer:
    """One instanced quad call draws every glyph on screen."""

    def __init__(self, ctx: moderngl.Context, assets: Path) -> None:
        self.ctx = ctx
        meta: dict[str, Any] = json.loads((assets / "fonts" / "atlas.json").read_text())
        size = int(meta["size"])
        raw = (assets / "fonts" / "atlas.bin").read_bytes()

        self.tex = ctx.texture((size, size), 1, raw)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = False
        self.tex.repeat_y = False

        self.faces: dict[str, Face] = {
            name: Face(name, blob) for name, blob in meta["faces"].items()
        }
        self.display = self.faces["display"]
        self.body = self.faces["body"]
        self.bodybold = self.faces["bodybold"]
        self.mono = self.faces["mono"]

        self.prog = ctx.program(vertex_shader=S.TEXT_VS, fragment_shader=S.TEXT_FS)
        set_uniform(self.prog, "u_atlas", 0)

        self.data = np.zeros((C.MAX_TEXT_GLYPHS, TEXT_FLOATS), dtype=F32)
        self.buf = ctx.buffer(reserve=self.data.nbytes, dynamic=True)
        quad = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=F32)
        self.quad = ctx.buffer(quad.tobytes())
        self.vao = ctx.vertex_array(
            self.prog,
            [
                (self.quad, "2f", "quad"),
                (self.buf, "4f 4f 4f 2f/i", "i_rect", "i_uv", "i_color", "i_params"),
            ],
        )
        self.n = 0
        self.dropped = 0

    # ------------------------------------------------------------------ layout

    def begin(self) -> None:
        self.n = 0
        self.dropped = 0

    def add(
        self,
        text: str,
        face: Face,
        x: float,
        y: float,
        px: float,
        col: tuple[float, float, float],
        a: float = 1.0,
        align: int = ALIGN_LEFT,
        tracking: float = 0.0,
        weight: float = 0.0,
    ) -> float:
        """Lay out one line. `y` is the top of the ascent. Returns the pen x it ended at."""
        glyphs = face.glyphs
        valid = face.valid
        data = self.data
        n = self.n

        inv_w = 1.0 / C.RENDER_W
        inv_h = 1.0 / C.RENDER_H
        track_uv = tracking * px * inv_w

        if align != ALIGN_LEFT:
            width = face.measure(text, px) * inv_w + track_uv * max(len(text) - 1, 0)
            x -= width if align == ALIGN_RIGHT else width * 0.5

        pen = x
        for ch in text:
            code = ord(ch)
            if code < FIRST_CHAR or code > LAST_CHAR:
                continue
            i = code - FIRST_CHAR
            row = glyphs[i]
            if valid[i]:
                if n >= C.MAX_TEXT_GLYPHS:
                    self.dropped += 1
                    break
                out = data[n]
                out[0] = pen + float(row[G_BX]) * px * inv_w
                out[1] = y + float(row[G_BY]) * px * inv_h
                out[2] = float(row[G_W]) * px * inv_w
                out[3] = float(row[G_H]) * px * inv_h
                out[4] = row[G_U]
                out[5] = row[G_V]
                out[6] = row[G_UW]
                out[7] = row[G_VH]
                out[8] = col[0]
                out[9] = col[1]
                out[10] = col[2]
                out[11] = a
                out[12] = px
                out[13] = weight
                n += 1
            pen += float(row[G_ADV]) * px * inv_w + track_uv
        self.n = n
        return pen

    def wrap(
        self,
        text: str,
        face: Face,
        x: float,
        y: float,
        px: float,
        max_w: float,
        col: tuple[float, float, float],
        a: float = 1.0,
        align: int = ALIGN_LEFT,
        leading: float = 1.22,
    ) -> float:
        """Word-wrap into `max_w` (uv). Returns the y below the last line.

        Only ever called on state changes, not per frame -- the word and its
        example sentence are laid out when the card appears and then just
        re-emitted from the same string each frame.
        """
        inv_w = 1.0 / C.RENDER_W
        line_h = face.line * px * leading / C.RENDER_H
        start = 0
        last_space = -1
        cursor = y
        i = 0
        n = len(text)
        while i <= n:
            if i == n or text[i] == " ":
                if face.measure(text[start:i], px) * inv_w > max_w and last_space > start:
                    self.add(text[start:last_space], face, x, cursor, px, col, a, align)
                    cursor += line_h
                    start = last_space + 1
                last_space = i
            i += 1
        if start < n:
            self.add(text[start:], face, x, cursor, px, col, a, align)
            cursor += line_h
        return cursor

    def flush(self) -> int:
        if self.n == 0:
            return 0
        self.buf.write(self.data)
        self.tex.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.n)
        return 1

    def warm(self, target: moderngl.Framebuffer) -> None:
        target.use()
        self.ctx.viewport = (0, 0, 8, 8)
        self.begin()
        self.add("WARM", self.display, 0.1, 0.1, 40.0, (1.0, 1.0, 1.0))
        self.add("warm", self.body, 0.1, 0.3, 20.0, (1.0, 1.0, 1.0))
        self.add("0.0", self.mono, 0.1, 0.5, 16.0, (1.0, 1.0, 1.0))
        self.flush()
        self.begin()

    def release(self) -> None:
        self.vao.release()
        self.buf.release()
        self.quad.release()
        self.tex.release()
        self.prog.release()


class ShapeRenderer:
    """Rounded rects, rings and bars. One instanced call, plus one for arcs."""

    MAX = 320

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=S.SHAPE_VS, fragment_shader=S.SHAPE_FS)
        self.arc_prog = ctx.program(vertex_shader=S.SHAPE_VS, fragment_shader=S.ARC_FS)
        aspect = C.RENDER_H / C.RENDER_W
        set_uniform(self.prog, "u_aspect", aspect)
        set_uniform(self.arc_prog, "u_aspect", aspect)

        quad = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=F32)
        self.quad = ctx.buffer(quad.tobytes())

        self.data = np.zeros((self.MAX, SHAPE_FLOATS), dtype=F32)
        self.buf = ctx.buffer(reserve=self.data.nbytes, dynamic=True)
        self.arc_data = np.zeros((8, SHAPE_FLOATS), dtype=F32)
        self.arc_buf = ctx.buffer(reserve=self.arc_data.nbytes, dynamic=True)

        layout = "4f 4f 4f/i"
        names = ("i_rect", "i_color", "i_params")
        self.vao = ctx.vertex_array(
            self.prog, [(self.quad, "2f", "quad"), (self.buf, layout, *names)]
        )
        self.arc_vao = ctx.vertex_array(
            self.arc_prog, [(self.quad, "2f", "quad"), (self.arc_buf, layout, *names)]
        )
        self.n = 0
        self.arc_n = 0

    def begin(self) -> None:
        self.n = 0
        self.arc_n = 0

    def rect(
        self,
        cx: float,
        cy: float,
        hw: float,
        hh: float,
        col: tuple[float, float, float],
        a: float = 1.0,
        radius: float = 0.0,
        border: float = 0.0,
    ) -> None:
        if self.n >= self.MAX:
            return
        r = self.data[self.n]
        r[0] = cx
        r[1] = cy
        r[2] = hw
        r[3] = hh
        r[4] = col[0]
        r[5] = col[1]
        r[6] = col[2]
        r[7] = a
        r[8] = radius
        r[9] = 1.0 if border > 0.0 else 0.0
        r[10] = border
        r[11] = 0.0
        self.n += 1

    def arc(
        self,
        cx: float,
        cy: float,
        radius: float,
        thickness: float,
        progress: float,
        col: tuple[float, float, float],
        a: float = 1.0,
    ) -> None:
        if self.arc_n >= 8:
            return
        r = self.arc_data[self.arc_n]
        r[0] = cx
        r[1] = cy
        r[2] = radius
        # `radius` is in units of frame width; the uv-y half-extent that covers
        # the same number of pixels is therefore smaller, not larger.
        r[3] = radius * (C.RENDER_W / C.RENDER_H)
        r[4] = col[0]
        r[5] = col[1]
        r[6] = col[2]
        r[7] = a
        r[8] = thickness
        r[9] = progress
        r[10] = 0.0
        r[11] = 0.0
        self.arc_n += 1

    def flush(self) -> int:
        calls = 0
        if self.n:
            self.buf.write(self.data)
            self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.n)
            calls += 1
        if self.arc_n:
            self.arc_buf.write(self.arc_data)
            self.arc_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.arc_n)
            calls += 1
        return calls

    def warm(self, target: moderngl.Framebuffer) -> None:
        target.use()
        self.ctx.viewport = (0, 0, 8, 8)
        self.begin()
        self.rect(0.5, 0.5, 0.2, 0.1, (1.0, 1.0, 1.0), 1.0, 0.05)
        self.rect(0.5, 0.5, 0.2, 0.1, (1.0, 1.0, 1.0), 1.0, 0.05, 0.01)
        self.arc(0.5, 0.5, 0.2, 0.01, 0.5, (1.0, 1.0, 1.0))
        self.flush()
        self.begin()

    def release(self) -> None:
        self.vao.release()
        self.arc_vao.release()
        self.buf.release()
        self.arc_buf.release()
        self.quad.release()
        self.prog.release()
        self.arc_prog.release()


# --------------------------------------------------------------------------- formatting


class NumBuf:
    """Formats numbers into a reused character buffer.

    f-strings and % formatting both allocate a new str every call, and the HUD
    formats a dozen numbers per frame at 60fps. This writes ASCII into a
    bytearray and hands back a memoryview-free str only when the value actually
    changed, so a steady readout costs nothing.
    """

    __slots__ = ("_cache", "_digits", "_last")

    def __init__(self, slots: int = 24) -> None:
        self._digits = bytearray(24)
        self._cache: list[str] = [""] * slots
        self._last: list[int] = [-(2**60)] * slots

    def fixed(self, slot: int, value: float, decimals: int = 1) -> str:
        """Format with fixed decimals, reusing the previous string if unchanged."""
        scale = 10**decimals
        key = int(value * scale + (0.5 if value >= 0.0 else -0.5))
        if key == self._last[slot]:
            return self._cache[slot]
        self._last[slot] = key
        buf = self._digits
        neg = key < 0
        if neg:
            key = -key
        i = len(buf)
        for _ in range(decimals):
            i -= 1
            buf[i] = 48 + key % 10
            key //= 10
        if decimals:
            i -= 1
            buf[i] = 46  # '.'
        if key == 0:
            i -= 1
            buf[i] = 48
        while key:
            i -= 1
            buf[i] = 48 + key % 10
            key //= 10
        if neg:
            i -= 1
            buf[i] = 45
        out = buf[i:].decode("ascii")
        self._cache[slot] = out
        return out

    def integer(self, slot: int, value: int) -> str:
        return self.fixed(slot, float(value), 0)
