"""Window, context and render targets.

The whole app composes into one 1080x1920 offscreen framebuffer which is then
blitted letterboxed into the window. Recording is therefore free: it reads the
framebuffer we already drew rather than running a second render pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import glfw
import moderngl
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.render import shaders as S

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

F32 = np.float32

# RGBA16F everywhere the image is still being composited. Bloom needs values
# above 1.0 to have anywhere to come from; an 8-bit target clips the rim to flat
# white and the glow stops reading as light.
HDR: Final = "f2"


def uniform(prog: moderngl.Program, name: str) -> moderngl.Uniform:
    """Fetch a uniform, narrowed to the type it actually is.

    `Program.__getitem__` can hand back a uniform, a uniform block, an attribute
    or a varying, so it is typed as the union of all four. Every lookup in this
    app is a uniform; going through here says so once, and turns a typo into an
    exception at init rather than a silent no-op at frame 1.
    """
    member = prog[name]
    if not isinstance(member, moderngl.Uniform):
        raise TypeError(f"program member {name!r} is a {type(member).__name__}, not a uniform")
    return member


def set_uniform(prog: moderngl.Program, name: str, value: object) -> None:
    """Init-time convenience. Hot paths bind the Uniform once and assign to it."""
    uniform(prog, name).value = value


class Window:
    """GLFW window plus a ModernGL 3.3-core context.

    OpenGL 4.1 is the macOS ceiling and 3.3 core is the common floor, so the
    context is requested as 3.3 core forward-compatible and nothing in the app
    uses anything newer.
    """

    def __init__(
        self, width: int = C.WINDOW_W, height: int = C.WINDOW_H, visible: bool = True
    ) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, visible)
        glfw.window_hint(glfw.SAMPLES, 0)  # we render offscreen; MSAA here would be waste
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.handle = glfw.create_window(width, height, C.WINDOW_TITLE, None, None)
        if not self.handle:
            glfw.terminate()
            raise RuntimeError("could not create an OpenGL 3.3 core window")

        glfw.make_context_current(self.handle)
        glfw.swap_interval(1)
        self.ctx = moderngl.create_context()
        self.vsync_interval = 1

        self.on_key: Callable[[int, int], None] | None = None
        self.on_click: Callable[[float, float], None] | None = None
        self.on_scroll: Callable[[float], None] | None = None
        self.fullscreen = False
        self._windowed = (0, 0, width, height)
        glfw.set_key_callback(self.handle, self._key_cb)
        glfw.set_mouse_button_callback(self.handle, self._mouse_cb)
        glfw.set_scroll_callback(self.handle, self._scroll_cb)

    # ------------------------------------------------------------------ callbacks

    def _key_cb(self, _win: Any, key: int, _sc: int, action: int, mods: int) -> None:
        if action in (glfw.PRESS, glfw.REPEAT) and self.on_key is not None:
            self.on_key(key, mods)

    def _mouse_cb(self, _win: Any, button: int, action: int, _mods: int) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and self.on_click:
            x, y = glfw.get_cursor_pos(self.handle)
            fw, fh = glfw.get_framebuffer_size(self.handle)
            ww, wh = glfw.get_window_size(self.handle)
            sx = fw / max(ww, 1)
            sy = fh / max(wh, 1)
            self.on_click(x * sx / max(fw, 1), y * sy / max(fh, 1))

    def _scroll_cb(self, _win: Any, _dx: float, dy: float) -> None:
        if self.on_scroll is not None:
            self.on_scroll(dy)

    # ------------------------------------------------------------------ frame

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def set_vsync(self, interval: int) -> None:
        """1 = every refresh, 2 = every other (the 30fps cap in low-power mode)."""
        if interval != self.vsync_interval:
            self.vsync_interval = interval
            glfw.swap_interval(interval)

    def toggle_fullscreen(self) -> None:
        """Borderless fullscreen on the monitor the window is currently on."""
        if self.fullscreen:
            x, y, w, h = self._windowed
            # A null monitor is how GLFW spells "go back to windowed"; the
            # binding stubs type the parameter as non-optional.
            glfw.set_window_monitor(self.handle, None, x, y, w, h, 0)  # type: ignore[arg-type]
            self.fullscreen = False
        else:
            x, y = glfw.get_window_pos(self.handle)
            w, h = glfw.get_window_size(self.handle)
            self._windowed = (x, y, w, h)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.handle, monitor, 0, 0, mode.size.width, mode.size.height,
                mode.refresh_rate,
            )  # fmt: skip
            self.fullscreen = True
        # The monitor swap resets the swap interval on some drivers.
        glfw.swap_interval(self.vsync_interval)

    def framebuffer_size(self) -> tuple[int, int]:
        w, h = glfw.get_framebuffer_size(self.handle)
        return max(int(w), 1), max(int(h), 1)

    def poll(self) -> None:
        glfw.poll_events()

    def swap(self) -> None:
        glfw.swap_buffers(self.handle)

    def close(self) -> None:
        glfw.set_window_should_close(self.handle, True)

    def terminate(self) -> None:
        glfw.terminate()


class Targets:
    """Every framebuffer the app owns, allocated exactly once at init.

    Nothing here is recreated on a resize: the composition resolution is fixed
    at 1080x1920 and only the final letterbox blit knows about the window size.
    That is also why opening and closing the portal repeatedly cannot leak — no
    framebuffer is ever created after startup.
    """

    def __init__(self, ctx: moderngl.Context, low_power: bool = False) -> None:
        self.ctx = ctx

        self.main_tex = ctx.texture((C.RENDER_W, C.RENDER_H), 4, dtype=HDR)
        self.main_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.main_tex.repeat_x = False
        self.main_tex.repeat_y = False
        self.main = ctx.framebuffer([self.main_tex])

        # Interior scene at half resolution: it is viewed through a small
        # ellipse and the difference is invisible, but it is the frame's biggest
        # single cost so halving it is the cheapest win available.
        self.scene_tex = ctx.texture((C.SCENE_W, C.SCENE_H), 4, dtype=HDR)
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_tex.repeat_x = False
        self.scene_tex.repeat_y = False
        self.scene_depth = ctx.depth_renderbuffer((C.SCENE_W, C.SCENE_H))
        self.scene = ctx.framebuffer([self.scene_tex], self.scene_depth)

        # Both bloom resolutions exist from the start so toggling low-power mode
        # mid-session never allocates.
        self.bloom_hi = _BloomChain(ctx, C.BLOOM_DIV)
        self.bloom_lo = _BloomChain(ctx, C.BLOOM_DIV_LOW)
        self.low_power = low_power

    @property
    def bloom(self) -> _BloomChain:
        return self.bloom_lo if self.low_power else self.bloom_hi

    def release(self) -> None:
        for obj in (
            self.main,
            self.main_tex,
            self.scene,
            self.scene_tex,
            self.scene_depth,
        ):
            obj.release()
        self.bloom_hi.release()
        self.bloom_lo.release()


class _BloomChain:
    """A single downsampled ping-pong pair. One blur pass, deliberately."""

    def __init__(self, ctx: moderngl.Context, div: int) -> None:
        self.w = max(C.RENDER_W // div, 4)
        self.h = max(C.RENDER_H // div, 4)
        self.tex_a = ctx.texture((self.w, self.h), 4, dtype=HDR)
        self.tex_b = ctx.texture((self.w, self.h), 4, dtype=HDR)
        for t in (self.tex_a, self.tex_b):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False
        self.fbo_a = ctx.framebuffer([self.tex_a])
        self.fbo_b = ctx.framebuffer([self.tex_b])

    def release(self) -> None:
        for obj in (self.fbo_a, self.fbo_b, self.tex_a, self.tex_b):
            obj.release()


class Fullscreen:
    """Shared machinery for the passes that draw one triangle over a target."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._empty = ctx.buffer(reserve=4)
        self._vaos: dict[int, moderngl.VertexArray] = {}

    def vao(self, prog: moderngl.Program) -> moderngl.VertexArray:
        key = id(prog)
        v = self._vaos.get(key)
        if v is None:
            # No attributes: the vertex shader synthesises positions from
            # gl_VertexID, so there is no vertex buffer to bind or upload.
            v = self.ctx.vertex_array(prog, [])
            v.vertices = 3
            self._vaos[key] = v
        return v

    def draw(self, prog: moderngl.Program) -> None:
        self.vao(prog).render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        for v in self._vaos.values():
            v.release()
        self._vaos.clear()
        self._empty.release()


class Presenter:
    """Bloom composite plus the letterboxed blit into the window."""

    def __init__(self, ctx: moderngl.Context, fs: Fullscreen) -> None:
        self.ctx = ctx
        self.fs = fs
        self.bright = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.BLOOM_BRIGHT_FS)
        self.blur = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.BLOOM_BLUR_FS)
        self.composite = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.COMPOSITE_FS)
        self.blit = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.BLIT_FS)

        set_uniform(self.bright, "u_src", 0)
        set_uniform(self.blur, "u_src", 0)
        set_uniform(self.composite, "u_scene", 0)
        set_uniform(self.composite, "u_bloom", 1)
        set_uniform(self.blit, "u_src", 0)

        # Composited LDR frame. Recording and the window blit both read this one.
        self.present_tex = ctx.texture((C.RENDER_W, C.RENDER_H), 4)
        self.present_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.present_tex.repeat_x = False
        self.present_tex.repeat_y = False
        self.present_fbo = ctx.framebuffer([self.present_tex])

        self._rect = np.zeros(4, dtype=F32)

    def run_bloom(self, targets: Targets, threshold: float, amount: float) -> int:
        """Bright pass, one separable blur, composite. Returns draw calls used."""
        ctx = self.ctx
        chain = targets.bloom
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)

        chain.fbo_a.use()
        ctx.viewport = (0, 0, chain.w, chain.h)
        targets.main_tex.use(0)
        set_uniform(self.bright, "u_threshold", threshold)
        self.fs.draw(self.bright)

        chain.fbo_b.use()
        chain.tex_a.use(0)
        set_uniform(self.blur, "u_dir", (1.0 / chain.w, 0.0))
        self.fs.draw(self.blur)

        chain.fbo_a.use()
        chain.tex_b.use(0)
        set_uniform(self.blur, "u_dir", (0.0, 1.0 / chain.h))
        self.fs.draw(self.blur)

        self.present_fbo.use()
        ctx.viewport = (0, 0, C.RENDER_W, C.RENDER_H)
        targets.main_tex.use(0)
        chain.tex_a.use(1)
        set_uniform(self.composite, "u_bloom_amount", amount)
        self.fs.draw(self.composite)
        return 4

    def to_window(self, win_w: int, win_h: int) -> None:
        """Blit the composed vertical frame into whatever shape the window is."""
        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, win_w, win_h)
        ctx.disable(moderngl.BLEND)

        target = C.RENDER_ASPECT
        actual = win_w / win_h
        if actual > target:  # window wider than 9:16 -> pillarbox
            sx = target / actual
            sy = 1.0
        else:
            sx = 1.0
            sy = actual / target
        set_uniform(self.blit, "u_rect", (0.5 - sx * 0.5, 0.5 - sy * 0.5, sx, sy))
        self.present_tex.use(0)
        self.fs.draw(self.blit)

    def release(self) -> None:
        self.present_fbo.release()
        self.present_tex.release()
        for p in (self.bright, self.blur, self.composite, self.blit):
            p.release()


def read_frame(fbo: moderngl.Framebuffer, out: NDArray[np.uint8]) -> None:
    """Synchronous readback. Debug and screenshots only -- never in the frame loop."""
    fbo.read_into(out, components=3)
