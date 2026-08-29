"""The portal itself: torn rim, masked interior, and the sparks it sheds.

The rim is the entire visual identity of the app, so it gets the whole budget
that a post stack would otherwise eat. A tiny 512x1 pass evaluates its
angle-only noise once, then one draw over a tightly bounded quad produces the
tear, filament, outward sparks, interior mask and glow. Premultiplied output is
what lets that main pass be both opaque (interior) and additive (glow) at the
same time.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.render import shaders as S
from portal_palms.render.gl import set_uniform, uniform

F32 = np.float32

# How far past the ellipse boundary the shader still writes anything.
RIM_MARGIN = 1.85
# Minor/major axis ratio: a closed portal is a slit between the palms, an open
# one is a full ellipse. This single number is what makes the gesture read as
# *tearing* rather than as scaling a circle.
MINOR_CLOSED = 0.035
MINOR_OPEN = 0.70
# Even with the wrists touching there should be a visible seam of light.
MIN_MAJOR = 0.075
# More than seven samples across the highest-frequency angular signal. The
# strip is tiny (8 KiB across both half-float targets) but removes millions of
# duplicate simplex-noise evaluations from a large open portal.
RIM_SIGNAL_SAMPLES = 512


class PortalRenderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.signal_prog = ctx.program(
            vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.RIM_SIGNAL_FS
        )
        self.signal_tex = [
            ctx.texture((RIM_SIGNAL_SAMPLES, 1), 4, dtype="f2"),
            ctx.texture((RIM_SIGNAL_SAMPLES, 1), 4, dtype="f2"),
        ]
        for tex in self.signal_tex:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = False
        self.signal_fbo = ctx.framebuffer(self.signal_tex)
        self.signal_vao = ctx.vertex_array(self.signal_prog, [])
        self.signal_vao.vertices = 3
        self.u_signal_time = uniform(self.signal_prog, "u_time")

        self.prog = ctx.program(vertex_shader=S.PORTAL_VS, fragment_shader=S.PORTAL_FS)
        set_uniform(self.prog, "u_scene", 0)
        set_uniform(self.prog, "u_signal0", 1)
        set_uniform(self.prog, "u_signal1", 2)
        set_uniform(self.prog, "u_aspect", C.RENDER_H / C.RENDER_W)
        self.vao = ctx.vertex_array(self.prog, [])
        self.vao.vertices = 4

        # Uniform handles are bound once. A missing one raises here, at init,
        # instead of on the first frame that happens to exercise it.
        self.u_center = uniform(self.prog, "u_center")
        self.u_extent = uniform(self.prog, "u_extent")
        self.u_radius = uniform(self.prog, "u_radius")
        self.u_rot = uniform(self.prog, "u_rot")
        self.u_time = uniform(self.prog, "u_time")
        self.u_open = uniform(self.prog, "u_open")
        self.u_intensity = uniform(self.prog, "u_intensity")
        self.u_charge = uniform(self.prog, "u_charge")
        self.u_flash = uniform(self.prog, "u_flash")
        self.u_parallax = uniform(self.prog, "u_parallax")
        self.u_scale01 = uniform(self.prog, "u_scale01")
        set_uniform(self.prog, "u_pxscale", float(C.RENDER_W))
        set_uniform(self.prog, "u_scene_aspect", C.SCENE_ASPECT)
        set_uniform(self.prog, "u_zoom", C.SCENE_ZOOM)

        self.major = 0.0
        self.minor = 0.0
        self.flash = 0.0
        self._par_x = 0.0
        self._par_y = 0.0

    def warm(self, target: moderngl.Framebuffer, scene_tex: moderngl.Texture) -> None:
        self.update_signals(0.0)
        target.use()
        self.ctx.viewport = (0, 0, 4, 4)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        scene_tex.use(0)
        self.set_shape(0.5, 0.5, 0.2, 0.5, 0.0)
        self.draw(0.0, 0.5, 1.0, 0.0)

    def update_signals(self, t: float) -> int:
        """Evaluate the rim's angle-only noise once for this frame."""
        self.signal_fbo.use()
        self.ctx.viewport = (0, 0, RIM_SIGNAL_SAMPLES, 1)
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        self.u_signal_time.value = t
        self.signal_vao.render(moderngl.TRIANGLES, vertices=3)
        return 1

    # ------------------------------------------------------------------ shape

    def set_shape(
        self, cx: float, cy: float, radius: float, openness: float, rotation: float
    ) -> None:
        """Compute the ellipse and the quad that tightly bounds its glow."""
        self.major = max(radius, MIN_MAJOR)
        self.minor = self.major * (MINOR_CLOSED + (MINOR_OPEN - MINOR_CLOSED) * openness)

        aspect = C.RENDER_H / C.RENDER_W
        a = self.major * RIM_MARGIN
        b = self.minor * RIM_MARGIN
        cs = math.cos(rotation)
        sn = math.sin(rotation)
        # Axis-aligned bounds of the rotated ellipse, so the quad stays tight at
        # every roll angle instead of being sized for the worst case.
        hx = math.sqrt(a * a * cs * cs + b * b * sn * sn)
        hy = math.sqrt(a * a * sn * sn + b * b * cs * cs) / aspect

        self.u_center.value = (cx, cy)
        self.u_extent.value = (hx, hy)
        self.u_radius.value = (self.major, self.minor)
        self.u_rot.value = rotation
        # Filament and spark thicknesses are in pixels, but a large portal
        # should still have a proportionally heavier edge than a small one.
        self.u_scale01.value = self.major / 0.30

    def set_parallax(self, vx: float, vy: float, cx: float, cy: float) -> None:
        """Shift the interior against hand movement, with a little lag.

        Without this the scene sits still while the frame around it moves and
        the whole thing reads as a sticker. The lag is deliberate: an instant
        offset looks like a texture slide, a lagging one looks like depth.
        """
        tx = (cx - 0.5) * -0.075 + vx * -0.012
        ty = (cy - 0.5) * -0.075 + vy * -0.012
        self._par_x += (tx - self._par_x) * 0.14
        self._par_y += (ty - self._par_y) * 0.14
        if self._par_x > 0.12:
            self._par_x = 0.12
        elif self._par_x < -0.12:
            self._par_x = -0.12
        if self._par_y > 0.12:
            self._par_y = 0.12
        elif self._par_y < -0.12:
            self._par_y = -0.12

    @property
    def parallax(self) -> tuple[float, float]:
        return self._par_x, self._par_y

    def trigger_flash(self) -> None:
        self.flash = 1.0

    def tick(self, dt: float) -> None:
        if self.flash > 0.0:
            self.flash -= dt * 2.4
            if self.flash < 0.0:
                self.flash = 0.0

    # ------------------------------------------------------------------ draw

    def draw(self, t: float, openness: float, intensity: float, charge: float) -> None:
        self.u_time.value = t
        self.u_open.value = openness
        self.u_intensity.value = intensity
        self.u_charge.value = charge
        self.u_flash.value = self.flash
        self.u_parallax.value = (self._par_x, self._par_y)
        self.signal_tex[0].use(1)
        self.signal_tex[1].use(2)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    def release(self) -> None:
        self.vao.release()
        self.prog.release()
        self.signal_vao.release()
        self.signal_fbo.release()
        for tex in self.signal_tex:
            tex.release()
        self.signal_prog.release()


class ParticleSystem:
    """GPU particles by transform feedback between paired VBOs.

    Transform feedback rather than ping-pong data textures because the ceiling
    here is OpenGL 4.1: no compute shaders, no SSBOs. Nothing about a particle
    is ever read back to the CPU, and the whole system is one instanced draw.
    """

    FLOATS = 6  # pos.xy, vel.xy, life, seed

    def __init__(self, ctx: moderngl.Context, count: int = C.PARTICLE_COUNT) -> None:
        self.ctx = ctx
        self.capacity = count
        self.count = count

        rng = np.random.default_rng(0xB0A7)
        state = np.zeros((count, self.FLOATS), dtype=F32)
        state[:, 5] = rng.random(count).astype(F32)  # per-particle seed, fixed for life
        blob = state.tobytes()

        self.update_prog = ctx.program(
            vertex_shader=S.PARTICLE_UPDATE_VS,
            varyings=("out_pos", "out_vel", "out_life", "out_seed"),
        )
        self.draw_prog = ctx.program(
            vertex_shader=S.PARTICLE_DRAW_VS, fragment_shader=S.PARTICLE_DRAW_FS
        )

        self.buf = [ctx.buffer(blob), ctx.buffer(blob)]
        quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=F32)
        self.quad = ctx.buffer(quad.tobytes())

        fmt = "2f 2f 1f 1f"
        names = ("in_pos", "in_vel", "in_life", "in_seed")
        self.update_vao = [
            ctx.vertex_array(self.update_prog, [(self.buf[i], fmt, *names)]) for i in range(2)
        ]
        self.draw_vao = [
            ctx.vertex_array(
                self.draw_prog,
                [(self.quad, "2f", "quad"), (self.buf[i], fmt + "/i", *names)],
            )
            for i in range(2)
        ]
        self.src = 0

        set_uniform(self.draw_prog, "u_aspect", C.RENDER_H / C.RENDER_W)
        set_uniform(self.draw_prog, "u_size", 0.0055)
        set_uniform(self.update_prog, "u_aspect", C.RENDER_H / C.RENDER_W)

        u = self.update_prog
        self.u_dt = uniform(u, "u_dt")
        self.u_time = uniform(u, "u_time")
        self.u_center = uniform(u, "u_center")
        self.u_radius = uniform(u, "u_radius")
        self.u_rot = uniform(u, "u_rot")
        self.u_emit = uniform(u, "u_emit")
        self.u_implode = uniform(u, "u_implode")
        self.u_intensity = uniform(self.draw_prog, "u_intensity")

    def set_low_power(self, low: bool) -> None:
        """Halve the live count. The buffers stay allocated, so this never stalls."""
        self.count = C.PARTICLE_COUNT_LOW if low else self.capacity
        if self.count > self.capacity:
            self.count = self.capacity

    def update(
        self,
        dt: float,
        t: float,
        cx: float,
        cy: float,
        major: float,
        minor: float,
        rot: float,
        emit: float,
        implode: float,
    ) -> None:
        self.u_dt.value = min(dt, 0.05)
        self.u_time.value = t
        self.u_center.value = (cx, cy)
        self.u_radius.value = (major, minor)
        self.u_rot.value = rot
        self.u_emit.value = emit
        self.u_implode.value = implode
        dst = 1 - self.src
        self.update_vao[self.src].transform(
            self.buf[dst], mode=moderngl.POINTS, vertices=self.count
        )
        self.src = dst

    def draw(self, intensity: float) -> int:
        self.u_intensity.value = intensity
        self.draw_vao[self.src].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.count)
        return 1

    def warm(self, target: moderngl.Framebuffer) -> None:
        target.use()
        self.ctx.viewport = (0, 0, 4, 4)
        self.update(0.016, 0.0, 0.5, 0.5, 0.2, 0.1, 0.0, 1.0, 0.0)
        self.draw(1.0)

    def release(self) -> None:
        for v in self.update_vao + self.draw_vao:
            v.release()
        for b in self.buf:
            b.release()
        self.quad.release()
        self.update_prog.release()
        self.draw_prog.release()


def peak_particle_alloc(count: int) -> NDArray[np.float32]:
    """Exposed for tests: the exact per-particle state footprint."""
    return np.zeros((count, ParticleSystem.FLOATS), dtype=F32)
