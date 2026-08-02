"""The interior scene: what you actually see through the portal.

Renders into a half-resolution framebuffer, and only while the portal is open.
When the portal is shut this pass is skipped entirely, which is the difference
between the app costing one scene per frame and costing two.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np
from numpy.typing import NDArray

from portal_palms import config as C
from portal_palms.render import shaders as S
from portal_palms.render.gl import Fullscreen, Targets, set_uniform, uniform
from portal_palms.render.primitives import Painter, PrimitivePool

F32 = np.float32
Mat = NDArray[np.float32]


def perspective(fovy: float, aspect: float, near: float, far: float, out: Mat) -> None:
    f = 1.0 / math.tan(fovy * 0.5)
    out[:] = 0.0
    out[0, 0] = f / aspect
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = -1.0
    out[3, 2] = (2.0 * far * near) / (near - far)


def look_at(
    ex: float, ey: float, ez: float,
    cx: float, cy: float, cz: float,
    out: Mat,
) -> None:  # fmt: skip
    fx, fy, fz = cx - ex, cy - ey, cz - ez
    fl = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    # right = normalize(cross(forward, world_up)), world_up = +Y
    rx, ry, rz = -fz, 0.0, fx
    rl = math.sqrt(rx * rx + rz * rz) or 1.0
    rx, rz = rx / rl, rz / rl
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx

    out[:] = 0.0
    out[0, 0], out[1, 0], out[2, 0] = rx, ry, rz
    out[0, 1], out[1, 1], out[2, 1] = ux, uy, uz
    out[0, 2], out[1, 2], out[2, 2] = -fx, -fy, -fz
    out[3, 0] = -(rx * ex + ry * ey + rz * ez)
    out[3, 1] = -(ux * ex + uy * ey + uz * ez)
    out[3, 2] = fx * ex + fy * ey + fz * ez
    out[3, 3] = 1.0


class SceneRenderer:
    """Owns the interior framebuffer, the primitive pool and the camera."""

    # A landscape 3:2 target at this distance shows roughly 6.1 x 4.1 world
    # units, which is the box every metaphor is composed inside.
    EYE_X = 0.0
    EYE_Y = 1.95
    EYE_Z = 7.6
    LOOK_Y = 1.15
    FOV_Y = 30.0

    def __init__(self, ctx: moderngl.Context, targets: Targets, fs: Fullscreen) -> None:
        self.ctx = ctx
        self.targets = targets
        self.fs = fs

        self.prog = ctx.program(vertex_shader=S.SCENE_VS, fragment_shader=S.SCENE_FS)
        self.bg = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.SCENE_BG_FS)
        self.pool = PrimitivePool(ctx, self.prog)
        self.painter = Painter(self.pool)

        self._proj: Mat = np.zeros((4, 4), dtype=F32)
        self._view: Mat = np.zeros((4, 4), dtype=F32)
        self._vp: Mat = np.zeros((4, 4), dtype=F32)
        perspective(math.radians(self.FOV_Y), C.SCENE_W / C.SCENE_H, 0.1, 90.0, self._proj)

        # Bright enough that the metaphor stays legible after the rim shader
        # darkens the interior toward the tear and the tonemap rolls the top off.
        set_uniform(self.prog, "u_key", (-0.45, 0.78, 0.44))
        set_uniform(self.prog, "u_keycol", (0.82, 0.90, 0.99))
        set_uniform(self.prog, "u_fillcol", (0.13, 0.19, 0.31))
        self.draw_calls = 0

    def warm(self) -> None:
        """Force both programs through a real draw so nothing compiles mid-session."""
        self.targets.scene.use()
        self.ctx.viewport = (0, 0, 4, 4)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.fs.draw(self.bg)
        self.painter.reset()
        self.painter.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.ball(0.0, 0.0, 0.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.column(0.0, 0.0, 0.0, 1.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.cone(0.0, 0.0, 0.0, 1.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.ring(0.0, 0.0, 0.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.shard(0.0, 0.0, 0.0, 1.0, (1.0, 1.0, 1.0))
        self.painter.floor()
        self._flush(0.0, 0.0, 0.0)
        self.painter.reset()

    def _flush(self, t: float, par_x: float, par_y: float) -> None:
        # Hand parallax moves the camera, not the image. That gives real
        # parallax between near and far objects in the scene, which is what
        # makes the portal read as a hole with depth behind it.
        ex = self.EYE_X + par_x
        ey = self.EYE_Y + par_y
        look_at(ex, ey, self.EYE_Z, 0.0, self.LOOK_Y, 0.0, self._view)
        np.matmul(self._view, self._proj, out=self._vp)
        uniform(self.prog, "u_viewproj").write(self._vp)
        set_uniform(self.prog, "u_eye", (ex, ey, self.EYE_Z))
        self.draw_calls += self.pool.draw(self.painter.counts)

    def render(self, tint: tuple[float, float, float], t: float, par_x: float, par_y: float) -> int:
        """Draw whatever the painter currently holds. Returns draw calls used."""
        ctx = self.ctx
        self.draw_calls = 0
        self.targets.scene.use()
        ctx.viewport = (0, 0, C.SCENE_W, C.SCENE_H)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        set_uniform(self.bg, "u_time", t)
        set_uniform(self.bg, "u_tint", tint)
        self.fs.draw(self.bg)
        self.draw_calls += 1

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self._flush(t, par_x, par_y)

        ctx.disable(moderngl.DEPTH_TEST)
        return self.draw_calls

    def release(self) -> None:
        self.pool.release()
        self.prog.release()
        self.bg.release()
