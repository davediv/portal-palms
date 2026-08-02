"""Render frames offscreen to PNG, with no camera and no window.

    uv run python tools/preview.py --out /tmp/shot.png --open 1.0 --time 2.0
    uv run python tools/preview.py --contact-sheet /tmp/sheet.png

This is how the rim shader and every metaphor scene get looked at during
development: hidden GLFW window, the app's real render path, one readback.
It is also what the "is every metaphor guessable with the label hidden?"
check runs on.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portal_palms import config as C
from portal_palms.render.gl import Fullscreen, Presenter, Targets, Window
from portal_palms.render.portal import ParticleSystem, PortalRenderer
from portal_palms.render.scene import SceneRenderer
from portal_palms.render.text import ShapeRenderer, TextRenderer


def _placeholder(p: object, t: float) -> None:
    """A stand-in interior so the portal composite can be judged before the
    metaphor scenes exist."""

    p.floor(0.0)  # type: ignore[attr-defined]
    for i in range(6):
        a = t * 0.5 + i * 1.047
        p.box(  # type: ignore[attr-defined]
            math.cos(a) * 2.0, 0.6 + 0.4 * math.sin(t + i), math.sin(a) * 2.0,
            0.5, 0.5, 0.5, (0.55, 0.62, 0.72), a, 1, 1.0, 0.0, 0.4,
        )  # fmt: skip
    p.ball(0.0, 1.4, 0.0, 0.45, (1.0, 0.773, 0.192), 1.0, 0.9)  # type: ignore[attr-defined]
    p.figure(1.6, 0.0, 1.0, 0.9, (0.85, 0.88, 0.92), stride=math.sin(t * 2.0))  # type: ignore[attr-defined]


class Preview:
    def __init__(self) -> None:
        self.win = Window(320, 240, visible=False)
        ctx = self.win.ctx
        self.ctx = ctx
        self.fs = Fullscreen(ctx)
        self.targets = Targets(ctx)
        self.presenter = Presenter(ctx, self.fs)
        self.scene = SceneRenderer(ctx, self.targets, self.fs)
        self.portal = PortalRenderer(ctx)
        self.particles = ParticleSystem(ctx)
        self.text = TextRenderer(ctx, C.resolve_asset_root())
        self.shapes = ShapeRenderer(ctx)
        self.out = np.zeros((C.RENDER_H, C.RENDER_W, 3), dtype=np.uint8)

    def frame(
        self,
        t: float,
        openness: float,
        radius: float,
        rotation: float = 0.0,
        charge: float = 0.0,
        paint: object = None,
        settle: int = 40,
    ) -> NDArray[np.uint8]:
        import moderngl

        ctx = self.ctx
        cx, cy = 0.5, 0.46

        # Let the particle sim reach a steady state so a still frame shows the
        # rim as it actually looks in motion, not one frame after a cold start.
        for i in range(settle):
            self.particles.update(
                1.0 / 60.0, t - (settle - i) / 60.0, cx, cy,
                max(radius, 0.075),
                max(radius, 0.075) * (0.035 + 0.665 * openness),
                rotation, openness, 0.0,
            )  # fmt: skip

        paint = paint or _placeholder
        self.scene.painter.reset()
        paint(self.scene.painter, t)  # type: ignore[operator]
        self.scene.render((0.247, 0.839, 0.549), t, 0.0, 0.0)

        self.targets.main.use()
        ctx.viewport = (0, 0, C.RENDER_W, C.RENDER_H)
        ctx.clear(0.043, 0.059, 0.102, 1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

        self.targets.scene_tex.use(0)
        self.portal.set_shape(cx, cy, radius, openness, rotation)
        self.portal.draw(t, openness, 1.0, charge)
        self.particles.draw(1.0)

        self.presenter.run_bloom(self.targets, 1.05, 0.55)
        self.presenter.present_fbo.read_into(self.out, components=3)
        return np.flipud(self.out).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/portal.png")
    ap.add_argument("--open", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=0.30)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--charge", type=float, default=0.0)
    ap.add_argument("--time", type=float, default=3.0)
    ap.add_argument("--strip", action="store_true", help="one image of the whole open sequence")
    ap.add_argument("--scene", action="store_true", help="dump the interior FBO on its own")
    args = ap.parse_args()

    p = Preview()
    if args.scene:
        p.frame(args.time, args.open, args.radius, args.rot, args.charge, None)
        buf = np.zeros((C.SCENE_H, C.SCENE_W, 3), dtype=np.uint8)
        p.targets.scene.read_into(buf, components=3)
        cv2.imwrite(args.out, cv2.cvtColor(np.flipud(buf), cv2.COLOR_RGB2BGR))
        print(f"wrote {args.out}")
        return 0
    if args.strip:
        shots = []
        for i in range(5):
            o = i / 4.0
            shots.append(p.frame(args.time + i * 0.7, o, 0.09 + 0.22 * o, args.rot, 0.0, None))
        sheet = np.concatenate([cv2.resize(s, (360, 640)) for s in shots], axis=1)
        cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    else:
        img = p.frame(args.time, args.open, args.radius, args.rot, args.charge, None)
        cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
