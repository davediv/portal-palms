"""Render every metaphor as a filmstrip, with the labels hidden.

    uv run python tools/contact_sheet.py --out /tmp/sheet.png

This is the "is it guessable?" check from the spec, run the only way it can
honestly be run: look at the animation with no word attached and see whether
the idea arrives. Pass --labels once you have guessed, to score yourself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portal_palms import config as C
from portal_palms.content.metaphors import SCENES
from portal_palms.render.gl import Fullscreen, Targets, Window
from portal_palms.render.scene import SceneRenderer

FRAMES = 5
TILE_W = 300


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/sheet.png")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--only", default="", help="comma separated scene names")
    args = ap.parse_args()

    win = Window(320, 240, visible=False)
    ctx = win.ctx
    fs = Fullscreen(ctx)
    targets = Targets(ctx)
    scene = SceneRenderer(ctx, targets, fs)
    buf = np.zeros((C.SCENE_H, C.SCENE_W, 3), dtype=np.uint8)

    names = [n for n in SCENES if not args.only or n in args.only.split(",")]
    tile_h = int(TILE_W * C.SCENE_ASPECT)
    rows = []
    for name in names:
        sc = SCENES[name]
        strip = []
        for i in range(FRAMES):
            t = sc.loop * (i / FRAMES)
            scene.painter.reset()
            sc.paint(scene.painter, t)
            if scene.painter.overflow:
                print(f"  !! {name}: {scene.painter.overflow} instances dropped", file=sys.stderr)
            scene.render(sc.tint, t, 0.0, 0.0)
            targets.scene.read_into(buf, components=3)
            img = cv2.cvtColor(np.flipud(buf), cv2.COLOR_RGB2BGR)
            strip.append(cv2.resize(img, (TILE_W, tile_h), interpolation=cv2.INTER_AREA))
        row = np.concatenate(strip, axis=1)
        cv2.line(row, (0, tile_h - 1), (row.shape[1], tile_h - 1), (40, 46, 58), 1)
        if args.labels:
            cv2.putText(row, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 220, 150), 1)
        rows.append(row)
        print(f"{name:15s} loop {sc.loop:.1f}s")

    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(names)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
