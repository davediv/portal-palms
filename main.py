"""Portal Palms -- entry point.

    uv run python main.py
    uv run python main.py --words my_deck.json --low-power

Deliberately thin, and deliberately careful about import order: the window has
to be on screen before anything expensive loads, so nothing heavier than argparse
is imported at module scope.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="portal-palms", description=__doc__)
    p.add_argument("--words", default="", help="path to a word list JSON (see README)")
    p.add_argument("--width", type=int, default=600, help="window width")
    p.add_argument("--height", type=int, default=1000, help="window height")
    p.add_argument(
        "--landscape",
        action="store_true",
        help="compose 1920x1080 so a maximised window fills a desktop screen",
    )
    p.add_argument(
        "--size",
        default="",
        metavar="WxH",
        help="composition size, e.g. 1920x1080 or 1440x1440 (default 1080x1920)",
    )
    p.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    p.add_argument("--low-power", action="store_true", help="start in low power mode")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera")
    p.add_argument("--seed", type=int, default=None, help="fix the word shuffle")
    p.add_argument(
        "--autostart",
        action="store_true",
        help="skip waiting for consent on the explainer (kiosk use)",
    )
    p.add_argument("--check", action="store_true", help="validate assets and exit")
    p.add_argument("--demo", action="store_true", help="run with synthetic hands, no camera")
    p.add_argument("--camera", action="store_true",
                   help="use the real camera even with --offscreen")  # fmt: skip
    p.add_argument("--offscreen", type=int, default=0, metavar="N",
                   help="render N frames headless to PNG and exit")  # fmt: skip
    p.add_argument("--out", default="shots", help="output directory for --offscreen")
    p.add_argument("--stride", type=int, default=10, help="save every Nth offscreen frame")
    return p


def check() -> int:
    """Fail loudly at install time rather than quietly at camp."""
    from portal_palms import config as C
    from portal_palms.content.metaphors import SCENES
    from portal_palms.content.words import load_words, unused_scenes

    assets = C.resolve_asset_root()
    ok = True
    for name, path in (
        ("hand landmarker", assets / "models" / "hand_landmarker.task"),
        ("font atlas", assets / "fonts" / "atlas.bin"),
        ("font metrics", assets / "fonts" / "atlas.json"),
    ):
        exists = path.exists()
        ok &= exists
        print(f"{'ok  ' if exists else 'MISS'} {name:16s} {path}")

    words, warnings = load_words()
    print(f"ok   {'words':16s} {len(words)} entries, {len(SCENES)} metaphors")
    for w in warnings:
        print("warn words:", w)
    spare = unused_scenes(words)
    if spare:
        print("note  metaphors with no word:", ", ".join(spare))
    if len(words) < 25:
        print(f"WARN only {len(words)} words; the deck is meant to be at least 25")
    return 0 if ok else 1


def _apply_size(args: argparse.Namespace) -> None:
    """Set the composition size before anything reads it."""
    from portal_palms import config as C

    if args.size:
        try:
            w, h = (int(v) for v in args.size.lower().split("x", 1))
        except ValueError:
            raise SystemExit(f"--size wants WxH, got {args.size!r}") from None
    elif args.landscape:
        w, h = 1920, 1080
    else:
        return
    C.set_render_size(w, h)
    # Match the window to it, unless the user asked for a specific window size.
    if args.width == 600 and args.height == 1000:
        scale = 900 / max(w, h)
        args.width = max(int(w * scale), 480)
        args.height = max(int(h * scale), 320)


def main() -> int:
    args = build_parser().parse_args()
    if args.check:
        return check()
    _apply_size(args)

    from portal_palms.app import App, prewarm_mediapipe

    if not (args.demo or args.offscreen):
        prewarm_mediapipe()
    app = App(args)
    print(f"startup {app.startup_ms:.0f} ms to first interactive frame")
    if args.offscreen:
        from pathlib import Path

        return app.run_offscreen(args.offscreen, Path(args.out), max(args.stride, 1))
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
