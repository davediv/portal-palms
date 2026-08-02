"""Check the PyInstaller exclusion list against what the app actually imports.

    uv run python tools/verify_excludes.py

Exercises the real startup path -- landmarker construction, GL, camera imports --
and then reports any module on the spec's exclusion list that ended up in
sys.modules anyway. Anything it names would produce a bundle that builds
cleanly, launches, and then dies at runtime with ModuleNotFoundError.

Run this after touching EXCLUDE_MODULES, and after any mediapipe upgrade.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def spec_exclusions() -> list[str]:
    """Read EXCLUDE_MODULES out of the spec without executing it."""
    tree = ast.parse((ROOT / "portal-palms.spec").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "EXCLUDE_MODULES" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            return [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    raise SystemExit("EXCLUDE_MODULES not found in portal-palms.spec")


def main() -> int:
    from portal_palms import config as C
    from portal_palms import mediapipe_slim

    excluded = spec_exclusions()
    # Exactly what the app does before it touches mediapipe. Without this the
    # matplotlib and sounddevice exclusions would look unsafe.
    mediapipe_slim.install()

    # Everything the real app touches on the way to its first tracked frame.
    import cv2  # noqa: F401
    import glfw  # noqa: F401
    import moderngl  # noqa: F401
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    model = C.resolve_asset_root() / "models" / "hand_landmarker.task"
    landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=C.NUM_HANDS,
        )
    )
    landmarker.close()

    # A stubbed name is present in sys.modules but none of the real library is,
    # so excluding it from the bundle is exactly right.
    loaded = {name for name, mod in sys.modules.items() if not mediapipe_slim.is_stub(mod)}
    bad = sorted(
        name for name in excluded if any(m == name or m.startswith(name + ".") for m in loaded)
    )
    stubbed = sorted(n for n in excluded if mediapipe_slim.is_stub(sys.modules.get(n)))
    print(f"checked {len(excluded)} exclusions against {len(loaded)} real modules")
    if stubbed:
        print(f"stubbed by mediapipe_slim: {', '.join(stubbed)}")
    if bad:
        print("\nEXCLUDED BUT ACTUALLY IMPORTED -- the bundle will break at runtime:")
        for name in bad:
            print(f"  {name}")
        return 1
    print("all exclusions safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
