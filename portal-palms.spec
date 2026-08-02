# PyInstaller spec for Portal Palms.  Build with:  uv run pyinstaller portal-palms.spec
#
# onedir, not onefile: onefile unpacks ~400MB to a temp directory on every
# launch, which alone blows the cold-start budget several times over.
#
# The exclusion lists below are the difference between a 900MB bundle and one
# that fits on a camp laptop. Everything excluded was verified unused by
# importing the app and checking sys.modules; if you add a feature, check here
# before assuming something is missing.

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH)  # noqa: F821 - injected by PyInstaller

# The glfw wheel finds its native library by walking up from glfw/__init__.py,
# so it has to land back inside a `glfw/` directory rather than in the generic
# binaries pile. Without this the bundle imports fine right up until it does
# not: "Failed to load GLFW3 shared library" at startup.
_glfw_libs = []
try:
    import glfw as _glfw_probe

    _glfw_dir = Path(_glfw_probe.__file__).parent
    _glfw_libs = [(str(p), "glfw") for p in _glfw_dir.glob("*.dylib")]
    _glfw_libs += [(str(p), "glfw") for p in _glfw_dir.glob("*.so")]
    _glfw_libs += [(str(p), "glfw") for p in _glfw_dir.glob("*.dll")]
except ImportError:  # pragma: no cover - build-time only
    pass

datas = [
    (str(ROOT / "assets" / "models" / "hand_landmarker.task"), "assets/models"),
    (str(ROOT / "assets" / "fonts" / "atlas.bin"), "assets/fonts"),
    (str(ROOT / "assets" / "fonts" / "atlas.json"), "assets/fonts"),
    (str(ROOT / "portal_palms" / "content" / "words.json"), "portal_palms/content"),
]
for lic in (ROOT / "assets" / "fonts").glob("OFL-*.txt"):
    datas.append((str(lic), "assets/fonts"))

# MediaPipe ships every task graph it knows about. We use exactly one.
MEDIAPIPE_KEEP = ("hand_landmark", "palm_detection", "hand_detector", "hand_roi")
# Verified with tools/verify_excludes.py, which builds a real landmarker and
# checks nothing on this list ended up in sys.modules. Run it again before
# adding an entry -- a wrong guess here produces a bundle that builds cleanly,
# launches, opens the camera, and only then dies with ModuleNotFoundError.
EXCLUDE_MODULES = [
    # Safe only because portal_palms/mediapipe_slim.py installs a meta-path
    # finder that stops mediapipe reaching for them at all. Remove that and
    # these three break the bundle at landmarker construction.
    "matplotlib",
    "PIL",
    "sounddevice",
    "scipy",
    "pandas",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "mypy",
    "pip",
    "tkinter",
    "pydoc",
    "doctest",
    "xmlrpc",
    # The old non-tasks solutions API. This app uses mediapipe.tasks only.
    "mediapipe.python.solutions",
    "mediapipe.examples",
    "mediapipe.model_maker",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=collect_dynamic_libs("mediapipe") + _glfw_libs,
    datas=datas,
    hiddenimports=[
        "mediapipe.tasks.python.vision",
        "mediapipe.tasks.python.core.base_options",
        "portal_palms.content.metaphors",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDE_MODULES,
    noarchive=False,
)


def _drop_unused_task_graphs(entries):
    """MediaPipe bundles binarypb graphs for every solution it supports."""
    kept = []
    for name, path, kind in entries:
        low = name.lower()
        if low.endswith(".binarypb") and not any(k in low for k in MEDIAPIPE_KEEP):
            continue
        if "/testdata/" in low or low.endswith(".tflite") and "hand" not in low:
            continue
        kept.append((name, path, kind))
    return kept


a.datas = _drop_unused_task_graphs(a.datas)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="portal-palms",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressed dylibs fail Gatekeeper checks on macOS
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="portal-palms",
)
