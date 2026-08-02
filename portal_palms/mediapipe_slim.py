"""Keep MediaPipe's unused optional dependencies out of the process.

`mediapipe.tasks.python.vision.__init__` eagerly imports `drawing_utils`, which
does `import matplotlib.pyplot`, and `...audio.__init__` eagerly imports
`audio_classifier`, which does `import sounddevice`. Neither is reachable from
this app -- it draws its own overlays and has no audio -- but both get loaded
the moment anything under `mediapipe.tasks` is imported.

That is not merely wasteful. In a PyInstaller bundle matplotlib rebuilds its
font cache **on every launch**, because the bundled `mpl-data` does not match
whatever is in the user's cache directory. Measured on the packaged build: 12.2
seconds to first tracked frame with it, against a 3 second budget. From source
it is invisible, because a developer's cache is already warm -- which is exactly
the kind of cost that only shows up on the machines it will actually run on.

The fix is a meta-path finder that hands back inert modules for those names.
Going through the real import machinery (rather than pre-seeding sys.modules)
matters: `mediapipe`'s package inits rely on `import a.b.c` binding `c` as an
attribute of `a.b`, and a raw sys.modules entry skips the step that does that.

Attribute access on a stub returns another stub rather than raising, because
`drawing_styles` reads constants off matplotlib at import time. Calling one
raises with a pointed message, so if anyone ever does route real work through
these they get a clear explanation instead of a mystery.
"""

from __future__ import annotations

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

# Roots to intercept. Submodules of these are intercepted too.
STUBBED: Final = ("matplotlib", "sounddevice")


class _Inert:
    """Returns itself for any attribute; complains loudly only if called."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> _Inert:
        return _Inert(f"{self._name}.{item}")

    def __getitem__(self, item: object) -> _Inert:
        return _Inert(f"{self._name}[...]")

    def __call__(self, *args: object, **kwargs: object) -> Any:
        raise RuntimeError(
            f"{self._name} was called, but Portal Palms stubs it out to keep "
            "matplotlib and sounddevice out of the process. See "
            "portal_palms/mediapipe_slim.py."
        )

    def __iter__(self) -> Any:
        return iter(())

    def __repr__(self) -> str:
        return f"<inert {self._name}>"


class _StubModule(ModuleType):
    def __getattr__(self, item: str) -> _Inert:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Inert(f"{self.__name__}.{item}")


class _StubLoader(Loader):
    def create_module(self, spec: ModuleSpec) -> ModuleType:
        mod = _StubModule(spec.name)
        # A __path__ makes it a package, so submodule imports resolve too.
        mod.__path__ = []
        return mod

    def exec_module(self, module: ModuleType) -> None:
        return None


class _StubFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        root = fullname.partition(".")[0]
        if root not in STUBBED:
            return None
        return ModuleSpec(fullname, _StubLoader(), is_package=True)


def is_stub(module: object) -> bool:
    """True if `module` is one of ours rather than the real library."""
    return isinstance(module, _StubModule)


_installed = False


def install() -> bool:
    """Install the finder. Returns False if the real modules already loaded.

    Must run before anything imports mediapipe. Called from the tracking
    thread's landmarker construction, and from prewarm, both of which happen
    behind the camera explainer.
    """
    global _installed  # one process-wide switch, set once
    if _installed:
        return True
    if any(m.partition(".")[0] in STUBBED for m in sys.modules):
        # Something already imported them for real; leave well alone.
        return False
    sys.meta_path.insert(0, _StubFinder())
    _installed = True
    return True
