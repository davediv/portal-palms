"""Session progress, stored as one small JSON file in the per-platform data dir.

Written atomically -- temp file plus os.replace -- because the most likely time
for this app to be killed is mid-session on a shared camp laptop, and a
half-written progress file that throws on next launch is worse than no file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

APP_DIR_NAME = "PortalPalms"


def user_data_dir() -> Path:
    # Table rather than if/elif: mypy narrows sys.platform to the host and calls
    # the other branches unreachable, which silences real errors in them.
    home = Path.home()
    bases: dict[str, Path] = {
        "darwin": home / "Library" / "Application Support",
        "win32": Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")),
    }
    base = bases.get(sys.platform, Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")))
    return base / APP_DIR_NAME


def clips_dir() -> Path:
    folder = "Movies" if sys.platform == "darwin" else "Videos"
    return Path.home() / folder / APP_DIR_NAME


@dataclass
class Progress:
    streak: int = 0
    best: int = 0
    completed: int = 0
    last_day: str = ""
    seen: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: Path | None = None) -> Progress:
        p = path or (user_data_dir() / "progress.json")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing or corrupt: start fresh rather than refusing to launch.
            return cls()
        return cls(
            streak=int(raw.get("streak", 0)),
            best=int(raw.get("best", 0)),
            completed=int(raw.get("completed", 0)),
            last_day=str(raw.get("last_day", "")),
            seen=[str(s) for s in raw.get("seen", [])][-500:],
        )

    def save(self, path: Path | None = None) -> None:
        p = path or (user_data_dir() / "progress.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "streak": self.streak,
                "best": self.best,
                "completed": self.completed,
                "last_day": self.last_day,
                "seen": self.seen[-500:],
            },
            indent=1,
        )
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".progress-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp).replace(p)
        except OSError:
            Path(tmp).unlink(missing_ok=True)

    # ------------------------------------------------------------------ logic

    def touch_today(self, today: date | None = None) -> None:
        """Roll the streak forward. Same day is a no-op; a skipped day resets."""
        d = today or date.today()
        iso = d.isoformat()
        if self.last_day == iso:
            return
        if self.last_day == (d - timedelta(days=1)).isoformat():
            self.streak += 1
        else:
            self.streak = 1
        self.last_day = iso
        self.best = max(self.best, self.streak)

    def record(self, word: str) -> None:
        self.completed += 1
        if word not in self.seen:
            self.seen.append(word)
        self.touch_today()
