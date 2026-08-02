"""The word deck: loading, validating, and ordering it.

Everything here runs at startup. The render loop only ever indexes into the
already-built list and reads attributes off a frozen object.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from portal_palms import config as C
from portal_palms.content.metaphors import SCENES, Scene


@dataclass(frozen=True, slots=True)
class Word:
    word: str
    pos: str
    example: str
    scene_name: str
    scene: Scene
    display: str  # pre-uppercased: the display face has no lowercase glyphs


class Deck:
    """An ordered deck with a shuffled-but-stable rotation."""

    def __init__(self, words: list[Word], seed: int | None = None) -> None:
        if not words:
            raise ValueError("the word list is empty")
        self.words = words
        self.order = list(range(len(words)))
        # Shuffled so a repeat session is not identical, but fixed for the run
        # so "next" and "previous" are symmetric.
        random.Random(seed).shuffle(self.order)
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.words)

    @property
    def current(self) -> Word:
        return self.words[self.order[self.cursor]]

    def advance(self, step: int = 1) -> Word:
        self.cursor = (self.cursor + step) % len(self.order)
        return self.current


def load_words(path: Path | None = None) -> tuple[list[Word], list[str]]:
    """Read the deck. Returns (words, warnings); never raises on bad entries."""
    src = path or C.WORDS_PATH
    warnings: list[str] = []
    raw = json.loads(src.read_text(encoding="utf-8"))
    entries = raw.get("words") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{src}: expected a list of words, or an object with a 'words' key")

    out: list[Word] = []
    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            warnings.append(f"entry {i} is not an object; skipped")
            continue
        word = str(item.get("word", "")).strip()
        if not word:
            warnings.append(f"entry {i} has no 'word'; skipped")
            continue
        scene_name = str(item.get("scene", word)).strip()
        scene = SCENES.get(scene_name)
        if scene is None:
            # A tutor adding vocabulary should get their word in the deck even
            # before anyone animates it, rather than a crash on startup.
            warnings.append(f"'{word}': no metaphor named '{scene_name}'; skipped")
            continue
        out.append(
            Word(
                word=word,
                pos=str(item.get("pos", "")).strip(),
                example=str(item.get("example", "")).strip(),
                scene_name=scene_name,
                scene=scene,
                display=word.upper(),
            )
        )
    return out, warnings


def unused_scenes(words: list[Word]) -> list[str]:
    used = {w.scene_name for w in words}
    return sorted(set(SCENES) - used)
