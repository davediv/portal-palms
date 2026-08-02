"""The deck and the metaphor scenes.

These run without a GL context: the Painter writes into plain numpy arrays, so
every scene can be exercised for loop continuity, instance budget and NaNs
without a window.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from portal_palms.content.metaphors import SCENES, fade_cycle, saw, tri
from portal_palms.content.words import Deck, load_words, unused_scenes
from portal_palms.render.primitives import INSTANCE_FLOATS, MAX_PER_PRIM, PRIM_COUNT, Painter


class FakePool:
    """A Painter target with no GL behind it."""

    def __init__(self) -> None:
        self.instances = [
            np.zeros((MAX_PER_PRIM, INSTANCE_FLOATS), dtype=np.float32) for _ in range(PRIM_COUNT)
        ]


@pytest.fixture
def painter() -> Painter:
    return Painter(FakePool())  # type: ignore[arg-type]


SCENE_NAMES = sorted(SCENES)


# --------------------------------------------------------------------------- deck


def test_the_shipped_deck_meets_the_spec_minimum() -> None:
    words, warnings = load_words()
    assert not warnings, warnings
    assert len(words) >= 25, f"the spec asks for at least 25 words, found {len(words)}"


def test_every_word_has_a_metaphor_and_a_sentence() -> None:
    words, _ = load_words()
    for w in words:
        assert w.scene_name in SCENES, f"{w.word} points at a missing scene"
        assert w.pos, f"{w.word} has no part of speech"
        assert len(w.example) > 10, f"{w.word} has no usable example sentence"
        # Match on a stem, so "cope" is satisfied by "coping" and "hesitate" by
        # "hesitated". Inflection is the normal case in a natural sentence.
        low = w.word.lower()
        stem = (low[:-1] if low.endswith("e") else low)[:5]
        assert stem in w.example.lower(), f"{w.word}'s example does not actually use the word"


def test_display_form_is_pre_uppercased() -> None:
    """The display face has no lowercase glyphs, and .upper() in a frame allocates."""
    words, _ = load_words()
    for w in words:
        assert w.display == w.word.upper()


def test_no_duplicate_words_or_scenes() -> None:
    words, _ = load_words()
    assert len({w.word for w in words}) == len(words)
    assert len({w.scene_name for w in words}) == len(words)


def test_every_metaphor_is_reachable_from_the_deck() -> None:
    words, _ = load_words()
    assert unused_scenes(words) == []


def test_words_json_is_valid_and_documented() -> None:
    from portal_palms import config as C

    raw = json.loads(C.WORDS_PATH.read_text())
    assert "_comment" in raw, "a tutor opening this file needs to know how to edit it"
    assert isinstance(raw["words"], list)


def test_deck_rotation_is_symmetric() -> None:
    words, _ = load_words()
    d = Deck(words, seed=1)
    first = d.current.word
    d.advance(1)
    d.advance(-1)
    assert d.current.word == first


def test_deck_wraps_in_both_directions() -> None:
    words, _ = load_words()
    d = Deck(words, seed=1)
    for _ in range(len(words)):
        d.advance(1)
    assert d.current.word == d.words[d.order[0]].word


def test_deck_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError):
        Deck([])


def test_unknown_scene_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A tutor adding vocabulary must not be able to make the app fail to launch."""
    p = tmp_path / "deck.json"
    p.write_text(
        json.dumps(
            {
                "words": [
                    {"word": "regret", "pos": "noun", "example": "I regret it.",
                     "scene": "regret"},
                    {"word": "gizmo", "pos": "noun", "example": "A gizmo.",
                     "scene": "no_such_scene"},
                ]
            }
        )
    )  # fmt: skip
    words, warnings = load_words(p)
    assert [w.word for w in words] == ["regret"]
    assert len(warnings) == 1 and "no_such_scene" in warnings[0]


# --------------------------------------------------------------------------- scenes


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_paints_within_the_instance_budget(name: str, painter: Painter) -> None:
    sc = SCENES[name]
    for i in range(24):
        painter.reset()
        sc.paint(painter, sc.loop * i / 24.0)
        assert painter.overflow == 0, f"{name} overflowed a primitive pool"


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_produces_finite_transforms(name: str, painter: Painter) -> None:
    sc = SCENES[name]
    for i in range(24):
        painter.reset()
        sc.paint(painter, sc.loop * i / 24.0)
        for kind in range(PRIM_COUNT):
            n = int(painter.counts[kind])
            if n:
                rows = painter.inst[kind][:n]
                assert np.isfinite(rows).all(), f"{name} produced NaN or inf"


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_stays_inside_the_camera_box(name: str, painter: Painter) -> None:
    """Anything far outside the visible box is a composition mistake, not a choice."""
    sc = SCENES[name]
    for i in range(24):
        painter.reset()
        sc.paint(painter, sc.loop * i / 24.0)
        for kind in range(PRIM_COUNT):
            n = int(painter.counts[kind])
            if not n:
                continue
            pos = painter.inst[kind][:n, 0:3]
            assert np.abs(pos[:, 0]).max() < 30.0, f"{name} placed something far off to the side"
            assert pos[:, 1].min() > -12.0 and pos[:, 1].max() < 16.0, f"{name} y is out of range"


def _visible_mass(painter: Painter) -> tuple[float, float]:
    """Alpha-weighted total and centroid of everything currently painted.

    Weighted by alpha on purpose. Several scenes depict something receding
    forever by running two copies half a period apart and cross-fading them, so
    an instance does teleport across the loop boundary -- but it is at zero
    alpha when it does, which is invisible and entirely correct.
    """
    total = 0.0
    weighted = 0.0
    for kind in range(PRIM_COUNT):
        n = int(painter.counts[kind])
        if not n:
            continue
        rows = painter.inst[kind][:n]
        alpha = rows[:, 13]
        total += float(alpha.sum())
        weighted += float((rows[:, 0:3].sum(axis=1) * alpha).sum())
    return total, weighted / max(total, 1e-6)


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_loops_without_a_visible_jump(name: str, painter: Painter) -> None:
    """t arrives wrapped into [0, loop), so the wrap must not be visible."""
    sc = SCENES[name]
    eps = 1.0 / 120.0

    def sample(t: float) -> tuple[float, float]:
        painter.reset()
        sc.paint(painter, t % sc.loop)
        return _visible_mass(painter)

    total_before, centroid_before = sample(sc.loop - eps)
    total_after, centroid_after = sample(0.0)
    assert abs(total_after - total_before) < 1.2, (
        f"{name} pops visible geometry in or out at the loop boundary"
    )
    assert abs(centroid_after - centroid_before) < 0.6, f"{name} jumps at the loop boundary"


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_actually_animates(name: str, painter: Painter) -> None:
    """A still image is not a metaphor. Something has to move.

    Measured per instance rather than on the centroid, because plenty of these
    move symmetrically -- a rocking deck, a box turned over in place -- and a
    centroid that stays put is exactly what those should produce.
    """
    sc = SCENES[name]
    frames = []
    for i in range(8):
        painter.reset()
        sc.paint(painter, sc.loop * i / 8.0)
        frames.append([painter.inst[k][: int(painter.counts[k])].copy() for k in range(PRIM_COUNT)])
    moved = 0.0
    for a, b in itertools.pairwise(frames):
        for ra, rb in zip(a, b, strict=True):
            n = min(ra.shape[0], rb.shape[0])
            if n:
                moved = max(moved, float(np.abs(ra[:n] - rb[:n]).max()))
    assert moved > 0.05, f"{name} barely moves"


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_alpha_and_colour_are_in_range(name: str, painter: Painter) -> None:
    sc = SCENES[name]
    for i in range(12):
        painter.reset()
        sc.paint(painter, sc.loop * i / 12.0)
        for kind in range(PRIM_COUNT):
            n = int(painter.counts[kind])
            if not n:
                continue
            rows = painter.inst[kind][:n]
            assert rows[:, 10:14].min() >= -0.001, f"{name} has a negative colour"
            assert rows[:, 13].max() <= 1.001, f"{name} has alpha above 1"


def test_loop_lengths_are_sane() -> None:
    for name, sc in SCENES.items():
        assert 1.5 <= sc.loop <= 8.0, f"{name} loops every {sc.loop}s"


# --------------------------------------------------------------------------- easing


def test_saw_and_tri_stay_in_range() -> None:
    for i in range(200):
        t = i * 0.137
        assert 0.0 <= saw(t, 3.0) < 1.0
        assert 0.0 <= tri(t, 3.0) <= 1.0


def test_fade_cycle_is_zero_at_both_ends() -> None:
    assert fade_cycle(0.0) == 0.0
    assert fade_cycle(1.0) == 0.0
    assert fade_cycle(0.5) == pytest.approx(1.0)


def test_painter_reports_overflow_instead_of_writing_out_of_bounds(painter: Painter) -> None:
    for i in range(MAX_PER_PRIM + 25):
        painter.ball(0.0, float(i), 0.0, 0.1, (1.0, 1.0, 1.0))
    assert painter.overflow == 25
    assert int(painter.counts[1]) == MAX_PER_PRIM


def test_painter_rod_handles_degenerate_and_antiparallel_cases(painter: Painter) -> None:
    painter.rod(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, (1.0, 1.0, 1.0))
    assert int(painter.counts[2]) == 0, "a zero-length rod should be skipped"
    painter.rod(0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.1, (1.0, 1.0, 1.0))
    row = painter.inst[2][0]
    q = row[6:10]
    assert math.isclose(float(np.dot(q, q)), 1.0, abs_tol=1e-4), "quaternion must be unit"
