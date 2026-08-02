"""Every screen the app can be showing, drawn from the SDF atlas and the SDF shapes.

Sizing note: the app composes into a 1080x1920 frame by default, letterboxed
into a window roughly 550px wide, so one render-frame pixel lands at about 0.52
display pixels. Everything the learner reads is therefore >= 32 render px at the
reference size, which is the "nothing below 16px at 1x" floor. Only the debug
HUD goes smaller.

The composition can be reshaped (--size / --landscape), so the sizes below are
quoted against the 1920-tall reference and scaled by C.UI_SCALE at use. Read
them through px(), never raw.

Microcopy code-switches into Indonesian where a camp student would actually
switch. Failure states are funny about the situation, never about the person --
nobody's accent or mistakes are ever the joke.
"""

from __future__ import annotations

import math
from typing import Final

from portal_palms import config as C
from portal_palms.content.words import Word
from portal_palms.render.text import ALIGN_CENTER, ALIGN_LEFT, ALIGN_RIGHT, ShapeRenderer
from portal_palms.render.text import TextRenderer as Text

Color = tuple[float, float, float]

BASE: Final[Color] = (0.043, 0.059, 0.102)
GREEN: Final[Color] = (0.247, 0.839, 0.549)
YELLOW: Final[Color] = (1.000, 0.773, 0.192)
MAGENTA: Final[Color] = (1.000, 0.239, 0.498)
WHITE: Final[Color] = (0.949, 0.957, 0.941)
DIM: Final[Color] = (0.42, 0.48, 0.57)

# Type scale, in render-frame pixels at the 1920-tall reference.
PX_HERO: Final = 185.0
PX_TITLE: Final = 108.0
PX_LEAD: Final = 46.0
PX_BODY: Final = 40.0
PX_META: Final = 34.0
PX_MICRO: Final = 32.0


def px(size: float) -> float:
    """Reference type size -> render-frame pixels for the current composition."""
    return size * C.UI_SCALE


def column() -> float:
    """Half-width of the centred content column, in uv.

    Near full-bleed in portrait. On a wide frame a full-bleed column would set
    the example sentence in one enormous line across the whole desktop, so it
    is capped to something with a readable measure.
    """
    return 0.435 if C.is_portrait() else 0.30


def _panel(s: ShapeRenderer, cx: float, cy: float, hw: float, hh: float, a: float = 0.88) -> None:
    s.rect(cx, cy, hw, hh, (0.02, 0.03, 0.055), a, 0.022)
    s.rect(cx, cy, hw, hh, GREEN, a * 0.20, 0.022, 0.0016)


def top_bar(t: Text, s: ShapeRenderer, streak: int, completed: int, num: object) -> None:
    t.add("PORTAL PALMS", t.mono, 0.06, 0.030, px(PX_MICRO), DIM, 0.75, ALIGN_LEFT, 0.10)
    label = "STREAK " + str(streak) + "  /  " + str(completed) + " WORDS"
    t.add(label, t.mono, 0.94, 0.030, px(PX_MICRO), GREEN, 0.8, ALIGN_RIGHT, 0.06)


# --------------------------------------------------------------------------- boot


def explainer(t: Text, s: ShapeRenderer, phase: float) -> None:
    """Shown BEFORE the camera device is opened.

    On macOS the OS permission sheet fires on the first VideoCapture.open, so
    this has to already be on screen when that happens or the user gets an
    unexplained system dialog.
    """
    _panel(s, 0.5, 0.5, min(column(), 0.40), 0.24)
    t.add("PORTAL", t.display, 0.5, 0.315, px(PX_TITLE), WHITE, 1.0, ALIGN_CENTER, -0.02)
    t.add("PALMS", t.display, 0.5, 0.372, px(PX_TITLE), GREEN, 1.0, ALIGN_CENTER, -0.02)

    t.wrap(
        "Two hands. Pull them apart to tear open a portal and see what a word "
        "actually means.",
        t.body, 0.5, 0.455, px(PX_BODY), column() * 1.38, WHITE, 0.92, ALIGN_CENTER,
    )  # fmt: skip
    t.wrap(
        "Next: your camera. It runs on this laptop only. Nothing leaves the "
        "machine, and nothing is saved unless you press R to record.",
        t.body, 0.5, 0.545, px(PX_META), column() * 1.38, DIM, 1.0, ALIGN_CENTER,
    )  # fmt: skip

    pulse = 0.65 + 0.35 * math.sin(phase * 3.4)
    t.add("PRESS SPACE TO START", t.bodybold, 0.5, 0.645, px(PX_LEAD), YELLOW, pulse, ALIGN_CENTER,
          0.08)  # fmt: skip
    t.add("tekan spasi untuk mulai", t.body, 0.5, 0.685, px(PX_META), DIM, 0.9, ALIGN_CENTER)


def loading(t: Text, s: ShapeRenderer, phase: float, message: str) -> None:
    s.arc(0.5, 0.47, 0.075, 0.006, 1.0, (0.14, 0.18, 0.26), 1.0)
    s.arc(0.5, 0.47, 0.075, 0.006, (phase * 0.55) % 1.0, GREEN, 1.0)
    t.add(message, t.body, 0.5, 0.565, px(PX_LEAD), WHITE, 0.9, ALIGN_CENTER)


def camera_error(t: Text, s: ShapeRenderer, detail: str) -> None:
    _panel(s, 0.5, 0.5, min(column(), 0.40), 0.17)
    t.add("NO CAMERA", t.display, 0.5, 0.395, px(PX_TITLE), MAGENTA, 1.0, ALIGN_CENTER, -0.02)
    t.wrap(
        "The camera did not answer. Another app may still be holding it, or "
        "the permission was declined.",
        t.body, 0.5, 0.475, px(PX_BODY), column() * 1.38, WHITE, 0.9, ALIGN_CENTER,
    )  # fmt: skip
    t.wrap(
        "Close Zoom or Photo Booth, then press R to try again. "
        "On macOS check System Settings > Privacy & Security > Camera.",
        t.body, 0.5, 0.555, px(PX_META), column() * 1.38, DIM, 1.0, ALIGN_CENTER,
    )  # fmt: skip
    if detail:
        t.add(detail[:48], t.mono, 0.5, 0.625, px(PX_MICRO), DIM, 0.6, ALIGN_CENTER)


# --------------------------------------------------------------------------- live


def hands_hint(t: Text, s: ShapeRenderer, phase: float, one_hand: bool) -> None:
    """The empty state. There is never a blank screen in this app."""
    y = 0.795
    pulse = 0.6 + 0.4 * math.sin(phase * 2.6)
    if one_hand:
        t.add("ONE MORE HAND", t.display, 0.5, y, px(PX_TITLE * 0.72), YELLOW, pulse, ALIGN_CENTER,
              -0.01)  # fmt: skip
        t.add("satu lagi -- it takes two to tear", t.body, 0.5, y + 0.048, px(PX_META), DIM, 1.0,
              ALIGN_CENTER)  # fmt: skip
    else:
        t.add("SHOW ME YOUR HANDS", t.display, 0.5, y, px(PX_TITLE * 0.72), WHITE, pulse,
              ALIGN_CENTER, -0.01)  # fmt: skip
        t.add("angkat dua tangan, palms toward the camera", t.body, 0.5, y + 0.048, px(PX_META),
              DIM, 1.0, ALIGN_CENTER)  # fmt: skip


def pull_hint(t: Text, s: ShapeRenderer, openness: float) -> None:
    a = (1.0 - openness) * 0.85
    if a < 0.03:
        return
    t.add("PULL THEM APART", t.bodybold, 0.5, 0.845, px(PX_LEAD), GREEN, a, ALIGN_CENTER, 0.10)


def hold_ring(
    t: Text, s: ShapeRenderer, cx: float, cy: float, radius: float, progress: float
) -> None:
    """The hold-to-lock countdown, drawn just outside the rim."""
    if progress <= 0.001:
        return
    r = radius * 1.32 + 0.03
    s.arc(cx, cy, r, 0.0035, 1.0, (0.16, 0.20, 0.28), 0.55)
    s.arc(cx, cy, r, 0.0055, progress, YELLOW, 0.95)
    t.add("HOLD", t.mono, cx, cy - r * (C.RENDER_W / C.RENDER_H) - 0.030, px(PX_MICRO), YELLOW,
          0.85, ALIGN_CENTER, 0.22)  # fmt: skip


# --------------------------------------------------------------------------- reveal


def word_card(t: Text, s: ShapeRenderer, w: Word, reveal: float, phase: float) -> None:
    """Word, part of speech, one natural example sentence.

    Slides up and fades in so it does not cut the scene off the moment the
    portal locks; the animation gets a beat on its own first.
    """
    if reveal <= 0.001:
        return
    k = reveal * reveal * (3.0 - 2.0 * reveal)
    a = k

    # Laid out in pixels and converted to uv once, rather than in hard-coded uv
    # fractions. `y` for text is the top of the *ascent*, and a capital in the
    # display face occupies about 1.06x its point size below that -- the number
    # everything here is spaced off. Fixed uv offsets look right at 1080x1920
    # and collide the moment the composition is reshaped.
    inv_h = 1.0 / C.RENDER_H
    hero = px(PX_HERO)
    meta = px(PX_META)
    body = px(PX_BODY)
    micro = px(PX_MICRO)
    measure = column() * 1.8
    lines = 2 if t.body.measure(w.example, body) > measure * C.RENDER_W else 1

    pad = hero * 0.17
    h_word = hero * 1.06
    h_pos = meta * 1.32 if w.pos else 0.0
    h_ex = (body * 1.21 * 1.22 * lines + body * 0.45) if w.example else 0.0
    h_hint = micro * 1.95
    half = (pad + h_word + h_pos + h_ex + h_hint + pad) * 0.5 * inv_h

    centre = 0.962 - half + (1.0 - k) * 0.045
    top = centre - half
    _panel(s, 0.5, centre, column(), half, 0.82 * a)

    y = top + pad * inv_h
    t.add(w.display, t.display, 0.5, y, hero, WHITE, a, ALIGN_CENTER, -0.025)
    y += h_word * inv_h
    if w.pos:
        t.add(w.pos.upper(), t.mono, 0.5, y, meta, GREEN, a * 0.95, ALIGN_CENTER, 0.18)
        y += h_pos * inv_h
    if w.example:
        t.wrap(w.example, t.body, 0.5, y, body, measure, WHITE, a * 0.92, ALIGN_CENTER)
        y += h_ex * inv_h

    if reveal > 0.85:
        pulse = 0.45 + 0.35 * math.sin(phase * 2.8)
        t.add("SWIPE OR PRESS SPACE FOR THE NEXT ONE", t.mono, 0.5, y, micro, DIM,
              pulse, ALIGN_CENTER, 0.10)  # fmt: skip


# --------------------------------------------------------------------------- recording


def countdown(t: Text, s: ShapeRenderer, remaining: float) -> None:
    n = math.ceil(remaining)
    if n <= 0:
        return
    frac = 1.0 - (remaining - math.floor(remaining))
    scale = 1.0 + (1.0 - frac) * 0.35
    s.rect(0.5, 0.5, 0.5, 0.5, BASE, 0.55)
    t.add(str(n), t.display, 0.5, 0.40, px(PX_HERO * 1.7) * scale, WHITE, 0.35 + frac * 0.65,
          ALIGN_CENTER)  # fmt: skip
    t.add("RECORDING IN", t.mono, 0.5, 0.355, px(PX_META), YELLOW, 0.9, ALIGN_CENTER, 0.22)


def recording_badge(t: Text, s: ShapeRenderer, seconds: float, phase: float) -> None:
    blink = 0.55 + 0.45 * math.sin(phase * 6.0)
    s.rect(0.5, 0.072, 0.115, 0.019, (0.02, 0.03, 0.055), 0.8, 0.019)
    s.rect(0.418, 0.072, 0.010, 0.010, MAGENTA, blink, 0.010)
    left = max(C.RECORD_MAX_SECONDS - seconds, 0.0)
    t.add("REC  " + str(int(left)) + "s", t.mono, 0.512, 0.0605, px(PX_MICRO), WHITE, 0.95,
          ALIGN_CENTER, 0.12)  # fmt: skip


def clip_preview(t: Text, s: ShapeRenderer, name: str, saved: bool, error: str | None) -> None:
    _panel(s, 0.5, 0.5, min(column(), 0.40), 0.16)
    if error:
        t.add("CLIP FAILED", t.display, 0.5, 0.400, px(PX_TITLE * 0.8), MAGENTA, 1.0, ALIGN_CENTER)
        t.wrap("The encoder gave up. The portal is fine, the recorder is not.",
               t.body, 0.5, 0.470, px(PX_BODY), 0.60, WHITE, 0.9, ALIGN_CENTER)  # fmt: skip
        t.add(error[:44], t.mono, 0.5, 0.545, px(PX_MICRO), DIM, 0.7, ALIGN_CENTER)
        t.add("PRESS SPACE TO CARRY ON", t.mono, 0.5, 0.600, px(PX_META), YELLOW, 0.9,
              ALIGN_CENTER, 0.14)  # fmt: skip
        return

    t.add("CLIP READY", t.display, 0.5, 0.392, px(PX_TITLE * 0.8), GREEN, 1.0, ALIGN_CENTER, -0.02)
    t.add(name, t.mono, 0.5, 0.462, px(PX_META), WHITE, 0.85, ALIGN_CENTER)
    if saved:
        t.add("saved", t.body, 0.5, 0.500, px(PX_META), GREEN, 0.9, ALIGN_CENTER)
    t.add("S   KEEP IT", t.bodybold, 0.5, 0.545, px(PX_LEAD), WHITE, 0.95, ALIGN_CENTER, 0.06)
    t.add("F   SHOW ME WHERE", t.bodybold, 0.5, 0.585, px(PX_LEAD), WHITE, 0.95, ALIGN_CENTER, 0.06)
    t.add("X   BIN IT", t.bodybold, 0.5, 0.625, px(PX_LEAD), DIM, 0.95, ALIGN_CENTER, 0.06)


# --------------------------------------------------------------------------- settings


def settings_drawer(
    t: Text,
    s: ShapeRenderer,
    slide: float,
    rows: list[tuple[str, str, bool]],
    selected: int,
) -> None:
    if slide <= 0.001:
        return
    k = slide * slide * (3.0 - 2.0 * slide)
    h = 0.30
    top = 1.0 - h * k
    s.rect(0.5, top + h * 0.5, 0.5, h * 0.5, (0.02, 0.03, 0.055), 0.95 * k, 0.028)
    s.rect(0.5, top + 0.004, 0.5, 0.0012, GREEN, 0.35 * k)

    t.add("SETTINGS", t.mono, 0.5, top + 0.022, px(PX_META), GREEN, k * 0.9, ALIGN_CENTER, 0.24)
    y = top + 0.066
    for i, (label, value, on) in enumerate(rows):
        sel = i == selected
        if sel:
            s.rect(0.5, y + 0.016, 0.44, 0.019, GREEN, 0.10 * k, 0.010)
        t.add(label, t.body, 0.10, y, px(PX_BODY), WHITE if sel else DIM, k, ALIGN_LEFT)
        t.add(value, t.bodybold, 0.90, y, px(PX_BODY), GREEN if on else DIM, k, ALIGN_RIGHT)
        y += 0.042
    t.add("ARROWS TO CHANGE   S TO CLOSE", t.mono, 0.5, top + h - 0.030, px(PX_MICRO), DIM,
          k * 0.8, ALIGN_CENTER, 0.10)  # fmt: skip


def toast(t: Text, s: ShapeRenderer, message: str, a: float) -> None:
    """Bottom-of-screen notice. Used for things the app did on its own."""
    if a <= 0.01:
        return
    s.rect(0.5, 0.905, 0.34, 0.022, (0.02, 0.03, 0.055), 0.88 * a, 0.022)
    s.rect(0.5, 0.905, 0.34, 0.022, YELLOW, 0.25 * a, 0.022, 0.0016)
    t.add(message, t.body, 0.5, 0.8925, px(PX_META), YELLOW, a, ALIGN_CENTER)
