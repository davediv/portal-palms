"""The metaphor scenes. This is the product.

Each scene is a small looping animation built from the shared primitive pool in
render/primitives.py. There is no per-word geometry: a scene is a function that
writes instance transforms, so switching words costs a buffer write.

Authoring rules, learned the hard way while making these guessable:

* Compose inside x in [-3, 3], y in [0, 3.2], z in [-4, 1.5]. That is what the
  scene camera sees through the ellipse.
* `t` arrives already wrapped into [0, loop), so every scene must be continuous
  across the wrap. Anything that travels one way forever needs two copies half a
  period apart, each fading, rather than a snap back to the start.
* One idea per scene. If a viewer has to watch twice, it is too complicated.
* Colour carries meaning: WARM is what the figure wants, HOT is what presses on
  them, COOL is neutral world. Never decorate with colour.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, NamedTuple

from portal_palms.render.primitives import AX_X, AX_Y, AX_Z, Painter

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

Color = tuple[float, float, float]

# Plain tuples, not numpy rows: indexing a tuple hands back a cached float and
# allocates nothing, which matters because these are read every frame.
BODY: Final[Color] = (0.78, 0.82, 0.87)
COOL: Final[Color] = (0.34, 0.40, 0.50)
DARK: Final[Color] = (0.13, 0.16, 0.23)
WARM: Final[Color] = (1.00, 0.773, 0.192)
HOT: Final[Color] = (1.00, 0.239, 0.498)
# Just self-lit enough to hold a silhouette against the interior's own darkness.
MASS: Final[Color] = (0.26, 0.10, 0.20)
LIFE: Final[Color] = (0.247, 0.839, 0.549)
PALE: Final[Color] = (0.949, 0.957, 0.941)

TAU: Final = math.tau


# --------------------------------------------------------------------------- easing


def saw(t: float, period: float) -> float:
    """0..1 ramp."""
    return (t % period) / period


def tri(t: float, period: float) -> float:
    """0..1..0 triangle."""
    x = (t % period) / period
    return x * 2.0 if x < 0.5 else 2.0 - 2.0 * x


def ease(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def pulse(x: float, centre: float, width: float) -> float:
    """A soft bump peaking at `centre`."""
    d = abs(x - centre) / width
    return 0.0 if d >= 1.0 else 1.0 - d * d


def loop_fade(u: float, tail: float = 0.09) -> float:
    """0 at both ends of the loop, 1 through the middle.

    For the scenes whose idea genuinely requires a reset -- the stack has to be
    empty again, the timeline has to start over. Fading through the seam is
    what makes the reset land as "and again tomorrow" instead of as a glitch.
    """
    return ease(min(u / tail, 1.0)) * ease(min((1.0 - u) / tail, 1.0))


def fade_cycle(x: float) -> float:
    """Full alpha through the middle of a pass, zero at both ends.

    This is what makes "something that keeps receding" loop without a visible
    snap: run two copies half a period apart and multiply each by this.
    """
    return ease(min(x, 1.0 - x) * 4.0)


class Scene(NamedTuple):
    paint: Callable[[Painter, float], None]
    loop: float
    tint: Color


SCENES: dict[str, Scene] = {}


def scene(
    name: str, loop: float, tint: Color
) -> Callable[[Callable[[Painter, float], None]], Callable[[Painter, float], None]]:
    def deco(fn: Callable[[Painter, float], None]) -> Callable[[Painter, float], None]:
        SCENES[name] = Scene(fn, loop, tint)
        return fn

    return deco


# --------------------------------------------------------------------------- shared props


def doorway(
    p: Painter, x: float, z: float, h: float, w: float, glow: float, a: float = 1.0
) -> None:
    """An upright frame with light behind it. Used by several scenes."""
    p.box(x, h * 0.5, z, 0.14, h, 0.14, COOL, 0.0, AX_Z, a)
    p.box(x + w, h * 0.5, z, 0.14, h, 0.14, COOL, 0.0, AX_Z, a)
    p.box(x + w * 0.5, h, z, w + 0.14, 0.14, 0.14, COOL, 0.0, AX_Z, a)
    if glow > 0.001:
        p.box(x + w * 0.5, h * 0.5, z - 0.1, w, h, 0.02, WARM, 0.0, AX_Z, a * glow, glow * 0.9)


def upright_ring(
    p: Painter,
    x: float,
    y: float,
    z: float,
    d: float,
    col: Color,
    a: float = 1.0,
    emis: float = 0.0,
) -> None:
    """A ring standing up facing the camera.

    The torus primitive lies in XZ, so it needs a quarter turn about X to be
    seen as a circle rather than as a flat ellipse on the ground.
    """
    p.ring(x, y, z, d, col, 1.5708, AX_X, a, emis)


# --------------------------------------------------------------------------- the scenes


@scene("regret", 4.4, HOT)
def m_regret(p: Painter, t: float) -> None:
    """A figure walking backward toward a door that keeps receding."""
    p.floor()
    step = math.sin(t * TAU / 1.1)
    # Backward walk: the body faces away from the door while the feet carry it
    # toward the door. The lean backwards is what sells "backward".
    sway = math.sin(t * TAU / 4.4) * 0.12
    p.figure(-1.4 + sway, 0.0, 0.4, 1.15, BODY,
             lean=-0.30, stride=step, arm=-step * 0.7, facing=-1.0)  # fmt: skip
    # Drag marks trailing behind the heels, pointing the way they came.
    for i in range(4):
        p.box(-2.1 - i * 0.42 + sway, 0.02, 0.55, 0.3, 0.02, 0.1, COOL,
              0.0, AX_Y, 0.30 - i * 0.06, 0.1)  # fmt: skip
    for phase in (0.0, 0.5):
        u = (t / 4.4 + phase) % 1.0
        # Recedes and shrinks; a second copy half a loop behind takes over so
        # the retreat never resets on screen.
        s = 1.0 - u * 0.55
        doorway(p, 0.9 + u * 2.4, -u * 4.0, 2.2 * s, 1.15 * s, 0.9 * (1.0 - u), fade_cycle(u))


@scene("curiosity", 3.6, WARM)
def m_curiosity(p: Painter, t: float) -> None:
    """A small light investigating a dark box that keeps reopening."""
    p.floor()
    u = saw(t, 3.6)
    lid = ease(pulse(u, 0.45, 0.34))  # opens, then shuts again
    p.box(0.0, 0.45, 0.0, 1.5, 0.9, 1.2, DARK)
    p.box(0.0, 0.92 + lid * 0.55, -0.55 - lid * 0.2, 1.55, 0.10, 1.25, COOL, -lid * 0.85, AX_X)
    if lid > 0.05:
        p.box(0.0, 0.88, 0.0, 1.2, 0.02, 0.95, WARM, 0.0, AX_Y, lid, lid * 1.4)

    # The light circles at a distance, then dives in the moment the lid lifts.
    ang = t * 1.9
    near = ease(pulse(u, 0.5, 0.26))
    r = 1.85 - near * 1.55
    lx = math.cos(ang) * r
    lz = math.sin(ang) * r * 0.6
    ly = 1.75 - near * 0.75 + math.sin(t * 5.1) * 0.07
    p.ball(lx, ly, lz, 0.15, WARM, 1.0, 1.0)
    p.ball(lx, ly, lz, 0.30, WARM, 0.22, 0.5)


@scene("deadline", 3.2, HOT)
def m_deadline(p: Painter, t: float) -> None:
    """A horizon line advancing toward a stationary figure who cannot move."""
    p.floor()
    p.figure(0.0, 0.0, 0.6, 1.05, BODY, arm=math.sin(t * 2.0) * 0.06)
    for phase in (0.0, 0.5):
        u = (t / 3.2 + phase) % 1.0
        z = -7.5 + u * 8.0
        a = fade_cycle(u)
        # The line itself, plus posts so its approach has parallax.
        p.box(0.0, 0.05, z, 9.0, 0.10, 0.16, HOT, 0.0, AX_Y, a, 0.85 + u * 0.9)
        for sx in (-2.4, 2.4):
            p.box(sx, 0.45, z, 0.10, 0.9, 0.10, HOT, 0.0, AX_Z, a * 0.7, 0.5)


@scene("overwhelmed", 5.0, HOT)
def m_overwhelmed(p: Painter, t: float) -> None:
    """Objects stacking on a shape until it buckles."""
    p.floor()
    u = saw(t, 5.0)
    n = int(u * 7.0)
    load = u
    buckle = ease(max(u - 0.72, 0.0) / 0.28)
    # The stack has to be empty again by the top of the loop, so the last slice
    # of the cycle takes the load back off rather than cutting to nothing.
    clear = loop_fade(u, 0.07)
    load *= clear
    buckle *= clear

    squash = 1.0 - load * 0.42 - buckle * 0.30
    p.box(0.0, 0.34 * squash, 0.0, 1.25 + load * 0.45, 0.68 * squash, 1.15, LIFE,
          buckle * 0.30, AX_Z, 1.0, 0.05)  # fmt: skip

    base = 0.68 * squash
    for i in range(min(n, 7)):
        drop = ease(min((u * 7.0 - i), 1.0))
        y = base + 0.20 + i * 0.36 + (1.0 - drop) * 3.0
        tilt = buckle * (0.14 + i * 0.10) * (1.0 if i % 2 else -1.0)
        p.box(tilt * 1.5, y, (i % 3 - 1) * 0.16, 0.95, 0.32, 0.85, COOL,
              tilt, AX_Z, drop * clear)  # fmt: skip
    if buckle > 0.3:
        p.box(0.0, 0.06, 0.0, 2.6, 0.04, 2.2, HOT, 0.0, AX_Y, (buckle - 0.3) * 1.2, 0.8)


@scene("hesitate", 2.2, WARM)
def m_hesitate(p: Painter, t: float) -> None:
    """A step that starts and retracts, over and over, at the edge of a drop."""
    u = tri(t, 2.2)
    reach = ease(u)
    # One ledge that simply stops. Nothing on the far side: a second platform
    # turns this into "crossing a gap", and the point is that there is no far
    # side to aim at.
    p.box(-2.0, -0.18, 0.0, 4.0, 0.36, 3.4, COOL)
    p.box(-2.0, 0.01, 0.0, 4.0, 0.03, 3.4, DARK, 0.0, AX_Y, 0.8)
    p.box(0.0, 0.03, 0.0, 0.07, 0.05, 3.4, HOT, 0.0, AX_Y, 1.0, 0.85)

    x = -0.95
    p.figure(x, 0.0, 0.0, 1.15, BODY, lean=reach * 0.34, arm=reach * 0.6, crouch=reach * 0.14)
    # The leading foot pushes out over nothing and comes straight back.
    p.rod(x + 0.1, 1.05, 0.12, x + 0.2 + reach * 1.25, 0.14 - reach * 0.22, 0.12,
          0.12, BODY)  # fmt: skip


@scene("procrastinate", 4.6, HOT)
def m_procrastinate(p: Painter, t: float) -> None:
    """The task keeps getting shoved further down the timeline. The deadline creeps in."""
    p.floor()
    u = saw(t, 4.6)
    shoves = math.floor(u * 3.0)
    frac = (u * 3.0) % 1.0
    push = ease(pulse(frac, 0.35, 0.35))
    slide = ease(frac)

    # The timeline has to start over, so the cycle fades through its own seam:
    # the block and the deadline are gone for an instant and then it is the
    # same morning again, which is the honest shape of the word.
    fade = loop_fade(u, 0.07)

    p.box(0.0, 0.62, 0.0, 5.6, 0.09, 0.22, COOL, 0.0, AX_Z, 1.0, 0.12)
    p.box(-2.7, 0.95, 0.0, 0.08, 0.75, 0.3, PALE, 0.0, AX_Z, 0.8, 0.25)  # "now"

    task_x = -2.1 + (shoves + slide) * 1.0
    p.box(task_x, 0.95, 0.0, 0.68, 0.58, 0.55, WARM, 0.0, AX_Z, fade, 0.4)
    # Someone is doing the shoving, and it takes visible effort each time.
    p.figure(task_x - 1.05, 0.0, 0.35, 1.0, BODY, lean=push * 0.32, arm=0.7 + push * 0.5,
             a=fade)  # fmt: skip

    # The deadline slides the other way by exactly as much as the task gains.
    dl = 2.5 - (shoves + slide) * 0.40
    p.box(dl, 1.05, 0.0, 0.11, 0.95, 0.36, HOT, 0.0, AX_Z, fade, 0.95)
    p.cone(dl, 1.72, 0.0, 0.36, 0.32, HOT, math.pi, AX_Z, fade, 0.95)
    if dl - task_x < 0.9:
        p.ball(dl, 1.05, 0.0, 0.5 * (0.9 - (dl - task_x)), HOT, 0.45 * fade, 1.0)


@scene("nostalgia", 5.0, WARM)
def m_nostalgia(p: Painter, t: float) -> None:
    """A warm room shrinking into the distance while a figure reaches back for it."""
    p.floor()
    reach = 0.55 + math.sin(t * TAU / 5.0) * 0.18
    p.figure(-1.5, 0.0, 1.2, 1.1, BODY, lean=0.16, arm=reach, facing=-1.0)
    for phase in (0.0, 0.5):
        u = (t / 5.0 + phase) % 1.0
        s = 1.0 - u * 0.72
        a = fade_cycle(u)
        # Pushed well back in z as it shrinks, so the two copies read as one
        # room going away rather than as nested squares.
        z = -0.5 - u * 7.0
        p.box(1.2, 1.05 * s, z, 1.9 * s, 2.1 * s, 0.06, WARM, 0.0, AX_Z, a * 0.9, 0.85 * s)
        p.box(1.2, 1.05 * s, z + 0.06, 2.2 * s, 2.4 * s, 0.06, COOL, 0.0, AX_Z, a * 0.55)


@scene("resent", 6.0, HOT)
def m_resent(p: Painter, t: float) -> None:
    """A seed lodged in someone, growing a little every time the other passes."""
    p.floor()
    u = saw(t, 6.0)
    passes = math.floor(u * 3.0)
    grown = 0.16 + passes * 0.13 + ease(pulse((u * 3.0) % 1.0, 0.5, 0.3)) * 0.06
    grown *= 1.0 - ease(max(u - 0.94, 0.0) / 0.06)  # let go, quietly, and restart

    p.figure(-0.9, 0.0, 0.4, 1.15, BODY)
    p.ball(-0.9, 1.42, 0.62, grown, HOT, 1.0, 0.55 + grown)

    # The other figure crosses behind, unaware, again and again. Faded at both
    # ends of each traverse so the wrap-around is never seen.
    walk = (u * 3.0) % 1.0
    px = -3.4 + walk * 6.8
    p.figure(px, 0.0, -1.8, 0.95, COOL, stride=math.sin(t * 6.0), a=loop_fade(walk, 0.14))


@scene("cope", 4.0, LIFE)
def m_cope(p: Painter, t: float) -> None:
    """Holding the stack level while the ground itself rocks."""
    rock = math.sin(t * TAU / 4.0) * 0.19
    # The ground itself is what is moving. Drawn as a slatted deck so the tilt
    # is unmistakable even in a single frame.
    for i in range(-4, 5):
        x = i * 0.62
        p.box(x, -0.15 + math.sin(rock) * x, -0.2, 0.5, 0.12, 3.4, COOL, rock, AX_Z, 1.0, 0.05)
    p.box(0.0, -0.15, -0.2, 5.6, 0.03, 3.5, HOT, rock, AX_Z, 0.35, 0.3)

    # The figure counter-leans; the load it is holding stays dead level.
    p.figure(0.0, -0.08, 0.0, 0.86, BODY, lean=-rock * 1.5, arm=-1.2, crouch=0.12)
    for i in range(3):
        p.box(0.0, 1.86 + i * 0.28, 0.0, 0.85 - i * 0.11, 0.23, 0.72, WARM,
              math.sin(t * 3.3 + i) * 0.06, AX_Z, 1.0, 0.14)  # fmt: skip


@scene("relieved", 4.2, LIFE)
def m_relieved(p: Painter, t: float) -> None:
    """The weight finally lifts and the body opens back up."""
    p.floor()
    u = saw(t, 4.2)
    lift = ease(min(max((u - 0.25) / 0.35, 0.0), 1.0))
    press = 1.0 - lift

    # Resting on the crown of the head at 1.86, clearing the stood-up head at
    # 2.42. Anything lower and it reads as a table the figure is standing under.
    slab_y = 1.98 + lift * 1.5
    p.box(0.0, slab_y, 0.0, 2.5, 0.32, 1.8, COOL, 0.0, AX_Z, 1.0 - lift * 0.6)
    for s in (-1.0, 1.0):
        p.box(s * 1.0, slab_y - 0.26, 0.0, 0.18, 0.28, 0.5, COOL,
              0.0, AX_Z, (1.0 - lift) * 0.8)  # fmt: skip
    if press > 0.2:
        p.box(0.0, slab_y - 0.20, 0.0, 1.6, 0.03, 0.9, HOT, 0.0, AX_Y, press * 0.55, 0.7)
    p.figure(0.0, 0.0, 0.0, 0.94 + lift * 0.16, BODY,
             crouch=press * 0.62, arm=-lift * 0.55, lean=press * 0.20)  # fmt: skip
    # The exhale.
    if lift > 0.4:
        for i in range(6):
            ph = ((u - 0.4) * 2.2 + i * 0.17) % 1.0
            p.ball(math.sin(i * 2.4) * 0.45 * ph, 1.75 + ph * 1.1, math.cos(i * 2.4) * 0.4 * ph,
                   0.09 * (1.0 - ph), LIFE, (1.0 - ph) * lift, 0.9)  # fmt: skip


@scene("reluctant", 3.4, WARM)
def m_reluctant(p: Painter, t: float) -> None:
    """Being pulled forward while leaning all the way back."""
    p.floor()
    drag = tri(t, 3.4)
    x = -0.9 + drag * 0.75
    p.figure(x, 0.0, 0.0, 1.1, BODY, lean=-0.52, arm=0.95, crouch=0.16)
    # The tether, taut, running off toward whatever is doing the pulling.
    p.rod(x + 0.30, 1.95, 0.0, 3.4, 1.55, 0.0, 0.05, WARM, 1.0, 0.45)
    # Heels dug in.
    for s in (-1.0, 1.0):
        p.box(x - 0.42, 0.05, s * 0.16, 0.5, 0.09, 0.2, BODY, -0.22, AX_Z)
    p.box(x - 0.62, 0.03, 0.0, 0.7, 0.03, 0.5, COOL, 0.0, AX_Y, 0.55, 0.2)


@scene("anticipate", 3.8, WARM)
def m_anticipate(p: Painter, t: float) -> None:
    """Leaning toward a door that has not opened. The light under it keeps swelling."""
    p.floor()
    swell = 0.35 + 0.65 * ease(tri(t, 3.8))
    doorway(p, 1.0, -0.5, 2.3, 1.3, 0.0)
    p.box(1.65, 1.15, -0.55, 1.28, 2.2, 0.05, DARK)
    # The only light is the strip at its foot.
    p.box(1.65, 0.055, -0.42, 1.28, 0.09, 0.18, WARM, 0.0, AX_Y, 1.0, 0.6 + swell * 1.5)
    p.box(1.65, 0.02, -0.05 - swell * 0.5, 1.28 + swell * 0.5, 0.02, 0.9 + swell,
          WARM, 0.0, AX_Y, 0.22 * swell, 0.8)  # fmt: skip
    p.figure(-0.55, 0.0, 0.3, 1.1, BODY, lean=0.20 + swell * 0.22, arm=swell * 0.30)


@scene("dread", 4.6, HOT)
def m_dread(p: Painter, t: float) -> None:
    """Something at the horizon getting bigger. The figure does not move."""
    p.floor()
    p.figure(0.0, 0.0, 1.9, 0.85, BODY)
    for phase in (0.0, 0.5):
        u = (t / 4.6 + phase) % 1.0
        s = 0.5 + u * u * 4.0
        z = -8.0 + u * 7.4
        a = fade_cycle(u)
        # Rim light alone is too thin to carry a shape this dark, so the mass
        # gets a faint self-lit body as well. Three offset lumps, because a
        # single box reads as a building and a building is not frightening.
        p.box(0.0, s * 0.80, z, s * 2.0, s * 1.6, s * 1.0, MASS, 0.0, AX_Z, a, 0.10, 1.0)
        p.box(-s * 0.62, s * 0.50, z + 0.35, s * 1.1, s * 1.0, s * 0.85, MASS,
              0.24, AX_Z, a, 0.08, 1.0)  # fmt: skip
        p.box(s * 0.66, s * 0.38, z + 0.25, s * 0.9, s * 0.8, s * 0.75, MASS,
              -0.32, AX_Z, a, 0.08, 1.0)  # fmt: skip


@scene("envy", 4.8, LIFE)
def m_envy(p: Painter, t: float) -> None:
    """Watching something orbit somebody else, reaching, never taking."""
    p.floor()
    reach = 0.75 + math.sin(t * TAU / 2.4) * 0.20
    p.figure(-1.7, 0.0, 0.4, 1.1, BODY, lean=0.24, arm=reach)
    p.figure(1.5, 0.0, 0.0, 1.1, COOL)
    a = t * 1.5
    ox = 1.5 + math.cos(a) * 0.95
    oz = math.sin(a) * 0.75
    p.ball(ox, 1.75 + math.sin(a * 2.0) * 0.18, oz, 0.22, WARM, 1.0, 1.0)
    p.ball(ox, 1.75 + math.sin(a * 2.0) * 0.18, oz, 0.44, WARM, 0.16, 0.5)


@scene("spite", 4.4, HOT)
def m_spite(p: Painter, t: float) -> None:
    """Knocking over your own tower so the other one cannot have it."""
    p.floor()
    u = saw(t, 4.4)
    swing = ease(min(max((u - 0.42) / 0.14, 0.0), 1.0))
    fall = ease(min(max((u - 0.52) / 0.30, 0.0), 1.0))
    rebuild = ease(min(max((u - 0.88) / 0.12, 0.0), 1.0))

    p.figure(-1.5, 0.0, 0.2, 1.1, BODY, arm=swing * 1.5 - rebuild * 0.8, lean=swing * 0.2)
    # The other one, walking in, arriving just too late.
    p.figure(3.4 - u * 2.3, 0.0, -0.9, 1.0, COOL, stride=math.sin(t * 5.0))

    for i in range(4):
        settle = max(fall - i * 0.12, 0.0)
        y = 0.24 + i * 0.46 - settle * (0.24 + i * 0.46 - 0.2)
        x = 0.1 + settle * (0.6 + i * 0.62)
        p.box(x, max(y, 0.2) * (1.0 - rebuild) + (0.24 + i * 0.46) * rebuild,
              settle * 0.25, 0.8, 0.42, 0.72, WARM,
              settle * (1.2 + i * 0.25) * (1.0 - rebuild), AX_Z, 1.0, 0.12)  # fmt: skip


@scene("vulnerable", 4.6, LIFE)
def m_vulnerable(p: Painter, t: float) -> None:
    """The shell opens, the soft light is exposed, and things are falling."""
    p.floor()
    op = ease(tri(t, 4.6))
    # A crate whose four walls fall outward. Cones read as wings; flat panels
    # hinged at the base read as armour coming off.
    p.figure(0.0, 0.0, 0.0, 1.0, BODY, crouch=0.24 - op * 0.16, emis=op * 0.30, arm=-op * 0.3)
    # A soft aura, not a solid ball: the whole point is that you can see the
    # person once the shell is off.
    p.ball(0.0, 1.35, 0.0, 0.55 + op * 0.65, LIFE, 0.07 + op * 0.09, 0.5 + op * 1.2)
    for i in range(4):
        a = i * 1.5708
        cx, cz = math.cos(a), math.sin(a)
        drop = op * 1.5708
        p.box(cx * (0.85 + op * 0.55), 1.0 * math.cos(drop) + 0.05, cz * (0.85 + op * 0.55),
              0.10 if i % 2 == 0 else 1.75, 2.0, 1.75 if i % 2 == 0 else 0.10,
              COOL, -drop * (1.0 if i % 2 == 0 else -1.0),
              AX_Z if i % 2 == 0 else AX_X, 0.95)  # fmt: skip
    for i in range(7):
        ph = ((t * 0.55) + i * 0.143) % 1.0
        p.shard(math.sin(i * 5.1) * 2.3, 3.2 - ph * 3.3, math.cos(i * 3.7) * 1.1, 0.20, HOT,
                ph * 6.0 + i, AX_Z, 0.35 + 0.5 * op)  # fmt: skip


@scene("resilient", 5.2, LIFE)
def m_resilient(p: Painter, t: float) -> None:
    """Knocked apart, and it puts itself back together — not quite the same way."""
    p.floor()
    u = saw(t, 5.2)
    apart = ease(min(u / 0.22, 1.0)) * (1.0 - ease(min(max((u - 0.42) / 0.38, 0.0), 1.0)))
    for i in range(6):
        ang = i * 1.047 + u * 0.6
        # Where each block ends up is offset a little each cycle: rebuilt, not restored.
        home_x = math.sin(i * 2.3) * 0.20
        y = 0.26 + i * 0.44
        p.box(
            home_x + math.cos(ang) * 2.4 * apart,
            y + apart * (1.4 - i * 0.12),
            math.sin(ang) * 1.1 * apart,
            0.82, 0.40, 0.72, LIFE if i % 2 else BODY,
            apart * (2.2 + i * 0.3), AX_Z, 1.0, 0.10,
        )  # fmt: skip


@scene("overthink", 4.0, WARM)
def m_overthink(p: Painter, t: float) -> None:
    """Every path out of the decision leads straight back into it."""
    p.ball(0.0, 1.55, 0.0, 0.34, WARM, 1.0, 0.9 + math.sin(t * 4.0) * 0.2)
    for i in range(5):
        a = i * 1.2566 + t * 0.35
        nx = math.cos(a) * 1.85
        ny = 1.55 + math.sin(a) * 1.05
        nz = math.sin(a * 1.7) * 0.7
        p.ball(nx, ny, nz, 0.15, COOL, 1.0, 0.25)
        p.rod(0.0, 1.55, 0.0, nx, ny, nz, 0.035, COOL, 0.7, 0.15)
        # A pulse runs out along the branch and comes straight back in.
        ph = ((t * 0.85) + i * 0.2) % 1.0
        k = tri(ph, 1.0)
        p.ball(nx * k, 1.55 + (ny - 1.55) * k, nz * k, 0.085, WARM, 1.0, 1.0)


@scene("burnout", 5.0, HOT)
def m_burnout(p: Painter, t: float) -> None:
    """Still spinning. Nothing left in it."""
    u = saw(t, 5.0)
    life = 1.0 - ease(min(u / 0.85, 1.0)) * 0.92
    fade = loop_fade(u, 0.06)
    p.ball(0.0, 1.35, 0.0, 0.60, WARM if life > 0.4 else HOT, fade, 0.10 + life * 1.6)
    p.ball(0.0, 1.35, 0.0, 1.0, WARM, 0.09 * life * fade, 0.6 * life)
    spin = t * (3.4 * life + 0.25)
    for i in range(6):
        a = spin + i * 1.047
        r = 1.55 * (0.4 + life * 0.6)
        # The satellites lose altitude as the core loses heat, and end up on
        # the floor as dead cinders.
        drop = (1.0 - life) * (0.9 + i * 0.16)
        y = max(1.35 - drop, 0.12)
        p.ball(math.cos(a) * r, y, math.sin(a) * r * 0.55, 0.17,
               WARM if y > 0.2 else DARK, fade, life * 0.9)  # fmt: skip
    p.floor(0.0, 46.0, HOT, life * 0.25)


@scene("homesick", 4.4, WARM)
def m_homesick(p: Painter, t: float) -> None:
    """Inside a cold grid. The warm thing is small, and it is very far away."""
    p.floor(0.0, 46.0, COOL)
    # Not a cage -- a cage means trapped. Just cold empty distance, and one warm
    # thing at the far end of it that keeps breathing.
    breathe = 0.9 + math.sin(t * TAU / 4.4) * 0.13
    reach = 0.35 + math.sin(t * TAU / 4.4) * 0.20
    p.figure(-1.5, 0.0, 1.9, 1.15, BODY, lean=0.14, arm=reach)

    for i in range(7):
        z = -1.0 - i * 1.3
        s = 1.0 - i * 0.10
        p.box(0.0, 0.02, z, 7.0 * s, 0.03, 0.05, COOL, 0.0, AX_Y, 0.30 * s, 0.10)

    p.box(1.7, 0.85, -6.2, 1.55, 1.3, 1.2, WARM, 0.0, AX_Z, 1.0, 0.5 * breathe)
    p.cone(1.7, 1.92, -6.2, 2.15, 1.0, WARM, 0.0, AX_Z, 1.0, 0.42 * breathe)
    p.box(1.7, 0.62, -5.58, 0.42, 0.85, 0.04, PALE, 0.0, AX_Z, 1.0, 1.3 * breathe)
    p.ball(1.7, 1.1, -6.2, 2.1, WARM, 0.028, 0.8 * breathe)


@scene("awkward", 3.6, WARM)
def m_awkward(p: Painter, t: float) -> None:
    """Two people trying to get past each other, choosing the same side every time."""
    p.floor()
    u = saw(t, 3.6)
    close = ease(min(u / 0.28, 1.0)) * (1.0 - ease(max(u - 0.72, 0.0) / 0.28))
    # Both sidestep the same way, twice, and stay blocked. The dodge has to be
    # big and exactly synchronised or the joke does not land.
    dodge = math.sin(u * TAU * 2.0) * close * 1.15
    gap = 2.5 - close * 1.55

    p.figure(-gap, 0.0, dodge, 1.1, BODY, stride=math.sin(t * 6.0) * close * 0.6)
    p.figure(gap, 0.0, dodge, 1.1, COOL, stride=math.sin(t * 6.0) * close * 0.6, facing=-1.0)
    # A dotted line between them makes the shared axis, and the deadlock, legible.
    for i in range(5):
        p.ball(-0.6 + i * 0.3, 1.35, dodge, 0.05, HOT, close * 0.55, 0.8)
    if close > 0.8:
        k = (close - 0.8) * 5.0
        p.box(0.0, 2.25, dodge, 0.09, 0.34, 0.09, HOT, 0.0, AX_Z, k, 1.0)
        p.ball(0.0, 1.98, dodge, 0.07, HOT, k, 1.0)


@scene("grateful", 4.4, LIFE)
def m_grateful(p: Painter, t: float) -> None:
    """Small lights arrive from outside and settle in. The person gets brighter."""
    p.floor()
    warmth = 0.0
    for i in range(7):
        # Two whole arrivals per loop. A fractional count leaves each light
        # part-way through its flight when the loop wraps, and they all jump.
        ph = ((t / 4.4) * 2.0 + i * 0.142) % 1.0
        k = ease(ph)
        a = i * 0.897
        x = math.cos(a) * 4.2 * (1.0 - k)
        y = 1.6 + math.sin(a) * 2.4 * (1.0 - k)
        z = math.sin(a * 1.3) * 2.0 * (1.0 - k)
        p.ball(x, y, z, 0.13 * (1.0 - k * 0.55), WARM, 1.0 - k * 0.75, 1.0)
        warmth += (1.0 if ph > 0.82 else 0.0) * 0.14
    glow = 0.18 + warmth + 0.10 * math.sin(t * 2.2)
    # The person has to stay readable, so the warmth goes into them rather than
    # into a halo that swallows them.
    p.figure(0.0, 0.0, 0.0, 1.15, BODY, arm=-0.4, emis=0.10 + glow * 0.55)
    p.ball(0.0, 1.5, 0.0, 0.42, LIFE, 0.10 + glow * 0.12, 0.9)


@scene("petty", 4.2, HOT)
def m_petty(p: Painter, t: float) -> None:
    """An enormous balance, tipped completely by one grain."""
    u = saw(t, 4.2)
    land = ease(min(max((u - 0.35) / 0.10, 0.0), 1.0))
    tip = ease(min(max((u - 0.42) / 0.28, 0.0), 1.0)) * (1.0 - ease(max(u - 0.88, 0.0) / 0.12))
    ang = -0.34 * tip

    p.cone(0.0, 0.55, 0.0, 0.7, 1.1, COOL)
    p.box(0.0, 1.15, 0.0, 5.2, 0.13, 0.5, COOL, ang, AX_Z)
    lx, ly = -2.3, 1.15 - math.sin(ang) * 2.3
    rx, ry = 2.3, 1.15 + math.sin(ang) * 2.3
    p.box(lx, ly + 0.75, 0.0, 1.7, 1.4, 1.4, BODY, ang, AX_Z, 1.0, 0.06)
    p.box(rx, ry + 0.09, 0.0, 1.2, 0.09, 1.0, COOL, ang, AX_Z)
    # The grain. It is nothing. It decides everything.
    gy = ry + 0.20 + (1.0 - land) * 2.6
    p.ball(rx, gy, 0.0, 0.09, HOT, 1.0, 1.0)


@scene("commitment", 5.0, LIFE)
def m_commitment(p: Painter, t: float) -> None:
    """Crossing, while the planks behind you drop away one at a time."""
    u = saw(t, 5.0)
    walk = u * 6.0
    for i in range(9):
        # Planks scroll toward the walker so the loop never has to snap.
        px = -3.4 + ((i - walk) % 9.0) * 0.86
        gone = ease(min(max((-0.6 - px) / 1.2, 0.0), 1.0))
        p.box(px, -0.1 - gone * 3.2, 0.0, 0.66, 0.13, 2.0, COOL,
              gone * 1.5, AX_X, 1.0 - gone * 0.75)  # fmt: skip
    p.figure(-0.2, 0.0, 0.0, 1.1, BODY, stride=math.sin(walk * TAU), arm=math.sin(walk * TAU))
    p.box(-3.6, -0.12, 0.0, 1.4, 0.16, 2.2, DARK)


@scene("distracted", 4.0, WARM)
def m_distracted(p: Painter, t: float) -> None:
    """Heading for the goal, yanked aside every single time."""
    p.floor()
    upright_ring(p, 2.45, 1.5, 0.0, 1.35, LIFE, 1.0, 0.35)
    u = saw(t, 4.0)
    lure = math.floor(u * 3.0)
    frac = (u * 3.0) % 1.0
    side = 1.0 if lure % 2 < 1 else -1.0
    pull = ease(pulse(frac, 0.55, 0.45))
    # Always moving toward the goal; always pulled off it before arriving, and
    # never getting more than a third of the way there.
    x = -2.3 + (lure + frac) * 0.5
    y = 1.5 + side * pull * 1.05
    z = -side * pull * 0.5
    # Losing the thread entirely and starting over from the left is the most
    # honest ending this scene has.
    fade = loop_fade(u, 0.07)
    p.ball(x, y, z, 0.19, WARM, fade, 1.0)
    p.ball(x, y, z, 0.38, WARM, 0.16 * fade, 0.5)
    # A dotted trail of where it has been, wandering rather than advancing.
    for i in range(4):
        k = (i + 1) * 0.22
        p.ball(x - k * 0.5, y - side * pull * k * 1.4, z, 0.06, WARM,
               (0.35 - i * 0.07) * fade, 0.6)  # fmt: skip
    lx = x + side * 1.25
    p.ball(lx, 1.5 + side * 1.75, -side * 0.8, 0.15 + 0.05 * math.sin(t * 14.0),
           HOT, (0.45 + 0.55 * pull) * fade, 1.0)  # fmt: skip


@scene("accountable", 4.2, LIFE)
def m_accountable(p: Painter, t: float) -> None:
    """The light is on you and you do not step out of it."""
    p.floor()
    drift = math.sin(t * TAU / 4.2) * 0.42
    p.cone(0.0, 2.05, 0.0, 2.9, 3.4, PALE, math.pi, AX_Z, 0.09, 0.35)
    # Low emissive on the floor rings: past about 0.25 the bloom fills them in
    # and a ring on the ground turns into a glowing puddle.
    p.ring(0.0, 0.02, 0.0, 2.5, LIFE, 0.0, AX_Y, 1.0, 0.16)
    p.figure(drift, 0.0, 0.0, 1.1, BODY, lean=-drift * 0.25, stride=math.sin(t * 1.6) * 0.3)
    # The mark stays under them wherever they shift. So does the light.
    p.ring(drift, 0.05, 0.0, 0.85, WARM, 0.0, AX_Y, 0.9, 0.22)


@scene("tempting", 3.4, WARM)
def m_tempting(p: Painter, t: float) -> None:
    """The line is right there. So is the thing on the other side of it."""
    p.floor()
    lean = ease(tri(t, 3.4))
    p.box(0.55, 0.03, 0.0, 0.09, 0.03, 4.2, HOT, 0.0, AX_Y, 1.0, 0.8)
    p.figure(-0.7, 0.0, 0.0, 1.1, BODY, lean=lean * 0.34, arm=lean * 0.85)
    # A toe over the line, then back.
    p.rod(-0.55, 0.9, 0.1, -0.4 + lean * 1.25, 0.08, 0.1, 0.11, BODY)
    bob = math.sin(t * 2.3) * 0.12
    p.ball(1.85, 1.35 + bob, 0.0, 0.30, WARM, 1.0, 0.9 + lean * 0.5)
    p.ball(1.85, 1.35 + bob, 0.0, 0.62, WARM, 0.13, 0.5)


@scene("exhausted", 5.4, COOL)
def m_exhausted(p: Painter, t: float) -> None:
    """The steps get shorter and lower until there is barely a step left."""
    p.floor()
    u = saw(t, 5.4)
    energy = 1.0 - ease(min(u / 0.88, 1.0)) * 0.88
    cadence = math.sin(t * (7.0 * energy + 0.8))
    p.figure(math.sin(u * 3.0) * 0.1, 0.0, 0.0, 1.15, BODY,
             stride=cadence * energy, arm=cadence * energy * 0.5,
             crouch=(1.0 - energy) * 0.68, lean=(1.0 - energy) * 0.40)  # fmt: skip
    # What is left in the tank, draining. Kept low and beside the figure so both
    # are in frame at once.
    p.box(-2.3, 1.05, 0.0, 0.34, 1.9, 0.34, COOL, 0.0, AX_Z, 0.45)
    p.box(-2.3, 0.16 + energy * 0.88, 0.0, 0.24, 1.76 * energy, 0.24, WARM,
          0.0, AX_Z, 1.0, 0.55 * energy)  # fmt: skip
    # Footprints that get shorter and fainter behind them.
    for i in range(5):
        e = 1.0 - (i + 1) * 0.16
        p.box(-0.45 - i * 0.42 * energy, 0.02, 0.35, 0.24, 0.02, 0.14, COOL,
              0.0, AX_Y, 0.35 * e, 0.1)  # fmt: skip


@scene("skeptical", 4.0, COOL)
def m_skeptical(p: Painter, t: float) -> None:
    """Held at arm's length, turned over, and not accepted."""
    p.floor()
    p.figure(1.5, 0.0, 0.0, 1.05, COOL, arm=-0.7, facing=-1.0)
    hold = 0.85 + math.sin(t * TAU / 4.0) * 0.12
    p.figure(-1.3, 0.0, 0.2, 1.1, BODY, lean=-0.16, arm=hold)
    ox = -1.3 + hold * 1.15
    p.box(ox, 1.45, 0.25, 0.45, 0.45, 0.45, WARM, t * 1.3, AX_Y, 1.0, 0.3)
    p.box(ox, 1.45, 0.25, 0.46, 0.46, 0.46, PALE, t * 1.3, AX_Y, 0.25, 0.0, 1.0)
    p.rod(-1.3, 1.85, 0.25, ox - 0.2, 1.5, 0.25, 0.09, BODY)


@scene("impatient", 4.6, HOT)
def m_impatient(p: Painter, t: float) -> None:
    """The bar is barely moving. Then it starts over."""
    p.floor()
    u = saw(t, 4.6)
    prog = ease(min(u / 0.82, 1.0)) * 0.88
    reset = ease(max(u - 0.9, 0.0) / 0.10)
    prog *= 1.0 - reset

    upright_ring(p, 0.55, 1.7, 0.0, 2.1, COOL, 0.45)
    steps = 16
    for i in range(steps):
        if i / steps > prog:
            continue
        a = -1.5708 + (i / steps) * TAU
        p.ball(0.55 + math.cos(a) * 0.93, 1.7 + math.sin(a) * 0.93, 0.06, 0.11, WARM, 1.0, 0.9)
    if reset > 0.05:
        upright_ring(p, 0.55, 1.7, 0.0, 2.1 + reset * 0.8, HOT, (1.0 - reset) * 0.9, 0.8)
    tap = abs(math.sin(t * 9.0))
    p.figure(-2.1, 0.0, 0.4, 1.1, BODY, arm=0.3 + tap * 0.5, stride=tap * 0.18)


@scene("stubborn", 4.0, COOL)
def m_stubborn(p: Painter, t: float) -> None:
    """Pushed hard from both sides. Does not move a millimetre."""
    p.floor()
    shove = abs(math.sin(t * TAU / 2.0))
    p.figure(0.0, 0.0, 0.0, 1.2, BODY, crouch=0.08)
    for s in (-1.0, 1.0):
        px = s * (2.3 - shove * 0.85)
        p.figure(px, 0.0, -0.5 * s, 1.0, COOL, lean=s * 0.4 * shove,
                 arm=0.9, facing=-s)  # fmt: skip
        p.box(s * 0.75, 1.5, 0.0, 0.5 * shove, 0.09, 0.09, HOT, 0.0, AX_Z, shove, 0.7)
    p.box(0.0, 0.03, 0.0, 1.1, 0.03, 1.1, LIFE, 0.0, AX_Y, 0.7, 0.35)


@scene("assume", 4.0, WARM)
def m_assume(p: Painter, t: float) -> None:
    """Filling in the shape you cannot see with the one you expect."""
    p.floor()
    u = saw(t, 4.0)
    guess = ease(pulse(u, 0.5, 0.42))
    # The real thing is round and bigger than the screen, so it shows over the
    # top and around the sides. The projected square never matches it.
    # Sized and placed so it clears the screen by a sliver on every side: enough
    # to know it is round and bigger, not enough to actually see it.
    p.ball(0.0, 1.35, -2.6, 1.62, LIFE, 1.0, 0.20)
    p.box(0.0, 1.35, -0.6, 2.5, 2.6, 0.08, COOL, 0.0, AX_Z, 1.0)
    p.box(0.0, 1.35, -0.5, 1.15, 1.15, 0.05, WARM, guess * 0.8, AX_Z, guess, guess * 0.85)
    p.figure(-2.1, 0.0, 1.7, 1.05, BODY, lean=0.14, arm=guess * 0.55)


@scene("bittersweet", 5.0, WARM)
def m_bittersweet(p: Painter, t: float) -> None:
    """The same moment, warm and cold at once, and it will not resolve."""
    p.floor()
    swing = math.sin(t * TAU / 5.0)
    p.figure(0.0, 0.0, 0.0, 1.1, BODY, lean=swing * 0.10)
    for s, col, e in ((-1.0, WARM, 0.9), (1.0, COOL, 0.2)):
        k = 0.5 + 0.5 * swing * s
        p.ball(s * 1.55, 1.75 + s * swing * 0.28, 0.0, 0.34 + k * 0.14, col,
               0.35 + k * 0.65, e * k)  # fmt: skip
        p.rod(0.0, 1.55, 0.0, s * 1.55, 1.75 + s * swing * 0.28, 0.0, 0.03, col, 0.5, 0.2)


@scene("belong", 4.6, LIFE)
def m_belong(p: Painter, t: float) -> None:
    """The gap in the circle is exactly your shape, and it closes behind you."""
    p.floor()
    u = saw(t, 4.6)
    step_in = ease(min(max((u - 0.15) / 0.4, 0.0), 1.0))
    n = 7
    for i in range(n):
        a = i / n * TAU + 0.35
        r = 1.75
        # The ring shuffles closed once the newcomer is inside.
        a += step_in * 0.16 * (1.0 if i > n // 2 else -1.0)
        p.figure(math.cos(a) * r, 0.0, math.sin(a) * r * 0.55 - 0.4, 0.85, COOL,
                 emis=step_in * 0.12)  # fmt: skip
    x = 0.0 + (1.0 - step_in) * 3.4
    p.figure(x, 0.0, 1.5 - step_in * 1.5, 0.95, BODY, stride=math.sin(t * 5.0) * (1.0 - step_in),
             emis=step_in * 0.35)  # fmt: skip
    p.ring(0.0, 0.03, -0.4, 4.0, LIFE, 0.0, AX_Y, 0.35 + step_in * 0.45, 0.12)
