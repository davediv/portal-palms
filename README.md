# Portal Palms

Hold up both hands. Pull them apart. You tear a hole in the screen, and inside
it a small scene shows you what an abstract English word actually means.

Built for Gen Z students at an English immersion camp in Kampung Inggris, Pare.
It targets the words that a bilingual word list cannot fix — the ones where an
Indonesian gloss sits in your notes and never becomes a thing you can feel.
*Regret* is a figure walking backward toward a door that keeps receding.
*Overwhelmed* is objects stacking on a shape until it buckles. *Hesitate* is a
step that starts and pulls back, and starts, and pulls back.

There are 34 of them. They are the product; everything else is scaffolding.

---

## Running it

Needs Python 3.11 or 3.12, a webcam, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python main.py
```

First launch shows a camera explainer. Press <kbd>space</kbd> and the camera
opens — on macOS that is the moment the OS permission sheet appears, which is
exactly why the explainer is on screen first.

No camera handy, or it is busy on a video call?

```bash
uv run python main.py --demo        # synthetic hands, whole loop, no device
uv run python main.py --check       # verify assets and the word deck, then exit
```

### Window shape

**The composition follows your display.** On a landscape monitor the app
composes 1920x1080 and a maximised window fills the screen; on a portrait one it
composes 1080x1920.

That default exists because a 9:16 composition on a 16:9 monitor can only ever
pillarbox — maximising makes it taller and never wider, and you get a
phone-shaped strip down the middle of your desktop.

The vertical format is still one flag away, and it is the one to use when the
clip shape matters:

```bash
uv run python main.py --portrait               # 1080x1920, vertical clips
uv run python main.py --landscape              # force 1920x1080
uv run python main.py --size 1440x1440         # or any shape you like
uv run python main.py --fullscreen             # start fullscreen
```

<kbd>f</kbd> toggles fullscreen at any time. The whole UI reflows rather than
stretching: type scales with the frame, the content column narrows to a readable
measure on wide screens, the word card measures its own type and sizes its panel
to fit, and the camera plate letterboxes on whichever axis binds. Recording
follows the composition, so `--portrait` is what gives you vertical clips.

### Gestures

| | |
|---|---|
| **Both hands up, close together** | a thin bright seam between your palms |
| **Pull apart** | the seam tears open into a portal; radius follows your wrists exactly |
| **Hold it open ~1.5s** | the portal locks and the scene plays |
| **Roll both hands** | the ring rolls with them |
| **Bring your hands back together** | it snaps shut with an inward implosion |
| **Flick one hand sideways** | next word |

One hand leaving frame collapses the portal smoothly. If a scene is already
playing it keeps playing on one hand — losing your word because you scratched
your nose is worse than holding the portal open a beat too long.

### Keys

| key | |
|---|---|
| <kbd>space</kbd> / <kbd>→</kbd> | next word |
| <kbd>←</kbd> | previous word |
| <kbd>f</kbd> | fullscreen |
| <kbd>d</kbd> | performance HUD |
| <kbd>s</kbd> | settings drawer |
| <kbd>r</kbd> | record a clip (3-2-1, 30s max) |
| <kbd>m</kbd> | mirror on/off |
| <kbd>l</kbd> | low-power mode |
| <kbd>esc</kbd> | close the drawer, or quit |

Clips land in `~/Movies/PortalPalms`. Progress lives in
`~/Library/Application Support/PortalPalms/progress.json` (per-platform
equivalents elsewhere), written atomically so a mid-session kill cannot corrupt it.

---

## Swapping in your own word list

Copy `portal_palms/content/words.json`, edit it, and point the app at it:

```bash
uv run python main.py --words my_deck.json
```

Each entry is four fields:

```json
{
  "word": "reluctant",
  "pos": "adjective",
  "example": "He was reluctant to go, but he came anyway.",
  "scene": "reluctant"
}
```

`scene` names one of the animations in `portal_palms/content/metaphors.py`. A
word pointing at a scene that does not exist is **skipped with a warning at
startup**, not a crash — you can add vocabulary before anyone has animated it,
and the rest of the deck still runs. `uv run python main.py --check` lists any
warnings plus any metaphors no word is currently using.

You can reuse a scene for a related word (`dread` and `apprehensive` share a
picture happily), or write a new one. A metaphor is a function of about twenty
lines that places primitives; the rules for making one that reads are in the
module docstring at the top of `metaphors.py`. To see what you made:

```bash
uv run python tools/contact_sheet.py --out sheet.png --only reluctant
```

That renders each scene as a filmstrip **with the label hidden**, which is the
only honest way to check the thing the whole app depends on: can someone guess
the word before they are told it?

---

## Performance architecture

The frame budget is 16.6ms and it outranks everything else here. On an M1 at
1080x1920 with the portal open, a scene playing and 4096 particles alive,
GPU-synced each frame so the number is honest:

```
mean 4.5ms   p95 7.0ms   p99 7.6ms   max 11.6ms   over budget 0 / 1140 frames
13 draw calls    RSS flat at 153MB after 20s
```

Cold start, measured on the packaged build:

```
window with content 135ms  ->  shaders warm 208ms  ->  tracking live 1.7-2.8s
```

The window is up and readable in a seventh of a second because nothing heavier
than argparse is imported at module scope, and every expensive init step draws
an explainer frame between the ones before and after it. MediaPipe imports on a
side thread behind that screen.

Five decisions do most of that work.

**Detection is decoupled from rendering.** Capture and MediaPipe inference run
on a worker thread at camera rate; the render loop owns the main thread and runs
at display rate. Inference costs 27ms of a 33ms budget on CPU — it would eat
every frame if it were inline. Both `VideoCapture.read` and MediaPipe release
the GIL, so a thread genuinely overlaps with rendering, and a thread rather than
a process avoids copying a frame through a pipe 30 times a second.

Results cross on a **triple buffer**: three preallocated slots rotate between
writer and reader, both critical sections are two integer assignments, and the
reader touches its slot with no lock held. Between detections the render thread
dead-reckons landmarks forward from last-known velocity, clamped, so 30Hz
tracking still looks like 60Hz motion.

**Nothing allocates in the loop.** Every vector, matrix and buffer is sized at
init; `gc.freeze()` then `gc.disable()` after startup means no collection can
land inside a frame. The HUD formats numbers into a reused character buffer
rather than f-strings. `tests/test_no_alloc.py` asserts this rather than
claiming it: it runs the gesture machine, the One-Euro filter and all 34 scenes
for 1500 frames and requires the live tracked-object count not to move.

**Particles live on the GPU**, moved by transform feedback between paired VBOs
and drawn as one instanced call. Transform feedback rather than compute shaders
because macOS caps OpenGL at 4.1: no compute, no SSBOs, anywhere in this
codebase.

**Geometry is shared.** All 34 scenes draw from one vertex buffer of reused
primitives with one instance buffer per primitive kind. Switching words is a
buffer write. No geometry is built at runtime, and no framebuffer is created
after startup — which is why opening and closing the portal a hundred times
cannot degrade anything.

The whole app composes into one offscreen framebuffer — 1080x1920 unless
`--size` says otherwise — that is then blitted letterboxed into the window, so
recording is a readback of the frame we already drew rather than a second render
pass. Composition size is the one global set before startup; every layout number
downstream is either normalised or derived from it, so reshaping the frame
reflows the app rather than stretching it. That readback goes into a GL
buffer, asynchronously, and is copied out one frame later, so the loop never
waits on the GPU.

Low-power mode halves particles, drops bloom to eighth resolution and caps at
30fps. It trips itself when the rolling average frame time exceeds 20ms for two
seconds, and tells you it did.

### Things that turned out to matter

A few of these were only findable by measuring, and are worth knowing if you
touch this code:

- **The One-Euro `beta` constant is unit-dependent.** The canonical 0.007 comes
  from a paper working in pixels. MediaPipe hands you [0,1] coordinates, which
  makes every speed ~640x smaller and the adaptive term inert — the filter
  degenerates to a flat 1Hz low-pass and a fast hand lags by about 100px.
  `OE_SPEED_SCALE` puts the derivative back into pixels.
- **MediaPipe's normalised space is anisotropic.** x is divided by width and y
  by height, so on a 4:3 sensor an upright hand measures 33% longer than the
  same hand held sideways. Every distance in `gestures.py` scales y first.
- **The GPU delegate is not merely unavailable on macOS, it is a trap.** It
  constructs without complaint and then aborts the process inside the first
  inference, in native code, where there is no exception to catch. Only Linux
  is allowed to attempt it.
- **A grid floor aliases to a solid fill** at grazing angles, because the cell
  shrinks below a pixel. It has to fade out before that happens.
- **MediaPipe drags matplotlib in, and in a bundle that costs 12 seconds every
  launch.** `mediapipe.tasks.python.vision` eagerly imports drawing helpers that
  import `matplotlib.pyplot`; frozen, matplotlib then rebuilds its font cache on
  every single start, because the bundled `mpl-data` never matches the user's
  cache. Invisible from source — a developer's cache is already warm — and fatal
  on the machines this actually ships to. `portal_palms/mediapipe_slim.py`
  installs a meta-path finder that hands back inert modules for matplotlib and
  sounddevice, neither of which this app can reach. 12.2s to 1.7s.
- **`gc.get_count()` cannot detect an allocation leak.** CPython's freelists
  recycle small tuples and lists without touching it: 100k tuple churns move it
  by 1, and a thousand *retained* tuples move it by 0. `len(gc.get_objects())`
  is the measure that works, and it is what the no-alloc tests use.

---

## Layout

```
main.py                     entry point; imports nothing heavy at module scope
portal_palms/
  config.py                 every tunable, in one place
  tracking.py               camera thread, MediaPipe, mirror, One-Euro, triple buffer
  gestures.py               hysteresis state machines; the portal and swipe recognisers
  synthetic.py              fake hands for --demo and for headless verification
  app.py                    the loop and the state machine
  render/
    shaders.py              all GLSL, targeting 330 core
    gl.py                   window, context, render targets, letterbox blit
    portal.py               the rim shader and the particle system
    primitives.py           shared geometry pool and the scene painter
    scene.py                the interior framebuffer and its camera
    text.py                 SDF text and 2D shapes, two instanced draws
  content/
    metaphors.py            the 34 animations
    words.json              the deck
  ui/
    screens.py              every screen state
    hud.py                  frame metrics, the debug overlay, the low-power governor
    recorder.py             PBO readback into ffmpeg
    progress.py             streak and history, written atomically
assets/models/              hand_landmarker.task, shipped locally, never downloaded
assets/fonts/               Barlow / Barlow Condensed / JetBrains Mono + baked atlas
tools/                      offline atlas baker, preview renderer, contact sheet
```

### Dependencies

Five at runtime: `mediapipe`, `numpy`, `opencv-python`, `moderngl`, `glfw`.

No Qt, no tkinter, no pygame, no scene graph, no asyncio in the render path. A
widget toolkit owns the main thread and repaints on its own schedule, and you
cannot hold a 16.6ms budget inside someone else's event loop.

Pillow bakes the font atlas but is a dev dependency only — the atlas is
pre-baked into `assets/fonts/` and shipped, which keeps Pillow out of the
runtime list entirely and takes glyph rasterisation off the cold-start path.

Recording needs an ffmpeg binary. `uv sync --extra recording` bundles one via
`imageio-ffmpeg`; failing that the app uses whatever `ffmpeg` is on PATH, and if
neither exists the `r` key is simply inert.

---

## Development

```bash
uv run pytest              # 318 tests
uv run mypy                # strict, clean
uv run ruff check .        # clean
uv run ruff format --check .
```

Rendering can be verified without a camera, a window, or a person:

```bash
uv run python main.py --offscreen 600 --out shots --stride 60   # the whole app
uv run python main.py --offscreen 300 --camera --out shots      # with a real device
uv run python tools/preview.py --strip --out open.png           # the tearing sequence
uv run python tools/contact_sheet.py --out sheet.png            # all 34 metaphors
uv run python tools/bake_font_atlas.py                          # re-bake the atlas
```

`--offscreen` drives the real update and render path and dumps PNGs, so what you
are looking at is the shipped code rather than a copy of it.

### Packaging

```bash
uv run pyinstaller portal-palms.spec
```

Produces `dist/portal-palms/`, 314MB. The spec drops unused MediaPipe task
graphs and a list of transitively-imported libraries the app never reaches.

That list is load-bearing and easy to get wrong — a bad guess builds cleanly,
launches, opens the camera, and only then dies with `ModuleNotFoundError`.
Before touching it:

```bash
uv run python tools/verify_excludes.py
```

which builds a real landmarker and reports any excluded module that ended up in
`sys.modules` anyway.

---

## Licences

App code MIT. Barlow, Barlow Condensed and JetBrains Mono are SIL OFL 1.1 —
licence text is in `assets/fonts/`. `hand_landmarker.task` is Google's, Apache
2.0, shipped locally on purpose: this runs on camp wifi and must never reach for
the network at runtime.
