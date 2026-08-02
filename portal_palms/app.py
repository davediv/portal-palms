"""The application: one plain while loop that owns the main thread.

No asyncio, no widget toolkit, no scheduler. The loop blocks on vsync at buffer
swap, which is the lowest-jitter pacing available and is not something a
scheduler in front of it can improve on.

After init the loop allocates nothing: every vector, buffer and matrix was sized
at startup, the cyclic collector is off and the surviving heap is frozen, and
formatting goes through a reused character buffer instead of f-strings. What is
left is refcounted scalars, which are freed immediately and never trigger a
pause.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import glfw
import moderngl
import numpy as np

from portal_palms import config as C
from portal_palms import gestures as G
from portal_palms import mediapipe_slim
from portal_palms.content.words import Deck, load_words
from portal_palms.render import shaders as S
from portal_palms.render.gl import Fullscreen, Presenter, Targets, Window, set_uniform
from portal_palms.render.portal import ParticleSystem, PortalRenderer
from portal_palms.render.scene import SceneRenderer
from portal_palms.render.text import NumBuf, ShapeRenderer, TextRenderer
from portal_palms.synthetic import SyntheticTracker
from portal_palms.tracking import HandState, HandTracker
from portal_palms.ui import screens
from portal_palms.ui.hud import Hud, LowPowerGovernor, Metrics
from portal_palms.ui.progress import Progress
from portal_palms.ui.recorder import Recorder, reveal

if TYPE_CHECKING:  # pragma: no cover
    import argparse

F32 = np.float32

REVEAL_DELAY: Final = 1.15  # let the animation play before the card arrives
TOAST_SECONDS: Final = 4.0
INTENSITY_STEPS: Final = (0.70, 1.00, 1.30)
INTENSITY_NAMES: Final = ("SUBTLE", "NORMAL", "MAXIMUM")


class Mode(IntEnum):
    EXPLAIN = 0
    LOADING = 1
    ERROR = 2
    LIVE = 3
    PREVIEW = 4


class App:
    # ------------------------------------------------------------------ init

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.t0 = time.perf_counter()

        self.offscreen = bool(getattr(args, "offscreen", 0))
        self.win = Window(args.width, args.height, visible=not self.offscreen)
        ctx = self.win.ctx
        self.ctx = ctx
        # Manual lifetime management: every GL object is owned for the whole
        # run and released explicitly at shutdown.
        ctx.gc_mode = None  # type: ignore[assignment]

        self.assets = C.resolve_asset_root()
        self.fs = Fullscreen(ctx)
        self.targets = Targets(ctx)
        self.presenter = Presenter(ctx, self.fs)
        self.text = TextRenderer(ctx, self.assets)
        self.shapes = ShapeRenderer(ctx)
        self.num = NumBuf(8)

        # The window is now able to draw. Everything expensive from here on
        # happens with the explainer already on screen.
        self.mode = Mode.EXPLAIN
        self.boot_message: str = "Waking up the tracker"
        self._draw_boot()
        # Milestones for the cold-start budget. What matters to the user is
        # "when did something appear", not "when did __init__ return".
        self.t_first_frame = time.perf_counter() - self.t0
        self.t_gl_ready = 0.0
        self.t_camera = 0.0
        self.t_tracking = 0.0

        self.cam_prog = ctx.program(vertex_shader=S.FULLSCREEN_VS, fragment_shader=S.CAMERA_FS)
        set_uniform(self.cam_prog, "u_cam", 0)
        self._draw_boot()
        self.portal = PortalRenderer(ctx)
        self._draw_boot()
        self.particles = ParticleSystem(ctx)
        self._draw_boot()
        self.scene = SceneRenderer(ctx, self.targets, self.fs)
        self._draw_boot()

        self.cam_tex = ctx.texture((C.CAM_W, C.CAM_H), 3)
        self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.cam_tex.repeat_x = False
        self.cam_tex.repeat_y = False
        self.fit_x = 0.5
        self.fit_y = 0.5
        self._fit_camera(C.CAM_W, C.CAM_H)
        set_uniform(self.cam_prog, "u_dim", 1.0)

        self.recorder = Recorder(ctx)
        self.hud = Hud(self.text, self.shapes)
        self.metrics = Metrics()
        self.governor = LowPowerGovernor()
        if args.low_power:
            self.governor.toggle(0.0)

        # ---- content
        words, warnings = load_words(Path(args.words) if args.words else None)
        for w in warnings:
            print("words:", w)
        self.deck = Deck(words, seed=args.seed)
        self.progress = Progress.load()

        # ---- tracking
        # --offscreen normally implies synthetic hands; --camera opts back in to
        # the real device so a headless run can check the camera composition.
        self.fake = bool(args.demo) or (self.offscreen and not getattr(args, "camera", False))
        self.tracker: HandTracker | SyntheticTracker = (
            SyntheticTracker(mirror=not args.no_mirror)
            if self.fake
            else HandTracker(
                str(self.assets / "models" / "hand_landmarker.task"), mirror=not args.no_mirror
            )
        )
        self.hands = HandState()
        self.gesture = G.PortalGesture()
        self.swipe = G.SwipeGesture()
        self.tracking_started = False
        self.first_tracked_at = 0.0

        # ---- session state
        self.mirror = not args.no_mirror
        self.intensity_step = 1
        self.settings_open = False
        self.settings_row = 0
        self.settings_slide = 0.0
        self.reveal_at = -1.0
        self.reveal_k = 0.0
        self.scored = False
        self.countdown_until = -1.0
        self.toast_text = ""
        self.toast_until = -1.0
        self.clip_saved = False
        self.scene_time = 0.0
        self._was_recording = False
        self.running = True

        self.win.on_key = self._on_key
        self.win.on_click = self._on_click

        if getattr(args, "fullscreen", False):
            self.win.toggle_fullscreen()

        self._warm()
        self.t_gl_ready = time.perf_counter() - self.t0
        self.startup_ms = self.t_gl_ready * 1000.0
        if args.autostart:
            self._begin_tracking()

        # Everything that survives init is permanent. Freezing it moves it out
        # of the collector's reach; disabling the collector then means no GC
        # pause can ever land inside a frame. A pause is a dropped frame, and a
        # dropped frame is visible in a 60fps recording.
        gc.collect()
        gc.freeze()
        gc.disable()

    def _fit_camera(self, cam_w: int, cam_h: int) -> None:
        """Letterbox the camera plate into the frame on whichever axis binds.

        Everything that converts a landmark into a screen position goes through
        fit_x / fit_y, so the gesture stays glued to the hands whatever shape
        the composition is.
        """
        a_cam = cam_w / max(cam_h, 1)
        if a_cam > C.RENDER_ASPECT:
            self.fit_x = 0.5
            self.fit_y = 0.5 * C.RENDER_ASPECT / a_cam
        else:
            self.fit_x = 0.5 * a_cam / C.RENDER_ASPECT
            self.fit_y = 0.5
        set_uniform(self.cam_prog, "u_fit", (self.fit_x, self.fit_y))

    def _warm(self) -> None:
        """Force every program through a real draw before the first visible frame.

        A shader that first compiles mid-session is a multi-hundred-millisecond
        stall, and drivers defer the real specialisation until a draw actually
        uses the program -- so warming means drawing, not just linking.
        """
        ctx = self.ctx
        self.scene.warm()
        self.portal.warm(self.targets.main, self.targets.scene_tex)
        self.particles.warm(self.targets.main)
        self.text.warm(self.targets.main)
        self.shapes.warm(self.targets.main)

        self.targets.main.use()
        ctx.viewport = (0, 0, 8, 8)
        ctx.disable(moderngl.BLEND)
        self.cam_tex.use(0)
        self.fs.draw(self.cam_prog)
        self.presenter.run_bloom(self.targets, 1.0, 0.5)
        for chain in (self.targets.bloom_hi, self.targets.bloom_lo):
            chain.fbo_a.use()
            ctx.viewport = (0, 0, 4, 4)
            self.targets.main_tex.use(0)
            self.fs.draw(self.presenter.bright)
        ctx.finish()

    # ------------------------------------------------------------------ boot frames

    def _draw_boot(self) -> None:
        """One explainer frame, drawn between the expensive init steps.

        This is what keeps the window from being blank while shaders compile
        and, later, while `import mediapipe` runs on the worker thread.
        """
        ctx = self.ctx
        phase = time.perf_counter() - self.t0
        self.targets.main.use()
        ctx.viewport = (0, 0, C.RENDER_W, C.RENDER_H)
        ctx.clear(C.BASE[0], C.BASE[1], C.BASE[2], 1.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        self.text.begin()
        self.shapes.begin()
        screens.explainer(self.text, self.shapes, phase)
        self.shapes.flush()
        self.text.flush()

        # Straight to the window: bloom is not compiled yet at the first call.
        ctx.screen.use()
        w, h = self.win.framebuffer_size()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.BLEND)
        self.presenter.present_fbo.use()
        ctx.viewport = (0, 0, C.RENDER_W, C.RENDER_H)
        self.targets.main_tex.use(0)
        set_uniform(self.presenter.blit, "u_rect", (0.0, 0.0, 1.0, 1.0))
        self.fs.draw(self.presenter.blit)
        self.presenter.to_window(w, h)
        self.win.swap()
        self.win.poll()

    # ------------------------------------------------------------------ input

    def _begin_tracking(self) -> None:
        if self.tracking_started:
            return
        self.tracking_started = True
        self.mode = Mode.LOADING
        self.boot_message = "Opening the camera"
        # Kicked off only after the explainer has been on screen for at least a
        # frame, because opening the device is what raises the OS prompt.
        self.tracker.start()

    def _on_key(self, key: int, mods: int) -> None:
        now = time.perf_counter()
        if key == glfw.KEY_ESCAPE:
            if self.settings_open:
                self.settings_open = False
            else:
                self.running = False
            return
        if key == glfw.KEY_Q and (mods & glfw.MOD_SUPER or mods & glfw.MOD_CONTROL):
            self.running = False
            return

        if self.mode is Mode.EXPLAIN:
            if key in (glfw.KEY_SPACE, glfw.KEY_ENTER):
                self._begin_tracking()
            return

        if self.mode is Mode.ERROR:
            if key == glfw.KEY_R:
                self.tracker.error = None
                self.tracker = HandTracker(
                    str(self.assets / "models" / "hand_landmarker.task"), mirror=self.mirror
                )
                self.tracking_started = False
                self._begin_tracking()
            return

        if self.mode is Mode.PREVIEW:
            self._preview_key(key)
            return

        if key == glfw.KEY_F:
            self.win.toggle_fullscreen()
            return
        if key == glfw.KEY_D:
            self.hud.visible = not self.hud.visible
            return
        if key == glfw.KEY_S:
            self.settings_open = not self.settings_open
            return

        if self.settings_open:
            self._settings_key(key, now)
            return

        if key == glfw.KEY_R and self.recorder.available:
            if self.recorder.recording:
                self._finish_clip()
            elif self.countdown_until < 0.0:
                self.countdown_until = now + C.RECORD_COUNTDOWN
            return
        if key in (glfw.KEY_SPACE, glfw.KEY_RIGHT):
            self._next_word(1)
        elif key == glfw.KEY_LEFT:
            self._next_word(-1)
        elif key == glfw.KEY_M:
            self._set_mirror(not self.mirror)
        elif key == glfw.KEY_L:
            self.governor.toggle(now)
            self._apply_power()

    def _settings_key(self, key: int, now: float) -> None:
        if key == glfw.KEY_DOWN:
            self.settings_row = (self.settings_row + 1) % 3
        elif key == glfw.KEY_UP:
            self.settings_row = (self.settings_row - 1) % 3
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_ENTER):
            step = 1 if key != glfw.KEY_LEFT else -1
            if self.settings_row == 0:
                self._set_mirror(not self.mirror)
            elif self.settings_row == 1:
                self.intensity_step = (self.intensity_step + step) % len(INTENSITY_STEPS)
            else:
                self.governor.toggle(now)
                self._apply_power()

    def _preview_key(self, key: int) -> None:
        if key == glfw.KEY_S:
            self.clip_saved = True
        elif key == glfw.KEY_F:
            if self.recorder.path is not None:
                reveal(self.recorder.path)
        elif key in (glfw.KEY_X, glfw.KEY_BACKSPACE):
            self.recorder.discard()
            self.mode = Mode.LIVE
        elif key in (glfw.KEY_SPACE, glfw.KEY_ESCAPE, glfw.KEY_ENTER):
            self.mode = Mode.LIVE

    def _on_click(self, _x: float, _y: float) -> None:
        if self.mode is Mode.EXPLAIN:
            self._begin_tracking()

    def _set_mirror(self, on: bool) -> None:
        self.mirror = on
        self.tracker.mirror = on
        self.gesture.reset()
        self.swipe.reset()

    def _apply_power(self) -> None:
        low = self.governor.active
        self.targets.low_power = low
        self.particles.set_low_power(low)
        self.win.set_vsync(2 if low else 1)

    def _toast(self, message: str, now: float) -> None:
        self.toast_text = message
        self.toast_until = now + TOAST_SECONDS

    # ------------------------------------------------------------------ content

    def _next_word(self, step: int) -> None:
        self.deck.advance(step)
        self.scene_time = 0.0
        self.reveal_at = -1.0
        self.reveal_k = 0.0
        self.scored = False
        self.gesture.unlock()

    def _finish_clip(self) -> None:
        self.recorder.stop()
        self.clip_saved = False
        self.mode = Mode.PREVIEW

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        win = self.win
        perf = time.perf_counter
        perf_ns = time.perf_counter_ns
        metrics = self.metrics

        last = perf()
        frame_start = perf_ns()

        while self.running and not win.should_close:
            win.poll()
            now = perf()
            dt = now - last
            if dt > 0.25:
                dt = 0.25  # a stall must not teleport the simulation
            last = now

            self._update(now, dt)
            cpu_mark = perf_ns()
            self._render(now, dt)
            metrics.cpu_ms = (perf_ns() - cpu_mark) * 1e-6

            self.recorder.capture(self.presenter.present_fbo, dt)

            w, h = win.framebuffer_size()
            self.presenter.to_window(w, h)
            win.swap()  # blocks on vsync; this is the pacing

            end = perf_ns()
            metrics.push((end - frame_start) * 1e-6, metrics.cpu_ms, now)
            frame_start = end

            if self.governor.update(metrics.rolling_ms, now):
                self._apply_power()
                self._toast("Eased off the effects to keep it smooth", now)

        self._shutdown()
        return 0

    # ------------------------------------------------------------------ update

    def _update(self, now: float, dt: float) -> None:
        tracker = self.tracker

        if self.mode is Mode.EXPLAIN:
            # Auto-continue for anyone who does not press anything, so a shared
            # laptop never sits on this screen forever.
            if self.tracking_started or (now - self.t0) > 12.0:
                self._begin_tracking()
            return

        if tracker.error is not None:
            self.mode = Mode.ERROR
            return

        if not tracker.model_ready:
            self.mode = Mode.LOADING
            self.boot_message = (
                "Opening the camera" if not tracker.camera_ready else ("Waking up the tracker")
            )
            return

        if self.mode is Mode.LOADING:
            self.mode = Mode.LIVE
            self.t_camera = now - self.t0
            print(
                f"cold start: window {self.t_first_frame * 1000:.0f}ms"
                f" -> shaders warm {self.t_gl_ready * 1000:.0f}ms"
                f" -> tracking live {self.t_camera * 1000:.0f}ms"
                f"  ({tracker.delegate_name}, {tracker.cam_w}x{tracker.cam_h})"
            )
            G.set_frame_aspect(tracker.cam_w, tracker.cam_h)
            self._fit_camera(tracker.cam_w, tracker.cam_h)

        self.hands.update(tracker, now)
        if self.first_tracked_at <= 0.0 and self.hands.any_hand:
            self.first_tracked_at = now
            self.t_tracking = now - self.t0

        was_locked = self.gesture.phase is G.PortalPhase.LOCKED
        self.gesture.update(self.hands, now, dt)
        locked = self.gesture.phase is G.PortalPhase.LOCKED

        if locked and not was_locked:
            self.portal.trigger_flash()
            self.reveal_at = now + REVEAL_DELAY
            self.scene_time = 0.0
        if not locked and was_locked:
            self.reveal_at = -1.0

        if locked:
            self.scene_time += dt
            if self.reveal_at > 0.0 and now >= self.reveal_at:
                self.reveal_k = min(self.reveal_k + dt * 2.2, 1.0)
                if not self.scored and self.reveal_k >= 1.0:
                    self.scored = True
                    self.progress.record(self.deck.current.word)
                    self.progress.save()
            if self.swipe.update(self.hands, now) != 0:
                self._next_word(1)
        else:
            self.reveal_k = max(self.reveal_k - dt * 3.5, 0.0)
            if self.gesture.close_requested:
                # Closing the hands shuts the portal; it does not skip the word.
                # Tearing the same portal open again replays the metaphor from
                # the start, which is what someone who did not get it will do.
                self.scene_time = 0.0
                self.scored = False

        self.portal.tick(dt)
        self.settings_slide += ((1.0 if self.settings_open else 0.0) - self.settings_slide) * min(
            dt * 12.0, 1.0
        )

        if self.countdown_until > 0.0 and now >= self.countdown_until:
            self.countdown_until = -1.0
            if not self.recorder.start():
                self._toast("Could not start the recorder", now)
        # The recorder clears its own flag when it hits the 30s cap, so the
        # preview has to be triggered from the edge rather than from a key.
        if self._was_recording and not self.recorder.recording:
            self._finish_clip()
        self._was_recording = self.recorder.recording

    # ------------------------------------------------------------------ render

    def _render(self, now: float, dt: float) -> None:
        t = now - self.t0
        intensity = INTENSITY_STEPS[self.intensity_step]
        calls = 0

        if self.mode in (Mode.EXPLAIN, Mode.LOADING, Mode.ERROR):
            calls = self._render_boot(t)
        else:
            calls = self._render_live(t, now, dt, intensity)

        self.metrics.draw_calls = calls + 4  # the bloom chain
        self.metrics.particles = self.particles.count if self.gesture.is_open else 0

        threshold = 1.15 if not self.governor.active else 1.35
        amount = (0.62 if not self.governor.active else 0.45) * intensity
        self.presenter.run_bloom(self.targets, threshold, amount)

    def _clear_main(self) -> None:
        ctx = self.ctx
        self.targets.main.use()
        ctx.viewport = (0, 0, C.RENDER_W, C.RENDER_H)
        ctx.clear(C.BASE[0], C.BASE[1], C.BASE[2], 1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    def _render_boot(self, t: float) -> int:
        self._clear_main()
        self.text.begin()
        self.shapes.begin()
        if self.mode is Mode.ERROR:
            screens.camera_error(self.text, self.shapes, self.tracker.error or "")
        elif self.mode is Mode.LOADING:
            screens.loading(self.text, self.shapes, t, self.boot_message)
        else:
            screens.explainer(self.text, self.shapes, t)
        return self.shapes.flush() + self.text.flush()

    def _render_live(self, t: float, now: float, dt: float, intensity: float) -> int:
        ctx = self.ctx
        g = self.gesture
        calls = 0

        # Portal centre, converted from mirrored camera space into the render
        # frame. The camera is fitted by width into a band, so y has to be
        # remapped through that band and then flipped into GL's y-up.
        cx = 0.5 + (g.center_x - 0.5) * (self.fit_x * 2.0)
        cy_gl = 0.5 + (0.5 - g.center_y) * (self.fit_y * 2.0)
        cy_ui = 1.0 - cy_gl
        # Landmark space is y-down, clip space is y-up, so the roll inverts.
        rot = -g.rotation

        word = self.deck.current
        sc = word.scene

        if g.is_open:
            loop = sc.loop
            self.portal.set_parallax(g.vel_x, g.vel_y, cx, g.center_y)
            par_x, par_y = self.portal.parallax
            self.scene.painter.reset()
            # Scenes only advance while locked. A portal being pulled open shows
            # the animation held on its first frame, which reads as anticipation.
            stime = self.scene_time if g.phase is G.PortalPhase.LOCKED else 0.0
            sc.paint(self.scene.painter, stime % loop)
            calls += self.scene.render(sc.tint, stime, par_x * 9.0, par_y * 9.0)

        self._clear_main()
        self.cam_tex.use(0)
        set_uniform(self.cam_prog, "u_mirror", 1.0 if self.mirror else 0.0)
        set_uniform(self.cam_prog, "u_dim", 1.0 - self.reveal_k * 0.42)
        ctx.disable(moderngl.BLEND)
        self.fs.draw(self.cam_prog)
        calls += 1
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

        if g.is_open or g.openness > 0.0 or self.hands.both:
            self.targets.scene_tex.use(0)
            self.portal.set_shape(cx, cy_gl, g.radius * self.fit_x * 2.0, g.openness, rot)
            self.portal.draw(t, g.openness, intensity, g.hold_progress)
            calls += 1
            self.particles.update(
                dt, t, cx, cy_gl, self.portal.major, self.portal.minor, rot,
                g.openness * (0.35 + 0.65 * g.hold_progress) if g.hold_progress > 0 else
                g.openness,
                g.implode,
            )  # fmt: skip
            calls += self.particles.draw(intensity)

        self.text.begin()
        self.shapes.begin()
        self._render_ui(t, now, cx, cy_ui)
        calls += self.shapes.flush() + self.text.flush()
        return calls

    def _render_ui(self, t: float, now: float, cx: float, cy_ui: float) -> None:
        txt = self.text
        sh = self.shapes
        g = self.gesture

        screens.top_bar(txt, sh, self.progress.streak, self.progress.completed, self.num)

        if self.mode is Mode.PREVIEW:
            name = self.recorder.path.name if self.recorder.path else ""
            screens.clip_preview(txt, sh, name, self.clip_saved, self.recorder.error)
            return

        if not self.hands.both:
            screens.hands_hint(txt, sh, t, self.hands.any_hand)
        elif g.phase in (G.PortalPhase.SEAM, G.PortalPhase.OPENING):
            screens.pull_hint(txt, sh, g.openness)

        if g.hold_progress > 0.0 and g.phase is not G.PortalPhase.LOCKED:
            screens.hold_ring(txt, sh, cx, cy_ui, g.radius * self.fit_x * 2.0, g.hold_progress)

        screens.word_card(txt, sh, self.deck.current, self.reveal_k, t)

        if self.countdown_until > 0.0:
            screens.countdown(txt, sh, self.countdown_until - now)
        elif self.recorder.recording:
            screens.recording_badge(txt, sh, self.recorder.seconds, t)

        if self.settings_slide > 0.001:
            screens.settings_drawer(
                txt, sh, self.settings_slide, self._settings_rows(), self.settings_row
            )

        if now < self.toast_until:
            fade = min((self.toast_until - now) / 0.8, 1.0)
            screens.toast(txt, sh, self.toast_text, fade)

        self.hud.draw(
            self.metrics,
            self.tracker.detect_ms,
            self.tracker.delegate_name,
            g.spread,
            self.governor.active,
        )

    def _settings_rows(self) -> list[tuple[str, str, bool]]:
        # Rebuilt only while the drawer is on screen, which is not a hot path.
        return [
            ("Mirror the camera", "ON" if self.mirror else "OFF", self.mirror),
            ("Effect intensity", INTENSITY_NAMES[self.intensity_step], True),
            ("Low power mode", "ON" if self.governor.active else "OFF", self.governor.active),
        ]

    # ------------------------------------------------------------------ teardown

    def run_offscreen(self, frames: int, out_dir: Path, stride: int, fps: float = 60.0) -> int:
        """Drive the whole app on a fixed clock and dump PNGs.

        Same update and render path as run(); only the clock and the presentation
        differ. That is the point -- it verifies the shipped code, not a copy.
        """
        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        buf = np.zeros((C.RENDER_H, C.RENDER_W, 3), dtype=np.uint8)
        dt = 1.0 / fps
        tracker = self.tracker
        synthetic = tracker if isinstance(tracker, SyntheticTracker) else None
        if synthetic is not None:
            synthetic.epoch = self.t0  # one clock for the tracker and the app
            self.mode = Mode.LIVE
            self.tracking_started = True
            G.set_frame_aspect(tracker.cam_w, tracker.cam_h)
        else:
            # Real device: let the normal state machine bring it up, and run on
            # the wall clock so detections land where the app expects them.
            self._begin_tracking()
        written = 0
        prev = time.perf_counter()

        for i in range(frames):
            if synthetic is not None:
                # Fixed clock: deterministic, and the same frame comes out every run.
                now = self.t0 + i * dt
                synthetic.step(now)
            else:
                # Real device: wall clock, paced roughly to the display rate so
                # detections arrive at the same cadence they would in run().
                now = time.perf_counter()
                dt = min(max(now - prev, 1.0 / 240.0), 0.25)
                prev = now
                time.sleep(max(1.0 / fps - dt, 0.0))
            self._update(now, dt)
            self._render(now, dt)
            if i % stride == 0:
                self.presenter.present_fbo.read_into(buf, components=3)
                path = out_dir / f"frame{written:04d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(np.flipud(buf), cv2.COLOR_RGB2BGR))
                written += 1
        print(f"wrote {written} frames to {out_dir}")
        self._shutdown()
        return 0

    def _shutdown(self) -> None:
        gc.enable()
        self.progress.save()
        self.recorder.release()
        self.tracker.stop()
        self.scene.release()
        self.portal.release()
        self.particles.release()
        self.text.release()
        self.shapes.release()
        self.presenter.release()
        self.targets.release()
        self.fs.release()
        self.cam_tex.release()
        self.cam_prog.release()
        self.win.terminate()


def prewarm_mediapipe() -> None:
    """Pull mediapipe into the module cache on a side thread during the explainer.

    `import mediapipe` costs the better part of a second and would otherwise be
    the first thing that happens after the user consents, right when they are
    watching for a response.
    """

    def work() -> None:
        # Any failure here is silent on purpose: the tracker thread reports the
        # real problem with a screen the user can act on.
        with contextlib.suppress(Exception):
            mediapipe_slim.install()
            import mediapipe  # noqa: F401

    threading.Thread(target=work, name="portal-prewarm", daemon=True).start()
