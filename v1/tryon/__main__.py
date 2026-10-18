"""python -m tryon : run the projector try-on.

Threads: capture (newest frame), pose worker (MediaPipe), checkout (payments).
The main thread renders once per camera frame from the latest pose, so a slow
model never lowers the projector frame rate.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from .calibration import load_calibration
from .capture import Camera
from .checkout import CheckoutClient
from .config import load_config
from .display import ProjectorWindow, cv2_key_name
from .garments import load_catalog
from .gestures import GestureDetector
from .orientation import OrientationEstimator
from .pose import PoseTracker
from .render import Renderer
from .smoothing import BodySmoother
from .state import CHARGING, IDLE, AppState

DEBUG_WINDOW = "Fitcast debug (mirror)"
KEYS_HELP = "<-/-> garment   space shirt/pants   enter checkout   c reload calib   d overlays   f fullscreen   q quit"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tryon", description="Projector virtual try-on")
    ap.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    ap.add_argument("--source", help="camera index or a video file to replay instead of the camera")
    ap.add_argument("--no-projector", action="store_true", help="debug window only, even if calib.json exists")
    ap.add_argument("--headless", action="store_true", help="no windows at all (testing)")
    ap.add_argument("--duration", type=float, default=0.0, help="quit after this many seconds")
    ap.add_argument("--record", help="save raw camera frames to this .mp4 so a session can be replayed with --source")
    ap.add_argument("--snapshot", help="on exit, save the last debug/projector frames as <prefix>_debug.png etc.")
    ap.add_argument("--keys", help="scripted keys for testing, e.g. '1.0:right,2.5:space' (seconds:key)")
    return ap.parse_args(argv)


class App:
    def __init__(self, args):
        self.args = args
        overrides = {}
        if args.source is not None:
            if args.source.isdigit():
                overrides.update(camera_index=int(args.source), camera_source="")
            else:
                overrides["camera_source"] = args.source
        self.cfg = cfg = load_config(args.config, overrides)
        self.catalog = load_catalog(cfg)
        self.state = AppState(cfg, self.catalog)
        self.renderer = Renderer(cfg, self.catalog)
        self.smoother = BodySmoother(cfg)
        self.orient = OrientationEstimator(cfg)
        self.gestures = GestureDetector(cfg)
        self.checkout = CheckoutClient(cfg)
        self.calib = None
        self.projector: ProjectorWindow | None = None
        self.camera = Camera(cfg).start()
        self.tracker = PoseTracker(cfg, self.camera).start()
        self.last_res = None
        self.stage_ms: dict[str, float] = {}
        self.fps = 0.0
        self.profile_printed = False
        self.scripted = []
        if args.keys:
            for item in args.keys.split(","):
                t, k = item.split(":")
                self.scripted.append((float(t), k.strip()))
            self.scripted.sort()
        self._load_calib(initial=True)
        if not args.headless:
            cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
            w = cfg.debug_window_width
            cv2.resizeWindow(DEBUG_WINDOW, w, int(w * cfg.camera_height / cfg.camera_width))
        self.debug_fullscreen = False
        self.recorder = None
        if args.record:
            self.recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), cfg.camera_fps,
                                            (cfg.camera_width, cfg.camera_height))
        print(f"[app] checkout provider: {self.checkout.provider}")
        print(f"[app] keys: {KEYS_HELP}")

    # -- calibration / projector -------------------------------------------
    def _load_calib(self, initial: bool = False) -> None:
        cfg = self.cfg
        path = cfg.path(cfg.calib_path)
        self.calib = load_calibration(path)
        if self.calib is None:
            print("=" * 72)
            print(f"[app] {path.name} not found -> DEBUG ONLY mode (magic mirror on the laptop).")
            print("      Connect the projector and run:  python tools/calibrate.py")
            print("      then press 'c' here (or restart) to start projecting.")
            print("=" * 72)
            return
        if tuple(self.calib.camera_size) != (cfg.camera_width, cfg.camera_height):
            print(f"[app] warning: calib.json was made at camera size {self.calib.camera_size}, "
                  f"app runs at {cfg.camera_width}x{cfg.camera_height}; recalibrate")
        print(f"[app] calibration loaded ({self.calib.method}, error {self.calib.error_px:.2f} px)")
        if self.args.no_projector or self.args.headless or (self.projector and self.projector.ok):
            return
        self.projector = ProjectorWindow(cfg)
        if not self.projector.ok:
            print(f"[app] projector window not opened: {self.projector.error}. Staying in debug-only mode.")
            self.projector = None
        else:
            print(f"[app] projecting via {self.projector.backend} on display {self.projector.display}")

    @property
    def projecting(self) -> bool:
        return self.calib is not None and (self.projector is not None or self.args.headless)

    # -- input --------------------------------------------------------------
    def handle(self, action: str, now: float) -> None:
        st = self.state
        if action in ("next", "right"):
            st.step(+1, now)
        elif action in ("prev", "left"):
            st.step(-1, now)
        elif action in ("toggle_mode", "space"):
            st.toggle_mode(now)
        elif action in ("checkout", "enter"):
            g = st.begin_checkout(now)
            if g is not None:
                order = self.checkout.submit(g)
                print(f"[checkout] {order.order_id}: {g.name} {g.price_text} -> {self.checkout.provider}")
        elif action == "c":
            self._load_calib()
        elif action == "d":
            st.debug_overlays = not st.debug_overlays
        elif action == "f":
            if self.projector:
                self.projector.toggle_fullscreen()
            elif not self.args.headless:
                self.debug_fullscreen = not self.debug_fullscreen
                cv2.setWindowProperty(DEBUG_WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if self.debug_fullscreen else cv2.WINDOW_NORMAL)
        elif action in ("q", "esc"):
            st.running = False

    def _poll_keys(self, now: float, t_start: float) -> None:
        keys = []
        if not self.args.headless:
            name = cv2_key_name(cv2.waitKeyEx(1))
            if name:
                keys.append(name)
        if self.projector:
            keys += self.projector.poll_keys()
        while self.scripted and now - t_start >= self.scripted[0][0]:
            keys.append(self.scripted.pop(0)[1])
        for k in keys:
            self.handle(k, now)

    # -- HUD ----------------------------------------------------------------
    def _hud(self) -> tuple[list[str], str]:
        st, o, g = self.state, self.orient, self.gestures
        ms = self.stage_ms
        shirt, pants = st.garment("shirt"), st.garment("pants")
        if self.projector:
            proj = f"projector: {self.projector.backend} | calib err {self.calib.error_px:.1f}px"
        elif self.calib:
            proj = "projector: calib loaded, no projector window"
        else:
            proj = "DEBUG ONLY: no calib.json  (python tools/calibrate.py, then press c)"
        lines = [
            f"FPS render {self.fps:4.1f} | camera {self.camera.read_fps:4.1f} | pose {self.tracker.infer_fps:4.1f}"
            f" | seg every {self.tracker.seg_every}",
            "ms: " + "  ".join(f"{k} {ms.get(k, 0):.1f}" for k in ("wait", "pose", "seg", "face", "warp", "proj", "debug", "show", "loop")),
            f"view {o.bucket.upper():5s} yaw {o.yaw:5.0f}  c {o.c:+.2f}  dz {o.dz:+.2f}  face {o.cue:.2f} ({o.hemisphere or '?'})"
            f"  fit {self.renderer.modes.get('shirt', '-')}/{self.renderer.modes.get('pants', '-')}",
            f"mode {st.mode.upper()}  shirt: {shirt.name if shirt else '-'}  pants: {pants.name if pants else '-'}",
            f"gesture: swipe {g.swipe_speed:+6.0f}px/s  raise {g.raise_progress:.2f}  palm {g.palm_progress:.2f}"
            f"  {('blocked: ' + g.blocked) if g.blocked else ''}",
            f"checkout: {st.checkout} ({self.checkout.provider})"
            + (f"  last: {st.checkout_result.status} {st.checkout_result.transaction_id}" if st.checkout_result else ""),
            proj,
        ]
        return lines, KEYS_HELP if st.debug_overlays else ""

    # -- main loop ------------------------------------------------------------
    def run(self) -> int:
        cfg, st = self.cfg, self.state
        last_fidx, last_seq = -1, -1
        t_start = time.monotonic()
        n_frames, fps_t0 = 0, t_start
        n_shown = 0
        dbg = proj_img = None
        try:
            while st.running:
                tw = time.perf_counter()
                frame, t_frame, fidx = self.camera.wait_for_new(last_fidx, 0.1)
                wait_ms = (time.perf_counter() - tw) * 1000
                now = time.monotonic()
                if self.args.duration and now - t_start >= self.args.duration:
                    break
                if not self.camera.running and self.camera.error:
                    print(f"[app] camera stopped: {self.camera.error}")
                    break
                if frame is None or fidx == last_fidx:
                    self._poll_keys(now, t_start)
                    continue
                last_fidx = fidx
                t_loop = time.perf_counter()
                if self.recorder is not None:
                    self.recorder.write(frame)

                res, seq = self.tracker.latest()
                if res is not None and seq != last_seq:
                    last_seq = seq
                    self.last_res = res
                    self.smoother.update(res)
                    if res.present:
                        self.orient.update(res)
                    for event in self.gestures.update(res, self.orient, st.checkout_ready):
                        print(f"[gesture] {event}")
                        self.handle(event, now)
                    for k, v in res.timings.items():
                        self.stage_ms[k] = 0.9 * self.stage_ms.get(k, v) + 0.1 * v
                if st.checkout in (IDLE, CHARGING):
                    st.set_charging(self.gestures.palm_progress)
                result = self.checkout.poll()
                if result is not None:
                    print(f"[checkout] {result.status} {result.transaction_id} ({result.latency_ms:.0f} ms)")
                    st.finish_checkout(result, now)
                st.tick(now)

                ahead = cfg.projector_latency_ms / 1000.0 if self.projecting else 0.0
                body = self.smoother.predict((now if self.projecting else t_frame) + ahead)
                mask = self.last_res.mask if self.last_res is not None else None
                self.renderer.build_layer(body, st, self.orient.view_weights(now), mask, now)
                if self.projecting:
                    proj_img = self.renderer.projector_frame(self.calib, st, self.gestures, now)
                every = cfg.debug_every_n_when_projecting if self.projector else 1
                show_debug = n_shown % max(1, every) == 0
                n_shown += 1
                if show_debug and (not self.args.headless or self.args.snapshot):
                    hud, footer = self._hud()
                    dbg = self.renderer.debug_frame(frame, body, st, self.gestures, hud, footer, now)

                ts = time.perf_counter()
                if self.projector and proj_img is not None:
                    self.projector.show(proj_img)
                if show_debug and not self.args.headless:
                    cv2.imshow(DEBUG_WINDOW, dbg)
                self._poll_keys(now, t_start)
                show_ms = (time.perf_counter() - ts) * 1000

                for k, v in {**self.renderer.timings, "wait": wait_ms, "show": show_ms,
                             "loop": (time.perf_counter() - t_loop) * 1000}.items():
                    self.stage_ms[k] = 0.9 * self.stage_ms.get(k, v) + 0.1 * v
                n_frames += 1
                if now - fps_t0 >= 1.0:
                    self.fps = n_frames / (now - fps_t0)
                    n_frames, fps_t0 = 0, now
                if not self.profile_printed and now - t_start >= cfg.profile_print_after_s:
                    self.profile_printed = True
                    self._print_profile()
        except KeyboardInterrupt:
            pass
        finally:
            self._print_profile()
            if self.args.snapshot:
                if dbg is not None:
                    cv2.imwrite(f"{self.args.snapshot}_debug.png", dbg)
                if proj_img is not None:
                    cv2.imwrite(f"{self.args.snapshot}_projector.png", proj_img)
            self.shutdown()
        return 0

    def _print_profile(self) -> None:
        ms = self.stage_ms
        print(f"[perf] render {self.fps:.1f} fps, camera {self.camera.read_fps:.1f} fps, pose {self.tracker.infer_fps:.1f} fps | "
              + " ".join(f"{k}={ms[k]:.1f}ms" for k in sorted(ms)))

    def shutdown(self) -> None:
        self.tracker.stop()
        self.camera.stop()
        if self.recorder is not None:
            self.recorder.release()
        if self.projector:
            self.projector.close()
        if not self.args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def main(argv=None) -> int:
    args = parse_args(argv)
    return App(args).run()


if __name__ == "__main__":
    sys.exit(main())
