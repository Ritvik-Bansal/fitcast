#!/usr/bin/env python
"""FitCast entry point: projection-mapped virtual try-on.

Examples:
  python main.py --list-cameras
  python main.py --source assets/test/white_turn.mp4            # offline, debug window only
  python main.py --source 0 --demo                               # live, fullscreen on the projector
  python main.py --source assets/test/white_turn.mp4 --no-window --record out/white_review.mp4 --max-frames 600
"""
from __future__ import annotations

import argparse
import logging
import sys
import subprocess
import time
import queue
from pathlib import Path

import cv2
import numpy as np
import yaml

from fitcast.calibration import Calibration
from fitcast.capture import FrameSource, list_cameras
from fitcast.compositor import Compositor
from fitcast.display import Display, draw_debug, draw_skeleton_for_projector
from fitcast.garments import load_outfits
from fitcast.gestures import GestureDetector
from fitcast.renderer import Renderer
from fitcast.sizing import SizeEstimator
from fitcast.smoothing import Smoother
from fitcast.tracker import Tracker
from fitcast.types import GalleryState

log = logging.getLogger("fitcast")

KEY_PREV = {2, 63234, 65361, ord("p")}        # left arrow (Cocoa / waitKeyEx / X11), p
KEY_NEXT = {3, 63235, 65363, ord("n")}        # right arrow, n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FitCast projection-mapped virtual try-on")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--calibration", help="calibration yaml to load (default: config calibration.file)")
    ap.add_argument("--source", help="camera index or video file (default: config capture.source)")
    ap.add_argument("--list-cameras", action="store_true", help="print capture devices and exit")
    ap.add_argument("--demo", action="store_true", help="fullscreen projector, auto-cycle outfits, auto-recover")
    ap.add_argument("--projector-x", type=int, help="override projector window x origin")
    ap.add_argument("--projector-y", type=int, help="override projector window y origin")
    ap.add_argument("--height", type=float, help="person height in cm for sizing (default: config)")
    ap.add_argument("--backend", choices=["mediapipe", "rtmlib"], help="pose backend override")
    ap.add_argument("--mask-mode", choices=["clothes", "skeleton", "intersect"])
    ap.add_argument("--outfit", type=int, default=0, help="starting outfit index")
    ap.add_argument("--record", help="write a side-by-side review video (debug | composite) to this path")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = run forever)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after N seconds wall clock")
    ap.add_argument("--no-window", action="store_true", help="headless: no windows, process every frame in order")
    ap.add_argument("--no-projector", action="store_true", help="debug window only (laptop-only development)")
    ap.add_argument("--no-realtime", action="store_true", help="video files: do not pace to native fps")
    ap.add_argument("--no-roi", action="store_true", help="track the whole camera frame, not just the projector footprint")
    ap.add_argument("--stylist", action="store_true", help="Gemini outfit picker: key v = talk, or write out/ask.txt")
    ap.add_argument("--show-skeleton", action="store_true", help="also project the tracked skeleton (key k toggles)")
    ap.add_argument("--sensor", choices=["webcam", "iphone"], help="webcam (default) or the iPhone LiDAR sensor app")
    ap.add_argument("--sensor-host", help="iphone sensor: connect to this IP instead of USB (usbmux)")
    ap.add_argument("--calibration3d", help="calibration3d.yaml for depth-correct rendering (default: config)")
    ap.add_argument("--loop", action="store_true", help="headless: keep looping the video file (soak tests)")
    ap.add_argument("--log-level", default=None)
    return ap.parse_args(argv)


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"config not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


class App:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg, self.args = cfg, args
        cap_cfg = dict(cfg["capture"])
        if args.no_realtime or args.no_window:
            cap_cfg["video_realtime"] = False
        if args.no_window and not args.loop:
            cap_cfg["video_loop"] = False          # headless review runs process the file exactly once
        self.sensor_kind = args.sensor or cfg["sensor"].get("kind", "webcam")
        if self.sensor_kind == "iphone":
            cs = cfg["sensor"].get("cam_size", [960, 720])          # 4:3 like the phone: square pixels for sizing
            cap_cfg["width"], cap_cfg["height"] = int(cs[0]), int(cs[1])
        self.size = (int(cap_cfg["width"]), int(cap_cfg["height"]))
        self.depth_mask = None
        self.proj3d = None
        self._bg_depth: list = []
        self._bg_conf: list = []
        if self.sensor_kind == "iphone":
            from fitcast.sensors.iphone import IPhoneSensor
            scfg = dict(cfg["sensor"])
            if args.sensor_host:
                scfg["usbmux"], scfg["host"] = False, args.sensor_host
            self.source = IPhoneSensor(scfg, self.size, tracker_cfg=cfg["tracker"])
        else:
            self.source = FrameSource(cap_cfg, args.source, sync=args.no_window)
        tcfg = dict(cfg["tracker"])
        if args.backend:
            tcfg["backend"] = args.backend
        if args.mask_mode:
            tcfg["mask"] = dict(tcfg["mask"], mode=args.mask_mode)
        self.tracker = Tracker(tcfg, self.size)
        self.smoother = Smoother(cfg["smoothing"], mask_alpha=float(tcfg["mask"]["temporal_alpha"]))
        self.renderer = Renderer(cfg["renderer"], self.size, tcfg["min_visibility"])
        self.compositor = Compositor(cfg["compositor"], self.size)
        self.calib = Calibration(cfg["calibration"], self.size)
        is_file = getattr(self.source, "is_file", False)
        self.calib3d = None
        if self.sensor_kind == "iphone":
            from pathlib import Path as _P
            c3d_path = args.calibration3d or cfg["calibration3d"]["file"]
            if _P(c3d_path).exists():
                from fitcast.calib3d import load_calibration3d
                self.calib3d = load_calibration3d(c3d_path)
                log.info("3D calibration %s loaded (reproj %.2f px, %d points)", c3d_path,
                         self.calib3d.reproj_error_px, self.calib3d.n_points)
            else:
                log.warning("no %s: depth-correct rendering disabled, falling back to the 2D homography "
                            "(calibration.yaml, webcam camera space!). Run scripts/calibrate3d.py --sensor iphone.", c3d_path)
        if self.calib3d is None:
            self.calib.load(args.calibration, allow_simulated=is_file)   # a fake homography is fine for clips, never live
        # depth mask runs at the sensor's depth resolution (set up on the first frame, see _step_sensor_track)
        self._proj_cache: np.ndarray | None = None
        self._K_warned = False
        if self.calib.ok and getattr(self.calib.data, "gallery_rect", None) and self.sensor_kind != "iphone":
            self._apply_layout()
        if self.calib.ok and tcfg.get("roi_from_calibration", True) and not args.no_roi and self.sensor_kind != "iphone":
            # only track inside the projector footprint (+margin): bystanders beside the rig are ignored
            c = self.calib.unproject_points(np.array([[0, 0], [self.calib.projector_size[0], 0],
                                                      list(self.calib.projector_size), [0, self.calib.projector_size[1]]], float))
            self.tracker.set_roi((c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()),
                                 float(tcfg.get("roi_margin_frac", 0.1)))
            self.tracker.set_footprint(c)
        self.gestures = GestureDetector(cfg["gestures"])
        self.sizing = SizeEstimator(cfg["sizing"], args.height)
        try:
            outfits = load_outfits(cfg["garments"])
        except RuntimeError as e:
            log.warning("%s -- generating the procedural demo set now", e)
            import subprocess
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "scripts" / "prepare_garments.py"),
                            "--demo", "--config", args.config], check=True)
            outfits = load_outfits(cfg["garments"])
        self.gallery = GalleryState(outfits=outfits, index=args.outfit % len(outfits), prev_index=args.outfit % len(outfits))
        gcfg = cfg["compositor"]["gallery"]
        hide_t = set(gcfg.get("hide_tops") or []); hide_b = set(gcfg.get("hide_bottoms") or [])
        seen: set = set()
        for i, o in enumerate(outfits):            # bottom swipes cycle distinct bottoms only (many demo outfits share "Jeans")
            key = o.name.split(" + ")[-1] if o.bottom_front is not None else None
            if key is not None and key not in seen and key not in hide_b:
                seen.add(key)
                self.gallery.bottom_cycle.append(i)
        seen_t: set = set()
        for i, o in enumerate(outfits):            # tops strip / swipes: distinct, not hidden (drawn garments are hidden)
            key = o.name.split(" + ")[0] if o.top_front is not None else None
            if key is not None and key not in seen_t and key not in hide_t:
                seen_t.add(key)
                self.gallery.top_cycle.append(i)
        if self.gallery.top_cycle and self.gallery.index not in self.gallery.top_cycle:
            self.gallery.index = self.gallery.prev_index = self.gallery.top_cycle[0]
        if self.gallery.bottom_cycle and self.gallery.outfits[self.gallery.index].name.split(" + ")[-1] in hide_b:
            self.gallery.bottom_index = self.gallery.bottom_cycle[0]
        log.info("%d outfits loaded: %s", len(outfits), ", ".join(o.id for o in outfits))
        proj = (int(cfg["calibration"]["projector_width"]), int(cfg["calibration"]["projector_height"]))
        self.display = Display(cfg["display"], proj, (args.projector_x, args.projector_y), headless=args.no_window)
        self.show_overlay = True
        self.show_skeleton = bool(args.show_skeleton)
        self.show_debug = bool(cfg["display"]["show_debug"])
        self.writer: cv2.VideoWriter | None = None
        self.fps = 0.0
        self.latency_ms = 0.0
        self.render_ms = 0.0
        self.last_gesture_t = time.perf_counter()
        self.last_cycle_t = time.perf_counter()
        self.frames = 0
        self.gesture_log: list[tuple[float, str]] = []
        from collections import deque
        self._stats = {k: deque(maxlen=int(cfg["logging"]["stats_window"])) for k in ("pose", "seg", "render", "loop")}
        self._black = np.zeros((int(cfg["calibration"]["projector_height"]), int(cfg["calibration"]["projector_width"]), 3), np.uint8)
        self._stale = False
        self.stale_s = float(cfg["capture"]["stale_ms"]) / 1000.0
        self.stall_exit_s = float(cfg["capture"]["stall_exit_s"])
        self.reopen_s = float(cfg["capture"]["reopen_s"])
        self.pump_every = max(1, int(cfg["display"].get("pump_every_n", 1)))
        self._torso_ema: float | None = None
        self._proj_person: np.ndarray | None = None
        self.stylist = None
        self.recorder = None
        self.wake = None
        self._dump_until, self._dump_n, self._dump_pending = 0.0, 0, -1
        self._frame_stats: list[str] | None = [] if cfg["logging"].get("frame_stats", False) else None
        if self._frame_stats is not None:
            Path(cfg["logging"].get("frame_stats_file", "out/frame_stats.csv")).write_text("t,ok,facing,sw,mask_px,lit_px,fallback,outfit\n")
        scfg = cfg.get("stylist", {})
        if (args.stylist or scfg.get("enabled", False)) and self.sensor_kind != "iphone":
            from fitcast.stylist import GeminiStylist, Recorder, load_api_key
            key = load_api_key(scfg)
            if key:
                self.stylist = GeminiStylist({**scfg, "hide_tops": cfg["compositor"]["gallery"].get("hide_tops") or [],
                                              "hide_bottoms": cfg["compositor"]["gallery"].get("hide_bottoms") or []}, outfits, key)
                self.recorder = Recorder(scfg)
                log.info("stylist on: press %s to talk (again to send) or write a question to %s",
                         scfg.get("record_key", "v"), scfg.get("ask_file", "out/ask.txt"))
                self._wake_events: queue.Queue = queue.Queue()
                wcfg = scfg.get("wake", {})
                if wcfg.get("enabled", True):
                    from fitcast.wakeword import WakeListener
                    self.wake = WakeListener(wcfg, on_wake=lambda: self._wake_events.put(("wake", None)),
                                             on_request=lambda path, hint: self._wake_events.put(("request", (path, hint))),
                                             on_error=lambda msg: self._wake_events.put(("error", msg)))
                    self.wake.start()
                    log.info("wake word on: say %r and then your request", wcfg.get("wake_words", ["fitbot"])[0])
        if self.sensor_kind != "iphone":
            self.compositor.use_separate_gallery_layer(True)
        self.depth_shift = (0.0, 0.0)
        self._depth_render: np.ndarray | None = None
        self._want_selfcal = False
        import signal
        def _on_usr1(*_):
            self._want_selfcal = True
            log.info("self-calibration trigger received (SIGUSR1); runs on the next tracked frame")
        signal.signal(signal.SIGUSR1, _on_usr1)   # kill -USR1 <pid> = key t
        self._last_reopen = 0.0
        self.predict_auto = bool(cfg["smoothing"]["predict_auto"])
        self.predict_max_ms = float(cfg["smoothing"]["predict_max_ms"])
        self._lat_ema = float(cfg["smoothing"]["predict_ms"])

    # -- outfit control -----------------------------------------------------------------------
    def _gallery_rects(self) -> tuple[list[int], list[int]]:
        """Bottoms strip on the left band and tops strip on the right band of the projector footprint (camera px),
        so both always land inside the beam."""
        pw, ph = self.calib.projector_size
        c = self.calib.unproject_points(np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float64))
        gx0, gy0 = max(c[0][0], c[3][0]), max(c[0][1], c[1][1])
        gx1, gy1 = min(c[1][0], c[2][0]), min(c[2][1], c[3][1])
        fw, fh = gx1 - gx0, gy1 - gy0
        gcfg = self.cfg["compositor"]["gallery"]
        strip_w = max(int(gcfg.get("fp_w_frac", 0.12) * fw), int(gcfg["thumb_size"] * gcfg["current_scale"]) + 24)
        mx, my = gcfg.get("fp_x_frac", 0.03) * fw, gcfg.get("fp_y_frac", 0.08) * fh
        left = [int(gx0 + mx), int(gy0 + my), int(gx0 + mx) + strip_w, int(gy1 - my)]
        right = [int(gx1 - mx) - strip_w, int(gy0 + my), int(gx1 - mx), int(gy1 - my)]
        return left, right

    def _apply_layout(self) -> None:
        """Strips on both sides of the beam and the stylist message band at the top between them."""
        left, right = self._gallery_rects()
        self.compositor.set_gallery_rects(left, right)
        band_h = int(0.3 * (left[3] - left[1]))
        self.compositor.set_message_rect([left[2] + 8, left[1], right[0] - 8, left[1] + band_h])

    def set_pair(self, top: int | None, bottom: int | None, now: float) -> None:
        """Stylist pick: a top and a bottom by catalogue index (None keeps the current one)."""
        g = self.gallery
        if top is not None:
            g.prev_index, g.index = g.index, top % len(g.outfits)
        if bottom is not None:
            g.bottom_index = -1 if bottom == g.index else bottom % len(g.outfits)
        g.strip, g.changed_at = "outfits", now
        log.info("stylist outfit -> %s", g.current.name)

    def set_outfit(self, idx: int, now: float) -> None:
        """Whole outfit (keys / auto-cycle): bottoms follow the top again."""
        g = self.gallery
        cyc = g.top_cycle or list(range(len(g.outfits)))
        pos = cyc.index(g.index) if g.index in cyc else 0
        g.prev_index, g.index = g.index, cyc[(pos + (1 if idx > g.index else -1)) % len(cyc)]
        g.bottom_index, g.strip, g.changed_at = -1, "outfits", now
        log.info("outfit -> %s", g.current.name)

    def set_top(self, idx: int, now: float) -> None:
        """Swipe above the hips: change the top only; the bottoms stay."""
        g = self.gallery
        if g.bottom_index < 0:
            g.bottom_index = g.index
        cyc = g.top_cycle or list(range(len(g.outfits)))
        pos = cyc.index(g.index) if g.index in cyc else 0
        step = 1 if idx > g.index else -1
        g.prev_index, g.index = g.index, cyc[(pos + step) % len(cyc)]
        g.strip, g.changed_at = "tops", now
        log.info("top -> %s", g.current.name)

    def set_bottom(self, idx: int, now: float) -> None:
        """Swipe below the hips: change the bottoms only; the top stays."""
        g = self.gallery
        cur = g.bottom_index if g.bottom_index >= 0 else g.index
        cyc = g.bottom_cycle or list(range(len(g.outfits)))
        cur_name = g.outfits[cur].name.split(" + ")[-1]
        pos = next((k for k, i in enumerate(cyc) if g.outfits[i].name.split(" + ")[-1] == cur_name), 0)
        step = 1 if idx > cur else -1
        g.prev_index, g.bottom_index = cur, cyc[(pos + step) % len(cyc)]
        g.strip, g.changed_at = "bottoms", now
        log.info("bottoms -> %s", g.current.name)

    def handle_key(self, k: int, now: float) -> bool:
        """Returns False when the app should quit."""
        if k in (-1, 0xFFFF):
            return True
        if k in (ord("q"), 27):
            return False
        if k in KEY_NEXT:
            self.set_outfit(self.gallery.index + 1, now)
            self.last_gesture_t = now          # operator input resets the demo idle timer (auto-cycle must not)
        elif k in KEY_PREV:
            self.set_outfit(self.gallery.index - 1, now)
            self.last_gesture_t = now
        elif k == ord("m"):
            self.tracker.cycle_mask_mode()
        elif self.stylist is not None and k == ord(str(self.cfg["stylist"].get("record_key", "v"))[0]):
            if self.recorder.recording:
                path = self.recorder.stop()
                if path is not None and self.stylist.ask_audio(path):
                    self.compositor.set_message(["Stylist is thinking..."], now, 30)
                else:
                    self.compositor.set_message(["Could not hear you. Press v and speak."], now, 6)
            elif self.recorder.start():
                self.compositor.set_message(["Listening... press v again to send"], now, 20)
        elif k == ord("d"):
            self.show_overlay = not self.show_overlay
        elif k == ord("f"):
            self.display.toggle_fullscreen()
        elif k == ord("c"):
            # 'c' = re-capture the frontal reference + sizing (projector recalibration is scripts/calibrate.py, 'r' reloads it)
            log.info("re-capturing frontal reference; stand facing the mirror")
            self.tracker.reset_reference()
            self.sizing.reset()
        elif k == ord("g"):
            self.gallery.visible = not self.gallery.visible
        elif k == ord("b"):
            self.relearn_background()
        elif k == ord("t"):
            self._want_selfcal = True             # run on the next tracked frame (needs the person's mask)
        elif k == ord("k"):
            self.show_skeleton = not self.show_skeleton
            log.info("skeleton on projector: %s", self.show_skeleton)
        elif k in (ord("["), ord("]")):
            self.renderer.vertical_offset_frac += -0.02 if k == ord("[") else 0.02
            log.info("renderer.vertical_offset_frac = %.2f (shoulder widths; put this in config.yaml to keep it)",
                     self.renderer.vertical_offset_frac)
        elif k == ord("r"):
            if self.calib.load(self.args.calibration, allow_simulated=getattr(self.source, "is_file", False)):
                if self.calib.data.gallery_rect:
                    self._apply_layout()
                log.info("calibration reloaded")
        return True

    # -- main loop ----------------------------------------------------------------------------
    def run(self) -> int:
        a = self.args
        self.source.start()
        if not a.no_window:
            if not a.no_projector:
                self.display.open_projector(fullscreen=a.demo or self.display.second is not None)
            if self.show_debug:
                self.display.open_debug()
        if a.record:
            Path(a.record).parent.mkdir(parents=True, exist_ok=True)
            w, h = self.size
            fps = float(self.source.native_fps)
            if not (1.0 < fps <= 120.0):
                fps = float(self.cfg["capture"]["fps"])        # sensors report 0 until frames flow
            self.writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
            log.info("recording review video to %s", a.record)
        t_start = time.perf_counter()
        last_seq = -1
        fps_t = t_start
        running = True
        try:
            while running:
                item = self.source.read()
                if item is None or item[2] == last_seq:
                    if a.no_window and self.sensor_kind != "iphone":
                        break
                    if a.no_window:                      # headless sensor run: wait for the next frame
                        time.sleep(0.002)
                        if time.perf_counter() - t_start > self.stall_exit_s and item is None:
                            log.error("no sensor frames within %.0f s", self.stall_exit_s)
                            break
                        continue
                    now = time.perf_counter()
                    age = now - (item[1] if item is not None else t_start)
                    if age > self.stale_s:
                        # SAFETY: never hold a stale garment frame on the projector while the person may move
                        if not self._stale:
                            log.error("no new camera frame for %.1f s: projector blanked", age)
                            self._stale = True
                        self.display.show_projector(self._black)
                    if age > self.reopen_s and now - self._last_reopen > self.reopen_s:
                        self._last_reopen = now
                        if self.source.reopen():
                            last_seq = -1
                            t_start = now          # restart the stall clock on a successful reopen
                    if age > self.stall_exit_s:
                        log.error("camera stalled for %.0f s; exiting. Check the cable / Continuity Camera and restart.", age)
                        break
                    running = self.handle_key(self.display.key(5 if item is None else 1), now)
                    continue
                frame, ts, seq = item
                last_seq = seq
                self._stale = False
                now = time.perf_counter()
                self.step(frame, ts, now)
                self.frames += 1
                # fps (EMA over loop period)
                dt = now - fps_t
                fps_t = now
                inst = 1.0 / dt if dt > 0 else 0.0
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
                self._stats["loop"].append(dt)
                if self.frames % self.pump_every == 0:
                    running = self.handle_key(self.display.key(1), now)
                if a.max_frames and self.frames >= a.max_frames:
                    break
                if a.max_seconds and now - t_start >= a.max_seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        self.report(time.perf_counter() - t_start)
        return 0

    def _step_sensor_track(self, frame: np.ndarray, ts: float):
        """iPhone path: phone joints + depth-based person mask -> tracker's mask/head/facing logic.
        The depth background is learned from frames where NOBODY is tracked (empty scene); the mask module runs at
        the sensor's depth resolution and upsamples itself."""
        lt = self.source.latest()
        st = None if lt is None else lt["track"]
        empty = self.tracker.process_external(frame, ts, np.zeros((33, 3), np.float32), np.zeros((33,), np.float32), None)
        if st is None or lt["depth_raw"] is None:
            return empty
        depth_raw, conf_raw = lt["depth_raw"], lt["conf_raw"]
        if conf_raw is None:
            conf_raw = np.full(depth_raw.shape, 2, np.uint8)
        dh, dw = depth_raw.shape[:2]
        if self.depth_mask is None:
            from fitcast.depthmask import DepthBodyMask
            self.depth_mask = DepthBodyMask(self.cfg["depthmask"], (dw, dh), self.size)
            log.info("depth mask at sensor resolution %dx%d -> camera space %dx%d", dw, dh, *self.size)
        if st.K_cam is not None:
            K_d = np.array(st.K_cam, np.float64).copy()
            K_d[0, :] *= dw / self.size[0]
            K_d[1, :] *= dh / self.size[1]
            self.depth_mask.set_intrinsics(K_d)
        if not self.depth_mask.ready:
            if not st.ok:                                   # only the empty scene goes into the background
                self._bg_depth.append(np.asarray(depth_raw, np.float32).copy()); self._bg_conf.append(conf_raw.copy())
                need = int(self.cfg["depthmask"]["background_frames"])
                if len(self._bg_depth) >= need:
                    self.depth_mask.learn_background(self._bg_depth, self._bg_conf)
                    self._bg_depth.clear(); self._bg_conf.clear()
                    log.info("depth background learned (%d empty-scene frames); person may step in", need)
            elif self.frames % 60 == 0:
                log.warning("depth background not learned yet: step OUT of the beam for 2 s (key b restarts)")
            return empty                                     # no garment until the background exists
        if not st.ok:
            return empty
        person = self.depth_mask.compute(np.asarray(depth_raw, np.float32), conf_raw, frame, person_seg=lt["seg"])
        track = self.tracker.process_external(frame, ts, st.joints, st.visibility, person)
        track.depth, track.joints3d, track.K_cam = st.depth, st.joints3d, st.K_cam
        return track

    def _depth_for_render(self, track) -> np.ndarray:
        """Depth used by the GPU projector: the head box is forced to an opaque surface at body depth so wall
        content seen beside/through a depth-less head can never be projected onto the head."""
        d = track.depth
        x0, y0, x1, y1 = track.head_box
        if x1 <= x0 or y1 <= y0:
            return d
        body = d[track.clothes_mask > 0] if track.clothes_mask is not None else d[d > 0]
        body = body[body > 0]
        fill = float(np.median(body)) if body.size else float(np.median(d[d > 0])) if (d > 0).any() else 0.0
        if fill <= 0:
            return d
        if self._depth_render is None or self._depth_render.shape != d.shape:
            self._depth_render = np.empty_like(d)
        np.copyto(self._depth_render, d)
        self._depth_render[y0:y1, x0:x1] = fill
        return self._depth_render

    def run_self_calibration(self, track) -> None:
        """Person-triggered calibration (T-pose / key t): project dim discs one at a time, use the ones that land on
        the body to solve the camera->projector mapping at the person's distance (second distance -> two-plane model).
        Blocks the loop for ~8 s; the person must stand still."""
        if self.args.no_window or self.sensor_kind == "iphone" or not track.ok:
            log.warning("self-calibration needs the projector window, the webcam path and a tracked person")
            return
        from fitcast import geometry as _geo
        from fitcast.selfcal import self_calibrate
        scfg = self.cfg["selfcal"]
        person = track.person_mask if track.person_mask is not None else track.clothes_mask
        torso_px = float(_geo.torso_length(track.joints))
        proj_size = (int(self.cfg["calibration"]["projector_width"]), int(self.cfg["calibration"]["projector_height"]))
        last_seq = [-1]

        def capture_fn():
            for attempt in range(2):
                t_end = time.perf_counter() + 2.5
                while time.perf_counter() < t_end:
                    it = self.source.read()
                    if it is not None and it[2] != last_seq[0]:
                        last_seq[0] = it[2]
                        return it[0]
                    self.display.key(1)
                    time.sleep(0.002)
                log.warning("camera stalled during self-calibration; reopening")   # the MacBook camera does this ~8 s after opening
                if not self.source.reopen():
                    break
                last_seq[0] = -1
            raise RuntimeError("camera stopped delivering frames during self-calibration")

        def show_fn(frame):
            self.display.show_projector(frame)
            self.display.key(1)

        log.info("self-calibration starting: hold still (%d discs, ~%d s)", int(scfg["grid"][0]) * int(scfg["grid"][1]),
                 int(scfg["grid"][0]) * int(scfg["grid"][1]) * (int(self.cfg["calibration"]["settle_frames"]) + int(self.cfg["calibration"]["median_frames"])) // 30 + 1)
        try:
            res = self_calibrate(scfg, self.cfg["calibration"], self.calib, proj_size, capture_fn, show_fn,
                                 person, track.head_box, torso_px, log_fn=log.info)
        except RuntimeError as e:
            log.error("self-calibration failed: %s", e)
            return
        if res["H_body"] is None:
            log.error("self-calibration failed: %d discs on the body, %d on the wall; no usable mapping. Stand centrally "
                      "in the beam, hold still, dim the room; see out/selfcal_disc_*.jpg", res["n_body"], res["n_wall"])
            return
        if res["n_body"] < int(scfg["min_body_discs"]):
            log.warning("self-calibration: only %d body discs; mapping comes mostly from the wall (depth correction weak)", res["n_body"])
        if scfg.get("dense", True):
            from fitcast.selfcal import dense_scan
            log.info("self-calibration: dense Gray-code scan (46 patterns, ~%d s): keep holding still",
                     46 * (int(scfg.get("settle_frames", 5)) + int(scfg.get("median_frames", 2))) // 30 + 1)
            try:
                hips_ok = track.visibility[23] >= self.tracker.min_vis and track.visibility[24] >= self.tracker.min_vis
                split_y = float(0.5 * (track.joints[23, 1] + track.joints[24, 1])) if hips_ok else None
                dres = dense_scan(scfg, self.cfg["calibration3d"], proj_size, capture_fn, show_fn, person, track.head_box,
                                  res["H_body"], log_fn=log.info, split_y=split_y)
            except RuntimeError as e:
                log.error("dense scan failed: %s (keeping the disc mapping)", e)
                dres = None
            if (dres is not None and dres["H_body"] is not None and dres["n_body"] >= int(scfg.get("dense_min_body_points", 300))
                    and dres["inlier_frac_body"] >= float(scfg.get("dense_min_inlier_frac", 0.5))):
                res["H_body"], res["rms_body"], res["n_body"] = dres["H_body"], dres["rms_body"], dres["n_body"]
                log.info("self-calibration: dense body plane (%d points, rms %.2f px, residual median %.1f / p95 %.1f px) "
                         "replaces the disc solution; residual map out/selfcal_residual.png", dres["n_body"], dres["rms_body"],
                         dres["res_med"], dres["res_p95"])
                nmin = int(scfg.get("split_min_points", 200))
                if (scfg.get("split_model", True) and dres["H_upper"] is not None and dres["H_lower"] is not None
                        and dres["n_upper"] >= nmin and dres["n_lower"] >= nmin):
                    inv = np.linalg.inv(dres["H_body"])
                    Du, Dl = dres["H_upper"] @ inv, dres["H_lower"] @ inv
                    self._pending_D = (Du / Du[2, 2], Dl / Dl[2, 2])
                    c = cv2.perspectiveTransform(np.array([[[self.size[0] / 2.0, float(split_y or self.size[1] / 2)]]], np.float64), dres["H_body"])[0, 0]
                    cu = cv2.perspectiveTransform(c.reshape(1, 1, 2), self._pending_D[0])[0, 0]
                    cl = cv2.perspectiveTransform(c.reshape(1, 1, 2), self._pending_D[1])[0, 0]
                    log.info("self-calibration: split model on (upper %d / lower %d points); at the hip line the two planes "
                             "differ by (%.0f, %.0f) projector px", dres["n_upper"], dres["n_lower"], cu[0] - cl[0], cu[1] - cl[1])
                else:
                    self._pending_D = (None, None)
            elif dres is not None:
                log.warning("self-calibration: dense scan rejected (%d body points, %.0f%% inliers); keeping the disc mapping",
                            dres["n_body"], 100 * dres["inlier_frac_body"])
        if float(scfg.get("mesh_show_s", 3.0)) > 0:
            from fitcast.selfcal import render_dot_mesh
            try:
                mesh = render_dot_mesh(proj_size, person, track.head_box, res["H_body"], scfg)
                cv2.imwrite("out/selfcal_mesh.png", mesh)
                show_fn(mesh)
                t_end = time.perf_counter() + float(scfg.get("mesh_show_s", 3.0))
                while time.perf_counter() < t_end:
                    capture_fn()
            except RuntimeError:
                pass
        which = self.calib.set_plane(res["H_body"], torso_px, float(scfg["change_frac"]))
        if getattr(self, "_pending_D", None) is not None:
            self.calib.data.D_upper, self.calib.data.D_lower = self._pending_D
            self._pending_D = None
        self.calib.data.reproj_error_px = float(res["rms_body"] if res["rms_body"] is not None else 0.0)
        self.calib.data.verified = True
        self.calib.data.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S") + f" selfcal({which})"
        left, right = self._gallery_rects()
        self.calib.data.gallery_rect = left
        self._apply_layout()
        self.calib.save()
        self.tracker.reset_reference()
        log.info("self-calibration saved as the %s plane: %d body discs, rms %.2f px, torso %.0f px%s", which,
                 res["n_body"], res["rms_body"], torso_px,
                 "" if which == "second" else " (T-pose again ~1 m closer or farther for depth tracking)")

    def _step_stylist(self, now: float) -> None:
        """Typed questions from the ask file; finished replies dress the person, speak and show the shopping list."""
        scfg = self.cfg["stylist"]
        ask = Path(scfg.get("ask_file", "out/ask.txt"))
        if self.frames % 10 == 0 and ask.exists():
            q = ask.read_text().strip()
            ask.unlink(missing_ok=True)
            if q and self.stylist.ask_text(q):
                log.info("stylist question: %s", q)
                self.compositor.set_message([q[:70], "Stylist is thinking..."], now, 30)
        while self.wake is not None and not self._wake_events.empty():
            kind, payload = self._wake_events.get_nowait()
            if kind == "wake":
                self.compositor.set_message(["Yes? I'm listening..."], now, 15)
                chime = scfg.get("wake", {}).get("chime", "/System/Library/Sounds/Glass.aiff")
                if chime:
                    subprocess.Popen(["afplay", chime], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif kind == "request":
                path, hint = payload
                if self.stylist.ask_audio(path, hint):
                    self.compositor.set_message([f"Heard: {hint[:60]}" if hint else "Heard you", "Stylist is thinking..."], now, 30)
            elif kind == "error":
                log.error("wake word listener: %s", payload)
                self.compositor.set_message(["Voice trigger off", str(payload)[:60]], now, 8)
                self.wake = None
        r = self.stylist.poll()
        if r is None:
            return
        if r.error:
            log.error("stylist failed: %s", r.error)
            self.compositor.set_message(["Stylist unavailable", r.error[:60]], now, 8)
            return
        self.set_pair(r.top_index, r.bottom_index, now)
        self.last_gesture_t = now
        lines = [r.question[:70], self.gallery.current.name]
        for sh in r.shopping:
            price = f" ~${sh['price_usd']}" if sh.get("price_usd") not in (None, "") else ""
            lines.append(f"{sh['item']} - {sh['store']}{price}")
        lines.append("Prices approximate - scan to shop")
        from fitcast.stylist import make_qr, share_links
        qr = make_qr(r.shopping[0]["url"], 240) if r.shopping else None
        self.compositor.set_message(lines, now, float(scfg.get("message_s", 30)), qr)
        share_links(r, self.gallery.current.name, scfg)
        log.info("stylist (%s, %.1f s): %s | %s | %s", r.model, r.latency_s, r.question, r.say,
                 "; ".join(f"{x['item']} @ {x['store']} ${x['price_usd']} {x['url']}" for x in r.shopping))
        if scfg.get("speak", True):
            from fitcast.stylist import speak
            speak(r.say)

    def relearn_background(self) -> None:
        """Key b: forget the depth background and learn it again from the next N frames (step out of the beam)."""
        if self.depth_mask is not None:
            self._bg_depth.clear(); self._bg_conf.clear()
            from fitcast.depthmask import DepthBodyMask
            self.depth_mask = DepthBodyMask(self.cfg["depthmask"], self.size, self.size)
            log.info("relearning the depth background: step out of the beam")

    def step(self, frame: np.ndarray, ts: float, now: float) -> None:
        t0 = time.perf_counter()
        if self.sensor_kind == "iphone":
            track = self._step_sensor_track(frame, ts)
        else:
            track = self.tracker.process(frame, ts)
        track = self.smoother.update(track)
        dump_file = Path("out/dump.now")
        if self.frames % 10 == 0 and dump_file.exists():          # `touch out/dump.now`: 15 s of paired frames to out/turn/
            dump_file.unlink(missing_ok=True)
            self._dump_until = now + float(self.cfg["logging"].get("dump_s", 15))
            self._dump_n = 0
            Path("out/turn").mkdir(parents=True, exist_ok=True)
            for f in Path("out/turn").glob("*.jpg"):
                f.unlink()
            log.info("frame dump started (%d s)", int(self.cfg["logging"].get("dump_s", 15)))
        dumping = now < self._dump_until and self.frames % int(self.cfg["logging"].get("dump_every_frames", 6)) == 0
        if self.frames % int(self.cfg["logging"].get("debug_jpeg_every_frames", 45)) == 0 or dumping:
            dbg = frame.copy()
            if not track.ok:
                cv2.putText(dbg, "no person tracked", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite("out/live_debug.jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
            elif track.clothes_mask is not None:
                dbg[track.clothes_mask > 0] = (0.55 * dbg[track.clothes_mask > 0] + np.array([0, 112, 0])).astype(np.uint8)
            if track.ok:
                x0, y0, x1, y1 = track.head_box
                cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2)
                for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)):
                    if track.visibility[a] >= self.tracker.min_vis and track.visibility[b] >= self.tracker.min_vis:
                        cv2.line(dbg, tuple(int(v) for v in track.joints[a, :2]), tuple(int(v) for v in track.joints[b, :2]), (0, 255, 255), 2)
                cv2.putText(dbg, f"{track.facing} sw {track.shoulder_width:.0f} ratio {track.facing_ratio:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imwrite("out/live_debug.jpg", dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if dumping:
                    cv2.imwrite(f"out/turn/{self._dump_n:03d}_cam.jpg", cv2.resize(dbg, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    self._dump_pending = self._dump_n
                    self._dump_n += 1
        if self.frames % int(self.cfg["logging"].get("status_every_frames", 150)) == 0:
            log.info("status: %s facing %s sw %.0f px fps %.1f calib %s", "TRACKED" if track.ok else "no person",
                     track.facing if track.ok else "-", track.shoulder_width, self.fps,
                     "ok" if self.calib.ok else "MISSING")
        if self.frames % 10 == 0 and Path(self.cfg["selfcal"].get("trigger_file", "out/calibrate.now")).exists():
            Path(self.cfg["selfcal"].get("trigger_file", "out/calibrate.now")).unlink(missing_ok=True)
            self._want_selfcal = True
            log.info("self-calibration trigger file seen; runs on the next tracked frame")
        if self._want_selfcal and track.ok:
            self._want_selfcal = False
            self.run_self_calibration(track)
        if self.stylist is not None:
            self._step_stylist(now)
        for ev in self.gestures.update(track):
            self.gesture_log.append((now, ev))
            log.info("gesture: %s", ev)
            self.last_gesture_t = now
            if ev == "next":
                self.set_outfit(self.gallery.index + 1, now)
            elif ev == "prev":
                self.set_outfit(self.gallery.index - 1, now)
            elif ev in ("next_top", "prev_top"):
                self.set_top(self.gallery.index + (1 if ev == "next_top" else -1), now)
            elif ev in ("next_bottom", "prev_bottom"):
                cur = self.gallery.bottom_index if self.gallery.bottom_index >= 0 else self.gallery.index
                self.set_bottom(cur + (1 if ev == "next_bottom" else -1), now)
            elif ev == "toggle_gallery":
                self.gallery.visible = not self.gallery.visible
            elif ev == "calibrate":
                self.run_self_calibration(track)
        self.sizing.update(track)
        self.gallery.size_text, self.gallery.measure_text = self.sizing.recommendation(self.gallery.current.size_table)
        if self.args.demo:
            if now - self.last_gesture_t > self.cfg["demo"]["idle_s"] and now - self.last_cycle_t > self.cfg["demo"]["auto_cycle_s"]:
                self.set_outfit(self.gallery.index + 1, now)
                self.last_cycle_t = now
        t1 = time.perf_counter()
        rgba = self.renderer.draw(track, self.gallery.current)
        comp = self.compositor.compose(rgba, track, self.gallery, now)
        if self._frame_stats is not None:
            self._frame_stats.append(f"{now:.3f},{int(track.ok)},{track.facing if track.ok else ''},{track.shoulder_width:.0f},"
                               f"{int(cv2.countNonZero(track.clothes_mask)) if track.ok else 0},{self.compositor.lit_px},"
                               f"{int(self.tracker.fallback_active)},{self.gallery.current.name}\n")
            if len(self._frame_stats) >= 150:
                with open(self.cfg["logging"].get("frame_stats_file", "out/frame_stats.csv"), "a") as f:
                    f.writelines(self._frame_stats)
                self._frame_stats.clear()
        if self.show_skeleton:
            draw_skeleton_for_projector(comp, track, self.tracker.min_vis)
        if track.ok:
            from fitcast import geometry as _geo
            t_px = _geo.torso_length(track.joints)
            self._torso_ema = t_px if self._torso_ema is None else 0.8 * self._torso_ema + 0.2 * t_px
        shift = self.calib.depth_shift(self._torso_ema)
        self.depth_shift = shift
        if self.sensor_kind == "iphone" and self.calib3d is not None:
            if self.proj3d is None:
                from fitcast.render3d import DepthMeshProjector
                c = self.calib3d
                self.proj3d = DepthMeshProjector(self.cfg["render3d"], self.size, tuple(c.proj_size), np.asarray(c.K_cam),
                                                 np.asarray(c.K_p), np.asarray(c.dist_p), np.asarray(c.R), np.asarray(c.t))
                self._proj_cache = np.zeros((int(c.proj_size[1]), int(c.proj_size[0]), 3), np.uint8)
                log.info("depth-correct GPU projection active")
            if track.ok and track.depth is not None:
                if track.K_cam is not None and not self._K_warned:
                    dev = float(np.abs(np.asarray(track.K_cam) - np.asarray(self.calib3d.K_cam)).max())
                    if dev > float(self.cfg["render3d"].get("k_drift_warn_px", 5.0)):
                        log.warning("live camera intrinsics differ from calibration3d by %.1f px (focus/zoom changed?): "
                                    "recalibrate with scripts/calibrate3d.py", dev)
                        self._K_warned = True
                proj = self.proj3d.render(comp, self._depth_for_render(track))
                np.copyto(self._proj_cache, proj)
            else:
                # tracking lost: fade exactly what was last projected (projector space); never the 2D mapping
                proj = cv2.convertScaleAbs(self._proj_cache, alpha=self.compositor.fade, beta=0)
        else:
            split_y = None
            if track.ok and track.visibility[23] >= self.tracker.min_vis and track.visibility[24] >= self.tracker.min_vis:
                split_y = float(0.5 * (track.joints[23, 1] + track.joints[24, 1]))
            proj = self.calib.to_projector(comp, shift, torso_px=self._torso_ema, split_y=split_y)
            gl = self.compositor.gallery_layer
            if gl is not None and self.calib.ok:
                # gallery on the wall: calibration-time plane, no depth shift; combined by max (disjoint regions)
                if self._proj_person is None or self._proj_person.shape != proj.shape:
                    self._proj_person = np.zeros_like(proj)
                np.copyto(self._proj_person, proj)
                pg = self.calib.to_projector(gl)
                cv2.max(self._proj_person, pg, dst=self._proj_person)
                proj = self._proj_person
        t2 = time.perf_counter()
        self.render_ms = (t2 - t1) * 1e3
        self.display.show_projector(proj)
        if self._dump_pending >= 0:
            cv2.imwrite(f"out/turn/{self._dump_pending:03d}_proj.jpg", cv2.resize(proj, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._dump_pending = -1
        self.latency_ms = (time.perf_counter() - ts) * 1e3
        if self.predict_auto:
            # predict forward by the measured capture->display latency (clamped), per spec 4.3
            self._lat_ema = 0.9 * self._lat_ema + 0.1 * min(self.latency_ms, self.predict_max_ms)
            self.smoother.set_predict_ms(self._lat_ema)
        tm = self.tracker.timings
        self._stats["pose"].append(tm.get("pose_ms", 0.0))
        self._stats["seg"].append(tm.get("seg_ms", 0.0))
        self._stats["render"].append(self.render_ms)
        if self.show_debug or self.writer is not None:
            info = {"fps": self.fps, "latency_ms": self.latency_ms, "pose_ms": tm.get("pose_ms", 0.0),
                    "seg_ms": tm.get("seg_ms", 0.0), "render_ms": self.render_ms,
                    "ratio": self.tracker.facing.ratio, "mask_mode": self.tracker.mask_mode,
                    "fallback": self.tracker.fallback_active, "outfit": self.gallery.current.name,
                    "calib": (f"ok {self.calib.data.reproj_error_px:.1f}px depth({self.depth_shift[0]:+.0f},{self.depth_shift[1]:+.0f})"
                              if self.calib.ok else "MISSING"),
                    "extra": f"frame {self.frames}" + (
                        f"  sensor {self.source.stats.get('fps', 0):.0f}fps lat {self.source.stats.get('capture_latency_ms') or 0:.0f}ms"
                        f"  depthmask {'ready' if self.depth_mask is not None and self.depth_mask.ready else 'learning'}"
                        f"  3D {'on' if self.proj3d is not None else 'off'}" if self.sensor_kind == "iphone" else "")}
            want_comp = self.writer is not None or bool(self.cfg["display"].get("debug_show_composite", False))
            dbg = draw_debug(frame, track, info, self.tracker.min_vis,
                             composite=comp if want_comp else None, show_overlay=self.show_overlay,
                             mask_tint=bool(self.cfg["display"].get("mask_tint", True)))
            if self.writer is not None:
                self.writer.write(dbg)
            if self.show_debug and not self.args.no_window and self.frames % int(self.cfg["display"]["debug_every_n"]) == 0:
                s = float(self.cfg["display"]["debug_scale"])
                if s != 1.0:
                    dbg = cv2.resize(dbg, None, fx=s, fy=s)
                self.display.show_debug(dbg)

    def shutdown(self) -> None:
        self.source.stop()
        if self.writer is not None:
            self.writer.release()
        self.tracker.close()
        self.display.close()

    def report(self, elapsed: float) -> None:
        if not self.frames:
            log.error("no frames processed")
            return
        st = {k: (float(np.mean(v)) if v else 0.0) for k, v in self._stats.items()}
        loop_ms = st["loop"] * 1e3
        log.info("processed %d frames in %.1f s: achieved %.1f fps | mean pose %.1f ms, seg %.1f ms (every %d), "
                 "render+compose+warp %.1f ms | loop %.1f ms -> capacity %.1f fps | last latency %.0f ms",
                 self.frames, elapsed, self.frames / elapsed, st["pose"], st["seg"], self.tracker.seg_every,
                 st["render"], loop_ms, 1000.0 / loop_ms if loop_ms else 0.0, self.latency_ms)
        if self.gesture_log:
            log.info("gestures: %s", ", ".join(f"{ev}@{t:.2f}" for t, ev in self.gesture_log))
        if self.wake is not None:
            self.wake.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(level=(args.log_level or cfg.get("logging", {}).get("level", "INFO")).upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.list_cameras:
        list_cameras()
        return 0
    try:
        app = App(cfg, args)
    except RuntimeError as e:
        log.error("%s", e)
        return 2
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
