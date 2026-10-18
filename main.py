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
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from fitcast.calibration import Calibration
from fitcast.capture import FrameSource, list_cameras
from fitcast.compositor import Compositor
from fitcast.display import Display, draw_debug
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
        self.size = (int(cap_cfg["width"]), int(cap_cfg["height"]))
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
        self.calib.load(args.calibration, allow_simulated=self.source.is_file)   # a fake homography is fine for clips, never live
        if self.calib.ok and getattr(self.calib.data, "gallery_rect", None):
            self.compositor.set_gallery_rect(self.calib.data.gallery_rect)
        if self.calib.ok and tcfg.get("roi_from_calibration", True) and not args.no_roi:
            # only track inside the projector footprint (+margin): bystanders beside the rig are ignored
            c = self.calib.unproject_points(np.array([[0, 0], [self.calib.projector_size[0], 0],
                                                      list(self.calib.projector_size), [0, self.calib.projector_size[1]]], float))
            self.tracker.set_roi((c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()),
                                 float(tcfg.get("roi_margin_frac", 0.1)))
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
        log.info("%d outfits loaded: %s", len(outfits), ", ".join(o.id for o in outfits))
        proj = (int(cfg["calibration"]["projector_width"]), int(cfg["calibration"]["projector_height"]))
        self.display = Display(cfg["display"], proj, (args.projector_x, args.projector_y), headless=args.no_window)
        self.show_overlay = True
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
        self.depth_shift = (0.0, 0.0)
        self._last_reopen = 0.0
        self.predict_auto = bool(cfg["smoothing"]["predict_auto"])
        self.predict_max_ms = float(cfg["smoothing"]["predict_max_ms"])
        self._lat_ema = float(cfg["smoothing"]["predict_ms"])

    # -- outfit control -----------------------------------------------------------------------
    def set_outfit(self, idx: int, now: float) -> None:
        g = self.gallery
        g.prev_index, g.index = g.index, idx % len(g.outfits)
        g.changed_at = now
        log.info("outfit -> %s", g.current.name)

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
        elif k in (ord("["), ord("]")):
            self.renderer.vertical_offset_frac += -0.02 if k == ord("[") else 0.02
            log.info("renderer.vertical_offset_frac = %.2f (shoulder widths; put this in config.yaml to keep it)",
                     self.renderer.vertical_offset_frac)
        elif k == ord("r"):
            if self.calib.load(self.args.calibration, allow_simulated=self.source.is_file):
                if self.calib.data.gallery_rect:
                    self.compositor.set_gallery_rect(self.calib.data.gallery_rect)
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
            self.writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), self.source.native_fps, (2 * w, h))
            log.info("recording review video to %s", a.record)
        t_start = time.perf_counter()
        last_seq = -1
        fps_t = t_start
        running = True
        try:
            while running:
                item = self.source.read()
                if item is None or item[2] == last_seq:
                    if a.no_window:
                        break
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

    def step(self, frame: np.ndarray, ts: float, now: float) -> None:
        t0 = time.perf_counter()
        track = self.tracker.process(frame, ts)
        track = self.smoother.update(track)
        for ev in self.gestures.update(track):
            self.gesture_log.append((now, ev))
            self.last_gesture_t = now
            if ev == "next":
                self.set_outfit(self.gallery.index + 1, now)
            elif ev == "prev":
                self.set_outfit(self.gallery.index - 1, now)
            elif ev == "toggle_gallery":
                self.gallery.visible = not self.gallery.visible
        self.sizing.update(track)
        self.gallery.size_text, self.gallery.measure_text = self.sizing.recommendation(self.gallery.current.size_table)
        if self.args.demo:
            if now - self.last_gesture_t > self.cfg["demo"]["idle_s"] and now - self.last_cycle_t > self.cfg["demo"]["auto_cycle_s"]:
                self.set_outfit(self.gallery.index + 1, now)
                self.last_cycle_t = now
        t1 = time.perf_counter()
        rgba = self.renderer.draw(track, self.gallery.current)
        comp = self.compositor.compose(rgba, track, self.gallery, now)
        if track.ok:
            from fitcast import geometry as _geo
            t_px = _geo.torso_length(track.joints)
            self._torso_ema = t_px if self._torso_ema is None else 0.8 * self._torso_ema + 0.2 * t_px
        shift = self.calib.depth_shift(self._torso_ema)
        self.depth_shift = shift
        proj = self.calib.to_projector(comp, shift)
        t2 = time.perf_counter()
        self.render_ms = (t2 - t1) * 1e3
        self.display.show_projector(proj)
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
                    "extra": f"frame {self.frames}"}
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
