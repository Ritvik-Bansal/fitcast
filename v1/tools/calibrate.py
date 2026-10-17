"""Camera -> projector calibration.

Projects a 9x6 checkerboard fullscreen on the projector. Hold a white foam
board flat at the taped standing spot, facing the projector, so the whole board
pattern lands on it. The tool finds the corners in the camera image, solves the
camera->projector homography, then projects a green dot on your nose for 10 s so
you can eyeball the alignment.

Keys (in either window):
  space   capture the board and solve
  r       retry (back to the checkerboard)
  q       save calib.json and quit (after a successful solve)
  esc     quit without saving
  + / -   grow / shrink the projected board so it fits on the foam board
  arrows  move the projected board
  g       fallback: dot-grid calibration (one bright dot at a time; keep the board still)

    python tools/calibrate.py
    python tools/calibrate.py --main-display     # test on the laptop screen without a projector
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tryon.calibration import (Calibration, checkerboard, dot_centroid, dot_grid, find_corners,  # noqa: E402
                               order_like, save_calibration, solve_homography)
from tryon.capture import Camera  # noqa: E402
from tryon.config import load_config  # noqa: E402
from tryon.display import ProjectorWindow, cv2_key_name  # noqa: E402
from tryon.pose import PoseTracker  # noqa: E402

PREVIEW = "Fitcast calibration (camera)"
VERIFY_S = 10.0
SETTLE_S = 0.35


def put(img, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


class Calibrator:
    def __init__(self, args):
        overrides = {}
        if args.main_display:
            overrides["projector_allow_main_display"] = True
        if args.camera is not None:
            overrides["camera_index"] = args.camera
        self.cfg = cfg = load_config(args.config, overrides)
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
        self.pattern = (cols, rows)
        self.proj_size = (cfg.projector_width, cfg.projector_height)
        self.center = [self.proj_size[0] / 2, self.proj_size[1] / 2]
        self.square = self.proj_size[1] * 0.055
        self.out_path = cfg.path(args.out or cfg.calib_path)
        self.camera = Camera(cfg).start()
        self.projector = ProjectorWindow(cfg)
        if not self.projector.ok:
            self.camera.stop()
            sys.exit(f"[calibrate] {self.projector.error}")
        print(f"[calibrate] projector window via {self.projector.backend} on {self.projector.display}")
        cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW, 960, int(960 * cfg.camera_height / cfg.camera_width))
        self.pose = PoseTracker(cfg)
        self.mode = "board"          # board | verify | review | dots
        self.H: np.ndarray | None = None
        self.err = 0.0
        self.method = "checkerboard"
        self.msg = "Hold the white board at the taped spot, then press SPACE"
        self.verify_until = 0.0
        self.last_idx = -1
        self.frame_n = 0
        self.fast_found = False
        self.dots: list = []

    # -- projector content ---------------------------------------------------
    def board_image(self):
        return checkerboard(self.proj_size, self.pattern, tuple(self.center), self.square)

    def show_projector(self, img):
        self.projector.show(img)

    # -- solving --------------------------------------------------------------
    def capture_board(self, frame) -> None:
        img, proj_pts = self.board_image()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_pts = find_corners(gray, self.pattern)
        if cam_pts is None:
            self.msg = "Board not found. Fit it on the foam board (+/- and arrows), reduce room light, retry SPACE"
            return
        cam_pts = order_like(cam_pts, proj_pts)
        H, err = solve_homography(cam_pts, proj_pts)
        if H is None:
            self.msg = "Homography failed; retry"
            return
        self.H, self.err, self.method = H, err, "checkerboard"
        self.start_verify()

    def start_verify(self) -> None:
        self.mode = "verify"
        self.verify_until = time.monotonic() + VERIFY_S
        self.msg = f"Solved (error {self.err:.2f} px). Step in: the green dot should sit on your nose"
        print(f"[calibrate] homography solved, reprojection error {self.err:.2f} px")

    # -- dot grid fallback ---------------------------------------------------------
    def start_dots(self) -> None:
        span = ((self.pattern[0] + 1) * self.square, (self.pattern[1] + 1) * self.square)
        self.dot_targets = dot_grid(self.proj_size, tuple(self.center), span, (5, 4))
        self.dots = []
        self.H = None
        self.dot_i = 0
        self.dot_phase = "dark"
        self.dot_t = time.monotonic()
        self.dark = None
        self.mode = "dots"
        self.msg = "Dot grid: keep the board still"

    def step_dots(self, frame, ts) -> np.ndarray:
        pw, ph = self.proj_size
        img = np.zeros((ph, pw, 3), np.uint8)
        if ts < self.dot_t + SETTLE_S:
            if self.dot_phase == "lit":
                p = self.dot_targets[self.dot_i]
                cv2.circle(img, (int(p[0]), int(p[1])), int(self.square * 0.35), (255, 255, 255), -1, cv2.LINE_AA)
            return img
        if self.dot_phase == "dark":
            self.dark = frame.copy()
            self.dot_phase = "lit"
            self.dot_t = time.monotonic()
            p = self.dot_targets[self.dot_i]
            cv2.circle(img, (int(p[0]), int(p[1])), int(self.square * 0.35), (255, 255, 255), -1, cv2.LINE_AA)
            return img
        c = dot_centroid(self.dark, frame)
        if c is not None:
            self.dots.append((c, self.dot_targets[self.dot_i]))
        self.dot_i += 1
        self.dot_phase = "dark"
        self.dot_t = time.monotonic()
        if self.dot_i >= len(self.dot_targets):
            if len(self.dots) >= 6:
                cam = np.array([d[0] for d in self.dots])
                proj = np.array([d[1] for d in self.dots])
                H, err = solve_homography(cam, proj)
                if H is not None:
                    self.H, self.err, self.method = H, err, "dots"
                    self.start_verify()
                    return img
            self.mode = "board"
            self.msg = f"Dot grid found only {len(self.dots)} dots; darken the room and retry (g)"
        return img

    # -- main loop -------------------------------------------------------------------
    def run(self) -> int:
        saved = False
        try:
            while True:
                frame, ts, idx = self.camera.wait_for_new(self.last_idx, 0.1)
                if frame is None or idx == self.last_idx:
                    if self.handle_keys(None):
                        break
                    continue
                self.last_idx = idx
                self.frame_n += 1
                preview = frame.copy()
                now = time.monotonic()
                if self.mode == "board":
                    img, _ = self.board_image()
                    self.show_projector(img)
                    if self.frame_n % 3 == 0:
                        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
                        pts = find_corners(small, self.pattern, fast=True)
                        self.fast_found = pts is not None
                        self.fast_pts = pts * 2 if pts is not None else None
                    if self.fast_found and self.fast_pts is not None:
                        cv2.drawChessboardCorners(preview, self.pattern, self.fast_pts.astype(np.float32).reshape(-1, 1, 2), True)
                    put(preview, "board detected - press SPACE" if self.fast_found else "board not detected yet",
                        60, (0, 255, 0) if self.fast_found else (0, 165, 255))
                elif self.mode == "dots":
                    self.show_projector(self.step_dots(frame, ts))
                    put(preview, f"dot {min(self.dot_i + 1, len(self.dot_targets))}/{len(self.dot_targets)}, found {len(self.dots)}", 60)
                elif self.mode in ("verify", "review"):
                    pw, ph = self.proj_size
                    img = np.zeros((ph, pw, 3), np.uint8)
                    res = self.pose.process(frame, ts, idx)
                    if res.present and self.mode == "verify":
                        nose = res.kp["nose"]
                        p = Calibration(self.H, (frame.shape[1], frame.shape[0]), self.proj_size).cam_to_proj([nose])[0]
                        cv2.circle(img, (int(p[0]), int(p[1])), 14, (0, 255, 0), -1, cv2.LINE_AA)
                        cv2.circle(preview, (int(nose[0]), int(nose[1])), 8, (0, 255, 0), 2, cv2.LINE_AA)
                    self.show_projector(img)
                    if self.mode == "verify":
                        left = self.verify_until - now
                        put(preview, f"verifying: {left:4.1f}s  (green dot projected on your nose)", 60, (0, 255, 0))
                        if left <= 0:
                            self.mode = "review"
                            self.msg = "q = save calib.json and quit,  r = retry,  esc = quit without saving"
                put(preview, self.msg, 30)
                hints = {"board": "SPACE capture  +/- size  arrows move  g dot grid  esc abort",
                         "dots": "dot grid running - keep the board still  (r cancel, esc abort)",
                         "verify": "r retry  q save+quit  esc abort",
                         "review": "r retry  q save+quit  esc abort"}
                put(preview, hints[self.mode], preview.shape[0] - 14, (0, 200, 255), 0.5)
                cv2.imshow(PREVIEW, preview)
                if self.frame_n > 5 and cv2.getWindowProperty(PREVIEW, cv2.WND_PROP_VISIBLE) < 1:
                    print("[calibrate] window closed; nothing saved")
                    break
                key = self.handle_keys(frame)
                if key == "save":
                    saved = True
                    break
                if key:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            self.pose.stop()
            self.projector.close()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return 0 if saved else 1

    def handle_keys(self, frame):
        keys = []
        name = cv2_key_name(cv2.waitKeyEx(1))
        if name:
            keys.append(name)
        keys += self.projector.poll_keys()
        for k in keys:
            if k == "esc":
                print("[calibrate] aborted; nothing saved")
                return "quit"
            if k == "q":
                if self.H is None:
                    print("[calibrate] nothing solved yet; press SPACE first (esc to abort)")
                    continue
                cal = Calibration(self.H, (self.cfg.camera_width, self.cfg.camera_height), self.proj_size,
                                  self.err, self.method, window=self.projector.geometry())
                save_calibration(self.out_path, cal)
                print(f"[calibrate] saved {self.out_path} (error {self.err:.2f} px)")
                return "save"
            if k == "r":
                self.mode = "board"
                self.H = None
                self.msg = "Hold the white board at the taped spot, then press SPACE"
            elif k == "space" and self.mode == "board" and frame is not None:
                self.capture_board(frame)
            elif k == "g" and self.mode == "board":
                self.start_dots()
            elif k in ("+", "="):
                self.square = min(self.proj_size[1] / 8, self.square * 1.1)
            elif k in ("-", "_"):
                self.square = max(12, self.square / 1.1)
            elif k in ("left", "right", "up", "down"):
                step = self.square * 0.5
                dx, dy = {"left": (-step, 0), "right": (step, 0), "up": (0, -step), "down": (0, step)}[k]
                self.center[0] = float(np.clip(self.center[0] + dx, 0, self.proj_size[0]))
                self.center[1] = float(np.clip(self.center[1] + dy, 0, self.proj_size[1]))
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pattern", default="9x6", help="inner corners, cols x rows (default 9x6)")
    ap.add_argument("--out", help="where to write the calibration (default: calib_path from config)")
    ap.add_argument("--camera", type=int, help="camera index override")
    ap.add_argument("--main-display", action="store_true", help="allow the projector window on the laptop screen")
    return Calibrator(ap.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
