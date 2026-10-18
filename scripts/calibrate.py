#!/usr/bin/env python
"""Camera <-> projector calibration tool (writes calibration.yaml).

  --simulate   headless: fake rig with a known homography, full procedure, saves --out.
  (default)    live: camera + fullscreen projector window + operator debug window.

Live steps: dark frame -> marker frame -> detect (or click by hand, `m`) -> solve ->
gallery rectangle (two clicks) -> verification grid -> save (`s`). `r` restarts, `q` quits.
Keep everyone out of the projection while calibrating.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitcast.calibration import (Calibration, SimulatedRig, black_frame,  # noqa: E402
                                 capture_pattern, detect_markers, make_sim_homography,
                                 marker_points, render_markers, run_calibration, verify_grid)

log = logging.getLogger("calibrate")

SIM_MAX_REPROJ_PX = 1.0     # --simulate acceptance gate (spec)

# Operator overlay cosmetics (not tunables).
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE, LINE_H, POINT_R, PREVIEW_THICKNESS = 0.6, 24, 6, 4
TEXT_COLOR, POINT_COLOR, RECT_COLOR = (255, 255, 255), (0, 255, 0), (255, 160, 0)
ENTER_KEYS = (13, 10, 32)   # Return (macOS), Enter (Linux), space
NO_KEY = -1


class Restart(Exception):
    """Operator pressed `r`."""


class Quit(Exception):
    """Operator pressed `q`."""


# --- simulate ------------------------------------------------------------------------------

def simulate(config: dict, out: str) -> int:
    """Run the full procedure against a SimulatedRig; return a process exit code."""
    cfg = config["calibration"]
    cam = (int(config["capture"]["width"]), int(config["capture"]["height"]))
    proj = (int(cfg["projector_width"]), int(cfg["projector_height"]))
    H_true = make_sim_homography(cam, proj)     # ~3 deg rotation, 80% fill, shift, keystone
    rig = SimulatedRig(cfg, cam, H_true)
    print(f"simulate: camera {cam[0]}x{cam[1]} -> projector {proj[0]}x{proj[1]}, "
          f"markers {cfg['marker_grid'][0]}x{cfg['marker_grid'][1]}, "
          f"noise sigma {rig.noise_sigma}, ambient {rig.ambient}")
    cal = run_calibration(cfg, cam, rig.capture, rig.show)
    corners = np.array([[0, 0], [cam[0] - 1, 0], [0, cam[1] - 1], [cam[0] - 1, cam[1] - 1]], float)
    truth = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    dev = float(np.linalg.norm(cal.project_points(corners) - truth, axis=1).max())
    print(f"recovered H vs H_true: max {dev:.3f} projector px over the camera frame corners")
    cal.data.gallery_rect = [int(v) for v in config["compositor"]["gallery"]["rect"]]
    cal.data.simulated = True
    cal.save(out)
    back = Calibration(cfg, cam)
    if not back.load(out, allow_simulated=True) or not np.allclose(back.H, cal.H):
        print(f"FAIL: {out} did not load back with the same H")
        return 1
    print(f"saved {out}: gallery_rect {back.data.gallery_rect}, verified={back.data.verified}, "
          f"reproj {back.data.reproj_error_px:.3f} px")
    return 0 if cal.data.verified and cal.data.reproj_error_px < SIM_MAX_REPROJ_PX else 1


# --- live ----------------------------------------------------------------------------------

class LiveRig:
    """Real camera + fullscreen projector window + operator debug window. All GUI lives here;
    `show`/`capture` match the capture_fn/show_fn contract of fitcast.calibration."""

    def __init__(self, config: dict, source, proj_xy: tuple[int, int]) -> None:
        cap_cfg, disp = config["capture"], config["display"]
        self.cam_size = (int(cap_cfg["width"]), int(cap_cfg["height"]))
        self.interp = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}[cap_cfg["resize_interpolation"]]
        from fitcast.capture import parse_source
        src = parse_source(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {source!r}. Fix: check --source; on macOS allow "
                               "your terminal app in System Settings > Privacy & Security > Camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])
        self.proj_win, self.dbg_win = disp["projector_window"], disp["debug_window"]
        self.dbg_scale = float(disp["debug_scale"])
        # same placement as fitcast.display: borderless NSWindow on the projector screen (OpenCV's own
        # fullscreen only ever covers the MAIN screen on macOS)
        from fitcast.display import cocoa_place_borderless, detect_displays
        cv2.namedWindow(self.proj_win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.proj_win, int(proj_xy[0]), int(proj_xy[1]))
        cv2.imshow(self.proj_win, np.zeros((int(config["calibration"]["projector_height"]),
                                           int(config["calibration"]["projector_width"]), 3), np.uint8))
        cv2.waitKey(1)
        second = next((d for d in detect_displays() if not d["main"]), None)
        placed = cocoa_place_borderless(self.proj_win, second["cocoa_index"]) if second else None
        if placed is None:
            print("WARNING: no second display / AppKit placement failed; projector window is on the main screen")
            cv2.setWindowProperty(self.proj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            print(f"projector window on '{second['name']}' frame {placed}")
        cv2.namedWindow(self.dbg_win, cv2.WINDOW_AUTOSIZE)   # 1:1 so click coords = image coords
        cv2.setMouseCallback(self.dbg_win, self._on_mouse)
        self.lines: list[str] = []
        self.points: np.ndarray | None = None   # overlay points, camera px
        self.rect: list[int] | None = None      # overlay rect, camera px
        self.clicks: list[tuple[float, float]] = []
        self.key = NO_KEY
        self._warned_size = False

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x / self.dbg_scale, y / self.dbg_scale))

    def pump(self) -> None:
        """Run the GUI event loop once; `q`/`r` raise, any other key is kept for take_key()."""
        k = cv2.waitKey(1)
        if k < 0:
            return
        k &= 0xFF
        if k == ord("q"):
            raise Quit
        if k == ord("r"):
            raise Restart
        self.key = k

    def take_key(self) -> int:
        """Last key pressed since the previous call, or NO_KEY."""
        k, self.key = self.key, NO_KEY
        return k

    def take_click(self) -> tuple[float, float] | None:
        """Oldest pending left click in camera px, or None."""
        return self.clicks.pop(0) if self.clicks else None

    def reset_input(self) -> None:
        """Drop stale clicks and keys before a step that reads them."""
        self.clicks.clear()
        self.key = NO_KEY

    def status(self, *lines: str) -> None:
        """Set the debug-window instructions (also echoed to the terminal when they change)."""
        if list(lines) != self.lines:
            self.lines = list(lines)
            print(" | ".join(lines))

    def show(self, frame: np.ndarray) -> None:
        """Put a frame on the projector."""
        cv2.imshow(self.proj_win, frame)
        self.pump()

    def capture(self) -> np.ndarray:
        """Read one camera frame in camera space and refresh the debug window."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("camera read failed. Fix: check the camera connection / macOS camera "
                               "permission (System Settings > Privacy & Security > Camera)")
        if (frame.shape[1], frame.shape[0]) != self.cam_size:
            if not self._warned_size:
                log.warning("camera delivers %dx%d, resizing to camera space %dx%d",
                            frame.shape[1], frame.shape[0], *self.cam_size)
                self._warned_size = True
            frame = cv2.resize(frame, self.cam_size, interpolation=self.interp)
        self._draw(frame)
        self.pump()
        return frame

    def _draw(self, frame: np.ndarray) -> None:
        s = self.dbg_scale
        vis = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if self.points is not None:
            for i, (x, y) in enumerate(self.points):
                c = (round(x * s), round(y * s))
                cv2.circle(vis, c, POINT_R, POINT_COLOR, 2, cv2.LINE_AA)
                cv2.putText(vis, str(i + 1), (c[0] + POINT_R, c[1] - POINT_R), FONT, FONT_SCALE,
                            POINT_COLOR, 1, cv2.LINE_AA)
        if self.rect is not None:
            x0, y0, x1, y1 = (round(v * s) for v in self.rect)
            cv2.rectangle(vis, (x0, y0), (x1, y1), RECT_COLOR, 2)
        for i, line in enumerate(self.lines):
            org = (10, LINE_H * (i + 1))
            cv2.putText(vis, line, org, FONT, FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, org, FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.imshow(self.dbg_win, vis)

    def close(self) -> None:
        """Release the camera and close all windows."""
        self.cap.release()
        cv2.destroyAllWindows()


def wait_forever(rig: LiveRig) -> None:
    """Keep the GUI alive until the operator presses r (Restart) or q (Quit)."""
    while True:
        rig.capture()


def click_markers(rig: LiveRig, cfg: dict, pts: np.ndarray) -> np.ndarray:
    """Manual fallback: light one marker at a time; the operator clicks each in the debug
    window in grid order. `u` undoes the last click. Returns (N, 2) camera px."""
    cols = int(cfg["marker_grid"][0])
    clicked: list[tuple[float, float]] = []
    lit = None
    rig.reset_input()
    while len(clicked) < len(pts):
        i = len(clicked)
        if i != lit:
            active = np.zeros(len(pts), bool)
            active[i] = True
            rig.show(render_markers(cfg, pts, active))
            rig.status(f"Step 3/6 manual: click the lit marker {i + 1}/{len(pts)} "
                       f"(row {i // cols + 1}, col {i % cols + 1})", "u undo   r restart   q quit")
            lit = i
        rig.points = np.array(clicked) if clicked else None
        rig.capture()
        k, c = rig.take_key(), rig.take_click()
        rig.clicks.clear()
        if k == ord("u") and clicked:
            clicked.pop()
        elif c is not None:
            clicked.append(c)
    rig.show(black_frame(cfg))
    return np.array(clicked, np.float64)


def _rect_from(corners: list[tuple[float, float]], w: int, h: int) -> list[int] | None:
    (ax, ay), (bx, by) = corners
    x0, x1 = sorted((int(round(min(max(ax, 0), w - 1))), int(round(min(max(bx, 0), w - 1)))))
    y0, y1 = sorted((int(round(min(max(ay, 0), h - 1))), int(round(min(max(by, 0), h - 1)))))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def pick_gallery_rect(rig: LiveRig, cal: Calibration, default: list[int] | None) -> list[int] | None:
    """Operator clicks two opposite corners of the gallery area (wall beside the person) in the
    debug window; the rect is previewed on the wall through H. ENTER with no clicks keeps
    `default`. Returns x0, y0, x1, y1 camera px."""
    w, h = rig.cam_size
    canvas = np.zeros((h, w, 3), np.uint8)
    corners: list[tuple[float, float]] = []
    previewed: object = ()
    rig.points = None
    rig.reset_input()
    while True:
        rect = _rect_from(corners, w, h) if len(corners) == 2 else None
        if len(corners) == 2 and rect is None:
            corners.pop()           # degenerate (zero width/height): take the click again
        current = rect if rect is not None else (default if not corners else None)
        if current != previewed:
            canvas[:] = 0
            if current is not None:
                cv2.rectangle(canvas, tuple(current[:2]), tuple(current[2:]), (255, 255, 255),
                              PREVIEW_THICKNESS)
            rig.show(cal.to_projector(canvas))
            previewed = current
        rig.rect = current
        if not corners:
            rig.status("Step 4/6: click two opposite corners of the gallery area (wall beside the person)",
                       f"ENTER keep {default}   r restart   q quit")
        elif len(corners) == 1:
            rig.status("Step 4/6: click the opposite corner", "u undo   r restart   q quit")
        else:
            rig.status(f"Step 4/6: gallery {rect}", "ENTER accept   u undo   r restart   q quit")
        rig.points = np.array(corners) if corners else None
        rig.capture()
        c = rig.take_click()
        if c is not None and len(corners) < 2:
            corners.append(c)
        k = rig.take_key()
        if k == ord("u") and corners:
            corners.pop()
        elif k in ENTER_KEYS and len(corners) != 1:
            rig.points = None
            rig.show(black_frame(cal.cfg))
            return rect if rect is not None else default


def _dump_debug(lit: np.ndarray, dark: np.ndarray, cam: np.ndarray | None, path: str) -> None:
    """Write lit frame (+ numbered detections) and dark frame side by side for remote inspection."""
    vis = lit.copy()
    if cam is not None:
        for i, (x, y) in enumerate(cam):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), 2)
            cv2.putText(vis, str(i), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, np.hstack([vis, dark]))


def live_session(rig: LiveRig, config: dict, out: str, auto: bool = False) -> None:
    """One pass of the live steps (a)-(g); returns after saving. Raises Restart/Quit.
    `auto`: no keyboard/mouse at all: accept detections, gallery rect from config, save immediately;
    detection failure writes out/calib_debug.jpg and raises RuntimeError."""
    cfg = config["calibration"]
    rows = int(cfg["marker_grid"][1])
    pts = marker_points(cfg)
    settle = int(cfg["settle_frames"])
    rig.points, rig.rect = None, None
    # (a) dark frame, (b) marker frame
    rig.status("Step 1/6: capturing dark frame (keep everyone out of the projection)",
               "r restart   q quit")
    dark = capture_pattern(black_frame(cfg), rig.capture, rig.show, settle)
    rig.status("Step 2/6: capturing markers", "r restart   q quit")
    lit = capture_pattern(render_markers(cfg, pts), rig.capture, rig.show, settle)
    # (c) detect, confirm, or click by hand
    cam = detect_markers(lit, dark, cfg, len(pts), rows)
    _dump_debug(lit, dark, cam, "out/calib_debug.jpg")
    proj_pts = pts
    if auto:
        if True:
            # robust path: light one marker at a time, one disc per capture (ambient light, bystanders)
            from fitcast.calibration import capture_median, detect_one_marker
            dark = capture_median(black_frame(cfg), rig.capture, rig.show, settle, int(cfg["median_frames"]))
            found, used = [], []
            for i in range(len(pts)):
                active = np.zeros(len(pts), bool); active[i] = True
                lit_i = capture_median(render_markers(cfg, pts, active), rig.capture, rig.show, settle,
                                       int(cfg["median_frames"]))
                c = detect_one_marker(lit_i, dark, cfg)
                if c is not None:
                    found.append(c); used.append(i)
            print(f"sequential detection: {len(found)}/{len(pts)} markers found (missing: "
                  f"{[i for i in range(len(pts)) if i not in used] or 'none'})")
            if len(found) < 4:
                raise RuntimeError("fewer than 4 markers detected (see out/calib_debug.jpg): step out of the beam, "
                                   "point the camera at the whole projection, or run without --auto and click them")
            cam, proj_pts = np.array(found), pts[used]
            _dump_debug(lit, dark, cam, "out/calib_debug.jpg")
        else:
            print(f"detected {len(cam)} markers automatically (out/calib_debug.jpg)")
        rig.points = cam
    elif cam is not None:
        rig.points = cam
        rig.status(f"Step 3/6: detected {len(cam)} markers (numbered in grid order)",
                   "ENTER accept   m click by hand   r restart   q quit")
        rig.reset_input()
        while True:
            rig.capture()
            k = rig.take_key()
            if k in ENTER_KEYS:
                break
            if k == ord("m"):
                cam = None
                break
    if cam is None:
        cam = click_markers(rig, cfg, pts)
    rig.points = cam
    # (d) solve
    cal = Calibration(cfg, rig.cam_size)
    try:
        err = cal.set_homography(cam, proj_pts)
    except (RuntimeError, ValueError) as e:
        rig.status(f"solve failed: {e}", "r restart   q quit")
        wait_forever(rig)
    print(f"reprojection error: {err:.3f} px (RMS, projector px)")
    # (e) gallery rectangle
    default = [int(v) for v in config["compositor"]["gallery"]["rect"]]
    cal.data.gallery_rect = default if auto else pick_gallery_rect(rig, cal, default)
    rig.rect = cal.data.gallery_rect
    # (f) verification grid
    rig.status("Step 5/6: projecting verification grid", "r restart   q quit")
    passed, max_err = verify_grid(cal, rig.capture, rig.show, cfg)
    verdict = (f"VERIFY {'PASS' if passed else 'FAIL'}: max error {max_err:.2f} px "
               f"(tolerance {cfg['verify_tolerance_px']} px)")
    print(verdict)
    # (g) save
    rig.status(f"Step 6/6: reprojection {err:.2f} px, {verdict}",
               "s save   r restart   q quit" + ("" if passed else "   (saving an unverified calibration)"))
    if auto and (not passed or err > float(cfg["max_reproj_px"])):
        raise RuntimeError(f"not saving: reprojection {err:.1f} px (limit {cfg['max_reproj_px']}), verify "
                           f"{'passed' if passed else 'FAILED'} (max {max_err:.1f} px). Nobody in the beam? Camera sees "
                           f"the whole projection? See out/calib_debug.jpg")
    if not auto:
        rig.reset_input()
        while rig.take_key() != ord("s"):
            rig.capture()
    cal.save(out)
    print(f"saved {out}: reproj {err:.3f} px, verified={passed}, gallery_rect {cal.data.gallery_rect}")


def preview(config: dict, args: argparse.Namespace, seconds: float) -> int:
    """Aiming aid: project all markers and show the camera feed on the laptop with instructions."""
    import time
    disp = config["display"]
    px, py = args.projector_x or 0, args.projector_y or 0
    source = args.source if args.source is not None else config["capture"]["source"]
    rig = LiveRig(config, source, (px, py))
    cfg = config["calibration"]
    pattern = render_markers(cfg, marker_points(cfg))
    cv2.rectangle(pattern, (4, 4), (int(cfg["projector_width"]) - 5, int(cfg["projector_height"]) - 5), (255, 255, 255), 6)
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            rig.status(f"AIM: all 9 dots + the white border inside this view, nobody in frame  ({seconds - (time.time() - t0):.0f}s)",
                       "camera must not move after this; q quit")
            rig.show(pattern)
            rig.capture()
            if rig.take_key() == ord("q"):
                break
    finally:
        rig.close()
    return 0


def live(config: dict, args: argparse.Namespace, out: str) -> int:
    """Interactive calibration with a real camera and projector; return an exit code."""
    disp = config["display"]
    px, py = args.projector_x, args.projector_y
    if px is None or py is None:
        from fitcast.display import detect_displays
        second = next((s for s in detect_displays() if not s["main"]), None)
        if px is None:
            px = disp.get("projector_x") if disp.get("projector_x") is not None else (second["x"] if second else 0)
        if py is None:
            py = disp.get("projector_y") if disp.get("projector_y") is not None else (second["y"] if second else 0)
    source = args.source if args.source is not None else config["capture"]["source"]
    rig = LiveRig(config, source, (px, py))
    try:
        while True:
            try:
                live_session(rig, config, out, auto=args.auto)
                return 0
            except RuntimeError as e:
                print(f"CALIBRATION FAILED: {e}")
                return 2
            except Restart:
                print("restart")
    except Quit:
        print("quit without saving")
        return 1
    finally:
        rig.close()


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None, help="camera index or path (default: capture.source)")
    ap.add_argument("--projector-x", type=int, default=None,
                    help="projector display origin x (default: display.projector_x, else 0)")
    ap.add_argument("--projector-y", type=int, default=None,
                    help="projector display origin y (default: display.projector_y, else 0)")
    ap.add_argument("--simulate", action="store_true", help="headless run against a fake rig")
    ap.add_argument("--preview", type=float, default=0.0, metavar="SECONDS",
                    help="aiming aid: project the dots and show the camera feed for this many seconds, then exit")
    ap.add_argument("--auto", action="store_true",
                    help="live mode without any keyboard/mouse: auto-detect, config gallery rect, save at once")
    ap.add_argument("--out", default=None,
                    help="output file (default: calibration.file; --simulate defaults to calibration.sim_file "
                         "so a fake homography never shadows a real one)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)    # keep prints and log lines in order
    config = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=config["logging"]["level"], format="%(levelname)s %(name)s: %(message)s")
    out = args.out or (config["calibration"]["sim_file"] if args.simulate else config["calibration"]["file"])
    if args.preview > 0:
        return preview(config, args, args.preview)
    return simulate(config, out) if args.simulate else live(config, args, out)


if __name__ == "__main__":
    sys.exit(main())
