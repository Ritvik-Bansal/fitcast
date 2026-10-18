#!/usr/bin/env python
"""3D projector-camera calibration with iPhone LiDAR (writes calibration3d.yaml). docs/BUILD_SPEC_3D.md section 6.

  --simulate        headless: SimulatedRig3D (wall + two boxes, known projector), full procedure, prints the
                    reprojection error and the K_p / R / t recovery error versus truth, saves --out.
  --sensor iphone   live: FitCastSensor stream (colour + LiDAR depth) + fullscreen projector window.

Live: put the non-planar target (boxes / foam boards at angles + wall) at the floor mark, dim the room, keep
everyone out of the projection, keep the rig still. The tool projects white/black + 44 Gray code patterns
(~1 min), solves, then projects the verification pattern (green = box edges, white = 10 cm grid on the
measured surfaces) and saves out/calib3d_verify.jpg (camera view before | after). `q` quits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitcast.calib3d import (DEFAULTS, SimulatedRig3D, load_calibration3d,  # noqa: E402
                             project_to_projector, rotation_error_deg, run_calibration3d, save_calibration3d, sim_truth,
                             verification_pattern, verification_points)

# --simulate acceptance gates (spec)
SIM_MAX_REPROJ_PX = 1.0
SIM_MAX_FOCAL_FRAC = 0.01
SIM_MAX_ROT_DEG = 0.2
SIM_MAX_T_M = 0.01
VERIFY_JPG = "out/calib3d_verify.jpg"


def _sizes(config: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    cam = (int(config["capture"]["width"]), int(config["capture"]["height"]))
    cal = config["calibration"]
    return cam, (int(cal["projector_width"]), int(cal["projector_height"]))


# --- simulate ------------------------------------------------------------------------------

def simulate(config: dict, out: str) -> int:
    """Run the full procedure on SimulatedRig3D; return a process exit code (0 = all gates pass)."""
    cfg = {**DEFAULTS, **config["calibration3d"], "settle_frames": 0}   # the fake rig has no latency to settle
    cam, proj = _sizes(config)
    truth = sim_truth(cam, proj, float(cfg["throw_ratio_guess"]))
    rig = SimulatedRig3D(cfg, cam, proj, **truth)
    Kt = truth["K_p_true"]
    print(f"simulate: camera {cam[0]}x{cam[1]} -> projector {proj[0]}x{proj[1]}; wall 3.2 m, boxes "
          f"{[(round(float(b[0]), 2), round(float(b[1]), 2), b[4]) for b in rig.box_rects]} (x0, x1, z m); "
          f"depth noise {rig.depth_noise_m * 1e3:.0f} mm, ambient {rig.ambient:.0f}, noise sigma {rig.noise_sigma}")
    print(f"truth: f {Kt[0, 0]:.1f} (guess {cfg['throw_ratio_guess'] * proj[0]:.1f}), c ({Kt[0, 2]:.1f}, "
          f"{Kt[1, 2]:.1f}), dist {truth['dist_p_true'].round(4).tolist()}")
    c = run_calibration3d(cfg, cam, proj, rig.capture, rig.show, rig.depth, rig.conf, K_cam=truth["K_cam"])
    c.notes = "SIMULATED (SimulatedRig3D) - not for a live rig; " + c.notes
    K = c.K_p
    focal = abs(K[0, 0] / Kt[0, 0] - 1.0)
    rot = rotation_error_deg(c.R, truth["R_true"])
    terr = float(np.linalg.norm(c.t - truth["t_true"]))
    print(f"K_p: f {K[0, 0]:.2f} vs {Kt[0, 0]:.2f} ({focal:.4%}), c ({K[0, 2]:.2f}, {K[1, 2]:.2f}) vs "
          f"({Kt[0, 2]:.2f}, {Kt[1, 2]:.2f}), dist {c.dist_p.round(4).tolist()}")
    print(f"R: {rot:.4f} deg from truth; t: {c.t.round(4).tolist()} vs {truth['t_true'].round(4).tolist()} "
          f"({terr * 1e3:.2f} mm)")
    edges = verification_points(c, rig.depth(), truth["K_cam"], cfg)[1]
    rvec, _ = cv2.Rodrigues(truth["R_true"])
    true_px, _ = cv2.projectPoints(edges.reshape(-1, 1, 3), rvec, truth["t_true"], Kt, truth["dist_p_true"])
    dev = np.linalg.norm(project_to_projector(c, edges) - true_px.reshape(-1, 2), axis=1)
    print(f"verification edges ({len(edges)} pts): recovered vs true projection mean {dev.mean():.3f} px, "
          f"max {dev.max():.3f} px")
    save_calibration3d(out, c)
    back = load_calibration3d(out)
    same = all(np.allclose(getattr(back, k), getattr(c, k)) for k in ("K_p", "dist_p", "R", "t", "K_cam"))
    print(f"saved {out}: reproj {back.reproj_error_px:.3f} px over {back.n_points} points, round trip "
          f"{'OK' if same else 'MISMATCH'}")
    gates = {"reproj < 1 px": c.reproj_error_px < SIM_MAX_REPROJ_PX, "focal within 1%": focal < SIM_MAX_FOCAL_FRAC,
             "rotation within 0.2 deg": rot < SIM_MAX_ROT_DEG, "translation within 1 cm": terr < SIM_MAX_T_M,
             "yaml round trip": same}
    print("  ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    return 0 if all(gates.values()) else 1


# --- live ----------------------------------------------------------------------------------

class Quit(Exception):
    """Operator pressed `q`."""


class LiveSensorRig:
    """IPhoneSensor (colour + depth) + projector window via fitcast.display.Display (same borderless
    second-screen placement as the main app). capture/show/depth/conf match run_calibration3d."""

    def __init__(self, config: dict, sensor, disp, timeout_s: float) -> None:
        self.sensor, self.disp, self.timeout_s = sensor, disp, float(timeout_s)
        self.dbg_scale = float(config["display"]["debug_scale"])
        self.seq: int | None = None
        self.key: int | None = None           # last key other than q
        self._warned_conf = False

    def pump(self) -> None:
        """Run the GUI event loop once; `q` raises Quit, any other key is kept in self.key."""
        k = self.disp.key(1)
        if k in (-1, 0xFFFF):
            return
        if k & 0xFF == ord("q"):
            raise Quit
        self.key = k

    def show(self, frame: np.ndarray) -> None:
        """Put a frame on the projector."""
        self.disp.show_projector(frame)
        self.pump()

    def capture(self) -> np.ndarray:
        """Block until the sensor delivers a NEW frame (latest-frame-only source) and return it."""
        deadline = time.perf_counter() + self.timeout_s
        while True:
            r = self.sensor.read()
            if r is not None and r[2] != self.seq:
                frame, _, self.seq = r
                self.disp.show_debug(cv2.resize(frame, None, fx=self.dbg_scale, fy=self.dbg_scale,
                                                interpolation=cv2.INTER_AREA))
                self.pump()
                return frame
            if time.perf_counter() > deadline:
                raise RuntimeError(f"no new frame from the iPhone for {self.timeout_s:.0f} s. Fix: open FitCastSensor "
                                   "on the phone, check the USB cable / sensor.port, see docs/BUILD_SPEC_3D.md")
            self.pump()
            time.sleep(0.002)

    def _track(self):
        tr = self.sensor.track()
        if tr is None or tr.depth is None or tr.K_cam is None:
            raise RuntimeError("the iPhone stream has no depth / intrinsics. Fix: FitCastSensor must send the "
                               "`depth` blob and K (LiDAR iPhone, sceneDepth enabled)")
        return tr

    def depth(self) -> np.ndarray:
        """Latest depth (H, W) float32 metres, camera space."""
        return self._track().depth

    def conf(self) -> np.ndarray:
        """Latest ARKit confidence in camera space (IPhoneSensor.last_conf); depth > 0 -> 2 when absent."""
        conf = getattr(self.sensor, "last_conf", None)
        if conf is not None:
            return conf
        if not self._warned_conf:
            print("WARNING: the stream has no depth confidence; every pixel with depth counts as confidence 2")
            self._warned_conf = True
        return np.where(self.depth() > 0, 2, 0).astype(np.uint8)

    def K_cam(self) -> np.ndarray:
        """Camera-space intrinsics of the stream."""
        return np.asarray(self._track().K_cam, np.float64)


def live(config: dict, args: argparse.Namespace, out: str) -> int:
    """Calibrate with the iPhone sensor and the real projector; return an exit code."""
    try:
        from fitcast.sensors.iphone import IPhoneSensor
    except ImportError as e:
        print(f"CALIBRATION FAILED: cannot import fitcast.sensors.iphone ({e}). Fix: build the Mac receiver "
              "(docs/BUILD_SPEC_3D.md section 4) and install its requirements")
        return 2
    from fitcast.display import Display
    cfg = {**DEFAULTS, **config["calibration3d"]}
    settle = int(cfg["settle_frames"])
    cam, proj = _sizes(config)
    sensor = IPhoneSensor(config["sensor"], cam, tracker_cfg=config["tracker"]).start()
    disp = Display(config["display"], proj, (args.projector_x, args.projector_y))
    disp.open_projector(fullscreen=True)
    disp.open_debug()
    rig = LiveSensorRig(config, sensor, disp, float(config["sensor"]["connect_timeout_s"]))
    try:
        print("waiting for the iPhone stream ...")
        rig.capture()
        K_cam = rig.K_cam()
        white = np.full((proj[1], proj[0], 3), 255, np.uint8)
        rig.show(white)
        for _ in range(settle):
            rig.capture()
        before = rig.capture()
        c = run_calibration3d(cfg, cam, proj, rig.capture, rig.show, rig.depth, rig.conf, K_cam=K_cam)
        rig.show(verification_pattern(c, rig.depth(), K_cam, proj, cfg))
        for _ in range(settle):
            rig.capture()
        after = rig.capture()
        Path(VERIFY_JPG).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(VERIFY_JPG, np.hstack([before, after]))
        print(f"wrote {VERIFY_JPG} (camera view: white | verification pattern). Check that the green lines sit on "
              "the box edges and the grid is straight on every face.")
        if c.reproj_error_px > float(cfg["target_reproj_px"]):
            print(f"NOT SAVED: reprojection {c.reproj_error_px:.2f} px > target {cfg['target_reproj_px']} px")
            return 2
        save_calibration3d(out, c)
        print(f"saved {out}: reproj {c.reproj_error_px:.3f} px over {c.n_points} points. "
              "The verification pattern stays up; press any key in the debug window to finish.")
        rig.key = None
        try:
            while rig.key is None:
                rig.capture()
        except Quit:
            pass
        return 0
    except RuntimeError as e:
        print(f"CALIBRATION FAILED: {e}")
        return 2
    except Quit:
        print("quit without saving")
        return 1
    finally:
        sensor.stop()
        disp.close()


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--simulate", action="store_true", help="headless run against SimulatedRig3D")
    ap.add_argument("--sensor", choices=["iphone"], default="iphone", help="live depth sensor (LiDAR needed)")
    ap.add_argument("--projector-x", type=int, default=None, help="projector display origin x (default: auto)")
    ap.add_argument("--projector-y", type=int, default=None, help="projector display origin y (default: auto)")
    ap.add_argument("--out", default=None, help="output file (default: calibration3d.file)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    config = yaml.safe_load(Path(args.config).read_text())
    out = args.out or config["calibration3d"]["file"]
    return simulate(config, out) if args.simulate else live(config, args, out)


if __name__ == "__main__":
    sys.exit(main())
