#!/usr/bin/env python
"""Projection accuracy check (docs/BUILD_SPEC_3D.md section 10): projected body boundary vs the real body, in mm.

  --simulate            headless: SimulatedRig3D wall + a flat person cutout `sim_person_offset_m` in front of it,
                        projector calibrated by run_calibration3d on the standard box rig; the outline is projected
                        through DepthMeshProjector with (a) the TRUE projector model, (b) the RECOVERED calibration,
                        (c) the recovered calibration with its principal point shifted by `sim_shift_px` (metric
                        sanity: the error must rise by about shift * z / f_p). The fake camera always sees the light
                        through the true projector (with the person's projector shadow on the wall).
  --sensor iphone       live: FitCastSensor stream + fullscreen projector. Learn the depth background (step out of
                        the beam), then phase "still" (and "slow turn" with --turn), `--seconds` each.
  --report              print the spec targets (still: median < 10 mm, p95 < 20 mm; slow movement: median < 25 mm)
                        with PASS / FAIL per phase.

What the live number means: every iteration projects the outline of the CURRENT depth mask (no Smoother
prediction) and the next camera frames show whatever the projector shows at their exposure, i.e. the outline of a
mask that is one pipeline latency old. Truth = the DepthBodyMask of the depth that arrived with the lit frame.
So "still" measures the static transfer accuracy (calibration + depth + mask edges), "slow turn" adds the
pipeline latency (motion during capture -> display). A dark frame (projector black) is captured every
`dark_every_n` evaluated frames; after every projector change `calibration.settle_frames` camera frames are
skipped. Lit pixels farther than `max_search_px` from the truth boundary are rejected (stray light, motion edges
against an old dark frame) and counted separately.

SAFETY: the head box of the sensor's TrackResult is blacked out in the outline image as the last step before
projection; when the phone tracks nobody (no head box) the projector shows black.

Writes out/accuracy_<timestamp>.json (out/accuracy_sim_<timestamp>.json for --simulate) and annotated frames
(truth boundary red, detected lit outline green) to out/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fitcast.accuracy import (annotate, boundary_errors_mm, detect_lit_outline, outline_image,  # noqa: E402
                              summarize, truth_boundary_from_depth)
from fitcast.calib3d import DEFAULTS as C3D_DEFAULTS  # noqa: E402
from fitcast.calib3d import SimulatedRig3D, run_calibration3d, sim_truth  # noqa: E402

# `accuracy:` keys (not yet in config.yaml; config values win). Listed in the builder report for config.yaml.
DEFAULTS = {
    "outline_thickness_px": 2,       # band just inside the mask boundary (camera px)
    "outline_color": [255, 255, 255],   # BGR of the projected band (white: strongest channel-max difference)
    "dark_every_n": 10,              # capture a fresh dark frame after this many evaluated lit frames
    "seconds": 10,                   # per phase (live)
    "max_search_px": 50,             # lit pixels farther than this from the truth boundary are rejected
    "min_blob_px": 8,                # thinned lit components smaller than this are noise
    "annotate_frames": 3,            # annotated jpgs written per phase
    "frame_timeout_s": 5.0,          # live: no new sensor frame for this long -> abort the phase
    "out_dir": "out",
    "sim_person_offset_m": 0.8,      # simulate: person plane this far in front of the wall
    "sim_frames": 4,                 # simulate: lit/dark iterations per case
    "sim_shift_px": 15.0,            # simulate: principal-point shift (projector px) of the sanity case
    "sim_frames_per_pattern": 1,     # simulate: Gray code frames averaged per pattern (speed)
    "sim_max_median_mm": 3.0,        # simulate gate: median with the perfect / recovered calibration
    "sim_shift_min_ratio": 3.0,      # simulate gate: shifted median >= this x the recovered median ...
    "sim_shift_min_frac": 0.4,       # ... and >= this fraction of shift * z / f_p
}

# Spec targets (BUILD_SPEC_3D section 10).
TARGETS = {"still": {"median": 10.0, "p95": 20.0}, "turn": {"median": 25.0}}

WALL_Z = 3.2


def acc_cfg(config: dict) -> dict:
    """`accuracy:` section merged over DEFAULTS."""
    return {**DEFAULTS, **(config.get("accuracy") or {})}


def detect_cfg(config: dict) -> dict:
    """Threshold keys for detect_lit_outline: calibration: section (diff_threshold ...) + accuracy overrides."""
    return {**config["calibration"], **acc_cfg(config)}


def _sizes(config: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    cam = (int(config["capture"]["width"]), int(config["capture"]["height"]))
    cal = config["calibration"]
    return cam, (int(cal["projector_width"]), int(cal["projector_height"]))


def black_head(img: np.ndarray, head_box: tuple[int, int, int, int]) -> np.ndarray:
    """SAFETY: zero the head box in place (clamped to the image); returns img."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in head_box)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = 0
    return img


def evaluate(frame_lit: np.ndarray, frame_dark: np.ndarray, truth_mask: np.ndarray, depth: np.ndarray,
             K_cam: np.ndarray, dcfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(errors_mm, lit_pts, truth_pts) of one lit frame against the depth mask captured with it."""
    lit = detect_lit_outline(frame_lit, frame_dark, dcfg)
    truth = truth_boundary_from_depth(truth_mask)
    err = boundary_errors_mm(lit, truth, depth, K_cam, max_px=float(dcfg["max_search_px"]), inside_mask=truth_mask,
                             offset_px=(int(dcfg["outline_thickness_px"]) - 1) / 2.0)
    return err, lit, truth


def _fmt(s: dict) -> str:
    if s["median"] is None:
        return f"no lit pixels matched (rejected {s['n_rejected']})"
    return (f"mean {s['mean']:.2f} mm, median {s['median']:.2f} mm, p95 {s['p95']:.2f} mm, max {s['max']:.1f} mm, "
            f"bias {s['bias']:+.2f} mm, n {s['n']} (rejected {s['n_rejected']})")


def report(phases: list[dict]) -> bool:
    """Print the spec targets with PASS/FAIL per phase (kind 'still' / 'turn'; other kinds are informational).
    Returns True when every target phase passes."""
    print("\naccuracy targets (BUILD_SPEC_3D section 10): still median < 10 mm, p95 < 20 mm; "
          "slow movement median < 25 mm")
    all_ok = True
    for p in phases:
        s, kind = p["summary"], p["kind"]
        tgt = TARGETS.get(kind)
        if tgt is None:
            print(f"  {p['name']:<38} INFO  {_fmt(s)}")
            continue
        checks = []
        for k, lim in tgt.items():
            v = s[k]
            ok = v is not None and v < lim
            checks.append((k, v, lim, ok))
        ok = all(c[3] for c in checks)
        all_ok &= ok
        detail = ", ".join(f"{k} {'n/a' if v is None else f'{v:.2f}'} < {lim:g}" for k, v, lim, _ in checks)
        print(f"  {p['name']:<38} {'PASS' if ok else 'FAIL'}  {detail} mm  (n {s['n']})")
    return all_ok


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


# --- simulate ------------------------------------------------------------------------------

class PersonRig3D(SimulatedRig3D):
    """SimulatedRig3D (no boxes) + a flat, fronto-parallel person cutout `offset_m` in front of the wall, centred
    on the projector's image centre ray. Unlike the base rig, the person's projector SHADOW on the wall is
    modelled (wall pixels whose ray to the projector crosses the person stay unlit), so light that misses the
    body lands where it would physically. `guide()` is a room-light colour image (white clothes on a grey wall)
    for DepthBodyMask's edge-aware upsampling."""

    # Person parts relative to the centre, metres, OpenCV axes (+y down): rectangles (x0, x1, y0, y1) + head disc.
    RECTS = ((-0.20, 0.20, -0.30, 0.22),                         # torso
             (-0.34, 0.34, -0.30, -0.22),                        # shoulders (joins the arms)
             (0.25, 0.34, -0.30, 0.25), (-0.34, -0.25, -0.30, 0.25),   # arms, 5 cm gap to the torso
             (-0.05, 0.05, -0.36, -0.29),                        # neck
             (0.03, 0.18, 0.22, 0.50), (-0.18, -0.03, 0.22, 0.50))     # legs
    HEAD = (0.0, -0.44, 0.09)                                    # centre x, y, radius
    HEAD_PAD_M = 0.04

    def __init__(self, cfg: dict, cam_size: tuple[int, int], proj_size: tuple[int, int], truth: dict,
                 offset_m: float, wall_z: float = WALL_Z, dx_m: float = 0.0, seed: int = 1) -> None:
        super().__init__(cfg, cam_size, proj_size, **truth, wall_z=wall_z, boxes=(), seed=seed)
        self.wall_z = float(wall_z)
        self.z_person = self.wall_z - float(offset_m)
        w, h = self.cam_size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        self._rx = (u - self.K_cam[0, 2]) / self.K_cam[0, 0]
        self._ry = (v - self.K_cam[1, 2]) / self.K_cam[1, 1]
        C = -self.R.T @ self.t
        d = self.R.T @ np.linalg.solve(self.K_p, [self.proj_size[0] / 2, self.proj_size[1] / 2, 1.0])
        self.centre = (C + (self.z_person - C[2]) / d[2] * d)[:2] + np.array([dx_m, 0.0])
        self._C = C
        self.person_mask = self._inside(self._rx * self.z_person, self._ry * self.z_person)
        hx, hy, hr = self.HEAD
        r = hr + self.HEAD_PAD_M
        X0, Y0 = self.centre + np.array([hx - r, hy - r])
        X1, Y1 = self.centre[0] + hx + r, self.centre[1] - 0.29   # head box ends at the neck base
        f, c = self.K_cam[[0, 1], [0, 1]], self.K_cam[:2, 2]
        p0 = f * np.array([X0, Y0]) / self.z_person + c
        p1 = f * np.array([X1, Y1]) / self.z_person + c
        self.head_box = (int(math.floor(p0[0])), int(math.floor(p0[1])), int(math.ceil(p1[0])) + 1,
                         int(math.ceil(p1[1])) + 1)
        self.set_person(False)

    def _inside(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x, y = X - self.centre[0], Y - self.centre[1]
        m = np.zeros(x.shape, bool)
        for x0, x1, y0, y1 in self.RECTS:
            m |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        hx, hy, hr = self.HEAD
        m |= (x - hx) ** 2 + (y - hy) ** 2 <= hr * hr
        return m

    def set_person(self, on: bool) -> None:
        """Person in the scene or not (the empty scene is what DepthBodyMask learns as background).
        Recomputes the true depth, the per-pixel projector map, the shadow and the confidence."""
        depth = np.full(self.true_depth.shape, self.wall_z, np.float64)
        if on:
            depth[self.person_mask] = self.z_person
        self.true_depth = depth.astype(np.float32)
        P = np.stack([self._rx * depth, self._ry * depth, depth], axis=-1).reshape(-1, 1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        pp, _ = cv2.projectPoints(P, rvec, self.t, self.K_p, self.dist_p)
        h, w = depth.shape
        self._map = pp.reshape(h, w, 2).astype(np.float32)
        if on:
            # wall point P is shadowed when the segment projector centre -> P crosses the person plane inside the person
            C = self._C
            wall = ~self.person_mask
            s = (self.z_person - C[2]) / (self.wall_z - C[2])
            QX = C[0] + s * (self._rx[wall] * self.wall_z - C[0])
            QY = C[1] + s * (self._ry[wall] * self.wall_z - C[1])
            shadow = np.zeros(depth.shape, bool)
            shadow[wall] = self._inside(QX, QY)
            self._map[shadow] = -10.0                       # outside the projector frame -> unlit
        k = np.ones((5, 5), np.uint8)
        grad = cv2.dilate(self.true_depth, k) - cv2.erode(self.true_depth, k)
        self._conf = np.where(grad > 0.05, 1, 2).astype(np.uint8)
        self.person_on = bool(on)

    def guide(self) -> np.ndarray:
        """Room-light camera image (no projector): grey wall 90, white clothes 200, Gaussian noise."""
        img = np.where(self.true_depth < self.wall_z - 0.01, 200.0, 90.0).astype(np.float32)
        img += self._rng.standard_normal(img.shape, dtype=np.float32) * self.noise_sigma
        return cv2.cvtColor(np.clip(np.rint(img), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _sim_case(name: str, proj3d, rig: PersonRig3D, dm, n: int, a: dict, dcfg: dict, K_cam: np.ndarray,
              out_dir: Path | None, tag: str) -> dict:
    """n lit/dark iterations of the static person through one projector model; returns the phase dict."""
    errs = []
    black = np.zeros((rig.proj_size[1], rig.proj_size[0], 3), np.uint8)
    color = tuple(int(c) for c in a["outline_color"])
    last = None
    for _ in range(n):
        depth = rig.depth()
        mask = dm.compute(depth, rig.conf(), rig.guide()).copy()
        out = black_head(outline_image(mask, int(a["outline_thickness_px"]), color), rig.head_box)
        rig.show(proj3d.render(out, depth))
        lit = rig.capture()
        depth_lit = rig.depth()                              # depth arriving with the lit frame -> truth mask
        truth_mask = dm.compute(depth_lit, rig.conf(), rig.guide()).copy()
        rig.show(black)
        dark = rig.capture()
        e, lit_pts, truth_pts = evaluate(lit, dark, truth_mask, depth_lit, K_cam, dcfg)
        errs.append(e)
        last = (lit, lit_pts, truth_pts)
    s = summarize(np.concatenate(errs))
    if out_dir is not None and last is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        img = annotate(cv2.convertScaleAbs(last[0], alpha=1.0, beta=40), last[1], last[2],
                       f"{name}: median {s['median'] if s['median'] is None else round(s['median'], 2)} mm")
        slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        cv2.imwrite(str(out_dir / f"accuracy_sim_{tag}_{slug}.jpg"), img)
    return {"summary": s, "frames": n}


def run_simulation(config: dict, out_dir: Path | None = None, log=print) -> dict:
    """Full --simulate procedure. Returns {'phases': [...], 'gates': {...}, 'expected_shift_mm', ...}.
    Raises RuntimeError (with a fix-it hint) when no offscreen OpenGL context is available."""
    from fitcast.depthmask import DepthBodyMask
    from fitcast.render3d import DepthMeshProjector

    a = acc_cfg(config)
    dcfg = detect_cfg(config)
    cam, proj = _sizes(config)
    c3cfg = {**C3D_DEFAULTS, **config["calibration3d"], "settle_frames": 0,
             "frames_per_pattern": int(a["sim_frames_per_pattern"])}
    truth = sim_truth(cam, proj, float(c3cfg["throw_ratio_guess"]))
    t0 = time.perf_counter()
    calib_rig = SimulatedRig3D(c3cfg, cam, proj, **truth)
    c = run_calibration3d(c3cfg, cam, proj, calib_rig.capture, calib_rig.show, calib_rig.depth, calib_rig.conf,
                          log=lambda s: None, K_cam=truth["K_cam"])
    del calib_rig
    log(f"calibration (box rig): reproj {c.reproj_error_px:.3f} px over {c.n_points} points, "
        f"cx {c.K_p[0, 2]:.2f} vs true {truth['K_p_true'][0, 2]:.2f}, f {c.K_p[0, 0]:.1f} vs "
        f"{truth['K_p_true'][0, 0]:.1f} ({time.perf_counter() - t0:.1f} s)")

    rig = PersonRig3D(c3cfg, cam, proj, truth, float(a["sim_person_offset_m"]))
    K_cam = truth["K_cam"]
    dm = DepthBodyMask(config["depthmask"], cam, cam)
    dm.set_intrinsics(K_cam)
    n_bg = max(3, min(10, int(config["depthmask"]["background_frames"])))
    dm.learn_background([rig.depth() for _ in range(n_bg)], [rig.conf() for _ in range(n_bg)])
    rig.set_person(True)
    m = dm.compute(rig.depth(), rig.conf(), rig.guide())
    if not dm.last_stats["body_pixels"]:
        raise RuntimeError("simulate: DepthBodyMask found no person (check depthmask.box_m / closer_than_background_m)")
    inter = np.logical_and(m > 0, rig.person_mask).sum()
    iou = inter / max(1, np.logical_or(m > 0, rig.person_mask).sum())
    log(f"scene: wall {rig.wall_z} m, person cutout at {rig.z_person:.2f} m, depth mask IoU vs exact silhouette "
        f"{iou:.4f}; head box {rig.head_box} blacked out")

    # metric floor: the outline image itself as the 'lit' frame (no projector)
    color = tuple(int(v) for v in a["outline_color"])
    depth0 = rig.depth()
    m0 = dm.compute(depth0, rig.conf(), rig.guide()).copy()
    floor = summarize(evaluate(outline_image(m0, int(a["outline_thickness_px"]), color),
                               np.zeros((cam[1], cam[0], 3), np.uint8), m0, depth0, K_cam, dcfg)[0])
    log(f"metric floor (outline image read back directly): {_fmt(floor)}")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    shift = float(a["sim_shift_px"])
    K_shift = c.K_p.copy()
    K_shift[0, 2] += shift
    cases = [("still (true projector model)", "still", truth["K_p_true"], truth["dist_p_true"], truth["R_true"],
              truth["t_true"]),
             ("still (recovered calibration)", "still", c.K_p, c.dist_p, c.R, c.t),
             (f"sanity (recovered, cx +{shift:g} px)", "sanity", K_shift, c.dist_p, c.R, c.t)]
    phases = []
    for name, kind, Kp, dist, R, t in cases:
        try:
            p3d = DepthMeshProjector(config["render3d"], cam, proj, K_cam, Kp, dist, R, t)
        except RuntimeError as e:
            raise RuntimeError(f"simulate needs an offscreen OpenGL context: {e}") from e
        try:
            ph = _sim_case(name, p3d, rig, dm, int(a["sim_frames"]), a, dcfg, K_cam, out_dir, tag)
        finally:
            p3d.release()
        ph.update(name=name, kind=kind)
        phases.append(ph)
        log(f"{name:<38} {_fmt(ph['summary'])}")

    expected = shift * rig.z_person * 1000.0 / float(c.K_p[0, 0])
    m_true, m_rec, m_shift = (p["summary"]["median"] for p in phases)
    gates = {
        f"true model median < {a['sim_max_median_mm']} mm": m_true is not None and m_true < a["sim_max_median_mm"],
        f"recovered median < {a['sim_max_median_mm']} mm": m_rec is not None and m_rec < a["sim_max_median_mm"],
        f"shifted median >= {a['sim_shift_min_ratio']}x recovered": (
            m_shift is not None and m_rec is not None and m_shift >= a["sim_shift_min_ratio"] * max(m_rec, 1e-6)),
        f"shifted median >= {a['sim_shift_min_frac']} x expected {expected:.1f} mm": (
            m_shift is not None and m_shift >= a["sim_shift_min_frac"] * expected),
    }
    log(f"sanity: a {shift:g} px principal-point shift moves the light by {shift:g} * z / f_p = {expected:.1f} mm on "
        f"the body plane (horizontally; horizontal edges see less, light that misses the body lands on the wall)")
    for k, ok in gates.items():
        log(f"  gate {k}: {'OK' if ok else 'FAILED'}")
    res = {"mode": "simulate", "timestamp": tag, "calibration": {"reproj_error_px": c.reproj_error_px,
                                                                 "n_points": c.n_points},
           "scene": {"wall_z_m": rig.wall_z, "person_z_m": rig.z_person, "depth_mask_iou": float(iou)},
           "metric_floor": floor, "expected_shift_mm": expected, "phases": phases, "gates": gates,
           "accuracy_config": a}
    if out_dir is not None:
        _write_json(out_dir / f"accuracy_sim_{tag}.json", res)
        log(f"wrote {out_dir / f'accuracy_sim_{tag}.json'} and annotated frames accuracy_sim_{tag}_*.jpg")
    return res


# --- live ----------------------------------------------------------------------------------

class _Live:
    """Live accuracy run on the iPhone sensor + projector (main thread only: GL context + Cocoa windows)."""

    def __init__(self, config: dict, args: argparse.Namespace) -> None:
        from fitcast.calibration import Calibration
        from fitcast.depthmask import DepthBodyMask
        from fitcast.display import Display
        from fitcast.sensors.iphone import IPhoneSensor

        self.config, self.args = config, args
        self.a = acc_cfg(config)
        self.dcfg = detect_cfg(config)
        self.cam, self.proj = _sizes(config)
        self.settle = int(config["calibration"]["settle_frames"])
        self.black = np.zeros((self.proj[1], self.proj[0], 3), np.uint8)
        self.c3d = None
        self.calib = None
        self.proj3d = None
        c3d_path = Path(args.calibration3d or config["calibration3d"]["file"])
        if c3d_path.exists():
            from fitcast.calib3d import load_calibration3d
            self.c3d = load_calibration3d(c3d_path)
            self.projection = f"depth mesh ({c3d_path}, reproj {self.c3d.reproj_error_px:.2f} px)"
        else:
            self.calib = Calibration(config["calibration"], self.cam)
            if not self.calib.load(args.calibration, allow_simulated=False):
                raise RuntimeError(f"neither {c3d_path} nor a valid 2D calibration loaded: the numbers would be "
                                   "meaningless. Fix: run scripts/calibrate3d.py --sensor iphone (or scripts/calibrate.py)")
            self.projection = "2D homography (calibration.yaml; no depth correction)"
            print(f"WARNING: no {c3d_path}: projecting through the 2D homography")
        scfg = dict(config["sensor"])
        if args.sensor_host:
            scfg["usbmux"], scfg["host"] = False, args.sensor_host
        self.sensor = IPhoneSensor(scfg, self.cam, tracker_cfg=config["tracker"])
        self.dm = DepthBodyMask(config["depthmask"], self.cam, self.cam)
        self.display = Display(config["display"], self.proj, (args.projector_x, args.projector_y))
        self.last_seq = -1
        self._warned_k = False

    # -- io ----------------------------------------------------------------------------------
    def show(self, img: np.ndarray) -> None:
        self.display.show_projector(img)
        self.display.key(1)

    def next_frame(self):
        """Next new sensor frame with its matching TrackResult: (frame, ts, track, conf, seg) or None on timeout."""
        t0 = time.perf_counter()
        timeout = float(self.a["frame_timeout_s"])
        while time.perf_counter() - t0 < timeout:
            item = self.sensor.read()
            if item is not None and item[2] != self.last_seq:
                st, conf, seg = self.sensor.track(), self.sensor.last_conf, self.sensor.last_seg
                again = self.sensor.read()
                if st is not None and st.timestamp == item[1] and again is not None and again[2] == item[2]:
                    self.last_seq = item[2]
                    return item[0], item[1], st, conf, seg
            self.display.key(1)
            time.sleep(0.001)
        return None

    def mask_of(self, frame: np.ndarray, st, conf, seg) -> np.ndarray | None:
        if st.depth is None:
            return None
        if st.K_cam is not None:
            self.dm.set_intrinsics(st.K_cam)
        conf = conf if conf is not None else np.full(st.depth.shape, 2, np.uint8)
        m = self.dm.compute(st.depth, conf, frame, person_seg=seg)
        return m.copy() if self.dm.last_stats["body_pixels"] else None

    def to_projector(self, img_cam: np.ndarray, st) -> np.ndarray:
        if self.c3d is not None:
            if self.proj3d is None:
                from fitcast.render3d import DepthMeshProjector
                c = self.c3d
                K = st.K_cam if st.K_cam is not None else c.K_cam
                self.proj3d = DepthMeshProjector(self.config["render3d"], self.cam, tuple(c.proj_size), np.asarray(K),
                                                 np.asarray(c.K_p), np.asarray(c.dist_p), np.asarray(c.R),
                                                 np.asarray(c.t))
            return self.proj3d.render(img_cam, st.depth)
        return self.calib.to_projector(img_cam)

    def skip(self, n: int) -> bool:
        for _ in range(n):
            if self.next_frame() is None:
                return False
        return True

    # -- procedure ---------------------------------------------------------------------------
    def learn_background(self) -> None:
        self.show(self.black)
        input("\nStep OUT of the beam (empty scene), then press Enter to learn the depth background... ")
        need = int(self.config["depthmask"]["background_frames"])
        depths, confs = [], []
        while len(depths) < need:
            f = self.next_frame()
            if f is None:
                raise RuntimeError("no sensor frames. Fix: start the FitCastSensor app, check USB / --sensor-host")
            _, _, st, conf, _ = f
            if st.depth is None:
                continue
            if st.K_cam is not None:
                self.dm.set_intrinsics(st.K_cam)
            depths.append(st.depth.copy())
            confs.append((conf if conf is not None else np.full(st.depth.shape, 2, np.uint8)).copy())
        self.dm.learn_background(depths, confs)
        print(f"depth background learned from {need} frames")

    def phase(self, name: str, seconds: float, out_dir: Path, tag: str) -> dict:
        a = self.a
        color = tuple(int(c) for c in a["outline_color"])
        thick = int(a["outline_thickness_px"])
        dark_every = max(1, int(a["dark_every_n"]))
        n_ann = max(0, int(a["annotate_frames"]))
        errs, n_eval, n_dark, n_skipped_untracked, saved = [], 0, 0, 0, 0
        dark = None
        lit_streak = 0                                       # consecutive frames shown with the outline on
        since_dark = 0                                       # evaluated frames since the last dark frame
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < seconds:
            if dark is None or since_dark >= dark_every:
                self.show(self.black)
                if not self.skip(self.settle):
                    break
                f = self.next_frame()
                if f is None:
                    break
                dark = f[0].copy()
                n_dark += 1
                lit_streak = 0
                since_dark = 0
            f = self.next_frame()
            if f is None:
                print("ERROR: sensor frames stopped")
                break
            frame, ts, st, conf, seg = f
            mask = self.mask_of(frame, st, conf, seg)
            if lit_streak > self.settle and mask is not None and st.K_cam is None and not self._warned_k:
                print("WARNING: sensor frames carry no K (intrinsics): cannot convert to mm, frames not evaluated")
                self._warned_k = True
            if lit_streak > self.settle and mask is not None and st.K_cam is not None:
                e, lit_pts, truth_pts = evaluate(frame, dark, mask, st.depth, st.K_cam, self.dcfg)
                errs.append(e)
                n_eval += 1
                since_dark += 1
                el = time.perf_counter() - t_start
                if saved < n_ann and el >= saved * seconds / n_ann:
                    med = summarize(e)["median"]
                    txt = f"{name} t={el:.1f}s median " + ("n/a" if med is None else f"{med:.1f} mm")
                    cv2.imwrite(str(out_dir / f"accuracy_{tag}_{name.replace(' ', '_')}_{saved}.jpg"),
                                annotate(frame, lit_pts, truth_pts, txt))
                    saved += 1
            if st.ok and mask is not None:
                out = black_head(outline_image(mask, thick, color), st.head_box)   # SAFETY: last op before projecting
                self.show(self.to_projector(out, st))
                lit_streak += 1
            else:
                n_skipped_untracked += int(not st.ok)
                self.show(self.black)                         # nobody tracked -> no head box -> no light
                lit_streak = 0
        self.show(self.black)
        e = np.concatenate(errs) if errs else np.zeros(0)
        s = summarize(e)
        print(f"{name}: {n_eval} lit frames, {n_dark} dark frames, {n_skipped_untracked} untracked; {_fmt(s)}")
        return {"name": name, "kind": "turn" if "turn" in name else "still", "summary": s, "lit_frames": n_eval,
                "dark_frames": n_dark, "untracked_frames": n_skipped_untracked}

    def run(self) -> dict:
        a, args = self.a, self.args
        out_dir = Path(a["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        seconds = float(args.seconds if args.seconds is not None else a["seconds"])
        self.sensor.start()
        self.display.open_projector(fullscreen=self.display.second is not None)
        phases = []
        try:
            self.show(self.black)
            self.learn_background()
            plan = [("still", "Stand STILL on the floor mark facing the camera, then press Enter... ")]
            if args.turn:
                plan.append(("slow turn", "Next: turn SLOWLY on the mark (about 90 deg over the phase). Press Enter... "))
            for name, prompt in plan:
                self.show(self.black)
                input(f"\n[{name}, {seconds:g} s] {prompt}")
                phases.append(self.phase(name, seconds, out_dir, tag))
        finally:
            self.show(self.black)
            if self.proj3d is not None:
                self.proj3d.release()
            self.display.close()
            self.sensor.stop()
        res = {"mode": "live", "timestamp": tag, "projection": self.projection, "phases": phases,
               "sensor_stats": {k: v for k, v in self.sensor.stats.items()}, "accuracy_config": a,
               "note": "outline of the CURRENT depth mask (no prediction): static transfer accuracy + pipeline latency"}
        path = out_dir / f"accuracy_{tag}.json"
        _write_json(path, res)
        print(f"wrote {path} and annotated frames {out_dir}/accuracy_{tag}_*.jpg")
        return res


# --- cli -----------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FitCast projection accuracy check (mm, spec section 10)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="headless simulated rig (no hardware)")
    mode.add_argument("--sensor", choices=["iphone"], help="live with the iPhone LiDAR sensor app")
    ap.add_argument("--sensor-host", help="iphone: connect to this IP instead of USB (usbmux)")
    ap.add_argument("--calibration3d", help="calibration3d.yaml (default: config calibration3d.file)")
    ap.add_argument("--calibration", help="2D calibration.yaml fallback (default: config calibration.file)")
    ap.add_argument("--projector-x", type=int)
    ap.add_argument("--projector-y", type=int)
    ap.add_argument("--seconds", type=float, default=None, help="seconds per phase (default: accuracy.seconds)")
    ap.add_argument("--turn", action="store_true", help="live: add a 'slow turn' phase after 'still'")
    ap.add_argument("--report", action="store_true", help="print spec targets with PASS/FAIL per phase")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.simulate:
        try:
            res = run_simulation(config, out_dir=Path(acc_cfg(config)["out_dir"]))
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 2
        gates_ok = all(res["gates"].values())
        if args.report:
            report(res["phases"])
        print(f"\nsimulate: {'all sanity gates passed' if gates_ok else 'SANITY GATE FAILED'}")
        return 0 if gates_ok else 1
    try:
        live = _Live(config, args)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    res = live.run()
    ok = True
    if args.report:
        ok = report(res["phases"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
