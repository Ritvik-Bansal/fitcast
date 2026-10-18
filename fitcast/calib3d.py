"""Full projector-camera calibration with LiDAR depth (docs/BUILD_SPEC_3D.md section 6).

Gray code structured light gives, per camera pixel, the projector pixel that lights it; LiDAR depth lifts
the camera pixel to a metric 3D point (OpenCV camera coords: +x right, +y down, +z forward). Those
3D <-> projector-pixel correspondences calibrate the projector as an inverse camera:
    p = K_p * distort(R X + t)          X in camera coords (metres), p in projector px.

Also here: a headless `SimulatedRig3D`, `run_calibration3d` (the procedure shared by
`scripts/calibrate3d.py` and the tests) and `verification_pattern`.

Gray code decode path: a vectorised numpy decoder (`decode_gray_code`). OpenCV's Python binding only has
the per-pixel `GrayCodePattern.getProjPixel(images, x, y) -> (error, (x, y))` (~3 us per call, ~3 s for a
1280x720 frame) and `decode()`, which needs a stereo pair. The numpy decoder reproduces getProjPixel's
bit order and thresholds exactly (tests/test_calib3d.py cross-checks it). Threshold semantics follow
OpenCV: `gray_code_white_threshold` = minimum |pattern - inverse| for an unambiguous bit,
`gray_code_black_threshold` = minimum (white - black) for a lit (non-shadow) pixel.
Pixel coordinates use OpenCV's pixel-centre convention (pixel i has centre i) in both images.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

# Keys not (yet) in config.yaml `calibration3d:`; used only when the config omits them (proposed for config.yaml).
DEFAULTS = {
    "settle_frames": 10,          # camera frames discarded after each pattern change (projector + sensor latency)
    "depth_frames": 10,           # depth/confidence frames combined (median depth, min confidence)
    "max_correspondences": 50000, # random subset of decoded pixels used for RANSAC / the final [R|t] refine
    "max_calib_points": 4000,     # subset passed to cv2.calibrateCamera (intrinsics)
    "min_points": 500,            # fewer valid correspondences -> RuntimeError
    "solve_rounds": 2,            # calibrateCamera -> RANSAC inliers -> calibrateCamera on inliers ...
    "ransac_iterations": 500,
    "coarse_ransac_px": 25.0,     # inlier threshold of the distortion-free DLT RANSAC that rejects gross decode errors
    "ransac_confidence": 0.999,
    "lm_max_iter": 200,           # calibrateCamera / solvePnPRefineLM iterations
    "fix_k3": True,               # one view of a few planes constrains k3 poorly
    "zero_tangent_dist": True,
    "seed": 0,                    # subsampling RNG
    "verify_grid_m": 0.1,         # verification grid spacing on the measured surface (metres)
    "verify_edge_m": 0.1,         # depth step between neighbouring pixels that counts as a silhouette edge
    "verify_line_px": 3,          # line thickness of the verification pattern (projector px)
}

_FIX = "run `.venv/bin/python scripts/calibrate3d.py --sensor iphone` (phone + projector) to create it"

# Verification pattern colours (BGR, cosmetic).
GRID_COLOR = (255, 255, 255)
EDGE_COLOR = (0, 255, 0)

CaptureFn = Callable[[], np.ndarray]
ShowFn = Callable[[np.ndarray], None]


def _opt(cfg: dict, key: str):
    return cfg.get(key, DEFAULTS[key])


# --- data ----------------------------------------------------------------------------------

@dataclass
class Calibration3D:
    """Camera + projector intrinsics and the camera -> projector rigid transform (X_p = R X_c + t, metres)."""
    K_cam: np.ndarray            # (3, 3) camera intrinsics at cam_size
    dist_cam: np.ndarray         # (5,) camera distortion (zeros for ARKit's pinhole intrinsics)
    K_p: np.ndarray              # (3, 3) projector intrinsics at proj_size
    dist_p: np.ndarray           # (5,) projector distortion k1 k2 p1 p2 k3
    R: np.ndarray                # (3, 3) rotation camera -> projector
    t: np.ndarray                # (3,) translation camera -> projector, metres
    cam_size: tuple[int, int]    # w, h
    proj_size: tuple[int, int]   # w, h
    reproj_error_px: float       # RMS over RANSAC inliers, projector px
    n_points: int                # inlier correspondences
    timestamp: str = ""
    notes: str = ""


_SHAPES = {"K_cam": (3, 3), "K_p": (3, 3), "R": (3, 3), "t": (3,)}
_DIST_LENGTHS = (4, 5, 8, 12, 14)


def save_calibration3d(path: str | Path, c: Calibration3D) -> None:
    """Write `c` as YAML (matrices as nested lists)."""
    d = {
        "K_cam": np.asarray(c.K_cam, float).tolist(),
        "dist_cam": np.asarray(c.dist_cam, float).ravel().tolist(),
        "K_p": np.asarray(c.K_p, float).tolist(),
        "dist_p": np.asarray(c.dist_p, float).ravel().tolist(),
        "R": np.asarray(c.R, float).tolist(),
        "t": np.asarray(c.t, float).ravel().tolist(),
        "cam_size": [int(v) for v in c.cam_size],
        "proj_size": [int(v) for v in c.proj_size],
        "reproj_error_px": float(c.reproj_error_px),
        "n_points": int(c.n_points),
        "timestamp": str(c.timestamp),
        "notes": str(c.notes),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=None))


def load_calibration3d(path: str | Path) -> Calibration3D:
    """Read a file written by save_calibration3d. Raises FileNotFoundError / ValueError with a fix-it hint."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"3D calibration {p} not found. Fix: {_FIX}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: not a calibration3d mapping. Fix: {_FIX}")
    missing = [k for k in Calibration3D.__dataclass_fields__ if k not in raw]
    if missing:
        raise ValueError(f"{p}: missing keys {missing}. Fix: {_FIX}")
    arr = {}
    for k, shape in _SHAPES.items():
        a = np.asarray(raw[k], np.float64)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"{p}: {k} has shape {a.shape}, expected {shape} (finite). Fix: {_FIX}")
        arr[k] = a
    for k in ("dist_cam", "dist_p"):
        a = np.asarray(raw[k], np.float64).ravel()
        if len(a) not in _DIST_LENGTHS or not np.isfinite(a).all():
            raise ValueError(f"{p}: {k} has {len(a)} coefficients, expected one of {_DIST_LENGTHS}. Fix: {_FIX}")
        arr[k] = a
    R = arr["R"]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or abs(np.linalg.det(R) - 1) > 1e-6:
        raise ValueError(f"{p}: R is not a rotation matrix. Fix: {_FIX}")
    sizes = {}
    for k in ("cam_size", "proj_size"):
        s = raw[k]
        if not (isinstance(s, (list, tuple)) and len(s) == 2 and all(int(v) > 0 for v in s)):
            raise ValueError(f"{p}: {k} must be [w, h] positive ints, got {s!r}. Fix: {_FIX}")
        sizes[k] = (int(s[0]), int(s[1]))
    return Calibration3D(K_cam=arr["K_cam"], dist_cam=arr["dist_cam"], K_p=arr["K_p"], dist_p=arr["dist_p"],
                         R=R, t=arr["t"], cam_size=sizes["cam_size"], proj_size=sizes["proj_size"],
                         reproj_error_px=float(raw["reproj_error_px"]), n_points=int(raw["n_points"]),
                         timestamp=str(raw["timestamp"]), notes=str(raw["notes"]))


# --- Gray code -----------------------------------------------------------------------------

def _n_bits(n: int) -> int:
    return int(math.ceil(math.log2(n)))


def gray_code_patterns(proj_size: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """OpenCV Gray code sequence for a pw x ph projector: (patterns, white, black), each (ph, pw) uint8 gray.
    patterns = 2 * (ceil(log2 pw) + ceil(log2 ph)) images: column bits then row bits, MSB first,
    each code image followed by its inverse (44 images at 1920x1080)."""
    pw, ph = int(proj_size[0]), int(proj_size[1])
    gc = cv2.structured_light.GrayCodePattern.create(width=pw, height=ph)
    ok, patterns = gc.generate()
    if not ok or len(patterns) != gc.getNumberOfPatternImages():
        raise RuntimeError(f"GrayCodePattern.generate failed for {pw}x{ph}")
    black, white = gc.getImagesForShadowMasks(np.zeros((ph, pw), np.uint8), np.zeros((ph, pw), np.uint8))
    return list(patterns), white, black


def _gray_f32(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32), cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32, copy=False)


def decode_gray_code(captures: list[np.ndarray], white: np.ndarray, black: np.ndarray,
                     proj_size: tuple[int, int], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per camera pixel, the projector pixel that lights it.

    captures: camera images (BGR or gray, uint8 or float) of the `gray_code_patterns` sequence in order;
    white / black: camera images of the all-white / all-black projector frames. cfg = config["calibration3d"].
    Returns proj_xy (H, W, 2) float32 projector (x, y) (pixel centres) and valid (H, W) bool: lit
    (white - black > gray_code_black_threshold), every bit unambiguous (|code - inverse| >=
    gray_code_white_threshold) and the code inside the projector frame. Same result as
    GrayCodePattern.getProjPixel, vectorised (~0.1 s at 1280x720 instead of ~3 s)."""
    pw, ph = int(proj_size[0]), int(proj_size[1])
    ncol, nrow = _n_bits(pw), _n_bits(ph)
    if len(captures) != 2 * (ncol + nrow):
        raise ValueError(f"decode_gray_code: got {len(captures)} captures, expected {2 * (ncol + nrow)} "
                         f"for a {pw}x{ph} projector")
    bit_thr = float(cfg["gray_code_white_threshold"])
    valid = (_gray_f32(white) - _gray_f32(black)) > float(cfg["gray_code_black_threshold"])
    out = []
    for offset, nbits in ((0, ncol), (2 * ncol, nrow)):
        value = np.zeros(valid.shape, np.int32)
        binary = np.zeros(valid.shape, bool)
        for i in range(nbits):
            d = _gray_f32(captures[offset + 2 * i]) - _gray_f32(captures[offset + 2 * i + 1])
            valid &= np.abs(d) >= bit_thr
            binary ^= d > 0                    # binary bit i = binary bit i-1 XOR gray bit i
            value = (value << 1) | binary
        out.append(value)
    x, y = out
    valid &= (x < pw) & (y < ph)
    return np.stack([x, y], axis=-1).astype(np.float32), valid


# --- geometry ------------------------------------------------------------------------------

def lift_to_3d(pixels: np.ndarray, depth_cam: np.ndarray, K_cam: np.ndarray) -> np.ndarray:
    """Camera pixels (N, 2) x, y -> (N, 3) float64 camera-coords points X = depth * K^-1 [x y 1].
    Depth is sampled nearest-neighbour (never mixes the two sides of a depth edge). Rows whose depth is
    invalid (<= 0, non-finite, outside the image) are NaN: drop them with `np.isfinite(P).all(1)`
    (kept in place so the result stays aligned with `pixels`)."""
    px = np.asarray(pixels, np.float64).reshape(-1, 2)
    h, w = depth_cam.shape[:2]
    xi, yi = np.rint(px[:, 0]).astype(np.int64), np.rint(px[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    z = np.full(len(px), np.nan)
    z[inside] = depth_cam[yi[inside], xi[inside]]
    z[~(z > 0) | ~np.isfinite(z)] = np.nan
    K = np.asarray(K_cam, np.float64)
    X = (px[:, 0] - K[0, 2]) / K[0, 0] * z
    Y = (px[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([X, Y, z], axis=1)


def _rms(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray, rvec: np.ndarray,
         tvec: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)


def _dlt(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct linear transform: 3x4 projection matrix P with x ~ P [X 1] (least squares, n >= 6)."""
    Xh = np.column_stack([X, np.ones(len(X))])
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4], A[0::2, 8:12] = Xh, -x[:, :1] * Xh
    A[1::2, 4:8], A[1::2, 8:12] = Xh, -x[:, 1:2] * Xh
    return np.linalg.svd(A)[2][-1].reshape(3, 4)


def _similarity(p: np.ndarray) -> np.ndarray:
    """Hartley normalisation: (d+1)x(d+1) similarity moving p to zero mean and mean norm sqrt(d)."""
    c = p.mean(axis=0)
    s = math.sqrt(p.shape[1]) / max(float(np.mean(np.linalg.norm(p - c, axis=1))), 1e-12)
    T = np.eye(p.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return T


def _coarse_inliers(obj: np.ndarray, img: np.ndarray, thr: float, iters: int, conf: float,
                    rng: np.random.Generator) -> np.ndarray:
    """RANSAC over 6-point DLT projection matrices (no intrinsics needed, so it runs before any
    calibration): (N,) bool mask of points within `thr` px of the best pinhole model. Rejects gross
    decode errors (wrong high Gray code bits) that would bias the least-squares calibrateCamera."""
    T3, T2 = _similarity(obj), _similarity(img)
    Xh = np.column_stack([obj, np.ones(len(obj))])
    Xn, xn = (Xh @ T3.T)[:, :3], (np.column_stack([img, np.ones(len(img))]) @ T2.T)[:, :2]
    T2inv = np.linalg.inv(T2)

    def support(P_n: np.ndarray) -> np.ndarray:
        q = Xh @ (T2inv @ P_n @ T3).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.linalg.norm(q[:, :2] / q[:, 2:3] - img, axis=1) < thr

    best = np.zeros(len(obj), bool)
    i, need = 0, iters
    while i < min(iters, need):
        best_i = support(_dlt(Xn[s := rng.choice(len(obj), 6, replace=False)], xn[s]))
        if best_i.sum() > best.sum():
            best = best_i
            w = best.mean()
            need = math.ceil(math.log(1 - conf) / math.log(max(1 - w ** 6, 1e-12))) if w < 1 else 0
        i += 1
    if best.sum() >= 6:
        best = support(_dlt(Xn[best], xn[best]))        # refit on all inliers
    return best


def solve_projector(cam_pts3d: np.ndarray, proj_pts2d: np.ndarray, proj_size: tuple[int, int],
                    throw_ratio_guess: float, cfg: dict
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Calibrate the projector from 3D (camera coords, m) <-> projector px correspondences.

    Initial K_p: fx = fy = throw_ratio_guess * pw, principal point at the frame centre (cy is free, so a
    lens offset is recovered, even beyond the frame edge: up to one frame outside). Gross outliers are
    removed first by a 6-point DLT RANSAC (coarse_ransac_px). Then `solve_rounds` x
    [cv2.calibrateCamera on <= max_calib_points inliers with USE_INTRINSIC_GUESS | FIX_ASPECT_RATIO (+ ZERO_TANGENT_DIST, FIX_K3), then
    solvePnPRansac (ransac_reproj_px) on <= max_correspondences], then solvePnPRefineLM on the inliers.
    Returns K_p (3,3), dist_p (5,), R (3,3), t (3,) (X_p = R X + t), RMS over inliers (projector px) and
    the (N,) bool inlier mask (final reprojection < ransac_reproj_px) over all input correspondences."""
    obj_all = np.asarray(cam_pts3d, np.float64).reshape(-1, 3)
    img_all = np.asarray(proj_pts2d, np.float64).reshape(-1, 2)
    if len(obj_all) != len(img_all):
        raise ValueError(f"solve_projector: {len(obj_all)} 3D points vs {len(img_all)} projector points")
    if len(obj_all) < int(_opt(cfg, "min_points")):
        raise RuntimeError(f"solve_projector: only {len(obj_all)} correspondences (need {_opt(cfg, 'min_points')})")
    pw, ph = int(proj_size[0]), int(proj_size[1])
    rng = np.random.default_rng(int(_opt(cfg, "seed")))
    max_iter = int(_opt(cfg, "lm_max_iter"))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, max_iter, 1e-12)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    if _opt(cfg, "zero_tangent_dist"):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if _opt(cfg, "fix_k3"):
        flags |= cv2.CALIB_FIX_K3
    f0 = float(throw_ratio_guess) * pw
    K = np.array([[f0, 0, pw / 2.0], [0, f0, ph / 2.0], [0, 0, 1]], np.float64)
    dist = np.zeros(5)
    thr = float(cfg["ransac_reproj_px"])

    idx = np.arange(len(obj_all))
    if len(idx) > int(_opt(cfg, "max_correspondences")):
        idx = np.sort(rng.choice(idx, int(_opt(cfg, "max_correspondences")), replace=False))
    obj, img = obj_all[idx], img_all[idx]
    inl = _coarse_inliers(obj, img, float(_opt(cfg, "coarse_ransac_px")), int(_opt(cfg, "ransac_iterations")),
                          float(_opt(cfg, "ransac_confidence")), rng)
    if inl.sum() < int(_opt(cfg, "min_points")):
        raise RuntimeError(f"solve_projector: only {inl.sum()} of {len(obj)} correspondences fit one pinhole "
                           f"projector within {_opt(cfg, 'coarse_ransac_px')} px. Fix: check the decode (dim the room, "
                           "keep the rig still, cover glossy surfaces)")
    for _ in range(max(1, int(_opt(cfg, "solve_rounds")))):
        cand = np.flatnonzero(inl)
        n_cal = min(len(cand), int(_opt(cfg, "max_calib_points")))
        sub = np.sort(rng.choice(cand, n_cal, replace=False))
        # calibrateCamera insists the principal point lies inside imageSize, but projector lens offsets put cy
        # at or beyond the bottom edge: solve in a frame padded by one projector frame on every side (pixel
        # coords and principal point shifted by (pw, ph); exactly equivalent), then shift back.
        pad = np.array([pw, ph], np.float64)
        Ks = K.copy()
        Ks[:2, 2] += pad
        _, Ks, d, rvecs, tvecs = cv2.calibrateCamera(
            [obj[sub].astype(np.float32).reshape(-1, 1, 3)], [(img[sub] + pad).astype(np.float32).reshape(-1, 1, 2)],
            (3 * pw, 3 * ph), Ks, dist, flags=flags, criteria=criteria)
        K = Ks.copy()
        K[:2, 2] -= pad
        dist = d.ravel()[:5]
        ok, rvec, tvec, ransac_inl = cv2.solvePnPRansac(
            obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, dist, rvecs[0].copy(), tvecs[0].copy(),
            useExtrinsicGuess=True, iterationsCount=int(_opt(cfg, "ransac_iterations")),
            reprojectionError=thr, confidence=float(_opt(cfg, "ransac_confidence")),
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or ransac_inl is None or len(ransac_inl) < int(_opt(cfg, "min_points")):
            raise RuntimeError(f"solve_projector: RANSAC found {0 if ransac_inl is None else len(ransac_inl)} "
                               f"inliers of {len(obj)} at {thr} px. Fix: check the decode (dim the room, "
                               f"keep the rig still) or raise calibration3d.ransac_reproj_px")
        inl = np.zeros(len(obj), bool)
        inl[ransac_inl.ravel()] = True
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl].reshape(-1, 1, 3), img[inl].reshape(-1, 1, 2), K, dist,
                                      rvec, tvec, criteria=criteria)
    err = _rms(obj_all, img_all, K, dist, rvec, tvec)
    inlier_mask = err < thr
    rms = float(np.sqrt(np.mean(err[inlier_mask] ** 2)))
    R, _ = cv2.Rodrigues(rvec)
    return K, dist.copy(), R, tvec.ravel().copy(), rms, inlier_mask


def project_to_projector(c: Calibration3D, pts3d: np.ndarray) -> np.ndarray:
    """Camera-coords points (N, 3) metres -> (N, 2) projector px through the calibration."""
    pts = np.asarray(pts3d, np.float64).reshape(-1, 1, 3)
    rvec, _ = cv2.Rodrigues(np.asarray(c.R, np.float64))
    out, _ = cv2.projectPoints(pts, rvec, np.asarray(c.t, np.float64), np.asarray(c.K_p, np.float64),
                               np.asarray(c.dist_p, np.float64))
    return out.reshape(-1, 2)


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of R_a^T R_b in degrees."""
    c = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --- simulation ----------------------------------------------------------------------------

def sim_truth(cam_size: tuple[int, int], proj_size: tuple[int, int], throw_ratio_guess: float
              ) -> dict[str, np.ndarray]:
    """Plausible ground truth for SimulatedRig3D: phone camera ~f 0.75 w, projector 3% longer throw than
    the guess with a full upward lens offset (cy = 1.02 ph, below the frame), mild barrel distortion, mounted 9 cm below
    and 6 cm beside the camera, rotated ~2.5 deg."""
    cw, ch = cam_size
    pw, ph = proj_size
    fc = 0.75 * cw
    K_cam = np.array([[fc, 0, cw / 2 - 0.5 + 3.0], [0, fc, ch / 2 - 0.5 - 2.0], [0, 0, 1]], np.float64)
    fp = 1.03 * throw_ratio_guess * pw
    K_p = np.array([[fp, 0, pw / 2 + 14.0], [0, fp, 1.02 * ph], [0, 0, 1]], np.float64)
    dist_p = np.array([-0.05, 0.02, 0.0, 0.0, 0.0])
    R, _ = cv2.Rodrigues(np.radians([1.5, -2.0, 0.6]).reshape(3, 1))
    C = np.array([0.06, 0.09, 0.02])           # projector centre in camera coords (m): right, below, ahead
    return {"K_cam": K_cam, "K_p_true": K_p, "dist_p_true": dist_p, "R_true": R, "t_true": -R @ C}


class SimulatedRig3D:
    """Headless fake phone (camera + LiDAR) and projector looking at a wall (z = wall_z) with two
    fronto-parallel boxes. Box faces are centred left/right of where the projector image centre lands at
    their depth, so the projector always covers them. Captures are rendered from the TRUE depth; `depth()`
    adds noise. Projector shadows are not modelled (occluded wall pixels are still lit; the
    correspondences stay geometrically exact, so the solve is unaffected)."""

    def __init__(self, cfg: dict, cam_size: tuple[int, int], proj_size: tuple[int, int], K_cam: np.ndarray,
                 K_p_true: np.ndarray, dist_p_true: np.ndarray, R_true: np.ndarray, t_true: np.ndarray,
                 wall_z: float = 3.2, boxes: tuple[tuple[float, float], ...] = ((-0.5, 2.4), (0.5, 2.7)),
                 box_size: tuple[float, float] = (0.6, 0.8), depth_noise_m: float = 0.005,
                 ambient: float = 20.0, noise_sigma: float = 2.0, edge_px: int = 2, seed: int = 0) -> None:
        """cfg = config["calibration3d"] (unused by the rig itself; kept for symmetry with SimulatedRig).
        boxes: (x offset from the projector image centre, face depth) per box, metres."""
        self.cfg = cfg
        self.cam_size = (int(cam_size[0]), int(cam_size[1]))
        self.proj_size = (int(proj_size[0]), int(proj_size[1]))
        self.K_cam = np.asarray(K_cam, np.float64)
        self.K_p, self.dist_p = np.asarray(K_p_true, np.float64), np.asarray(dist_p_true, np.float64)
        self.R, self.t = np.asarray(R_true, np.float64), np.asarray(t_true, np.float64).ravel()
        self.depth_noise_m, self.ambient, self.noise_sigma = float(depth_noise_m), float(ambient), float(noise_sigma)
        self._rng = np.random.default_rng(seed)
        w, h = self.cam_size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        rx = (u - self.K_cam[0, 2]) / self.K_cam[0, 0]       # ray direction (rx, ry, 1)
        ry = (v - self.K_cam[1, 2]) / self.K_cam[1, 1]
        depth = np.full((h, w), float(wall_z))
        # projector image centre ray, in camera coords: C + s * d
        C = -self.R.T @ self.t
        d = self.R.T @ np.linalg.solve(self.K_p, [self.proj_size[0] / 2, self.proj_size[1] / 2, 1.0])
        self.box_rects = []                                  # (x0, x1, y0, y1, z) camera coords, metres
        for dx, z in boxes:
            cx, cy = (C + (z - C[2]) / d[2] * d)[:2]
            x0, x1 = cx + dx - box_size[0] / 2, cx + dx + box_size[0] / 2
            y0, y1 = cy - box_size[1] / 2, cy + box_size[1] / 2
            self.box_rects.append((x0, x1, y0, y1, float(z)))
            X, Y = rx * z, ry * z
            hit = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (z < depth)
            depth[hit] = z
        self.true_depth = depth.astype(np.float32)
        P = np.stack([rx * depth, ry * depth, depth], axis=-1).reshape(-1, 1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        pp, _ = cv2.projectPoints(P, rvec, self.t, self.K_p, self.dist_p)
        self._map = pp.reshape(h, w, 2).astype(np.float32)
        k = np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)
        grad = cv2.dilate(self.true_depth, k) - cv2.erode(self.true_depth, k)
        self._conf = np.where(grad > 0.05, 1, 2).astype(np.uint8)
        self._frame = np.zeros((self.proj_size[1], self.proj_size[0], 3), np.uint8)

    def true_proj_xy(self) -> np.ndarray:
        """(H, W, 2) float32 exact projector coordinates of every camera pixel's surface point."""
        return self._map

    def show(self, proj_frame: np.ndarray) -> None:
        """Put a frame (gray or BGR, projector size) on the fake projector."""
        f = np.asarray(proj_frame)
        self._frame = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f.copy()

    def capture(self) -> np.ndarray:
        """What the camera sees: the projector frame sampled at each pixel's projector coordinate
        (bilinear, 0 outside) + ambient + Gaussian noise, uint8 BGR."""
        img = cv2.remap(self._frame, self._map[..., 0], self._map[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out = img.astype(np.float32) + self.ambient
        if self.noise_sigma > 0:
            out += self._rng.standard_normal(out.shape, dtype=np.float32) * self.noise_sigma
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def depth(self) -> np.ndarray:
        """(H, W) float32 metres: true depth + fresh Gaussian noise (depth_noise_m)."""
        n = self._rng.standard_normal(self.true_depth.shape, dtype=np.float32) * self.depth_noise_m
        return self.true_depth + n

    def conf(self) -> np.ndarray:
        """(H, W) uint8 ARKit-style confidence: 2, 1 within edge_px of a box edge."""
        return self._conf.copy()


# --- procedure -----------------------------------------------------------------------------

def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def capture_mean(pattern_bgr: np.ndarray, capture_fn: CaptureFn, show_fn: ShowFn, settle: int, n: int) -> np.ndarray:
    """Show a projector frame, drop `settle` camera frames, return the mean gray (float32) of the next n."""
    show_fn(pattern_bgr)
    for _ in range(settle):
        capture_fn()
    acc = _gray_f32(capture_fn()).copy()
    for _ in range(n - 1):
        acc += _gray_f32(capture_fn())
    return acc / float(n)


def run_calibration3d(cfg: dict, cam_size: tuple[int, int], proj_size: tuple[int, int], capture_fn: CaptureFn,
                      show_fn: ShowFn, depth_fn: Callable[[], np.ndarray], conf_fn: Callable[[], np.ndarray],
                      log: Callable[[str], None] = print, *, K_cam: np.ndarray,
                      dist_cam: np.ndarray | None = None) -> Calibration3D:
    """Full procedure: depth + confidence of the static scene (median / min over depth_frames), white and
    black references, the Gray code sequence (frames_per_pattern averaged after settle_frames), decode,
    keep pixels with a valid decode and confidence >= min_depth_confidence, lift to 3D, solve.
    cfg = config["calibration3d"]; K_cam = intrinsics of the camera-space image (depth at camera res).
    Raises RuntimeError when too few correspondences survive."""
    t0 = time.perf_counter()
    cw, ch = int(cam_size[0]), int(cam_size[1])
    settle, n = int(_opt(cfg, "settle_frames")), max(1, int(cfg["frames_per_pattern"]))
    patterns, white, black = gray_code_patterns(proj_size)
    black_bgr = _to_bgr(black)
    log(f"depth: {_opt(cfg, 'depth_frames')} frames of the static scene")
    show_fn(black_bgr)
    for _ in range(settle):
        capture_fn()
    depths, confs = [], []
    for _ in range(max(1, int(_opt(cfg, "depth_frames")))):
        capture_fn()
        depths.append(np.asarray(depth_fn(), np.float32))
        confs.append(np.asarray(conf_fn()))
    for a in (depths[0], confs[0]):
        if a.shape[:2] != (ch, cw):
            raise ValueError(f"depth/conf maps are {a.shape[1]}x{a.shape[0]}, camera space is {cw}x{ch}")
    dstack = np.stack(depths)
    depth = np.median(dstack, axis=0)
    good_depth = np.all(dstack > 0, axis=0) & (np.min(np.stack(confs), axis=0) >= int(cfg["min_depth_confidence"]))
    del depths, confs, dstack
    log(f"gray code: {len(patterns)} patterns + white/black, {n} frame(s) each after {settle} settle frames")
    white_cap = capture_mean(_to_bgr(white), capture_fn, show_fn, settle, n)
    if white_cap.shape != (ch, cw):
        raise ValueError(f"capture_fn returned {white_cap.shape[1]}x{white_cap.shape[0]} frames, "
                         f"camera space is {cw}x{ch}")
    black_cap = capture_mean(black_bgr, capture_fn, show_fn, settle, n)
    caps = [capture_mean(_to_bgr(p), capture_fn, show_fn, settle, n) for p in patterns]
    show_fn(black_bgr)
    td = time.perf_counter()
    proj_xy, valid = decode_gray_code(caps, white_cap, black_cap, proj_size, cfg)
    log(f"decode: {valid.sum()} valid pixels ({valid.mean():.1%}) in {(time.perf_counter() - td) * 1e3:.0f} ms")
    del caps
    use = valid & good_depth
    ys, xs = np.nonzero(use)
    if len(xs) < int(_opt(cfg, "min_points")):
        raise RuntimeError(f"only {len(xs)} decoded pixels with depth confidence >= {cfg['min_depth_confidence']} "
                           f"(need {_opt(cfg, 'min_points')}). Fix: dim the room, make sure the projection covers "
                           f"the boxes and wall, keep everyone out of the scene")
    pix = np.stack([xs, ys], axis=1).astype(np.float64)
    pts3d = lift_to_3d(pix, depth, K_cam)
    ok = np.isfinite(pts3d).all(axis=1)
    pts3d, pts2d = pts3d[ok], proj_xy[ys[ok], xs[ok]].astype(np.float64)
    log(f"solve: {len(pts3d)} correspondences (depth range {pts3d[:, 2].min():.2f}-{pts3d[:, 2].max():.2f} m)")
    K_p, dist_p, R, t, rms, inl = solve_projector(pts3d, pts2d, proj_size, float(cfg["throw_ratio_guess"]), cfg)
    log(f"reprojection error: {rms:.3f} px RMS over {inl.sum()} inliers (projector px; "
        f"target {cfg['target_reproj_px']} px), total {time.perf_counter() - t0:.1f} s")
    if rms > float(cfg["target_reproj_px"]):
        log(f"WARNING: reprojection {rms:.2f} px > target {cfg['target_reproj_px']} px. Fix: add a second target "
            "configuration (move/rotate the boxes) and recalibrate, keep the rig and boxes still")
    notes = (f"gray code {len(patterns)} patterns, frames_per_pattern {n}, settle {settle}, "
             f"min_depth_confidence {cfg['min_depth_confidence']}, ransac {cfg['ransac_reproj_px']} px, "
             f"{len(pts3d)} correspondences, throw guess {cfg['throw_ratio_guess']}")
    return Calibration3D(K_cam=np.asarray(K_cam, np.float64).copy(),
                         dist_cam=np.zeros(5) if dist_cam is None else np.asarray(dist_cam, np.float64).ravel(),
                         K_p=K_p, dist_p=dist_p, R=R, t=t, cam_size=(cw, ch),
                         proj_size=(int(proj_size[0]), int(proj_size[1])), reproj_error_px=rms,
                         n_points=int(inl.sum()), timestamp=datetime.now().isoformat(timespec="seconds"),
                         notes=notes)


# --- verification --------------------------------------------------------------------------

def verification_points(c: Calibration3D, depth_cam: np.ndarray, K_cam: np.ndarray,
                        cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """3D points (camera coords) of the verification pattern from the measured depth: (grid (N, 3), edges (M, 3)).
    grid: where X or Y crosses a multiple of verify_grid_m between neighbouring pixels of one surface
    (interpolated onto the line); edges: the nearer pixel of every neighbour pair whose depth differs by
    more than verify_edge_m (box silhouettes)."""
    cfg = cfg or {}
    step, edge_m = float(_opt(cfg, "verify_grid_m")), float(_opt(cfg, "verify_edge_m"))
    h, w = depth_cam.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.asarray(depth_cam, np.float64)
    good = np.isfinite(z) & (z > 0)
    z = np.where(good, z, np.nan)
    K = np.asarray(K_cam, np.float64)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    grid, edges = [], []
    for a, b in ((P[:, :-1], P[:, 1:]), (P[:-1], P[1:])):       # right and down neighbours
        with np.errstate(invalid="ignore"):
            dz = np.abs(a[..., 2] - b[..., 2])
            same = dz <= edge_m
            for axis in (0, 1):
                ka, kb = np.floor(a[..., axis] / step), np.floor(b[..., axis] / step)
                m = same & (ka != kb)
                line = np.maximum(ka[m], kb[m]) * step
                s = ((line - a[..., axis][m]) / (b[..., axis][m] - a[..., axis][m]))[:, None]
                grid.append(a[m] + s * (b[m] - a[m]))
            m = dz > edge_m
            near = np.where((a[..., 2] < b[..., 2])[..., None], a, b)
            edges.append(near[m])
    return np.concatenate(grid), np.concatenate(edges)


def verification_pattern(c: Calibration3D, depth_cam: np.ndarray, K_cam: np.ndarray, proj_size: tuple[int, int],
                         cfg: dict | None = None) -> np.ndarray:
    """(ph, pw, 3) BGR projector image: a verify_grid_m grid drawn on the measured wall / box surfaces
    (white) plus the box silhouette edges (green), projected through the calibration. Projected onto the
    scene, the green lines must land on the physical box edges and the grid must look straight on each face."""
    cfg = cfg or {}
    pw, ph = int(proj_size[0]), int(proj_size[1])
    out = np.zeros((ph, pw, 3), np.uint8)
    k = np.ones((int(_opt(cfg, "verify_line_px")),) * 2, np.uint8)
    for pts, color in zip(verification_points(c, depth_cam, K_cam, cfg), (GRID_COLOR, EDGE_COLOR)):
        if not len(pts):
            continue
        q = np.rint(project_to_projector(c, pts)).astype(np.int64)
        q = q[(q[:, 0] >= 0) & (q[:, 0] < pw) & (q[:, 1] >= 0) & (q[:, 1] < ph)]
        m = np.zeros((ph, pw), np.uint8)
        m[q[:, 1], q[:, 0]] = 255
        out[cv2.dilate(m, k) > 0] = color
    return out
