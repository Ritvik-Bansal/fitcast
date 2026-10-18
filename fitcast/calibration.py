"""Planar homography calibration: camera space -> projector space, plus the gallery rect.

Camera space is the processed frame (`capture.width` x `capture.height`), projector space
is `calibration.projector_width` x `projector_height`. `H` maps camera px -> projector px.

Also here: the marker-pattern helpers, a headless `SimulatedRig`, and `run_calibration`,
the procedure shared by `scripts/calibrate.py` (--simulate and live) and the tests.
Pixel coordinates use OpenCV's pixel-center convention (pixel i has center i).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Keys not (yet) in config.yaml `calibration:`; used only when the config omits them.
_DEFAULTS = {
    "diff_threshold_min": 12,     # adaptive threshold floor
    "adaptive_frac": 0.4,         # threshold = this fraction of the lit-dark contrast peak

    "max_blob_area_px": 6000,     # larger blobs are people / glare, not a marker disc
    "min_marker_mean_diff": 20,   # a real disc is clearly brighter than the ambient change

    "min_fill_ratio": 0.6,      # blob area / bbox area; a disc is ~0.785
    "max_aspect": 2.0,          # bbox aspect ratio limit

    "blur_ksize": 5,            # Gaussian blur on the lit-dark difference before thresholding (odd)
    "ransac_reproj_px": 8.0,    # RANSAC inlier threshold, projector px (~5 camera px; manual clicks)
}

_FIX = "run `.venv/bin/python scripts/calibrate.py` (camera + projector) to create it"

CaptureFn = Callable[[], np.ndarray]
ShowFn = Callable[[np.ndarray], None]


def _opt(cfg: dict, key: str):
    return cfg.get(key, _DEFAULTS[key])


@dataclass
class CalibrationData:
    """Everything stored in calibration.yaml."""
    H: np.ndarray                       # (3, 3) float64 camera -> projector
    camera_size: tuple[int, int]        # (w, h) camera space H was computed in
    projector_size: tuple[int, int]     # (w, h)
    reproj_error_px: float              # RMS over all markers, projector px
    timestamp: str                      # ISO 8601
    gallery_rect: list[int] | None      # x0, y0, x1, y1 camera px
    verified: bool
    verify_max_err_px: float | None     # camera px
    simulated: bool = False             # produced by --simulate: never valid for a real camera
    depth_ref_torso_px: float | None = None   # torso length (px) at the standing spot H was refined for
    depth_slope: list[float] | None = None    # camera-px shift per px of torso length away from that spot (dx, dy)
    H2: np.ndarray | None = None              # second camera->projector homography at another standing distance ...
    depth_ref2_torso_px: float | None = None  # ... with the torso length (px) measured there (self-calibration)
    D_upper: np.ndarray | None = None         # projector-space correction for the upper body (above the hips) ...
    D_lower: np.ndarray | None = None         # ... and the lower body: H_part = D_part @ H  (dense self-calibration)


def _size(s) -> tuple[int, int]:
    w, h = int(s[0]), int(s[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"size must be positive, got {s}")
    return w, h


def _apply(M: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) or (2,) points; returns (N, 2) float64."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, M).reshape(-1, 2)


def _resize_matrix(src: tuple[int, int], dst: tuple[int, int]) -> np.ndarray:
    """Scale matrix mapping pixel coords of a (w, h)=src image to the same scene point after
    cv2.resize to dst. Includes the half-pixel-center offset: x' = (x + 0.5) * s - 0.5."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])


def _rescale_data(d: CalibrationData, new_size: tuple[int, int]) -> CalibrationData:
    """Return a copy of `d` whose H and gallery rect live in camera space `new_size`."""
    H = d.H @ np.linalg.inv(_resize_matrix(d.camera_size, new_size))
    sx, sy = new_size[0] / d.camera_size[0], new_size[1] / d.camera_size[1]
    rect = None
    if d.gallery_rect is not None:
        x0, y0, x1, y1 = d.gallery_rect
        rect = [round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)]
    verr = None if d.verify_max_err_px is None else d.verify_max_err_px * (sx + sy) / 2
    return dataclasses.replace(d, H=H / H[2, 2], camera_size=new_size, gallery_rect=rect,
                               verify_max_err_px=verr, simulated=d.simulated,
                               depth_ref_torso_px=None if d.depth_ref_torso_px is None else d.depth_ref_torso_px * (sx + sy) / 2,
                               depth_slope=d.depth_slope, H2=None if d.H2 is None else (lambda M: M / M[2, 2])(d.H2 @ np.linalg.inv(_resize_matrix(d.camera_size, new_size))),
                               depth_ref2_torso_px=None if d.depth_ref2_torso_px is None else d.depth_ref2_torso_px * (sx + sy) / 2, D_upper=d.D_upper, D_lower=d.D_lower)


def _data_from_dict(raw) -> CalibrationData:
    """Parse and validate a calibration.yaml mapping. Raises ValueError/KeyError/TypeError."""
    if not isinstance(raw, dict):
        raise ValueError("file is not a YAML mapping")
    H = np.asarray(raw["H"], np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H has shape {H.shape}, expected (3, 3)")
    if not np.all(np.isfinite(H)) or H[2, 2] == 0:
        raise ValueError("H has non-finite values or H[2][2] == 0")
    if not np.all(np.isfinite(np.linalg.inv(H))):   # LinAlgError when singular
        raise ValueError("H is not invertible")
    rect = raw.get("gallery_rect")
    verr = raw.get("verify_max_err_px")
    return CalibrationData(
        H=H / H[2, 2],
        camera_size=_size(raw["camera_size"]),
        projector_size=_size(raw["projector_size"]),
        reproj_error_px=float(raw["reproj_error_px"]),
        timestamp=str(raw.get("timestamp", "")),
        gallery_rect=None if rect is None else [int(v) for v in rect],
        verified=bool(raw.get("verified", False)),
        verify_max_err_px=None if verr is None else float(verr),
        simulated=bool(raw.get("simulated", False)),
        depth_ref_torso_px=None if raw.get("depth_ref_torso_px") is None else float(raw["depth_ref_torso_px"]),
        depth_slope=None if raw.get("depth_slope") is None else [float(v) for v in raw["depth_slope"]],
        H2=None if raw.get("H2") is None else np.asarray(raw["H2"], np.float64),
        depth_ref2_torso_px=None if raw.get("depth_ref2_torso_px") is None else float(raw["depth_ref2_torso_px"]),
        D_upper=None if raw.get("D_upper") is None else np.asarray(raw["D_upper"], np.float64),
        D_lower=None if raw.get("D_lower") is None else np.asarray(raw["D_lower"], np.float64),
    )


class Calibration:
    """Camera -> projector homography: load/save, point mapping and the per-frame warp."""

    def __init__(self, cfg: dict, camera_size: tuple[int, int]) -> None:
        """cfg is config["calibration"]; camera_size is the runtime (w, h) camera space."""
        self.cfg = cfg
        self.camera_size = _size(camera_size)
        self.projector_size = _size((cfg["projector_width"], cfg["projector_height"]))
        self.data: CalibrationData | None = None
        self._H_inv: np.ndarray | None = None
        self._buf: np.ndarray | None = None
        self._buf2 = None
        self._letterbox_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._warned_fallback = False

    @property
    def ok(self) -> bool:
        """True when a usable homography is loaded or computed."""
        return self.data is not None

    @property
    def H(self) -> np.ndarray | None:
        """The camera -> projector homography, or None when not calibrated."""
        return None if self.data is None else self.data.H

    def _set_data(self, d: CalibrationData) -> None:
        H_inv = np.linalg.inv(d.H)
        self._H_inv = H_inv / H_inv[2, 2]
        self.data = d

    def _require(self) -> CalibrationData:
        if self.data is None:
            raise RuntimeError(f"no calibration loaded; {_FIX}")
        return self.data

    def load(self, path: str | None = None, allow_simulated: bool = False) -> bool:
        """Load calibration.yaml (default cfg['file']). Returns False, with a fix-it log line,
        when the file is missing or invalid. Rescales H when it was made at another camera size.
        A file written by `calibrate.py --simulate` is refused unless `allow_simulated` (video-file runs)."""
        p = Path(path or self.cfg["file"])
        if not p.is_file():
            logger.error("Calibration file %s not found: projector shows an uncalibrated "
                         "letterboxed camera image. Fix: %s.", p, _FIX)
            return False
        try:
            d = _data_from_dict(yaml.safe_load(p.read_text()))
        except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError,
                np.linalg.LinAlgError) as e:
            logger.error("Calibration file %s is invalid (%s). Fix: %s.", p, e, _FIX)
            return False
        if d.simulated and not allow_simulated:
            logger.error("Calibration file %s is a SIMULATED calibration (fake homography). Refusing to use it "
                         "with a real camera: the garment and the head blackout would land in the wrong place. "
                         "Fix: %s.", p, _FIX)
            return False
        if d.reproj_error_px > float(self.cfg.get("max_reproj_px", 10.0)):
            logger.error("Calibration file %s has reprojection error %.1f px (limit %.1f): refusing it. Fix: %s.",
                         p, d.reproj_error_px, float(self.cfg.get("max_reproj_px", 10.0)), _FIX)
            return False
        if d.projector_size != self.projector_size:
            logger.error("Calibration file %s was made for projector %s but config says %s. "
                         "Fix: %s.", p, d.projector_size, self.projector_size, _FIX)
            return False
        if d.camera_size != self.camera_size:
            logger.warning("Calibration %s was made at camera size %s, runtime is %s: rescaling H. "
                           "Re-run scripts/calibrate.py at this size for best accuracy.",
                           p, d.camera_size, self.camera_size)
            d = _rescale_data(d, self.camera_size)
        self._set_data(d)
        logger.info("Loaded calibration %s (reproj %.2f px, verified=%s, %s)",
                    p, d.reproj_error_px, d.verified, d.timestamp)
        return True

    def save(self, path: str | None = None) -> None:
        """Write the current calibration to YAML (default cfg['file'])."""
        d = self._require()
        p = Path(path or self.cfg["file"])
        doc = {
            "H": [[float(v) for v in row] for row in d.H],
            "camera_size": list(d.camera_size),
            "projector_size": list(d.projector_size),
            "reproj_error_px": float(d.reproj_error_px),
            "timestamp": d.timestamp,
            "gallery_rect": None if d.gallery_rect is None else [int(v) for v in d.gallery_rect],
            "verified": bool(d.verified),
            "verify_max_err_px": None if d.verify_max_err_px is None else float(d.verify_max_err_px),
            "simulated": bool(d.simulated),
            "depth_ref_torso_px": None if d.depth_ref_torso_px is None else float(d.depth_ref_torso_px),
            "depth_slope": None if d.depth_slope is None else [float(v) for v in d.depth_slope],
            "H2": None if d.H2 is None else [[float(v) for v in row] for row in d.H2],
            "depth_ref2_torso_px": None if d.depth_ref2_torso_px is None else float(d.depth_ref2_torso_px),
            "D_upper": None if d.D_upper is None else [[float(v) for v in row] for row in d.D_upper],
            "D_lower": None if d.D_lower is None else [[float(v) for v in row] for row in d.D_lower],
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# FitCast calibration: H maps camera px -> projector px. "
                     "Written by scripts/calibrate.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        logger.info("Saved calibration to %s", p)

    def _letterbox(self, w: int, h: int) -> np.ndarray:
        if self._letterbox_cache is None or self._letterbox_cache[0] != (w, h):
            pw, ph = self.projector_size
            s = min(pw / w, ph / h)
            M = np.array([[s, 0.0, (pw - w * s) / 2], [0.0, s, (ph - h * s) / 2], [0.0, 0.0, 1.0]])
            self._letterbox_cache = ((w, h), M)
        return self._letterbox_cache[1]

    def depth_shift(self, torso_px: float | None) -> tuple[float, float]:
        """Camera-space shift to apply for a person whose torso length is `torso_px`, from the two-point depth
        model recorded by `calibrate.py --refine` (zero when the model is absent)."""
        d = self.data
        if d is None or torso_px is None or d.depth_ref_torso_px is None or d.depth_slope is None:
            return 0.0, 0.0
        dt = float(torso_px) - float(d.depth_ref_torso_px)
        return float(d.depth_slope[0]) * dt, float(d.depth_slope[1]) * dt

    def H_for(self, torso_px: float | None) -> np.ndarray:
        """Camera->projector homography for a person whose torso length is `torso_px`: linear interpolation
        between the two self-calibrated planes (exact for a pinhole pair, since H is affine in 1/distance and
        1/distance is proportional to the apparent torso length). Falls back to H without a second plane."""
        d = self._require()
        if torso_px is None or d.H2 is None or d.depth_ref_torso_px is None or d.depth_ref2_torso_px is None:
            return d.H
        t0, t2 = float(d.depth_ref_torso_px), float(d.depth_ref2_torso_px)
        if abs(t2 - t0) < 1e-6:
            return d.H
        a = (float(torso_px) - t0) / (t2 - t0)
        a = float(np.clip(a, float(self.cfg.get("depth_interp_min", -0.5)), float(self.cfg.get("depth_interp_max", 1.5))))
        H = d.H + a * (d.H2 - d.H)
        return H / H[2, 2]

    def set_plane(self, H: np.ndarray, torso_px: float, change_frac: float = 0.25) -> str:
        """Store a self-calibrated plane. Same distance as the reference (within `change_frac`) -> replace the
        primary H; clearly different -> becomes the second plane. Returns 'primary' | 'second'."""
        H = np.asarray(H, np.float64); H = H / H[2, 2]
        d = self.data
        if d is None or d.depth_ref_torso_px is None or abs(torso_px - d.depth_ref_torso_px) / d.depth_ref_torso_px <= change_frac:
            if d is None:
                d = CalibrationData(H=H, camera_size=self.camera_size, projector_size=self.projector_size,
                                    reproj_error_px=0.0, timestamp=datetime.now().isoformat(timespec="seconds"),
                                    gallery_rect=None, verified=False, verify_max_err_px=None)
            d.H, d.depth_ref_torso_px, d.depth_slope = H, float(torso_px), None
            self._set_data(d)
            return "primary"
        d.H2, d.depth_ref2_torso_px, d.depth_slope = H, float(torso_px), None
        self._set_data(d)
        return "second"

    def to_projector(self, img_cam: np.ndarray, shift: tuple[float, float] | None = None,
                     torso_px: float | None = None, split_y: float | None = None) -> np.ndarray:
        """Warp a camera-space image into the (ph, pw) projector frame. `shift` = (dx, dy) camera px the light
        would land away from the intended spot at the person's current depth (see `depth_shift`); it is
        compensated by pre-translating the image.

        Returns an internal buffer reused on every call: copy it to keep it past the next call.
        Without a calibration the image is letterboxed into the frame instead (logged once).
        Raises ValueError when calibrated and the image is not camera_size.
        """
        h, w = img_cam.shape[:2]
        if self.data is not None:
            if (w, h) != self.camera_size:
                raise ValueError(f"to_projector got a {w}x{h} image, calibration camera space is "
                                 f"{self.camera_size[0]}x{self.camera_size[1]}")
            M = self.H_for(torso_px) if torso_px is not None else self.data.H
            if shift is not None and (shift[0] or shift[1]):
                M = M @ np.array([[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]], [0.0, 0.0, 1.0]])
            if split_y is not None and self.data.D_upper is not None and self.data.D_lower is not None:
                return self._warp_split(img_cam, M, float(split_y))
        else:
            if not self._warned_fallback:
                logger.warning("No calibration: projecting a letterboxed camera image. Fix: %s.", _FIX)
                self._warned_fallback = True
            M = self._letterbox(w, h)
        shape = (self.projector_size[1], self.projector_size[0]) + img_cam.shape[2:]
        if self._buf is None or self._buf.shape != shape or self._buf.dtype != img_cam.dtype:
            self._buf = np.zeros(shape, img_cam.dtype)
        return cv2.warpPerspective(img_cam, M, self.projector_size, dst=self._buf,
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    def _warp_split(self, img_cam: np.ndarray, M: np.ndarray, split_y: float) -> np.ndarray:
        """Upper body through D_upper @ M, lower body through D_lower @ M, feathered across the hip line
        (`split_y`, camera px) over `seam_feather_px` projector rows. Chest and shins are not one plane when the
        camera sits well away from the projector lens; the dense scan measures both."""
        pw, ph = self.projector_size
        d = self.data
        Mu, Ml = d.D_upper @ M, d.D_lower @ M
        shape = (ph, pw) + img_cam.shape[2:]
        if self._buf is None or self._buf.shape != shape or self._buf.dtype != img_cam.dtype:
            self._buf = np.zeros(shape, img_cam.dtype)
        if self._buf2 is None or self._buf2.shape != shape or self._buf2.dtype != img_cam.dtype:
            self._buf2 = np.zeros(shape, img_cam.dtype)
        cv2.warpPerspective(img_cam, Mu, (pw, ph), dst=self._buf, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.warpPerspective(img_cam, Ml, (pw, ph), dst=self._buf2, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        xc = np.array([[[self.camera_size[0] / 2.0, split_y]]], np.float64)
        ys = 0.5 * (cv2.perspectiveTransform(xc, Mu)[0, 0, 1] + cv2.perspectiveTransform(xc, Ml)[0, 0, 1])
        f = int(self.cfg.get("seam_feather_px", 60))
        y0 = int(np.clip(round(ys - f / 2), 0, ph))
        y1 = int(np.clip(round(ys + f / 2), 0, ph))
        out = self._buf
        if y1 < ph:
            out[y1:] = self._buf2[y1:]
        if y1 > y0:
            w = np.linspace(0.0, 1.0, y1 - y0, dtype=np.float32).reshape(-1, 1, *([1] * (out.ndim - 2)))
            band = out[y0:y1] * (1.0 - w) + self._buf2[y0:y1] * w      # ponytail: a ~60-row slab; fine per frame
            out[y0:y1] = np.rint(band).astype(out.dtype)
        return out

    def set_homography(self, cam_pts: np.ndarray, proj_pts: np.ndarray) -> float:
        """Solve H from point pairs (RANSAC when > 4, exact when 4), store it, and return the
        RMS reprojection error in projector px over ALL points (inliers and outliers)."""
        cam = np.asarray(cam_pts, np.float64).reshape(-1, 2)
        proj = np.asarray(proj_pts, np.float64).reshape(-1, 2)
        n = len(cam)
        if n < 4 or len(proj) != n:
            raise ValueError(f"need >= 4 matching point pairs, got {n} camera / {len(proj)} projector")
        if n == 4:
            H, mask = cv2.findHomography(cam, proj, 0)
        else:
            H, mask = cv2.findHomography(cam, proj, cv2.RANSAC, float(_opt(self.cfg, "ransac_reproj_px")))
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < np.finfo(float).eps:
            raise RuntimeError("findHomography failed: points are degenerate (collinear, duplicated "
                               "or clicked out of order?)")
        H = H / H[2, 2]
        sq = np.sum((_apply(H, cam) - proj) ** 2, axis=1)
        err = float(np.sqrt(np.mean(sq)))
        if mask is not None and not mask.all():
            inl = mask.ravel() != 0
            logger.warning("RANSAC rejected %d of %d points (indices %s, inlier RMS %.2f px): "
                           "mis-detected or mis-clicked markers", int((~inl).sum()), n,
                           np.flatnonzero(~inl).tolist(), float(np.sqrt(np.mean(sq[inl]))))
        self._set_data(CalibrationData(
            H=H, camera_size=self.camera_size, projector_size=self.projector_size,
            reproj_error_px=err, timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            gallery_rect=None if self.data is None else self.data.gallery_rect,
            verified=False, verify_max_err_px=None))
        return err

    def project_points(self, cam_pts) -> np.ndarray:
        """Camera px -> projector px, (N, 2) float64."""
        return _apply(self._require().H, cam_pts)

    def unproject_points(self, proj_pts) -> np.ndarray:
        """Projector px -> camera px (inverse H), (N, 2) float64."""
        self._require()
        return _apply(self._H_inv, proj_pts)

    def rescaled(self, new_camera_size: tuple[int, int]) -> "Calibration":
        """A new Calibration whose H (and gallery rect) apply to frames resized to new_camera_size."""
        out = Calibration(self.cfg, new_camera_size)
        if self.data is not None:
            out._set_data(_rescale_data(self.data, out.camera_size))
        return out


# --- marker patterns -----------------------------------------------------------------------

def _grid(cfg: dict) -> tuple[int, int]:
    cols, rows = (int(v) for v in cfg["marker_grid"])
    if cols < 2 or rows < 2:
        raise ValueError(f"calibration.marker_grid must be at least [2, 2], got {cfg['marker_grid']}")
    return cols, rows


def _lattice(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys)        # row-major: top row first, left to right
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)


def marker_points(cfg: dict) -> np.ndarray:
    """(cols*rows, 2) projector px marker centers: a `marker_grid` lattice inset by
    `marker_margin_frac`, in row-major order (the order detect_markers returns)."""
    cols, rows = _grid(cfg)
    pw, ph, m = cfg["projector_width"], cfg["projector_height"], cfg["marker_margin_frac"]
    return _lattice(np.linspace(m * pw, (1 - m) * pw, cols), np.linspace(m * ph, (1 - m) * ph, rows))


def _verify_points(cfg: dict) -> np.ndarray:
    """Cell centers between calibration markers: a (cols-1) x (rows-1) row-major lattice."""
    g = marker_points(cfg)
    cols, rows = _grid(cfg)
    xs, ys = g[:cols, 0], g[::cols, 1]
    return _lattice((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)


def black_frame(cfg: dict) -> np.ndarray:
    """An all-black (ph, pw, 3) projector frame."""
    return np.zeros((cfg["projector_height"], cfg["projector_width"], 3), np.uint8)


def render_markers(cfg: dict, pts: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """(ph, pw, 3) black frame with white filled circles of `marker_radius_px` at `pts`
    (sub-pixel centers). `active` (bool per point) selects which to draw; None draws all."""
    img = black_frame(cfg)
    r = int(cfg["marker_radius_px"]) * 16
    for i, (x, y) in enumerate(np.asarray(pts, np.float64).reshape(-1, 2)):
        if active is None or active[i]:
            cv2.circle(img, (round(x * 16), round(y * 16)), r, (255, 255, 255), -1, cv2.LINE_AA, 4)
    return img


def _diff_gray(frame_lit: np.ndarray, frame_dark: np.ndarray) -> np.ndarray:
    """Per-pixel maximum over colour channels of (lit - dark): projector white reads bluish under warm
    room light, so the strongest channel carries 2-3x the contrast of a grey conversion."""
    diff = cv2.subtract(frame_lit, frame_dark)
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    return diff


def _marker_binary(diff_gray: np.ndarray, cfg: dict, peak: float | None = None) -> np.ndarray:
    """Blur, remove the global brightness shift between captures (auto-exposure / projector black level),
    then threshold adaptively: `adaptive_frac` of the (peak - shift) contrast, clamped to
    [diff_threshold_min, diff_threshold]. `peak` defaults to the 99.9th percentile (several discs);
    single-marker callers pass the max."""
    k = int(_opt(cfg, "blur_ksize")) | 1
    d = cv2.GaussianBlur(diff_gray, (k, k), 0)
    shift = float(np.median(d))
    if shift > 0:
        d = cv2.subtract(d, int(round(shift)))
    pk = float(np.percentile(d, 99.9)) if peak is None else float(peak) - shift
    thr = _opt(cfg, "adaptive_frac") * pk
    thr = float(min(cfg["diff_threshold"], max(_opt(cfg, "diff_threshold_min"), thr)))
    _, binary = cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)
    return binary


def detect_one_marker(frame_lit: np.ndarray, frame_dark: np.ndarray, cfg: dict) -> np.ndarray | None:
    """Centroid (2,) of the single projected disc in `frame_lit` (one marker lit) or None.
    Candidates are disc-shaped blobs of plausible size; the one with the highest mean difference
    intensity wins (a projected disc is the brightest change in the scene, a moving person is not)."""
    diff = _diff_gray(frame_lit, frame_dark)
    binary = _marker_binary(diff, cfg, peak=float(cv2.GaussianBlur(diff, (5, 5), 0).max()))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    stats, cents = stats[1:], cents[1:]
    area = stats[:, cv2.CC_STAT_AREA].astype(float)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH].astype(float), stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    fill = area / np.maximum(bw * bh, 1.0)
    aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1.0)
    keep = ((area >= cfg["min_blob_area_px"]) & (area <= _opt(cfg, "max_blob_area_px"))
            & (fill >= _opt(cfg, "min_fill_ratio")) & (aspect <= _opt(cfg, "max_aspect")))
    if not keep.any():
        return None
    best, best_score = None, -1.0
    for i in np.flatnonzero(keep):
        score = float(cv2.mean(diff, mask=(labels == i + 1).astype(np.uint8))[0])
        if score > best_score:
            best, best_score = i, score
    if best_score < _opt(cfg, "min_marker_mean_diff"):
        return None                      # absolute floor: a disc must stand out of the noise
    return cents[best].astype(np.float64)


def capture_median(pattern: np.ndarray, capture_fn: CaptureFn, show_fn: ShowFn, settle_frames: int,
                   n: int = 3) -> np.ndarray:
    """Like capture_pattern but returns the per-pixel median of `n` frames (kills moving people)."""
    show_fn(pattern)
    for _ in range(int(settle_frames)):
        capture_fn()
    frames = np.stack([capture_fn() for _ in range(max(1, int(n)))])
    return np.median(frames, axis=0).astype(np.uint8)


def detect_markers(frame_lit: np.ndarray, frame_dark: np.ndarray, cfg: dict,
                   expected: int, rows: int) -> np.ndarray | None:
    """Find projected markers in camera frames: (expected, 2) camera px centroids in projector
    grid order (sorted by y, split into `rows` rows, each sorted by x), or None when the blob
    count after the `min_blob_area_px` filter is not `expected`."""
    if rows <= 0 or expected % rows:
        raise ValueError(f"expected={expected} markers do not split into {rows} rows")
    diff = _diff_gray(frame_lit, frame_dark)
    binary = _marker_binary(diff, cfg)
    _, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    stats, cents = stats[1:], cents[1:]
    area = stats[:, cv2.CC_STAT_AREA].astype(float)
    bw, bh = stats[:, cv2.CC_STAT_WIDTH].astype(float), stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    # projected markers are filled discs: reject thin / sparse blobs (moving people, edges, glare streaks)
    fill = area / np.maximum(bw * bh, 1.0)
    aspect = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1.0)
    keep = (area >= cfg["min_blob_area_px"]) & (fill >= _opt(cfg, "min_fill_ratio")) & (aspect <= _opt(cfg, "max_aspect"))
    pts, area = cents[keep], area[keep]
    if len(pts) > expected:
        # discs of one grid have similar areas: keep the `expected` blobs nearest the median area
        med = np.median(area)
        order = np.argsort(np.abs(area - med), kind="stable")[:expected]
        pts = pts[np.sort(order)]
    if len(pts) != expected:
        logger.warning("detect_markers: found %d blobs, expected %d. Fix: dim the room, make sure "
                       "the camera sees the whole projection, or tune calibration.diff_threshold "
                       "(%s) / min_blob_area_px (%s).", len(pts), expected,
                       cfg["diff_threshold"], cfg["min_blob_area_px"])
        return None
    pts = pts[np.argsort(pts[:, 1], kind="stable")]
    return np.concatenate([r[np.argsort(r[:, 0], kind="stable")] for r in np.split(pts, rows)])


def capture_pattern(pattern: np.ndarray, capture_fn: CaptureFn, show_fn: ShowFn,
                    settle_frames: int) -> np.ndarray:
    """Show a projector frame, discard `settle_frames` camera frames (projector latency,
    camera buffering, auto-exposure), and return the next camera frame."""
    show_fn(pattern)
    for _ in range(int(settle_frames)):
        capture_fn()
    return capture_fn()


def verify_grid(cal: Calibration, capture_fn: CaptureFn, show_fn: ShowFn,
                cfg: dict) -> tuple[bool, float]:
    """Project markers at the cell centers between the calibration markers, detect them, and
    compare with unproject(projector points). Returns (passed, max error in camera px; inf when
    detection failed) and records the result on cal.data. Leaves the projector black."""
    _, rows = _grid(cfg)
    proj = _verify_points(cfg)
    settle = int(cfg["settle_frames"])
    black = black_frame(cfg)
    dark = capture_pattern(black, capture_fn, show_fn, settle)
    lit = capture_pattern(render_markers(cfg, proj), capture_fn, show_fn, settle)
    show_fn(black)
    found = detect_markers(lit, dark, cfg, len(proj), rows - 1)
    if found is None:
        passed, err = False, math.inf
    else:
        err = float(np.max(np.linalg.norm(found - cal.unproject_points(proj), axis=1)))
        passed = err <= float(cfg["verify_tolerance_px"])
    if cal.data is not None:
        cal.data.verified = passed
        cal.data.verify_max_err_px = err if math.isfinite(err) else None
    return passed, err


# --- simulation ----------------------------------------------------------------------------

def make_sim_homography(camera_size: tuple[int, int], projector_size: tuple[int, int],
                        rot_deg: float = 3.0, fill: float = 0.8,
                        offset_px: tuple[float, float] = (20.0, -12.0),
                        keystone: tuple[float, float] = (6e-5, 1e-4)) -> np.ndarray:
    """A plausible camera -> projector H for simulation: the projector image covers `fill` of
    the camera frame, rotated `rot_deg`, shifted `offset_px` and keystoned (perspective terms
    per camera px). Test fixture values, not runtime tunables."""
    (cw, ch), (pw, ph) = camera_size, projector_size
    s = fill * min(cw / pw, ch / ph)
    c, sn = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    to_center = np.array([[1, 0, -pw / 2], [0, 1, -ph / 2], [0, 0, 1]], np.float64)
    rot_scale = np.array([[s * c, -s * sn, 0], [s * sn, s * c, 0], [0, 0, 1]])
    keyst = np.array([[1, 0, 0], [0, 1, 0], [keystone[0], keystone[1], 1]])
    to_cam = np.array([[1, 0, cw / 2 + offset_px[0]], [0, 1, ch / 2 + offset_px[1]], [0, 0, 1]])
    G = to_cam @ keyst @ rot_scale @ to_center      # projector -> camera
    H = np.linalg.inv(G)
    return H / H[2, 2]


class SimulatedRig:
    """Headless fake projector + camera: the camera sees the projector frame through H_true."""

    def __init__(self, cfg: dict, camera_size: tuple[int, int], H_true: np.ndarray | None = None,
                 noise_sigma: float = 2.0, ambient: float = 25, seed: int = 0) -> None:
        """cfg is config["calibration"]; H_true maps camera -> projector (default:
        make_sim_homography). `seed` makes the camera noise deterministic."""
        self.camera_size = _size(camera_size)
        proj_size = _size((cfg["projector_width"], cfg["projector_height"]))
        self.H_true = (make_sim_homography(self.camera_size, proj_size) if H_true is None
                       else np.asarray(H_true, np.float64))
        self.noise_sigma = float(noise_sigma)
        self.ambient = float(ambient)
        self._rng = np.random.default_rng(seed)
        self._frame = black_frame(cfg)

    def show(self, proj_frame: np.ndarray) -> None:
        """Put a frame on the fake projector."""
        self._frame = proj_frame.copy()

    def capture(self) -> np.ndarray:
        """What the camera sees: the projector frame warped into camera space + ambient + noise."""
        img = cv2.warpPerspective(self._frame, self.H_true, self.camera_size,
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        out = img.astype(np.float32) + self.ambient
        if self.noise_sigma > 0:
            out += self._rng.standard_normal(out.shape, dtype=np.float32) * self.noise_sigma
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def run_calibration(cfg: dict, camera_size: tuple[int, int], capture_fn: CaptureFn,
                    show_fn: ShowFn, log: Callable[[str], None] = print) -> Calibration:
    """Automatic procedure: dark frame, marker frame, detect, solve, verify. Returns the
    Calibration (cal.data holds error + verify result). Raises RuntimeError when markers are
    not detected (live mode then falls back to manual clicking)."""
    cols, rows = _grid(cfg)
    pts = marker_points(cfg)
    settle = int(cfg["settle_frames"])
    cam_size = _size(camera_size)
    log("capturing dark frame")
    dark = capture_pattern(black_frame(cfg), capture_fn, show_fn, settle)
    log(f"capturing {cols}x{rows} marker frame")
    lit = capture_pattern(render_markers(cfg, pts), capture_fn, show_fn, settle)
    show_fn(black_frame(cfg))
    if (lit.shape[1], lit.shape[0]) != cam_size:
        raise ValueError(f"capture_fn returned {lit.shape[1]}x{lit.shape[0]} frames, "
                         f"camera space is {cam_size[0]}x{cam_size[1]}")
    cam = detect_markers(lit, dark, cfg, len(pts), rows)
    if cam is None:
        raise RuntimeError("marker detection failed (see warning above); in live mode press m "
                           "to click the markers by hand")
    cal = Calibration(cfg, cam_size)
    err = cal.set_homography(cam, pts)
    log(f"reprojection error: {err:.3f} px (RMS over {len(pts)} markers, projector px)")
    passed, max_err = verify_grid(cal, capture_fn, show_fn, cfg)
    log(f"VERIFY {'PASS' if passed else 'FAIL'}: max error {max_err:.2f} px "
        f"(tolerance {cfg['verify_tolerance_px']} px, camera px)")
    return cal
