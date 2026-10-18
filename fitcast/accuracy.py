"""Projection accuracy metric (docs/BUILD_SPEC_3D.md section 10): how far the projected body boundary lands
from the real body, in millimetres, measured with the camera.

Procedure (driven by `scripts/accuracy_check.py`): a thin band just INSIDE the person mask's boundary
(`outline_image`) is projected through the normal pipeline; the camera frame with the outline lit is
differenced against a frame with the projector dark (`detect_lit_outline`), thinned to a 1-px curve, and every
lit pixel is matched to the nearest pixel of the truth boundary (`truth_boundary_from_depth`: the contour of
the depth-based DepthBodyMask captured together with the lit frame). The pixel distance is converted to mm with
the metric scale at the matched truth point: mm_per_px = depth * 1000 / mean(fx, fy) (`boundary_errors_mm`).

The band is drawn inside the mask so all its light falls on the body: a band straddling the silhouette would be
split by DepthMeshProjector's black depth-gap curtains, its outer half would hit the wall (in the projector
shadow / at the wrong depth). The thinned band centre sits (thickness - 1) / 2 px inside the truth contour:
the metric's own floor (sub-pixel for thickness 2; `tests/test_accuracy.py` pins it).

All functions are pure (no I/O, no globals); camera pixel coordinates are (x, y) with pixel centres at
integers (OpenCV convention).
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibration import _diff_gray, _marker_binary

_HAS_THINNING = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), np.uint8)

# Optional detection keys (accuracy config); the threshold keys come from `calibration:` (see detect_lit_outline).
DEFAULTS = {
    "min_blob_px": 8,       # connected components of the thinned outline smaller than this are noise
}


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be (H, W), got shape {m.shape}")
    return np.where(m > 0, 255, 0).astype(np.uint8) if m.dtype != np.uint8 or m.max() not in (0, 255) else m


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """(N, 2) int32 camera px (x, y) of the outer contours of `mask` (nonzero = inside): cv2.findContours,
    RETR_EXTERNAL, CHAIN_APPROX_NONE (every boundary pixel; the pixels are the outermost mask pixels).
    Holes are not boundaries. Empty mask -> (0, 2)."""
    contours, _ = cv2.findContours(_mask_u8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), np.int32)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours]).astype(np.int32)
    return np.unique(pts, axis=0)


def truth_boundary_from_depth(depth_mask: np.ndarray) -> np.ndarray:
    """(N, 2) int32 camera px of the depth-based silhouette boundary. The DepthBodyMask (depth decides who is
    person, so the projected outline cannot change it) is the ground truth of the accuracy check."""
    return boundary_pixels(depth_mask)


def outline_image(mask: np.ndarray, thickness_px: int, color: tuple[int, int, int] | list[int]) -> np.ndarray:
    """(H, W, 3) BGR uint8 camera-space image: black except a `thickness_px` wide band just inside the mask
    boundary, lit with `color` (BGR). The band is `mask & ~erode(mask)` with a 3x3 square eroded
    `thickness_px` times, so its outer edge is exactly the `boundary_pixels` contour and it never leaves the
    body. The caller must still black out the head box (safety rule) before projecting it."""
    m = _mask_u8(mask)
    t = int(thickness_px)
    if t < 1:
        raise ValueError(f"thickness_px must be >= 1, got {thickness_px}")
    inner = cv2.erode(m, _SQUARE, iterations=t, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    band = cv2.subtract(m, inner)
    out = np.zeros(m.shape + (3,), np.uint8)
    out[band > 0] = np.asarray(color, np.uint8).reshape(3)
    return out


def _skeleton(binary: np.ndarray) -> np.ndarray:
    """Morphological skeleton (Lantuejoul) of a 0/255 image: fallback when cv2.ximgproc.thinning is missing."""
    img = binary.copy()
    skel = np.zeros_like(img)
    while cv2.countNonZero(img):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, _CROSS)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = cv2.erode(img, _CROSS)
    return skel


def thin(binary: np.ndarray) -> np.ndarray:
    """1-px wide centre lines of a 0/255 image (Zhang-Suen via cv2.ximgproc.thinning, else a morphological
    skeleton)."""
    b = _mask_u8(binary)
    if _HAS_THINNING:
        return cv2.ximgproc.thinning(b, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return _skeleton(b)


def detect_lit_outline(frame_lit: np.ndarray, frame_dark: np.ndarray, cfg: dict) -> np.ndarray:
    """(M, 2) int32 camera px (x, y) where the projected outline is visible.

    Channel-max difference lit - dark (`calibration._diff_gray`), the adaptive threshold of the marker
    detector (`calibration._marker_binary`, peak = max of the blurred difference as for a single marker, so a
    thin outline covering < 0.1 % of the frame still sets the contrast), thinned to 1 px; components of the
    thinned image smaller than `min_blob_px` are dropped. cfg needs the `calibration:` threshold keys
    (`diff_threshold`; optional `diff_threshold_min`, `adaptive_frac`, `blur_ksize`) plus optional
    `min_blob_px`: the script passes {**config["calibration"], **config["accuracy"]}."""
    if frame_lit.shape != frame_dark.shape:
        raise ValueError(f"lit {frame_lit.shape} and dark {frame_dark.shape} frames differ in shape")
    diff = _diff_gray(frame_lit, frame_dark)
    k = int(cfg.get("blur_ksize", 5)) | 1
    peak = float(cv2.GaussianBlur(diff, (k, k), 0).max())
    binary = _marker_binary(diff, cfg, peak=peak)
    skel = thin(binary)
    min_px = int(cfg.get("min_blob_px", DEFAULTS["min_blob_px"]))
    if min_px > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_px
        keep[0] = False
        skel = np.where(keep[labels], 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(skel)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def boundary_errors_mm(lit_pts: np.ndarray, truth_pts: np.ndarray, depth_cam: np.ndarray, K_cam: np.ndarray,
                       max_px: float | None = None) -> np.ndarray:
    """Per lit pixel, the distance (mm) to the nearest truth boundary pixel.

    lit_pts / truth_pts: (N, 2) / (M, 2) camera px (x, y); depth_cam (H, W) metres at camera resolution
    (0 / NaN = invalid); K_cam (3, 3) camera-space intrinsics. Pixel distance = exact Euclidean distance
    transform of the inverted truth-boundary image (DIST_MASK_PRECISE). Metric scale = depth at the NEAREST
    TRUTH pixel (always on the body; a lit pixel that missed the body may sit on the wall or have no depth)
    * 1000 / mean(fx, fy); when that depth is invalid the lit pixel's own depth is used. Returns (N,) float64;
    NaN for lit pixels outside the image, without any valid depth, or farther than `max_px` from the truth
    (when given: stray light / motion edges, counted as rejected by `summarize`). No truth -> all NaN."""
    lit = np.asarray(lit_pts, np.int64).reshape(-1, 2)
    tru = np.asarray(truth_pts, np.int64).reshape(-1, 2)
    depth = np.asarray(depth_cam, np.float64)
    h, w = depth.shape[:2]
    out = np.full(len(lit), np.nan)
    if not len(lit):
        return out
    tin = (tru[:, 0] >= 0) & (tru[:, 0] < w) & (tru[:, 1] >= 0) & (tru[:, 1] < h)
    tru = tru[tin]
    if not len(tru):
        return out
    img = np.full((h, w), 255, np.uint8)
    img[tru[:, 1], tru[:, 0]] = 0
    dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # nearest truth pixel (approximate 5x5 transform; used only to pick which depth to read)
    _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(img == 0)
    lab_xy = np.zeros((int(labels.max()) + 1, 2), np.int64)
    lab_xy[labels[zy, zx]] = np.stack([zx, zy], axis=1)

    inside = (lit[:, 0] >= 0) & (lit[:, 0] < w) & (lit[:, 1] >= 0) & (lit[:, 1] < h)
    lx, ly = lit[inside, 0], lit[inside, 1]
    d_px = dist[ly, lx].astype(np.float64)
    near = lab_xy[labels[ly, lx]]
    z = depth[near[:, 1], near[:, 0]]
    z_own = depth[ly, lx]
    bad = ~(np.isfinite(z) & (z > 0))
    z = np.where(bad, z_own, z)
    K = np.asarray(K_cam, np.float64)
    f = 0.5 * (K[0, 0] + K[1, 1])
    with np.errstate(invalid="ignore"):
        mm = d_px * z * 1000.0 / f
        mm[~(np.isfinite(z) & (z > 0))] = np.nan
        if max_px is not None:
            mm[d_px > float(max_px)] = np.nan
    out[inside] = mm
    return out


def summarize(errors_mm: np.ndarray) -> dict:
    """{'mean', 'median', 'p95', 'max', 'n', 'n_rejected'} over the finite errors (mm); NaN entries are
    counted in n_rejected. Statistics are None when no finite error exists."""
    e = np.asarray(errors_mm, np.float64).ravel()
    ok = e[np.isfinite(e)]
    res = {"n": int(ok.size), "n_rejected": int(e.size - ok.size)}
    if ok.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, **res}
    return {"mean": float(ok.mean()), "median": float(np.median(ok)), "p95": float(np.percentile(ok, 95)),
            "max": float(ok.max()), **res}


def annotate(frame: np.ndarray, lit_pts: np.ndarray, truth_pts: np.ndarray, text: str = "") -> np.ndarray:
    """Copy of `frame` (BGR) with the truth boundary in red, the detected lit outline in green, `text` top-left."""
    img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    for pts, color in ((truth_pts, (0, 0, 255)), (lit_pts, (0, 255, 0))):
        p = np.asarray(pts, np.int64).reshape(-1, 2)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)]
        img[p[:, 1], p[:, 0]] = color
    if text:
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img
