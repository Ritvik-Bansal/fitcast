"""Camera -> projector homography: pattern generation, solving, load/save.

The camera sits against the projector lens, so at the standing plane the map
from camera pixels to projector pixels is close to a single homography. It is
solved from a projected checkerboard (or a dot grid fallback) seen on a white
board held at the taped spot.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    H: np.ndarray
    camera_size: tuple[int, int]
    projector_size: tuple[int, int]
    error_px: float = 0.0
    method: str = "checkerboard"
    created: str = ""
    window: dict = field(default_factory=dict)

    def cam_to_proj(self, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def proj_to_cam(self, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)


def load_calibration(path: str | Path) -> Calibration | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
        H = np.array(d["H"], np.float64).reshape(3, 3)
        if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("homography is singular or not finite")
        return Calibration(
            H=H,
            camera_size=tuple(d.get("camera_size", (1280, 720))),
            projector_size=tuple(d.get("projector_size", (1920, 1080))),
            error_px=float(d.get("error_px", 0.0)),
            method=d.get("method", "checkerboard"),
            created=d.get("created", ""),
            window=d.get("window", {}),
        )
    except (KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as e:
        print(f"[calib] {path} is unreadable ({e}); ignoring it")
        return None


def save_calibration(path: str | Path, calib: Calibration) -> None:
    d = {
        "H": calib.H.tolist(),
        "camera_size": list(calib.camera_size),
        "projector_size": list(calib.projector_size),
        "error_px": round(float(calib.error_px), 3),
        "method": calib.method,
        "created": calib.created or time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": calib.window,
    }
    Path(path).write_text(json.dumps(d, indent=2))


def checkerboard(proj_size: tuple[int, int], pattern: tuple[int, int], center: tuple[float, float],
                 square_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Projector image with a checkerboard of `pattern` inner corners (cols, rows).

    Returns the BGR image and the inner-corner positions in projector pixels, in
    the same row-major order OpenCV reports them. A white quiet zone one square
    wide surrounds the board so detection works.
    """
    pw, ph = proj_size
    cols, rows = pattern
    img = np.zeros((ph, pw, 3), np.uint8)
    sq = float(square_px)
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    x0, y0 = center[0] - bw / 2, center[1] - bh / 2
    pad = sq
    cv2.rectangle(img, (int(x0 - pad), int(y0 - pad)), (int(x0 + bw + pad), int(y0 + bh + pad)), (255, 255, 255), -1)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                p0 = (int(round(x0 + c * sq)), int(round(y0 + r * sq)))
                p1 = (int(round(x0 + (c + 1) * sq)) - 1, int(round(y0 + (r + 1) * sq)) - 1)
                cv2.rectangle(img, p0, p1, (0, 0, 0), -1)
    # squares end at pixel round(edge) - 1, so the true corner sits half a pixel before round(edge)
    corners = np.array([[round(x0 + (c + 1) * sq) - 0.5, round(y0 + (r + 1) * sq) - 0.5]
                        for r in range(rows) for c in range(cols)], np.float64)
    return img, corners


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = False) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        if fast:
            return None
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        return corners.reshape(-1, 2).astype(np.float64)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    return corners.reshape(-1, 2).astype(np.float64)


def order_like(cam_pts: np.ndarray, proj_pts: np.ndarray) -> np.ndarray:
    """Fix OpenCV's 180-degree corner-order ambiguity.

    Camera and projector point the same way, so the first->last corner vector
    must point roughly the same direction in both images.
    """
    v_cam = cam_pts[-1] - cam_pts[0]
    v_proj = proj_pts[-1] - proj_pts[0]
    if float(np.dot(v_cam, v_proj)) < 0:
        return cam_pts[::-1].copy()
    return cam_pts


def solve_homography(cam_pts: np.ndarray, proj_pts: np.ndarray) -> tuple[np.ndarray | None, float]:
    if len(cam_pts) < 4:
        return None, float("inf")
    H, inliers = cv2.findHomography(cam_pts, proj_pts, cv2.RANSAC, 4.0)
    if H is None:
        return None, float("inf")
    mapped = cv2.perspectiveTransform(cam_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(np.mean(np.sum((mapped - proj_pts) ** 2, axis=1))))
    return H, err


def dot_grid(proj_size: tuple[int, int], center: tuple[float, float], span: tuple[float, float],
             grid: tuple[int, int] = (5, 4)) -> np.ndarray:
    gx, gy = grid
    xs = np.linspace(center[0] - span[0] / 2, center[0] + span[0] / 2, gx)
    ys = np.linspace(center[1] - span[1] / 2, center[1] + span[1] / 2, gy)
    pw, ph = proj_size
    return np.array([[min(max(x, 20), pw - 20), min(max(y, 20), ph - 20)] for y in ys for x in xs], np.float64)


def dot_centroid(dark: np.ndarray, lit: np.ndarray, min_area: int = 12) -> np.ndarray | None:
    """Centroid of the brightest blob that appeared between a dark and a lit frame."""
    diff = cv2.subtract(cv2.cvtColor(lit, cv2.COLOR_BGR2GRAY), cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    peak = int(diff.max())
    if peak < 25:
        return None
    _, bw = cv2.threshold(diff, max(20, int(peak * 0.5)), 255, cv2.THRESH_BINARY)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return None
    return cents[best].astype(np.float64)
