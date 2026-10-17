"""Pure 2D helpers shared by the tracker (skeleton mask) and the renderer (garment quads)."""
from __future__ import annotations

import numpy as np

from .types import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-6 else np.array([1.0, 0.0], np.float32)


def torso_quad(joints: np.ndarray, sw: float, margin_frac: float, neck_raise_frac: float,
               hem_drop_frac: float, min_width: float = 0.0) -> np.ndarray:
    """Torso quad [image-left shoulder, image-right shoulder, image-right hem, image-left hem] (4,2) float32.

    Built from shoulders and hips, expanded outward by `margin_frac * sw`, raised at the neckline by
    `neck_raise_frac * sw`, extended below the hips by `hem_drop_frac * torso_len`. The hem is at least
    shoulder width wide (shirts hang straight). `min_width` enforces a minimum width (side view).
    """
    ls, rs = joints[L_SHOULDER, :2], joints[R_SHOULDER, :2]
    lh, rh = joints[L_HIP, :2], joints[R_HIP, :2]
    sc, hc = (ls + rs) * 0.5, (lh + rh) * 0.5
    down = hc - sc
    torso_len = float(np.hypot(*down)) or sw * 1.3
    down_u = unit(down)
    # shoulder axis pointing to image-right, orthogonal to the torso axis
    axis = np.array([-down_u[1], down_u[0]], np.float32)
    if axis[0] < 0:
        axis = -axis
    half = max(abs(float(np.dot(ls - sc, axis))), abs(float(np.dot(rs - sc, axis))))
    half = max(half + margin_frac * sw, min_width * 0.5)
    top = sc - down_u * (neck_raise_frac * sw)
    bot = hc + down_u * (hem_drop_frac * torso_len)
    return np.array([top - axis * half, top + axis * half, bot + axis * half, bot - axis * half], np.float32)


def limb_quad(p0: np.ndarray, p1: np.ndarray, w0: float, w1: float, extend0: float = 0.0,
              extend1: float = 0.0) -> np.ndarray:
    """Quad around segment p0->p1 with half-widths w0 at p0 and w1 at p1, optionally extended along the axis.

    Corner order: [p0-left, p0-right, p1-right, p1-left] where left/right are relative to the direction p0->p1.
    """
    d = unit(np.asarray(p1, np.float32) - np.asarray(p0, np.float32))
    n = np.array([-d[1], d[0]], np.float32)
    a = np.asarray(p0, np.float32) - d * extend0
    b = np.asarray(p1, np.float32) + d * extend1
    return np.array([a - n * w0, a + n * w0, b + n * w1, b - n * w1], np.float32)


def shoulder_width(joints: np.ndarray) -> float:
    return float(np.hypot(*(joints[L_SHOULDER, :2] - joints[R_SHOULDER, :2])))


def torso_length(joints: np.ndarray) -> float:
    sc = (joints[L_SHOULDER, :2] + joints[R_SHOULDER, :2]) * 0.5
    hc = (joints[L_HIP, :2] + joints[R_HIP, :2]) * 0.5
    return float(np.hypot(*(hc - sc)))


def up_direction(joints: np.ndarray) -> np.ndarray:
    """Unit vector from shoulder centre toward the nose (image coordinates)."""
    sc = (joints[L_SHOULDER, :2] + joints[R_SHOULDER, :2]) * 0.5
    return unit(joints[NOSE, :2] - sc)
