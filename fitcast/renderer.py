"""Warps garment textures onto body parts in camera space. Output: RGBA (H,W,4) uint8."""
from __future__ import annotations

import numpy as np
import cv2

from . import geometry as geo
from .types import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, R_ANKLE, R_ELBOW, R_HIP, R_KNEE,
                    R_SHOULDER, R_WRIST, Outfit, TrackResult)


def _src_quad(img: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """Normalised (x0,y0,x1,y1) region of `img` -> pixel corners [TL, TR, BR, BL]."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    return np.array([[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]], np.float32)


class Renderer:
    """Per-part perspective warps: bottoms, torso, sleeves (draw order), feathered and alpha-composited."""

    def __init__(self, cfg: dict, size: tuple[int, int], min_visibility: float = 0.5) -> None:
        self.cfg = cfg
        self.W, self.H = size
        self.min_vis = float(min_visibility)
        self.feather = int(cfg["feather_px"])
        self.canvas = np.zeros((self.H, self.W, 4), np.uint8)
        self._warp = np.zeros((self.H, self.W, 4), np.uint8)
        self.parts_drawn = 0

    # -- public ------------------------------------------------------------------------------
    def draw(self, track: TrackResult, outfit: Outfit) -> np.ndarray:
        """Render `outfit` onto the body described by `track`. Returns the reusable RGBA canvas."""
        self.canvas.fill(0)
        self.parts_drawn = 0
        if not track.ok:
            return self.canvas
        j, v = track.joints, track.visibility
        sw = max(track.shoulder_width, 1.0)
        ref = max(track.ref_shoulder_width, sw)
        back = track.facing == "back"
        self._draw_bottoms(j, v, sw, outfit, back)
        self._draw_torso(j, v, sw, ref, outfit, track.facing)
        self._draw_sleeves(j, v, sw, outfit)
        return self.canvas

    # -- parts -------------------------------------------------------------------------------
    def _vis(self, v: np.ndarray, *idx: int) -> bool:
        return all(v[i] >= self.min_vis for i in idx)

    def _draw_torso(self, j: np.ndarray, v: np.ndarray, sw: float, ref: float, outfit: Outfit, facing: str) -> None:
        if not self._vis(v, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            return
        tex = outfit.top_back if (facing == "back" and outfit.top_back is not None) else outfit.top_front
        if tex is None:
            return
        c = self.cfg
        min_w = c["side_min_width_frac"] * ref if facing == "side" else 0.0
        dst = geo.torso_quad(j, sw, c["torso_margin_frac"], c["neck_raise_frac"], c["hem_drop_frac"], min_w)
        # dst is [image-left shoulder, image-right shoulder, image-right hem, image-left hem] which matches
        # the texture's [TL, TR, BR, BL] for BOTH a front image seen from the front and a back image seen
        # from behind (viewer-left is always image-left), so no winding swap is needed.
        self._warp_part(tex, _src_quad(tex, outfit.top_body_region), dst)

    def _draw_sleeves(self, j: np.ndarray, v: np.ndarray, sw: float, outfit: Outfit) -> None:
        if outfit.sleeve == "none" or outfit.sleeve_tex is None:
            return
        c = self.cfg
        tex = outfit.sleeve_tex
        src = _src_quad(tex, (0.0, 0.0, 1.0, 1.0))
        wu, wf = 0.5 * c["upper_arm_width_frac"] * sw, 0.5 * c["forearm_width_frac"] * sw
        overlap = c["sleeve_overlap_frac"] * sw
        for sh, el, wr in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
            if self._vis(v, sh, el):
                self._warp_part(tex, src, geo.limb_quad(j[sh, :2], j[el, :2], wu, wu * 0.85, extend0=overlap))
            if outfit.sleeve == "long" and self._vis(v, el, wr):
                self._warp_part(tex, src, geo.limb_quad(j[el, :2], j[wr, :2], wf, wf * 0.8, extend0=overlap * 0.5))

    def _draw_bottoms(self, j: np.ndarray, v: np.ndarray, sw: float, outfit: Outfit, back: bool) -> None:
        tex = outfit.bottom_back if (back and outfit.bottom_back is not None) else outfit.bottom_front
        if tex is None or not self._vis(v, L_HIP, R_HIP):
            return
        c = self.cfg
        band_h = c["waist_band_height_frac"]
        knee_frac = c.get("bottom_knee_frac", 0.55)
        lh, rh = j[L_HIP, :2], j[R_HIP, :2]
        hc = 0.5 * (lh + rh)
        axis = geo.unit(lh - rh)
        if axis[0] < 0:
            axis = -axis                       # points image-right
        down = np.array([-axis[1], axis[0]], np.float32)
        if down[1] < 0:
            down = -down
        hip_dist = float(np.hypot(*(lh - rh)))
        half = max(0.5 * hip_dist + 0.5 * c["leg_width_frac"] * sw, 0.45 * sw)
        band = c["waist_band_height_frac"] * sw
        # waist band quad: full width between the (expanded) hips
        top = hc - down * band * 0.6
        bot = hc + down * band * 0.6
        dst = np.array([top - axis * half, top + axis * half, bot + axis * half, bot - axis * half], np.float32)
        self._warp_part(tex, _src_quad(tex, (0.0, 0.0, 1.0, band_h)), dst)
        # legs: image-left half of the texture goes on the image-left leg
        legs = sorted(((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)), key=lambda t: j[t[0], 0])
        wl, ws = 0.5 * c["leg_width_frac"] * sw, 0.5 * c["shin_width_frac"] * sw
        for side, (hip, knee, ankle) in enumerate(legs):
            x0, x1 = (0.0, 0.5) if side == 0 else (0.5, 1.0)
            if self._vis(v, hip, knee):
                self._warp_part(tex, _src_quad(tex, (x0, band_h * 0.8, x1, knee_frac)),
                                geo.limb_quad(j[hip, :2], j[knee, :2], wl, wl * 0.85, extend0=band * 0.3))
            if self._vis(v, knee, ankle):
                self._warp_part(tex, _src_quad(tex, (x0, knee_frac, x1, 1.0)),
                                geo.limb_quad(j[knee, :2], j[ankle, :2], ws, ws * 0.9, extend0=wl * 0.3))

    # -- core warp ---------------------------------------------------------------------------
    def _warp_part(self, tex: np.ndarray, src: np.ndarray, dst: np.ndarray) -> None:
        """Perspective-warp `tex[src quad]` onto `dst` quad (ROI only), feather the alpha, alpha-over onto canvas."""
        f = self.feather
        x0 = int(np.floor(dst[:, 0].min())) - f
        y0 = int(np.floor(dst[:, 1].min())) - f
        x1 = int(np.ceil(dst[:, 0].max())) + f
        y1 = int(np.ceil(dst[:, 1].max())) + f
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.W), min(y1, self.H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        if not np.isfinite(dst).all():
            return
        H = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], np.float32))
        roi = self._warp[y0:y1, x0:x1]
        cv2.warpPerspective(tex, H, (x1 - x0, y1 - y0), dst=roi, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = roi[:, :, 3]
        if f > 0:
            k = 2 * f + 1
            alpha = cv2.blur(cv2.erode(alpha, np.ones((f, f), np.uint8)), (k, k))
        a = alpha.astype(np.float32) * (1.0 / 255.0)
        a3 = a[:, :, None]
        out = self.canvas[y0:y1, x0:x1]
        rgb = out[:, :, :3].astype(np.float32)
        rgb *= (1.0 - a3)
        rgb += roi[:, :, :3].astype(np.float32) * a3
        oa = out[:, :, 3].astype(np.float32) * (1.0 / 255.0)
        oa = a + oa * (1.0 - a)
        out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[:, :, 3] = (oa * 255.0).astype(np.uint8)
        self.parts_drawn += 1
