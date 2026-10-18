"""Depth-based whole-person mask for the 3D sensor path (iPhone LiDAR).

Membership is decided by depth alone, so projected garment light cannot change who is "person":

1. Background model: per-pixel median of the empty scene (`learn_background`). Pixels never seen with
   enough confidence become +inf (unknown background = far).
2. Body (depth resolution): valid depth (conf >= `min_confidence`) that is at least
   `closer_than_background_m` in front of the background AND whose 3D point (via `K_depth`, OpenCV camera
   coords) lies inside `box_m`; opened, closed, and components smaller than `min_component_frac` dropped.
3. Edge-aware upsampling to camera resolution: bilinear resize, then an RGB-guided filter, threshold 0.5.
4. Optional fusion: ARKit person segmentation and the skeleton mask may extend the depth mask only within
   `fuse_reach_px` of it (edge refinement); they never create regions on their own.

Assumption: the depth map and `rgb_cam` cover the same field of view (ARKit sceneDepth is aligned to the
captured image), so depth -> camera is an independent x/y scale even when the aspect ratios differ.
The output is a PERSON mask; the caller intersects it with clothes/skin logic and applies the head blackout.
"""
from __future__ import annotations

import logging
import math
import time
import warnings

import cv2
import numpy as np

log = logging.getLogger("fitcast.depthmask")

_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


def _ellipse(radius: int) -> np.ndarray | None:
    """Elliptical structuring element of diameter 2*radius+1 (None when radius <= 0)."""
    r = int(radius)
    return None if r <= 0 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


class DepthBodyMask:
    """Person mask from LiDAR depth vs a learned empty-scene background, upsampled to camera resolution.

    Config (`config["depthmask"]`): `closer_than_background_m`, `box_m` [x0,x1,y0,y1,z0,z1] metres,
    `min_confidence` (ARKit 0..2), `open_px` / `close_px` (morphology RADII in depth pixels, kernel 2r+1),
    `guided_radius_px` (camera px), `guided_eps` (normalised-intensity units, guide in [0,1]),
    `fuse_person_seg`, `fuse_skeleton`; optional `min_component_frac` (0.002), `fuse_reach_px` (12, camera px),
    `guided_scale` (1.0; < 1 uses OpenCV's fast guided filter). `background_frames` is used by the caller.
    """

    def __init__(self, cfg: dict, depth_size: tuple[int, int], cam_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.dw, self.dh = int(depth_size[0]), int(depth_size[1])
        self.cw, self.ch = int(cam_size[0]), int(cam_size[1])
        self.thr = float(cfg["closer_than_background_m"])
        box = [float(v) for v in cfg["box_m"]]
        if len(box) != 6:
            raise ValueError("depthmask.box_m must be [x0, x1, y0, y1, z0, z1]")
        self.box = box
        self.min_conf = int(cfg["min_confidence"])
        self.k_open = _ellipse(cfg["open_px"])
        self.k_close = _ellipse(cfg["close_px"])
        self.radius = int(cfg["guided_radius_px"])
        self.eps = float(cfg["guided_eps"])
        self.fuse_seg = bool(cfg["fuse_person_seg"])
        self.fuse_skel = bool(cfg["fuse_skeleton"])
        self.min_area = max(1, int(math.ceil(float(cfg.get("min_component_frac", 0.002)) * self.dw * self.dh)))
        self.reach = int(cfg.get("fuse_reach_px", 12))
        self.guided_scale = float(cfg.get("guided_scale", 1.0))
        r = max(self.reach, 0)
        self.k_reach = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)) if r > 0 else None
        self.sx, self.sy = self.cw / self.dw, self.ch / self.dh
        # ROI padding (camera px) that makes the ROI-only guided filter + fusion exact: bilinear support (1 cell)
        # + guided support (2r) + fusion reach.
        g = 1
        if _HAS_XIMGPROC:
            g = 2 * self.radius + 2 + (int(math.ceil(1.0 / self.guided_scale)) if self.guided_scale < 1.0 else 0)
        self.pad_x = int(math.ceil(self.sx)) + g + r
        self.pad_y = int(math.ceil(self.sy)) + g + r
        if not _HAS_XIMGPROC:
            log.warning("cv2.ximgproc missing (install opencv-contrib-python): depth mask upsampling falls back to "
                        "bilinear + threshold (no RGB edge snapping)")

        self.background: np.ndarray | None = None          # (dh, dw) float32 metres, +inf = unknown/far
        self._K: np.ndarray | None = None
        self._xn: np.ndarray | None = None                  # (dh, dw) (u - cx) / fx
        self._yn: np.ndarray | None = None
        self._zlo = np.full((self.dh, self.dw), max(box[4], 1e-6), np.float32)   # per-pixel depth bounds
        self._zhi = np.full((self.dh, self.dw), box[5], np.float32)
        self._warned: set[str] = set()
        # reusable buffers
        dshape, cshape = (self.dh, self.dw), (self.ch, self.cw)
        self._d = np.zeros(dshape, np.float32)
        self._b = np.zeros(dshape, bool)
        self._b2 = np.zeros(dshape, bool)
        self._low = np.zeros(dshape, np.uint8)
        self._low2 = np.zeros(dshape, np.uint8)
        self._labels = np.zeros(dshape, np.int32)
        self._up = np.zeros(cshape, np.uint8)               # bilinear upsampled low-res mask (0..255)
        self._gf = np.zeros(cshape, np.float32)             # guided filter output
        self._seg = np.zeros(cshape, np.uint8)
        self._extra = np.zeros(cshape, np.uint8)
        self._dil = np.zeros(cshape, np.uint8)
        self._out = [np.zeros(cshape, np.uint8), np.zeros(cshape, np.uint8)]
        self._n = 0
        self.last_stats: dict = {"valid_fraction": 0.0, "body_pixels": 0, "n_components": 0,
                                 "roi": (0, 0, 0, 0), "ms": 0.0, "guided": _HAS_XIMGPROC}

    # -- setup -----------------------------------------------------------------------------------
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)

    def set_intrinsics(self, K_depth: np.ndarray) -> None:
        """Set the (3,3) depth-image intrinsics (image K scaled by depth_w/img_w, depth_h/img_h).
        Enables the x/y part of the `box_m` test; without it only the z range is applied."""
        K = np.asarray(K_depth, np.float64)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"K_depth must be a (3,3) intrinsics matrix with fx, fy > 0, got {K!r}")
        if self._K is not None and np.array_equal(K, self._K):
            return                                          # cheap when called every frame with the same K
        self._K = K.copy()
        u = (np.arange(self.dw, dtype=np.float64) - K[0, 2]) / K[0, 0]
        v = (np.arange(self.dh, dtype=np.float64) - K[1, 2]) / K[1, 1]
        self._xn = np.broadcast_to(u[None, :], (self.dh, self.dw)).astype(np.float32)
        self._yn = np.broadcast_to(v[:, None], (self.dh, self.dw)).astype(np.float32)
        self._update_bounds()

    def learn_background(self, depth_frames: list[np.ndarray], conf_frames: list[np.ndarray]) -> None:
        """Per-pixel median of the empty scene over pixels with conf >= `min_confidence` and valid depth.
        Pixels never valid get +inf (unknown background is treated as far)."""
        if len(depth_frames) == 0 or len(depth_frames) != len(conf_frames):
            raise ValueError("learn_background needs >= 1 depth frame and one confidence frame per depth frame")
        d = np.stack([np.asarray(f, np.float32) for f in depth_frames])
        c = np.stack([np.asarray(f) for f in conf_frames])
        if d.shape[1:] != (self.dh, self.dw) or c.shape != d.shape:
            raise ValueError(f"background frames must be (h, w) = {(self.dh, self.dw)}, got {d.shape[1:]} / {c.shape[1:]}")
        bad = ~np.isfinite(d) | (d <= 0) | (c < self.min_conf)
        d[bad] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN pixels -> NaN, handled below
            med = np.nanmedian(d, axis=0)
        self.background = np.where(np.isnan(med), np.inf, med).astype(np.float32)
        self._update_bounds()
        log.info("depth background learned from %d frames (%.0f%% of pixels known)", len(depth_frames),
                 100.0 * float(np.isfinite(self.background).mean()))

    def _update_bounds(self) -> None:
        """Fold background, box and intrinsics into per-pixel [zlo, zhi] so compute() is two comparisons."""
        x0, x1, y0, y1, z0, z1 = self.box
        zlo = np.full((self.dh, self.dw), max(z0, 1e-6), np.float64)
        zhi = np.full((self.dh, self.dw), z1, np.float64)
        if self.background is not None:
            # strictly closer than background - thr (+inf background stays +inf)
            np.minimum(zhi, np.nextafter(self.background.astype(np.float64) - self.thr, -np.inf), out=zhi)
        if self._xn is not None:
            for n, a, b in ((self._xn, x0, x1), (self._yn, y0, y1)):
                # coordinate = n * Z in [a, b]  <=>  Z in [min(a/n, b/n), max(a/n, b/n)] for n != 0
                n = n.astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pa, pb = a / n, b / n
                lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
                zero = n == 0
                inside = a <= 0.0 <= b
                lo[zero] = -np.inf if inside else np.inf
                hi[zero] = np.inf if inside else -np.inf
                np.maximum(zlo, lo, out=zlo)
                np.minimum(zhi, hi, out=zhi)
        self._zlo[:] = zlo
        self._zhi[:] = zhi

    @property
    def ready(self) -> bool:
        """True once a background has been learned."""
        return self.background is not None

    # -- per frame -------------------------------------------------------------------------------
    def compute(self, depth: np.ndarray, conf: np.ndarray, rgb_cam: np.ndarray,
                person_seg: np.ndarray | None = None, skeleton_mask: np.ndarray | None = None) -> np.ndarray:
        """Whole-person mask (cam_h, cam_w) uint8 0/255 from depth (metres, 0/NaN invalid) + conf (0..2).

        `rgb_cam` (cam_h, cam_w, 3) BGR guides the edge-aware upsampling; `person_seg` (any size, 0/255) and
        `skeleton_mask` (cam size, 0/255) may only extend the depth mask within `fuse_reach_px` of it.
        Returns one of two alternating internal buffers (valid until the call after next).
        Before `learn_background` it returns an all-zero mask (warned once)."""
        t0 = time.perf_counter()
        self._n += 1
        out = self._out[self._n & 1]
        out.fill(0)
        st = self.last_stats
        if not self.ready:
            self._warn_once("ready", "DepthBodyMask.compute() called before learn_background(): empty mask "
                                     "(capture the empty scene first)")
            st.update(valid_fraction=0.0, body_pixels=0, n_components=0, roi=(0, 0, 0, 0),
                      ms=(time.perf_counter() - t0) * 1e3)
            return out
        if self._xn is None:
            self._warn_once("K", "DepthBodyMask: set_intrinsics() not called; box_m applies to z only")
        if depth.shape != (self.dh, self.dw) or conf.shape != (self.dh, self.dw):
            raise ValueError(f"depth/conf must be {(self.dh, self.dw)}, got {depth.shape} / {conf.shape}")
        if rgb_cam.shape[:2] != (self.ch, self.cw) or rgb_cam.dtype != np.uint8:
            raise ValueError(f"rgb_cam must be uint8 BGR {(self.ch, self.cw)}, got {rgb_cam.dtype} {rgb_cam.shape[:2]}")

        # ---- low-res body: valid & conf & zlo <= d <= zhi (box + background folded into the bounds)
        d, b, b2 = self._d, self._b, self._b2
        np.copyto(d, depth, casting="unsafe")              # float16/float32 -> float32, no allocation
        np.greater(d, 0.0, out=b)                          # NaN -> False
        np.greater_equal(conf, self.min_conf, out=b2)
        np.logical_and(b, b2, out=b)
        valid_n = int(np.count_nonzero(b))
        np.greater_equal(d, self._zlo, out=b2)
        np.logical_and(b, b2, out=b)
        np.less_equal(d, self._zhi, out=b2)
        np.logical_and(b, b2, out=b)
        low = self._low
        np.multiply(b, np.uint8(255), out=low)
        if self.k_open is not None:
            cv2.morphologyEx(low, cv2.MORPH_OPEN, self.k_open, dst=self._low2)
            low, spare = self._low2, self._low
        else:
            spare = self._low2
        if self.k_close is not None:
            cv2.morphologyEx(low, cv2.MORPH_CLOSE, self.k_close, dst=spare)
            low, spare = spare, low
        n, _, stats, _ = cv2.connectedComponentsWithStats(low, labels=self._labels, connectivity=8,
                                                          ltype=cv2.CV_32S)
        keep = stats[:, cv2.CC_STAT_AREA] >= self.min_area
        keep[0] = False
        st["valid_fraction"] = valid_n / float(self.dw * self.dh)
        st["n_components"] = int(keep.sum())
        if not keep.any():
            st.update(body_pixels=0, roi=(0, 0, 0, 0), ms=(time.perf_counter() - t0) * 1e3)
            return out
        lut = np.where(keep, 255, 0).astype(np.uint8)
        np.take(lut, self._labels, out=low)
        st["body_pixels"] = int(stats[keep, cv2.CC_STAT_AREA].sum())

        # ---- camera-res ROI around the kept components (padding makes ROI-only processing exact)
        ks = stats[keep]
        bx0, by0 = int(ks[:, 0].min()), int(ks[:, 1].min())
        bx1, by1 = int((ks[:, 0] + ks[:, 2]).max()), int((ks[:, 1] + ks[:, 3]).max())
        x0 = max(0, int(math.floor(bx0 * self.sx)) - self.pad_x)
        y0 = max(0, int(math.floor(by0 * self.sy)) - self.pad_y)
        x1 = min(self.cw, int(math.ceil(bx1 * self.sx)) + self.pad_x)
        y1 = min(self.ch, int(math.ceil(by1 * self.sy)) + self.pad_y)
        st["roi"] = (x0, y0, x1, y1)

        cv2.resize(low, (self.cw, self.ch), dst=self._up, interpolation=cv2.INTER_LINEAR)
        up = self._up[y0:y1, x0:x1]
        o = out[y0:y1, x0:x1]
        if _HAS_XIMGPROC:
            gf = self._gf[y0:y1, x0:x1]
            # uint8 guide is not normalised by OpenCV: eps in [0,1] units -> * 255^2
            cv2.ximgproc.guidedFilter(rgb_cam[y0:y1, x0:x1], up, self.radius, self.eps * 65025.0, dst=gf,
                                      dDepth=cv2.CV_32F, scale=self.guided_scale)
            cv2.compare(gf, 127.5, cv2.CMP_GT, dst=o)
        else:
            cv2.compare(up, 127, cv2.CMP_GT, dst=o)

        # ---- fusion: RGB segmentation / skeleton may extend the depth mask only within fuse_reach_px of it
        add = None
        if self.fuse_seg and person_seg is not None:
            if person_seg.shape[:2] == (self.ch, self.cw):
                seg = person_seg
            else:
                cv2.resize(person_seg, (self.cw, self.ch), dst=self._seg, interpolation=cv2.INTER_LINEAR)
                seg = self._seg
            add = self._extra[y0:y1, x0:x1]
            cv2.compare(seg[y0:y1, x0:x1], 127, cv2.CMP_GT, dst=add)
        if self.fuse_skel and skeleton_mask is not None:
            if skeleton_mask.shape[:2] != (self.ch, self.cw):
                raise ValueError(f"skeleton_mask must be {(self.ch, self.cw)}, got {skeleton_mask.shape[:2]}")
            sk = skeleton_mask[y0:y1, x0:x1]
            if add is None:
                add = self._extra[y0:y1, x0:x1]
                cv2.compare(sk, 127, cv2.CMP_GT, dst=add)
            else:
                cv2.bitwise_or(add, cv2.compare(sk, 127, cv2.CMP_GT), dst=add)
        if add is not None and self.k_reach is not None:
            dil = self._dil[y0:y1, x0:x1]
            cv2.dilate(o, self.k_reach, dst=dil)
            cv2.bitwise_and(add, dil, dst=add)
            cv2.bitwise_or(o, add, dst=o)
        st["ms"] = (time.perf_counter() - t0) * 1e3
        return out
