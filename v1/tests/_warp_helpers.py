"""Shared builders for the warp/render tests: synthetic Body objects and a cached catalog."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tryon.config import load_config  # noqa: E402
from tryon.garments import load_catalog  # noqa: E402
from tryon.smoothing import Body  # noqa: E402
from tryon.warp import Layer, SideAssigner, ViewWarper  # noqa: E402

SIZE = (1280, 720)
VIEWS = ("front", "left", "back", "right")
JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]

_cache = {}


def cfg_and_catalog():
    if "c" not in _cache:
        import contextlib
        import io
        cfg = load_config()
        with contextlib.redirect_stdout(io.StringIO()):
            cat = load_catalog(cfg)
        _cache["c"] = (cfg, cat)
    return _cache["c"]


def warpers(cfg=None):
    """{(garment_id, view): (garment, ViewWarper)} for the real placeholder catalog."""
    if "w" not in _cache:
        import contextlib
        import io
        cfg, cat = cfg_and_catalog()
        out = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for g in cat.garments:
                for vn, v in g.views.items():
                    out[(g.id, vn)] = (g, ViewWarper(v, g.category, g.sleeve_length, cfg))
        _cache["w"] = out
    return _cache["w"]


def swap_labels(kp: dict, vis: dict | None = None):
    def sw(k):
        if k.startswith("left_"):
            return "right_" + k[5:]
        if k.startswith("right_"):
            return "left_" + k[6:]
        return k
    kp2 = {sw(k): v.copy() for k, v in kp.items()}
    if vis is None:
        return kp2
    return kp2, {sw(k): v for k, v in vis.items()}


def make_body(pose="neutral", cx=640.0, top=40.0, s=1.0, facing="front", swapped=False,
              size=SIZE, vis_lower=1.0):
    """Synthetic standing person. `facing='front'`: MediaPipe 'left' joints at image-right.
    facing='back': 'left' joints at image-left. `swapped` exchanges every left/right key afterwards."""
    sh_y = top + 80 * s
    hip_y = sh_y + 170 * s
    knee_y = hip_y + 200 * s
    ank_y = knee_y + 200 * s
    sh, hw = 55 * s, 40 * s
    if pose == "sideon":
        sh, hw = 6 * s, 5 * s
    if pose == "degenerate":
        sh, hw, hip_y = 0.5 * s, 0.5 * s, sh_y + 3 * s
        knee_y, ank_y = hip_y + 200 * s, hip_y + 400 * s
    # unit vector for "mp left" side in the image
    lsign = 1.0 if facing == "front" else -1.0
    kp = {"nose": (cx, top), "left_eye": (cx + lsign * 8 * s, top - 6 * s), "right_eye": (cx - lsign * 8 * s, top - 6 * s),
          "left_ear": (cx + lsign * 16 * s, top), "right_ear": (cx - lsign * 16 * s, top)}
    for side, sg in (("left", lsign), ("right", -lsign)):
        kp[f"{side}_shoulder"] = (cx + sg * sh, sh_y)
        kp[f"{side}_hip"] = (cx + sg * hw, hip_y)
        kp[f"{side}_knee"] = (cx + sg * hw * 1.05, knee_y)
        kp[f"{side}_ankle"] = (cx + sg * hw * 1.05, ank_y)
    # arms: dict image-side sign -> (elbow, wrist) offsets relative to the shoulder (in units of s)
    def arm(sg, name):
        shx = cx + sg * sh
        if name in ("neutral", "sideon", "degenerate"):
            return (shx + sg * 15 * s, sh_y + 90 * s), (shx + sg * 20 * s, sh_y + 170 * s)
        if name == "t":
            return (shx + sg * 80 * s, sh_y), (shx + sg * 160 * s, sh_y)
        if name == "overhead":
            return (shx + sg * 20 * s, sh_y - 90 * s), (shx, sh_y - 180 * s)
        if name == "crossed":
            return (shx + sg * 5 * s, sh_y + 85 * s), (cx - sg * 35 * s, sh_y + 55 * s)
        if name == "bent":
            return (shx + sg * 15 * s, sh_y + 90 * s), (shx + sg * 15 * s + sg * 90 * s, sh_y + 90 * s) if sg > 0 else \
                   (shx + sg * 20 * s, sh_y + 170 * s)
        return (shx + sg * 15 * s, sh_y + 90 * s), (shx + sg * 20 * s, sh_y + 170 * s)
    arm_pose = {"tpose": "t", "t": "t"}.get(pose, pose)
    for side, sg in (("left", lsign), ("right", -lsign)):
        e, w = arm(sg, arm_pose)
        kp[f"{side}_elbow"], kp[f"{side}_wrist"] = e, w
    if pose == "offscreen":
        kp["left_wrist"] = (-5000.0, 100.0)
        kp["right_wrist"] = (6000.0, -3000.0)
    kp = {k: np.array(v, np.float32) for k, v in kp.items()}
    vis = {k: 1.0 for k in kp}
    if pose == "novis":
        for k in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
            vis[k] = 0.0
    for k in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
        vis[k] = min(vis[k], vis_lower)
    if swapped:
        kp, vis = swap_labels(kp, vis)
    return Body(kp=kp, vis=vis, size=size, t=0.0)


def render(warper, body, size=SIZE, assigner=None):
    layer = Layer.empty(size)
    names = (assigner or SideAssigner()).assign(body.kp)
    mode = warper.render(layer, body, names)
    return layer, mode


def alpha_bbox(layer, thresh=0):
    a = layer.img[..., 3]
    ys, xs = np.nonzero(a > thresh)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def kp_bbox(body, names=None, limit=3000):
    pts = np.array([v for k, v in body.kp.items() if (names is None or k in names)], np.float64)
    pts = pts[(np.abs(pts) < limit).all(axis=1)]
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def solid_warper(garment_id, view, cfg=None):
    """ViewWarper for a garment view whose alpha is filled solid inside its outer contour."""
    import contextlib
    import io
    import cv2
    from tryon.garments import GarmentView
    g, _ = warpers()[(garment_id, view)]
    v = g.views[view]
    img = v.image.copy()
    mask = (img[..., 3] > 8).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    img[..., 3] = np.where(filled > 0, 255, 0).astype(np.uint8)
    cfg = cfg or cfg_and_catalog()[0]
    with contextlib.redirect_stdout(io.StringIO()):
        return ViewWarper(GarmentView(view, img, v.anchors, "solid"), g.category, g.sleeve_length, cfg)
