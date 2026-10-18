"""Self-service calibration: the person stands anywhere in the beam (T-pose), dim discs are projected one at a
time over the projector frame, and the discs that land on the body give the camera->projector homography at the
person's distance (discs on the wall give the wall plane). No boxes, no operator, no depth sensor."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .calibration import Calibration, capture_median, detect_one_marker

log = logging.getLogger("fitcast.selfcal")


def disc_grid(cfg: dict, proj_size: tuple[int, int]) -> np.ndarray:
    """(N, 2) projector-pixel disc centres: `grid` = [cols, rows] inset by `margin_frac`."""
    cols, rows = int(cfg["grid"][0]), int(cfg["grid"][1])
    pw, ph = proj_size
    m = float(cfg["margin_frac"])
    xs = np.linspace(pw * m, pw * (1 - m), cols)
    ys = np.linspace(ph * m, ph * (1 - m), rows)
    return np.array([(x, y) for y in ys for x in xs], np.float64)


def render_disc(proj_size: tuple[int, int], pt: np.ndarray, radius_px: int, level: int) -> np.ndarray:
    """Black projector frame with one grey disc (dim on purpose: it may briefly cross the face)."""
    pw, ph = proj_size
    f = np.zeros((ph, pw, 3), np.uint8)
    cv2.circle(f, (int(round(pt[0])), int(round(pt[1]))), int(radius_px), (level, level, level), -1, cv2.LINE_AA)
    return f


def _inside(box: tuple[int, int, int, int], p: np.ndarray, pad: float = 0.0) -> bool:
    x0, y0, x1, y1 = box
    return (x0 - pad) <= p[0] <= (x1 + pad) and (y0 - pad) <= p[1] <= (y1 + pad)


def self_calibrate(cfg: dict, calib_cfg: dict, cal: Calibration, proj_size: tuple[int, int], capture_fn, show_fn,
                   person_mask: np.ndarray, head_box: tuple[int, int, int, int], torso_px: float,
                   log_fn=log.info) -> dict:
    """Project `grid` discs one by one; classify each detected disc as body (inside `person_mask`) or wall.

    Returns {"H_body", "rms_body", "n_body", "H_wall", "rms_wall", "n_wall", "n_discs", "skipped_head", "torso_px"}.
    H_* are camera->projector homographies (None when fewer than 4 points). Discs whose predicted landing point
    (through the current calibration, when one exists) or detected position falls in the head box are skipped.
    """
    pts = disc_grid(cfg, proj_size)
    settle, nmed = int(calib_cfg["settle_frames"]), int(calib_cfg["median_frames"])
    radius, level = int(cfg["disc_radius_px"]), int(cfg["disc_level"])
    head_pad = float(cfg["head_pad_px"])
    black = np.zeros((proj_size[1], proj_size[0], 3), np.uint8)
    dark = capture_median(black, capture_fn, show_fn, settle, nmed)
    body_cam, body_proj, wall_cam, wall_proj = [], [], [], []
    skipped = 0
    for pt in pts:
        if cal.ok:
            pred = cal.unproject_points(pt.reshape(1, 2))[0]
            if _inside(head_box, pred, head_pad):
                skipped += 1
                continue
        lit = capture_median(render_disc(proj_size, pt, radius, level), capture_fn, show_fn, settle, nmed)
        c = detect_one_marker(lit, dark, calib_cfg)
        if c is None:
            continue
        if _inside(head_box, c, head_pad):
            skipped += 1
            continue
        xi, yi = int(round(c[0])), int(round(c[1]))
        on_body = 0 <= yi < person_mask.shape[0] and 0 <= xi < person_mask.shape[1] and person_mask[yi, xi] > 0
        (body_cam if on_body else wall_cam).append(c)
        (body_proj if on_body else wall_proj).append(pt)
    show_fn(black)
    out = {"n_discs": len(pts), "skipped_head": skipped, "torso_px": float(torso_px),
           "H_body": None, "rms_body": None, "n_body": len(body_cam), "H_wall": None, "rms_wall": None, "n_wall": len(wall_cam)}
    thr = float(calib_cfg["ransac_reproj_px"])
    for key, cam, proj in (("body", body_cam, body_proj), ("wall", wall_cam, wall_proj)):
        if len(cam) >= 4:
            cam_a, proj_a = np.asarray(cam, np.float64), np.asarray(proj, np.float64)
            method = cv2.RANSAC if len(cam) > 4 else 0
            H, inl = cv2.findHomography(cam_a, proj_a, method, thr)
            if H is not None:
                H = H / H[2, 2]
                pr = cv2.perspectiveTransform(cam_a.reshape(-1, 1, 2), H).reshape(-1, 2)
                err = np.linalg.norm(pr - proj_a, axis=1)
                keep = err < thr if inl is None else inl.ravel().astype(bool)
                rms = float(np.sqrt(np.mean(err[keep] ** 2))) if keep.any() else float("inf")
                out[f"H_{key}"], out[f"rms_{key}"], out[f"n_{key}"] = H, rms, int(keep.sum())
    log_fn(f"self-calibration: {len(pts)} discs, {out['n_body']} on the body (rms {out['rms_body']}), "
           f"{out['n_wall']} on the wall (rms {out['rms_wall']}), {skipped} skipped near the head, torso {torso_px:.0f} px")
    return out
