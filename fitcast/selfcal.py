"""Self-service calibration: the person stands anywhere in the beam (T-pose), dim discs are projected one at a
time over the projector frame, and the discs that land on the body give the camera->projector homography at the
person's distance (discs on the wall give the wall plane). No boxes, no operator, no depth sensor."""
from __future__ import annotations

import logging
import time

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
    settle = int(cfg.get("settle_frames", calib_cfg["settle_frames"]))
    nmed = int(cfg.get("median_frames", calib_cfg["median_frames"]))
    radius, level = int(cfg["disc_radius_px"]), int(cfg["disc_level"])
    head_pad = float(cfg["head_pad_px"])
    black = np.zeros((proj_size[1], proj_size[0], 3), np.uint8)
    # let the camera's auto exposure settle on the black-projected scene before any reference is taken
    show_fn(black)
    t_end = time.perf_counter() + float(cfg.get("initial_settle_s", 1.0))
    while time.perf_counter() < t_end:
        capture_fn()
    body_cam, body_proj, wall_cam, wall_proj = [], [], [], []
    skipped = 0
    n_dbg = 0
    x0 = max(0, int(head_box[0] - head_pad)); y0 = max(0, int(head_box[1] - head_pad))
    x1 = min(person_mask.shape[1], int(head_box[2] + head_pad)); y1 = min(person_mask.shape[0], int(head_box[3] + head_pad))
    # pixels that may never win a detection: the padded head box and a band along the silhouette (a few px of
    # body sway between the dark and the lit capture makes those edges far brighter in the difference than a disc)
    keep = np.full(person_mask.shape, 255, np.uint8)
    if x1 > x0 and y1 > y0:
        keep[y0:y1, x0:x1] = 0
    r = int(cfg.get("edge_band_px", 18))
    if r > 0:
        k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        band = cv2.dilate(person_mask, k) & cv2.bitwise_not(cv2.erode(person_mask, k))
        keep[band > 0] = 0

    def probe(pt: np.ndarray, tag: str = "disc"):
        """Dark reference immediately before each disc: immune to auto-exposure drift and slow body sway."""
        nonlocal n_dbg
        if tag != "disc":
            n_dbg = 0 if not hasattr(probe, "_t") else probe._t
        dark = capture_median(black, capture_fn, show_fn, settle, nmed)
        lit = capture_median(render_disc(proj_size, pt, radius, level), capture_fn, show_fn, settle, nmed)
        lit_m = cv2.bitwise_and(lit, lit, mask=keep)
        dark_m = cv2.bitwise_and(dark, dark, mask=keep)
        c = detect_one_marker(lit_m, dark_m, calib_cfg)
        if n_dbg < int(cfg.get("debug_images", 6)):
            vis = lit.copy()
            if c is not None:
                cv2.circle(vis, (int(c[0]), int(c[1])), 12, (0, 255, 0), 2)
            diff = cv2.convertScaleAbs(cv2.absdiff(lit_m, dark_m), alpha=4)
            cv2.imwrite(f"out/selfcal_{tag}_{n_dbg:02d}.jpg", np.hstack([vis, diff]))
            n_dbg += 1
            if tag != "disc":
                probe._t = n_dbg
        return c

    def on_body(c) -> bool:
        xi, yi = int(round(c[0])), int(round(c[1]))
        return 0 <= yi < person_mask.shape[0] and 0 <= xi < person_mask.shape[1] and person_mask[yi, xi] > 0

    blind_top = float(cfg.get("blind_top_frac", 0.0)) * proj_size[1]
    for pt in pts:
        if cal.ok:
            if _inside(head_box, cal.unproject_points(pt.reshape(1, 2))[0], head_pad):
                skipped += 1
                continue
        elif pt[1] < blind_top:          # no mapping yet: keep the first pass out of the top band where the head is
            skipped += 1
            continue
        c = probe(pt)
        if c is None:
            continue
        (body_cam if on_body(c) else wall_cam).append(c)
        (body_proj if on_body(c) else wall_proj).append(pt)
    show_fn(black)
    out = {"n_discs": len(pts), "skipped_head": skipped, "torso_px": float(torso_px),
           "H_body": None, "rms_body": None, "n_body": len(body_cam), "H_wall": None, "rms_wall": None, "n_wall": len(wall_cam)}
    thr = float(calib_cfg["ransac_reproj_px"])

    def _fit_h(cam, proj):
        cam_a, proj_a = np.asarray(cam, np.float64), np.asarray(proj, np.float64)
        H, inl = cv2.findHomography(cam_a, proj_a, cv2.RANSAC if len(cam) > 4 else 0, thr)
        if H is None:
            return None, None, 0
        H = H / H[2, 2]
        pr = cv2.perspectiveTransform(cam_a.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(pr - proj_a, axis=1)
        keep = err < thr if inl is None else inl.ravel().astype(bool)
        return H, (float(np.sqrt(np.mean(err[keep] ** 2))) if keep.any() else float("inf")), int(keep.sum())

    def _sane(H, cam_size_wh):
        """The projector frame unprojected into the camera must be a convex quad of reasonable size."""
        pw, ph = proj_size
        corners = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float64).reshape(-1, 1, 2)
        try:
            q = cv2.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return False
        area = abs(cv2.contourArea(q.astype(np.float32)))
        ok = bool(np.isfinite(q).all() and 0.1 * cam_size_wh[0] * cam_size_wh[1] < area < 30.0 * cam_size_wh[0] * cam_size_wh[1])
        log_fn(f"self-calibration: projector footprint in camera px {q.round(0).tolist()} area/frame {area / (cam_size_wh[0] * cam_size_wh[1]):.2f} -> {'ok' if ok else 'REJECTED'}")
        return ok

    cam_wh = (person_mask.shape[1], person_mask.shape[0])
    H_wall = rms_wall = None
    if len(wall_cam) >= 4:
        H_wall, rms_wall, n_w = _fit_h(wall_cam, wall_proj)
        if H_wall is not None and not _sane(H_wall, cam_wh):
            log_fn("self-calibration: wall mapping rejected (degenerate footprint)")
            H_wall = None
        else:
            out["n_wall"] = n_w
    if H_wall is not None and len(body_cam) < int(cfg.get("body_target_min", 8)):
        # second pass: aim a finer grid of discs at the body (projector-space bbox of the person mask via the wall
        # mapping), skipping the head region; these give the person's plane correction with many points
        ys, xs = np.nonzero(person_mask)
        if len(xs):
            bbox_cam = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float64)
            bp = cv2.perspectiveTransform(bbox_cam.reshape(-1, 1, 2), H_wall).reshape(-1, 2)
            bx0, by0 = bp.min(axis=0); bx1, by1 = bp.max(axis=0)
            bx0, bx1 = max(0, bx0), min(proj_size[0] - 1, bx1); by0, by1 = max(0, by0), min(proj_size[1] - 1, by1)
            gc, gr = int(cfg.get("body_grid", [4, 6])[0]), int(cfg.get("body_grid", [4, 6])[1])
            targets = [(x, y) for y in np.linspace(by0, by1, gr + 2)[1:-1] for x in np.linspace(bx0, bx1, gc + 2)[1:-1]]
            n2 = 0
            for pt in targets:
                pt = np.array(pt, np.float64)
                pred = cv2.perspectiveTransform(pt.reshape(1, 1, 2), np.linalg.inv(H_wall)).reshape(2)
                if _inside(head_box, pred, head_pad):
                    skipped += 1
                    continue
                c = probe(pt, "body")
                if c is not None and on_body(c):
                    body_cam.append(c); body_proj.append(pt); n2 += 1
            show_fn(black)
            log_fn(f"self-calibration: targeted pass added {n2} body discs ({len(targets)} aimed)")
            out["n_body"] = len(body_cam)
    H_body = rms_body = None
    if H_wall is not None and len(body_cam) >= 2:
        # the person's plane differs from the wall plane by (nearly) a similarity in projector space:
        # fit it on the body discs on top of the well-conditioned wall mapping
        cam_b, proj_b = np.asarray(body_cam, np.float64), np.asarray(body_proj, np.float64)
        via_wall = cv2.perspectiveTransform(cam_b.reshape(-1, 1, 2), H_wall).reshape(-1, 2)
        A, inl = cv2.estimateAffinePartial2D(via_wall, proj_b, method=cv2.RANSAC, ransacReprojThreshold=thr)
        if A is not None:
            sc = float(np.hypot(A[0, 0], A[1, 0]))
            n_in = int(inl.sum()) if inl is not None else 0
            if not (0.5 < sc < 2.0) or n_in < int(cfg.get("min_body_inliers", 6)) or n_in < 0.6 * len(body_cam):
                log_fn(f"self-calibration: body correction rejected (scale {sc:.2f}, inliers {n_in}/{len(body_cam)})")
                A = None
            else:
                log_fn(f"self-calibration: body correction scale {sc:.3f} shift ({A[0, 2]:.0f}, {A[1, 2]:.0f}) proj px, inliers {n_in}/{len(body_cam)}")
        if A is not None:
            H_body = np.vstack([A, [0, 0, 1]]) @ H_wall
            H_body = H_body / H_body[2, 2]
            pr = cv2.perspectiveTransform(cam_b.reshape(-1, 1, 2), H_body).reshape(-1, 2)
            keep = inl.ravel().astype(bool)
            rms_body = float(np.sqrt(np.mean(np.sum((pr[keep] - proj_b[keep]) ** 2, axis=1))))
            if not _sane(H_body, cam_wh):
                H_body, rms_body = None, None
    elif H_wall is None and len(body_cam) >= 6:
        H_body, rms_body, _ = _fit_h(body_cam, body_proj)
        if H_body is not None and not _sane(H_body, cam_wh):
            H_body, rms_body = None, None
    if H_body is None and H_wall is not None:
        H_body, rms_body = H_wall, rms_wall          # no usable body correction: the wall mapping is still far better than nothing
        log_fn("self-calibration: using the wall mapping for the body (too few body discs for a correction)")
    out.update({"H_body": H_body, "rms_body": rms_body, "H_wall": H_wall, "rms_wall": rms_wall})
    log_fn(f"self-calibration: {len(pts)} discs, {out['n_body']} on the body (rms {out['rms_body']}), "
           f"{out['n_wall']} on the wall (rms {out['rms_wall']}), {skipped} skipped near the head, torso {torso_px:.0f} px")
    return out


# ---------------------------------------------------------------------------------------------
# Dense scan: Gray code over the whole beam -> per-pixel camera->projector map on the body and the wall
# ---------------------------------------------------------------------------------------------
def _head_rect_proj(head_box, head_pad: float, H: np.ndarray, proj_size: tuple[int, int], pad_proj: float) -> tuple[int, int, int, int]:
    """Padded head box mapped into projector pixels (bounding rect, clipped). Blacked out in every projected frame."""
    x0, y0, x1, y1 = head_box
    pts = np.array([[x0 - head_pad, y0 - head_pad], [x1 + head_pad, y0 - head_pad],
                    [x1 + head_pad, y1 + head_pad], [x0 - head_pad, y1 + head_pad]], np.float64).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    pw, ph = proj_size
    px0, py0 = int(max(0, np.floor(p[:, 0].min() - pad_proj))), int(max(0, np.floor(p[:, 1].min() - pad_proj)))
    px1, py1 = int(min(pw, np.ceil(p[:, 0].max() + pad_proj))), int(min(ph, np.ceil(p[:, 1].max() + pad_proj)))
    return px0, py0, px1, py1


def dense_scan(cfg: dict, cal3d_cfg: dict, proj_size: tuple[int, int], capture_fn, show_fn, person_mask: np.ndarray,
               head_box: tuple[int, int, int, int], H_guess: np.ndarray, log_fn=log.info,
               split_y: float | None = None) -> dict:
    """Gray-code scan (44 patterns + white + black, ~10 s): every lit camera pixel gets its projector pixel.
    Body pixels (eroded person mask, head excluded) and wall pixels are fitted with RANSAC homographies.
    The head region (through `H_guess`, the disc-pass body plane) is black in every pattern.

    Returns {"H_body", "rms_body", "n_body", "inlier_frac_body", "res_med", "res_p95", "H_wall", "rms_wall", "n_wall",
             "proj_xy", "valid", "body"}; H_* None when the fit failed."""
    from .calib3d import gray_code_patterns, decode_gray_code
    patterns, white, black = gray_code_patterns(proj_size)
    settle, nmed = int(cfg.get("settle_frames", 5)), int(cfg.get("median_frames", 2))
    head_pad = float(cfg["head_pad_px"])
    hx0, hy0, hx1, hy1 = _head_rect_proj(head_box, head_pad, H_guess, proj_size, float(cfg.get("dense_head_pad_proj_px", 40)))

    lvl = int(cfg.get("dense_level", 170))

    def to_frame(g: np.ndarray) -> np.ndarray:
        f = cv2.cvtColor((g.astype(np.uint16) * lvl // 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if hx1 > hx0 and hy1 > hy0:
            f[hy0:hy1, hx0:hx1] = 0          # SAFETY: never light the head, even during the scan
        return f

    caps = [capture_median(to_frame(pat), capture_fn, show_fn, settle, nmed) for pat in patterns]
    wcap = capture_median(to_frame(white), capture_fn, show_fn, settle, nmed)
    bcap = capture_median(to_frame(black), capture_fn, show_fn, settle, nmed)
    show_fn(np.zeros((proj_size[1], proj_size[0], 3), np.uint8))
    dcfg = dict(cal3d_cfg)
    dcfg["gray_code_black_threshold"] = float(cfg.get("dense_black_threshold", 12))
    dcfg["gray_code_white_threshold"] = float(cfg.get("dense_white_threshold", 4))
    proj_xy, valid = decode_gray_code(caps, wcap, bcap, proj_size, dcfg)
    wg = wcap if wcap.ndim == 2 else wcap.max(axis=2)
    valid &= wg < int(cfg.get("dense_sat_level", 250))        # clipped pixels decode wrong bits (white fabric)

    r = int(cfg.get("edge_band_px", 18))
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    body = valid & (cv2.erode(person_mask, k) > 0)
    x0 = max(0, int(head_box[0] - head_pad)); y0 = max(0, int(head_box[1] - head_pad))
    x1 = min(person_mask.shape[1], int(head_box[2] + head_pad)); y1 = min(person_mask.shape[0], int(head_box[3] + head_pad))
    body[y0:y1, x0:x1] = False
    wall = valid & (cv2.dilate(person_mask, k) == 0)
    thr = float(cfg.get("dense_ransac_px", 6.0))
    max_pts = int(cfg.get("dense_max_points", 4000))

    def fit(sel: np.ndarray):
        ys, xs = np.nonzero(sel)
        n = len(xs)
        if n < 50:
            return None, None, 0, 0.0, None
        idx = np.linspace(0, n - 1, min(n, max_pts)).astype(int)
        cam = np.stack([xs[idx], ys[idx]], 1).astype(np.float64)
        proj = proj_xy[ys[idx], xs[idx]].astype(np.float64)
        # coarse RANSAC (wide gate: the body is one tilted plane, chest forward of the shins) then least squares on
        # the inliers so the whole body, not just its largest flat part, defines the plane
        H, inl = cv2.findHomography(cam, proj, cv2.RANSAC, float(cfg.get("dense_coarse_px", 20.0)))
        if H is None:
            return None, None, 0, 0.0, None
        keep = inl.ravel().astype(bool)
        if keep.sum() >= 8:
            H2, _ = cv2.findHomography(cam[keep], proj[keep], 0)
            if H2 is not None:
                H = H2
        H = H / H[2, 2]
        err0 = np.linalg.norm(cv2.perspectiveTransform(cam.reshape(-1, 1, 2), H).reshape(-1, 2) - proj, axis=1)
        keep = err0 < thr
        err = np.linalg.norm(cv2.perspectiveTransform(cam.reshape(-1, 1, 2), H).reshape(-1, 2) - proj, axis=1)
        rms = float(np.sqrt(np.mean(err[keep] ** 2))) if keep.any() else float("inf")
        return H, rms, int(keep.sum()), float(keep.mean()), err

    H_body, rms_body, n_body, frac_body, err_body = fit(body)
    H_wall, rms_wall, n_wall, frac_wall, _ = fit(wall)
    # upper / lower body planes split at the hip line (camera y): the chest sits in front of the shins
    H_up = H_lo = None
    n_up = n_lo = 0
    rms_up = rms_lo = None
    if split_y is not None:
        rows = np.arange(body.shape[0], dtype=np.float32)[:, None]
        H_up, rms_up, n_up, _, _ = fit(body & (rows < float(split_y)))
        H_lo, rms_lo, n_lo, _, _ = fit(body & (rows >= float(split_y)))
        log_fn(f"dense scan: upper body plane {n_up} inliers (rms {None if rms_up is None else round(rms_up, 2)} px), "
               f"lower body plane {n_lo} inliers (rms {None if rms_lo is None else round(rms_lo, 2)} px)")
    res_med = res_p95 = None
    if err_body is not None:
        res_med, res_p95 = float(np.median(err_body)), float(np.percentile(err_body, 95))
        # residual map of the body-plane model on every decoded body pixel (diagnostic image)
        ys, xs = np.nonzero(body)
        cam_all = np.stack([xs, ys], 1).astype(np.float64)
        e_all = np.linalg.norm(cv2.perspectiveTransform(cam_all.reshape(-1, 1, 2), H_body).reshape(-1, 2)
                               - proj_xy[ys, xs].astype(np.float64), axis=1)
        img = np.zeros(person_mask.shape, np.float32)
        img[ys, xs] = e_all
        heat = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=255.0 / 20.0), cv2.COLORMAP_JET)
        heat[~body] = 0
        cv2.imwrite("out/selfcal_residual.png", heat)
        np.savez_compressed("out/selfcal_dense.npz", proj_xy=proj_xy, valid=valid, body=body, person_mask=person_mask,
                            head_box=np.array(head_box), H_body=H_body, H_guess=H_guess)
        # upper vs lower body residual (split at the vertical middle of the body pixels): a large gap means the
        # body is not one plane for this camera/projector baseline
        ymid = 0.5 * (ys.min() + ys.max())
        up, lo = e_all[ys < ymid], e_all[ys >= ymid]
        if len(up) and len(lo):
            log_fn(f"dense scan: residual median upper body {np.median(up):.1f} px, lower body {np.median(lo):.1f} px")
    log_fn(f"dense scan: {int(valid.sum())} decoded px, body {int(body.sum())} px -> plane from {n_body} inliers "
           f"(rms {rms_body if rms_body is None else round(rms_body, 2)} px, {frac_body:.0%} inliers, residual median "
           f"{None if res_med is None else round(res_med, 1)} / p95 {None if res_p95 is None else round(res_p95, 1)} px); "
           f"wall {n_wall} inliers (rms {rms_wall if rms_wall is None else round(rms_wall, 2)} px)")
    return {"H_body": H_body, "rms_body": rms_body, "n_body": n_body, "inlier_frac_body": frac_body,
            "res_med": res_med, "res_p95": res_p95, "H_wall": H_wall, "rms_wall": rms_wall, "n_wall": n_wall,
            "H_upper": H_up, "n_upper": n_up, "rms_upper": rms_up, "H_lower": H_lo, "n_lower": n_lo, "rms_lower": rms_lo,
            "proj_xy": proj_xy, "valid": valid, "body": body}


def render_dot_mesh(proj_size: tuple[int, int], person_mask: np.ndarray, head_box: tuple[int, int, int, int],
                    H: np.ndarray, cfg: dict) -> np.ndarray:
    """Green dot mesh over the body plus a dot outline of the silhouette, mapped through `H`: shown for a few
    seconds after calibration so the mapping can be checked by eye (the dots must sit on the body). Head black."""
    pw, ph = proj_size
    f = np.zeros((ph, pw, 3), np.uint8)
    rad = int(cfg.get("mesh_dot_radius_px", 5))
    sp = int(cfg.get("mesh_spacing_px", 24))
    step = int(cfg.get("mesh_outline_step_px", 12))
    head_pad = float(cfg["head_pad_px"])
    inner = cv2.erode(person_mask, np.ones((7, 7), np.uint8))
    ys, xs = np.mgrid[sp // 2:person_mask.shape[0]:sp, sp // 2:person_mask.shape[1]:sp]
    on = inner[ys, xs] > 0
    pts = [np.stack([xs[on], ys[on]], 1).astype(np.float64)]
    cnts, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if len(c) >= step * 3:
            pts.append(c[::step, 0, :].astype(np.float64))
    cam = np.concatenate(pts, 0) if pts else np.zeros((0, 2))
    x0, y0, x1, y1 = head_box
    keep = ~((cam[:, 0] >= x0 - head_pad) & (cam[:, 0] <= x1 + head_pad) & (cam[:, 1] >= y0 - head_pad) & (cam[:, 1] <= y1 + head_pad))
    cam = cam[keep]
    if len(cam):
        proj = cv2.perspectiveTransform(cam.reshape(-1, 1, 2), H).reshape(-1, 2)
        for x, y in proj:
            if 0 <= x < pw and 0 <= y < ph:
                cv2.circle(f, (int(round(x)), int(round(y))), rad, (0, 255, 0), -1, cv2.LINE_AA)
    hx0, hy0, hx1, hy1 = _head_rect_proj(head_box, head_pad, H, proj_size, float(cfg.get("dense_head_pad_proj_px", 40)))
    if hx1 > hx0 and hy1 > hy0:
        f[hy0:hy1, hx0:hx1] = 0              # SAFETY: last operation, head black
    return f
