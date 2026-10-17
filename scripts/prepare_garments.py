#!/usr/bin/env python
"""Prepare FitCast garment assets into assets/garments/<outfit_id>/ (see fitcast/garments.py).

Modes:
  --demo         >= 12 procedural outfits drawn with OpenCV primitives, zero downloads
  --import DIR   VITON-HD (cloth/ + cloth-mask/), DressCode (upper_body/, lower_body/, dresses/),
                 any image + same-stem mask under a "*mask*" dir, or plain product photos on white
  --backs        fill in missing top_back / sleeve / bottom_back / thumb in existing outfit folders

Every generated or imported garment is saturation/brightness boosted (projection on white fabric in a
lit room washes color out) and a warning is logged when its mean luminance is too low to project.
Missing backs are synthesized: mirror the front, then remove the collar and any chest graphic by
copying matching fabric from elsewhere on the garment (cv2.inpaint blends seams / is the fallback).
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitcast.garments import PARTS, ensure_rgba, fit_rgba, garment_cfg, make_thumb  # noqa: E402

log = logging.getLogger("fitcast.prepare_garments")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BOTTOM_WORDS = ("lower", "bottom", "pants", "trousers", "jeans", "skirt")
REF_W = 512.0  # demo line widths are given in px at this canvas width and scale with top_size/bottom_size


# ----------------------------------------------------------------------------- config / color

def load_config(path: Path | None = None) -> dict:
    """Read config.yaml and return its `garments:` section completed with fitcast.garments.DEFAULTS."""
    path = Path(path) if path else ROOT / "config.yaml"
    return garment_cfg(yaml.safe_load(path.read_text())["garments"])


def boost_colors(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Scale HSV saturation by `saturation_boost` and value by `brightness_boost` (clipped). Alpha kept."""
    c = garment_cfg(cfg)
    hsv = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(c["saturation_boost"])
    hsv[..., 2] *= float(c["brightness_boost"])
    bgr = cv2.cvtColor(hsv.clip(0, 255).round().astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.dstack([bgr, img[..., 3]]) if img.ndim == 3 and img.shape[2] == 4 else bgr


def mean_luminance(rgba: np.ndarray, cfg: dict) -> float:
    """Mean gray level (0-255) of the opaque pixels (alpha >= `luminance_alpha`); 0 when none."""
    c = garment_cfg(cfg)
    opaque = rgba[..., 3] >= c["luminance_alpha"]
    if not opaque.any():
        return 0.0
    return float(cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2GRAY)[opaque].mean())


def check_luminance(rgba: np.ndarray, cfg: dict, label: str) -> float:
    """Return mean luminance; log a warning when it is below `min_mean_luminance`."""
    c = garment_cfg(cfg)
    lum = mean_luminance(rgba, c)
    if lum < c["min_mean_luminance"]:
        log.warning("%s: mean luminance %.0f < %s; dark garments barely show when projected, "
                    "use a lighter/brighter garment", label, lum, c["min_mean_luminance"])
    return lum


def _hex(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def _median_color(rgba: np.ndarray, cfg: dict) -> tuple[int, int, int]:
    solid = rgba[..., 3] >= garment_cfg(cfg)["luminance_alpha"]
    px = rgba[..., :3][solid] if solid.any() else rgba[..., :3].reshape(-1, 3)
    return tuple(int(v) for v in np.median(px, axis=0))  # type: ignore[return-value]


def _kernel(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


# ----------------------------------------------------------------------------- geometry estimates

def estimate_body_region(rgba: np.ndarray, cfg: dict) -> tuple[float, float, float, float]:
    """Torso sub-rectangle of a top: x from the hem rows (bottom `hem_rows_frac`, sleeves excluded),
    y = the garment's full alpha extent. Normalized, clamped to [0, 1]."""
    c = garment_cfg(cfg)
    a = rgba[..., 3] > 127
    h, w = a.shape
    rows = np.flatnonzero(a.any(axis=1))
    if not len(rows):
        return (0.0, 0.0, 1.0, 1.0)
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    starts, ends = [], []
    for row in a[max(y0, y1 - max(1, round((y1 - y0) * c["hem_rows_frac"]))):y1]:
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(s):
            k = int(np.argmax(e - s))  # longest run = the body; sleeve cuffs are separate runs
            starts.append(s[k])
            ends.append(e[k])
    x0, x1 = (float(np.median(starts)), float(np.median(ends))) if starts else (0.0, float(w))
    r = [x0 / w, y0 / h, x1 / w, y1 / h]
    return tuple(float(min(1.0, max(0.0, v))) for v in r)  # type: ignore[return-value]


def estimate_sleeve(rgba: np.ndarray, cfg: dict) -> str:
    """'long' when a noticeable share of the lower half lies outside the body x-range, else 'short'."""
    c = garment_cfg(cfg)
    a = rgba[..., 3] > 127
    h, w = a.shape
    x0, _, x1, _ = estimate_body_region(rgba, c)
    slack = (x1 - x0) * c["long_sleeve_slack"]
    bx0, bx1 = max(0, int((x0 - slack) * w)), min(w, int(np.ceil((x1 + slack) * w)))
    rows = np.flatnonzero(a.any(axis=1))
    if not len(rows):
        return "short"
    lower = a[(rows[0] + rows[-1]) // 2:]
    outside = int(lower[:, :bx0].sum() + lower[:, bx1:].sum())
    return "long" if outside > c["long_sleeve_frac"] * max(1, int(lower.sum())) else "short"


# ----------------------------------------------------------------------------- back / sleeve synthesis

def detect_graphic(rgba: np.ndarray, cfg: dict) -> np.ndarray:
    """Mask (0/255) of a large chest blob whose colors are not in the fabric palette.

    The palette is k-means (Lab) of the torso around the chest box, so stripes, plaid and color
    blocks that also appear outside the chest do not count as a graphic.
    """
    c = garment_cfg(cfg)
    alpha = rgba[..., 3]
    h, w = alpha.shape
    out = np.zeros((h, w), np.uint8)
    solid = alpha >= c["opaque_alpha"]
    rows = np.flatnonzero((alpha > 0).any(axis=1))
    if not len(rows):
        return out
    gy0, gy1 = int(rows[0]), int(rows[-1]) + 1
    gh = gy1 - gy0
    rx0, _, rx1, _ = estimate_body_region(rgba, c)
    bx0, bx1 = int(rx0 * w), int(np.ceil(rx1 * w))
    bw = bx1 - bx0
    fx0, fy0, fx1, fy1 = c["graphic_chest_box"]
    cx0, cx1 = bx0 + int(fx0 * bw), bx0 + int(fx1 * bw)
    cy0, cy1 = gy0 + int(fy0 * gh), gy0 + int(fy1 * gh)
    k = int(c["graphic_palette_k"])
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return out
    frame = np.zeros((h, w), bool)
    frame[gy0 + int(c["back_collar_frac"] * gh):gy1, bx0:bx1] = True
    frame[cy0:cy1, cx0:cx1] = False
    frame &= solid
    lab = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[frame]
    if len(samples) < 10 * k:
        return out
    samples = np.ascontiguousarray(samples[::max(1, len(samples) // int(c["graphic_samples"]))])
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(samples, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    roi = lab[cy0:cy1, cx0:cx1]
    dist = np.sqrt(((roi[:, :, None, :] - centers[None, None]) ** 2).sum(-1)).min(-1)
    hit = ((dist > c["graphic_color_dist"]) & solid[cy0:cy1, cx0:cx1]).astype(np.uint8) * 255
    hit = cv2.morphologyEx(hit, cv2.MORPH_CLOSE, _kernel(c["graphic_close_px"] // 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hit, connectivity=8)
    area = (cx1 - cx0) * (cy1 - cy0)
    lo, hi = c["graphic_min_area_frac"] * area, c["graphic_max_area_frac"] * area
    keep = [i for i in range(1, n) if lo <= stats[i, cv2.CC_STAT_AREA] <= hi]
    if keep:
        blob = np.isin(labels, keep).astype(np.uint8) * 255
        out[cy0:cy1, cx0:cx1] = cv2.dilate(blob, _kernel(c["graphic_dilate_px"]))
        out[alpha == 0] = 0
    return out


def _patch_fill(color: np.ndarray, comp: np.ndarray, valid: np.ndarray, c: dict) -> bool:
    """Copy fabric into hole `comp` from the offset whose surrounding ring matches best (keeps pattern
    phase). Source pixels must all be `valid`. Returns False when no such offset exists."""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    r = int(c["patch_ring_px"])
    x0, y0 = max(0, int(xs.min()) - r), max(0, int(ys.min()) - r)
    x1, y1 = min(w, int(xs.max()) + 1 + r), min(h, int(ys.max()) + 1 + r)
    if x1 - x0 >= w or y1 - y0 >= h:
        return False
    hole = comp[y0:y1, x0:x1]
    ring = cv2.dilate(hole.astype(np.uint8), _kernel(r)).astype(bool) & ~hole & valid[y0:y1, x0:x1]
    need = (hole | ring).astype(np.float32)
    bad = cv2.matchTemplate((~valid).astype(np.float32), need, cv2.TM_CCORR)
    ok = bad < 0.5
    if not ok.any():
        return False
    if ring.any():
        f = color.astype(np.float32)
        ssd = cv2.matchTemplate(f, f[y0:y1, x0:x1], cv2.TM_SQDIFF, mask=ring.astype(np.uint8) * 255)
        ssd = np.where(ok, ssd, np.inf)
        ok = ssd <= ssd.min() + float(c["patch_ssd_tol"]) * int(ring.sum())
    # Among equally good matches prefer the source straight below/above (no sideways shift), then nearest.
    sy, sx = np.nonzero(ok)
    k = np.lexsort((np.abs(sy - y0), np.abs(sx - x0)))[0]
    color[ys, xs] = color[ys + (sy[k] - y0), xs + (sx[k] - x0)]
    return True


def _fill_component(color: np.ndarray, comp: np.ndarray, valid: np.ndarray, c: dict, depth: int) -> np.ndarray:
    """Patch-fill `comp`; if no source fits, split it in half along its longer side and recurse.
    Filled pixels become `valid` sources for the rest. Returns the pixels left unfilled."""
    if _patch_fill(color, comp, valid, c):
        valid |= comp
        return np.zeros_like(comp)
    ys, xs = np.nonzero(comp)
    if depth <= 0 or len(ys) < 2:
        return comp
    if np.ptp(ys) >= np.ptp(xs):
        first = comp & (np.arange(comp.shape[0])[:, None] < (int(ys.min()) + int(ys.max()) + 1) // 2)
    else:
        first = comp & (np.arange(comp.shape[1])[None, :] < (int(xs.min()) + int(xs.max()) + 1) // 2)
    unfilled = _fill_component(color, first, valid, c, depth - 1)
    return unfilled | _fill_component(color, comp & ~first, valid, c, depth - 1)


def _fill_holes(color: np.ndarray, alpha: np.ndarray, passes: list[np.ndarray], c: dict) -> np.ndarray:
    """Fill hole masks pass by pass, components smallest first, by fabric patch copy; cv2.inpaint
    blends the seams and fills whatever could not be patched. Transparent pixels are never sources."""
    holes = np.zeros(alpha.shape, bool)
    for m in passes:
        holes |= m > 0
    valid = (alpha >= c["opaque_alpha"]) & ~holes
    unfilled = np.zeros(holes.shape, bool)
    for m in passes:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        for i in sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA]):
            unfilled |= _fill_component(color, labels == i, valid, c, int(c["patch_split_depth"]))
    f8 = (holes & ~unfilled).astype(np.uint8)
    seam = int(c["seam_px"])
    band = (cv2.dilate(f8, _kernel(seam)) > 0) & (cv2.erode(f8, _kernel(seam)) == 0) & (alpha > 0)
    todo = (band | unfilled).astype(np.uint8)
    if todo.any():
        r = float(c["inpaint_radius"])
        near = cv2.dilate(todo, _kernel(int(r) + 2)) > 0
        todo = ((todo > 0) | (near & (alpha == 0))).astype(np.uint8) * 255
        color = cv2.inpaint(color, todo, r, cv2.INPAINT_TELEA)
    return color


def synthesize_back(front: np.ndarray, cfg: dict, extra_mask: np.ndarray | None = None) -> np.ndarray:
    """Back view of a top: mirrored front with the collar region and any chest graphic replaced by
    fabric. `extra_mask` (front coordinates, 0/255) marks the front-only details (graphics, pockets,
    buttons) when the caller knows them; only without it is a chest graphic auto-detected.
    Alpha is exactly the mirrored front alpha."""
    c = garment_cfg(cfg)
    back = cv2.flip(front, 1)
    alpha = back[..., 3].copy()
    color = np.ascontiguousarray(back[..., :3])
    if not (alpha >= c["opaque_alpha"]).any():
        return back
    # Transparent pixels borrow the fabric color so inpainting never pulls in black.
    color[alpha == 0] = _median_color(back, c)
    ys, xs = np.nonzero(alpha)
    gx0, gy0 = int(xs.min()), int(ys.min())
    gw, gh = int(xs.max()) + 1 - gx0, int(ys.max()) + 1 - gy0
    collar = np.zeros_like(alpha)
    collar[gy0:gy0 + round(gh * c["back_collar_frac"]), gx0 + round(gw / 3):gx0 + round(2 * gw / 3)] = 255
    collar[alpha == 0] = 0
    if extra_mask is not None:
        details = cv2.dilate(cv2.flip(extra_mask, 1), _kernel(c["graphic_dilate_px"]))
    else:
        details = detect_graphic(np.dstack([color, alpha]), c)
    details[(alpha == 0) | (collar > 0)] = 0
    # Details first (e.g. a zipper running into the collar), then the collar can copy from straight below.
    return np.dstack([_fill_holes(color, alpha, [details, collar], c), alpha])


def make_sleeve_tile(rgba: np.ndarray, cfg: dict) -> np.ndarray:
    """Opaque `sleeve_tile_size` square tile of the main fabric: a patch at the most interior torso
    point, mirror-tiled (reflect) so the repeats have no hard seams."""
    c = garment_cfg(cfg)
    n = int(c["sleeve_tile_size"])
    h, w = rgba.shape[:2]
    solid = (rgba[..., 3] >= c["opaque_alpha"]).astype(np.uint8)
    x0, _, x1, _ = estimate_body_region(rgba, c)
    solid[:, :int(x0 * w)] = 0
    solid[:, int(np.ceil(x1 * w)):] = 0
    dist = cv2.distanceTransform(np.pad(solid, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    half = min(n // 4, int(dist[cy, cx] / np.sqrt(2)) - 1)
    if half < 2:
        tile = np.full((n, n, 3), _median_color(rgba, c), np.uint8)
    else:
        p = rgba[cy - half:cy + half, cx - half:cx + half, :3]
        quad = np.vstack([np.hstack([p, p[:, ::-1]]), np.hstack([p[::-1], p[::-1, ::-1]])])
        reps = -(-n // quad.shape[0])
        tile = np.tile(quad, (reps, reps, 1))[:n, :n]
    return np.dstack([tile, np.full((n, n), 255, np.uint8)])


# ----------------------------------------------------------------------------- import

def _white_bg_mask(bgr: np.ndarray, c: dict) -> np.ndarray:
    fg = (bgr.min(axis=2) <= c["white_bg_threshold"]).astype(np.uint8) * 255
    k = 2 * int(c["import_open_px"]) + 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))  # square: keeps corners
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(fg)
    if contours:  # largest blob, drawn filled = holes closed
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def extract_garment(img: np.ndarray, cfg: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """RGBA garment tightly cropped to its mask. Mask: given (white = garment), else the image's own
    alpha if it has one, else near-white background removal (`white_bg_threshold`)."""
    c = garment_cfg(cfg)
    rgba = ensure_rgba(img)
    bgr = np.ascontiguousarray(rgba[..., :3])
    if mask is None and (rgba[..., 3] < 255).any():
        mask = rgba[..., 3]
    if mask is None:
        m = _white_bg_mask(bgr, c)
    else:
        if mask.ndim == 3:
            mask = cv2.cvtColor(np.ascontiguousarray(mask[..., :3]), cv2.COLOR_BGR2GRAY)
        if mask.shape != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = np.where(mask > 127, 255, 0).astype(np.uint8)
    if cv2.countNonZero(m) < c["import_min_area_frac"] * m.size:
        raise ValueError("garment mask is empty or tiny")
    x, y, w, h = cv2.boundingRect(m)
    return np.dstack([bgr[y:y + h, x:x + w], m[y:y + h, x:x + w]])


def find_garments(src: Path, cfg: dict) -> list[tuple[Path, Path | None, str]]:
    """(image, mask or None, kind) for every garment image under `src`; kind is top|bottom|dress.

    Files under a directory whose name contains "mask" are masks, paired with images by stem.
    VITON-HD: only `cloth/` images are used. DressCode: only `*_1` (garment) photos.
    """
    c = garment_cfg(cfg)
    skip = {s.lower() for s in c["import_skip_dirs"]}
    src = Path(src)

    def dirs(p: Path) -> list[str]:
        return [d.lower() for d in p.relative_to(src).parts[:-1]]

    masks: dict[str, list[Path]] = {}
    images: list[Path] = []
    for p in sorted(q for q in src.rglob("*") if q.is_file() and q.suffix.lower() in IMG_EXT):
        ds = dirs(p)
        if any("mask" in d for d in ds):
            masks.setdefault(p.stem, []).append(p)
        elif not any(d in skip for d in ds):
            images.append(p)
    if any("cloth" in dirs(p) for p in images):
        images = [p for p in images if "cloth" in dirs(p)]
    by_dir: dict[Path, list[Path]] = {}
    for p in images:
        by_dir.setdefault(p.parent, []).append(p)
    images = sorted(p for ps in by_dir.values() for p in ps
                    if p.stem.endswith("_1") or not any(q.stem.endswith("_1") for q in ps))
    out = []
    for p in images:
        cands = masks.get(p.stem, [])
        top_dir = p.relative_to(src).parts[0]
        mask = next((m for m in cands if m.relative_to(src).parts[0] == top_dir), cands[0] if cands else None)
        path_l = "/".join(dirs(p) + [p.stem.lower()])
        kind = "bottom" if any(k in path_l for k in BOTTOM_WORDS) else "dress" if "dress" in path_l else "top"
        out.append((p, mask, kind))
    return out


def import_dataset(src: Path, out_dir: Path, cfg: dict, limit: int | None = None) -> list[str]:
    """Import every garment found under `src` as its own outfit folder; returns summary lines."""
    c = garment_cfg(cfg)
    found = find_garments(src, c)[:limit] if limit else find_garments(src, c)
    if not found:
        log.error("no garment images found under %s", src)
    lines, used = [], set()
    for img_path, mask_path, kind in found:
        oid = f"{kind}_{re.sub(r'[^A-Za-z0-9]+', '_', img_path.stem).strip('_').lower()}"
        while oid in used:
            oid += "_x"
        used.add(oid)
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("unreadable image")
            mask = None
            if mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"unreadable mask {mask_path}")
            rgba = extract_garment(img, c, mask)
        except (ValueError, cv2.error) as e:
            log.error("skipping %s: %s", img_path, e)
            continue
        name = img_path.stem
        if kind == "bottom":
            meta = dict(name=name, category="bottom", sleeve="none", size_table="mens",
                        colors=[_hex(_median_color(rgba, c))])
            lines.append(write_outfit(out_dir, oid, meta, c, bottom=rgba))
        else:
            top = fit_rgba(rgba, tuple(c["top_size"]))
            meta = dict(name=name, category=kind, sleeve=estimate_sleeve(top, c),
                        size_table="womens" if kind == "dress" else "mens", colors=[_hex(_median_color(top, c))])
            lines.append(write_outfit(out_dir, oid, meta, c, top=top))
    return lines


# ----------------------------------------------------------------------------- writing

def write_outfit(out_dir: Path, oid: str, meta: dict, cfg: dict, top: np.ndarray | None = None,
                 bottom: np.ndarray | None = None, top_back: np.ndarray | None = None,
                 bottom_back: np.ndarray | None = None, top_detail: np.ndarray | None = None,
                 region: tuple[float, float, float, float] | None = None) -> str:
    """Boost, derive missing parts (back, sleeve tile, thumb, body region), write `out_dir/oid/`.

    `meta` needs name, category, sleeve, size_table and optionally colors. Returns a summary line.
    """
    c = garment_cfg(cfg)
    tw, th = (int(v) for v in c["top_size"])
    bw, bh = (int(v) for v in c["bottom_size"])
    folder = Path(out_dir) / oid
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    parts: dict[str, np.ndarray] = {}
    lums = []
    if top is not None:
        if top.shape[:2] != (th, tw):
            top, top_detail, region = fit_rgba(top, (tw, th)), None, None
        top = boost_colors(top, c)
        parts["top_front"] = top
        if top_back is not None:
            parts["top_back"] = boost_colors(fit_rgba(top_back, (tw, th)), c)
        else:
            parts["top_back"] = synthesize_back(top, c, top_detail)
        if meta["sleeve"] != "none":
            parts["sleeve"] = make_sleeve_tile(parts["top_back"], c)
        region = region or estimate_body_region(top, c)
        lums.append(f"top={check_luminance(top, c, f'{oid}/top_front'):.0f}")
    if bottom is not None:
        if bottom.shape[:2] != (bh, bw):
            bottom = fit_rgba(bottom, (bw, bh))
        bottom = boost_colors(bottom, c)
        parts["bottom_front"] = bottom
        parts["bottom_back"] = (boost_colors(fit_rgba(bottom_back, (bw, bh)), c) if bottom_back is not None
                                else cv2.flip(bottom, 1))
        lums.append(f"bottom={check_luminance(bottom, c, f'{oid}/bottom_front'):.0f}")
    if not parts:
        raise ValueError(f"{oid}: needs a top or a bottom")
    for part, img in parts.items():
        cv2.imwrite(str(folder / PARTS[part][0]), img)
    cv2.imwrite(str(folder / "thumb.png"), make_thumb(top if top is not None else bottom, c))
    out_meta = {
        "name": str(meta["name"]),
        "category": str(meta["category"]),
        "sleeve": str(meta["sleeve"]),
        "size_table": str(meta["size_table"]),
        "parts": list(parts),
        "top_body_region": [round(float(v), 4) for v in (region or (0.0, 0.0, 1.0, 1.0))],
        "colors": [str(x) for x in meta.get("colors", [])],
    }
    (folder / "meta.yaml").write_text(yaml.safe_dump(out_meta, sort_keys=False))
    return f"{oid:<22} parts={','.join(parts):<50} lum {' '.join(lums)}"


def fill_missing(garments_dir: Path, cfg: dict) -> list[str]:
    """Add missing top_back / sleeve / bottom_back / thumb / top_body_region to existing outfit
    folders (no boost: the fronts are assumed prepared already). Returns summary lines."""
    c = garment_cfg(cfg)
    lines = []
    for folder in sorted(p for p in Path(garments_dir).iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        try:
            meta = yaml.safe_load((folder / "meta.yaml").read_text())
            if not isinstance(meta, dict):
                raise ValueError("meta.yaml is not a mapping")
        except (OSError, ValueError, yaml.YAMLError) as e:
            log.error("skipping %s: %s", folder.name, e)
            continue

        def load(part: str) -> np.ndarray | None:
            path = folder / PARTS[part][0]
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
            return None if img is None else ensure_rgba(img)

        added, lums = [], []
        top, bottom = load("top_front"), load("bottom_front")
        if top is not None:
            back = load("top_back")
            if back is None:
                back = synthesize_back(top, c)
                cv2.imwrite(str(folder / PARTS["top_back"][0]), back)
                added.append("top_back")
            if meta.get("sleeve", "short") != "none" and load("sleeve") is None:
                cv2.imwrite(str(folder / PARTS["sleeve"][0]), make_sleeve_tile(back, c))
                added.append("sleeve")
            if "top_body_region" not in meta:
                meta["top_body_region"] = [round(float(v), 4) for v in estimate_body_region(top, c)]
                added.append("top_body_region")
            lums.append(f"top={check_luminance(top, c, f'{folder.name}/top_front'):.0f}")
        if bottom is not None:
            if load("bottom_back") is None:
                cv2.imwrite(str(folder / PARTS["bottom_back"][0]), cv2.flip(bottom, 1))
                added.append("bottom_back")
            lums.append(f"bottom={check_luminance(bottom, c, f'{folder.name}/bottom_front'):.0f}")
        if top is None and bottom is None:
            log.error("skipping %s: no top_front.png or bottom_front.png", folder.name)
            continue
        if not (folder / "thumb.png").is_file():
            cv2.imwrite(str(folder / "thumb.png"), make_thumb(top if top is not None else bottom, c))
            added.append("thumb")
        meta["parts"] = [p for p, (f, _) in PARTS.items() if (folder / f).is_file()]
        (folder / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        lines.append(f"{folder.name:<22} added={','.join(added) or '-':<40} lum {' '.join(lums)}")
    return lines


# ----------------------------------------------------------------------------- procedural demo

# Palette (BGR). Bold, saturated, nothing near black: the projector can only add light.
RED, YELLOW, CYAN = (40, 40, 235), (20, 215, 255), (230, 200, 0)
BLUE, WHITE, VIOLET = (225, 105, 15), (245, 245, 245), (230, 60, 170)
ORANGE, PINK, CREAM = (0, 140, 255), (180, 40, 240), (200, 245, 255)
EMERALD, CORAL, GOLD, PURPLE = (120, 175, 0), (90, 110, 255), (0, 190, 255), (190, 50, 140)
LIME, GREEN, DENIM, TAN = (0, 230, 160), (40, 160, 60), (220, 120, 50), (120, 175, 215)
SILVER, COPPER, GRAY = (205, 205, 205), (40, 130, 200), (160, 160, 160)

# Silhouettes in normalized canvas coords (x right, y down). Image left = the wearer's right.
TEE_BODY = [(0.39, 0.035), (0.25, 0.09), (0.25, 0.31), (0.27, 0.96), (0.73, 0.96), (0.75, 0.31),
            (0.75, 0.09), (0.61, 0.035)]
TEE_BODY_X = (0.25, 0.75)
SHORT_SLEEVE = [(0.30, 0.07), (0.25, 0.09), (0.08, 0.235), (0.15, 0.345), (0.255, 0.29)]
LONG_SLEEVE = [(0.30, 0.07), (0.25, 0.09), (0.17, 0.20), (0.075, 0.835), (0.165, 0.86), (0.255, 0.30)]
LONG_SLEEVE_OUTER = LONG_SLEEVE[1:4]
TANK_BODY = [(0.335, 0.03), (0.42, 0.03), (0.58, 0.03), (0.665, 0.03), (0.675, 0.15), (0.715, 0.25),
             (0.75, 0.31), (0.73, 0.96), (0.27, 0.96), (0.25, 0.31), (0.285, 0.25), (0.325, 0.15)]
DRESS_BODY = [(0.335, 0.03), (0.42, 0.03), (0.58, 0.03), (0.665, 0.03), (0.675, 0.11), (0.70, 0.17),
              (0.67, 0.36), (0.87, 0.96), (0.13, 0.96), (0.33, 0.36), (0.30, 0.17), (0.325, 0.11)]
DRESS_TORSO = (0.29, 0.71, 0.55)  # x0, x1, y1 of the torso part (hip line); the skirt hangs below
PANTS = [(0.22, 0.03), (0.78, 0.03), (0.80, 0.30), (0.76, 0.97), (0.535, 0.97), (0.5, 0.37),
         (0.465, 0.97), (0.24, 0.97), (0.20, 0.30)]
# Pants coordinates above are stretched horizontally about the center by this factor so each leg fills its
# half of the texture (the renderer maps the image-left half onto the image-left leg), corners stay clear.
PANTS_STRETCH = 1.6


def _mirror(pts: list) -> list:
    return [(1.0 - x, y) for x, y in pts]


def _lerp(a, b, t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _shade(c, f: float) -> tuple[int, int, int]:
    return tuple(int(min(255, round(v * f))) for v in c)  # type: ignore[return-value]


class _Draw:
    """Normalized-coordinate drawing on three layers: fabric color, silhouette alpha and a mask of
    front-only details (removed when the back is synthesized)."""

    def __init__(self, fabric: np.ndarray, sx: float = 1.0):
        self.sx = sx  # horizontal stretch about the canvas center applied to every point
        self.img = np.ascontiguousarray(fabric)
        self.h, self.w = fabric.shape[:2]
        self.alpha = np.zeros((self.h, self.w), np.uint8)
        self.detail = np.zeros((self.h, self.w), np.uint8)

    def p(self, pts) -> np.ndarray:
        return np.array([[round((0.5 + (x - 0.5) * self.sx) * (self.w - 1)), round(y * (self.h - 1))]
                         for x, y in pts], np.int32)

    def t(self, px: float) -> int:
        return max(1, round(px * self.w / REF_W))

    def _apply(self, fn, img, alpha, detail) -> None:
        if img is not None:
            fn(self.img, tuple(int(v) for v in img))
        if alpha is not None:
            fn(self.alpha, int(alpha))
        if detail:
            fn(self.detail, 255)

    def poly(self, pts, img=None, alpha=None, detail=False) -> None:
        p = self.p(pts)
        self._apply(lambda L, col: cv2.fillPoly(L, [p], col, cv2.LINE_AA), img, alpha, detail)

    def line(self, pts, px, img=None, alpha=None, detail=False, closed=False) -> None:
        p, t = self.p(pts), self.t(px)
        self._apply(lambda L, col: cv2.polylines(L, [p], closed, col, t, cv2.LINE_AA), img, alpha, detail)

    def ellipse(self, center, axes, px=-1, img=None, alpha=None, detail=False) -> None:
        cc = tuple(int(v) for v in self.p([center])[0])
        ax = (round(axes[0] * self.w), round(axes[1] * self.h))
        t = -1 if px < 0 else self.t(px)
        self._apply(lambda L, col: cv2.ellipse(L, cc, ax, 0, 0, 360, col, t, cv2.LINE_AA), img, alpha, detail)

    def circle(self, center, r, px=-1, img=None, detail=False) -> None:
        self.ellipse(center, (r, r * self.w / self.h), px, img=img, detail=detail)

    def text(self, s, center, scale, px, img, outline=None, detail=False) -> None:
        font, k = cv2.FONT_HERSHEY_DUPLEX, self.w / REF_W
        ol = self.t(px * 1.8)
        (tw, th), _ = cv2.getTextSize(s, font, scale * k, ol)
        org = (round(center[0] * self.w - tw / 2), round(center[1] * self.h + th / 2))
        if outline is not None:
            cv2.putText(self.img, s, org, font, scale * k, outline, ol, cv2.LINE_AA)
        cv2.putText(self.img, s, org, font, scale * k, img, self.t(px), cv2.LINE_AA)
        if detail:
            cv2.putText(self.detail, s, org, font, scale * k, 255, ol, cv2.LINE_AA)

    def rgba(self) -> np.ndarray:
        return np.dstack([self.img, self.alpha])

    def region(self, x0: float, x1: float, y1: float | None = None) -> tuple[float, float, float, float]:
        rows = np.flatnonzero((self.alpha > 127).any(axis=1))
        return (x0, rows[0] / self.h, x1, (rows[-1] + 1) / self.h if y1 is None else y1)


# --- fabrics (BGR, full canvas, so colors under alpha==0 are fabric too)

def _solid(w: int, h: int, c) -> np.ndarray:
    return np.full((h, w, 3), c, np.uint8)


def _hstripes(w: int, h: int, c1, c2, period: float, frac: float = 0.5) -> np.ndarray:
    p = max(2, round(period * h))
    img = _solid(w, h, c1)
    img[(np.arange(h) % p) < p * frac] = c2
    return img


def _plaid(w: int, h: int, base, bands) -> np.ndarray:
    """bands: (color, opacity, period, width, offset), sizes as fractions of the width."""
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    x, y = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    for color, opacity, period, width, offset in bands:
        p, bw, off = period * w, width * w, offset * w
        vx = (((x - off) % p) < bw).astype(np.float32)
        hy = (((y - off) % p) < bw).astype(np.float32)
        m = np.minimum(1.0, opacity * 0.6 * (vx[None, :] + hy[:, None]))[..., None]
        img = img * (1 - m) + np.array(color, np.float32) * m
    return img.round().astype(np.uint8)


def _tiedye(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - 0.5 * w, yy - 0.42 * h
    r = np.hypot(dx, dy) / w
    ang = np.degrees(np.arctan2(dy, dx))
    hue = ((2 * ang + 760 * r + 14 * np.sin(r * 45)) % 360) / 2
    hsv = np.dstack([hue, np.full_like(hue, 235), np.full_like(hue, 255)]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _dots(w: int, h: int, base, dot, spacing: float, radius: float) -> np.ndarray:
    img = _solid(w, h, base)
    s, r = spacing * w, max(1, round(radius * w))
    for row, y in enumerate(np.arange(s / 2, h + s, s * 0.866)):
        for x in np.arange(s / 2 + (row % 2) * s / 2 - s, w + s, s):
            cv2.circle(img, (round(x), round(y)), r, dot, -1, cv2.LINE_AA)
    return img


def _chevron(w: int, h: int, colors, period: float, amp: float, wave: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tri = np.abs((xx / (wave * w)) % 1.0 - 0.5) * 2
    idx = np.floor((yy + amp * h * tri) / (period * h)).astype(np.int64) % len(colors)
    return np.array(colors, np.uint8)[idx]


def _twill(w: int, h: int, base, amp: float, noise: float, fade: float, seed: int) -> np.ndarray:
    """Denim/chino weave: diagonal twill ridges + soft noise + optional lighter thigh/knee fades."""
    yy, xx = np.mgrid[0:h, 0:w]
    step = max(1, round(3 * w / REF_W))
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    img += ((((xx + yy) // step) % 2) * 2 - 1)[..., None].astype(np.float32) * amp
    if noise:
        n = np.random.default_rng(seed).normal(0, noise, (h, w)).astype(np.float32)
        img += cv2.GaussianBlur(n, (0, 0), 1.5)[..., None]
    if fade:
        f = np.zeros((h, w), np.float32)
        for cx in (0.26, 0.74):  # thighs/knees of the (stretched) pants
            cv2.ellipse(f, (round(cx * w), round(0.5 * h)), (round(0.09 * w), round(0.17 * h)), 0, 0, 360, 1.0, -1)
        f = cv2.GaussianBlur(f, (0, 0), 0.05 * w)[..., None]
        img += (255 - img) * f * fade
    return img.clip(0, 255).round().astype(np.uint8)


# --- garments

def _top(fabric: np.ndarray, base, sleeve: str, rib=None, neck=(0.11, 0.065)) -> _Draw:
    """T-shirt silhouette: tapered body, short/long/no sleeves, crew neckline (neck=None: no cut)."""
    d = _Draw(fabric)
    d.poly(TEE_BODY, alpha=255)
    shape = {"short": SHORT_SLEEVE, "long": LONG_SLEEVE}.get(sleeve)
    for s in ([shape, _mirror(shape)] if shape else []):
        d.poly(s, alpha=255)
    stitch = _shade(base, 0.72)
    d.line([(0.272, 0.935), (0.728, 0.935)], 3, img=stitch)
    for m in (lambda p: p, _mirror):
        if sleeve == "short":
            d.line(m([_lerp((0.08, 0.235), (0.25, 0.09), 0.13), _lerp((0.15, 0.345), (0.255, 0.29), 0.2)]), 3, img=stitch)
        elif sleeve == "long":
            a, b = (0.075, 0.835), (0.165, 0.86)
            cuff = [a, b, _lerp(b, (0.255, 0.30), 0.09), _lerp(a, (0.17, 0.20), 0.09)]
            d.poly(m(cuff), img=_shade(base, 0.85))
    if neck is not None:
        d.ellipse((0.5, 0.035), neck, alpha=0)
        if rib is not None:
            d.ellipse((0.5, 0.035), (neck[0] + 0.012, neck[1] + 0.0095), px=12, img=rib)
    return d


def _graphic_tee(w, h):
    d = _top(_solid(w, h, VIOLET), VIOLET, "short", rib=YELLOW)
    r, cx, cy = 0.13 * w, 0.5, 0.31
    star = [(cx + (r if i % 2 == 0 else 0.45 * r) * np.cos(np.pi / 2 + i * np.pi / 5) / w,
             cy - (r if i % 2 == 0 else 0.45 * r) * np.sin(np.pi / 2 + i * np.pi / 5) / h) for i in range(10)]
    d.poly(star, img=YELLOW, detail=True)
    d.line(star, 6, img=ORANGE, closed=True, detail=True)
    d.text("FC", (0.5, 0.53), 3.2, 12, img=WHITE, outline=ORANGE, detail=True)
    return d


def _plaid_shirt(w, h):
    fab = _plaid(w, h, RED, [(GREEN, 0.9, 0.19, 0.075, 0.0), (YELLOW, 1.2, 0.19, 0.014, 0.11),
                              (WHITE, 1.0, 0.095, 0.006, 0.05)])
    d = _top(fab, RED, "long", neck=(0.10, 0.05))
    dark = _shade(RED, 0.5)
    collar = [(0.40, 0.03), (0.50, 0.085), (0.455, 0.165), (0.35, 0.075)]
    for cl in (collar, _mirror(collar)):
        d.poly(cl, alpha=255, detail=True)
        d.line(cl, 3, img=dark, closed=True)
    d.poly([(0.48, 0.09), (0.52, 0.09), (0.52, 0.96), (0.48, 0.96)], detail=True)
    for x in (0.484, 0.516):
        d.line([(x, 0.10), (x, 0.955)], 2, img=dark)
    for y in np.linspace(0.16, 0.86, 6):
        d.circle((0.5, y), 0.011, img=CREAM, detail=True)
    pocket = [(0.56, 0.22), (0.68, 0.22), (0.68, 0.335), (0.56, 0.335)]
    d.poly(pocket, detail=True)
    d.line(pocket, 3, img=dark, closed=True)
    d.line([(0.56, 0.25), (0.68, 0.25)], 2, img=dark)
    return d


def _hoodie(w, h):
    d = _top(_solid(w, h, ORANGE), ORANGE, "long", neck=None)
    dark = _shade(ORANGE, 0.62)
    d.ellipse((0.5, 0.095), (0.165, 0.08), img=_shade(ORANGE, 0.88), alpha=255)  # hood bunched at the neck
    d.ellipse((0.5, 0.095), (0.165, 0.08), px=3, img=dark)
    d.ellipse((0.5, 0.075), (0.085, 0.06), alpha=0)
    d.ellipse((0.5, 0.075), (0.097, 0.0695), px=12, img=SILVER)  # hood lining
    for x0, x1 in ((0.46, 0.45), (0.54, 0.55)):
        d.circle((x0, 0.14), 0.009, img=GRAY, detail=True)
        d.line([(x0, 0.14), (x1, 0.33)], 5, img=WHITE, detail=True)
        d.line([(x1, 0.33), (x1, 0.355)], 8, img=GRAY, detail=True)
    pocket = [(0.34, 0.60), (0.66, 0.60), (0.71, 0.86), (0.29, 0.86)]
    d.poly(pocket, img=_shade(ORANGE, 0.93), detail=True)
    d.line(pocket, 3, img=dark, closed=True, detail=True)
    for m in (lambda p: p, _mirror):
        d.line(m([(0.355, 0.62), (0.33, 0.72), (0.322, 0.845)]), 4, img=dark)
    d.poly([(0.26, 0.90), (0.74, 0.90), (0.74, 0.97), (0.26, 0.97)], img=_shade(ORANGE, 0.85))
    return d


def _polo(w, h):
    d = _top(_solid(w, h, CORAL), CORAL, "short", neck=(0.09, 0.05))
    dark, collar_c = _shade(CORAL, 0.6), _shade(CORAL, 0.92)
    for m in (lambda p: p, _mirror):  # ribbed sleeve cuffs
        a, b = (0.08, 0.235), (0.15, 0.345)
        d.poly(m([a, b, _lerp(b, (0.255, 0.29), 0.16), _lerp(a, (0.25, 0.09), 0.12)]), img=_shade(CORAL, 0.86))
    d.ellipse((0.5, 0.035), (0.10, 0.058), px=10, img=collar_c)
    flap = [(0.41, 0.035), (0.50, 0.085), (0.465, 0.155), (0.36, 0.075)]
    for f in (flap, _mirror(flap)):
        d.poly(f, img=collar_c, alpha=255, detail=True)
        d.line(f, 3, img=dark, closed=True)
        d.line([_lerp(f[3], f[0], 0.25), _lerp(f[2], f[1], 0.25)], 3, img=WHITE)  # tipping
    placket = [(0.475, 0.09), (0.525, 0.09), (0.525, 0.25), (0.475, 0.25)]
    d.poly(placket, img=_shade(CORAL, 0.96), detail=True)
    d.line(placket, 2, img=dark, closed=True)
    for y in (0.135, 0.205):
        d.circle((0.5, y), 0.011, img=WHITE, detail=True)
    return d


def _tracksuit_top(w, h, base):
    d = _top(_solid(w, h, base), base, "long", neck=(0.10, 0.055))
    d.ellipse((0.5, 0.035), (0.112, 0.067), px=14, img=_shade(base, 0.8), alpha=255)  # stand collar
    for off in (0.022, 0.047):
        s = [(x + off, y) for x, y in LONG_SLEEVE_OUTER]
        d.line(s, 6, img=WHITE)
        d.line(_mirror(s), 6, img=WHITE)
    d.line([(0.5, 0.10), (0.5, 0.955)], 5, img=SILVER, detail=True)
    d.poly([(0.488, 0.10), (0.512, 0.10), (0.512, 0.16), (0.488, 0.16)], img=SILVER, detail=True)
    return d


def _jersey(w, h):
    d = _Draw(_solid(w, h, GOLD))
    d.poly(TANK_BODY, alpha=255)
    d.ellipse((0.5, 0.03), (0.085, 0.13), alpha=0)
    d.ellipse((0.5, 0.03), (0.085, 0.13), px=14, img=PURPLE)
    arm = [TANK_BODY[0]] + [TANK_BODY[11], TANK_BODY[10], TANK_BODY[9]]
    for a in (arm, _mirror(arm)):
        d.line(a, 14, img=PURPLE)
    d.line([(0.272, 0.935), (0.728, 0.935)], 4, img=PURPLE)
    d.text("23", (0.5, 0.45), 4.6, 14, img=PURPLE, outline=WHITE, detail=True)
    return d


def _dress(w, h):
    d = _Draw(_dots(w, h, PINK, CREAM, spacing=0.10, radius=0.028))
    d.poly(DRESS_BODY, alpha=255)
    d.ellipse((0.5, 0.03), (0.08, 0.10), alpha=0)
    d.ellipse((0.5, 0.03), (0.08, 0.10), px=8, img=_shade(PINK, 0.8))
    d.poly([(0.30, 0.335), (0.70, 0.335), (0.70, 0.39), (0.30, 0.39)], img=YELLOW)  # sash
    d.line([(0.135, 0.935), (0.865, 0.935)], 3, img=CREAM)
    return d


def _pants(fabric: np.ndarray, base, style: str) -> np.ndarray:
    """Two legs from a waistband; style: jeans | chinos | track."""
    d = _Draw(fabric, sx=PANTS_STRETCH)
    d.poly(PANTS, alpha=255)
    dark = _shade(base, 0.62)
    both = (lambda p: p, _mirror)
    if style == "track":
        d.poly([(0.2, 0.03), (0.8, 0.03), (0.8, 0.11), (0.2, 0.11)], img=_shade(base, 0.85))
        for y in (0.05, 0.07, 0.09):
            d.line([(0.22, y), (0.78, y)], 1, img=_shade(base, 0.72))
        for m in both:
            d.line(m([(0.49, 0.07), (0.475, 0.17)]), 4, img=WHITE)
            for off in (0.02, 0.045):
                d.line(m([(0.215 + off, 0.11), (0.205 + off, 0.30), (0.245 + off, 0.95)]), 7, img=WHITE)
            d.poly(m([(0.23, 0.925), (0.47, 0.925), (0.47, 0.98), (0.23, 0.98)]), img=_shade(base, 0.8))
        return d.rgba()
    stitch = (40, 170, 240) if style == "jeans" else _shade(base, 0.8)
    d.line([(0.215, 0.09), (0.785, 0.09)], 3, img=dark)
    for y in (0.042, 0.08):
        d.line([(0.22, y), (0.78, y)], 2, img=stitch)
    for x in (0.27, 0.40, 0.60, 0.73):
        loop = [(x - 0.012, 0.025), (x + 0.012, 0.025), (x + 0.012, 0.105), (x - 0.012, 0.105)]
        d.poly(loop, img=_shade(base, 0.9))
        d.line(loop, 2, img=dark, closed=True)
    d.circle((0.5, 0.058), 0.02, img=COPPER if style == "jeans" else _shade(base, 0.6))
    d.line([(0.5, 0.095), (0.5, 0.34)], 3, img=dark)
    d.line([(0.545, 0.095), (0.545, 0.29), (0.535, 0.32), (0.505, 0.335)], 2, img=stitch)
    for m in both:
        d.line(m([(0.243, 0.945), (0.462, 0.945)]), 3, img=stitch)
        if style == "jeans":
            d.line(m([(0.33, 0.095), (0.315, 0.15), (0.28, 0.19), (0.21, 0.21)]), 4, img=dark)
            d.line(m([(0.35, 0.095), (0.335, 0.16), (0.29, 0.205), (0.212, 0.228)]), 2, img=stitch)
            d.line(m([(0.218, 0.30), (0.257, 0.955)]), 2, img=stitch)
            for p in ((0.33, 0.1), (0.215, 0.21)):
                d.circle(m([p])[0], 0.008, img=COPPER)
        else:
            d.line(m([(0.30, 0.095), (0.215, 0.23)]), 4, img=dark)
            d.line(m([(0.345, 0.40), (0.355, 0.955)]), 3, img=_shade(base, 1.12))
    return d.rgba()


def demo_outfits(cfg: dict) -> list[dict]:
    """Specs (meta + RGBA layers) of the procedural demo outfits, sized from top_size/bottom_size."""
    c = garment_cfg(cfg)
    tw, th = (int(v) for v in c["top_size"])
    bw, bh = (int(v) for v in c["bottom_size"])
    jeans = lambda: _pants(_twill(bw, bh, DENIM, 8, 9, 0.35, seed=1), DENIM, "jeans")  # noqa: E731
    chinos = lambda: _pants(_twill(bw, bh, TAN, 4, 4, 0.0, seed=2), TAN, "chinos")  # noqa: E731
    track = lambda col: _pants(_solid(bw, bh, col), col, "track")  # noqa: E731
    x0, x1 = TEE_BODY_X

    def tee(color, rib_f=0.8):
        return _top(_solid(tw, th, color), color, "short", rib=_shade(color, rib_f))

    specs = [
        ("01_tee_red", "Red Tee + Jeans", "short", tee(RED), jeans(), [RED, DENIM]),
        ("02_tee_yellow", "Yellow Tee + Jeans", "short", tee(YELLOW), jeans(), [YELLOW, DENIM]),
        ("03_tee_cyan", "Cyan Tee + Chinos", "short", tee(CYAN), chinos(), [CYAN, TAN]),
        ("04_tee_stripes", "Breton Stripe Tee + Chinos", "short",
         _top(_hstripes(tw, th, WHITE, BLUE, 0.09), BLUE, "short", rib=BLUE), chinos(), [WHITE, BLUE, TAN]),
        ("05_plaid_shirt", "Tartan Shirt + Jeans", "long", _plaid_shirt(tw, th), jeans(), [RED, GREEN, YELLOW, DENIM]),
        ("06_graphic_tee", "Star Graphic Tee + Jeans", "short", _graphic_tee(tw, th), jeans(), [VIOLET, YELLOW, DENIM]),
        ("07_hoodie", "Orange Hoodie + Joggers", "long", _hoodie(tw, th), track(GRAY), [ORANGE, GRAY]),
        ("08_dress", "Polka Dot Dress", "none", _dress(tw, th), None, [PINK, CREAM, YELLOW]),
        ("09_tracksuit", "Emerald Tracksuit", "long", _tracksuit_top(tw, th, EMERALD), track(EMERALD), [EMERALD, WHITE]),
        ("10_polo", "Coral Polo + Chinos", "short", _polo(tw, th), chinos(), [CORAL, TAN]),
        ("11_tiedye_tee", "Tie-Dye Tee + Jeans", "short",
         _top(_tiedye(tw, th), WHITE, "short", rib=WHITE), jeans(), [RED, YELLOW, CYAN, PINK, DENIM]),
        ("12_jersey", "Gold Jersey + Track Pants", "none", _jersey(tw, th), track(PURPLE), [GOLD, PURPLE]),
        ("13_chevron_ls", "Chevron Long Sleeve + Jeans", "long",
         _top(_chevron(tw, th, [LIME, BLUE, WHITE], 0.075, 0.05, 0.16), BLUE, "long", rib=BLUE), jeans(),
         [LIME, BLUE, WHITE, DENIM]),
    ]
    out = []
    for oid, name, sleeve, top, bottom, colors in specs:
        if oid == "08_dress":
            region = top.region(DRESS_TORSO[0], DRESS_TORSO[1], DRESS_TORSO[2])
        else:
            region = top.region(x0, x1)
        out.append(dict(
            oid=oid, top=top.rgba(), top_detail=top.detail, region=region, bottom=bottom,
            meta=dict(name=name, category="dress" if bottom is None else "outfit", sleeve=sleeve,
                      size_table="womens" if oid == "08_dress" else "mens", colors=[_hex(x) for x in colors]),
        ))
    return out


def generate_demo(out_dir: Path, cfg: dict) -> list[str]:
    """Write all procedural demo outfits into `out_dir`; returns one summary line per outfit."""
    c = garment_cfg(cfg)
    return [write_outfit(out_dir, s["oid"], s["meta"], c, top=s["top"], bottom=s["bottom"],
                         top_detail=s["top_detail"], region=s["region"]) for s in demo_outfits(c)]


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="generate the procedural demo outfits")
    mode.add_argument("--import", dest="import_dir", metavar="DIR", type=Path, help="import garments from DIR")
    mode.add_argument("--backs", action="store_true", help="fill missing back/sleeve/thumb in existing outfits")
    ap.add_argument("--out", type=Path, help="garments dir (default: config garments.dir)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--limit", type=int, help="--import: import at most N garments")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    out = args.out or Path(cfg["dir"])
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        lines = generate_demo(out, cfg)
    elif args.import_dir is not None:
        if not args.import_dir.is_dir():
            log.error("--import: %s is not a directory", args.import_dir)
            return 2
        lines = import_dataset(args.import_dir, out, cfg, args.limit)
    else:
        lines = fill_missing(out, cfg)
    for line in lines:
        print(line)
    print(f"{len(lines)} outfits -> {out}")
    return 0 if lines else 1


if __name__ == "__main__":
    sys.exit(main())
