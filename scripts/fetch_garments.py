#!/usr/bin/env python
"""Fetch free-licensed garment photos from Wikimedia Commons and build them into FitCast outfits.

Steps (default: both):
  fetch   download every MANIFEST photo (Commons thumbnail, <= `FETCH_WIDTH` px wide) into
          assets/garments/_raw/ and record source URL, author and license in _raw/sources.json and
          assets/garments/SOURCES.md. Folders starting with "_" are ignored by fitcast.garments.
  build   cut each photo out of its white background (near-white threshold -> cv2.grabCut -> 2 px
          feather), crop to the garment, fit it to `garments.top_size` / `bottom_size`, compute the
          torso `top_body_region` from the alpha, pair tops with bottoms (PAIRS) and write the outfit
          folders with scripts/prepare_garments.write_outfit (colour boost, synthesized back view,
          sleeve tile, thumbnail, meta.yaml). Garments without a usable free photo are drawn
          procedurally (scripts/procedural_garments.py) and listed as such in SOURCES.md.

Only the outfit folders named in PAIRS are (re)written; the 13 demo outfits are never touched.

    .venv/bin/python scripts/fetch_garments.py            # fetch + build
    .venv/bin/python scripts/fetch_garments.py build      # rebuild offline from _raw/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fitcast.garments import fit_rgba, garment_cfg, resize_rgba  # noqa: E402
from prepare_garments import _hex, _median_color, estimate_body_region, load_config, write_outfit  # noqa: E402
import procedural_garments as pgen  # noqa: E402

log = logging.getLogger("fitcast.fetch_garments")

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FitCastGarmentFetcher/1.0 (offline asset prep; https://commons.wikimedia.org/wiki/Commons:API)"
FETCH_WIDTH = 1600          # Commons thumbnail width requested (originals are up to 50 MP)
WORK_MAX_PX = 1400          # photos are downscaled to this longest side before cutting out
GRABCUT_MAX_PX = 700        # grabCut runs at this longest side (it is O(pixels)); mask is upsampled
GRABCUT_ITERS = 5
FG_ERODE_FRAC = 0.03        # sure-foreground = threshold mask eroded by this fraction of the garment size
BG_DILATE_FRAC = 0.03       # sure-background = outside the threshold mask dilated by this fraction
BG_DIST = 28.0              # Lab distance to the border colour below which a pixel looks like background
FEATHER_PX = 2              # alpha feather at output resolution
MARGIN_FRAC = 0.01          # transparent border kept around bottoms (corners stay clear)
TORSO_HEM_TOL = 0.06        # columns whose alpha reaches within this fraction of the height of the hem = torso
TORSO_MIN_FRAC = 0.30       # a torso span narrower than this fraction of the garment width is not trusted
RETRIES = 3
SMOOTH_D, SMOOTH_SIGMA_COLOR, SMOOTH_SIGMA_SPACE = 9, 30.0, 7.0   # bilateral filter for `smooth` photos
PRINT_CLOSE_PX = 9          # merge print strokes into one blob (px at the top canvas size)
PRINT_DILATE_PX = 4

ALLOWED_LICENSES = ("CC0", "Public domain", "PDM", "CC BY 2.0", "CC BY 2.5", "CC BY 3.0", "CC BY 4.0",
                    "CC BY-SA 2.0", "CC BY-SA 3.0", "CC BY-SA 4.0")

# key -> Commons file, garment kind and per-photo cut-out hints.
#   crop: (x0, y0, x1, y1) normalized region of the photo that holds only this garment
#   white: near-white threshold override (white garments on a white background need a stricter one)
#   bg_dist: Lab distance to the border colour that still counts as background (0 = threshold only)
#   gamma: < 1 brightens a dark garment so it still shows when projected
#   open_frac: remove thin attachments (mannequin stand pole), fraction of the garment size
#   smooth: edge-preserving smoothing (JPEG blocking on near-white fabric gets amplified by the colour boost)
#   print_dist: Lab distance from the fabric colour that marks a chest print, removed from the synthesized back
#   flat_chroma: give fabric-coloured pixels (within print_dist) the median fabric hue (kills JPEG colour blotches)
#   back_title: Commons file with the back view of the same garment
#   attribution: who to credit when the Commons "Artist" field names the garment maker, not the photographer
#   region: manual top_body_region override when the alpha-based estimate is wrong
AM_CREDIT = "Auckland War Memorial Museum Tamaki Paenga Hira"
MANIFEST: dict[str, dict] = {
    "tee_oversized_white": dict(kind="top", title="File:Banfu.jpg", white=250, bg_dist=0, smooth=True, print_dist=40, flat_chroma=True,
                                desc="oversized drop-shoulder white tee with blue bubble-letter print"),
    "tee_green": dict(kind="top", title="File:Ambigram Ideal, polysymmetrical logo printed on a green T-shirt.jpg",
                      desc="olive-green crew tee with white print"),
    "shirt_madras": dict(kind="top", title="File:Madraskarohemd Sir Oliver.jpg",
                         desc="short-sleeve madras plaid button-up shirt"),
    "overshirt_olive": dict(kind="top", title="File:Shirt, military (AM 2015.19.20-1).jpg",
                            back_title="File:Shirt, military (AM 2015.19.20-4).jpg", open_frac=0.012,
                            attribution=AM_CREDIT, desc="olive cotton military overshirt with flap pockets (front + back photo)"),
    "overshirt_scout": dict(kind="top", title="File:Chilean Scouting shirt of San Ignacio.jpg",
                            desc="light grey long-sleeve button overshirt with patches"),
    "sweater_crew": dict(kind="top", title="File:Polo Ralph Lauren Gun Patch Sweater (13973497074).jpg",
                         desc="red-brown crewneck sweater with suede elbow patches"),
    "jeans_flared": dict(kind="bottom",
                         title="File:Flared and Patched Jeans Worn by the Welders - DPLA - "
                               "a69e6a5fdac98d8871aee4ac77de9944 (page 1).jpg",
                         attribution="Missouri Historical Society (garment: Levi Strauss)",
                         desc="faded flared jeans with knee patch"),
    "jeans_grey": dict(kind="bottom", title="File:Puffa Jeans BW (3213392471).jpg", crop=(0.0, 0.40, 1.0, 1.0),
                       desc="washed grey straight jeans (vest above cropped away)"),
    "jeans_worn": dict(kind="bottom", title="File:Jeans BW 2 (3213391837).jpg",
                       desc="heavily faded baggy jeans"),
    "trackpants_navy": dict(kind="bottom", title="File:Tracksuit pants (AM 2017.66.26-3).jpg",
                            crop=(0.0, 0.10, 1.0, 1.0), gamma=0.5, open_frac=0.012, attribution=AM_CREDIT,
                            desc="navy track pants with white side stripes and elastic cuffs"),
    "trousers_khaki": dict(kind="bottom", title="File:Trousers, army (AM 2014.21.11.2-1).jpg",
                           attribution=AM_CREDIT,
                           desc="khaki drill straight trousers (chino stand-in)"),
}

# Outfit folders written by `build`: (id, name, sleeve, top key, bottom key). Keys not in MANIFEST are
# procedural garments from procedural_garments.GARMENTS.
PAIRS: list[tuple[str, str, str, str, str]] = [
    ("14_oversized_tee_flares", "Oversized White Tee + Flared Jeans", "short", "tee_oversized_white", "jeans_flared"),
    ("15_green_tee_trackpants", "Green Tee + Navy Track Pants", "short", "tee_green", "trackpants_navy"),
    ("16_madras_grey_jeans", "Madras Plaid Shirt + Grey Jeans", "short", "shirt_madras", "jeans_grey"),
    ("17_olive_overshirt_cargo", "Olive Overshirt + Cargo Pants", "long", "overshirt_olive", "cargo_olive"),
    ("18_sweater_khakis", "Crewneck Sweater + Khaki Trousers", "long", "sweater_crew", "trousers_khaki"),
    ("19_hoodie_sweatpants", "Heather Hoodie + Sweatpants", "long", "hoodie_heather", "sweatpants_heather"),
    ("20_jersey_baggy_jeans", "Basketball Jersey + Baggy Jeans", "none", "jersey_basketball", "jeans_worn"),
    ("21_oversized_tee_cargo", "Oversized White Tee + Cargo Pants", "short", "tee_oversized_white", "cargo_olive"),
    ("22_scout_overshirt_chinos", "Grey Overshirt + Chinos", "long", "overshirt_scout", "chinos_khaki"),
    ("23_hoodie_trackpants", "Heather Hoodie + Navy Track Pants", "long", "hoodie_heather", "trackpants_navy"),
]


# ----------------------------------------------------------------------------- fetch

def _get(url: str, timeout: float = 60.0) -> bytes:
    """GET with a descriptive User-Agent (Wikimedia policy) and a few retries."""
    err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError as e:  # URLError/HTTPError/timeouts
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {RETRIES} tries: {url}: {err}")


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_info(titles: list[str]) -> dict[str, dict]:
    """title -> {thumb_url, page_url, width, height, license, license_url, artist, credit} from the Commons API."""
    params = dict(action="query", titles="|".join(titles), prop="imageinfo", format="json",
                  iiprop="url|size|mime|extmetadata", iiurlwidth=str(FETCH_WIDTH))
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    norm = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
    out = {}
    for pg in data.get("query", {}).get("pages", {}).values():
        title = norm.get(pg["title"], pg["title"])
        if "imageinfo" not in pg:
            log.error("%s: not found on Commons", title)
            continue
        ii = pg["imageinfo"][0]
        md = ii.get("extmetadata", {})
        out[title] = dict(
            thumb_url=ii.get("thumburl") or ii["url"], page_url=ii["descriptionurl"], width=ii["width"],
            height=ii["height"], license=md.get("LicenseShortName", {}).get("value", "unknown"),
            license_url=md.get("LicenseUrl", {}).get("value", ""),
            artist=_strip_html(md.get("Artist", {}).get("value", "")),
            credit=_strip_html(md.get("Credit", {}).get("value", "")),
        )
    return out


def fetch(raw_dir: Path) -> dict[str, dict]:
    """Download every MANIFEST photo into `raw_dir` and write sources.json. Returns key -> source record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    photos = [(k, m["title"], m) for k, m in MANIFEST.items()]
    photos += [(f"{k}_back", m["back_title"], m) for k, m in MANIFEST.items() if m.get("back_title")]
    info = commons_info([t for _, t, _ in photos])
    sources = {}
    for key, title, m in photos:
        rec = info.get(title)
        if rec is None:
            continue
        if not rec["license"].startswith(ALLOWED_LICENSES):
            log.error("%s: license %r is not in the allowed list; skipped", key, rec["license"])
            continue
        if rec["height"] < 600:
            log.warning("%s: only %d px tall", key, rec["height"])
        path = raw_dir / f"{key}.jpg"
        try:
            path.write_bytes(_get(rec["thumb_url"]))
        except RuntimeError as e:
            log.error("%s: %s; keeping any earlier copy", key, e)
            continue
        if cv2.imread(str(path)) is None:
            path.unlink()
            log.error("%s: downloaded file is not a readable image", key)
            continue
        sources[key] = dict(rec, title=title, file=path.name, kind=m["kind"], desc=m["desc"])
        log.info("fetched %-22s %s (%s)", key, title, rec["license"])
    old = raw_dir / "sources.json"
    if old.is_file():
        for k, v in json.loads(old.read_text()).items():
            if k not in sources and (raw_dir / v["file"]).is_file():
                sources[k] = v
    (raw_dir / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    return sources


# ----------------------------------------------------------------------------- cut-out

def _largest_filled(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def _disk(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def cutout(bgr: np.ndarray, white: int, bg_dist: float = BG_DIST, open_frac: float = 0.0) -> np.ndarray:
    """Hard 0/255 garment mask of a product photo on a (near-)white background.

    Background = near-white (min channel >= `white`) or within `bg_dist` (Lab) of the median border colour;
    the largest remaining blob seeds grabCut (eroded = sure fg, far outside = sure bg). `open_frac` > 0
    removes thin attachments (a mannequin stand pole) with an opening of that fraction of the garment size."""
    h, w = bgr.shape[:2]
    s = min(1.0, GRABCUT_MAX_PX / max(h, w))
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                             lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    bg = (small.min(axis=2) >= white) | (dist < bg_dist)
    fg = cv2.morphologyEx((~bg).astype(np.uint8) * 255, cv2.MORPH_OPEN, _disk(1))
    fg = _largest_filled(fg)
    if cv2.countNonZero(fg) < 0.01 * fg.size:
        raise ValueError("no garment found against the background")
    x, y, bw, bh = cv2.boundingRect(fg)
    size = max(bw, bh)
    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    gc[fg > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(fg, _disk(FG_ERODE_FRAC * size)) > 0] = cv2.GC_FGD
    gc[cv2.dilate(fg, _disk(BG_DILATE_FRAC * size)) == 0] = cv2.GC_BGD
    gc[:2], gc[-2:], gc[:, :2], gc[:, -2:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgm, fgm, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error as e:
        log.warning("grabCut failed (%s); using the threshold mask", e)
        m = fg
    m = _largest_filled(cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(max(1.0, open_frac * size))))
    if s < 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.where(m > 127, 255, 0).astype(np.uint8)
    return m


def garment_rgba(bgr: np.ndarray, white: int, crop: tuple | None = None, bg_dist: float = BG_DIST,
                 gamma: float = 1.0, open_frac: float = 0.0, smooth: bool = False) -> np.ndarray:
    """Cut out and tightly crop a garment photo; RGBA with a hard alpha. `gamma` < 1 lifts dark fabric
    (the projector can only add light; very dark garments barely show)."""
    h, w = bgr.shape[:2]
    if crop is not None:
        x0, y0, x1, y1 = crop
        bgr = bgr[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
        h, w = bgr.shape[:2]
    s = min(1.0, WORK_MAX_PX / max(h, w))
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    m = cutout(bgr, white, bg_dist, open_frac)
    if smooth:
        bgr = cv2.bilateralFilter(bgr, SMOOTH_D, SMOOTH_SIGMA_COLOR, SMOOTH_SIGMA_SPACE)
    if gamma != 1.0:
        lut = (255.0 * (np.arange(256) / 255.0) ** gamma).round().astype(np.uint8)
        bgr = cv2.LUT(bgr, lut)
    x, y, bw, bh = cv2.boundingRect(m)
    return np.dstack([bgr[y:y + bh, x:x + bw], m[y:y + bh, x:x + bw]])


def feather(rgba: np.ndarray, px: int = FEATHER_PX) -> np.ndarray:
    """Soften the alpha edge by ~`px` pixels (inside only: fully transparent stays transparent)."""
    a = rgba[..., 3]
    soft = cv2.GaussianBlur(a, (2 * px + 1, 2 * px + 1), px / 2.0)
    out = rgba.copy()
    out[..., 3] = np.minimum(a, soft)
    return out


# ----------------------------------------------------------------------------- layout

def torso_region(top: np.ndarray, cfg: dict) -> tuple[float, float, float, float]:
    """Normalized torso box of a top canvas: the central column span whose alpha reaches the hem.

    Sleeves stop above the hem, so they fall outside; if they hang down to the hem too (long sleeves
    lying along the body), the span is intersected with the hem-row estimate of prepare_garments."""
    a = top[..., 3] > 127
    h, w = a.shape
    rows = np.flatnonzero(a.any(axis=1))
    cols = np.flatnonzero(a.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    bottom = np.where(a.any(axis=0), h - 1 - np.argmax(a[::-1], axis=0), -1)
    torso = bottom >= y1 - 1 - TORSO_HEM_TOL * (y1 - y0)
    cx = int(round(0.5 * (cols[0] + cols[-1])))
    if not torso[cx]:
        cx = int(np.flatnonzero(torso)[np.argmin(np.abs(np.flatnonzero(torso) - cx))])
    x0 = cx
    while x0 > 0 and torso[x0 - 1]:
        x0 -= 1
    x1 = cx + 1
    while x1 < w and torso[x1]:
        x1 += 1
    hx0, _, hx1, _ = estimate_body_region(top, cfg)
    x0, x1 = max(x0 / w, hx0), min(x1 / w, hx1)
    if x1 - x0 < TORSO_MIN_FRAC * (cols[-1] + 1 - cols[0]) / w:
        x0, x1 = hx0, hx1
    return (float(x0), y0 / h, float(x1), y1 / h)


def flatten_chroma(rgba: np.ndarray, dist: float) -> np.ndarray:
    """Set the Lab a/b of every opaque pixel within `dist` of the median fabric colour to the median a/b."""
    solid = rgba[..., 3] > 127
    lab = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2LAB)
    labf = lab.astype(np.float32)
    med = np.median(labf[solid], axis=0)
    near = (np.linalg.norm(labf - med, axis=2) <= dist) & solid
    lab[..., 1][near] = np.uint8(round(med[1]))
    lab[..., 2][near] = np.uint8(round(med[2]))
    return np.dstack([cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), rgba[..., 3]])


def print_mask(top: np.ndarray, region: tuple[float, float, float, float], dist: float) -> np.ndarray:
    """0/255 mask of a chest print: torso pixels farther than `dist` (Lab) from the median fabric colour,
    closed into blobs and dilated; prepare_garments fills them with fabric on the synthesized back."""
    h, w = top.shape[:2]
    solid = top[..., 3] > 127
    lab = cv2.cvtColor(np.ascontiguousarray(top[..., :3]), cv2.COLOR_BGR2LAB).astype(np.float32)
    far = (np.linalg.norm(lab - np.median(lab[solid], axis=0), axis=2) > dist) & solid
    box = np.zeros_like(far)
    x0, y0, x1, y1 = region
    box[int(y0 * h):int(y1 * h), int(x0 * w):int(np.ceil(x1 * w))] = True
    m = (far & box).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(PRINT_CLOSE_PX))
    m = cv2.dilate(m, _disk(PRINT_DILATE_PX))
    m[~solid] = 0
    return m


def layout_top(rgba: np.ndarray, cfg: dict) -> np.ndarray:
    """Fit a cut-out top into `top_size` (aspect kept, centered) and feather it."""
    return feather(fit_rgba(rgba, tuple(cfg["top_size"])))


def _crotch_x(alpha: np.ndarray) -> int:
    """Column of the gap between the legs: middle of the transparent run nearest the center, in the lower half."""
    h, w = alpha.shape
    gaps = []
    for row in alpha[h // 2:] > 127:
        d = np.diff(np.concatenate([[1], row.astype(np.int8), [1]]))
        s, e = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
        inner = [(a_, b_) for a_, b_ in zip(s, e) if a_ > 0 and b_ < w]
        if inner:
            a_, b_ = min(inner, key=lambda r: abs(0.5 * (r[0] + r[1]) - w / 2))
            gaps.append(0.5 * (a_ + b_))
    return int(round(np.median(gaps))) if gaps else w // 2


def layout_bottom(rgba: np.ndarray, cfg: dict) -> np.ndarray:
    """Stretch a cut-out pair of pants over `bottom_size`: waist on the top row, hems on the bottom row
    (MARGIN_FRAC kept clear), the crotch on the center line so each leg fills its half of the texture
    (the renderer maps the image-left half onto the image-left leg)."""
    bw, bh = (int(v) for v in cfg["bottom_size"])
    mx, my = max(1, round(MARGIN_FRAC * bw)), max(1, round(MARGIN_FRAC * bh))
    cx = min(max(_crotch_x(rgba[..., 3]), 1), rgba.shape[1] - 1)
    half_w, inner_h = bw // 2 - mx, bh - 2 * my
    out = np.zeros((bh, bw, 4), np.uint8)
    out[my:my + inner_h, mx:mx + half_w] = resize_rgba(rgba[:, :cx], (half_w, inner_h))
    out[my:my + inner_h, bw // 2:bw // 2 + half_w] = resize_rgba(rgba[:, cx:], (half_w, inner_h))
    return feather(out)


# ----------------------------------------------------------------------------- build

# (canvas RGBA, kind, front-only detail mask, top_body_region, back-view canvas) - None where not applicable
Loaded = tuple[np.ndarray, str, "np.ndarray | None", "tuple[float, float, float, float] | None", "np.ndarray | None"]


def _photo(raw_dir: Path, name: str, m: dict, cfg: dict) -> np.ndarray:
    bgr = cv2.imread(str(raw_dir / f"{name}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"{raw_dir / (name + '.jpg')} missing; run `scripts/fetch_garments.py fetch`")
    return garment_rgba(bgr, int(m.get("white", cfg["white_bg_threshold"])), m.get("crop"),
                        float(m.get("bg_dist", BG_DIST)), float(m.get("gamma", 1.0)), float(m.get("open_frac", 0.0)),
                        bool(m.get("smooth", False)))


def load_garment(key: str, raw_dir: Path, cfg: dict) -> Loaded:
    """Load a MANIFEST photo (cut out + laid out) or draw a procedural garment; see `Loaded`.
    Photo tops get their region from the alpha (or a MANIFEST override); a `back_title` photo becomes
    the back view, otherwise prepare_garments synthesizes one."""
    if key in MANIFEST:
        m = MANIFEST[key]
        layout = layout_top if m["kind"] == "top" else layout_bottom
        canvas = layout(_photo(raw_dir, key, m, cfg), cfg)
        back = layout(_photo(raw_dir, f"{key}_back", m, cfg), cfg) if m.get("back_title") else None
        if m["kind"] == "top":
            region = tuple(m.get("region") or torso_region(canvas, cfg))
            if m.get("flat_chroma"):
                canvas = flatten_chroma(canvas, float(m["print_dist"]))
            detail = print_mask(canvas, region, float(m["print_dist"])) if m.get("print_dist") else None
            return canvas, "top", detail, region, back
        return canvas, "bottom", None, None, back
    if key in pgen.GARMENTS:
        kind, fn = pgen.GARMENTS[key]
        size = cfg["top_size"] if kind == "top" else cfg["bottom_size"]
        rgba, detail, region = fn(int(size[0]), int(size[1]))
        return rgba, kind, detail, region, None
    raise KeyError(f"unknown garment {key!r}")


def build(raw_dir: Path, out_dir: Path, cfg: dict) -> list[str]:
    """Write every PAIRS outfit folder into `out_dir`; returns prepare_garments summary lines."""
    cache: dict[str, Loaded] = {}
    lines = []
    for oid, name, sleeve, top_key, bottom_key in PAIRS:
        try:
            for k in (top_key, bottom_key):
                if k not in cache:
                    cache[k] = load_garment(k, raw_dir, cfg)
        except (OSError, KeyError, ValueError, cv2.error) as e:
            log.error("skipping %s: %s", oid, e)
            continue
        top, top_kind, detail, region, top_back = cache[top_key]
        bottom, bottom_kind, _, _, bottom_back = cache[bottom_key]
        if (top_kind, bottom_kind) != ("top", "bottom"):
            log.error("skipping %s: %s is a %s and %s a %s", oid, top_key, top_kind, bottom_key, bottom_kind)
            continue
        meta = dict(name=name, category="outfit", sleeve=sleeve, size_table="mens",
                    colors=[_hex(_median_color(top, cfg)), _hex(_median_color(bottom, cfg))])
        lines.append(write_outfit(out_dir, oid, meta, cfg, top=top, bottom=bottom, top_back=top_back,
                                  bottom_back=bottom_back, top_detail=detail, region=tuple(region)))
    return lines


def write_sources_md(path: Path, raw_dir: Path) -> None:
    """Human-readable attribution list for every downloaded photo and every procedural garment."""
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
    used = {k for _, _, _, t, b in PAIRS for k in (t, b)}
    lines = ["# Garment sources", "",
             "Photos below were downloaded from Wikimedia Commons by `scripts/fetch_garments.py`, cut out of",
             "their backgrounds, colour-boosted and resized; the PNGs in the outfit folders listed are derivative",
             "works under the same license (CC BY-SA items stay CC BY-SA). Raw downloads: `_raw/`.", "",
             "| garment | outfits | file | author | license |", "|---|---|---|---|---|"]
    for photo, s in sources.items():
        key = photo[:-5] if photo.endswith("_back") and photo[:-5] in MANIFEST else photo
        if key not in used:
            continue
        outfits = ", ".join(oid for oid, _, _, t, b in PAIRS if key in (t, b))
        if photo != key:
            s = dict(s, desc=s["desc"] + " - back view")
        lic = f"[{s['license']}]({s['license_url']})" if s["license_url"] else s["license"]
        author = (MANIFEST[key].get("attribution") or s["artist"] or s["credit"] or "unknown").replace("|", "/")
        lines.append(f"| {s['desc']} | {outfits} | [{s['title'][5:]}]({s['page_url']}) | {author} | {lic} |")
    lines += ["", "## Procedural (no free photo found)", "",
              "Drawn by `scripts/procedural_garments.py` (no third-party content):", ""]
    for key in pgen.GARMENTS:
        if key in used:
            outfits = ", ".join(oid for oid, _, _, t, b in PAIRS if key in (t, b))
            lines.append(f"- `{key}`: {pgen.DESCRIPTIONS[key]} ({outfits})")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=("all", "fetch", "build"), default="all")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, help="garments dir (default: config garments.dir)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = garment_cfg(load_config(args.config))
    out = args.out or Path(cfg["dir"])
    if not out.is_absolute():
        out = ROOT / out
    raw = out / "_raw"
    if args.step in ("all", "fetch"):
        try:
            fetch(raw)
        except RuntimeError as e:
            log.error("%s; falling back to whatever is already in %s (procedural garments still build)", e, raw)
    lines = []
    if args.step in ("all", "build"):
        lines = build(raw, out, cfg)
        for line in lines:
            print(line)
        print(f"{len(lines)} outfits -> {out}")
    write_sources_md(out / "SOURCES.md", raw)
    return 0 if lines or args.step == "fetch" else 1


if __name__ == "__main__":
    sys.exit(main())
