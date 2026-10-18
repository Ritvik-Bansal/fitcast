#!/usr/bin/env python
"""Fetch real streetwear product photos (Stussy, Fear of God ESSENTIALS, Nike Tech Fleece, Carhartt WIP,
Chrome Hearts, Lakers) and build them into FitCast outfits 40_* .. 59_*.

No free-licence photo of these products exists on Wikimedia Commons / Openverse (checked 2026-10-18), so the
photos come from the brand's or a retailer's own product page: Shopify `/products/<handle>.json` image list
(front + back flat-lay / ghost-mannequin shot) or the brand CDN. They are copyrighted retailer product photos,
used for the local demo only and NOT for redistribution; every URL is recorded in SOURCES.md.

Steps (default: all):
  fetch   download each ITEMS photo (front + back) into assets/garments/_raw_streetwear/ with a browser
          User-Agent and record page URL, image URLs and the usage note in _raw_streetwear/sources.json.
          A shop answering 403/503 is logged and skipped (earlier downloads are kept).
  build   cut each photo out (transparent PNG: its own alpha; otherwise the fetch_garments near-white /
          border-colour threshold -> grabCut -> feather), lay tops out with fetch_garments.layout_top +
          torso_region and bottoms with layout_bottom + fetch_garments_bottoms.rectify_bottom (legs straight
          and vertical, waist on top, hems at the bottom), then write each PAIRS outfit with
          prepare_garments.write_outfit (colour boost, real back photos, sleeve tile, thumbnail, meta.yaml).
  sheet   out/streetwear_sheet.jpg: thumb | top_front | top_back | bottom_front | bottom_back per outfit.
Fetch/build also append (or refresh only) the "Streetwear" section of assets/garments/SOURCES.md.

    .venv/bin/python scripts/fetch_garments_streetwear.py            # fetch + build + sheet
    .venv/bin/python scripts/fetch_garments_streetwear.py build      # rebuild offline from _raw_streetwear/
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fitcast.garments import composite, garment_cfg, load_outfit  # noqa: E402
from fetch_garments import (BG_DIST, TORSO_HEM_TOL, TORSO_MIN_FRAC, WORK_MAX_PX, _disk, _largest_filled,  # noqa: E402
                            flatten_chroma, garment_rgba, layout_bottom, layout_top, print_mask, torso_region)
from fetch_garments_bottoms import rectify_bottom  # noqa: E402
from prepare_garments import _hex, _median_color, estimate_body_region, load_config, write_outfit  # noqa: E402

log = logging.getLogger("fitcast.fetch_garments_streetwear")

RAW_SUBDIR = "_raw_streetwear"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0 Safari/537.36")
FETCH_WIDTH = 1600          # requested image width (Shopify `width=` / Amplience `w=`)
RETRIES = 3
REQUEST_GAP_S = 1.2         # politeness gap between requests
ALPHA_CUT = 127             # transparent-PNG photos: alpha above this is garment
SOURCES_HEADING = "## Streetwear (scripts/fetch_garments_streetwear.py)"
NOTE = "retailer product photo, demo use only, not for redistribution"
SHEET_ROW_H = 320
SHEET_GAP = 8

STUSSY, FOG, JR = "https://www.stussy.com", "https://fearofgod.com", "https://justinreed.com"
CWIP = "https://cdn.media.amplience.net/i/carhartt_wip"

# key -> photo source + cut-out hints (hint names as in fetch_garments.MANIFEST: white, bg_dist, crop, gamma,
# open_frac, smooth, print_dist, flat_chroma) plus `region`: (x0, y0, x1, y1) or "hem_rows" (see _region).
# Flat-laid hoodies get a hand-measured region: their sleeves lie along the body down to the hem, so no
# alpha-profile estimate separates sleeve from torso.
#   shopify store: `store` + `handle`, `front` / `back` = index into the product's image list
#   direct: `page` + `front_url` / `back_url`
ITEMS: dict[str, dict] = {
    # ---------------------------------------------------------------- tops
    "stussy_tee_khaki": dict(kind="top", brand="Stussy", store=STUSSY, handle="1905000-basic-stussy-tee-khaki",
                             front=0, back=1, desc="Stussy Basic Stussy Tee, khaki (back: big script logo)"),
    "stussy_8ball_tee": dict(kind="top", brand="Stussy", store=STUSSY, handle="1905022-8-ball-tee-natural",
                             front=0, back=1, white=250, bg_dist=6,
                             desc="Stussy 8 Ball Tee, natural (back: 8-ball graphic)"),
    "stussy_hoodie_ash": dict(kind="top", brand="Stussy", store=STUSSY, handle="1925000-basic-stussy-hoodie-ash-heather",
                              front=0, back=1, white=250, bg_dist=6, region=(0.20, 0.01, 0.82, 0.90),
                              desc="Stussy Basic Stussy Hoodie, ash heather (back: big script logo)"),
    "fog_hoodie": dict(kind="top", brand="Fear of God ESSENTIALS", store=FOG, handle="classic-hoodie-rw01501-oatmeal-heather",
                       front=2, back=3, white=250, bg_dist=5, region=(0.20, 0.04, 0.80, 0.87),
                       desc="Fear of God ESSENTIALS Classic Hoodie, oatmeal heather"),
    "fog_tee": dict(kind="top", brand="Fear of God ESSENTIALS", store=FOG, handle="classic-short-sleeve-tee-rw01496-warm-heather",
                    front=0, back=1, white=250, bg_dist=5,
                    desc="Fear of God ESSENTIALS Classic Tee, warm heather"),
    "nike_tech_fleece": dict(kind="top", brand="Nike", store="https://www.feature.com",
                             handle="nike-sportswear-tech-fleece-full-zip-up-hoodie-signal-blue-black", front=0, back=1,
                             desc="Nike Sportswear Tech Fleece Full-Zip Hoodie, signal blue / black (CU4489-403)"),
    "lakers_jersey": dict(kind="top", brand="Nike / Los Angeles Lakers", store="https://lakersstore.com",
                          handle="doncic-icon-swingman-jersey", front=0, back=1, region="hem_rows",
                          desc="Los Angeles Lakers Icon Swingman Jersey (Doncic #77), gold"),
    "ch_hoodie": dict(kind="top", brand="Chrome Hearts", store=JR, handle="chrome-hearts-vertical-logo-hoodie-54684684684",
                      front=0, back=1, region=(0.16, 0.06, 0.86, 0.85), desc="Chrome Hearts Vertical Logo Hoodie, orange (resale listing)"),
    # ---------------------------------------------------------------- bottoms
    "fog_sweatpant": dict(kind="bottom", brand="Fear of God ESSENTIALS", store=FOG,
                          handle="classic-sweatpant-rw01503-warm-heather", front=0, back=1,
                          desc="Fear of God ESSENTIALS Classic Sweatpant, warm heather"),
    "stussy_bigol_jean": dict(kind="bottom", brand="Stussy", store=STUSSY, handle="116761-big-ol-jean-denim-light-wash-indigo",
                              front=6, back=7, desc="Stussy Big Ol' Jean, light wash indigo (baggy denim)"),
    "stussy_fleece_pant": dict(kind="bottom", brand="Stussy", store=STUSSY, handle="116812-fleece-pant-grey-heather",
                               front=4, back=5, white=250, bg_dist=6, desc="Stussy Fleece Pant, grey heather"),
    "stussy_track_pant": dict(kind="bottom", brand="Stussy", store=STUSSY, handle="116785-side-stripe-track-pant-olive",
                              front=5, back=6, white=245, bg_dist=10,
                              desc="Stussy Side Stripe Track Pant, olive (cream side stripes)"),
    "carhartt_cargo": dict(kind="bottom", brand="Carhartt WIP",
                           page="https://www.carhartt-wip.com/en-de/p/og-cargo-pant-wip-dusty-h-stone-washed-46",
                           front_url=f"{CWIP}/I037585_3TF_06-ST-02", back_url=f"{CWIP}/I037585_3TF_06-ST-01", open_frac=0.012,
                           desc="Carhartt WIP OG Cargo Pant (I037585), dusty H brown, stone washed"),
    "ch_cross_jeans": dict(kind="bottom", brand="Chrome Hearts", store=JR,
                           handle="chrome-hearts-levis-517-orange-tab-cross-patch-jeans-58962478541845448", front=0, back=1,
                           desc="Chrome Hearts x Levi's 517 Orange Tab Cross Patch Jeans (pink leather crosses, resale)"),
}

# Outfit folders written by `build`: (id, name, sleeve, top key, bottom key)
PAIRS: list[tuple[str, str, str, str, str]] = [
    ("40_essentials_hoodie_sweatpants", "Essentials Hoodie + Essentials Sweatpants", "long", "fog_hoodie", "fog_sweatpant"),
    ("41_stussy_tee_baggy_jeans", "Stussy Tee + Stussy Big Ol' Jeans", "short", "stussy_tee_khaki", "stussy_bigol_jean"),
    ("42_tech_fleece_carhartt_cargo", "Nike Tech Fleece + Carhartt WIP Cargo", "long", "nike_tech_fleece", "carhartt_cargo"),
    ("43_lakers_jersey_baggy_jeans", "Lakers Jersey + Baggy Jeans", "none", "lakers_jersey", "stussy_bigol_jean"),
    ("44_chrome_hearts_hoodie_cross_jeans", "Chrome Hearts Hoodie + Cross Patch Jeans", "long", "ch_hoodie", "ch_cross_jeans"),
    ("45_stussy_hoodie_fleece_pant", "Stussy Hoodie + Stussy Fleece Pant", "long", "stussy_hoodie_ash", "stussy_fleece_pant"),
    ("46_essentials_tee_carhartt_cargo", "Essentials Tee + Carhartt WIP Cargo", "short", "fog_tee", "carhartt_cargo"),
    ("47_stussy_8ball_cross_jeans", "Stussy 8 Ball Tee + Chrome Hearts Jeans", "short", "stussy_8ball_tee", "ch_cross_jeans"),
    ("48_stussy_hoodie_track_pant", "Stussy Hoodie + Side Stripe Track Pant", "long", "stussy_hoodie_ash", "stussy_track_pant"),
    ("49_essentials_tee_baggy_jeans", "Essentials Tee + Stussy Big Ol' Jeans", "short", "fog_tee", "stussy_bigol_jean"),
    ("50_tech_fleece_essentials_sweats", "Nike Tech Fleece + Essentials Sweatpants", "long", "nike_tech_fleece", "fog_sweatpant"),
    ("51_stussy_tee_carhartt_cargo", "Stussy Tee + Carhartt WIP Cargo", "short", "stussy_tee_khaki", "carhartt_cargo"),
]


# ----------------------------------------------------------------------------- fetch

def _get(url: str, timeout: float = 60.0) -> bytes:
    """GET with a browser User-Agent; retries transient errors, gives up at once on 403/404/503."""
    err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (403, 404, 503):
                break
        except OSError as e:
            err = e
        time.sleep(REQUEST_GAP_S * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {err}")


def _with_width(url: str) -> str:
    """Ask the CDN for a FETCH_WIDTH-wide rendition (Shopify `width`, Amplience `w`)."""
    key = "w" if "amplience" in url else "width"
    return f"{url}{'&' if '?' in url else '?'}{key}={FETCH_WIDTH}"


def image_urls(m: dict) -> tuple[str, str, str | None, str]:
    """(page URL, front image URL, back image URL or None, product title) for an ITEMS entry."""
    if "store" in m:
        page = f"{m['store']}/products/{m['handle']}"
        prod = json.loads(_get(page + ".json"))["product"]
        imgs = [i["src"] for i in prod["images"]]
        back = imgs[m["back"]] if m.get("back") is not None and m["back"] < len(imgs) else None
        return page, imgs[m["front"]], back, prod["title"]
    return m["page"], m["front_url"], m.get("back_url"), m["desc"]


def _save(raw_dir: Path, stem: str, data: bytes) -> Path:
    """Write image bytes as <stem>.png / .jpg by magic number; removes the other extension."""
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    for old in raw_dir.glob(f"{stem}.*"):
        old.unlink()
    path = raw_dir / f"{stem}{ext}"
    path.write_bytes(data)
    if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
        path.unlink()
        raise RuntimeError(f"{stem}: downloaded file is not a readable image")
    return path


def fetch(raw_dir: Path) -> dict[str, dict]:
    """Download every ITEMS front/back photo into `raw_dir`; merge records into sources.json and return them."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
    for key, m in ITEMS.items():
        try:
            page, front, back, title = image_urls(m)
            time.sleep(REQUEST_GAP_S)
            files = [_save(raw_dir, key, _get(_with_width(front))).name]
            if back:
                time.sleep(REQUEST_GAP_S)
                files.append(_save(raw_dir, f"{key}_back", _get(_with_width(back))).name)
        except (RuntimeError, KeyError, IndexError, ValueError) as e:
            log.error("%s: %s; keeping any earlier copy", key, e)
            continue
        h, w = cv2.imread(str(raw_dir / files[0]), cv2.IMREAD_UNCHANGED).shape[:2]
        if h < 600:
            log.warning("%s: only %d px tall", key, h)
        sources[key] = dict(brand=m["brand"], title=title, desc=m["desc"], page_url=page, front_url=front,
                            back_url=back, files=files, size=[w, h], note=NOTE)
        log.info("fetched %-20s %dx%d %s%s", key, w, h, page, " (+back)" if back else "")
    src_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    return sources


# ----------------------------------------------------------------------------- cut-out + layout

def _raw_file(raw_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg"):
        if (raw_dir / f"{stem}{ext}").is_file():
            return raw_dir / f"{stem}{ext}"
    return None


def _alpha_rgba(img: np.ndarray, m: dict) -> np.ndarray:
    """Cut-out of a transparent PNG: the largest opaque blob of its own alpha, tightly cropped (hard alpha)."""
    h, w = img.shape[:2]
    s = min(1.0, WORK_MAX_PX / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    bgr = np.ascontiguousarray(img[..., :3])
    mask = np.where(img[..., 3] > ALPHA_CUT, 255, 0).astype(np.uint8)
    if m.get("open_frac"):
        _, _, bw, bh = cv2.boundingRect(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disk(m["open_frac"] * max(bw, bh)))
    mask = _largest_filled(mask) & mask      # keep holes (e.g. between straps) transparent
    if cv2.countNonZero(mask) == 0:
        raise ValueError("transparent photo has no opaque garment")
    if float(m.get("gamma", 1.0)) != 1.0:
        lut = (255.0 * (np.arange(256) / 255.0) ** float(m["gamma"])).round().astype(np.uint8)
        bgr = cv2.LUT(bgr, lut)
    x, y, bw, bh = cv2.boundingRect(mask)
    return np.dstack([bgr[y:y + bh, x:x + bw], mask[y:y + bh, x:x + bw]])


def photo_rgba(raw_dir: Path, stem: str, m: dict, cfg: dict) -> np.ndarray:
    """Tightly cropped RGBA garment (hard alpha) from a raw download."""
    path = _raw_file(raw_dir, stem)
    if path is None:
        raise FileNotFoundError(f"{raw_dir / stem}.(png|jpg) missing; run `scripts/fetch_garments_streetwear.py fetch`")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4 and int(img[..., 3].min()) <= ALPHA_CUT:
        if m.get("crop") is not None:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = m["crop"]
            img = img[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
        return _alpha_rgba(img, m)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return garment_rgba(np.ascontiguousarray(img[..., :3]), int(m.get("white", cfg["white_bg_threshold"])),
                        m.get("crop"), float(m.get("bg_dist", BG_DIST)), float(m.get("gamma", 1.0)),
                        float(m.get("open_frac", 0.0)), bool(m.get("smooth", False)))


def torso_box(top: np.ndarray, cfg: dict) -> tuple[float, float, float, float]:
    """Normalized torso box of a top canvas, measured from the hem at the CENTRE column.

    fetch_garments.torso_region measures from the lowest garment row, which on a flat-laid hoodie is a sleeve
    cuff hanging below the hem (it then returns a sleeve). Here the hem is the bottom of the centre column and
    the torso is the contiguous column run around the centre whose alpha ends within TORSO_HEM_TOL of it.
    Falls back to torso_region when that run is implausibly narrow."""
    a = top[..., 3] > 127
    h, w = a.shape
    rows, cols = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    has = a.any(axis=0)
    bottom = np.where(has, h - 1 - np.argmax(a[::-1], axis=0), -1)
    cx = int(round(0.5 * (cols[0] + cols[-1])))
    if not has[cx]:
        cx = int(cols[np.argmin(np.abs(cols - cx))])
    hem = int(bottom[cx])
    torso = np.abs(bottom - hem) <= TORSO_HEM_TOL * (y1 - y0)      # cuffs hanging below the hem are excluded
    x0 = cx
    while x0 > 0 and torso[x0 - 1]:
        x0 -= 1
    x1 = cx + 1
    while x1 < w and torso[x1]:
        x1 += 1
    if x1 - x0 < TORSO_MIN_FRAC * (cols[-1] + 1 - cols[0]):
        return torso_region(top, cfg)
    return (x0 / w, y0 / h, x1 / w, (hem + 1) / h)


def _region(m: dict, front: np.ndarray, cfg: dict) -> tuple[float, float, float, float]:
    """ITEMS `region` hint: explicit box, "hem_rows" (prepare_garments estimate, for sleeveless tops) or auto."""
    r = m.get("region")
    if r == "hem_rows":
        return tuple(float(v) for v in estimate_body_region(front, cfg))
    return tuple(float(v) for v in r) if r else torso_box(front, cfg)


def load_item(key: str, raw_dir: Path, cfg: dict) -> dict:
    """Laid-out canvases of an ITEMS key: {kind, front, back, region, detail} (region/detail for tops only)."""
    m = ITEMS[key]
    has_back = _raw_file(raw_dir, f"{key}_back") is not None
    if m["kind"] == "top":
        front = layout_top(photo_rgba(raw_dir, key, m, cfg), cfg)
        back = layout_top(photo_rgba(raw_dir, f"{key}_back", m, cfg), cfg) if has_back else None
        region = _region(m, front, cfg)
        if m.get("flat_chroma"):
            front = flatten_chroma(front, float(m["print_dist"]))
        detail = print_mask(front, region, float(m["print_dist"])) if m.get("print_dist") and back is None else None
        return dict(kind="top", front=front, back=back, region=region, detail=detail)
    front = rectify_bottom(layout_bottom(photo_rgba(raw_dir, key, m, cfg), cfg))
    # a back photo shows the person's left leg on the image left as seen from behind: used as-is
    back = rectify_bottom(layout_bottom(photo_rgba(raw_dir, f"{key}_back", m, cfg), cfg)) if has_back else None
    return dict(kind="bottom", front=front, back=back, region=None, detail=None)


def build(raw_dir: Path, out_dir: Path, cfg: dict) -> list[str]:
    """Write every PAIRS outfit folder into `out_dir`; returns write_outfit summary lines."""
    cache: dict[str, dict] = {}
    lines = []
    for oid, name, sleeve, top_key, bottom_key in PAIRS:
        try:
            for k in (top_key, bottom_key):
                if k not in cache:
                    cache[k] = load_item(k, raw_dir, cfg)
        except (OSError, ValueError, cv2.error) as e:
            log.error("skipping %s: %s", oid, e)
            continue
        top, bottom = cache[top_key], cache[bottom_key]
        if (top["kind"], bottom["kind"]) != ("top", "bottom"):
            log.error("skipping %s: %s is a %s, %s a %s", oid, top_key, top["kind"], bottom_key, bottom["kind"])
            continue
        meta = dict(name=name, category="outfit", sleeve=sleeve, size_table="mens",
                    colors=[_hex(_median_color(top["front"], cfg)), _hex(_median_color(bottom["front"], cfg))])
        lines.append(write_outfit(out_dir, oid, meta, cfg, top=top["front"], bottom=bottom["front"],
                                  top_back=top["back"], bottom_back=bottom["back"], top_detail=top["detail"],
                                  region=top["region"]))
    return lines


# ----------------------------------------------------------------------------- sources + sheet

def sources_section(raw_dir: Path) -> str:
    """Markdown section listing every product photo used by PAIRS."""
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
    lines = [SOURCES_HEADING, "",
             "No free-licence photos of these products exist on Wikimedia Commons / Openverse, so these are the",
             "brand's or a retailer's own product photos (front + back flat-lay / ghost mannequin), downloaded by",
             "`scripts/fetch_garments_streetwear.py`, cut out, laid out and colour-boosted. **Copyrighted: "
             f"{NOTE}.** Raw downloads: `_raw_streetwear/`.", "",
             "| garment | outfits | product page (shop) | front image | back image | note |", "|---|---|---|---|---|---|"]
    for key, s in sources.items():
        outfits = ", ".join(oid for oid, _, _, t, b in PAIRS if key in (t, b))
        if key not in ITEMS or not outfits:
            continue
        back = f"[back]({s['back_url']})" if s.get("back_url") else "synthesized"
        host = s["page_url"].split("/")[2].removeprefix("www.")
        lines.append(f"| {ITEMS[key]['desc'].replace('|', '/')} | {outfits} | [{host}]({s['page_url']}) | "
                     f"[front]({s['front_url']}) | {back} | {s['note']} |")
    return "\n".join(lines) + "\n"


def update_sources_md(path: Path, raw_dir: Path) -> None:
    """Append our section to SOURCES.md, or replace only our own section if it is already there."""
    text = path.read_text() if path.is_file() else "# Garment sources\n"
    section = sources_section(raw_dir)
    pat = re.compile(rf"^{re.escape(SOURCES_HEADING)}\n.*?(?=^## |\Z)", re.S | re.M)
    if pat.search(text):
        text = pat.sub(lambda _: section + "\n", text, count=1).rstrip("\n") + "\n"
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text)


def contact_sheet(out_dir: Path, cfg: dict, path: Path) -> Path:
    """One row per built PAIRS outfit: thumb | top_front | top_back | bottom_front | bottom_back."""
    c = garment_cfg(cfg)
    rows = []
    for oid, name, _, _, _ in PAIRS:
        if not (out_dir / oid).is_dir():
            continue
        o = load_outfit(out_dir / oid, cfg)
        cells = []
        for img in (o.thumb, o.top_front, o.top_back, o.bottom_front, o.bottom_back):
            if img is None:
                continue
            bgr = composite(img, c["thumb_bg"]) if img.shape[2] == 4 else img
            s = SHEET_ROW_H / bgr.shape[0]
            cells.append(cv2.resize(bgr, (round(bgr.shape[1] * s), SHEET_ROW_H), interpolation=cv2.INTER_AREA))
            cells.append(np.full((SHEET_ROW_H, SHEET_GAP, 3), 255, np.uint8))
        row = np.hstack(cells)
        cv2.putText(row, f"{oid}  {name}", (8, SHEET_ROW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no streetwear outfits built yet")
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, SHEET_GAP), (0, width - r.shape[1]), (0, 0))) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=("all", "fetch", "build", "sheet"), default="all")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, help="garments dir (default: config garments.dir)")
    ap.add_argument("--sheet", type=Path, default=ROOT / "out" / "streetwear_sheet.jpg")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = args.out or Path(garment_cfg(cfg)["dir"])
    if not out.is_absolute():
        out = ROOT / out
    raw = out / RAW_SUBDIR
    if args.step in ("all", "fetch"):
        fetch(raw)
    ok = True
    if args.step in ("all", "build"):
        lines = build(raw, out, cfg)
        for line in lines:
            print(line)
        print(f"{len(lines)} outfits -> {out}")
        ok = len(lines) == len(PAIRS)
    if args.step in ("all", "fetch", "build"):
        update_sources_md(out / "SOURCES.md", raw)
    if args.step in ("all", "build", "sheet"):
        print(f"contact sheet -> {contact_sheet(out, cfg, args.sheet)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
