#!/usr/bin/env python
"""Fetch free-licensed real-photo TOPS and build them into FitCast outfits 24_* .. 31_*.

Companion to scripts/fetch_garments.py (whose cut-out, layout and torso-region helpers are reused).

Steps (default: all):
  fetch   download every TOPS photo into assets/garments/_raw_tops/ (Wikimedia Commons thumbnails via the
          API, or a direct URL with a hand-recorded licence for Flickr / Smithsonian photos) and record
          source, author and licence in _raw_tops/sources.json. Honors HTTP 429 Retry-After.
  build   cut each photo out (near-white / border-colour threshold -> grabCut -> feather), fit it to
          `garments.top_size`, estimate `top_body_region`, then write the outfit folder with
          prepare_garments.write_outfit (colour boost, back view: real photo when there is one, otherwise
          synthesized with the chest print removed; sleeve tile, thumbnail, meta.yaml). The bottom is
          COPIED unchanged (already boosted PNGs) from the existing outfit folder named in PAIRS, and
          meta.yaml gets the bottom parts and colour added. Also appends/refreshes the "Tops" section of
          assets/garments/SOURCES.md (nothing else in that file is touched) and writes out/tops_sheet.jpg
          (thumb | top_front | top_back per outfit, torso region drawn in green).

    .venv/bin/python scripts/fetch_garments_tops.py            # fetch + build
    .venv/bin/python scripts/fetch_garments_tops.py build      # rebuild offline from _raw_tops/
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fitcast.garments import composite, garment_cfg  # noqa: E402
from fetch_garments import (ALLOWED_LICENSES, API, BG_DIST, SMOOTH_D, SMOOTH_SIGMA_COLOR,  # noqa: E402
                            SMOOTH_SIGMA_SPACE, USER_AGENT, WORK_MAX_PX, cutout, flatten_chroma, garment_rgba,
                            layout_top, print_mask, torso_region)
from prepare_garments import _hex, _median_color, load_config, write_outfit  # noqa: E402

log = logging.getLogger("fitcast.fetch_garments_tops")

RAW_DIRNAME = "_raw_tops"
FETCH_WIDTH = 1600          # Commons thumbnail width requested
RETRIES = 5
RETRY_WAIT_S = 5.0          # wait per attempt when a 429 carries no Retry-After
REQUEST_GAP_S = 1.0         # politeness gap between downloads (Commons rate-limits shared IPs)
SOURCES_HEADING = "## Tops (scripts/fetch_garments_tops.py)"
SHEET_PATH = ROOT / "out" / "tops_sheet.jpg"
SHEET_ROW_H = 320
SHEET_BG = (90, 90, 90)
REGION_COLOR = (0, 255, 0)

MOHIST = "Missouri Historical Society"
SHEIE = "Robert Sheie"
OWN_LICENSES = ALLOWED_LICENSES + ("CC BY 2.0 (Flickr)",)

# key -> photo + cut-out hints (same keys as fetch_garments.MANIFEST; see the comment block there), plus:
#   title / back_title: Commons file of the front / back view
#   url / back_url + page_url, license, license_url, author: direct download with a hand-recorded licence
#   region: manual top_body_region override
TOPS: dict[str, dict] = {
    "jersey_lakers": dict(
        url="https://live.staticflickr.com/7390/15873625204_42534ace80_b.jpg",
        page_url="https://www.flickr.com/photos/127365614@N08/15873625204",
        title="ballcap hat Lakers jersey basketball pink roses poms game tickets shoes alstroemeria lilies",
        license="CC BY 2.0", license_url="https://creativecommons.org/licenses/by/2.0/", author="ProFlowers.com",
        crop=(0.215, 0.355, 0.795, 0.965), white=246, bg_dist=0, smooth=True, print_dist=45,
        print_box=(0.15, 0.08, 0.85, 0.66),
        desc="white Los Angeles Lakers #24 (Kobe Bryant) adidas swingman jersey, purple/gold trim (flat-lay; "
             "cropped out of a styled flat-lay with cap, shoes, ball and flowers)"),
    "tee_welders": dict(
        title='File:Pink "Welders" T-Shirt Worn by the Welders - DPLA - b0a00013064a4280e484f55149cb7174 (page 1).jpg',
        back_title='File:Pink "Welders" T-Shirt Worn by the Welders - DPLA - b0a00013064a4280e484f55149cb7174 (page 4).jpg',
        attribution=MOHIST, bg_dist=45,
        desc='hot-pink "the Welders" band tee (St. Louis punk band), front + back photo'),
    "tee_coral_court": dict(
        title="File:Coral Court Motel T-Shirt - DPLA - cd249cf4a00e22c6e6821cb73633bb23 (page 4).jpg",
        attribution=MOHIST, print_dist=40,
        desc="coral-pink Coral Court Motel souvenir tee with black screen print"),
    "tee_wayne": dict(
        url="https://ids.si.edu/ids/deliveryService?id=NMAAHC-2016_140_1_1_001&max=2000",
        back_url="https://ids.si.edu/ids/deliveryService?id=NMAAHC-2016_140_1_1_002&max=2000",
        page_url="https://nmaahc.si.edu/object/nmaahc_2016.140.1.1",
        title="Basketball jersey for Lockland Wayne High School (NMAAHC 2016.140.1.1)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Smithsonian National Museum of African American History and Culture (maker: P. Goldsmith & Sons)",
        bg_dist=45, print_dist=40,
        desc='royal-blue mesh "WAYNE 14" vintage basketball shirt with gold striped shoulders, front + back photo'),
    "sweatshirt_lenders": dict(
        title="File:Lender's Sweatshirt (15844939901).jpg", print_dist=40,
        desc="red fleece crewneck sweatshirt with white Lender's Bagels logo (sleeves folded across the front)"),
    "flannel_buffalo": dict(
        title="File:Polo Ralph Lauren winter jacket, red and black plaid.jpg", attribution=SHEIE, open_frac=0.012,
        erase=[[(0.413, 0.0), (0.562, 0.0), (0.556, 0.084), (0.548, 0.147), (0.531, 0.191), (0.500, 0.242),
                (0.482, 0.231), (0.447, 0.182), (0.420, 0.142), (0.411, 0.084)]],
        desc="red and black buffalo-check flannel shirt jacket"),
    "jacket_denim": dict(
        title="File:Polo Ralph Lauren denim jacket.jpg", attribution=SHEIE, open_frac=0.012,
        erase=[[(0.447, 0.067), (0.580, 0.067), (0.582, 0.129), (0.562, 0.147), (0.547, 0.187), (0.533, 0.233),
                (0.509, 0.209), (0.487, 0.182), (0.469, 0.164), (0.453, 0.124)]],
        desc="mid-blue denim trucker jacket with corduroy collar"),
    "polo_freddies": dict(
        title="File:Freddie's Polo Shirt - DPLA - 74c5a614754328992dc0c7ecf5102d2c (page 4).jpg",
        back_title="File:Freddie's Polo Shirt - DPLA - 74c5a614754328992dc0c7ecf5102d2c (page 6).jpg",
        attribution=MOHIST, print_dist=40,
        desc="burnt-orange pique polo shirt with small embroidered logo, front + back photo"),
}

# (outfit id, name, sleeve, top key, existing outfit folder whose bottom PNGs are copied)
PAIRS: list[tuple[str, str, str, str, str]] = [
    ("24_lakers_jersey_jeans", "Lakers Jersey + Baggy Jeans", "none", "jersey_lakers", "20_jersey_baggy_jeans"),
    ("25_band_tee_flares", "Welders Band Tee + Flared Jeans", "short", "tee_welders", "14_oversized_tee_flares"),
    ("26_motel_tee_cargo", "Coral Court Motel Tee + Cargo Pants", "short", "tee_coral_court", "17_olive_overshirt_cargo"),
    ("27_wayne_tee_trackpants", "Wayne 14 Athletic Tee + Track Pants", "short", "tee_wayne", "15_green_tee_trackpants"),
    ("28_crewneck_sweatpants", "Red Logo Crewneck + Sweatpants", "long", "sweatshirt_lenders", "19_hoodie_sweatpants"),
    ("29_flannel_grey_jeans", "Buffalo Plaid Flannel + Grey Jeans", "long", "flannel_buffalo", "16_madras_grey_jeans"),
    ("30_denim_jacket_khakis", "Denim Jacket + Khaki Trousers", "long", "jacket_denim", "18_sweater_khakis"),
    ("31_polo_chinos", "Orange Polo + Chinos", "short", "polo_freddies", "22_scout_overshirt_chinos"),
]
BOTTOM_PARTS = ("bottom_front", "bottom_back")


# ----------------------------------------------------------------------------- fetch

def _get(url: str, timeout: float = 60.0) -> bytes:
    """GET with the Wikimedia-style User-Agent; retries, sleeping Retry-After on HTTP 429."""
    err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            time.sleep(REQUEST_GAP_S)
            return data
        except urllib.error.HTTPError as e:
            err = e
            if e.code not in (429, 500, 502, 503, 504):
                break
            ra = e.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else RETRY_WAIT_S * (attempt + 1)
            log.info("HTTP %d, waiting %.0f s: %s", e.code, wait, url[:90])
            time.sleep(wait)
        except OSError as e:
            err = e
            time.sleep(RETRY_WAIT_S * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {err}")


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_info(titles: list[str]) -> dict[str, dict]:
    """title -> {url, page_url, width, height, license, license_url, author} from the Commons API."""
    params = dict(action="query", titles="|".join(titles), prop="imageinfo", format="json",
                  iiprop="url|size|extmetadata", iiurlwidth=str(FETCH_WIDTH))
    data = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    norm = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
    out = {}
    for pg in data.get("query", {}).get("pages", {}).values():
        title = norm.get(pg["title"], pg["title"])
        if "imageinfo" not in pg:
            log.error("%s: not found on Commons", title)
            continue
        ii, md = pg["imageinfo"][0], pg["imageinfo"][0].get("extmetadata", {})
        out[title] = dict(url=ii.get("thumburl") or ii["url"], page_url=ii["descriptionurl"],
                          width=ii["width"], height=ii["height"],
                          license=md.get("LicenseShortName", {}).get("value", "unknown"),
                          license_url=md.get("LicenseUrl", {}).get("value", ""),
                          author=_strip_html(md.get("Artist", {}).get("value", ""))
                          or _strip_html(md.get("Credit", {}).get("value", "")))
    return out


def _photo_jobs() -> list[tuple[str, str, dict]]:
    """(raw file stem, top key, record) for every front/back photo; Commons records carry `title` only."""
    jobs = []
    for key, m in TOPS.items():
        if m.get("url"):
            meta = {f: m[f] for f in ("page_url", "title", "license", "license_url", "author")}
            jobs.append((key, key, dict(meta, url=m["url"])))
            if m.get("back_url"):
                jobs.append((f"{key}_back", key, dict(meta, url=m["back_url"])))
        else:
            jobs.append((key, key, dict(title=m["title"])))
            if m.get("back_title"):
                jobs.append((f"{key}_back", key, dict(title=m["back_title"])))
    return jobs


def fetch(raw_dir: Path) -> dict[str, dict]:
    """Download every TOPS photo into `raw_dir`, write sources.json; returns stem -> source record."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = _photo_jobs()
    info = commons_info([r["title"] for _, _, r in jobs if "url" not in r])
    old_path = raw_dir / "sources.json"
    sources = json.loads(old_path.read_text()) if old_path.is_file() else {}
    for stem, key, rec in jobs:
        if "url" not in rec:
            if rec["title"] not in info:
                continue
            rec = dict(info[rec["title"]], title=rec["title"])
        if not rec["license"].startswith(OWN_LICENSES):
            log.error("%s: license %r is not allowed; skipped", stem, rec["license"])
            continue
        path = raw_dir / f"{stem}.jpg"
        try:
            path.write_bytes(_get(rec["url"]))
        except RuntimeError as e:
            log.error("%s: %s; keeping any earlier copy", stem, e)
            continue
        img = cv2.imread(str(path))
        if img is None:
            path.unlink()
            log.error("%s: downloaded file is not a readable image", stem)
            continue
        if img.shape[0] < 600:
            log.warning("%s: only %d px tall", stem, img.shape[0])
        sources[stem] = dict(rec, file=path.name, key=key, desc=TOPS[key]["desc"])
        log.info("fetched %-22s %s (%s)", stem, rec["title"][:70], rec["license"])
    old_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    return sources


# ----------------------------------------------------------------------------- build

def cut_photo(bgr: np.ndarray, m: dict, cfg: dict) -> np.ndarray:
    """fetch_garments.garment_rgba plus `erase`: polygons (normalized full-photo coords) forced transparent
    after the cut-out, for mannequin neck forms the garment's collar encloses. Returns the tight RGBA crop."""
    if not m.get("erase"):
        return garment_rgba(bgr, int(m.get("white", cfg["white_bg_threshold"])), m.get("crop"),
                            float(m.get("bg_dist", BG_DIST)), float(m.get("gamma", 1.0)),
                            float(m.get("open_frac", 0.0)), bool(m.get("smooth", False)))
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = m.get("crop") or (0.0, 0.0, 1.0, 1.0)
    bgr = bgr[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
    s = min(1.0, WORK_MAX_PX / max(bgr.shape[:2]))
    if s < 1:
        bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    mask = cutout(bgr, int(m.get("white", cfg["white_bg_threshold"])), float(m.get("bg_dist", BG_DIST)),
                  float(m.get("open_frac", 0.0)))
    for poly in m["erase"]:
        pts = np.array([[(px * w - round(x0 * w)) * s, (py * h - round(y0 * h)) * s] for px, py in poly])
        cv2.fillPoly(mask, [pts.round().astype(np.int32)], 0)
    mask = _largest_blob(mask)
    if m.get("smooth"):
        bgr = cv2.bilateralFilter(bgr, SMOOTH_D, SMOOTH_SIGMA_COLOR, SMOOTH_SIGMA_SPACE)
    if float(m.get("gamma", 1.0)) != 1.0:
        lut = (255.0 * (np.arange(256) / 255.0) ** float(m["gamma"])).round().astype(np.uint8)
        bgr = cv2.LUT(bgr, lut)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return np.dstack([bgr[by:by + bh, bx:bx + bw], mask[by:by + bh, bx:bx + bw]])


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (holes stay open, unlike fetch_garments._largest_filled)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def _cut(raw_dir: Path, stem: str, m: dict, cfg: dict) -> np.ndarray:
    bgr = cv2.imread(str(raw_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"{raw_dir / (stem + '.jpg')} missing; run `scripts/fetch_garments_tops.py fetch`")
    hints = dict(m, **m.get("back_hints", {})) if stem.endswith("_back") else m
    return layout_top(cut_photo(bgr, hints, cfg), cfg)


def load_top(key: str, raw_dir: Path, cfg: dict):
    """(front canvas, front-only print mask or None, top_body_region, back canvas or None) for a TOPS key.

    A real back photo is used when there is one; otherwise the print mask (torso pixels far from the
    fabric colour, limited to `print_box` when given) tells synthesize_back what to replace with fabric."""
    m = TOPS[key]
    front = _cut(raw_dir, key, m, cfg)
    has_back = (raw_dir / f"{key}_back.jpg").is_file() and (m.get("back_title") or m.get("back_url"))
    back = _cut(raw_dir, f"{key}_back", m, cfg) if has_back else None
    region = tuple(m.get("region") or torso_region(front, cfg))
    if m.get("flat_chroma"):
        front = flatten_chroma(front, float(m["print_dist"]))
    detail = None
    if m.get("print_dist") and back is None:
        detail = print_mask(front, tuple(m.get("print_box") or region), float(m["print_dist"]))
    return front, detail, region, back


def copy_bottom(src: Path, dst: Path, cfg: dict) -> str:
    """Copy the bottom PNGs of outfit folder `src` into `dst`, add them to dst/meta.yaml; returns bottom colour hex."""
    meta_path = dst / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    colour = ""
    for part in BOTTOM_PARTS:
        f = src / f"{part}.png"
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing; build the bottom outfit {src.name} first")
        shutil.copy2(f, dst / f.name)
        if part not in meta["parts"]:
            meta["parts"].append(part)
        if part == "bottom_front":
            colour = _hex(_median_color(cv2.imread(str(f), cv2.IMREAD_UNCHANGED), cfg))
    meta["colors"] = list(meta.get("colors", []))[:1] + [colour]
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return colour


def build(raw_dir: Path, out_dir: Path, cfg: dict) -> list[str]:
    """Write every PAIRS outfit folder; returns summary lines."""
    lines = []
    for oid, name, sleeve, top_key, bottom_oid in PAIRS:
        src = out_dir / bottom_oid
        if not (src / "bottom_front.png").is_file():
            log.error("skipping %s: bottom outfit %s has no bottom_front.png", oid, bottom_oid)
            continue
        try:
            front, detail, region, back = load_top(top_key, raw_dir, cfg)
        except (OSError, ValueError, cv2.error) as e:
            log.error("skipping %s: %s", oid, e)
            continue
        meta = dict(name=name, category="outfit", sleeve=sleeve, size_table="mens",
                    colors=[_hex(_median_color(front, cfg))])
        line = write_outfit(out_dir, oid, meta, cfg, top=front, top_back=back, top_detail=detail, region=region)
        copy_bottom(src, out_dir / oid, cfg)
        lines.append(f"{line} + bottom from {bottom_oid}")
    return lines


# ----------------------------------------------------------------------------- report

def write_sources_section(path: Path, raw_dir: Path) -> None:
    """Append (or refresh in place) this script's section of SOURCES.md; the rest of the file is kept verbatim."""
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
    rows = [SOURCES_HEADING, "",
            "Real product / museum photos downloaded by `scripts/fetch_garments_tops.py` (raw files in `_raw_tops/`),",
            "cut out, colour-boosted and resized; derivative PNGs keep the source license. Bottoms are copied from",
            "the outfit folder named in the script's PAIRS (sources above).", "",
            "| garment | outfits | file | author | license |", "|---|---|---|---|---|"]
    for stem, s in sources.items():
        key = s["key"]
        outfits = ", ".join(oid for oid, _, _, t, _ in PAIRS if t == key)
        if not outfits:
            continue
        desc = s["desc"] + (" - back view" if stem.endswith("_back") else "")
        lic = f"[{s['license']}]({s['license_url']})" if s.get("license_url") else s["license"]
        author = (TOPS[key].get("attribution") or s.get("author") or "unknown").replace("|", "/")
        name = s["title"][5:] if s["title"].startswith("File:") else s["title"]
        rows.append(f"| {desc} | {outfits} | [{name}]({s['page_url']}) | {author} | {lic} |")
    section = "\n".join(rows) + "\n"
    text = path.read_text() if path.is_file() else "# Garment sources\n"
    start = text.find(SOURCES_HEADING)
    if start < 0:
        text = text.rstrip("\n") + "\n\n" + section
    else:  # refresh only our own section (up to the next "## " heading)
        nxt = text.find("\n## ", start + len(SOURCES_HEADING))
        rest = "" if nxt < 0 else "\n" + text[nxt + 1:]
        text = text[:start] + section + rest
    path.write_text(text)


def _tile(rgba: np.ndarray, region: list | None) -> np.ndarray:
    img = composite(rgba, SHEET_BG)
    if region is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = region
        cv2.rectangle(img, (round(x0 * w), round(y0 * h)), (round(x1 * w) - 1, round(y1 * h) - 1), REGION_COLOR, 2)
    s = SHEET_ROW_H / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * s), SHEET_ROW_H), interpolation=cv2.INTER_AREA)


def contact_sheet(out_dir: Path, path: Path = SHEET_PATH) -> Path:
    """thumb | top_front (torso region boxed) | top_back for every PAIRS outfit, one row each."""
    rows = []
    for oid, *_ in PAIRS:
        f = out_dir / oid
        if not (f / "meta.yaml").is_file():
            continue
        meta = yaml.safe_load((f / "meta.yaml").read_text())
        thumb = cv2.imread(str(f / "thumb.png"))
        thumb = cv2.resize(thumb, (SHEET_ROW_H, SHEET_ROW_H), interpolation=cv2.INTER_NEAREST)
        front = _tile(cv2.imread(str(f / "top_front.png"), cv2.IMREAD_UNCHANGED), meta["top_body_region"])
        back = _tile(cv2.imread(str(f / "top_back.png"), cv2.IMREAD_UNCHANGED), None)
        row = np.hstack([thumb, front, back])
        cv2.putText(row, meta["name"], (8, SHEET_ROW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        rows.append(row)
    if not rows:
        raise RuntimeError("no outfits to put on the contact sheet")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))
    return path


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
    raw = out / RAW_DIRNAME
    if args.step in ("all", "fetch"):
        try:
            fetch(raw)
        except RuntimeError as e:
            log.error("%s; falling back to whatever is already in %s", e, raw)
    lines = []
    if args.step in ("all", "build"):
        lines = build(raw, out, cfg)
        for line in lines:
            print(line)
        print(f"{len(lines)} outfits -> {out}")
        if lines:
            print(f"contact sheet -> {contact_sheet(out)}")
    write_sources_section(out / "SOURCES.md", raw)
    return 0 if lines or args.step == "fetch" else 1


if __name__ == "__main__":
    sys.exit(main())
