#!/usr/bin/env python
"""Fetch free-licensed real-photo BOTTOMS from Wikimedia Commons and pair them with existing photo tops.

Companion of scripts/fetch_garments.py (whose cut-out and bottoms layout it reuses). Steps (default: both):
  fetch   download every MANIFEST photo into assets/garments/_raw_bottoms/ (own folder and sources.json,
          so the main fetcher's _raw/ is never touched) and append/refresh the "Bottoms" section of
          assets/garments/SOURCES.md (other sections are left as they are).
  build   cut each photo out (fetch_garments.garment_rgba: threshold -> grabCut), lay it out over
          `garments.bottom_size` (fetch_garments.layout_bottom: crotch on the centre line, waist on top,
          hems at the bottom) and write outfit folders 32_* .. 39_* with prepare_garments.write_outfit.
          The top of each outfit is file-copied (already colour-boosted) from an existing real-photo outfit
          folder (TOP_FROM), together with its sleeve type and top_body_region. The thumbnail shows the
          new bottom (several outfits share a top, so a top thumbnail would not tell them apart).

    .venv/bin/python scripts/fetch_garments_bottoms.py            # fetch + build
    .venv/bin/python scripts/fetch_garments_bottoms.py build      # rebuild offline from _raw_bottoms/
    .venv/bin/python scripts/fetch_garments_bottoms.py sheet      # contact sheet out/bottoms_sheet.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fitcast.garments import composite, garment_cfg, load_outfit  # noqa: E402
from fetch_garments import ALLOWED_LICENSES, BG_DIST, _get, commons_info, garment_rgba, layout_bottom  # noqa: E402
from prepare_garments import _hex, _median_color, load_config, write_outfit  # noqa: E402

log = logging.getLogger("fitcast.fetch_garments_bottoms")

RAW_SUBDIR = "_raw_bottoms"
DOWNLOAD_PAUSE_S = 1.5      # between downloads (Commons answers 429 to bursts)
SOURCES_HEADING = "## Bottoms (scripts/fetch_garments_bottoms.py)"
SHEET_ROW_H = 384           # contact sheet row height (px)

AM_CREDIT = "Auckland War Memorial Museum Tamaki Paenga Hira"
NM_CREDIT = "Nordiska museet"

# key -> Commons file + cut-out hints (same meaning as fetch_garments.MANIFEST):
#   crop (x0, y0, x1, y1) normalized, white threshold, bg_dist, gamma (< 1 lifts dark fabric),
#   open_frac (drops thin attachments: hanger hooks, stand poles), back_title (back-view photo of the same item),
#   attribution (credit when Commons "Artist" names the maker or is empty), note (shown in SOURCES.md).
MANIFEST: dict[str, dict] = {
    "cargo_olive_photo": dict(title="File:Cargo pants 001.jpg", back_title="File:Cargo pants 002.jpg",
                              desc="olive ripstop cargo pants with bellows pockets (front + back photo)"),
    "jeans_mid_blue": dict(title="File:Women's Pair of Blue Jeans Worn by Grace Strobel - DPLA - "
                                 "e94c33e652b48ea0513c90bc9b383e89 (page 1).jpg",
                           back_title="File:Women's Pair of Blue Jeans Worn by Grace Strobel - DPLA - "
                                      "e94c33e652b48ea0513c90bc9b383e89 (page 3).jpg",
                           desc="mid-blue straight jeans with cuffed hems (front + back photo)"),
    "jeans_light_501": dict(title="File:Spijkerbroek, merk Levi\u2019s 501, objectnr 75213(1).JPG",
                            back_title="File:Spijkerbroek, merk Levi\u2019s 501, objectnr 75213(2).JPG",
                            desc="light-wash Levi's 501 jeans with patched knees (front + back photo, on a mannequin)"),
    "trackpants_black": dict(title="File:Tracksuit bottoms.jpg",
                             desc="black tapered track pants with gold side stripes"),
    "chinos_beige": dict(title="File:Chino pants.jpg", desc="beige straight chinos"),
    "jeans_black_501": dict(title="File:Levi's 501 black.jpg", desc="black Levi's 501 jeans (back view used for both sides)"),
    "jeans_dark_dallas": dict(title="File:Jeans Gul&Bl\u00e5 i modellen Dallas.jpg", attribution=NM_CREDIT,
                              desc="dark rinse straight jeans"),
    "jeans_worn_mended": dict(title="File:Slitna jeans f\u00f6r m\u00e4n i bomull med lagade med sick-sacks\u00f6m "
                                    "p\u00e5 kn\u00e4na. 032s8YepsD7C.jpg", attribution=NM_CREDIT,
                              desc="worn mid-wash men's jeans with darned knees"),
    "cargo_black_work": dict(title="File:Zwarte werkbroek met witte bies, van Bulgaarse bouwvakker, objectnr 91220-3.JPG",
                             desc="black cotton work cargo pants with white piping"),
    "joggers_grey": dict(title="File:Unisex-Jogginghose.jpg", desc="grey cotton joggers with cuffed ankles"),
}

# Outfit folders: (id, name, top source folder, bottom key)
PAIRS: list[tuple[str, str, str, str]] = []


# ----------------------------------------------------------------------------- fetch

def fetch(raw_dir: Path) -> dict[str, dict]:
    """Download every MANIFEST photo (and back view) into `raw_dir`; write and return sources.json records."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    photos = [(k, m["title"], m) for k, m in MANIFEST.items()]
    photos += [(f"{k}_back", m["back_title"], m) for k, m in MANIFEST.items() if m.get("back_title")]
    info = commons_info([t for _, t, _ in photos])
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
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
        if not path.is_file():
            try:
                path.write_bytes(_get(rec["thumb_url"]))
            except RuntimeError as e:
                log.error("%s: %s", key, e)
                continue
            time.sleep(DOWNLOAD_PAUSE_S)
        if cv2.imread(str(path)) is None:
            path.unlink()
            log.error("%s: downloaded file is not a readable image", key)
            continue
        sources[key] = dict(rec, title=title, file=path.name, desc=m["desc"])
        log.info("fetched %-22s %s (%s)", key, title, rec["license"])
    src_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    return sources


# ----------------------------------------------------------------------------- build

def _photo(raw_dir: Path, name: str, m: dict, cfg: dict) -> np.ndarray:
    bgr = cv2.imread(str(raw_dir / f"{name}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"{raw_dir / (name + '.jpg')} missing; run `scripts/fetch_garments_bottoms.py fetch`")
    rgba = garment_rgba(bgr, int(m.get("white", cfg["white_bg_threshold"])), m.get("crop"),
                        float(m.get("bg_dist", BG_DIST)), float(m.get("gamma", 1.0)),
                        float(m.get("open_frac", 0.0)), bool(m.get("smooth", False)))
    return cv2.flip(rgba, 1) if m.get("mirror") else rgba


def load_bottom(key: str, raw_dir: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """(front canvas, back canvas or None) laid out over `bottom_size`."""
    m = MANIFEST[key]
    front = layout_bottom(_photo(raw_dir, key, m, cfg), cfg)
    back = None
    if m.get("back_title"):
        # used as-is: the renderer maps bottoms in image space (image-left half -> image-left leg) and a
        # back photo already shows the person's left leg on the image left, as seen from behind.
        back = layout_bottom(_photo(raw_dir, f"{key}_back", dict(m, **m.get("back_hints", {})), cfg), cfg)
    return front, back


def write_pair(out_dir: Path, oid: str, name: str, top_src: Path, bottom: np.ndarray,
               bottom_back: np.ndarray | None, cfg: dict) -> str:
    """Write the bottom with write_outfit, then copy the (already boosted) top parts of `top_src` in and
    merge meta.yaml (sleeve and top_body_region follow the top)."""
    top_meta = yaml.safe_load((top_src / "meta.yaml").read_text())
    meta = dict(name=name, category="outfit", sleeve=top_meta["sleeve"], size_table=top_meta.get("size_table", "mens"),
                colors=[_hex(_median_color(bottom, cfg))])
    line = write_outfit(out_dir, oid, meta, cfg, bottom=bottom, bottom_back=bottom_back)
    folder = out_dir / oid
    top_parts = [p for p in ("top_front", "top_back", "sleeve") if p in top_meta.get("parts", [])]
    for part in top_parts:
        shutil.copyfile(top_src / f"{part}.png", folder / f"{part}.png")
    out_meta = yaml.safe_load((folder / "meta.yaml").read_text())
    out_meta["parts"] = top_parts + [p for p in out_meta["parts"] if p not in top_parts]
    out_meta["top_body_region"] = top_meta["top_body_region"]
    out_meta["colors"] = list(top_meta.get("colors", [])[:1]) + out_meta["colors"]
    (folder / "meta.yaml").write_text(yaml.safe_dump(out_meta, sort_keys=False))
    return f"{line} top<-{top_src.name}"


def build(raw_dir: Path, out_dir: Path, cfg: dict) -> list[str]:
    """Write every PAIRS outfit folder; returns summary lines."""
    cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    lines = []
    for oid, name, top_folder, key in PAIRS:
        top_src = out_dir / top_folder
        try:
            if key not in cache:
                cache[key] = load_bottom(key, raw_dir, cfg)
            if not (top_src / "meta.yaml").is_file():
                raise FileNotFoundError(f"top source {top_src} has no meta.yaml")
        except (OSError, KeyError, ValueError, cv2.error) as e:
            log.error("skipping %s: %s", oid, e)
            continue
        lines.append(write_pair(out_dir, oid, name, top_src, *cache[key], cfg))
    return lines


# ----------------------------------------------------------------------------- sources + sheet

def sources_section(raw_dir: Path) -> str:
    """Markdown section crediting every photo used by PAIRS."""
    src_path = raw_dir / "sources.json"
    sources = json.loads(src_path.read_text()) if src_path.is_file() else {}
    lines = [SOURCES_HEADING, "",
             "Real product/museum photos downloaded from Wikimedia Commons by `scripts/fetch_garments_bottoms.py`,",
             "cut out, laid out over `bottom_size` and colour-boosted (derivative works under the same license).",
             "Tops are copied from the outfit folder named in the build log. Raw downloads: `_raw_bottoms/`.", "",
             "| garment | outfits | file | author | license |", "|---|---|---|---|---|"]
    for photo, s in sources.items():
        key = photo[:-5] if photo.endswith("_back") and photo[:-5] in MANIFEST else photo
        if key not in MANIFEST:
            continue
        outfits = ", ".join(oid for oid, _, _, b in PAIRS if b == key)
        if not outfits:
            continue
        desc = s["desc"] + (" - back view" if photo != key else "")
        lic = f"[{s['license']}]({s['license_url']})" if s["license_url"] else s["license"]
        author = (MANIFEST[key].get("attribution") or s["artist"] or s["credit"] or "unknown").replace("|", "/")
        lines.append(f"| {desc} | {outfits} | [{s['title'][5:]}]({s['page_url']}) | {author} | {lic} |")
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
    """One row per PAIRS outfit: thumb | bottom_front | bottom_back | top_front (on the thumbnail grey)."""
    c = garment_cfg(cfg)
    rows = []
    for oid, name, _, _ in PAIRS:
        if not (out_dir / oid).is_dir():
            continue
        o = load_outfit(out_dir / oid, cfg)
        cells = []
        for img in (o.thumb, o.bottom_front, o.bottom_back, o.top_front):
            if img is None:
                continue
            bgr = composite(img, c["thumb_bg"]) if img.shape[2] == 4 else img
            s = SHEET_ROW_H / bgr.shape[0]
            cells.append(cv2.resize(bgr, (round(bgr.shape[1] * s), SHEET_ROW_H), interpolation=cv2.INTER_AREA))
            cells.append(np.full((SHEET_ROW_H, 8, 3), 255, np.uint8))
        row = np.hstack(cells)
        cv2.putText(row, f"{oid}  {name}", (8, SHEET_ROW_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no bottoms outfits built yet")
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0))) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=("all", "fetch", "build", "sheet"), default="all")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, help="garments dir (default: config garments.dir)")
    ap.add_argument("--sheet", type=Path, default=ROOT / "out" / "bottoms_sheet.jpg")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = args.out or Path(cfg["dir"])
    if not out.is_absolute():
        out = ROOT / out
    raw = out / RAW_SUBDIR
    if args.step in ("all", "fetch"):
        try:
            fetch(raw)
        except RuntimeError as e:
            log.error("%s; building from whatever is already in %s", e, raw)
        update_sources_md(out / "SOURCES.md", raw)
    ok = True
    if args.step in ("all", "build"):
        lines = build(raw, out, cfg)
        for line in lines:
            print(line)
        print(f"{len(lines)} outfits -> {out}")
        ok = len(lines) == len(PAIRS)
    if args.step in ("all", "build", "sheet"):
        print(f"contact sheet -> {contact_sheet(out, cfg, args.sheet)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
