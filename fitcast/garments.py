"""Load prepared outfits from `assets/garments/<outfit_id>/` (written by scripts/prepare_garments.py).

Folder layout: `meta.yaml` + RGBA PNGs `top_front.png`, `top_back.png`, `sleeve.png` (tile),
`bottom_front.png`, `bottom_back.png`, and a BGR `thumb.png`. Any part may be absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from fitcast.types import Outfit

log = logging.getLogger("fitcast.garments")

ROOT = Path(__file__).resolve().parents[1]

# meta.yaml part name -> (file name, Outfit field)
PARTS: dict[str, tuple[str, str]] = {
    "top_front": ("top_front.png", "top_front"),
    "top_back": ("top_back.png", "top_back"),
    "sleeve": ("sleeve.png", "sleeve_tex"),
    "bottom_front": ("bottom_front.png", "bottom_front"),
    "bottom_back": ("bottom_back.png", "bottom_back"),
}
SLEEVES = ("short", "long", "none")
SIZE_TABLES = ("mens", "womens")

# Fallbacks for tunables not (yet) in config.yaml `garments:`. Values from the config always win.
DEFAULTS: dict = {
    "thumb_bg": [48, 48, 48],          # BGR dark gray behind thumbnails
    "thumb_margin_frac": 0.06,         # empty border around the garment in a thumbnail
    "sleeve_tile_size": 256,           # square sleeve texture tile, px
    "hem_rows_frac": 0.15,             # bottom fraction of garment rows used to estimate top_body_region
    "opaque_alpha": 250,               # alpha at/above this counts as solid fabric
    "luminance_alpha": 128,            # alpha at/above this counts for mean luminance
    "inpaint_radius": 5,
    "seam_px": 1,                      # half-width of the seam band blended after a patch fill
    "patch_ring_px": 6,                # ring of known pixels matched when copying fabric into a hole
    "patch_ssd_tol": 30.0,             # matches within this per-pixel SSD of the best count as ties
    "patch_split_depth": 4,            # a hole no single patch fits is halved up to this many times
    "graphic_chest_box": [0.12, 0.15, 0.88, 0.70],  # x: frac of body x-range, y: frac of garment height
    "graphic_palette_k": 8,            # fabric colors (k-means in Lab) sampled around the chest box
    "graphic_color_dist": 40.0,        # Lab distance to the nearest fabric color that counts as "graphic"
    "graphic_close_px": 15,            # merge letters/strokes into one blob
    "graphic_min_area_frac": 0.02,     # blob area bounds, fraction of the chest box
    "graphic_max_area_frac": 0.70,
    "graphic_dilate_px": 4,
    "graphic_samples": 4000,
    "long_sleeve_frac": 0.04,          # imported tops: alpha share outside the body in the lower half -> long
    "long_sleeve_slack": 0.10,         # ... body x-range widened by this fraction of its width per side
    "import_open_px": 1,               # morphological open radius cleaning white-background cutouts
    "import_min_area_frac": 0.01,      # imported masks smaller than this share of the image are rejected
    "import_skip_dirs": ["image", "image-parse-v3", "image-parse-agnostic-v3.2", "agnostic-v3.2",
                         "openpose_img", "openpose_json", "image-densepose", "label_maps",
                         "keypoints", "dense", "skeletons"],
}


def garment_cfg(cfg: dict) -> dict:
    """Return `cfg` (config["garments"]) completed with DEFAULTS for keys it lacks."""
    return {**DEFAULTS, **cfg}


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    """Return a 4-channel uint8 image; gray/BGR inputs get an opaque alpha, 16-bit is reduced to 8-bit."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f"unsupported image dtype {img.dtype}")
    if img.ndim == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def resize_rgba(rgba: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Resize with premultiplied alpha so transparent pixels do not bleed color into the edges."""
    w, h = int(wh[0]), int(wh[1])
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    pre = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    shrink = w * h < rgba.shape[0] * rgba.shape[1]
    r = cv2.resize(pre, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    alpha = r[..., 3:4]
    color = np.where(alpha > 0, r[..., :3] * 255.0 / np.maximum(alpha, 1e-6), 0.0)
    return np.concatenate([color, alpha], axis=2).clip(0, 255).round().astype(np.uint8)


def fit_rgba(rgba: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Scale `rgba` to fit inside a transparent (w, h) canvas, keeping aspect ratio, centered."""
    w, h = int(wh[0]), int(wh[1])
    ih, iw = rgba.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, min(w, round(iw * s))), max(1, min(h, round(ih * s)))
    out = np.zeros((h, w, 4), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resize_rgba(rgba, (nw, nh))
    return out


def composite(rgba: np.ndarray, bg_bgr) -> np.ndarray:
    """Alpha-blend an RGBA image over a solid BGR color; returns BGR uint8."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = np.asarray(bg_bgr, np.float32).reshape(1, 1, 3)
    return (rgba[..., :3].astype(np.float32) * a + bg * (1.0 - a)).round().astype(np.uint8)


def make_thumb(rgba: np.ndarray, cfg: dict) -> np.ndarray:
    """Square BGR thumbnail (`thumb_size`) of a garment part on the dark-gray thumbnail background."""
    c = garment_cfg(cfg)
    size = int(c["thumb_size"])
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha)
    if len(ys):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    inner = max(1, round(size * (1.0 - 2.0 * c["thumb_margin_frac"])))
    canvas = np.zeros((size, size, 4), np.uint8)
    o = (size - inner) // 2
    canvas[o:o + inner, o:o + inner] = fit_rgba(rgba, (inner, inner))
    return composite(canvas, c["thumb_bg"])


def _region(value) -> tuple[float, float, float, float]:
    if value is None:
        return (0.0, 0.0, 1.0, 1.0)
    vals = [min(1.0, max(0.0, float(v))) for v in value]
    if len(vals) != 4 or not (vals[0] < vals[2] and vals[1] < vals[3]):
        raise ValueError(f"top_body_region must be [x0, y0, x1, y1] with x0<x1, y0<y1 in [0,1], got {value}")
    return tuple(vals)  # type: ignore[return-value]


def load_outfit(folder: Path, cfg: dict) -> Outfit:
    """Load one outfit folder. Raises ValueError/FileNotFoundError when meta.yaml is missing or broken."""
    folder = Path(folder)
    meta_path = folder / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} is missing")
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{meta_path} is not valid YAML: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} must be a mapping, got {type(meta).__name__}")

    listed = meta.get("parts")
    if listed is None:  # no list: take whatever files are there
        listed = [p for p, (fname, _) in PARTS.items() if (folder / fname).is_file()]
    unknown = set(listed) - set(PARTS)
    if unknown:
        raise ValueError(f"{meta_path}: unknown parts {sorted(unknown)}; valid: {list(PARTS)}")

    images: dict[str, np.ndarray | None] = {field: None for _, field in PARTS.values()}
    for part in listed:
        fname, field = PARTS[part]
        path = folder / fname
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if img is None:
            log.warning("%s: part '%s' listed in meta.yaml but %s is missing/unreadable; using None",
                        folder.name, part, fname)
            continue
        images[field] = ensure_rgba(img)
    if images["top_front"] is None and images["bottom_front"] is None:
        raise ValueError(f"{folder}: no top_front.png or bottom_front.png to show")

    has_top = images["top_front"] is not None
    sleeve = meta.get("sleeve", "short" if has_top else "none")
    size_table = meta.get("size_table", "mens")
    if sleeve not in SLEEVES:
        raise ValueError(f"{meta_path}: sleeve must be one of {SLEEVES}, got {sleeve!r}")
    if size_table not in SIZE_TABLES:
        raise ValueError(f"{meta_path}: size_table must be one of {SIZE_TABLES}, got {size_table!r}")

    thumb_path = folder / "thumb.png"
    thumb = cv2.imread(str(thumb_path), cv2.IMREAD_UNCHANGED) if thumb_path.is_file() else None
    if thumb is None:
        log.warning("%s: thumb.png missing; building one from the garment", folder.name)
        thumb = make_thumb(images["top_front"] if has_top else images["bottom_front"], cfg)
    elif thumb.ndim == 2:
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    elif thumb.shape[2] == 4:
        thumb = composite(ensure_rgba(thumb), garment_cfg(cfg)["thumb_bg"])

    return Outfit(
        id=folder.name,
        name=str(meta.get("name", folder.name)),
        sleeve=sleeve,
        top_front=images["top_front"],
        top_back=images["top_back"],
        sleeve_tex=images["sleeve_tex"],
        bottom_front=images["bottom_front"],
        bottom_back=images["bottom_back"],
        thumb=thumb,
        size_table=size_table,
        top_body_region=_region(meta.get("top_body_region")),
    )


def load_outfits(cfg: dict, garments_dir: str | None = None) -> list[Outfit]:
    """Load every outfit folder under `garments_dir` (default cfg["dir"]), sorted by folder name.

    Relative paths resolve against the repo root. Broken folders are logged and skipped.
    Raises RuntimeError when no outfit loads.
    """
    d = Path(garments_dir if garments_dir is not None else cfg["dir"])
    if not d.is_absolute():
        d = ROOT / d
    fix = f"run `.venv/bin/python scripts/prepare_garments.py --demo` to generate demo outfits into {d}"
    if not d.is_dir():
        raise RuntimeError(f"garments dir {d} does not exist; {fix}")
    outfits = []
    excl = tuple(str(x) for x in (cfg.get("exclude_prefixes") or [])) if garments_dir is None else ()   # live dir only
    for folder in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        if excl and folder.name.startswith(excl):
            continue
        try:
            outfits.append(load_outfit(folder, cfg))
        except (OSError, ValueError, TypeError, cv2.error) as e:
            log.error("skipping outfit %s: %s", folder.name, e)
    if not outfits:
        raise RuntimeError(f"no outfits found in {d}; {fix}")
    log.info("loaded %d outfits from %s", len(outfits), d)
    return outfits
