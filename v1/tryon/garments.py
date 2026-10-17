"""Garment catalog: four view PNGs with alpha, anchors.json, meta.json per folder.

anchors.json maps each view to named points in that PNG's pixel coordinates:
  {"front": {"left_shoulder": [x, y], ...}, "back": {...}, "left": {...}, "right": {...}}
"left"/"right" in anchor names always mean the wearer's left/right. The left.png
view is the garment seen from the wearer's left side. Missing views are filled
in (back <- front, left <-> right mirrored) so a garment with only front.png runs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIEWS = ("front", "left", "back", "right")
SHIRT_ANCHORS = ("left_shoulder", "right_shoulder", "left_armpit", "right_armpit", "left_hem", "right_hem",
                 "left_sleeve_end", "right_sleeve_end", "collar")
PANTS_ANCHORS = ("left_waist", "right_waist", "left_knee", "right_knee", "left_ankle", "right_ankle", "crotch")
ANCHORS = {"shirt": SHIRT_ANCHORS, "pants": PANTS_ANCHORS}


@dataclass
class GarmentView:
    name: str
    image: np.ndarray
    anchors: dict[str, np.ndarray]
    source: str
    warper: object = None


@dataclass
class Garment:
    id: str
    name: str
    price: float
    category: str
    sleeve_length: str
    folder: Path
    views: dict[str, GarmentView] = field(default_factory=dict)

    @property
    def price_text(self) -> str:
        return f"${self.price:,.2f}"


def _swap_side(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _mirror(view: GarmentView, new_name: str) -> GarmentView:
    w = view.image.shape[1]
    img = np.ascontiguousarray(view.image[:, ::-1])
    anchors = {_swap_side(k): np.array([w - 1 - v[0], v[1]], np.float32) for k, v in view.anchors.items()}
    return GarmentView(new_name, img, anchors, f"mirror of {view.name}")


def _load_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        print(f"[garments] {path} has no alpha channel; treating it as fully opaque")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def load_garment(folder: Path, max_size: int) -> Garment:
    meta = {}
    if (folder / "meta.json").exists():
        meta = json.loads((folder / "meta.json").read_text())
    category = meta.get("category", "shirt")
    if category not in ANCHORS:
        raise ValueError(f"{folder}/meta.json: category must be 'shirt' or 'pants', got {category!r}")
    anchors_all = json.loads((folder / "anchors.json").read_text()) if (folder / "anchors.json").exists() else {}
    if not isinstance(anchors_all, dict) or not all(isinstance(v, dict) for v in anchors_all.values()):
        raise ValueError("anchors.json must map each view name to {anchor_name: [x, y]}")
    g = Garment(
        id=folder.name,
        name=meta.get("name", folder.name),
        price=float(meta.get("price", 0.0)),
        category=category,
        sleeve_length=meta.get("sleeve_length", "short"),
        folder=folder,
    )
    for view in VIEWS:
        path = folder / f"{view}.png"
        if not path.exists():
            continue
        img = _load_png(path)
        scale = min(1.0, max_size / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        pts = {k: np.array(v, np.float32) * scale for k, v in anchors_all.get(view, {}).items()
               if isinstance(v, (list, tuple)) and len(v) == 2}
        g.views[view] = GarmentView(view, img, pts, str(path))
    if "front" not in g.views:
        raise ValueError(f"{folder} has no front.png")
    missing = [a for a in ANCHORS[category] if a not in g.views["front"].anchors]
    if missing:
        print(f"[garments] {g.id}: front view is missing anchors {missing}; it will use the simple fit")
    if "back" not in g.views:
        g.views["back"] = GarmentView("back", g.views["front"].image, dict(g.views["front"].anchors), "copy of front")
    if "left" not in g.views and "right" in g.views:
        g.views["left"] = _mirror(g.views["right"], "left")
    if "right" not in g.views and "left" in g.views:
        g.views["right"] = _mirror(g.views["left"], "right")
    for side in ("left", "right"):
        if side not in g.views:
            g.views[side] = GarmentView(side, g.views["front"].image, dict(g.views["front"].anchors), "copy of front")
    return g


class Catalog:
    def __init__(self, garments: list[Garment]):
        self.garments = garments
        self.by_category = {c: [g for g in garments if g.category == c] for c in ANCHORS}

    def items(self, category: str) -> list[Garment]:
        return self.by_category.get(category, [])


def load_catalog(cfg) -> Catalog:
    root = cfg.path(cfg.garments_dir)
    if not root.exists():
        sys.exit(f"[garments] {root} does not exist. Run: python tools/make_placeholders.py")
    folders = [root / gid for gid in cfg.catalog] if cfg.catalog else sorted(p for p in root.iterdir() if p.is_dir())
    garments = []
    for folder in folders:
        if not folder.is_dir():
            print(f"[garments] unknown garment id '{folder.name}' in catalog (no folder {folder})")
            continue
        try:
            garments.append(load_garment(folder, cfg.garment_max_size))
        except Exception as e:  # one malformed garment folder must not take the whole demo down
            print(f"[garments] skipping {folder.name}: {type(e).__name__}: {e}")
    if not garments:
        sys.exit(f"[garments] no garments found in {root}. Run: python tools/make_placeholders.py")
    cat = Catalog(garments)
    print(f"[garments] loaded {len(cat.items('shirt'))} shirts, {len(cat.items('pants'))} pants")
    return cat
