"""Tests for fitcast.garments (loading) and scripts/prepare_garments.py (asset generation)."""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fitcast.garments import load_outfit, load_outfits  # noqa: E402

_spec = importlib.util.spec_from_file_location("prepare_garments", ROOT / "scripts" / "prepare_garments.py")
pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pg)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return pg.load_config()


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory, cfg) -> Path:
    out = tmp_path_factory.mktemp("garments")
    assert len(pg.generate_demo(out, cfg)) >= 12
    return out


def _solid_top(h: int = 320, w: int = 256, fabric=(40, 40, 220)) -> np.ndarray:
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = fabric
    img[16:310, 40:216, 3] = 255
    return img


# ----------------------------------------------------------------------------- demo + loader

def test_demo_outfits_load_with_expected_parts(demo_dir, cfg):
    folders = sorted(p for p in demo_dir.iterdir() if p.is_dir())
    assert len(folders) >= 12
    tw, th = cfg["top_size"]
    bw, bh = cfg["bottom_size"]
    n, ts = cfg["sleeve_tile_size"], cfg["thumb_size"]
    sizes = {"top_front": (th, tw), "top_back": (th, tw), "bottom_front": (bh, bw), "bottom_back": (bh, bw)}
    for folder in folders:
        o = load_outfit(folder, cfg)
        assert o.top_front is not None or o.bottom_front is not None
        if o.top_front is not None:
            assert o.top_back is not None
        for field, (h, w) in sizes.items():
            img = getattr(o, field)
            if img is None:
                continue
            assert img.dtype == np.uint8 and img.shape == (h, w, 4), (folder.name, field)
            a = img[..., 3]
            assert a[0, 0] == a[0, -1] == a[-1, 0] == a[-1, -1] == 0, (folder.name, field)
            assert (a == 0).mean() > 0.15 and (a == 255).any(), (folder.name, field)
        if o.sleeve_tex is not None:  # a fabric tile: opaque by design
            assert o.sleeve_tex.dtype == np.uint8 and o.sleeve_tex.shape == (n, n, 4)
            assert (o.sleeve_tex[..., 3] == 255).all()
        assert o.thumb.dtype == np.uint8 and o.thumb.shape == (ts, ts, 3)
        x0, y0, x1, y1 = o.top_body_region
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
        assert o.sleeve in ("short", "long", "none") and o.size_table in ("mens", "womens")


def test_demo_is_bright_and_sleeves_match_estimate(demo_dir, cfg):
    for o in load_outfits(cfg, str(demo_dir)):
        for img in (o.top_front, o.bottom_front):
            if img is not None:
                assert pg.mean_luminance(img, cfg) >= cfg["min_mean_luminance"], o.id
        if o.top_front is not None and o.sleeve != "none":
            assert pg.estimate_sleeve(o.top_front, cfg) == o.sleeve, o.id
            x0, _, x1, _ = pg.estimate_body_region(o.top_front, cfg)
            assert abs(x0 - o.top_body_region[0]) < 0.05 and abs(x1 - o.top_body_region[2]) < 0.05, o.id


def test_load_outfits_sorted_and_skips_broken(demo_dir, cfg, tmp_path, caplog):
    keep = sorted(p.name for p in demo_dir.iterdir())[:2]
    for name in keep:
        shutil.copytree(demo_dir / name, tmp_path / name)
    (tmp_path / "00_broken").mkdir()
    (tmp_path / "00_broken" / "meta.yaml").write_text("name: [unclosed\n")
    (tmp_path / "00_nometa").mkdir()
    with caplog.at_level(logging.ERROR, logger="fitcast.garments"):
        outfits = load_outfits(cfg, str(tmp_path))
    assert [o.id for o in outfits] == keep
    assert "00_broken" in caplog.text and "00_nometa" in caplog.text


def test_load_outfits_empty_dir_raises(cfg, tmp_path):
    with pytest.raises(RuntimeError, match=r"prepare_garments\.py --demo"):
        load_outfits(cfg, str(tmp_path))


def test_three_channel_png_gets_opaque_alpha_and_missing_parts_are_none(cfg, tmp_path):
    f = tmp_path / "plain"
    f.mkdir()
    cv2.imwrite(str(f / "top_front.png"), np.full((20, 16, 3), 200, np.uint8))
    meta = {"name": "plain", "category": "top", "sleeve": "short", "size_table": "mens", "parts": ["top_front"]}
    (f / "meta.yaml").write_text(yaml.safe_dump(meta))
    o = load_outfit(f, cfg)
    assert o.top_front.shape == (20, 16, 4) and (o.top_front[..., 3] == 255).all()
    assert o.top_back is None and o.sleeve_tex is None and o.bottom_front is None
    assert o.thumb.shape == (cfg["thumb_size"], cfg["thumb_size"], 3)  # rebuilt when thumb.png is missing


# ----------------------------------------------------------------------------- import

def test_plain_product_import(cfg, tmp_path):
    img = np.full((400, 400, 3), 255, np.uint8)
    img[120:260, 100:300] = (40, 160, 30)
    rgba = pg.extract_garment(img, cfg)
    assert rgba.shape == (140, 200, 4) and (rgba[..., 3] == 255).all()  # tight crop of the rectangle

    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "shirt.png"), img)
    assert len(pg.import_dataset(src, tmp_path / "out", cfg)) == 1
    top = cv2.imread(str(tmp_path / "out" / "top_shirt" / "top_front.png"), cv2.IMREAD_UNCHANGED)
    tw, th = cfg["top_size"]
    assert top.shape == (th, tw, 4)
    nh = round(140 * tw / 200)  # 200x140 fitted to the width, centered vertically
    y0 = (th - nh) // 2
    a = top[..., 3]
    assert (a[:y0] == 0).all() and (a[y0 + nh:] == 0).all() and (a[y0:y0 + nh] == 255).all()
    o = load_outfit(tmp_path / "out" / "top_shirt", cfg)
    assert o.sleeve == "short" and o.top_back is not None and o.sleeve_tex is not None


def test_dataset_layouts(cfg, tmp_path):
    garment = np.full((80, 60, 3), (30, 30, 200), np.uint8)
    mask = np.zeros((80, 60), np.uint8)
    mask[10:70, 15:45] = 255

    def put(rel: str, im: np.ndarray) -> None:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), im)

    put("viton/train/cloth/00001_00.jpg", garment)
    put("viton/train/cloth-mask/00001_00.png", mask)
    put("viton/train/image/00001_00.jpg", garment)          # person photo: ignored
    put("dc/lower_body/images/013_0.jpg", garment)           # DressCode person photo: ignored
    put("dc/lower_body/images/013_1.jpg", garment)
    put("dc/lower_body/cloth_mask/013_1.png", mask)

    viton = pg.find_garments(tmp_path / "viton", cfg)
    assert [(p.name, m.name if m else None, k) for p, m, k in viton] == [("00001_00.jpg", "00001_00.png", "top")]
    dc = pg.find_garments(tmp_path / "dc", cfg)
    assert [(p.name, m.name if m else None, k) for p, m, k in dc] == [("013_1.jpg", "013_1.png", "bottom")]

    out = tmp_path / "out"
    pg.import_dataset(tmp_path / "dc", out, cfg)
    o = load_outfit(out / "bottom_013_1", cfg)
    assert o.top_front is None and o.bottom_front is not None and o.sleeve == "none"
    bw, bh = cfg["bottom_size"]
    a = o.bottom_front[..., 3]  # 30x60 mask crop fitted to the height, centered horizontally
    nw = round(30 * bh / 60)
    x0 = (bw - nw) // 2
    assert (a[:, :x0 - 1] == 0).all() and (a[:, x0 + nw + 1:] == 0).all() and a[bh // 2, bw // 2] == 255
    assert np.array_equal(o.bottom_back, cv2.flip(o.bottom_front, 1))


# ----------------------------------------------------------------------------- back / color

def test_back_generation_removes_center_graphic(cfg):
    fabric = (40, 40, 220)
    front = _solid_top(fabric=fabric)
    front[110:190, 95:160, :3] = (0, 230, 255)  # yellow chest graphic
    back = pg.synthesize_back(front, cfg)
    w = front.shape[1]
    assert np.array_equal(back[..., 3], cv2.flip(front, 1)[..., 3])
    chest = back[110:190, w - 160:w - 95, :3].reshape(-1, 3).astype(int)
    assert np.abs(chest.mean(axis=0) - fabric).max() < 15
    assert np.abs(chest - fabric).max() < 40


def test_no_graphic_detected_on_stripes(cfg):
    front = _solid_top()
    front[(np.arange(front.shape[0]) // 12) % 2 == 0, :, :3] = (245, 245, 245)
    assert pg.detect_graphic(front, cfg).max() == 0


def test_boost_raises_saturation_and_brightness(cfg):
    img = np.zeros((40, 40, 4), np.uint8)
    img[..., :3] = (90, 120, 150)
    img[..., 3] = 255
    out = pg.boost_colors(img, cfg)
    hsv0 = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2HSV).astype(float)
    hsv1 = cv2.cvtColor(np.ascontiguousarray(out[..., :3]), cv2.COLOR_BGR2HSV).astype(float)
    assert hsv1[..., 1].mean() > hsv0[..., 1].mean() and hsv1[..., 2].mean() > hsv0[..., 2].mean()
    assert np.array_equal(out[..., 3], img[..., 3])


def test_dark_garment_warns(cfg, caplog):
    dark = np.zeros((40, 40, 4), np.uint8)
    dark[..., :3] = (25, 20, 30)
    dark[..., 3] = 255
    with caplog.at_level(logging.WARNING):
        assert pg.check_luminance(dark, cfg, "dark_tee") < cfg["min_mean_luminance"]
    assert any(r.levelno == logging.WARNING and "dark_tee" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        pg.check_luminance(_solid_top(fabric=(20, 215, 255)), cfg, "bright_tee")
    assert not caplog.records
