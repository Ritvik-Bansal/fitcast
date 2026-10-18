"""Accuracy metric (fitcast/accuracy.py) + the --simulate procedure of scripts/accuracy_check.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from fitcast import accuracy
from fitcast.accuracy import (boundary_errors_mm, boundary_pixels, detect_lit_outline, outline_image, summarize,
                              truth_boundary_from_depth)

ROOT = Path(__file__).resolve().parents[1]
K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])


def _load_script():
    spec = importlib.util.spec_from_file_location("accuracy_check", ROOT / "scripts" / "accuracy_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["accuracy_check"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def det_cfg(config) -> dict:
    return {**config["calibration"], "min_blob_px": 8}


def _rect_mask(x0=100, y0=50, x1=300, y1=200, shape=(360, 640)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[y0:y1 + 1, x0:x1 + 1] = 255
    return m


def test_boundary_pixels_rectangle():
    pts = boundary_pixels(_rect_mask())
    assert pts.dtype == np.int32 and pts.shape[1] == 2
    assert len(pts) == 2 * 201 + 2 * 151 - 4                   # every perimeter pixel exactly once
    on_edge = (np.isin(pts[:, 0], (100, 300)) & (pts[:, 1] >= 50) & (pts[:, 1] <= 200)) | \
              (np.isin(pts[:, 1], (50, 200)) & (pts[:, 0] >= 100) & (pts[:, 0] <= 300))
    assert on_edge.all()
    assert boundary_pixels(np.zeros((10, 10), np.uint8)).shape == (0, 2)
    assert np.array_equal(truth_boundary_from_depth(_rect_mask()), pts)


def test_outline_image_band_is_inside_the_mask():
    m = _rect_mask()
    img = outline_image(m, 2, (0, 255, 0))
    assert img.shape == m.shape + (3,) and img.dtype == np.uint8
    lit = img[..., 1] > 0
    assert not (lit & (m == 0)).any()                           # never spills off the body
    assert lit[125, 100] and lit[125, 101] and not lit[125, 102]   # 2 px band on the left edge
    assert (img[..., 0] == 0).all() and (img[..., 2] == 0).all()


def test_boundary_errors_mm_known_shift():
    """Truth = vertical segment at x = 400; lit = the same segment 4 px right; 3 m depth, fx 1000 -> 12 mm."""
    ys = np.arange(100, 300)
    truth = np.stack([np.full_like(ys, 400), ys], axis=1)
    lit = np.stack([np.full_like(ys, 404), ys], axis=1)
    depth = np.full((720, 1280), 3.0, np.float32)
    e = boundary_errors_mm(lit, truth, depth, K)
    assert e.shape == (len(ys),)
    assert np.allclose(e[10:-10], 12.0, atol=1e-6)               # away from the segment ends: exactly 4 px
    assert np.all(e >= 12.0 - 1e-6)
    # signed variant: the side of the truth mask decides the sign, offset_px is subtracted
    body = np.zeros((720, 1280), np.uint8)
    body[:, :401] = 255                                         # body left of the edge: lit pixels are outside
    s = boundary_errors_mm(lit, truth, depth, K, inside_mask=body, offset_px=0.5)
    assert np.allclose(s[10:-10], -4.5 * 3.0, atol=1e-6)
    # max_px rejects, invalid depth everywhere -> NaN
    assert np.isnan(boundary_errors_mm(lit, truth, depth, K, max_px=3.0)).all()
    assert np.isnan(boundary_errors_mm(lit, truth, np.zeros_like(depth), K)).all()


def test_boundary_errors_use_truth_depth():
    """A lit pixel that missed the body (wall behind, deeper) is scaled at the body's depth."""
    truth = np.array([[400, 200]])
    lit = np.array([[410, 200]])
    depth = np.full((720, 1280), 3.2, np.float32)
    depth[200, 400] = 2.0
    assert boundary_errors_mm(lit, truth, depth, K)[0] == pytest.approx(10 * 2.0, abs=1e-6)


def test_summarize():
    s = summarize(np.array([1.0, 2.0, 3.0, 4.0, np.nan, -10.0]))
    assert s["n"] == 5 and s["n_rejected"] == 1
    assert s["mean"] == pytest.approx(4.0) and s["median"] == pytest.approx(3.0)
    assert s["p95"] == pytest.approx(np.percentile([1, 2, 3, 4, 10], 95)) and s["max"] == 10.0
    assert s["bias"] == pytest.approx(0.0)
    empty = summarize(np.array([np.nan]))
    assert empty["median"] is None and empty["n"] == 0 and empty["n_rejected"] == 1


def test_metric_floor_outline_read_back(det_cfg):
    """The outline image itself as the 'lit' frame: the detected curve sits on the band, within 0.5 px of
    its nominal centre ((thickness - 1) / 2 px inside the contour)."""
    m = np.zeros((720, 1280), np.uint8)
    cv2.ellipse(m, (640, 360), (220, 160), 15, 0, 360, 255, -1)
    m[300:420, 200:260] = 255
    lit_img = outline_image(m, 2, (255, 255, 255))
    lit = detect_lit_outline(lit_img, np.zeros_like(lit_img), det_cfg)
    truth = truth_boundary_from_depth(m)
    assert len(lit) > 0.8 * len(truth)
    depth = np.full(m.shape, 2.0, np.float32)
    s = summarize(boundary_errors_mm(lit, truth, depth, K, inside_mask=m, offset_px=0.5))
    px = 2.0 * 1000 / 1000.0                                    # mm per px at 2 m, f 1000
    assert s["median"] <= 0.5 * px + 1e-6 and abs(s["bias"]) < 0.5 * px


def test_detect_lit_outline_ignores_noise(det_cfg):
    rng = np.random.default_rng(0)
    dark = rng.integers(20, 30, (720, 1280, 3), dtype=np.uint8)
    lit = dark.copy()
    cv2.rectangle(lit, (300, 200), (900, 500), (0, 200, 0), 2)   # a projected green outline
    pts = detect_lit_outline(lit, dark, det_cfg)
    assert len(pts) > 1500
    d = np.minimum.reduce([np.abs(pts[:, 0] - 300), np.abs(pts[:, 0] - 900),
                           np.abs(pts[:, 1] - 200), np.abs(pts[:, 1] - 500)])
    assert d.max() <= 2


def test_skeleton_fallback_thins_a_band(monkeypatch):
    band = np.zeros((100, 100), np.uint8)
    band[20:80, 40:44] = 255
    monkeypatch.setattr(accuracy, "_HAS_THINNING", False)
    sk = accuracy.thin(band)
    cols = np.nonzero(sk[30:70].any(axis=0))[0]
    assert sk.any() and len(cols) <= 2 and 40 <= cols.min() and cols.max() <= 43


def test_simulate_perfect_calibration_and_shift_detection(config):
    """--simulate end to end (~3 s): median < 3 mm with the true and the recovered calibration; a 15 px
    principal-point shift raises the median to a large fraction of 15 * z / f_p."""
    mod = _load_script()
    try:
        res = mod.run_simulation(config, out_dir=None, log=lambda s: None)
    except RuntimeError as e:
        if "OpenGL" in str(e):
            pytest.skip(str(e))
        raise
    true_ph, rec_ph, shift_ph = res["phases"]
    assert true_ph["summary"]["median"] < 3.0
    assert rec_ph["summary"]["median"] < 3.0 and rec_ph["summary"]["p95"] < 5.0
    assert rec_ph["summary"]["n"] > 5000
    exp = res["expected_shift_mm"]
    assert 14.0 < exp < 17.0
    assert shift_ph["summary"]["median"] > 0.5 * exp
    assert shift_ph["summary"]["median"] > 3 * rec_ph["summary"]["median"]
    assert all(res["gates"].values())
    assert res["scene"]["depth_mask_iou"] > 0.98
