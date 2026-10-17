"""Calibration tests: homography math, simulated end-to-end run, detection, warp, rescale."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fitcast.calibration import (Calibration, SimulatedRig, black_frame, detect_markers,
                                 marker_points, render_markers, run_calibration)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
CFG = CONFIG["calibration"]
CAM = (int(CONFIG["capture"]["width"]), int(CONFIG["capture"]["height"]))
PROJ = (int(CFG["projector_width"]), int(CFG["projector_height"]))


def _warp(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), M).reshape(-1, 2)


def _random_cam_pts(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform((0, 0), (CAM[0] - 1, CAM[1] - 1), (n, 2))


def _exact_cal(rig: SimulatedRig) -> Calibration:
    """A Calibration solved from noise-free correspondences of the rig's true homography."""
    cam = _random_cam_pts(20, seed=1)
    cal = Calibration(CFG, CAM)
    assert cal.set_homography(cam, _warp(rig.H_true, cam)) < 1e-3
    return cal


@pytest.fixture(scope="module")
def sim():
    rig = SimulatedRig(CFG, CAM)
    cal = run_calibration(CFG, CAM, rig.capture, rig.show, log=lambda *_: None)
    return rig, cal


def test_homography_round_trip():
    cal = _exact_cal(SimulatedRig(CFG, CAM))
    pts = _random_cam_pts(200, seed=2)
    back = cal.unproject_points(cal.project_points(pts))
    assert np.abs(back - pts).max() < 1e-6


def test_simulated_end_to_end(sim, tmp_path):
    rig, cal = sim
    assert cal.ok
    assert cal.data.reproj_error_px < 1.0
    assert cal.data.verified and cal.data.verify_max_err_px <= CFG["verify_tolerance_px"]
    corners = np.array([[0, 0], [CAM[0] - 1, 0], [0, CAM[1] - 1], [CAM[0] - 1, CAM[1] - 1]], float)
    assert np.linalg.norm(cal.project_points(corners) - _warp(rig.H_true, corners), axis=1).max() < 2.0
    path = tmp_path / "cal.yaml"
    cal.data.gallery_rect = [20, 40, 300, 700]
    cal.save(str(path))
    loaded = Calibration(CFG, CAM)
    assert loaded.load(str(path))
    assert np.allclose(loaded.H, cal.H)
    assert loaded.data.gallery_rect == [20, 40, 300, 700] and loaded.data.verified


def test_detect_markers_matches_ground_truth():
    rig = SimulatedRig(CFG, CAM, seed=3)
    pts = marker_points(CFG)
    rig.show(black_frame(CFG))
    dark = rig.capture()
    rig.show(render_markers(CFG, pts))
    lit = rig.capture()
    found = detect_markers(lit, dark, CFG, len(pts), int(CFG["marker_grid"][1]))
    assert found is not None and found.shape == (len(pts), 2)
    truth = _warp(np.linalg.inv(rig.H_true), pts)
    assert np.linalg.norm(found - truth, axis=1).max() < 1.5
    # wrong expected count -> None
    assert detect_markers(lit, dark, CFG, len(pts) + int(CFG["marker_grid"][1]),
                          int(CFG["marker_grid"][1])) is None


def test_to_projector_shape_and_pixel_mapping():
    cal = _exact_cal(SimulatedRig(CFG, CAM))
    img = np.zeros((CAM[1], CAM[0], 3), np.uint8)
    cx, cy, color = 400, 300, (10, 200, 60)
    img[cy - 4:cy + 5, cx - 4:cx + 5] = color
    out = cal.to_projector(img)
    assert out.shape == (PROJ[1], PROJ[0], 3) and out.dtype == np.uint8
    px, py = np.rint(cal.project_points([[cx, cy]])[0]).astype(int)
    assert tuple(int(v) for v in out[py, px]) == color
    assert cal.to_projector(img) is out          # reused buffer, no per-frame allocation


def test_to_projector_fallback_without_calibration():
    cal = Calibration(CFG, CAM)
    assert not cal.ok and not cal.load(str(ROOT / "does-not-exist.yaml"))
    img = np.full((CAM[1], CAM[0], 3), 77, np.uint8)
    out = cal.to_projector(img)
    assert out.shape == (PROJ[1], PROJ[0], 3)
    assert tuple(out[PROJ[1] // 2, PROJ[0] // 2]) == (77, 77, 77)


def test_load_rejects_invalid_homography(tmp_path):
    path = tmp_path / "bad.yaml"
    for H in ([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]]):
        path.write_text(yaml.safe_dump({"H": H, "camera_size": list(CAM), "projector_size": list(PROJ),
                                        "reproj_error_px": 0.1}))
        assert not Calibration(CFG, CAM).load(str(path))


def test_load_rescales_to_runtime_camera_size(sim, tmp_path):
    _, cal = sim
    path = tmp_path / "cal.yaml"
    cal.data.gallery_rect = [20, 40, 300, 700]
    cal.save(str(path))
    small = (CAM[0] // 2, CAM[1] // 2)
    half = Calibration(CFG, small)
    assert half.load(str(path)) and half.data.camera_size == small
    # cv2.resize maps pixel centers: x_small = (x_big + 0.5) * s - 0.5
    big_pts = _random_cam_pts(50, seed=4)
    small_pts = (big_pts + 0.5) * (np.array(small) / np.array(CAM)) - 0.5
    assert np.abs(half.project_points(small_pts) - cal.project_points(big_pts)).max() < 1e-6
    assert half.data.gallery_rect == [10, 20, 150, 350]
    resc = cal.rescaled(small)
    assert np.allclose(resc.H, half.H)
