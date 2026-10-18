"""3D calibration: Gray code, decode, solve on SimulatedRig3D, YAML round trip, verification pattern."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fitcast.calib3d import (DEFAULTS, Calibration3D, SimulatedRig3D, decode_gray_code, gray_code_patterns,
                             lift_to_3d, load_calibration3d, rotation_error_deg, run_calibration3d,
                             save_calibration3d, sim_truth, verification_pattern)

CAM, PROJ = (1280, 720), (1920, 1080)


@pytest.fixture(scope="module")
def cfg(config) -> dict:
    return {**DEFAULTS, **config["calibration3d"], "settle_frames": 0}


@pytest.fixture(scope="module")
def truth(cfg) -> dict:
    return sim_truth(CAM, PROJ, float(cfg["throw_ratio_guess"]))


@pytest.fixture(scope="module")
def rig(cfg, truth) -> SimulatedRig3D:
    return SimulatedRig3D(cfg, CAM, PROJ, **truth)


@pytest.fixture(scope="module")
def decoded(cfg, rig):
    """Camera captures of [white, black] + the sequence (uint8 BGR, one frame each) and their decode."""
    patterns, white, black = gray_code_patterns(PROJ)
    caps = []
    for p in [white, black] + patterns:
        rig.show(p)
        caps.append(rig.capture())
    return caps, decode_gray_code(caps[2:], caps[0], caps[1], PROJ, cfg)


@pytest.fixture(scope="module")
def calib(cfg, rig, truth) -> Calibration3D:
    return run_calibration3d(cfg, CAM, PROJ, rig.capture, rig.show, rig.depth, rig.conf,
                             log=lambda s: None, K_cam=truth["K_cam"])


def test_gray_code_pattern_count_and_size():
    patterns, white, black = gray_code_patterns(PROJ)
    gc = cv2.structured_light.GrayCodePattern.create(width=PROJ[0], height=PROJ[1])
    assert len(patterns) == gc.getNumberOfPatternImages() == 2 * (11 + 11)
    for img in patterns + [white, black]:
        assert img.shape == (PROJ[1], PROJ[0]) and img.dtype == np.uint8
    assert white.min() == 255 and black.max() == 0
    for a, b in zip(patterns[0::2], patterns[1::2]):          # every code image is followed by its inverse
        assert np.array_equal(a, 255 - b)


def test_decode_recovers_projector_coordinates(rig, decoded):
    _, (proj_xy, valid) = decoded
    assert proj_xy.shape == (CAM[1], CAM[0], 2) and proj_xy.dtype == np.float32 and valid.dtype == bool
    truth = rig.true_proj_xy()
    inside = (truth[..., 0] >= 0) & (truth[..., 0] < PROJ[0] - 1) & (truth[..., 1] >= 0) & (truth[..., 1] < PROJ[1] - 1)
    assert valid.sum() > 0.9 * inside.sum()                   # nearly every lit pixel decodes
    err = np.linalg.norm(proj_xy[valid] - truth[valid], axis=1)
    assert np.sqrt(np.mean(err ** 2)) < 0.6
    assert err.max() < 1.5                                    # Gray code: a wrong bit near a stripe edge is off by one


def test_decode_matches_opencv_getprojpixel(cfg, decoded):
    caps, (proj_xy, valid) = decoded
    gray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in caps[2:]]
    gc = cv2.structured_light.GrayCodePattern.create(width=PROJ[0], height=PROJ[1])
    gc.setWhiteThreshold(int(cfg["gray_code_white_threshold"]))
    ys, xs = np.nonzero(valid)
    pick = np.random.default_rng(1).choice(len(xs), 300, replace=False)
    for x, y in zip(xs[pick], ys[pick]):
        err, (px, py) = gc.getProjPixel(gray, int(x), int(y))
        assert not err
        assert (px, py) == tuple(proj_xy[y, x].astype(int))


def test_lift_to_3d_drops_invalid_depth():
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    depth = np.full((720, 1280), 2.0, np.float32)
    depth[0, 0] = 0
    P = lift_to_3d(np.array([[640, 360], [1640, 360], [0, 0], [1140, 360]]), depth, K)
    assert P.shape == (4, 3)
    assert np.allclose(P[0], [0, 0, 2]) and np.allclose(P[3], [1.0, 0, 2])
    assert np.isnan(P[1]).all() and np.isnan(P[2]).all()    # outside the image / zero depth


def test_run_calibration3d_recovers_truth(calib, truth):
    Kt = truth["K_p_true"]
    assert calib.reproj_error_px < 1.0
    assert abs(calib.K_p[0, 0] / Kt[0, 0] - 1) < 0.01
    assert abs(calib.K_p[1, 1] / Kt[1, 1] - 1) < 0.01
    assert rotation_error_deg(calib.R, truth["R_true"]) < 0.2
    assert np.linalg.norm(calib.t - truth["t_true"]) < 0.01
    assert calib.n_points > 10000


def test_yaml_round_trip(calib, tmp_path):
    path = tmp_path / "c3d.yaml"
    save_calibration3d(path, calib)
    back = load_calibration3d(path)
    for k in ("K_cam", "dist_cam", "K_p", "dist_p", "R", "t"):
        assert np.allclose(getattr(back, k), getattr(calib, k), rtol=0, atol=1e-12), k
    assert back.cam_size == CAM and back.proj_size == PROJ
    assert back.reproj_error_px == pytest.approx(calib.reproj_error_px)
    assert back.n_points == calib.n_points and back.timestamp == calib.timestamp and back.notes == calib.notes


def test_load_rejects_bad_shapes(calib, tmp_path):
    import yaml
    path = tmp_path / "bad.yaml"
    save_calibration3d(path, calib)
    raw = yaml.safe_load(path.read_text())
    raw["K_p"] = raw["K_p"][:2]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="K_p"):
        load_calibration3d(path)
    with pytest.raises(FileNotFoundError):
        load_calibration3d(tmp_path / "missing.yaml")


def test_verification_pattern_lands_on_box_edges(calib, rig, truth):
    img = verification_pattern(calib, rig.depth(), truth["K_cam"], PROJ)
    assert img.shape == (PROJ[1], PROJ[0], 3) and img.dtype == np.uint8
    lit = img.any(axis=2)
    assert 0.001 < lit.mean() < 0.3
    h, w = lit.shape
    for qy in (slice(0, h // 2), slice(h // 2, h)):           # the grid covers the whole projector frame
        for qx in (slice(0, w // 2), slice(w // 2, w)):
            assert lit[qy, qx].any()
    # the true box outlines, projected with the TRUE projector, fall on lit edge (green) pixels
    green = (img[..., 1] == 255) & (img[..., 0] == 0)
    green = cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rvec, _ = cv2.Rodrigues(truth["R_true"])
    for x0, x1, y0, y1, z in rig.box_rects:
        s = np.linspace(0.05, 0.95, 50)
        outline = np.concatenate([np.stack([x0 + (x1 - x0) * s, np.full_like(s, y0)], 1),
                                  np.stack([x0 + (x1 - x0) * s, np.full_like(s, y1)], 1),
                                  np.stack([np.full_like(s, x0), y0 + (y1 - y0) * s], 1),
                                  np.stack([np.full_like(s, x1), y0 + (y1 - y0) * s], 1)])
        pts = np.column_stack([outline, np.full(len(outline), z)])
        q, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, truth["t_true"], truth["K_p_true"], truth["dist_p_true"])
        q = np.rint(q.reshape(-1, 2)).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        assert ok.all()
        assert green[q[:, 1], q[:, 0]].mean() > 0.9
