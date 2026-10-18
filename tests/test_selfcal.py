"""Self-calibration on a synthetic two-plane scene and the two-plane homography model."""
import numpy as np
import cv2
import pytest

from fitcast.calibration import Calibration, SimulatedRig, make_sim_homography
from fitcast.selfcal import disc_grid, self_calibrate


class TwoPlaneRig(SimulatedRig):
    """Wall through H_true; a 'person' rectangle in camera space seen through H_person (nearer plane)."""

    def __init__(self, cfg, cam, H_wall, H_person, person_rect):
        super().__init__(cfg, cam, H_wall)
        self.H_person = H_person
        x0, y0, x1, y1 = person_rect
        self.person = np.zeros((cam[1], cam[0]), np.uint8)
        self.person[y0:y1, x0:x1] = 255

    def capture(self):
        wall = super().capture()
        body = cv2.warpPerspective(self._frame, np.linalg.inv(self.H_person), (self.person.shape[1], self.person.shape[0]))
        if body.ndim == 2:
            body = cv2.cvtColor(body, cv2.COLOR_GRAY2BGR)
        body = np.clip(body.astype(np.float32) + self.ambient + 10, 0, 255).astype(np.uint8)
        out = wall.copy()
        out[self.person > 0] = body[self.person > 0]
        return out


def test_self_calibration_two_planes(config):
    cam = (1280, 720)
    cfg = config["calibration"]
    H_wall = make_sim_homography(cam, (1920, 1080))
    # person plane: the same projector seen 10% larger and shifted (nearer plane)
    S = np.array([[1.1, 0, -60.0], [0, 1.1, -40.0], [0, 0, 1]])
    H_person = S @ H_wall
    rig = TwoPlaneRig(cfg, cam, H_wall, H_person, (520, 120, 760, 700))
    cal = Calibration(cfg, cam)
    res = self_calibrate(config["selfcal"], cfg, cal, (1920, 1080), rig.capture, rig.show, rig.person,
                         head_box=(560, 40, 720, 130), torso_px=200.0, log_fn=lambda m: None)
    assert res["n_body"] >= 4 and res["H_body"] is not None, res
    # recovered body homography maps camera -> projector like H_person (check on points inside the body)
    pts = np.array([[600, 300], [700, 500], [550, 650], [740, 200]], np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, res["H_body"]).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, H_person / H_person[2, 2]).reshape(-1, 2)
    assert np.abs(a - b).max() < 2.5, np.abs(a - b).max()
    if res["H_wall"] is not None:
        a = cv2.perspectiveTransform(pts, res["H_wall"]).reshape(-1, 2)
        b = cv2.perspectiveTransform(pts, H_wall / H_wall[2, 2]).reshape(-1, 2)
        assert np.abs(a - b).max() < 2.5


def test_two_plane_model_interpolates(config):
    cam = (1280, 720)
    cal = Calibration(config["calibration"], cam)
    H0 = make_sim_homography(cam, (1920, 1080)); H0 = H0 / H0[2, 2]
    H1 = H0.copy(); H1[0, 2] += 100.0; H1[1, 2] += 50.0
    assert cal.set_plane(H0, 100.0) == "primary"
    assert cal.set_plane(H1, 150.0) == "second"
    Hm = cal.H_for(125.0)
    assert abs(Hm[0, 2] - (H0[0, 2] + 50.0)) < 1e-6 and abs(Hm[1, 2] - (H0[1, 2] + 25.0)) < 1e-6
    assert np.allclose(cal.H_for(None), H0)
    img = np.zeros((720, 1280, 3), np.uint8); img[300:310, 600:610] = 255
    p1 = cal.to_projector(img, torso_px=150.0).copy()
    ref = cv2.warpPerspective(img, H1, (1920, 1080))
    ya, xa = np.nonzero(ref[:, :, 0]); yb, xb = np.nonzero(p1[:, :, 0])
    assert abs(xb.mean() - xa.mean()) < 1.5 and abs(yb.mean() - ya.mean()) < 1.5   # second plane -> warp with H1
    assert cal.set_plane(H0, 105.0) == "primary"          # same distance -> replaces the primary (reference 105 now)
    assert cal.data.depth_ref_torso_px == 105.0 and cal.data.H2 is not None


def test_split_warp_identity_and_shift():
    """Identity part corrections reproduce the single warp; a translated upper correction shifts only the top."""
    import yaml
    from fitcast.calibration import Calibration, CalibrationData, make_sim_homography
    cfg = yaml.safe_load(open("config.yaml"))["calibration"]
    cam, proj = (1280, 720), (1920, 1080)
    cal = Calibration(cfg, cam)
    cal.data = CalibrationData(H=make_sim_homography(cam, proj), camera_size=cam, projector_size=proj, reproj_error_px=1.0,
                               timestamp="t", gallery_rect=None, verified=True, verify_max_err_px=None)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (720, 1280, 3), np.uint8)
    single = cal.to_projector(img).copy()
    cal.data.D_upper, cal.data.D_lower = np.eye(3), np.eye(3)
    split = cal.to_projector(img, split_y=360.0).copy()
    assert np.array_equal(single, split)
    cal.data.D_upper = np.array([[1, 0, 0], [0, 1, -40.0], [0, 0, 1]])      # upper body 40 projector px up
    moved = cal.to_projector(img, split_y=360.0).copy()
    # warpPerspective computes its maps in fixed point per block, so a shifted warp is equal only up to rounding
    assert np.abs(moved[100:200].astype(int) - single[140:240].astype(int)).mean() < 2.0   # top rows: shifted copy
    assert np.array_equal(moved[900:1000], single[900:1000])                                 # bottom rows: untouched
