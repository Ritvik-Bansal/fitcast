import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    from _warp_helpers import ROOT  # noqa: F401  (puts the repo root on sys.path)
except ImportError:
    from tests._warp_helpers import ROOT  # noqa: F401
from tryon.calibration import (Calibration, checkerboard, dot_centroid, dot_grid, find_corners, load_calibration,
                               order_like, save_calibration, solve_homography)

PROJ = (1920, 1080)
CAM = (1280, 720)
PATTERN = (9, 6)


def true_cam_from_proj(scale=0.52, deg=3.0, tx=170.0, ty=60.0, persp=(2.0e-5, 1.5e-5)):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])
    P = np.eye(3)
    P[2, 0], P[2, 1] = persp
    return T @ P @ R @ S


def synth_camera(Hcp, center=(960, 540), square=100, seed=0, noise=3.0, blur=1.2, ambient=90):
    """Camera view of a projected checkerboard falling on a gray (lit) wall."""
    proj_img, proj_corners = checkerboard(PROJ, PATTERN, center, square)
    g = cv2.cvtColor(proj_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)  # projector/lens softness; avoids aliasing at exact 2:1 downscales
    warped = cv2.warpPerspective(g, Hcp, CAM, flags=cv2.INTER_LINEAR, borderValue=0)
    cover = cv2.warpPerspective(np.ones_like(g), Hcp, CAM, flags=cv2.INTER_LINEAR, borderValue=0)
    # projected black is the ambient wall (gray), white adds light; outside the projector footprint: darker gray
    cam = np.where(cover > 0.99, ambient * 0.5 + 0.55 * warped + ambient * 0.1, ambient * 0.6)
    cam = cv2.GaussianBlur(cam, (0, 0), blur)
    rng = np.random.RandomState(seed)
    cam = np.clip(cam + rng.normal(0, noise, cam.shape), 0, 255).astype(np.uint8)
    cam_corners = cv2.perspectiveTransform(proj_corners.reshape(-1, 1, 2), Hcp).reshape(-1, 2)
    return cam, proj_corners, cam_corners


class TestCheckerboardPipeline(unittest.TestCase):
    def _run(self, Hcp, **kw):
        cam, proj_corners, true_cam = synth_camera(Hcp, **kw)
        found = find_corners(cam, PATTERN)
        self.assertIsNotNone(found, "checkerboard not found")
        self.assertEqual(found.shape, (PATTERN[0] * PATTERN[1], 2))
        ordered = order_like(found, proj_corners)
        H, err = solve_homography(ordered, proj_corners)
        self.assertIsNotNone(H)
        return H, err, proj_corners, true_cam, ordered

    def assertMapsWithin(self, H, true_cam, proj_corners, tol=1.0):
        mapped = cv2.perspectiveTransform(true_cam.reshape(-1, 1, 2), H).reshape(-1, 2)
        e = np.linalg.norm(mapped - proj_corners, axis=1)
        self.assertLess(float(e.max()), tol, f"max projector-px error {e.max():.2f}, mean {e.mean():.2f}")

    def test_recovered_homography_accuracy(self):
        H, err, pc, tc, _ = self._run(true_cam_from_proj())
        self.assertLess(err, 1.0)
        self.assertMapsWithin(H, tc, pc, 1.2)

    def test_several_geometries(self):
        for kw in (dict(scale=0.45, deg=-4.0, tx=250, ty=90), dict(scale=0.6, deg=6.0, tx=120, ty=30, persp=(4e-5, -2e-5)),
                   dict(scale=0.5, deg=0.0, tx=200, ty=80, persp=(0, 0))):
            with self.subTest(**kw):
                H, err, pc, tc, _ = self._run(true_cam_from_proj(**kw), seed=3)
                self.assertMapsWithin(H, tc, pc, 1.2)

    def test_heavier_noise_and_blur(self):
        H, err, pc, tc, _ = self._run(true_cam_from_proj(), noise=8.0, blur=2.0, seed=5)
        self.assertMapsWithin(H, tc, pc, 1.5)

    def test_ordering_ambiguity_reversed(self):
        Hcp = true_cam_from_proj()
        cam, pc, tc = synth_camera(Hcp)
        found = find_corners(cam, PATTERN)
        fixed = order_like(found, pc)
        # OpenCV may return either order; feeding the reversed list must give the same result
        fixed_rev = order_like(found[::-1].copy(), pc)
        np.testing.assert_allclose(fixed, fixed_rev)
        self.assertLess(np.linalg.norm(fixed - tc, axis=1).max(), 2.0)

    def test_order_like_unit(self):
        pc = np.array([[x, y] for y in range(3) for x in range(4)], np.float64) * 100 + 50
        cam = pc * 0.5 + 10
        np.testing.assert_allclose(order_like(cam, pc), cam)
        np.testing.assert_allclose(order_like(cam[::-1].copy(), pc), cam)
        # camera rotated 180 degrees relative to projector can't be fixed by design; make sure it doesn't crash
        order_like(cam[::-1] * -1 + 500, pc)

    def test_checkerboard_corner_positions_match_detection(self):
        # No warp at all: the corner list returned by checkerboard() must be where OpenCV finds the corners.
        img, corners = checkerboard(PROJ, PATTERN, (960, 540), 100)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = find_corners(gray, PATTERN)
        self.assertIsNotNone(found)
        found = order_like(found, corners)
        err = np.linalg.norm(found - corners, axis=1)
        print(f"\n[calib] checkerboard() vs detected corners, no warp: max {err.max():.2f} px, mean {err.mean():.2f} px")
        self.assertLess(err.max(), 1.0)

    # regression test for a bug found during testing (fixed)
    def test_checkerboard_corner_list_has_no_half_pixel_bias(self):
        img, corners = checkerboard(PROJ, PATTERN, (960, 540), 100)
        found = order_like(find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), PATTERN), corners)
        self.assertLess(np.linalg.norm(found - corners, axis=1).max(), 0.3)

    def test_checkerboard_geometry(self):
        img, corners = checkerboard(PROJ, PATTERN, (960, 540), 100)
        self.assertEqual(img.shape, (1080, 1920, 3))
        self.assertEqual(corners.shape, (54, 2))
        np.testing.assert_allclose(corners.mean(axis=0), [960, 540], atol=1.0)  # corners sit on pixel boundaries
        self.assertEqual(int(img[0, 0].max()), 0)
        # corners are ordered row-major
        self.assertLess(corners[0, 0], corners[1, 0])
        self.assertLess(corners[0, 1], corners[PATTERN[0], 1])

    def test_not_found_returns_none(self):
        blank = np.full((720, 1280), 120, np.uint8)
        self.assertIsNone(find_corners(blank, PATTERN))
        self.assertIsNone(find_corners(blank, PATTERN, fast=True))

    def test_solve_homography_degenerate(self):
        H, err = solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))
        self.assertIsNone(H)
        self.assertEqual(err, float("inf"))
        # all-collinear points: must not crash
        pts = np.stack([np.arange(8.0), np.arange(8.0)], 1)
        H, err = solve_homography(pts, pts * 2)
        self.assertTrue(H is None or np.isfinite(err) or err == float("inf"))

    def test_solve_homography_outlier_rejected(self):
        pc = np.array([[x, y] for y in range(6) for x in range(9)], np.float64) * 100 + 400
        Hcp = true_cam_from_proj()
        cam = cv2.perspectiveTransform(pc.reshape(-1, 1, 2), Hcp).reshape(-1, 2)
        cam[7] += [60, -40]  # a misdetected corner
        H, err = solve_homography(cam, pc)
        good = np.delete(np.arange(len(pc)), 7)
        clean = cv2.perspectiveTransform(cam[good].reshape(-1, 1, 2), H).reshape(-1, 2)
        self.assertLess(np.linalg.norm(clean - pc[good], axis=1).max(), 0.5)


class TestDots(unittest.TestCase):
    def _pair(self, pos, radius=9, bright=200, noise=2.0, base=35):
        rng = np.random.RandomState(1)
        dark = np.clip(base + rng.normal(0, noise, (720, 1280, 3)), 0, 255).astype(np.uint8)
        lit = dark.copy()
        cv2.circle(lit, (int(round(pos[0])), int(round(pos[1]))), radius, (bright,) * 3, -1, cv2.LINE_AA)
        lit = cv2.GaussianBlur(lit, (0, 0), 1.0)
        return dark, lit

    def test_centroid_accuracy(self):
        for pos in ((300, 200), (1000, 500), (640.0, 360.0)):
            dark, lit = self._pair(pos)
            c = dot_centroid(dark, lit)
            self.assertIsNotNone(c)
            self.assertLess(np.linalg.norm(c - np.array(pos)), 0.7)

    def test_no_dot_returns_none(self):
        dark, _ = self._pair((100, 100))
        self.assertIsNone(dot_centroid(dark, dark.copy()))
        faint = dark.copy()
        faint[100:110, 100:110] += 10
        self.assertIsNone(dot_centroid(dark, faint))

    def test_tiny_blob_rejected(self):
        dark, lit = self._pair((400, 400), radius=1)
        self.assertIsNone(dot_centroid(dark, lit))

    def test_picks_largest_blob(self):
        dark, lit = self._pair((400, 400), radius=12)
        cv2.circle(lit, (900, 300), 4, (255, 255, 255), -1)
        c = dot_centroid(dark, lit)
        self.assertLess(np.linalg.norm(c - np.array([400, 400])), 1.0)

    def test_ambient_flicker_gain_change(self):
        # whole frame brightens a bit between dark and lit frames (auto-exposure): dot must still be found
        dark, lit = self._pair((700, 300))
        lit = np.clip(lit.astype(np.int16) + 6, 0, 255).astype(np.uint8)
        c = dot_centroid(dark, lit)
        self.assertIsNotNone(c)
        self.assertLess(np.linalg.norm(c - np.array([700, 300])), 1.0)

    def test_dot_homography_roundtrip(self):
        Hcp = true_cam_from_proj()
        targets = dot_grid(PROJ, (960, 540), (1000, 600), (5, 4))
        self.assertEqual(len(targets), 20)
        cams = []
        for p in targets:
            cp = cv2.perspectiveTransform(p.reshape(1, 1, 2), Hcp).reshape(2)
            dark, lit = self._pair(cp)
            c = dot_centroid(dark, lit)
            self.assertIsNotNone(c)
            cams.append(c)
        H, err = solve_homography(np.array(cams), targets)
        self.assertLess(err, 1.0)

    def test_dot_grid_clamped(self):
        pts = dot_grid(PROJ, (30, 30), (400, 400), (3, 3))
        self.assertTrue((pts >= 20).all())
        self.assertTrue((pts[:, 0] <= PROJ[0] - 20).all() and (pts[:, 1] <= PROJ[1] - 20).all())


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        H = true_cam_from_proj()
        c = Calibration(H=H, camera_size=CAM, projector_size=PROJ, error_px=0.4321, method="dots",
                        window={"x": 1920, "y": 0})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calib.json"
            save_calibration(p, c)
            d = load_calibration(p)
            np.testing.assert_allclose(d.H, H, rtol=1e-12)
            self.assertEqual(tuple(d.camera_size), CAM)
            self.assertEqual(tuple(d.projector_size), PROJ)
            self.assertAlmostEqual(d.error_px, 0.432, places=3)
            self.assertEqual(d.method, "dots")
            self.assertEqual(d.window, {"x": 1920, "y": 0})
            self.assertTrue(d.created)
            pts = np.array([[10.0, 20.0], [1000.0, 700.0]])
            np.testing.assert_allclose(d.proj_to_cam(d.cam_to_proj(pts)), pts, atol=1e-6)

    def test_missing_file(self):
        self.assertIsNone(load_calibration("/nonexistent/calib.json"))

    def test_corrupt_files_return_none(self):
        bad_payloads = ["{not json", "", '{"H": [1,2,3]}', '{"nope": 1}', '{"H": "abc"}',
                        json.dumps({"H": [[1, 0, 0], [0, 1, 0]]}), '{"H": [[1,2,3],[4,5,6],[7,8,"x"]]}',
                        "\x00\x01\x02", '{"H": null}']
        with tempfile.TemporaryDirectory() as tmp:
            for i, payload in enumerate(bad_payloads):
                with self.subTest(payload=payload[:30]):
                    p = Path(tmp) / f"c{i}.json"
                    p.write_text(payload)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertIsNone(load_calibration(p))

    # regression test for a bug found during testing (fixed)
    def test_non_object_json_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, payload in enumerate(["[]", "null", "42", '"str"']):
                p = Path(tmp) / f"n{i}.json"
                p.write_text(payload)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(load_calibration(p))

    def test_flat_9_element_H_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps({"H": list(np.eye(3).ravel())}))
            c = load_calibration(p)
            self.assertIsNotNone(c)
            self.assertEqual(tuple(c.camera_size), (1280, 720))

    def test_singular_H_loaded_without_validation(self):
        # Product risk: a zero/singular H loads fine and only fails later (proj_to_cam raises LinAlgError).
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            p.write_text(json.dumps({"H": np.zeros((3, 3)).tolist()}))
            c = load_calibration(p)
            print(f"\n[calib] singular H accepted by load_calibration: {c is not None}")


if __name__ == "__main__":
    unittest.main()
