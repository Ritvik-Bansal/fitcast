import contextlib
import io
import shutil
import statistics
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    from _warp_helpers import (SIZE, VIEWS, Layer, SideAssigner, ViewWarper, alpha_bbox, cfg_and_catalog,
                               kp_bbox, make_body, render, solid_warper, swap_labels, warpers)
except ImportError:  # run as tests.test_warp
    from tests._warp_helpers import (SIZE, VIEWS, Layer, SideAssigner, ViewWarper, alpha_bbox, cfg_and_catalog,
                                     kp_bbox, make_body, render, solid_warper, swap_labels, warpers)
from tryon.garments import load_garment
from tryon.smoothing import Body

MODES = {"mesh", "similarity", None}
NORMAL_POSES = ["neutral", "t", "overhead", "crossed", "bent", "offscreen", "novis"]


def _in_bounds(layer):
    x0, y0, x1, y1 = layer.roi
    h, w = layer.img.shape[:2]
    return 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h


class TestWarpPoses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = warpers()

    def _check(self, body, pose, expect_mesh=True, expect_nonempty=True, tag=""):
        kb = kp_bbox(body, names=[k for k in body.kp if "wrist" not in k] if pose == "offscreen" else None)
        cx, cy = (kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2
        bw, bh = (kb[2] - kb[0]), (kb[3] - kb[1])
        for (gid, view), (g, w) in self.W.items():
            with self.subTest(pose=pose, garment=gid, view=view, tag=tag):
                layer, mode = render(w, body)
                self.assertIn(mode, MODES)
                self.assertTrue(_in_bounds(layer), layer.roi)
                bb = alpha_bbox(layer)
                if expect_nonempty:
                    self.assertIsNotNone(bb, "empty layer for a normal pose")
                    self.assertIsNotNone(mode)
                if mode is None:
                    continue
                if expect_mesh:
                    self.assertEqual(mode, "mesh" if view in ("front", "back") else "similarity")
                if bb is not None:
                    # alpha must live inside roi
                    rx0, ry0, rx1, ry1 = layer.roi
                    self.assertTrue(bb[0] >= rx0 and bb[1] >= ry0 and bb[2] <= rx1 and bb[3] <= ry1)
                    # no explosion: within 2.5x the keypoint bbox (about its centre); sleeves may reach out
                    # to the wrists so the bbox already includes them, except for the off-screen wrist case.
                    lim_w, lim_h = 2.5 * max(bw, 40), 2.5 * max(bh, 40)
                    if pose == "offscreen":  # wrist at +-5000 px: sleeve is clamped by max_limb_stretch only
                        lim_w = 2.5 * max(bw, bh)
                    self.assertLessEqual(bb[2] - bb[0], lim_w, f"{bb} vs body {kb}")
                    self.assertLessEqual(bb[3] - bb[1], lim_h, f"{bb} vs body {kb}")
                    self.assertGreaterEqual(bb[0], cx - lim_w / 2)
                    self.assertLessEqual(bb[2], cx + lim_w / 2)
                    self.assertGreaterEqual(bb[1], cy - lim_h / 2)
                    self.assertLessEqual(bb[3], cy + lim_h / 2)

    def test_normal_poses_front_facing(self):
        for pose in NORMAL_POSES:
            self._check(make_body(pose), pose)

    def test_normal_poses_back_facing_correct_labels(self):
        for pose in NORMAL_POSES:
            self._check(make_body(pose, facing="back"), pose, tag="back")

    def test_back_facing_labels_swapped(self):
        for pose in ("neutral", "t", "overhead"):
            self._check(make_body(pose, facing="back", swapped=True), pose, tag="back-swapped")

    def test_side_on_shoulders_overlapping(self):
        # Views are bucketed by orientation.py, but the warper must survive any view with overlapping shoulders.
        self._check(make_body("sideon"), "sideon", expect_mesh=False, expect_nonempty=False)

    # regression test for a bug found during testing (fixed)
    def test_pants_legs_not_truncated_when_hips_narrow(self):
        body = make_body("sideon")
        w = warpers()[("pants_01_indigo_jeans", "front")][1]
        layer, mode = render(w, body)
        bb = alpha_bbox(layer, 20)
        leg_len = body.kp["left_ankle"][1] - body.kp["left_hip"][1]
        self.assertGreater(bb[3] - bb[1], 0.9 * leg_len, f"pants height {bb[3] - bb[1]} vs hip-ankle {leg_len}")

    def test_tiny_person(self):
        self._check(make_body("neutral", top=300, s=100 / 650), "tiny")

    def test_half_out_of_frame(self):
        body = make_body("neutral", top=430)  # hips at y~680+, knees/ankles far below the frame
        self._check(body, "halfout", expect_mesh=False, expect_nonempty=False)

    def test_far_below_frame_and_negative(self):
        for top in (700, -400):
            self._check(make_body("neutral", top=top), "offframe", expect_mesh=False, expect_nonempty=False)

    def test_degenerate_coincident_joints(self):
        for facing in ("front", "back"):
            self._check(make_body("degenerate", facing=facing), "degenerate", expect_mesh=False, expect_nonempty=False)

    def test_all_zero_keypoints(self):
        kp = {k: np.zeros(2, np.float32) for k in make_body().kp}
        body = Body(kp=kp, vis={k: 1.0 for k in kp}, size=SIZE, t=0.0)
        for (gid, view), (g, w) in self.W.items():
            layer, mode = render(w, body)
            self.assertIn(mode, MODES)

    def test_nan_keypoints_do_not_crash(self):
        body = make_body("neutral")
        body.kp["left_wrist"] = np.array([np.nan, np.nan], np.float32)
        body.kp["left_knee"] = np.array([np.inf, 0], np.float32)
        for (gid, view), (g, w) in self.W.items():
            with self.subTest(garment=gid, view=view):
                layer, mode = render(w, body)
                self.assertIn(mode, MODES)
                self.assertTrue(_in_bounds(layer))

    def test_missing_keys(self):
        body = make_body("neutral")
        for k in ("left_hip", "right_knee", "left_elbow"):
            body.kp.pop(k)
        for (gid, view), (g, w) in self.W.items():
            try:
                layer, mode = render(w, body, assigner=None)
            except KeyError as e:  # SideAssigner.assign requires shoulders/hips; documents behaviour
                self.assertIn("hip", str(e))
                return
            self.assertIn(mode, MODES)

    def test_roi_matches_clear(self):
        w = self.W[("shirt_01_crimson_tee", "front")][1]
        layer, _ = render(w, make_body("t"))
        self.assertTrue(layer.img.any())
        layer.clear()
        self.assertFalse(layer.img.any(), "Layer.clear left pixels outside the roi")


class TestMirroring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = warpers()

    def test_label_swap_identical_all_views(self):
        for facing in ("front", "back"):
            for pose in ("neutral", "t", "crossed"):
                a = make_body(pose, facing=facing)
                b = make_body(pose, facing=facing, swapped=True)
                for (gid, view), (g, w) in self.W.items():
                    with self.subTest(facing=facing, pose=pose, garment=gid, view=view):
                        la, ma = render(w, a)
                        lb, mb = render(w, b)
                        self.assertEqual(ma, mb)
                        self.assertTrue(np.array_equal(la.img, lb.img), "label swap changed the render")

    def test_label_swap_back_view_on_back_body(self):
        # Spec: back view of a back-facing body, labels "correct" (left_* image-left) vs swapped.
        for gid in ("shirt_01_crimson_tee", "shirt_02_ocean_longsleeve", "pants_01_indigo_jeans"):
            w = self.W[(gid, "back")][1]
            la, _ = render(w, make_body("t", facing="back"))
            lb, _ = render(w, make_body("t", facing="back", swapped=True))
            self.assertTrue(np.array_equal(la.img, lb.img))

    def test_label_swap_midway_sticky_assigner(self):
        # MediaPipe flips labels between frames: with one persistent SideAssigner the result must not change.
        w = self.W[("shirt_02_ocean_longsleeve", "back")][1]
        asg = SideAssigner()
        ref, _ = render(w, make_body("neutral", facing="back"), assigner=asg)
        for swapped in (True, False, True):
            lay, _ = render(w, make_body("neutral", facing="back", swapped=swapped), assigner=asg)
            self.assertTrue(np.array_equal(ref.img, lay.img))

    def _marker_garment(self, tmp):
        """Copy shirt_01 with a green patch at the back view's anchor left_hem and blue at right_hem."""
        cfg, cat = cfg_and_catalog()
        src = Path(cfg.path(cfg.garments_dir)) / "shirt_01_crimson_tee"
        dst = Path(tmp) / "marker_shirt"
        shutil.copytree(src, dst)
        import json
        anchors = json.loads((dst / "anchors.json").read_text())
        for view in ("front", "back"):
            img = cv2.imread(str(dst / f"{view}.png"), cv2.IMREAD_UNCHANGED)
            a = anchors[view]
            for name, col in (("left_hem", (0, 255, 0, 255)), ("right_hem", (255, 0, 0, 255))):
                x, y = a[name]
                cv2.circle(img, (int(x), int(y) - 60), 35, col, -1)
            cv2.imwrite(str(dst / f"{view}.png"), img)
        with contextlib.redirect_stdout(io.StringIO()):
            g = load_garment(dst, 1024)
        return g, anchors

    def test_back_view_left_marker_on_image_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, anchors = self._marker_garment(tmp)
            self.assertLess(anchors["back"]["left_hem"][0], anchors["back"]["right_hem"][0],
                            "placeholder back anchors: wearer-left hem must be image-left")
            cfg, _ = cfg_and_catalog()
            with contextlib.redirect_stdout(io.StringIO()):
                w = ViewWarper(g.views["back"], g.category, g.sleeve_length, cfg)
            for swapped in (False, True):
                body = make_body("neutral", facing="back", swapped=swapped)
                layer, mode = render(w, body)
                self.assertEqual(mode, "mesh")
                img = layer.img
                green = (img[..., 1] > 150) & (img[..., 0] < 80) & (img[..., 2] < 80)
                blue = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
                self.assertGreater(green.sum(), 200)
                self.assertGreater(blue.sum(), 200)
                gx = np.nonzero(green)[1].mean()
                bx = np.nonzero(blue)[1].mean()
                self.assertLess(gx, 640, f"swapped={swapped}: wearer-left (green) marker not on image-left")
                self.assertGreater(bx, 640)

    def test_front_view_left_marker_on_image_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self._marker_garment(tmp)
            cfg, _ = cfg_and_catalog()
            with contextlib.redirect_stdout(io.StringIO()):
                w = ViewWarper(g.views["front"], g.category, g.sleeve_length, cfg)
            for swapped in (False, True):
                layer, _ = render(w, make_body("neutral", facing="front", swapped=swapped))
                img = layer.img
                green = (img[..., 1] > 150) & (img[..., 0] < 80) & (img[..., 2] < 80)
                self.assertGreater(green.sum(), 200)
                self.assertGreater(np.nonzero(green)[1].mean(), 640)

    def test_real_placeholder_L_on_back_is_on_wearer_left(self):
        # The white "L" glyph in the real back.png must land on image-left of a back-facing body.
        w = self.warpers_back = warpers()[("shirt_01_crimson_tee", "back")]
        g, ww = w
        img = g.views["back"].image
        white = (img[..., :3].min(axis=2) > 235) & (img[..., 3] > 200)
        ys, xs = np.nonzero(white)
        self.assertGreater(len(xs), 50, "no white L marker found in back.png")
        self.assertLess(xs.mean(), img.shape[1] / 2, "'L' marker sits on the image-right half of back.png")
        layer, _ = render(ww, make_body("neutral", facing="back"))
        wm = (layer.img[..., :3].min(axis=2) > 200) & (layer.img[..., 3] > 200)
        self.assertGreater(wm.sum(), 30)
        self.assertLess(np.nonzero(wm)[1].mean(), 640)


class TestLimbs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = warpers()

    def test_long_sleeve_follows_wrist(self):
        w = self.W[("shirt_02_ocean_longsleeve", "front")][1]
        b0 = make_body("neutral")
        b1 = make_body("neutral")
        sh = b1.kp["left_shoulder"]  # image-right arm
        new_wrist = sh + np.array([120, 70], np.float32)
        b1.kp["left_wrist"] = new_wrist.astype(np.float32)
        b1.kp["left_elbow"] = (sh + np.array([60, 40], np.float32)).astype(np.float32)
        l0, _ = render(w, b0)
        l1, _ = render(w, b1)
        a0, a1 = l0.img[..., 3].astype(int), l1.img[..., 3].astype(int)
        x, y = int(new_wrist[0]), int(new_wrist[1])
        self.assertLess(a0[y - 8:y + 8, x - 8:x + 8].mean(), 20, "sleeve already at new wrist spot")
        self.assertGreater(a1[y - 8:y + 8, x - 8:x + 8].mean(), 100, "sleeve did not follow the wrist")
        # old wrist location (hanging down) is now empty
        ow = b0.kp["left_wrist"]
        ox, oy = int(ow[0]), int(ow[1])
        self.assertGreater(a0[oy - 8:oy + 8, ox - 8:ox + 8].mean(), 100)
        self.assertLess(a1[oy - 8:oy + 8, ox - 8:ox + 8].mean(), 60)
        # torso interior (between the armpits, below the collar) identical; opposite arm unchanged
        cx = 640
        torso = (slice(int(b0.kp["left_shoulder"][1]) + 60, int(b0.kp["left_hip"][1]) - 20), slice(cx - 30, cx + 30))
        self.assertTrue(np.array_equal(l0.img[torso], l1.img[torso]))
        far_left = (slice(0, 720), slice(0, cx - 100))
        self.assertTrue(np.array_equal(l0.img[far_left], l1.img[far_left]))

    def test_short_sleeve_follows_elbow(self):
        w = self.W[("shirt_01_crimson_tee", "front")][1]
        b0, b1 = make_body("neutral"), make_body("neutral")
        sh = b1.kp["right_shoulder"]
        b1.kp["right_elbow"] = (sh + np.array([-100, 20], np.float32)).astype(np.float32)
        l0, _ = render(w, b0)
        l1, _ = render(w, b1)
        x, y = int(b1.kp["right_elbow"][0]), int(b1.kp["right_elbow"][1])
        self.assertLess(l0.img[y - 8:y + 8, x - 8:x + 8, 3].mean(), 20)
        self.assertGreater(l1.img[y - 8:y + 8, x - 8:x + 8, 3].mean(), 60)

    def test_pants_follow_ankle(self):
        w = self.W[("pants_01_indigo_jeans", "front")][1]
        b0, b1 = make_body("neutral"), make_body("neutral")
        b1.kp["left_ankle"] = np.array([b0.kp["left_ankle"][0] + 150, b0.kp["left_ankle"][1] - 60], np.float32)
        l0, _ = render(w, b0)
        l1, _ = render(w, b1)
        x, y = int(b1.kp["left_ankle"][0]), int(b1.kp["left_ankle"][1])
        self.assertLess(l0.img[y - 6:y + 6, x - 6:x + 6, 3].mean(), 20)
        self.assertGreater(l1.img[y - 6:y + 6, x - 6:x + 6, 3].mean(), 60)


class TestSeams(unittest.TestCase):
    """Seam tests use a copy of each garment whose alpha is filled solid (the placeholder art has
    anti-aliased text cut-outs that would look like holes)."""

    @staticmethod
    def _holes(alpha, poly, min_alpha=200):
        mask = np.zeros(alpha.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        inside = mask > 0
        return int((alpha[inside] < min_alpha).sum()), int(inside.sum())

    def test_shirt_torso_has_no_holes(self):
        body = make_body("neutral")
        for gid in ("shirt_01_crimson_tee", "shirt_02_ocean_longsleeve", "shirt_03_lime_stripe_tee"):
            for view in ("front", "back"):
                with self.subTest(garment=gid, view=view):
                    w = solid_warper(gid, view)
                    names = SideAssigner().assign(body.kp)
                    t = w.targets(body, names)
                    layer, mode = render(w, body)
                    self.assertEqual(mode, "mesh")
                    T = t["T"]
                    shrink = lambda p: T + (p - T) * 0.92  # noqa: E731
                    quad = np.array([shrink(t[k]) for k in ("Sl", "Sr", "Hr", "Hl")])
                    n_bad, n = self._holes(layer.img[..., 3], quad)
                    self.assertGreater(n, 5000)
                    self.assertEqual(n_bad, 0, f"{n_bad}/{n} low-alpha pixels inside torso quad")
                    a = layer.img[..., 3]
                    ys = np.arange(int(quad[:, 1].min()) + 3, int(quad[:, 1].max()) - 3)
                    x0, x1 = int(T[0]) - 20, int(T[0]) + 20
                    self.assertTrue((a[ys, x0:x1].min(axis=1) > 200).all())

    def test_shirt_torso_holes_under_bent_and_raised_arms(self):
        for pose in ("t", "overhead", "crossed", "bent"):
            body = make_body(pose)
            for gid in ("shirt_01_crimson_tee", "shirt_02_ocean_longsleeve"):
                with self.subTest(pose=pose, garment=gid):
                    w = solid_warper(gid, "front")
                    t = w.targets(body, SideAssigner().assign(body.kp))
                    layer, _ = render(w, body)
                    T = t["T"]
                    quad = np.array([T + (t[k] - T) * 0.9 for k in ("Al", "Ar", "Hr", "Hl")])
                    n_bad, n = self._holes(layer.img[..., 3], quad)
                    self.assertEqual(n_bad, 0)

    def test_pants_legs_have_no_holes(self):
        body = make_body("neutral")
        for gid in ("pants_01_indigo_jeans", "pants_02_khaki_chinos"):
            for view in ("front", "back"):
                with self.subTest(garment=gid, view=view):
                    w = solid_warper(gid, view)
                    layer, mode = render(w, body)
                    self.assertEqual(mode, "mesh")
                    a = layer.img[..., 3]
                    for side in ("left", "right"):
                        hip, ank = body.kp[f"{side}_hip"], body.kp[f"{side}_ankle"]
                        knee = body.kp[f"{side}_knee"]
                        x = int(knee[0])
                        ys = np.arange(int(hip[1]) + 30, int(ank[1]) - 10)
                        col = a[ys, x]
                        self.assertTrue((col > 200).all(), f"hole in {side} leg centre line at y={ys[col <= 200][:5]}")
                        ky = int(knee[1])
                        band = a[ky - 6:ky + 6, x - 8:x + 8]
                        self.assertTrue((band > 200).all(), "gap at the knee hinge")

    def test_pants_bent_knee_no_gap(self):
        body = make_body("neutral")
        body.kp["left_knee"] = body.kp["left_knee"] + np.array([40, -60], np.float32)
        body.kp["left_ankle"] = body.kp["left_ankle"] + np.array([20, -150], np.float32)
        w = solid_warper("pants_01_indigo_jeans", "front")
        layer, _ = render(w, body)
        a = layer.img[..., 3]
        hip, knee, ank = (body.kp[k] for k in ("left_hip", "left_knee", "left_ankle"))
        for p, q in ((hip, knee), (knee, ank)):
            for f in np.linspace(0.3, 0.95, 10):
                x, y = (p + (q - p) * f).astype(int)
                self.assertGreater(a[y - 2:y + 3, x - 2:x + 3].max(), 200, f"leg gap near {(x, y)}")

    # regression test for a bug found during testing (fixed)
    def test_pants_crotch_no_hole_above_crotch_point(self):
        body = make_body("neutral")
        w = solid_warper("pants_01_indigo_jeans", "front")
        t = w.targets(body, SideAssigner().assign(body.kp))
        layer, _ = render(w, body)
        a = layer.img[..., 3]
        x = int(round((t["Wl"][0] + t["Wr"][0]) / 2))
        ys = np.arange(int(t["Wl"][1]) + 3, int(t["X"][1]) - 3)
        col = a[ys, x]
        holes = ys[col <= 200]
        self.assertEqual(holes.size, 0, f"hole above crotch at x={x}, y={holes.min() if holes.size else 0}..{holes.max() if holes.size else 0}")

    # regression test for a bug found during testing (fixed)
    def test_pants_nothing_drawn_above_waistband(self):
        body = make_body("neutral")
        w = solid_warper("pants_01_indigo_jeans", "front")
        t = w.targets(body, SideAssigner().assign(body.kp))
        layer, _ = render(w, body)
        top = alpha_bbox(layer, 20)[1]
        waist_y = min(t["Wl"][1], t["Wr"][1])
        self.assertGreaterEqual(top, waist_y - 0.05 * t["torso_len"], f"pants drawn {waist_y - top:.0f}px above waistband")

    def test_bent_elbow_no_gap_in_long_sleeve(self):
        w = solid_warper("shirt_02_ocean_longsleeve", "front")
        body = make_body("bent")
        layer, _ = render(w, body)
        a = layer.img[..., 3]
        sh, el, wr = (body.kp[k] for k in ("left_shoulder", "left_elbow", "left_wrist"))
        for p, q in ((sh, el), (el, wr)):
            for f in np.linspace(0.2, 0.9, 12):
                x, y = (p + (q - p) * f).astype(int)
                self.assertGreater(a[y - 2:y + 3, x - 2:x + 3].max(), 200, f"sleeve gap near {(x, y)}")


class TestPerformance(unittest.TestCase):
    def test_render_time_front_shirt_and_pants(self):
        W = warpers()
        ws = [W[("shirt_01_crimson_tee", "front")][1], W[("pants_01_indigo_jeans", "front")][1]]
        body = make_body("neutral")
        names = SideAssigner().assign(body.kp)
        layer = Layer.empty(SIZE)
        for _ in range(20):
            layer.clear()
            for w in ws:
                w.render(layer, body, names)
        ts = []
        for _ in range(200):
            layer.clear()
            t = time.perf_counter()
            for w in ws:
                w.render(layer, body, names)
            ts.append((time.perf_counter() - t) * 1000)
        med = statistics.median(ts)
        print(f"\n[perf] warp shirt+pants front: median {med:.2f} ms, p95 {sorted(ts)[int(0.95 * len(ts))]:.2f} ms")
        self.assertLess(med, 8.0)


class TestSideAssigner(unittest.TestCase):
    def test_sticky_within_margin(self):
        b = make_body("neutral")
        asg = SideAssigner()
        asg.assign(b.kp)
        self.assertFalse(asg.upper_swap)
        # shoulders cross by less than the margin: keep decision
        b.kp["left_shoulder"] = np.array([638, 120], np.float32)
        b.kp["right_shoulder"] = np.array([642, 120], np.float32)
        asg.assign(b.kp)
        self.assertFalse(asg.upper_swap)
        b.kp["left_shoulder"] = np.array([600, 120], np.float32)
        b.kp["right_shoulder"] = np.array([680, 120], np.float32)
        names = asg.assign(b.kp)
        self.assertTrue(asg.upper_swap)
        self.assertEqual(names["il_shoulder"], "left_shoulder")


if __name__ == "__main__":
    unittest.main()
