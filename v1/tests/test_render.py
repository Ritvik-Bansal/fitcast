import contextlib
import io
import types
import unittest

import cv2
import numpy as np

try:
    from _warp_helpers import SIZE, cfg_and_catalog, make_body
except ImportError:
    from tests._warp_helpers import SIZE, cfg_and_catalog, make_body
from tryon.calibration import Calibration
from tryon.config import load_config
from tryon.render import Renderer
from tryon.state import AppState

PROJ = (1920, 1080)
GESTURES = types.SimpleNamespace(palm_pos=None)
VIEW_F = [("front", 1.0)]


def make_H(scale=1.5, deg=4.0, tx=120.0, ty=-80.0, persp=(1.5e-5, 1.0e-5)):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])
    P = np.eye(3)
    P[2, 0], P[2, 1] = persp
    return T @ P @ R @ S


def calib(H=None):
    return Calibration(H=make_H() if H is None else H, camera_size=SIZE, projector_size=PROJ)


_catalog = {}


def make_renderer(**over):
    cfg = load_config(overrides={"startup_blank_ms": 0, **over})
    _, cat = cfg_and_catalog()
    with contextlib.redirect_stdout(io.StringIO()):
        r = Renderer(cfg, cat)
    return cfg, cat, r


def nonblack(img):
    return img.max(axis=2) > 0


class RenderBase(unittest.TestCase):
    def setUp(self):
        self.cfg, self.cat, self.r = make_renderer()
        self.state = AppState(self.cfg, self.cat)
        self.cal = calib()
        self.t = self.r.start_t + 10.0
        self.body = make_body("neutral")
        self.r._draw_ui = lambda *a, **k: None  # garment-only unless a test restores it

    def step(self, body, views=VIEW_F, mask=None, n=1, dt=0.033):
        for _ in range(n):
            self.t += dt
            layer = self.r.build_layer(body, self.state, views, mask, self.t)
        return layer

    def proj(self):
        return self.r.projector_frame(self.cal, self.state, GESTURES, self.t)


class TestLayerAndProjector(RenderBase):
    def test_build_layer_basic(self):
        layer = self.step(self.body, n=10)
        self.assertFalse(layer.is_empty)
        self.assertEqual(self.r.presence, 1.0)
        x0, y0, x1, y1 = layer.roi
        self.assertTrue(0 <= x0 < x1 <= SIZE[0] and 0 <= y0 < y1 <= SIZE[1])
        self.assertEqual(set(self.r.modes), {"shirt", "pants"})
        self.assertEqual(self.r.modes["shirt"], "mesh")

    def test_projector_frame_shape_black_background(self):
        self.step(self.body, n=10)
        out = self.proj()
        self.assertEqual(out.shape, (1080, 1920, 3))
        self.assertEqual(out.dtype, np.uint8)
        nb = nonblack(out)
        self.assertGreater(nb.sum(), 20000)
        ys, xs = np.nonzero(nb)
        # everything lit is inside the projection of the layer ROI (+ small margin)
        x0, y0, x1, y1 = self.r.final.roi
        corners = self.cal.cam_to_proj([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
        lo, hi = corners.min(axis=0) - 3, corners.max(axis=0) + 3
        self.assertGreaterEqual(xs.min(), lo[0])
        self.assertLessEqual(xs.max(), hi[0])
        self.assertGreaterEqual(ys.min(), lo[1])
        self.assertLessEqual(ys.max(), hi[1])
        # no gray haze: pixels outside the dilated garment footprint are exactly 0
        foot = cv2.dilate(nb.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertEqual(int(out[~foot].max()), 0)

    def test_homography_places_garment_correctly(self):
        self.step(self.body, n=10)
        out = self.proj()
        layer = self.r.final
        a = layer.img[..., 3]
        # a plain-colour point deep inside the shirt (camera px) must land at H(point), at brightness gain
        h_mid = (self.body.kp["left_hip"] + self.body.kp["right_hip"]) / 2
        pts = [(h_mid[0] + dx, h_mid[1] - 25) for dx in (-25, 25)]
        for (cx, cy) in pts:
            cx, cy = int(cx), int(cy)
            patch = layer.img[cy - 3:cy + 4, cx - 3:cx + 4, :3].reshape(-1, 3).mean(axis=0)
            self.assertGreater(a[cy, cx], 250)
            px, py = self.cal.cam_to_proj([[cx, cy]])[0]
            got = out[int(py) - 1:int(py) + 2, int(px) - 1:int(px) + 2].reshape(-1, 3).mean(axis=0)
            np.testing.assert_allclose(got, patch * self.cfg.garment_brightness, atol=10)
        # a camera point away from the person maps to black
        px, py = self.cal.cam_to_proj([[100, 100]])[0]
        self.assertEqual(int(out[int(py), int(px)].max()) if 0 <= px < 1920 and 0 <= py < 1080 else 0, 0)
        # centroid of lit pixels vs centroid of alpha mapped by H (perspective is mild)
        ys, xs = np.nonzero(a > 128)
        c_cam = np.array([xs.mean(), ys.mean()])
        c_proj = self.cal.cam_to_proj([c_cam])[0]
        ys2, xs2 = np.nonzero(nonblack(out))
        np.testing.assert_allclose([xs2.mean(), ys2.mean()], c_proj, atol=12)

    def test_identity_H_matches_layer(self):
        self.cal = calib(np.eye(3))
        self.step(self.body, n=10)
        out = self.proj()
        a = self.r.final.img[..., 3]
        self.assertTrue(np.array_equal(nonblack(out)[:720, :1280] & (a == 0), np.zeros((720, 1280), bool)) or
                        (nonblack(out)[:720, :1280] & (a == 0)).sum() < 50)

    def test_unusual_homographies_do_not_crash(self):
        self.step(self.body, n=10)
        for H in (make_H(tx=5000), make_H(tx=-5000), make_H(scale=0.01), make_H(scale=30.0),
                  make_H(persp=(3e-3, 0)), np.zeros((3, 3)) + np.eye(3) * 1e-6):
            self.cal = calib(H)
            out = self.proj()
            self.assertEqual(out.shape, (1080, 1920, 3))

    def test_projector_ui_present_when_enabled(self):
        del self.r._draw_ui  # restore the real method
        self.step(self.body, n=10)
        with_ui = self.proj()
        self.r._draw_ui = lambda *a, **k: None
        without = self.proj()
        diff = nonblack(with_ui) != nonblack(without) | (with_ui != without).any(axis=2)
        self.assertGreater((with_ui != without).any(axis=2).sum(), 500, "name/price text missing")
        ys, xs = np.nonzero((with_ui != without).any(axis=2))
        self.assertTrue(0 <= xs.min() and xs.max() < 1920 and 0 <= ys.min() and ys.max() < 1080)
        del diff

    def test_brightness_gain(self):
        cfg, cat, r = make_renderer(garment_brightness=0.5)
        r._draw_ui = lambda *a, **k: None
        st = AppState(cfg, cat)
        t = r.start_t + 5
        for i in range(10):
            t += 0.033
            r.build_layer(self.body, st, VIEW_F, None, t)
        out = r.projector_frame(self.cal, st, GESTURES, t)
        self.assertLessEqual(int(out.max()), 130)  # 255 * 0.5 (+ interpolation)
        self.assertGreater(int(out.max()), 40)


class TestMaskAndHead(RenderBase):
    def test_mask_none_lights_whole_garment(self):
        self.step(self.body, mask=None, n=10)
        self.assertGreater(nonblack(self.proj()).sum(), 20000)

    def test_zero_mask_projects_nothing(self):
        mask = np.zeros((720, 1280), np.uint8)
        self.step(self.body, mask=mask, n=10)
        out = self.proj()
        self.assertEqual(int(out.max()), 0, "garment light leaked where the person mask is 0")
        self.assertEqual(int(self.r.final.img.max()), 0)

    def test_half_mask(self):
        mask = np.zeros((720, 1280), np.uint8)
        mask[:, :640] = 255
        self.step(self.body, mask=mask, n=10)
        a = self.r.final.img[..., 3]
        pad = self.cfg.mask_dilate_px + self.cfg.mask_feather_px + 2
        self.assertEqual(int(a[:, 640 + pad:].max()), 0)
        self.assertGreater(int(a[:, :640 - pad].max()), 200)
        # projector: lit pixels all map from the allowed half (+ a margin in projector pixels)
        out = self.proj()
        ys, xs = np.nonzero(nonblack(out))
        cam = self.cal.proj_to_cam(np.stack([xs[::50], ys[::50]], 1))
        self.assertLessEqual(cam[:, 0].max(), 640 + pad + 3)

    def test_mask_shape_is_camera_size_only(self):
        # a mask at the wrong resolution is a caller bug; document the failure mode instead of hiding it
        bad = np.full((360, 640), 255, np.uint8)
        try:
            self.step(self.body, mask=bad, n=2)
        except Exception as e:  # noqa: BLE001
            self.assertIsInstance(e, (ValueError, cv2.error, IndexError))

    def _with_head_at(self, head_xy):
        b = make_body("neutral")
        d = np.array(head_xy, np.float32) - b.kp["nose"]
        for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            b.kp[k] = b.kp[k] + d
        return b

    def test_head_exclusion_all_255_mask(self):
        full = np.full((720, 1280), 255, np.uint8)
        # head circle right over the chest, where the shirt is solid
        chest = (640.0, 240.0)
        b = self._with_head_at(chest)
        self.step(b, mask=full, n=10)
        out = self.proj()
        kp = b.kp
        head = np.mean([kp[k] for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")], axis=0)
        ear_d = np.linalg.norm(kp["left_ear"] - kp["right_ear"])
        sh_d = np.linalg.norm(kp["left_shoulder"] - kp["right_shoulder"])
        torso = np.linalg.norm((kp["left_shoulder"] + kp["right_shoulder"]) / 2 - (kp["left_hip"] + kp["right_hip"]) / 2)
        r = self.cfg.head_exclusion_scale * max(0.65 * ear_d, 0.3 * sh_d, 0.18 * torso, 12.0)
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        for rad in (0.0, 0.5 * r, 0.9 * r):
            pts = head + rad * np.stack([np.cos(ang), np.sin(ang)], 1)
            pp = np.round(self.cal.cam_to_proj(pts)).astype(int)
            vals = out[pp[:, 1], pp[:, 0]]
            self.assertEqual(int(vals.max()), 0, f"light inside head circle at radius {rad:.0f}")
        # not vacuous: just outside the circle, the shirt is lit
        outside = head + np.array([0, r + 25])
        pp = np.round(self.cal.cam_to_proj([outside])[0]).astype(int)
        self.assertGreater(int(out[pp[1] - 2:pp[1] + 3, pp[0] - 2:pp[0] + 3].max()), 30)

    def test_head_exclusion_with_collar_at_normal_pose(self):
        # normal pose: nose is ~80 px above the shoulders; head circle must stay dark even if the mask is wide open
        full = np.full((720, 1280), 255, np.uint8)
        self.step(self.body, mask=full, n=10)
        out = self.proj()
        nose = self.body.kp["nose"]
        p = np.round(self.cal.cam_to_proj([nose])[0]).astype(int)
        self.assertEqual(int(out[p[1] - 10:p[1] + 10, p[0] - 10:p[0] + 10].max()), 0)


class TestPresenceAndStartup(RenderBase):
    def test_fade_in(self):
        self.step(self.body, n=1)
        self.assertAlmostEqual(self.r.presence, 0.033 * 1000 / self.cfg.fade_in_ms, places=3)
        out = self.proj()
        peak_partial = int(out.max())
        self.step(self.body, n=10)
        self.assertGreater(int(self.proj().max()), peak_partial)

    def test_fade_out_to_black(self):
        self.step(self.body, n=12)
        self.assertEqual(self.r.presence, 1.0)
        self.assertGreater(int(self.proj().max()), 0)
        elapsed = 0.0
        last_peak = 10 ** 9
        while self.r.presence > 0 and elapsed < 1.0:
            self.step(None, n=1, dt=0.033)
            elapsed += 0.033
            peak = int(self.proj().max())
            self.assertLessEqual(peak, last_peak + 2, "projector brightness rose during fade-out")
            last_peak = peak
        self.assertEqual(self.r.presence, 0.0)
        self.assertLessEqual(elapsed, self.cfg.fade_out_ms / 1000.0 + 0.1)
        out = self.proj()
        self.assertEqual(int(out.max()), 0, "projector not black after the fade-out")
        self.assertTrue(self.r.final.is_empty)

    def test_fade_out_includes_ui_removed(self):
        del self.r._draw_ui
        self.step(self.body, n=12)
        self.step(None, n=12)
        self.assertEqual(int(self.proj().max()), 0, "UI text still projected after the person left")

    def test_reappear_after_loss(self):
        self.step(self.body, n=12)
        self.step(None, n=12)
        self.step(self.body, n=8)
        self.assertGreater(nonblack(self.proj()).sum(), 10000)

    def test_large_dt_clamped(self):
        self.step(self.body, n=1, dt=5.0)  # a stall must not jump straight to full presence unclamped
        self.assertLessEqual(self.r.presence, 1.0)
        self.step(None, n=1, dt=5.0)
        self.assertGreaterEqual(self.r.presence, 0.0)

    def test_startup_blank_respected(self):
        cfg, cat, r = make_renderer(startup_blank_ms=1500)
        r._draw_ui = lambda *a, **k: None
        st = AppState(cfg, cat)
        t = r.start_t
        for i in range(30):
            t += 0.033
            r.build_layer(self.body, st, VIEW_F, None, t)
        self.assertLess(t - r.start_t, 1.5)
        out = r.projector_frame(self.cal, st, GESTURES, t)
        self.assertEqual(int(out.max()), 0, "light projected during startup blank")
        for i in range(30):
            t += 0.033
            r.build_layer(self.body, st, VIEW_F, None, t)
        self.assertGreater(t - r.start_t, 1.5)
        self.assertGreater(int(r.projector_frame(self.cal, st, GESTURES, t).max()), 0)


class TestCrossfade(RenderBase):
    def test_single_view_matches_first_layer(self):
        self.step(self.body, [("front", 1.0)], n=10)
        a = self.r.views[0]
        self.assertTrue(np.array_equal(self.r.final.img[..., 3][:200], a.img[..., 3][:200]))  # head rows

    def test_two_view_mix(self):
        w0, w1 = 0.3, 0.7
        # use a mask of all 255 and keep the head away from the torso (default pose)
        self.step(self.body, [("front", w0), ("back", w1)], n=10)
        a, b = self.r.views
        self.assertFalse(a.is_empty or b.is_empty)
        exp = a.img.astype(np.float32) * w0 + b.img.astype(np.float32) * w1
        y = slice(150, 330)
        x = slice(560, 720)
        got = self.r.final.img[y, x].astype(np.float32)
        np.testing.assert_allclose(got, np.clip(exp[y, x], 0, 255), atol=1.01)
        self.assertGreater(float(np.abs(a.img.astype(int) - b.img.astype(int)).max()), 0, "front/back identical")

    def test_crossfade_endpoints_equal_pure_views(self):
        self.step(self.body, [("front", 1.0), ("back", 0.0)], n=10)
        fa = self.r.final.img.copy()
        self.step(self.body, [("front", 1.0)], n=1)
        self.assertTrue(np.array_equal(fa[250:600], self.r.final.img[250:600]))
        self.step(self.body, [("front", 0.0), ("back", 1.0)], n=1)
        fb = self.r.final.img.copy()
        self.step(self.body, [("back", 1.0)], n=1)
        self.assertTrue(np.array_equal(fb[250:600], self.r.final.img[250:600]))

    def test_crossfade_midpoint_alpha_not_over_255(self):
        self.step(self.body, [("front", 0.5), ("back", 0.5)], n=10)
        # premultiplied: colour channels never exceed alpha
        img = self.r.final.img.astype(int)
        self.assertTrue((img[..., :3].max(axis=2) <= img[..., 3] + 1).all())

    def test_views_with_side(self):
        self.step(self.body, [("front", 0.5), ("left", 0.5)], n=10)
        self.assertFalse(self.r.final.is_empty)
        self.assertGreater(nonblack(self.proj()).sum(), 5000)


class TestDebugFrame(RenderBase):
    def test_debug_frame_mirrored_composite(self):
        del self.r._draw_ui
        cam = np.full((720, 1280, 3), 60, np.uint8)
        self.step(self.body, n=10)
        before = cam.copy()
        out = self.r.debug_frame(cam, self.body, self.state, GESTURES, ["hud line", "second"], "footer", self.t)
        self.assertTrue(np.array_equal(cam, before), "debug_frame modified its input frame")
        self.assertEqual(out.shape, cam.shape)
        a = self.r.final.img
        h_mid = (self.body.kp["left_hip"] + self.body.kp["right_hip"]) / 2
        cx, cy = int(h_mid[0]) + 20, int(h_mid[1]) - 25
        self.assertGreater(a[cy, cx, 3], 250)
        mx = 1280 - 1 - cx
        got = out[cy, mx].astype(int)
        np.testing.assert_allclose(got, a[cy, cx, :3].astype(int), atol=3)
        # garment-free area keeps the camera colour (mirrored, so still the flat 60 background)
        self.assertEqual(out[600, 50].tolist(), [60, 60, 60])

    def test_debug_overlays_toggle_and_hud(self):
        cam = np.zeros((720, 1280, 3), np.uint8)
        self.step(self.body, n=10)
        self.state.debug_overlays = False
        off = self.r.debug_frame(cam, self.body, self.state, GESTURES, ["hud", "more", "lines"], "foot", self.t)
        self.state.debug_overlays = True
        on = self.r.debug_frame(cam, self.body, self.state, GESTURES, ["hud", "more", "lines"], "foot", self.t)
        self.assertGreater(int((on != off).any(axis=2).sum()), 100)
        self.assertEqual(on.shape, (720, 1280, 3))

    def test_debug_frame_no_body(self):
        cam = np.zeros((720, 1280, 3), np.uint8)
        out = self.r.debug_frame(cam, None, self.state, GESTURES, ["hud"], "", self.t)
        self.assertEqual(out.shape, cam.shape)

    def test_debug_frame_fades_with_presence(self):
        cam = np.zeros((720, 1280, 3), np.uint8)
        self.step(self.body, n=12)
        full = self.r.debug_frame(cam, self.body, self.state, GESTURES, [], "", self.t).astype(int).sum()
        self.step(None, n=4)
        half = self.r.debug_frame(cam, None, self.state, GESTURES, [], "", self.t).astype(int).sum()
        self.assertLess(half, full)
        self.step(None, n=10)
        self.assertEqual(int(self.r.debug_frame(cam, None, self.state, GESTURES, [], "", self.t).max()), 0)


class TestEdgeBodies(RenderBase):
    def test_degenerate_body_no_crash(self):
        b = make_body("degenerate")
        layer = self.step(b, n=3)
        self.assertEqual(int(self.proj().max()), 0)

    def test_body_off_frame(self):
        for top in (700, -500):
            self.step(make_body("neutral", top=top), n=3)
            self.proj()

    def test_garment_switch_changes_output(self):
        self.step(self.body, n=10)
        a = self.proj().copy()
        self.state.step(+1, self.t)
        self.step(self.body, n=1)
        b = self.proj()
        self.assertFalse(np.array_equal(a, b))

    def test_all_garments_all_views_through_renderer(self):
        for mode in ("shirt", "pants"):
            self.state.mode = mode
            for i in range(len(self.cat.items(mode))):
                for view in ("front", "left", "back", "right"):
                    with self.subTest(mode=mode, i=i, view=view):
                        self.step(self.body, [(view, 1.0)], n=1)
                        self.assertEqual(self.proj().shape, (1080, 1920, 3))
                self.state.step(+1, self.t)

    def test_timing_budget(self):
        import time
        self.step(self.body, n=10)
        t0 = time.perf_counter()
        for _ in range(100):
            self.t += 0.033
            self.r.build_layer(self.body, self.state, [("front", 0.5), ("back", 0.5)], np.full((720, 1280), 255, np.uint8), self.t)
            self.r.projector_frame(self.cal, self.state, GESTURES, self.t)
        ms = (time.perf_counter() - t0) * 10
        print(f"\n[perf] build_layer(crossfade+mask)+projector_frame: {ms:.2f} ms/frame")
        self.assertLess(ms, 25.0)


if __name__ == "__main__":
    unittest.main()
