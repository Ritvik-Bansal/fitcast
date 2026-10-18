"""DepthBodyMask: synthetic LiDAR scenes (256x192 depth, 1280x720 camera)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from fitcast import depthmask
from fitcast.depthmask import DepthBodyMask

DW, DH = 256, 192
CW, CH = 1280, 720
K_DEPTH = np.array([[200.0, 0, 128.0], [0, 200.0, 96.0], [0, 0, 1]])
BG_Z, PERSON_Z = 3.0, 2.2


def person_shape(w: int, h: int, cx: float = 0.5, scale: float = 1.0) -> np.ndarray:
    """Analytic person (ellipse torso + rectangle legs) in normalised coords, sampled at pixel centres,
    so the depth-res and camera-res versions describe the same shape. (h, w) uint8 0/255."""
    x = (np.arange(w) + 0.5) / w
    y = (np.arange(h) + 0.5) / h
    X, Y = np.meshgrid(x, y)
    torso = ((X - cx) / (0.09 * scale)) ** 2 + ((Y - 0.45) / (0.2 * scale)) ** 2 <= 1.0
    legs = (np.abs(X - cx) <= 0.06 * scale) & (Y >= 0.6) & (Y <= min(0.6 + 0.32 * scale, 0.99))
    return ((torso | legs) * 255).astype(np.uint8)


@pytest.fixture
def cfg(config) -> dict:
    return dict(config["depthmask"])


def scene(rng: np.random.Generator, person: bool = True, cx: float = 0.5, scale: float = 1.0
          ) -> tuple[np.ndarray, np.ndarray]:
    """One depth frame (background plane + optional person) with 1 cm noise, and an all-high confidence map."""
    d = (BG_Z + rng.normal(0, 0.01, (DH, DW))).astype(np.float32)
    if person:
        m = person_shape(DW, DH, cx, scale) > 0
        d[m] = PERSON_Z + rng.normal(0, 0.01, int(m.sum()))
    return d, np.full((DH, DW), 2, np.uint8)


def rgb_for(gt_cam: np.ndarray) -> np.ndarray:
    """Grey background, white person (the all-white clothes under room light)."""
    img = np.full((CH, CW, 3), 128, np.uint8)
    img[gt_cam > 0] = 255
    return img


def make_mask(cfg: dict, rng: np.random.Generator, intrinsics: bool = True) -> DepthBodyMask:
    dm = DepthBodyMask(cfg, (DW, DH), (CW, CH))
    if intrinsics:
        dm.set_intrinsics(K_DEPTH)
    frames = [scene(rng, person=False) for _ in range(30)]
    dm.learn_background([f[0] for f in frames], [f[1] for f in frames])
    return dm


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_ximgproc_available() -> None:
    assert hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


def test_bilinear_fallback_without_ximgproc(cfg, monkeypatch) -> None:
    monkeypatch.setattr(depthmask, "_HAS_XIMGPROC", False)
    rng = np.random.default_rng(8)
    dm = make_mask(cfg, rng)
    gt = person_shape(CW, CH)
    d, c = scene(rng)
    assert iou(dm.compute(d, c, rgb_for(gt)), gt) > 0.9


def test_rejects_non_uint8_guide(cfg) -> None:
    dm = make_mask(cfg, np.random.default_rng(10))
    d, c = scene(np.random.default_rng(11))
    with pytest.raises(ValueError):
        dm.compute(d, c, np.zeros((CH, CW, 3), np.float32))


def test_not_ready_returns_empty(cfg) -> None:
    dm = DepthBodyMask(cfg, (DW, DH), (CW, CH))
    assert not dm.ready
    d, c = scene(np.random.default_rng(0))
    out = dm.compute(d, c, np.zeros((CH, CW, 3), np.uint8))
    assert out.shape == (CH, CW) and out.dtype == np.uint8 and out.max() == 0


def test_synthetic_person_iou(cfg) -> None:
    rng = np.random.default_rng(1)
    dm = make_mask(cfg, rng)
    assert dm.ready
    assert np.isfinite(dm.background).all() and abs(float(np.median(dm.background)) - BG_Z) < 0.005
    gt = person_shape(CW, CH)
    d, c = scene(rng)
    out = dm.compute(d, c, rgb_for(gt))
    assert out.shape == (CH, CW) and out.dtype == np.uint8 and set(np.unique(out)) <= {0, 255}
    score = iou(out, gt)
    assert score > 0.9, score
    assert dm.last_stats["n_components"] == 1 and dm.last_stats["valid_fraction"] == 1.0
    # float16 depth (the wire format) gives the same result
    out16 = dm.compute(d.astype(np.float16), c, rgb_for(gt))
    assert iou(out16, out) > 0.99


def test_projector_invariance(cfg) -> None:
    """Projected saturated garment colours inside the person must not change membership (depth decides)."""
    rng = np.random.default_rng(2)
    dm = make_mask(cfg, rng)
    gt = person_shape(CW, CH)
    d, c = scene(rng)
    rgb = rgb_for(gt)
    a = dm.compute(d, c, rgb).copy()
    proj = rgb.copy()
    xx = np.arange(CW)[None, :].repeat(CH, 0)
    stripes = ((xx // 24) % 2).astype(bool)
    inside = gt > 0
    proj[inside & stripes] = (0, 0, 255)      # saturated red
    proj[inside & ~stripes] = (255, 0, 0)     # saturated blue
    b = dm.compute(d, c, proj)
    changed = cv2.countNonZero(cv2.bitwise_xor(a, b))
    assert changed < 0.02 * cv2.countNonZero(a), (changed, cv2.countNonZero(a))


def test_box_excludes_near_objects(cfg) -> None:
    rng = np.random.default_rng(3)
    dm = make_mask(cfg, rng)
    gt = person_shape(CW, CH)
    d, c = scene(rng)
    d[140:192, 0:60] = 1.0          # table at z = 1.0 m (in front of box z0 = 1.5) in the bottom-left corner
    d[10:60, 226:256] = PERSON_Z    # object at a valid z but X = (226-128)/200*2.2 > 1.08 m: outside box x1 = 1.0
    out = dm.compute(d, c, rgb_for(gt))
    assert out[int(140 * CH / DH):, : int(60 * CW / DW)].max() == 0
    assert out[: int(60 * CH / DH), int(226 * CW / DW):].max() == 0
    assert iou(out, gt) > 0.9
    # control: with an unbounded x range the side object IS kept, so the exclusion above is the box test
    wide = dict(cfg, box_m=[-10.0, 10.0] + list(cfg["box_m"][2:]))
    dm2 = make_mask(wide, rng)
    out2 = dm2.compute(d, c, rgb_for(gt))
    assert out2[: int(60 * CH / DH), int(226 * CW / DW):].max() == 255


def test_low_confidence_and_invalid_excluded(cfg) -> None:
    rng = np.random.default_rng(4)
    dm = make_mask(cfg, rng)
    gt = person_shape(CW, CH)
    rgb = rgb_for(gt)
    d, c = scene(rng)
    person = person_shape(DW, DH) > 0
    c0 = c.copy()
    c0[person] = 0
    assert dm.compute(d, c0, rgb).max() == 0
    assert dm.last_stats["body_pixels"] == 0
    c0[:] = 0
    assert dm.compute(d, c0, rgb).max() == 0
    for bad in (np.nan, 0.0):
        dbad = d.copy()
        dbad[person] = bad
        assert dm.compute(dbad, c, rgb).max() == 0
    # conf 0 on the left half of the person: only the right half survives
    c1 = c.copy()
    c1[:, :128] = 0
    out = dm.compute(d, c1, rgb)
    assert out[:, : CW // 2 - 10].max() == 0 and out[:, CW // 2 + 10:].max() == 255


def test_background_unknown_pixels_are_far(cfg) -> None:
    rng = np.random.default_rng(5)
    dm = DepthBodyMask(cfg, (DW, DH), (CW, CH))
    dm.set_intrinsics(K_DEPTH)
    frames = [scene(rng, person=False) for _ in range(10)]
    confs = [f[1].copy() for f in frames]
    for cf in confs:
        cf[:, 100:156] = 0          # never confidently seen
    dm.learn_background([f[0] for f in frames], confs)
    assert np.isinf(dm.background[:, 100:156]).all() and np.isfinite(dm.background[:, :100]).all()
    # the wall there stays unmeasurable (e.g. beyond LiDAR range); the person standing in front of it is found
    gt = person_shape(CW, CH)
    d, c = scene(rng)
    c[:, 100:156] = 0
    c[person_shape(DW, DH) > 0] = 2
    assert iou(dm.compute(d, c, rgb_for(gt)), gt) > 0.9


def test_fusion_only_near_depth_body(cfg) -> None:
    rng = np.random.default_rng(6)
    reach = int(cfg.get("fuse_reach_px", 12))
    dm = make_mask(dict(cfg, fuse_person_seg=True, fuse_skeleton=True), rng)
    gt = person_shape(CW, CH)
    rgb = rgb_for(gt)
    d, c = scene(rng)
    base = dm.compute(d, c, rgb).copy()
    near = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * reach + 1)))

    seg = gt.copy()
    seg[50:250, 50:250] = 255                  # a second person only in RGB segmentation, far from the body
    r0, r1 = 520, 560                          # legs: straight vertical body edge in these rows
    row_edges = [int(np.nonzero(row)[0].max()) for row in base[r0:r1]]
    edge_x = max(row_edges)
    assert edge_x - min(row_edges) <= 1, "test band must have a straight body edge"
    seg[r0:r1, edge_x - 5:edge_x + 80] = 255    # arm only in RGB, touching the body edge, extending 80 px out
    seg_small = cv2.resize(seg, (DW, DH), interpolation=cv2.INTER_NEAREST)  # "any size"
    out = dm.compute(d, c, rgb, person_seg=seg_small).copy()
    added = cv2.bitwise_and(out, cv2.bitwise_not(base))
    assert added[50:250, 50:250].max() == 0, "RGB-only region far from the depth body must not be added"
    assert added[r0:r1, edge_x + 1:edge_x + reach].max() == 255, "seg touching the body edge is added"
    assert cv2.countNonZero(cv2.bitwise_and(added, cv2.bitwise_not(near))) == 0, "nothing beyond fuse_reach_px"
    assert added[r0:r1, edge_x + reach + 2:].max() == 0

    skel = np.zeros((CH, CW), np.uint8)
    skel[300:500, 1000:1100] = 255             # far skeleton capsule: not added
    skel[r0:r1, edge_x - 5:edge_x + 80] = 255
    out = dm.compute(d, c, rgb, skeleton_mask=skel)
    added = cv2.bitwise_and(out, cv2.bitwise_not(base))
    assert added[300:500, 1000:1100].max() == 0
    assert added[r0:r1, edge_x + 1:edge_x + reach].max() == 255
    assert cv2.countNonZero(cv2.bitwise_and(added, cv2.bitwise_not(near))) == 0

    # fusion disabled: inputs ignored
    dm_off = make_mask(dict(cfg, fuse_person_seg=False, fuse_skeleton=False), rng)
    off = dm_off.compute(d, c, rgb, person_seg=seg_small, skeleton_mask=skel)
    assert cv2.countNonZero(cv2.bitwise_and(off, cv2.bitwise_not(near))) == 0


@pytest.mark.parametrize("label,scale", [("typical", 1.0), ("large", 2.4)])
def test_timing(cfg, capsys, label: str, scale: float) -> None:
    rng = np.random.default_rng(7)
    dm = make_mask(dict(cfg, fuse_person_seg=True, fuse_skeleton=True), rng)
    gt = person_shape(CW, CH, scale=scale)
    rgb = rgb_for(gt)
    d, c = scene(rng, scale=scale)
    seg = cv2.resize(gt, (DW, DH), interpolation=cv2.INTER_NEAREST)
    skel = cv2.erode(gt, np.ones((15, 15), np.uint8))
    for _ in range(3):
        dm.compute(d, c, rgb, seg, skel)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        dm.compute(d, c, rgb, seg, skel)
    ms = (time.perf_counter() - t0) / n * 1e3
    x0, y0, x1, y1 = dm.last_stats["roi"]
    roi_frac = (x1 - x0) * (y1 - y0) / float(CW * CH)
    with capsys.disabled():
        print(f"\n[depthmask timing {label}] compute() mean {ms:.2f} ms over {n} calls at {CW}x{CH} "
              f"(depth {DW}x{DH}, ROI {roi_frac:.0%} of frame, seg+skeleton fusion on)")
    assert ms > 0
