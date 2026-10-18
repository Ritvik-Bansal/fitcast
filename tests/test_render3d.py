"""DepthMeshProjector: planar equivalence with a homography warp, occlusion, depth gaps, distortion, hole fill,
benchmark. Headless (offscreen ModernGL context)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fitcast.render3d import DepthMeshProjector

CAM = (1280, 720)
PROJ = (1920, 1080)
K_CAM = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
K_P = np.array([[2200.0, 0.0, 960.0], [0.0, 2200.0, 540.0], [0.0, 0.0, 1.0]])
NO_DIST = np.zeros(5)
R_3DEG = cv2.Rodrigues(np.array([0.0, np.deg2rad(3.0), 0.0]))[0]
T_PLANE = np.array([0.15, -0.1, 0.0])
Z_PLANE = 3.0

# Planar scene: depth valid only inside REGION (x0, y0, x1, y1 inclusive camera px); grey background there,
# coloured rectangles (BGR) inside it, black elsewhere. Rectangles are placed asymmetrically so a flipped axis
# moves their centroids.
REGION = (200, 160, 900, 600)
RECTS = {(0, 0, 255): (300, 200, 420, 290), (0, 255, 0): (650, 420, 800, 560),
         (255, 0, 0): (480, 330, 560, 380), (0, 255, 255): (760, 180, 860, 260)}

# Occlusion scene: far plane everywhere, a near box in front; projector offset along +x, no rotation.
Z_FAR, Z_NEAR = 3.0, 2.2
BOX = (540, 260, 740, 460)
T_OCC = np.array([0.3, 0.0, 0.0])
FAR_BGR, NEAR_BGR = (0, 200, 0), (0, 0, 220)


@pytest.fixture
def make(config):
    """Factory for projectors with config['render3d'] plus overrides; releases every GL context afterwards."""
    made: list[DepthMeshProjector] = []

    def _make(R=np.eye(3), t=np.zeros(3), dist=NO_DIST, **overrides) -> DepthMeshProjector:
        p = DepthMeshProjector({**config["render3d"], **overrides}, CAM, PROJ, K_CAM, K_P, dist, R, t)
        made.append(p)
        return p

    yield _make
    for p in made:
        p.release()


def _box(img: np.ndarray, rect, value) -> None:
    x0, y0, x1, y1 = rect
    img[y0:y1 + 1, x0:x1 + 1] = value


def _plane_scene() -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((CAM[1], CAM[0], 3), np.uint8)
    depth = np.zeros((CAM[1], CAM[0]), np.float32)
    _box(img, REGION, 100)
    _box(depth, REGION, Z_PLANE)
    for bgr, rect in RECTS.items():
        _box(img, rect, bgr)
    return img, depth


def _occlusion_scene() -> tuple[np.ndarray, np.ndarray]:
    img = np.empty((CAM[1], CAM[0], 3), np.uint8)
    depth = np.empty((CAM[1], CAM[0]), np.float32)
    img[:] = FAR_BGR
    depth[:] = Z_FAR
    _box(img, BOX, NEAR_BGR)
    _box(depth, BOX, Z_NEAR)
    return img, depth


def _plane_H(R: np.ndarray, t: np.ndarray, z: float) -> np.ndarray:
    """Camera px -> projector px for the fronto-parallel plane z = const: K_p (R + t n^T / z) K_cam^-1."""
    return K_P @ (R + np.outer(t, [0.0, 0.0, 1.0]) / z) @ np.linalg.inv(K_CAM)


def _warp_pts(H: np.ndarray, pts) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def _coverage(img: np.ndarray) -> np.ndarray:
    return img.max(axis=2) > 30


def _centroid(img: np.ndarray, bgr) -> np.ndarray:
    m = np.all(np.abs(img.astype(np.int16) - np.array(bgr, np.int16)) < 60, axis=2)
    assert m.sum() > 100, f"colour {bgr} not found"
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def _is(img: np.ndarray, bgr) -> np.ndarray:
    return np.all(np.abs(img.astype(np.int16) - np.array(bgr, np.int16)) < 30, axis=-1)


def test_planar_equivalence_with_homography(make):
    img, depth = _plane_scene()
    H = _plane_H(R_3DEG, T_PLANE, Z_PLANE)
    x0, y0, x1, y1 = REGION
    corners = _warp_pts(H, [(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
    assert np.all(corners > 0) and np.all(corners < PROJ), "test scene must fit inside the projector frame"

    out = make(R_3DEG, T_PLANE).render(img, depth).copy()
    ref = cv2.warpPerspective(img, H, PROJ, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov_out, cov_ref = _coverage(out), _coverage(ref)
    assert np.mean(cov_out != cov_ref) < 0.01
    for bgr in RECTS:
        err = np.linalg.norm(_centroid(out, bgr) - _centroid(ref, bgr))
        assert err < 1.5, f"rectangle {bgr} centroid off by {err:.2f} px"

    # Watertight mesh: without hole filling there is no dark pixel inside the footprint.
    raw = make(R_3DEG, T_PLANE, fill_holes_px=0).render(img, depth)
    inner = cv2.erode(cov_ref.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert not np.any(raw.max(axis=2)[inner] == 0)


def _occlusion_bands() -> tuple[tuple[float, float], tuple[float, float]]:
    """Projector x ranges (gap band, overlap band) on the rows the box covers. Gap = far-plane hole not covered by the near box
    (nothing visible to the camera lands there); overlap = near box over far plane (both project there)."""
    near = _warp_pts(_plane_H(np.eye(3), T_OCC, Z_NEAR), [(BOX[0], 360), (BOX[2], 360)])[:, 0]
    hole = _warp_pts(_plane_H(np.eye(3), T_OCC, Z_FAR), [(BOX[0], 360), (BOX[2], 360)])[:, 0]
    assert near[0] > hole[0] and near[1] > hole[1], "projector at +x: the near box shifts further right"
    return (hole[0], near[0]), (hole[1], near[1])


def test_near_surface_wins(make):
    img, depth = _occlusion_scene()
    out = make(t=T_OCC).render(img, depth)
    y = PROJ[1] // 2
    _, (ox0, ox1) = _occlusion_bands()
    band = out[y - 50:y + 51, int(np.ceil(ox0)) + 3:int(ox1) - 3]
    assert band.shape[1] > 50
    assert _is(band, NEAR_BGR).mean() > 0.99
    assert not _is(band, FAR_BGR).any()


def test_no_light_across_depth_gaps(make):
    img, depth = _occlusion_scene()
    y = PROJ[1] // 2
    (gx0, gx1), _ = _occlusion_bands()
    seg = slice(int(np.ceil(gx0)) + 2, int(gx1) - 2)

    out = make(t=T_OCC).render(img, depth)
    dark = out[y, seg].max(axis=1) == 0
    assert dark.sum() > 0 and dark.mean() > 0.9, f"gap band {gx0:.0f}..{gx1:.0f} px should be dark"

    # The far plane covers the whole projector frame: with the gap test disabled nothing is dark anywhere.
    out = make(t=T_OCC, depth_gap_m=1e6).render(img, depth)
    assert np.count_nonzero(out[y].max(axis=1) == 0) == 0


def test_benchmark(make, config, capsys):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (CAM[1], CAM[0], 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:CAM[1], 0:CAM[0]].astype(np.float32)
    depth = (3.0 + 0.0004 * (xx - 640) + 0.05 * np.sin(yy / 50.0)).astype(np.float32)   # tilted wavy wall
    person = ((xx - 640) / 220) ** 2 + ((yy - 400) / 330) ** 2 < 1
    depth[person] = 2.2 + 0.1 * np.cos((xx[person] - 640) / 140)                        # curved body in front
    depth[rng.random(depth.shape) < 0.01] = 0.0                                          # 1 % dropouts
    valid = (depth > 0).astype(np.uint8) * 255
    results = {}
    for step in sorted({int(config["render3d"]["mesh_step"]), 2}):
        p = make(R_3DEG, T_PLANE, mesh_step=step)
        for _ in range(5):                                       # warm-up: shader compile, driver allocations
            p.render(img, depth, valid)
        ms = []
        for i in range(50):
            p.render(img if i % 2 else img[::-1].copy(), depth, valid)
            ms.append(p.last_render_ms)
        results[step] = (float(np.mean(ms)), float(np.percentile(ms, 95)))
    with capsys.disabled():
        for step, (mean, p95) in results.items():
            print(f"\n[render3d benchmark] {CAM[0]}x{CAM[1]} -> {PROJ[0]}x{PROJ[1]}, mesh_step={step}: "
                  f"mean {mean:.2f} ms, p95 {p95:.2f} ms over 50 frames")
    assert results[int(config["render3d"]["mesh_step"])][0] < 20.0


def _plane_depth() -> np.ndarray:
    return np.full((CAM[1], CAM[0]), Z_PLANE, np.float32)


def test_distortion_matches_opencv_and_extrinsics_apply(make):
    img, _ = _plane_scene()
    depth = _plane_depth()
    plain = make(R_3DEG, T_PLANE).render(img, depth).copy()
    dist = np.array([-0.1, 0.0, 0.0, 0.0, 0.0])
    p = make(R_3DEG, T_PLANE, dist=dist)
    distorted = p.render(img, depth).copy()
    assert np.mean(_coverage(plain) != _coverage(distorted)) > 0.001

    # Full 5-term model agrees with cv2.projectPoints (same coefficient order and sign conventions).
    dist = np.array([-0.1, 0.05, 0.002, -0.003, 0.01])
    out = make(R_3DEG, T_PLANE, dist=dist).render(img, depth)
    rvec = cv2.Rodrigues(R_3DEG)[0]
    for bgr, (x0, y0, x1, y1) in RECTS.items():
        X = Z_PLANE * np.linalg.inv(K_CAM) @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, 1.0])
        expect = cv2.projectPoints(X.reshape(1, 1, 3), rvec, T_PLANE, K_P, dist)[0].ravel()
        err = np.linalg.norm(_centroid(out, bgr) - expect)
        assert err < 2.0, f"rectangle {bgr}: {err:.2f} px from cv2.projectPoints"

    p.set_extrinsics(R_3DEG, T_PLANE + [0.05, 0.0, 0.0])
    moved = p.render(img, depth).copy()
    assert np.mean(moved != distorted) > 0.01
    p.set_extrinsics(R_3DEG, T_PLANE)
    assert np.array_equal(p.render(img, depth), distorted)


def test_fill_small_holes_but_not_real_gaps(make):
    """Isolated invalid depth pixels (4.4 px holes) are filled; a curtain across a 15 cm depth step (about 4 px wide
    in the projector) stays dark although it is narrower than fill_holes_px."""
    img = np.full((CAM[1], CAM[0], 3), 180, np.uint8)
    depth = _plane_depth()
    depth[:, 640:] = Z_PLANE - 0.15
    holes = np.zeros(depth.shape, bool)
    holes[150:571:23, 220:561:29] = True                     # isolated invalid pixels, far half, inside the frame
    t = np.array([0.05, 0.0, 0.0])
    H_far, H_near = _plane_H(np.eye(3), t, Z_PLANE), _plane_H(np.eye(3), t, Z_PLANE - 0.15)
    edge_far = _warp_pts(H_far, [(639, 360)])[0, 0]
    edge_near = _warp_pts(H_near, [(640, 360)])[0, 0]
    assert 2 < edge_near - edge_far < 6
    (hx0, hy0), (_, hy1) = _warp_pts(H_far, [(210, 140), (210, 580)])
    hole_area = (slice(int(hy0), int(hy1)), slice(int(hx0), int(edge_far) - 20))
    y = PROJ[1] // 2

    raw = make(t=t, fill_holes_px=0).render(img, depth, ~holes)
    assert np.count_nonzero(raw[hole_area].max(axis=2) == 0) > 100
    filled = make(t=t, fill_holes_px=6).render(img, depth, ~holes)
    assert np.count_nonzero(filled[hole_area].max(axis=2) == 0) == 0
    curtain = filled[y, int(np.ceil(edge_far)) + 1:int(edge_near)]
    assert curtain.size and np.all(curtain.max(axis=1) == 0)
