import numpy as np

from fitcast.tracker import FacingClassifier
from fitcast.types import L_EYE, L_SHOULDER, NOSE, R_EYE, R_SHOULDER
from tests.conftest import frontal_joints


def _run(fc, joints, vis, n, min_vis=0.5):
    out = None
    for _ in range(n):
        out = fc.update(joints, vis, min_vis)
    return out


def test_front_back_side_with_hysteresis(config):
    cfg = config["tracker"]["facing"]
    fc = FacingClassifier(cfg)
    vis = np.ones((33,), np.float32)
    front = frontal_joints()
    assert _run(fc, front, vis, 40) == "front"
    # back: swap shoulder x order (person's left now on image left)
    back = front.copy()
    back[L_SHOULDER, 0], back[R_SHOULDER, 0] = front[R_SHOULDER, 0], front[L_SHOULDER, 0]
    hold = cfg["hold_frames"]
    for _ in range(hold - 1):
        fc.update(back, vis, 0.5)
    assert fc.state == "front", "must hold for hold_frames before switching"
    fc.update(back, vis, 0.5)
    assert fc.state == "back"
    # side: shoulders nearly overlapping -> ratio far below side_ratio
    side = front.copy()
    side[L_SHOULDER, 0] = 640 + 20
    side[R_SHOULDER, 0] = 640 - 20
    assert _run(fc, side, vis, hold) == "side"
    # a single flicker frame must not switch state
    fc.update(front, vis, 0.5)
    assert fc.state == "side"
    assert _run(fc, front, vis, hold) == "front"


def test_face_visibility_weak_hint(config):
    cfg = dict(config["tracker"]["facing"], face_vis_back_threshold=0.3)
    fc = FacingClassifier(cfg)
    vis = np.ones((33,), np.float32)
    front = frontal_joints()
    _run(fc, front, vis, 40)
    vis2 = vis.copy()
    vis2[[NOSE, L_EYE, R_EYE]] = 0.1
    assert _run(fc, front, vis2, cfg["hold_frames"] + 1) == "back"


def test_distance_invariance(config):
    """Stepping closer (bigger px) must not look like a frontal reference change that makes 'side' impossible."""
    fc = FacingClassifier(config["tracker"]["facing"])
    vis = np.ones((33,), np.float32)
    far = frontal_joints(sw=120)
    _run(fc, far, vis, 40)
    near = frontal_joints(sw=240)          # same person, closer: torso length doubles too
    near[:, 1] = (near[:, 1] - 120) * 2 + 120
    _run(fc, near, vis, 40)
    assert 0.8 < fc.ratio < 1.25
