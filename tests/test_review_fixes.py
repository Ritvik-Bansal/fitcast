"""Regression tests for the issues found in the M1-M5 review."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fitcast.smoothing import Smoother
from fitcast.tracker import FacingClassifier, SegWorker
from fitcast.types import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, FACE_IDX
from tests.conftest import frontal_joints, make_track


def _run(fc, joints, vis, n):
    for _ in range(n):
        fc.update(joints, vis, 0.5)
    return fc.state


def test_hidden_hips_do_not_poison_the_facing_reference(config):
    """Reviewer MAJOR: one stretch of hidden hips used to switch the reference to raw px and lock 'side'."""
    fc = FacingClassifier(config["tracker"]["facing"])
    vis = np.ones((33,), np.float32)
    front = frontal_joints()
    assert _run(fc, front, vis, 60) == "front"
    no_hips = vis.copy()
    no_hips[[L_HIP, R_HIP]] = 0.0
    assert _run(fc, front, no_hips, 30) == "front"
    assert _run(fc, front, vis, 30) == "front"
    assert 0.8 < fc.ratio < 1.25


def test_side_first_session_recovers_when_the_person_turns_front(config):
    fc = FacingClassifier(config["tracker"]["facing"])
    vis = np.ones((33,), np.float32)
    side = frontal_joints()
    side[L_SHOULDER, 0], side[R_SHOULDER, 0] = 640 + 20, 640 - 20
    _run(fc, side, vis, 60)                      # narrow reference seeded side-on
    front = frontal_joints()
    hold = config["tracker"]["facing"]["hold_frames"]
    assert _run(fc, front, vis, hold + 2) == "front"
    assert fc.ratio > 0.9                         # reference jumped to the frontal width at once


def test_key_map_has_no_letter_collisions():
    import main
    for letter in "cdfgmqr":
        assert ord(letter) not in main.KEY_NEXT and ord(letter) not in main.KEY_PREV


class _RaisingBackend:
    def segment_image(self, img, ms):
        raise RuntimeError("boom")


def test_seg_worker_failure_is_reported_not_silent(caplog):
    w = SegWorker(_RaisingBackend())
    w.start()
    assert w.submit(object(), 1)
    w.join(timeout=2.0)
    assert not w.is_alive()
    assert w.failed is True and w.busy is False
    assert any("segmenter worker crashed" in r.getMessage() for r in caplog.records)


def test_smoother_grows_head_box_by_prediction_shift(config):
    cfg = dict(config["smoothing"], predict_ms=100, max_predict_px=40, predict_auto=False)
    sm = Smoother(cfg, mask_alpha=1.0)
    base = make_track(ts=0.0)
    box0 = base.head_box
    out = None
    for i in range(1, 15):
        tr = make_track(ts=i / 30)
        tr.joints[:, 0] += 12 * i                 # fast move to the right
        tr.head_box = (box0[0] + 12 * i, box0[1], box0[2] + 12 * i, box0[3])
        out = sm.update(tr)
    # predicted face landmarks must still be inside the (grown) head box
    fx = out.joints[list(FACE_IDX), 0]
    assert out.head_box[0] <= fx.min() and fx.max() <= out.head_box[2]
    assert out.head_box[2] > box0[2] + 12 * 14    # grew on the leading side


@pytest.mark.skipif(not Path("assets/models/pose_landmarker_full.task").exists(), reason="models not downloaded")
def test_mask_fallback_logic_on_area_jump(config):
    """First mask after an empty one is not a 'jump'; a real >40% jump triggers the hold then clears."""
    from fitcast.tracker import Tracker
    cfg = dict(config["tracker"], seg_async=False)
    trk = Tracker(cfg, (1280, 720))
    try:
        trk._prev_area, trk._fallback_left = 0.0, 0
        # emulate the bookkeeping directly
        for area, expect in ((0.0, False), (50000.0, False), (52000.0, False), (90000.0, True)):
            if trk._prev_area > 0 and area > 0 and abs(area - trk._prev_area) / trk._prev_area > trk.area_jump:
                trk._fallback_left = trk.fallback_hold
            trk._prev_area = area
            active = trk._fallback_left > 0
            if trk._fallback_left > 0:
                trk._fallback_left -= 1
            assert active == expect, (area, active)
        assert trk.fallback_hold >= 1
    finally:
        trk.close()


def test_simulated_calibration_refused_for_live_sources(config, tmp_path):
    from fitcast.calibration import Calibration, SimulatedRig, make_sim_homography, run_calibration
    cam = (1280, 720)
    rig = SimulatedRig(config["calibration"], cam, make_sim_homography(cam, (1920, 1080)))
    cal = run_calibration(config["calibration"], cam, rig.capture, rig.show, log=lambda *a, **k: None)
    cal.data.simulated = True
    out = tmp_path / "sim.yaml"
    cal.save(str(out))
    live = Calibration(config["calibration"], cam)
    assert live.load(str(out)) is False and not live.ok
    clip = Calibration(config["calibration"], cam)
    assert clip.load(str(out), allow_simulated=True) is True and clip.ok
