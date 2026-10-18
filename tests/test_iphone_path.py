"""End-to-end check of `main.py --sensor iphone` against the fake phone (reviewer findings: background must come
from the empty scene, the head box stays black on the 3D path, and tracking loss never falls back to the 2D map)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from fitcast.calib3d import Calibration3D, save_calibration3d, sim_truth
from fitcast.sensors.fake_phone import FakePhone

pytestmark = pytest.mark.skipif(not Path("assets/models/pose_landmarker_full.task").exists(), reason="models not downloaded")


def _c3d(path: Path, cam=(960, 720), proj=(1920, 1080)) -> None:
    tr = sim_truth(cam, proj, 1.19)
    c = Calibration3D(K_cam=np.array([[750, 0, 480], [0, 750, 360], [0, 0, 1]], float), dist_cam=np.zeros(5),
                      K_p=tr["K_p_true"], dist_p=tr["dist_p_true"], R=tr["R_true"], t=tr["t_true"], cam_size=cam, proj_size=proj,
                      reproj_error_px=0.3, n_points=1000, timestamp="test", notes="synthetic")
    save_calibration3d(str(path), c)


def test_iphone_path_background_head_and_loss(config, tmp_path):
    import main as m
    c3d = tmp_path / "c3d.yaml"; _c3d(c3d)
    phone = FakePhone("127.0.0.1", 0, fps=60.0, person_after_s=1.5).start()
    cfg = dict(config); cfg["sensor"] = dict(config["sensor"], port=phone.port, usbmux=False, host="127.0.0.1")
    cfg["depthmask"] = dict(config["depthmask"], background_frames=30)
    args = m.parse_args(["--sensor", "iphone", "--no-window", "--outfit", "0", "--calibration3d", str(c3d)])
    app = m.App(cfg, args)
    calls = {"to_projector": 0}
    orig = app.calib.to_projector
    def spy(*a, **k):
        calls["to_projector"] += 1
        return orig(*a, **k)
    app.calib.to_projector = spy
    app.source.start()
    try:
        t0 = time.perf_counter(); last = -1; n_ok = 0; bg_ready_at = None
        while time.perf_counter() - t0 < 6.0:
            it = app.source.read()
            if it is None or it[2] == last:
                time.sleep(0.002); continue
            frame, ts, seq = it; last = seq
            app.step(frame, ts, time.perf_counter()); app.frames += 1
            if app.depth_mask is not None and app.depth_mask.ready and bg_ready_at is None:
                bg_ready_at = time.perf_counter() - t0
            tr = app.source.track()
            if tr is not None and tr.ok and app.depth_mask is not None and app.depth_mask.ready:
                n_ok += 1
        # background learned from the empty first 1.5 s, before the person appeared
        assert bg_ready_at is not None and bg_ready_at < 1.6, bg_ready_at
        assert n_ok > 30, "person frames should have been rendered through the 3D path"
        assert app.proj3d is not None and calls["to_projector"] == 0, "3D path must never use the 2D homography"
        # head box black in the composite on the last frame
        comp = app.compositor.out; tr = app.source.track()
        x0, y0, x1, y1 = app.smoother.update(app.tracker.process_external(frame, ts, tr.joints, tr.visibility, None)).head_box
        assert comp[y0:y1, x0:x1].max() == 0
        # tracking loss: the projector shows a dimmed copy of the last 3D frame, never the 2D mapping
        cache_before = app._proj_cache.copy()
        lost = dict(app.source.latest()); lost["track"] = None
        app.source.latest = lambda: lost
        app.step(frame, time.perf_counter(), time.perf_counter())
        assert calls["to_projector"] == 0
        assert app.compositor.fade < 1.0
    finally:
        app.shutdown(); phone.stop()
