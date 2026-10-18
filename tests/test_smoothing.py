import numpy as np

from fitcast.smoothing import OneEuroArray, Smoother
from tests.conftest import make_track


def test_one_euro_removes_jitter_and_tracks_motion():
    f = OneEuroArray(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    rng = np.random.default_rng(0)
    t = 0.0
    err_raw, err_f = [], []
    for i in range(200):
        t += 1 / 30
        truth = np.array([100.0 + 2.0 * i, 50.0], np.float32)
        noisy = truth + rng.normal(0, 3.0, 2).astype(np.float32)
        out = f(noisy, t)
        if i > 30:
            err_raw.append(np.abs(noisy - truth).mean())
            err_f.append(np.abs(out - truth).mean())
    assert np.mean(err_f) < np.mean(err_raw)
    assert np.mean(err_f) < 4.0


def test_smoother_predicts_forward_and_clamps(config):
    cfg = dict(config["smoothing"], predict_ms=100, max_predict_px=5, min_cutoff=1.0, beta=0.05)   # the mechanism, not the live tuning
    sm = Smoother(cfg, mask_alpha=1.0)
    tr = make_track(ts=0.0)
    base = tr.joints.copy()
    for i in range(1, 20):
        tr = make_track(ts=i / 30)
        tr.joints[:, 0] = base[:, 0] + 10 * i          # 300 px/s to the right
        out = sm.update(tr)
    assert out.joints[0, 0] > base[0, 0] + 10 * 19        # predicted ahead
    assert out.joints[0, 0] <= base[0, 0] + 10 * 19 + 5 + 1e-3


def test_mask_ema(config):
    sm = Smoother(config["smoothing"], mask_alpha=0.5)
    sm.update(make_track(ts=0.0, mask_full=True))
    out = sm.update(make_track(ts=1 / 30, mask_full=False))
    assert out.clothes_mask.max() == 255      # one empty frame does not clear the mask (EMA hysteresis)
    out = sm.update(make_track(ts=2 / 30, mask_full=False))
    assert out.clothes_mask.max() == 0        # two do
