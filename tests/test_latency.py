"""fitcast.latency pure functions + a headless simulated run of scripts/latency_probe.py (no windows, no camera)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fitcast import latency as L

ROOT = Path(__file__).resolve().parents[1]


def exposure_signal(edges_on: np.ndarray, edges_off: np.ndarray, times: np.ndarray, T: float,
                    lo: float = 20.0, hi: float = 220.0) -> np.ndarray:
    """Brightness of frames whose exposure is the box [t - T/2, t + T/2] (t = exposure centre) of a step signal:
    the 50% crossing then sits exactly at each step time."""
    def on_at(x: np.ndarray) -> np.ndarray:
        s = np.zeros_like(x)
        for a, b in zip(edges_on, edges_off):
            s += (x >= a) & (x < b)
        return s
    sub = np.linspace(-T / 2, T / 2, 201)
    frac = np.stack([on_at(times + d) for d in sub]).mean(axis=0)
    return lo + (hi - lo) * frac


def test_flash_schedule() -> None:
    s = L.flash_schedule(0.5, 3, t0=10.0)
    assert s == [(10.0, 10.5), (11.0, 11.5), (12.0, 12.5)]


def test_detect_transitions_subframe() -> None:
    rng = np.random.default_rng(1)
    fps = 30.0
    T = 1.0 / fps
    times = np.arange(0, 12.0, T) + 0.0123                     # exposure centres, arbitrary phase
    on = 1.0 + 2.0 * np.arange(5) + rng.uniform(0, T, 5)       # arbitrary sub-frame phases
    off = on + 1.0 + rng.uniform(0, T, 5)
    b = exposure_signal(on, off, times, T) + rng.normal(0, 2.0, times.size)
    rises = L.detect_transitions(b, times)
    falls = L.detect_transitions(b, times, edge="falling")
    assert rises.size == 5 and falls.size == 5
    assert np.max(np.abs(rises - on)) < 0.002
    assert np.max(np.abs(falls - off)) < 0.002


def test_detect_transitions_hysteresis_and_start_bright() -> None:
    t = np.arange(12, dtype=np.float64)
    # starts bright (no edge at 0), one real rise at 5.5, noise wobble around the threshold at 8..9 is ignored
    b = np.array([200, 200, 10, 10, 10, 10, 200, 200, 104, 96, 104, 200], np.float64)
    r = L.detect_transitions(b, t)
    assert r.size == 1 and abs(r[0] - 5.5) < 1e-9
    assert L.detect_transitions(np.full(10, 50.0), np.arange(10.0)).size == 0     # no contrast


def test_match_delays_70ms() -> None:
    rng = np.random.default_rng(2)
    trig = np.array([t for t, _ in L.flash_schedule(1.0, 20, t0=100.0)])
    obs = trig + 0.070 + rng.normal(0, 0.001, trig.size)
    obs = np.delete(obs, 7)                          # one flash never observed
    obs = np.concatenate([[99.5], obs, [obs[3] + 0.01]])   # spurious edges before the run and a double edge
    d, s = L.match_delays(trig, obs, max_delay_s=0.8)
    assert np.isnan(d[7])
    assert s["n"] == 19 and s["n_triggers"] == 20
    assert abs(s["median_ms"] - 70.0) < 2.0
    assert abs(s["mean_ms"] - 70.0) < 2.0 and s["p95_ms"] < 75.0
    _, s2 = L.match_delays(trig, obs, max_delay_s=0.05)   # nothing within 50 ms
    assert s2["n"] == 0 and s2["median_ms"] is None


def test_roi_brightness_synthetic_video(tmp_path: Path) -> None:
    path = tmp_path / "flash.mp4"
    fps, w, h = 30.0, 64, 48
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        pytest.skip("OpenCV build cannot write mp4v")
    white_from = [10, 40, 70]           # frame indices where white starts; each lasts 15 frames
    n = 100
    for i in range(n):
        on = any(k <= i < k + 15 for k in white_from)
        f = np.full((h, w, 3), 230 if on else 15, np.uint8)
        f[:4, :4] = 0                   # outside the ROI; must not matter
        wr.write(f)
    wr.release()
    t, b = L.roi_brightness(path, L.centre_roi(w, h, 0.5))
    assert t.size == n and abs(t[1] - 1.0 / fps) < 1e-9
    rises = L.detect_transitions(b, t)
    assert rises.size == 3
    for r, k in zip(rises, white_from):     # hard step between frames k-1 and k -> lands inside that interval
        assert (k - 1) / fps <= r < k / fps
        assert int(round(r * fps + 0.5)) == k
    falls = L.detect_transitions(b, t, edge="falling")
    assert [int(round(x * fps + 0.5)) for x in falls] == [k + 15 for k in white_from]
    t2, _ = L.roi_brightness(path, None, fps=240.0, max_frames=10)    # fps override + frame cap
    assert t2.size == 10 and abs(t2[1] - 1 / 240.0) < 1e-12


def test_roi_helpers() -> None:
    assert L.centre_roi(1280, 720, 0.2) == (512, 288, 256, 144)
    assert L.parse_roi("1, 2,30,40") == (1, 2, 30, 40)
    with pytest.raises(ValueError):
        L.parse_roi("1,2,3")
    f = np.zeros((10, 10, 3), np.uint8)
    f[2:4, 2:4] = 100
    assert L.frame_brightness(f, (2, 2, 2, 2)) == 100.0
    assert L.time_scale_ratio(np.array([0.0, 0.25, 0.5]), 2.0) == pytest.approx(0.125)
    assert L.first_transition_offset(np.array([5.0, 7.0]), np.array([1.0, 3.0])) == 4.0
    assert L.latency_cfg({"latency": None})["period_s"] == 1.0
    assert L.latency_cfg({"latency": {"count": 3}})["count"] == 3


def _load_probe():
    spec = importlib.util.spec_from_file_location("latency_probe", ROOT / "scripts" / "latency_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_simulated_headless(tmp_path: Path, config: dict) -> None:
    """Whole loopback flow on a fake camera (60 fps, 50 ms delay): phase-1 median ~ 50 ms + half a frame."""
    probe = _load_probe()
    cfg_path = tmp_path / "config.yaml"
    import yaml
    cfg = dict(config)
    cfg["latency"] = {"settle_s": 0.2, "max_delay_s": 0.15, "out_dir": str(tmp_path)}
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "res.json"
    code = probe.main(["--config", str(cfg_path), "--method", "loopback-camera", "--simulate", "50", "--headless",
                       "--period", "0.2", "--count", "4", "--out", str(out), "--main-latency-ms", "40"])
    assert code == 0
    r = json.loads(out.read_text())
    med = r["phase1"]["summary"]["median_ms"]
    assert r["phase1"]["summary"]["n"] == 8
    assert 50.0 + 8.3 - 6.0 < med < 50.0 + 8.3 + 6.0
    assert r["phase2"]["summary_seen"]["n"] >= 6
    assert r["estimate"]["end_to_end_ms"] == pytest.approx(med + 40.0)


def test_probe_video_analysis(tmp_path: Path, config: dict) -> None:
    """Synthetic 240 fps 'slow-mo' clip: Mac-screen patch 10 ms after hand-off, projection 70 ms after; the video
    clock starts 1.234 s before the first logged flash. Expect ~60 ms projector-vs-screen delay within 1 frame."""
    probe = _load_probe()
    fps, period, n = 240.0, 0.25, 6
    t0_log = 500.0
    sched = L.flash_schedule(period, n, t0=t0_log)
    log_path = tmp_path / "flash_log.json"
    log_path.write_text(json.dumps({"period_s": period, "count": n, "t_on": [a for a, _ in sched],
                                    "t_off": [b for _, b in sched]}))
    vid = tmp_path / "slomo.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 64))
    if not wr.isOpened():
        pytest.skip("OpenCV build cannot write mp4v")
    video_minus_log = 1.234 - t0_log

    def lit(t_video: float, lag: float) -> bool:
        t = t_video - video_minus_log - lag
        return any(a <= t < b for a, b in sched)
    for i in range(int((1.234 + 2 * period * n + 0.3) * fps)):
        tv = i / fps
        f = np.zeros((64, 160, 3), np.uint8)
        f[:, :80] = 220 if lit(tv, 0.010) else 20        # Mac screen (ref)
        f[:, 80:] = 220 if lit(tv, 0.070) else 20        # projection
        wr.write(f)
    wr.release()
    out = tmp_path / "video_res.json"
    code = probe.main(["--config", str(ROOT / "config.yaml"), "--method", "video", str(vid), "--roi", "90,10,60,44",
                       "--ref-roi", "10,10,60,44", "--log", str(log_path), "--out", str(out)])
    assert code == 0
    r = json.loads(out.read_text())
    assert r["summary"]["n"] == 2 * n
    assert abs(r["summary"]["median_ms"] - 60.0) < 1e3 / fps + 0.5
    assert abs(r["direct_vs_ref"]["summary_on"]["median_ms"] - 60.0) < 1e3 / fps + 0.5
    assert abs(r["time_scale_ratio"] - 1.0) < 0.05
