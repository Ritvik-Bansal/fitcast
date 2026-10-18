"""End-to-end latency measurement helpers (pure functions; the hardware loop lives in scripts/latency_probe.py).

Conventions
- A flash cycle is WHITE for `period_s` then BLACK for `period_s`: t_on_i = t0 + 2*i*period, t_off_i = t_on_i + period.
- A brightness signal is a 1-D array of per-frame ROI means with one timestamp per sample. Each timestamp means
  whatever the producer says (frame arrival perf_counter for a live camera, frame_index / fps for a video file);
  detected transition times inherit that meaning, so delays inherit its offset.
- Transitions are found with an adaptive threshold at the midpoint of the robust (percentile) min/max, with
  hysteresis so a single noisy sample near the threshold cannot produce two transitions. The time is sub-frame:
  exposure integrates the change (a frame that saw half the flash reads half bright), see detect_transitions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Defaults for `config["latency"]` keys; config values always win. Proposed for config.yaml `latency:`.
DEFAULTS: dict[str, Any] = {
    "period_s": 1.0,          # white for period_s, then black for period_s (one cycle = 2 * period_s)
    "count": 20,              # flash cycles per phase
    "roi_frac": 0.2,          # default camera ROI: centred box this fraction of width and height
    "max_delay_s": 0.8,       # an observation later than this after a trigger is not matched to it
    "settle_s": 1.5,          # black before phase 1 (camera warm-up / auto exposure settles)
    "robust_pct": 5.0,        # robust min/max = this percentile and 100 - this percentile
    "hysteresis_frac": 0.15,  # +- this fraction of the robust range around the threshold
    "min_contrast": 12.0,     # robust range (0..255 grey levels) below this = camera does not see the flash
    "loop_timeout_s": 2.0,    # round-trip phase: give up on an edge after this long
    "poll_s": 0.001,          # camera sampler thread poll interval
    "out_dir": "out",
}


def latency_cfg(config: dict | None) -> dict:
    """`config["latency"]` (absent or null is fine) merged over DEFAULTS."""
    sub = (config or {}).get("latency") or {}
    return {**DEFAULTS, **sub}


def flash_schedule(period_s: float, n: int, t0: float = 0.0) -> list[tuple[float, float]]:
    """`n` (t_on, t_off) pairs: on at t0 + 2*i*period_s, off period_s later."""
    p = float(period_s)
    return [(t0 + 2.0 * i * p, t0 + 2.0 * i * p + p) for i in range(int(n))]


def robust_range(brightness: np.ndarray, pct: float = DEFAULTS["robust_pct"]) -> tuple[float, float]:
    """(low, high) = the `pct` and `100 - pct` percentiles of the signal (NaNs ignored)."""
    b = np.asarray(brightness, np.float64)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(b, [pct, 100.0 - pct])
    return float(lo), float(hi)


def detect_transitions(brightness: np.ndarray, times: np.ndarray, thr: float | None = None,
                       edge: str = "rising", hysteresis_frac: float = DEFAULTS["hysteresis_frac"],
                       pct: float = DEFAULTS["robust_pct"], method: str = "area") -> np.ndarray:
    """Times at which `brightness` crosses from dark to bright (`edge="rising"`) or bright to dark ("falling").

    thr: crossing level; None = midpoint of the robust min/max (percentiles `pct`, 100-pct). A transition is
    armed only after the signal has been on the other side of thr by `hysteresis_frac * robust range`, and is
    registered when it passes thr by the same margin. A signal that starts bright has no rising edge at sample 0.

    Sub-frame time (`method`):
      "linear": linear interpolation of thr on the last sample pair that straddles it. Exact when the exposure
                smear spans >= 2 frames; up to ~0.09 frame off when the exposure equals the frame period.
      "area"  (default): the step time that explains the partially exposed frames around the crossing. Each
                sample is normalised to its local dark/bright plateaus, f in [0, 1], and
                t_step = left edge of the window + sum((1 - f_k) * frame_width_k). Exact when the exposure equals
                the frame period (dim room, full-duty shutter), the pair midpoint for a hard step. Falls back to
                "linear" when the window is incomplete or the two disagree by more than one frame.
    With an exposure E shorter than the frame period T the step is only known to +-(T - E)/2 whatever the method
    (+-T/2 for a hard step); over many flashes with random phase the error averages out (simulated mean error
    < 1 ms at 30 fps for every E). Returns float64 seconds (empty when the signal has no contrast)."""
    b = np.asarray(brightness, np.float64)
    t = np.asarray(times, np.float64)
    if b.shape != t.shape or b.ndim != 1:
        raise ValueError(f"brightness {b.shape} and times {t.shape} must be equal-length 1-D arrays")
    if edge not in ("rising", "falling"):
        raise ValueError(f"edge must be 'rising' or 'falling', not {edge!r}")
    if method not in ("area", "linear"):
        raise ValueError(f"method must be 'area' or 'linear', not {method!r}")
    ok = np.isfinite(b) & np.isfinite(t)
    b, t = b[ok], t[ok]
    if b.size < 2:
        return np.zeros(0, np.float64)
    lo, hi = robust_range(b, pct)
    rng = hi - lo
    if rng <= 0:
        return np.zeros(0, np.float64)
    level = 0.5 * (lo + hi) if thr is None else float(thr)
    if edge == "falling":                       # same code on the negated signal
        b, level, lo, hi = -b, -level, -hi, -lo
    h = float(hysteresis_frac) * rng
    arm, fire = level - h, level + h
    out: list[float] = []
    armed = b[0] < arm
    last_below = -1                             # start index of the latest pair crossing level upwards
    for i in range(1, b.size):
        if b[i - 1] < level <= b[i]:
            last_below = i - 1
        if not armed:
            if b[i] < arm:
                armed = True
            continue
        if b[i] >= fire and last_below >= 0:
            j = last_below
            b0, b1, t0, t1 = b[j], b[j + 1], t[j], t[j + 1]
            frac = (level - b0) / (b1 - b0) if b1 != b0 else 0.0
            t_lin = float(t0 + frac * (t1 - t0))
            t_area = _area_time(b, t, j, lo, hi) if method == "area" else None
            if t_area is None or abs(t_area - t_lin) > (t1 - t0):
                t_area = t_lin
            out.append(t_area)
            armed = False
    return np.asarray(out, np.float64)


_PLATEAU_N = 3   # samples on each side of the transition window used for the local dark / bright levels


def _area_time(b: np.ndarray, t: np.ndarray, j: int, lo_g: float, hi_g: float) -> float | None:
    """Rising step time from samples j-1..j+2 (pair j, j+1 straddles the level) normalised to the local plateaus;
    None when the window or a plateau does not fit in the signal."""
    a, c = j - 1, j + 2
    if a - 1 < 0 or c + 1 >= b.size:
        return None
    lo = float(np.median(b[max(0, a - _PLATEAU_N):a]))
    hi = float(np.median(b[c + 1:c + 1 + _PLATEAU_N]))
    if hi - lo < 0.5 * (hi_g - lo_g):           # plateau too short / disturbed: use the global levels
        lo, hi = lo_g, hi_g
    if hi <= lo:
        return None
    f = np.clip((b[a:c + 1] - lo) / (hi - lo), 0.0, 1.0)
    edges = 0.5 * (t[a - 1:c + 1] + t[a:c + 2])  # cell boundaries between consecutive samples
    return float(edges[0] + np.sum((1.0 - f) * np.diff(edges)))


def match_delays(trigger_times: np.ndarray, observed_times: np.ndarray,
                 max_delay_s: float = DEFAULTS["max_delay_s"], min_delay_s: float = 0.0) -> tuple[np.ndarray, dict]:
    """Per-trigger delay to the first observation in [trigger + min_delay_s, trigger + max_delay_s]
    (seconds, NaN = missed). min_delay_s = 0 means "nearest LATER observation"; a small negative value tolerates
    two clocks that were only aligned approximately (video method).

    Observations are consumed in order, so one observation never matches two triggers. Summary (milliseconds,
    over matched triggers only): mean_ms, median_ms, p95_ms, min_ms, max_ms, std_ms, n, n_triggers."""
    trig = np.sort(np.asarray(trigger_times, np.float64))
    obs = np.sort(np.asarray(observed_times, np.float64))
    delays = np.full(trig.shape, np.nan)
    k = 0
    for i, tt in enumerate(trig):
        while k < obs.size and obs[k] < tt + min_delay_s:
            k += 1
        if k < obs.size and obs[k] - tt <= max_delay_s:
            delays[i] = obs[k] - tt
            k += 1
    return delays, summarize(delays, n_triggers=int(trig.size))


def summarize(delays_s: np.ndarray, n_triggers: int | None = None) -> dict:
    """mean / median / p95 / min / max / std in ms of the finite entries, plus n and n_triggers."""
    d = np.asarray(delays_s, np.float64)
    d = d[np.isfinite(d)] * 1e3
    s: dict[str, Any] = {"n": int(d.size), "n_triggers": int(n_triggers if n_triggers is not None else d.size)}
    if d.size:
        s.update(mean_ms=float(d.mean()), median_ms=float(np.median(d)), p95_ms=float(np.percentile(d, 95)),
                 min_ms=float(d.min()), max_ms=float(d.max()), std_ms=float(d.std()))
    else:
        s.update(mean_ms=None, median_ms=None, p95_ms=None, min_ms=None, max_ms=None, std_ms=None)
    return s


def centre_roi(width: int, height: int, frac: float = DEFAULTS["roi_frac"]) -> tuple[int, int, int, int]:
    """Centred (x, y, w, h) box covering `frac` of the width and of the height (at least 1 px)."""
    w, h = max(1, int(round(width * frac))), max(1, int(round(height * frac)))
    return (int(width) - w) // 2, (int(height) - h) // 2, w, h


def parse_roi(s: str) -> tuple[int, int, int, int]:
    """'x,y,w,h' -> ints (raises ValueError with the expected format)."""
    parts = [p for p in s.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"ROI must be x,y,w,h (4 integers), got {s!r}")
    x, y, w, h = (int(float(p)) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be positive, got {s!r}")
    return x, y, w, h


def frame_brightness(frame: np.ndarray, roi_xy_wh: tuple[int, int, int, int] | None = None) -> float:
    """Mean grey level (mean over channels) inside the (x, y, w, h) ROI, clipped to the frame; None ROI = whole frame."""
    if roi_xy_wh is None:
        return float(frame.mean())
    x, y, w, h = roi_xy_wh
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi_xy_wh} lies outside the {W}x{H} frame")
    return float(frame[y0:y1, x0:x1].mean())


def roi_brightness(video_path: str | Path, roi_xy_wh: tuple[int, int, int, int] | None,
                   fps: float | None = None, max_frames: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(times, brightness) of a video file: time = frame_index / fps (container fps unless `fps` overrides it,
    e.g. a slow-motion clip whose metadata reports the playback rate), brightness = ROI mean grey level."""
    p = Path(video_path)
    if not p.exists():
        raise RuntimeError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {p} (codec?); convert it with ffmpeg -i in.mov -c:v libx264 out.mp4")
    rate = float(fps) if fps else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if rate <= 0:
        cap.release()
        raise RuntimeError(f"{p} reports no frame rate; pass the capture fps explicitly (--video-fps 240)")
    vals: list[float] = []
    try:
        while not max_frames or len(vals) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            vals.append(frame_brightness(frame, roi_xy_wh))
    finally:
        cap.release()
    b = np.asarray(vals, np.float64)
    return np.arange(b.size, dtype=np.float64) / rate, b


def first_transition_offset(reference_times: np.ndarray, log_times: np.ndarray) -> float | None:
    """Clock offset (video - log) that aligns the first detected reference transition with the first logged one."""
    r, l = np.asarray(reference_times, np.float64), np.asarray(log_times, np.float64)
    if r.size == 0 or l.size == 0:
        return None
    return float(r[0] - l[0])


def time_scale_ratio(observed_times: np.ndarray, expected_spacing_s: float) -> float | None:
    """Median spacing of observed transitions / expected spacing (1.0 = clocks agree; 8.0 = a 240 fps slow-mo
    clip read at its 30 fps playback rate). None with fewer than two transitions."""
    o = np.asarray(observed_times, np.float64)
    if o.size < 2 or expected_spacing_s <= 0:
        return None
    return float(np.median(np.diff(o)) / expected_spacing_s)
