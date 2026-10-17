"""Every tunable in one dataclass, overridden by config.yaml.

Relative paths in the config resolve against the repo root, so the app and the
tools behave the same no matter which directory they are launched from.
"""
from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # camera
    camera_index: int = 0
    camera_source: str = ""
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    video_loop: bool = True

    # models + inference
    pose_model: str = "models/pose_landmarker_full.task"
    segmenter_model: str = "models/selfie_segmenter.tflite"
    face_model: str = "models/blaze_face_short_range.tflite"
    mask_source: str = "segmenter"
    seg_input_width: int = 256
    seg_threshold: float = 0.5
    seg_every_n_frames: int = 0
    min_pose_confidence: float = 0.5
    min_visibility: float = 0.5
    face_detect_every_n: int = 3

    # smoothing + latency
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.02
    one_euro_d_cutoff: float = 1.0
    projector_latency_ms: float = 60.0
    max_prediction_px: float = 30.0
    lost_reset_ms: float = 500.0

    # garments + fit
    garments_dir: str = "assets/garments"
    catalog: list = field(default_factory=list)
    garment_max_size: int = 1024
    garment_scale: float = 1.0
    torso_width_scale: float = 1.15
    armpit_drop: float = 0.30
    hem_drop: float = 0.05
    collar_raise: float = 0.08
    pants_width_scale: float = 1.35
    waist_raise: float = 0.15
    crotch_drop: float = 0.0
    long_sleeve_elbow_frac: float = 0.47
    max_limb_stretch: float = 2.5

    # mask + light
    mask_dilate_px: int = 8
    mask_feather_px: int = 7
    head_exclusion_scale: float = 1.4
    garment_brightness: float = 0.9
    fade_out_ms: float = 300.0
    fade_in_ms: float = 150.0
    startup_blank_ms: float = 1500.0

    # orientation
    orientation_hysteresis_deg: float = 15.0
    orientation_hold_ms: float = 250.0
    view_crossfade_ms: float = 150.0
    shoulder_ratio_ref: float = 0.70
    orientation_z_sign: float = 1.0
    orientation_z_deadband: float = 0.06
    face_front_threshold: float = 0.5
    face_back_threshold: float = 0.2

    # gestures
    gestures_front_only: bool = True
    swipe_speed_px_s: float = 1000.0
    swipe_min_distance_px: float = 110.0
    swipe_window_ms: float = 160.0
    swipe_debounce_ms: float = 800.0
    swipe_left_is_next: bool = True
    raise_hold_ms: float = 500.0
    raise_debounce_ms: float = 1500.0
    palm_hold_ms: float = 1500.0
    palm_still_speed_px_s: float = 140.0
    gesture_cooldown_ms: float = 600.0
    turn_rate_suppress: float = 0.8

    # projector + windows
    calib_path: str = "calib.json"
    projector_width: int = 1920
    projector_height: int = 1080
    projector_display: int = -1
    projector_backend: str = "auto"
    projector_allow_main_display: bool = False
    debug_window_width: int = 1280
    debug_every_n_when_projecting: int = 2
    show_debug_overlays: bool = True
    ui_scale: float = 1.0

    # checkout
    currency: str = "USD"
    checkout_display_ms: float = 3000.0
    checkout_mock_latency_ms: float = 900.0
    checkout_mock_decline_rate: float = 0.0
    transactions_path: str = "transactions.jsonl"
    env_path: str = ".env"

    # perf
    profile_print_after_s: float = 5.0

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p


def _coerce(name: str, value, default):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        return list(value or [])
    return "" if value is None else str(value)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    data = {}
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    elif path != ROOT / "config.yaml":
        sys.exit(f"[config] {path} not found")
    data.update(overrides or {})
    known = {f.name: f for f in dataclasses.fields(Config)}
    for key, value in data.items():
        if key not in known:
            print(f"[config] ignoring unknown key '{key}'")
            continue
        setattr(cfg, key, _coerce(key, value, getattr(cfg, key)))
    return cfg
