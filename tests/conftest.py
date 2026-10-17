"""Shared fixtures: config + a synthetic frontal skeleton in a 1280x720 camera space."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fitcast.types import (L_ANKLE, L_ELBOW, L_EYE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_ELBOW,
                           R_EYE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, L_EAR, R_EAR, TrackResult)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def frontal_joints(cx: float = 640.0, sw: float = 200.0, top: float = 120.0) -> np.ndarray:
    """Synthetic 33x3 joints of a person facing the camera (person's LEFT on image RIGHT)."""
    j = np.zeros((33, 3), np.float32)
    j[NOSE] = (cx, top, 0)
    j[L_EYE] = (cx + 15, top - 10, 0)
    j[R_EYE] = (cx - 15, top - 10, 0)
    j[L_EAR] = (cx + 35, top - 5, 0)
    j[R_EAR] = (cx - 35, top - 5, 0)
    for i in (1, 3):
        j[i] = j[L_EYE]
    for i in (4, 6):
        j[i] = j[R_EYE]
    j[9] = (cx + 10, top + 25, 0)
    j[10] = (cx - 10, top + 25, 0)
    sy = top + 90
    j[L_SHOULDER] = (cx + sw / 2, sy, 0)
    j[R_SHOULDER] = (cx - sw / 2, sy, 0)
    j[L_ELBOW] = (cx + sw / 2 + 20, sy + 150, 0)
    j[R_ELBOW] = (cx - sw / 2 - 20, sy + 150, 0)
    j[L_WRIST] = (cx + sw / 2 + 30, sy + 290, 0)
    j[R_WRIST] = (cx - sw / 2 - 30, sy + 290, 0)
    hy = sy + 260
    j[L_HIP] = (cx + sw * 0.3, hy, 0)
    j[R_HIP] = (cx - sw * 0.3, hy, 0)
    j[L_KNEE] = (cx + sw * 0.3, hy + 160, 0)
    j[R_KNEE] = (cx - sw * 0.3, hy + 160, 0)
    j[L_ANKLE] = (cx + sw * 0.3, hy + 320, 0)
    j[R_ANKLE] = (cx - sw * 0.3, hy + 320, 0)
    return j


def make_track(joints: np.ndarray | None = None, facing: str = "front", ok: bool = True,
               mask_full: bool = True, ts: float = 0.0) -> TrackResult:
    j = frontal_joints() if joints is None else joints
    t = TrackResult.empty(720, 1280, ts)
    t.ok = ok
    t.joints = j.copy()
    t.visibility = np.ones((33,), np.float32)
    if mask_full:
        t.clothes_mask[:] = 255
    t.facing = facing
    t.shoulder_width = float(np.hypot(*(j[L_SHOULDER, :2] - j[R_SHOULDER, :2])))
    t.ref_shoulder_width = t.shoulder_width
    t.head_box = (int(j[R_EAR, 0] - 60), int(j[NOSE, 1] - 120), int(j[L_EAR, 0] + 60), int(j[L_SHOULDER, 1] - 10))
    return t
