"""Synthetic skeleton generator for orientation / gesture tests.

Body frame (meters): lx = toward the wearer's LEFT when they face the camera,
ly = down, lf = forward (the direction the person looks).  World/camera frame:
x right, y down, z away from the camera (smaller z = closer).

  x = lx*cos(yaw) - lf*sin(yaw)
  z = -lx*sin(yaw) - lf*cos(yaw)

yaw 0 = facing camera, 90 = wearer's LEFT side toward camera, 180 = back,
270 = right side.  Orthographic projection, SCALE px per meter.
"""
from __future__ import annotations

import math

import numpy as np

from tryon.pose import LANDMARKS, PoseResult

W, H = 1280, 720
SCALE = 290.0  # px / m  -> ~500 px tall person
FPS = 30
DT = 1.0 / FPS

# (lx, ly, lf) with origin at hip centre
BODY = {
    "nose": (0.0, -0.74, 0.10),
    "left_eye": (0.03, -0.77, 0.08), "right_eye": (-0.03, -0.77, 0.08),
    "left_ear": (0.08, -0.75, -0.02), "right_ear": (-0.08, -0.75, -0.02),
    "left_shoulder": (0.20, -0.50, 0.0), "right_shoulder": (-0.20, -0.50, 0.0),
    "left_elbow": (0.24, -0.22, 0.0), "right_elbow": (-0.24, -0.22, 0.0),
    "left_wrist": (0.25, 0.02, 0.0), "right_wrist": (-0.25, 0.02, 0.0),
    "left_hip": (0.09, 0.0, 0.0), "right_hip": (-0.09, 0.0, 0.0),
    "left_knee": (0.09, 0.45, 0.0), "right_knee": (-0.09, 0.45, 0.0),
    "left_ankle": (0.09, 0.90, 0.0), "right_ankle": (-0.09, 0.90, 0.0),
}
WORLD_KEYS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def swap_name(k: str) -> str:
    if k.startswith("left_"):
        return "right_" + k[5:]
    if k.startswith("right_"):
        return "left_" + k[6:]
    return k


def default_face_score(yaw: float) -> float:
    y = yaw % 360.0
    d = min(y, 360.0 - y)
    return 0.9 if d < 70 else 0.0


def make_pose(t, yaw=0.0, cx=640.0, hip_y=400.0, mp_swap_when_back=False,
              face_score="auto", vis=1.0, noise_px=0.0, rng=None, wrists=None,
              frame_idx=0, present=True, vis_overrides=None, face_vis="auto"):
    """Build one PoseResult.

    wrists: optional {'left_wrist': (x, y), 'right_wrist': (x, y)} in image px that
            replace the projected wrist (mp label, i.e. before any swap).
    face_score: 'auto' (0.9 if |yaw|<70 else 0.0), None, or a float.
    """
    if not present:
        return PoseResult(t=t, frame_idx=frame_idx, size=(W, H), present=False)
    if face_score == "auto":
        face_score = default_face_score(yaw)
    r = math.radians(yaw)
    c, s = math.cos(r), math.sin(r)
    kp, world = {}, {}
    for name, (lx, ly, lf) in BODY.items():
        x = lx * c - lf * s
        z = -lx * s - lf * c
        kp[name] = np.array([cx + x * SCALE, hip_y + ly * SCALE], np.float32)
        if name in WORLD_KEYS:
            world[name] = np.array([x, ly, z], np.float32)
    if wrists:
        for k, v in wrists.items():
            kp[k] = np.array(v, np.float32)
    if noise_px and rng is not None:
        for k in kp:
            kp[k] = kp[k] + rng.normal(0, noise_px, 2).astype(np.float32)
    visd = {k: vis for k in kp}
    if face_vis == "auto":
        face_vis = vis if c > 0.17 else min(vis, 0.1)   # head looks away: MediaPipe face visibility drops
    if face_vis is not None:
        for k in ("nose", "left_eye", "right_eye"):
            visd[k] = face_vis
    if vis_overrides:
        visd.update(vis_overrides)
    if mp_swap_when_back and c < 0:
        kp = {swap_name(k): v for k, v in kp.items()}
        world = {swap_name(k): v for k, v in world.items()}
        visd = {swap_name(k): v for k, v in visd.items()}
    assert set(kp) == set(LANDMARKS)
    return PoseResult(t=t, frame_idx=frame_idx, size=(W, H), present=True, kp=kp,
                      vis=visd, world=world, face_score=face_score)


def yaw_sweep(rev_s=4.0, direction=+1, start=0.0, total_deg=360.0, fps=FPS, t0=0.0):
    """Yield (t, yaw) for a constant-rate turn."""
    n = int(total_deg / (360.0 / rev_s) * fps)
    for i in range(n + 1):
        t = t0 + i / fps
        yield t, (start + direction * 360.0 / rev_s * (i / fps)) % 360.0


class FakeOrientation:
    def __init__(self, bucket="front", turning=False, transition=None):
        self.bucket, self.turning, self.transition = bucket, turning, transition
