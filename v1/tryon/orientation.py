"""Which way the person faces: front / left / back / right, with hysteresis.

Yaw is 0 facing the camera, 90 when the wearer's left side faces the camera
(they turned to their right), 180 facing away, 270 showing their right side.

  c = (x of MediaPipe-left shoulder - x of MediaPipe-right shoulder) / torso length / ref
      ~ cos(yaw)  (shoulders look narrow when side-on)
  sign of the world-z shoulder difference ~ sign(sin(yaw))  (which shoulder is nearer)

MediaPipe often labels a person facing away as if they faced the camera, which
flips both measurements by 180 degrees. So every measurement yields two
candidates, yaw and yaw + 180, and we keep the one closest to the previous yaw
(people turn continuously). A face detector on the head crop re-anchors the
front/back hemisphere whenever the shoulders are clearly square to the camera.
"""
from __future__ import annotations

import math

import numpy as np

BUCKETS = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}


def cdist(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class OrientationEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        # A missing face only means "facing away" once the detector has proven it can see
        # this wearer's face from the taped spot; otherwise a detector that is simply out
        # of range would flip a front-facing person to the back view.
        self.face_validated = False
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.bucket = "front"
        self.yaw = 0.0
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.transition: tuple[str, str, float] | None = None
        self.face_ema: float | None = None
        self.cue = 0.5
        self.hemisphere: str | None = None
        self.ref = self.cfg.shoulder_ratio_ref
        self.c = 1.0
        self.dz = 0.0
        self.turn_rate = 0.0
        self.last_t: float | None = None
        self.last_ratio: float | None = None

    def update(self, res) -> None:
        cfg = self.cfg
        kp = res.kp
        t = res.t
        dt = 0.033 if self.last_t is None else max(1e-3, min(0.5, t - self.last_t))
        if self.last_t is not None and t - self.last_t > 1.0:
            self.reset()
            dt = 0.033
        self.last_t = t

        s_mid = (kp["left_shoulder"] + kp["right_shoulder"]) / 2
        h_mid = (kp["left_hip"] + kp["right_hip"]) / 2
        torso = float(np.linalg.norm(s_mid - h_mid))
        if torso < 10:
            return
        ratio = float(kp["left_shoulder"][0] - kp["right_shoulder"][0]) / torso
        if self.last_ratio is not None:
            rate = abs(ratio - self.last_ratio) / dt
            self.turn_rate = 0.7 * self.turn_rate + 0.3 * rate
        self.last_ratio = ratio

        # first frame after a reset: seed the filters with the measurement itself
        a = 1.0 - math.exp(-dt / 0.06) if self.initialized else 1.0
        c_new = max(-1.0, min(1.0, ratio / self.ref))
        self.c += a * (c_new - self.c)
        if len(res.world) == 4:
            w = res.world
            tw = float(np.linalg.norm((w["left_shoulder"] + w["right_shoulder"]) / 2 - (w["left_hip"] + w["right_hip"]) / 2))
            if tw > 1e-3:
                dz = float(w["left_shoulder"][2] - w["right_shoulder"][2]) / tw * cfg.orientation_z_sign
                self.dz += a * (dz - self.dz)

        # front/back cue: the head-crop face detector; MediaPipe's own face visibility stays
        # high from behind, so it is only used when the detector is off
        if res.face_score is not None:
            self.face_ema = res.face_score if self.face_ema is None else 0.65 * self.face_ema + 0.35 * res.face_score
            if res.face_score >= 0.6:
                self.face_validated = True
        vis = float(np.mean([res.vis.get(k, 0.0) for k in ("nose", "left_eye", "right_eye")]))
        self.cue = self.face_ema if self.face_ema is not None else vis
        if self.cue >= cfg.face_front_threshold:
            self.hemisphere = "front"
        elif self.cue <= cfg.face_back_threshold and (self.face_validated or self.face_ema is None):
            self.hemisphere = "back"

        c = self.c
        s_mag = math.sqrt(max(0.0, 1.0 - c * c))
        if abs(self.dz) > cfg.orientation_z_deadband:
            s = -math.copysign(s_mag, self.dz)
            base = [math.degrees(math.atan2(s, c))]
        else:
            base = [math.degrees(math.atan2(s_mag, c)), math.degrees(math.atan2(-s_mag, c))]
        cands = [(b + off) % 360.0 for b in base for off in (0.0, 180.0)]

        square = abs(c) > 0.6
        if not self.initialized:
            if not square and len(base) == 1:
                # side-on: MediaPipe labels profiles correctly, so trust them as measured
                new = base[0] % 360.0
            else:
                hemi = self.hemisphere or "front"
                pool = [y for y in cands if (math.cos(math.radians(y)) > 1e-6) == (hemi == "front")] or cands
                ref_yaw = 0.0 if hemi == "front" else 180.0
                new = min(pool, key=lambda y: cdist(y, ref_yaw))
            self.yaw = new
            self.bucket = min(BUCKETS, key=lambda b: cdist(new, BUCKETS[b]))
            self.initialized = True
            return
        if square and self.hemisphere is not None:
            pool = [y for y in cands if (math.cos(math.radians(y)) > 1e-6) == (self.hemisphere == "front")] or cands
        else:
            pool = cands
        new = min(pool, key=lambda y: cdist(y, self.yaw))
        diff = (new - self.yaw + 180.0) % 360.0 - 180.0
        if abs(diff) > 100.0:
            self.yaw = new % 360.0       # hemisphere correction: snap, never sweep through a side view
        else:
            self.yaw = (self.yaw + diff * (1.0 - math.exp(-dt / 0.08))) % 360.0

        # Adapt the "square to camera" shoulder ratio to this wearer.
        if self.bucket in ("front", "back") and abs(ratio) > self.ref:
            self.ref = min(abs(ratio), 1.2)
        else:
            self.ref += (cfg.shoulder_ratio_ref - self.ref) * min(1.0, 0.03 * dt)

        self._update_bucket(t)

    def _update_bucket(self, t: float) -> None:
        cfg = self.cfg
        if cdist(self.yaw, BUCKETS[self.bucket]) <= 45.0 + cfg.orientation_hysteresis_deg:
            self.candidate = None
            return
        nearest = min(BUCKETS, key=lambda b: cdist(self.yaw, BUCKETS[b]))
        if nearest != self.candidate:
            self.candidate = nearest
            self.candidate_since = t
            return
        if (t - self.candidate_since) * 1000.0 >= cfg.orientation_hold_ms:
            self.transition = (self.bucket, nearest, t)
            self.bucket = nearest
            self.candidate = None

    def view_weights(self, now: float) -> list[tuple[str, float]]:
        """Views to draw and their crossfade weights."""
        if self.transition is not None:
            old, new, t0 = self.transition
            f = (now - t0) * 1000.0 / max(1.0, self.cfg.view_crossfade_ms)
            if 0.0 <= f < 1.0:
                return [(old, 1.0 - f), (new, f)]
            self.transition = None
        return [(self.bucket, 1.0)]

    @property
    def turning(self) -> bool:
        return self.turn_rate > self.cfg.turn_rate_suppress or self.candidate is not None
