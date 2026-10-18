"""One Euro filtering of landmarks (+ velocity prediction) and temporal mask smoothing."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import FACE_IDX, TrackResult


class OneEuroArray:
    """Vectorised One Euro filter over an array of signals sharing one timestamp."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = self.dx_prev = None
        self.t_prev = None

    @property
    def velocity(self) -> np.ndarray | None:
        """Filtered derivative (units per second) from the last call."""
        return self.dx_prev

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.x_prev is None or self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class Smoother:
    """Smooths joints (x, y) with One Euro, predicts forward by `predict_ms`, EMA-smooths the clothes mask."""

    def __init__(self, cfg: dict, mask_alpha: float = 1.0) -> None:
        self.cfg = cfg
        self.filt = OneEuroArray(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
        self.predict_s = float(cfg["predict_ms"]) / 1000.0
        self.max_predict_px = float(cfg["max_predict_px"])
        self.mask_alpha = float(mask_alpha)
        self._prev_xy: np.ndarray | None = None
        self._frozen = np.zeros(33, np.int32)
        self.max_freeze = int(cfg.get("max_freeze_frames", 5))
        self.jump_floor_px = float(cfg.get("max_jump_px_floor", 24))
        self._jump_lim = np.full(33, float(cfg.get("max_jump_sw_limbs", 0.9)), np.float32)
        self._jump_lim[[11, 12, 23, 24]] = float(cfg.get("max_jump_sw", 0.4))       # shoulders, hips
        self._mask_acc: np.ndarray | None = None
        self._mask_out: np.ndarray | None = None
        self._last_ok = False

    def reset(self) -> None:
        self.filt.reset()
        self._mask_acc = None
        self._prev_xy = None
        self._frozen[:] = 0

    def set_predict_ms(self, ms: float) -> None:
        """Set the prediction horizon (call with the measured pipeline latency)."""
        self.predict_s = max(0.0, float(ms)) / 1000.0

    def update(self, track: TrackResult) -> TrackResult:
        """Return `track` with smoothed + predicted joints and a temporally smoothed mask (in place)."""
        if not track.ok:
            if self._last_ok:
                self.reset()
            self._last_ok = False
            return track
        self._last_ok = True
        raw_face = track.joints[list(FACE_IDX), :2].copy()
        # outlier gate: a landmark that jumps farther than a body can move in one frame (torso: `max_jump_sw`,
        # limbs: `max_jump_sw_limbs`, in shoulder widths) keeps its previous position for up to `max_freeze_frames`
        if self._prev_xy is not None and track.shoulder_width > 1.0:
            lim = np.maximum(self._jump_lim * float(max(track.shoulder_width, track.ref_shoulder_width)), self.jump_floor_px)
            jump = np.linalg.norm(track.joints[:, :2] - self._prev_xy, axis=1)
            bad = (jump > lim) & (self._frozen < self.max_freeze)
            if bad.any():
                track.joints[bad, :2] = self._prev_xy[bad]
                self._frozen[bad] += 1
            self._frozen[~bad] = 0
        xy = self.filt(track.joints[:, :2], track.timestamp)
        self._prev_xy = xy.copy()
        if self.predict_s > 0 and self.filt.velocity is not None:
            step = self.filt.velocity * self.predict_s
            np.clip(step, -self.max_predict_px, self.max_predict_px, out=step)
            xy = xy + step
        track.joints[:, :2] = xy
        # the head box was built from raw landmarks: grow it by the smoothing/prediction shift of the face
        d = xy[list(FACE_IDX)] - raw_face
        dx0, dy0 = float(min(d[:, 0].min(), 0.0)), float(min(d[:, 1].min(), 0.0))
        dx1, dy1 = float(max(d[:, 0].max(), 0.0)), float(max(d[:, 1].max(), 0.0))
        h, w = track.clothes_mask.shape[:2]
        x0, y0, x1, y1 = track.head_box
        track.head_box = (max(0, int(np.floor(x0 + dx0))), max(0, int(np.floor(y0 + dy0))),
                          min(w, int(np.ceil(x1 + dx1))), min(h, int(np.ceil(y1 + dy1))))
        if self.mask_alpha < 1.0:
            m = track.clothes_mask
            if self._mask_acc is None or self._mask_acc.shape != m.shape:
                self._mask_acc = m.astype(np.float32)
                self._mask_out = np.empty_like(m)
            else:
                cv2.accumulateWeighted(m, self._mask_acc, self.mask_alpha)
            cv2.threshold(self._mask_acc, 127, 255, cv2.THRESH_BINARY, dst=self._mask_acc_bin())
            np.copyto(self._mask_out, self._mask_acc_bin(), casting="unsafe")
            track.clothes_mask = self._mask_out
        return track

    def _mask_acc_bin(self) -> np.ndarray:
        if getattr(self, "_bin", None) is None or self._bin.shape != self._mask_acc.shape:
            self._bin = np.empty_like(self._mask_acc)
        return self._bin
