"""One Euro filtering of landmarks (+ velocity prediction) and temporal mask smoothing."""
from __future__ import annotations

import math

import numpy as np

from .types import TrackResult


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
        self._mask_acc: np.ndarray | None = None
        self._last_ok = False

    def reset(self) -> None:
        self.filt.reset()
        self._mask_acc = None

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
        xy = self.filt(track.joints[:, :2], track.timestamp)
        if self.predict_s > 0 and self.filt.velocity is not None:
            step = self.filt.velocity * self.predict_s
            np.clip(step, -self.max_predict_px, self.max_predict_px, out=step)
            xy = xy + step
        track.joints[:, :2] = xy
        if self.mask_alpha < 1.0:
            m = track.clothes_mask
            if self._mask_acc is None or self._mask_acc.shape != m.shape:
                self._mask_acc = m.astype(np.float32)
            else:
                cv2_accumulate(self._mask_acc, m, self.mask_alpha)
            track.clothes_mask = (self._mask_acc > 127).astype(np.uint8) * 255
        return track


def cv2_accumulate(acc: np.ndarray, m: np.ndarray, alpha: float) -> None:
    """acc = (1-alpha)*acc + alpha*m, in place, without allocating a float copy of `m`."""
    acc *= (1.0 - alpha)
    acc += m * alpha  # m is uint8 0/255; numpy promotes the product to float
