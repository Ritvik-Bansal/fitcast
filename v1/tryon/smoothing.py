"""One Euro filter per keypoint, plus a small constant-velocity look-ahead.

Pose arrives at the inference rate; the renderer asks for the body at "now plus
projector latency". The filter's own derivative estimate drives the look-ahead,
which is clamped so a noisy velocity can never throw the garment off the body.
Beta is expressed for a 720p frame so the defaults hold at other resolutions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, speed_scale: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.speed_scale = speed_scale
        self.x_hat: np.ndarray | None = None
        self.dx_hat: np.ndarray | None = None
        self.t_prev = 0.0

    def reset(self) -> None:
        self.x_hat = None
        self.dx_hat = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x_hat is None:
            self.x_hat = x.copy()
            self.dx_hat = np.zeros_like(x)
            self.t_prev = t
            return self.x_hat
        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_hat
        dx = (x - self.x_hat) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx_hat = a_d * dx + (1 - a_d) * self.dx_hat
        speed = float(np.linalg.norm(self.dx_hat)) * self.speed_scale
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self.x_hat = a * x + (1 - a) * self.x_hat
        self.t_prev = t
        return self.x_hat


@dataclass
class Body:
    kp: dict[str, np.ndarray]
    vis: dict[str, float]
    size: tuple[int, int]
    t: float

    def ok(self, name: str, min_vis: float) -> bool:
        return name in self.kp and self.vis.get(name, 0.0) >= min_vis


class BodySmoother:
    def __init__(self, cfg):
        self.cfg = cfg
        self.filters: dict[str, OneEuro] = {}
        self.vis: dict[str, float] = {}
        self.size = (cfg.camera_width, cfg.camera_height)
        self.t_meas = 0.0
        self.last_seen = -1e9
        self.present = False

    def reset(self) -> None:
        self.filters.clear()
        self.vis.clear()

    def update(self, res) -> None:
        if not res.present:
            self.present = False
            if res.t - self.last_seen > self.cfg.lost_reset_ms / 1000.0:
                self.reset()
            return
        if res.t - self.last_seen > self.cfg.lost_reset_ms / 1000.0:
            self.reset()
        self.present = True
        self.last_seen = res.t
        self.t_meas = res.t
        self.size = res.size
        speed_scale = 720.0 / max(1, res.size[1])
        c = self.cfg
        for name, p in res.kp.items():
            f = self.filters.get(name)
            if f is None:
                f = self.filters[name] = OneEuro(c.one_euro_min_cutoff, c.one_euro_beta, c.one_euro_d_cutoff, speed_scale)
            f(p, res.t)
            v = res.vis.get(name, 0.0)
            self.vis[name] = v if name not in self.vis else 0.6 * self.vis[name] + 0.4 * v

    def predict(self, t: float) -> Body | None:
        if not self.present or not self.filters:
            return None
        ahead = min(max(0.0, t - self.t_meas), 0.2)
        max_px = self.cfg.max_prediction_px * self.size[1] / 720.0
        kp = {}
        for name, f in self.filters.items():
            if f.x_hat is None:
                continue
            step = f.dx_hat * ahead
            n = float(np.linalg.norm(step))
            if n > max_px:
                step *= max_px / n
            kp[name] = (f.x_hat + step).astype(np.float32)
        return Body(kp=kp, vis=dict(self.vis), size=self.size, t=t)
