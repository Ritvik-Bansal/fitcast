"""Mid-air gestures from pose tracks: horizontal swipes and raise-both-hands.

Each wrist is tracked relative to the shoulder center, in shoulder widths, over
the last ``window_s`` seconds. A fast horizontal sweep that stays between head
and hip height emits "next" or "prev"; holding both wrists above the nose for
``raise_hands_s`` emits "toggle_gallery" once.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from fitcast.types import (
    L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, TrackResult,
)

log = logging.getLogger("fitcast.gestures")

# Spec-mandated wrist visibility threshold, used only until config.yaml gains
# gestures.min_wrist_visibility.
_DEFAULT_MIN_WRIST_VISIBILITY = 0.5


class GestureDetector:
    """Turns a stream of ``TrackResult`` into discrete UI events."""

    def __init__(self, cfg: dict) -> None:
        """Read tunables from ``cfg`` (the ``gestures`` section of config.yaml)."""
        self.window_s = float(cfg["window_s"])
        self.min_travel = float(cfg["min_travel_sw"])
        self.min_velocity = float(cfg["min_velocity_sw_per_s"])
        self.cooldown_s = float(cfg["cooldown_s"])
        self.band_top_frac = float(cfg["band_top_frac"])
        self.band_bottom_frac = float(cfg["band_bottom_frac"])
        self.person_right_is_next = bool(cfg["person_right_is_next"])
        self.mirror_x = bool(cfg["mirror_x"])
        self.raise_hands_s = float(cfg["raise_hands_s"])  # <= 0 disables the gesture
        self.log_events = bool(cfg["log_events"])
        self.min_vis = float(cfg.get("min_wrist_visibility", _DEFAULT_MIN_WRIST_VISIBILITY))
        self.min_samples = int(cfg.get("min_samples", 4))   # guards against a single-frame landmark jump
        self.max_velocity = float(cfg.get("max_velocity_sw_per_s", 12.0))
        self.max_travel = float(cfg.get("max_travel_sw", 2.6))
        # Per wrist: (landmark index, label, ring buffer of (timestamp, nx, ny)).
        # The head-to-hip band is checked when a sample arrives; an out-of-band
        # sample clears the buffer, so every buffered window is entirely in band.
        self._wrists = ((L_WRIST, "left", deque()), (R_WRIST, "right", deque()))
        self._raise_start: float | None = None
        self.reset()

    def reset(self) -> None:
        """Forget motion history, cooldown and raise-hands state."""
        self._clear()
        self._cooldown_until = -math.inf
        self._raise_fired = False

    def update(self, track: TrackResult) -> list[str]:
        """Feed one frame; return [] or one of ["next"], ["prev"], ["toggle_gallery"].

        Gestures are ignored (and history cleared) while ``track.ok`` is False
        or the person is not facing the camera ("back"/"side"). No events at
        all are emitted for ``cooldown_s`` after a swipe.
        """
        now = track.timestamp
        if not track.ok or track.facing != "front":
            self._clear()
            return []
        if now < self._cooldown_until:
            return []  # buffers were cleared when the swipe fired; keep them empty

        j, vis = track.joints, track.visibility
        lsx, lsy = float(j[L_SHOULDER, 0]), float(j[L_SHOULDER, 1])
        rsx, rsy = float(j[R_SHOULDER, 0]), float(j[R_SHOULDER, 1])
        sw = math.hypot(lsx - rsx, lsy - rsy)
        if sw <= 0.0:
            self._clear()
            return []
        cx, cy = (lsx + rsx) * 0.5, (lsy + rsy) * 0.5
        nose_y = float(j[NOSE, 1])
        span = (float(j[L_HIP, 1]) + float(j[R_HIP, 1])) * 0.5 - nose_y
        band_top = nose_y + self.band_top_frac * span
        band_bottom = nose_y + self.band_bottom_frac * span

        if self._raise_hands(now, j, vis, nose_y):
            return ["toggle_gallery"]

        for idx, side, buf in self._wrists:
            wx, wy = float(j[idx, 0]), float(j[idx, 1])
            if vis[idx] < self.min_vis or not band_top <= wy <= band_bottom:
                buf.clear()
                continue
            buf.append((now, (wx - cx) / sw, (wy - cy) / sw))
            while buf[0][0] < now - self.window_s:
                buf.popleft()
            t0, nx0, _ = buf[0]
            dt = now - t0
            dx = buf[-1][1] - nx0
            if len(buf) < self.min_samples or dt <= 0.0 or abs(dx) <= self.min_travel:
                continue
            velocity = abs(dx) / dt
            if velocity <= self.min_velocity:
                continue
            if velocity > self.max_velocity or abs(dx) > self.max_travel:
                buf.clear()                 # faster/farther than an arm can move: a landmark jump, not a swipe
                continue
            event = self._direction(dx)
            self._cooldown_until = now + self.cooldown_s
            self._clear()
            if self.log_events:
                log.info("%s t=%.3f wrist=%s travel=%.2fsw velocity=%.2fsw/s",
                         event, now, side, dx, velocity)
            return [event]
        return []

    def _direction(self, dx: float) -> str:
        """Map normalized image-x travel to "next"/"prev".

        Facing the camera, the person's own right is image left (-x); a
        mirrored feed flips that.
        """
        toward_person_right = (dx < 0.0) != self.mirror_x
        return "next" if toward_person_right == self.person_right_is_next else "prev"

    def _raise_hands(self, now: float, j: np.ndarray, vis: np.ndarray, nose_y: float) -> bool:
        """Advance the both-hands-above-head timer; True on the frame it fires."""
        if self.raise_hands_s <= 0.0:
            return False
        raised = (vis[L_WRIST] >= self.min_vis and vis[R_WRIST] >= self.min_vis
                  and j[L_WRIST, 1] < nose_y and j[R_WRIST, 1] < nose_y)
        if not raised:
            self._raise_start = None
            self._raise_fired = False  # re-arm only once hands drop
            return False
        if self._raise_start is None:
            self._raise_start = now
        held = now - self._raise_start
        if self._raise_fired or held < self.raise_hands_s:
            return False
        self._raise_fired = True
        if self.log_events:
            log.info("toggle_gallery t=%.3f wrist=both held=%.2fs", now, held)
        return True

    def _clear(self) -> None:
        """Drop wrist histories and any in-progress raise-hands hold."""
        for _, _, buf in self._wrists:
            buf.clear()
        self._raise_start = None
