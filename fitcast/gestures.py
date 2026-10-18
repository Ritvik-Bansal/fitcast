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
        self.tpose_s = float(cfg.get("tpose_s", 2.0))         # <= 0 disables the T-pose calibration trigger
        self.tpose_level_sw = float(cfg.get("tpose_level_sw", 0.3))
        self.tpose_reach_sw = float(cfg.get("tpose_reach_sw", 1.1))
        self._tpose_since: float | None = None
        self._tpose_fired = False
        self.log_events = bool(cfg["log_events"])
        self.min_vis = float(cfg.get("min_wrist_visibility", _DEFAULT_MIN_WRIST_VISIBILITY))
        self.zone_swipes = bool(cfg.get("zone_swipes", False))
        self.zone_line = float(cfg.get("zone_line_sw", -0.3))            # relative to the hip line (negative = above)
        self.min_travel_zone = float(cfg.get("min_travel_sw_zone", 0.8))
        self.min_velocity_zone = float(cfg.get("min_velocity_sw_per_s_zone", 1.8))
        self._facing_warned = 0.0
        self.max_body_travel = float(cfg.get("max_body_travel_sw", 0.5))
        self.vertical_swipes = bool(cfg.get("vertical_swipes", False))
        self.window_v = float(cfg.get("window_s_vertical", cfg.get("window_s", 0.4)))
        self._miss_logged = 0.0
        self.hand_for_bottoms = str(cfg.get("hand_for_bottoms", "right"))
        self.min_travel_v = float(cfg.get("min_travel_sw_vertical", 0.9))
        self.min_velocity_v = float(cfg.get("min_velocity_sw_per_s_vertical", 1.8))
        self.other_hand_max_travel = float(cfg.get("other_hand_max_travel_sw", 0.5))
        self._body = deque()
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
        self._tpose_since, self._tpose_fired = None, False
        """Forget motion history, cooldown and raise-hands state."""
        self._clear()
        self._cooldown_until = -math.inf
        self._raise_fired = False

    def update(self, track: TrackResult) -> list[str]:
        """Feed one frame; return [] or one of ["next"], ["prev"], ["toggle_gallery"], ["calibrate"];
        with `zone_swipes` the swipe events carry a zone suffix: "next_top", "prev_bottom", ...

        Gestures are ignored (and history cleared) while ``track.ok`` is False
        or the person is not facing the camera ("back"/"side"). No events at
        all are emitted for ``cooldown_s`` after a swipe.
        """
        now = track.timestamp
        if not track.ok or track.facing != "front":
            if track.ok and self.log_events and now - self._facing_warned > 2.0:
                log.info("swipes ignored: facing=%s (ratio %.2f)", track.facing, track.facing_ratio)
                self._facing_warned = now
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
        hip_ny = ((float(j[L_HIP, 1]) + float(j[R_HIP, 1])) * 0.5 - cy) / sw   # hip line, shoulder-width units

        if self._raise_hands(now, j, vis, nose_y):
            return ["toggle_gallery"]
        if self._tpose(now, j, vis, sw):
            return ["calibrate"]

        # a swipe is made standing still: while the body itself moves (walking sideways swings both arms) nothing fires
        self._body.append((now, cx / sw, cy / sw))
        while self._body[0][0] < now - self.window_s:
            self._body.popleft()
        if math.hypot(self._body[-1][1] - self._body[0][1], self._body[-1][2] - self._body[0][2]) > self.max_body_travel:
            for _, _, b in self._wrists:
                b.clear()
            return []

        window = self.window_v if self.vertical_swipes else self.window_s
        for idx, side, buf in self._wrists:
            wx, wy = float(j[idx, 0]), float(j[idx, 1])
            in_band = self.vertical_swipes or band_top <= wy <= band_bottom   # vertical: anywhere in the frame
            if vis[idx] < self.min_vis or not in_band:
                buf.clear()
                continue
            buf.append((now, (wx - cx) / sw, (wy - cy) / sw))
            while buf[0][0] < now - window:
                buf.popleft()
            t0, nx0, ny0 = buf[0]
            dt = now - t0
            dx = buf[-1][1] - nx0
            if self.vertical_swipes:
                # travel = extreme-to-extreme within the window, so a raise that already peaked still counts
                ys = [b[2] for b in buf]
                i_min, i_max = int(np.argmin(ys)), int(np.argmax(ys))
                dy = ys[-1] - ys[0]
                span = ys[i_max] - ys[i_min]
                if abs(dy) < 0.5 * span:          # the hand went up and came back: use the dominant direction
                    dy = span if i_max > i_min else -span
                if len(buf) < self.min_samples or dt <= 0.0 or abs(dy) <= self.min_travel_v:
                    continue
                vel = abs(dy) / dt
                if vel <= self.min_velocity_v:
                    if self.log_events and now - self._miss_logged > 1.0:
                        log.info("vertical swipe too slow: wrist=%s travel=%.2fsw in %.2fs", side, dy, dt)
                        self._miss_logged = now
                    continue
                if vel > self.max_velocity or abs(dy) > self.max_travel:
                    buf.clear()
                    continue
                other = next(b for i2, _, b in self._wrists if i2 != idx)
                if len(other) >= 2 and abs(other[-1][2] - other[0][2]) > self.other_hand_max_travel:
                    buf.clear()                 # both arms moving = not a deliberate swipe
                    continue
                part = "bottom" if side == self.hand_for_bottoms else "top"
                event = ("prev" if dy < 0 else "next") + "_" + part      # image y grows downwards: up = prev
                self._cooldown_until = now + self.cooldown_s
                self._clear()
                if self.log_events:
                    log.info("%s t=%.3f wrist=%s vertical travel=%.2fsw velocity=%.2fsw/s", event, now, side, dy, vel)
                return [event]
            # zone first: swipes high up (above `zone_line_sw`, shoulder-width units from the shoulder centre,
            # negative = above) are for tops; everything lower is for bottoms, which may be subtler
            mean_ny = sum(b[2] for b in buf) / len(buf)
            zone = "top" if mean_ny < hip_ny + self.zone_line else "bottom"
            travel_min = self.min_travel if not self.zone_swipes else self.min_travel_zone
            vel_min = self.min_velocity if not self.zone_swipes else self.min_velocity_zone
            if len(buf) < self.min_samples or dt <= 0.0 or abs(dx) <= travel_min:
                continue
            velocity = abs(dx) / dt
            if velocity <= vel_min:
                continue
            # one-handed: if the other wrist travelled as well, this is arm swing, not a swipe
            other = next(b for i2, _, b in self._wrists if i2 != idx)
            if len(other) >= 2 and abs(other[-1][1] - other[0][1]) > self.other_hand_max_travel:
                buf.clear()
                continue
            if velocity > self.max_velocity or abs(dx) > self.max_travel:
                buf.clear()                 # faster/farther than an arm can move: a landmark jump, not a swipe
                continue
            event = self._direction(dx)
            if self.zone_swipes:
                event += "_" + zone
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

    def _tpose(self, now: float, j: np.ndarray, vis: np.ndarray, sw: float) -> bool:
        """Both arms straight out to the sides (wrists at shoulder height, beyond the shoulders by > tpose_reach_sw)
        held for `tpose_s` seconds -> one "calibrate" event; re-arms when the pose breaks."""
        if self.tpose_s <= 0.0:
            return False
        ok = (vis[L_WRIST] >= self.min_vis and vis[R_WRIST] >= self.min_vis and
              abs(j[L_WRIST, 1] - j[L_SHOULDER, 1]) < self.tpose_level_sw * sw and
              abs(j[R_WRIST, 1] - j[R_SHOULDER, 1]) < self.tpose_level_sw * sw and
              abs(j[L_WRIST, 0] - j[L_SHOULDER, 0]) > self.tpose_reach_sw * sw and
              abs(j[R_WRIST, 0] - j[R_SHOULDER, 0]) > self.tpose_reach_sw * sw and
              (j[L_WRIST, 0] - j[L_SHOULDER, 0]) * (j[R_WRIST, 0] - j[R_SHOULDER, 0]) < 0)   # opposite sides
        if not ok:
            self._tpose_since, self._tpose_fired = None, False
            return False
        if self._tpose_since is None:
            self._tpose_since = now
        if self._tpose_fired or now - self._tpose_since < self.tpose_s:
            return False
        self._tpose_fired = True
        if self.log_events:
            log.info("calibrate (T-pose) t=%.3f held=%.2fs", now, now - self._tpose_since)
        return True

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
        self._body.clear()
        for _, _, buf in self._wrists:
            buf.clear()
        self._raise_start = None
