"""Hand gestures from raw (unsmoothed) wrist keypoints.

  swipe       wrist moves sideways fast (relative to the torso, so body sway or a
              turn does not count) while above the hips -> next / previous garment
  raise hand  a wrist above the nose for raise_hold_ms -> toggle shirt / pants
  palm hold   a wrist held still between shoulder and nose height for
              palm_hold_ms -> checkout; progress drives the projected ring

Speeds and distances are in pixels of a 720p frame. Gestures only fire while
the person faces the camera and is not mid-turn, which is when a turning body
would otherwise fling the wrists sideways.
"""
from __future__ import annotations

from collections import deque

import numpy as np

WRISTS = ("left_wrist", "right_wrist")


class GestureDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist = {w: deque() for w in WRISTS}
        self.torso_hist: deque = deque()
        self.last_swipe_t = -1e9
        self.last_swipe_dir = 0
        self.last_any_t = -1e9
        self.raise_since = {w: None for w in WRISTS}
        self.raise_armed = True
        self.last_raise_t = -1e9
        self.palm_since = {w: None for w in WRISTS}
        self.palm_last_ok = {w: -1e9 for w in WRISTS}
        self.palm_armed = True
        self.palm_progress = 0.0
        self.palm_pos: np.ndarray | None = None
        self.raise_progress = 0.0
        self.swipe_speed = 0.0
        self.blocked = ""

    def _reset_timers(self) -> None:
        for w in WRISTS:
            self.raise_since[w] = None
            self.palm_since[w] = None
        self.palm_progress = 0.0
        self.raise_progress = 0.0
        self.palm_pos = None

    def update(self, res, orientation, checkout_ready: bool = True) -> list[str]:
        cfg = self.cfg
        events: list[str] = []
        if not res.present:
            self._reset_timers()
            for d in self.hist.values():
                d.clear()
            self.torso_hist.clear()
            self.blocked = "no person"
            return events
        t = res.t
        kp, vis = res.kp, res.vis
        k = 720.0 / max(1, res.size[1])
        torso_x = float(kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2
        self.torso_hist.append((t, torso_x))
        while self.torso_hist and t - self.torso_hist[0][0] > 0.6:
            self.torso_hist.popleft()
        for w in WRISTS:
            h = self.hist[w]
            if vis.get(w, 0.0) >= cfg.min_visibility:
                h.append((t, float(kp[w][0]), float(kp[w][1])))
            while h and t - h[0][0] > 0.6:
                h.popleft()

        if cfg.gestures_front_only and orientation.bucket != "front":
            self.blocked = f"facing {orientation.bucket}"
        elif orientation.turning or orientation.transition is not None:
            self.blocked = "turning"
        else:
            self.blocked = ""
        if self.blocked:
            self._reset_timers()
            return events

        nose_y = float(kp["nose"][1])
        shoulder_y = float(kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
        hip_y = float(kp["left_hip"][1] + kp["right_hip"][1]) / 2
        cooled = (t - self.last_any_t) * 1000 >= cfg.gesture_cooldown_ms

        # --- swipe -----------------------------------------------------------
        self.swipe_speed = 0.0
        win = cfg.swipe_window_ms / 1000.0
        for w in WRISTS:
            h = self.hist[w]
            if len(h) < 3 or h[-1][0] != t:
                continue
            old = next((s for s in h if t - s[0] <= win), None)
            if old is None or t - old[0] < 0.5 * win:
                continue
            seg = [s for s in h if s[0] >= old[0]]
            if any(s[2] > hip_y for s in seg):
                continue
            t0x = _value_at(self.torso_hist, old[0])
            dx = (h[-1][1] - old[1]) - (torso_x - t0x)
            steps = [abs(b[1] - a[1]) for a, b in zip(seg, seg[1:])]
            if len(steps) < 2 or max(steps) > 0.7 * abs(dx):
                continue  # one-frame keypoint jump (tracker glitch), not a hand moving
            dy = h[-1][2] - old[2]
            v = dx / (t - old[0]) * k
            if abs(v) > abs(self.swipe_speed):
                self.swipe_speed = v
            if (abs(v) >= cfg.swipe_speed_px_s and abs(dx) * k >= cfg.swipe_min_distance_px
                    and abs(dx) > 2.0 * abs(dy) and cooled
                    and (t - self.last_swipe_t) * 1000 >= cfg.swipe_debounce_ms):
                direction = 1 if v > 0 else -1
                if direction == -self.last_swipe_dir and t - self.last_swipe_t < 1.2:
                    continue  # the hand travelling back after a swipe
                # +x in the camera image is the wearer's left when they face the camera
                events.append("next" if (direction > 0) == cfg.swipe_left_is_next else "prev")
                self.last_swipe_t = self.last_any_t = t
                self.last_swipe_dir = direction
                h.clear()
                self._reset_timers()
                return events

        # --- raise hand --------------------------------------------------------
        any_raised = False
        self.raise_progress = 0.0
        for w in WRISTS:
            up = vis.get(w, 0.0) >= cfg.min_visibility and kp[w][1] < nose_y
            if not up:
                self.raise_since[w] = None
                continue
            any_raised = True
            if self.raise_since[w] is None:
                self.raise_since[w] = t
            held = (t - self.raise_since[w]) * 1000
            self.raise_progress = max(self.raise_progress, min(1.0, held / cfg.raise_hold_ms))
            if (held >= cfg.raise_hold_ms and self.raise_armed and cooled
                    and (t - self.last_raise_t) * 1000 >= cfg.raise_debounce_ms):
                events.append("toggle_mode")
                self.raise_armed = False
                self.last_raise_t = self.last_any_t = t
                self._reset_timers()
                return events
        if not any_raised:
            self.raise_armed = True

        # --- palm hold -----------------------------------------------------------
        in_zone_any = False
        best, best_w = 0.0, None
        for w in WRISTS:
            ok_vis = vis.get(w, 0.0) >= cfg.min_visibility
            y = float(kp[w][1])
            in_zone = ok_vis and nose_y <= y < shoulder_y
            still = in_zone and _speed(self.hist[w], 0.2) * k < cfg.palm_still_speed_px_s
            in_zone_any |= in_zone
            if still:
                self.palm_last_ok[w] = t
                if self.palm_since[w] is None:
                    self.palm_since[w] = t
            elif self.palm_since[w] is not None and t - self.palm_last_ok[w] > 0.15:
                self.palm_since[w] = None
            if self.palm_since[w] is not None:
                p = (t - self.palm_since[w]) * 1000 / cfg.palm_hold_ms
                if p > best:
                    best, best_w = p, w
        if not checkout_ready or not self.palm_armed:
            best, best_w = 0.0, None
            for w in WRISTS:
                self.palm_since[w] = None
        self.palm_progress = min(1.0, best)
        self.palm_pos = kp[best_w].copy() if best_w else None
        if best >= 1.0 and cooled:
            events.append("checkout")
            self.palm_armed = False
            self.last_any_t = t
            self._reset_timers()
        if not in_zone_any:
            self.palm_armed = True
        return events


def _value_at(hist: deque, t: float) -> float:
    for ts, x in hist:
        if ts >= t:
            return x
    return hist[-1][1] if hist else 0.0


def _speed(h: deque, window: float) -> float:
    if len(h) < 2:
        return 1e9
    t = h[-1][0]
    old = next((s for s in h if t - s[0] <= window), h[0])
    dt = t - old[0]
    if dt < 0.05:
        return 1e9
    return float(np.hypot(h[-1][1] - old[1], h[-1][2] - old[2])) / dt
