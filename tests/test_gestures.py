"""Synthetic-track tests for fitcast.gestures.GestureDetector."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from fitcast.gestures import GestureDetector
from fitcast.types import (
    L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, TrackResult,
)

FPS = 30.0
CHEST_Y = 300.0          # between nose (120) and hips (500)
LEFT_X, RIGHT_X = 500.0, 800.0   # 300 px = 1.5 shoulder widths
IDLE_LW = (780.0, 450.0)          # resting left wrist, in band, never moves
ABOVE_HEAD_Y = 80.0


def load_cfg(**overrides: object) -> dict:
    """The real gestures config, optionally with keys overridden."""
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = dict(yaml.safe_load(path.read_text())["gestures"])
    cfg.update(overrides)
    return cfg


def make_track(t: float, rw: tuple[float, float], lw: tuple[float, float],
               facing: str = "front", ok: bool = True,
               rw_vis: float = 1.0, lw_vis: float = 1.0) -> TrackResult:
    """Frontal person, shoulder width 200 px, nose y=120, hips y=500."""
    tr = TrackResult.empty(720, 1280, timestamp=t)
    tr.ok, tr.facing = ok, facing
    j = tr.joints
    j[L_SHOULDER, :2] = (740, 200)
    j[R_SHOULDER, :2] = (540, 200)
    j[L_HIP, :2] = (720, 500)
    j[R_HIP, :2] = (560, 500)
    j[NOSE, :2] = (640, 120)
    j[L_WRIST, :2] = lw
    j[R_WRIST, :2] = rw
    tr.visibility[:] = 1.0
    tr.visibility[L_WRIST] = lw_vis
    tr.visibility[R_WRIST] = rw_vis
    return tr


class Rig:
    """Feeds a detector a 30 fps stream, moving the right wrist (and optionally the left)."""

    def __init__(self, cfg: dict | None = None, **track_kw: object) -> None:
        self.det = GestureDetector(cfg if cfg is not None else load_cfg())
        self.t = 0.0
        self.events: list[tuple[float, str]] = []
        self.track_kw = track_kw

    def step(self, rw: tuple[float, float], lw: tuple[float, float] = IDLE_LW) -> None:
        for e in self.det.update(make_track(self.t, rw, lw, **self.track_kw)):
            self.events.append((self.t, e))
        self.t += 1.0 / FPS

    def move(self, seconds: float, a: tuple[float, float], b: tuple[float, float]) -> None:
        """Linear right-wrist path a -> b, both endpoints inclusive, over ``seconds``."""
        n = max(1, round(seconds * FPS))
        for k in range(n + 1):
            f = k / n
            self.step((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))

    def hold(self, seconds: float, rw: tuple[float, float],
             lw: tuple[float, float] = IDLE_LW) -> None:
        for _ in range(round(seconds * FPS)):
            self.step(rw, lw)

    def hold_until(self, t_end: float, rw: tuple[float, float]) -> None:
        while self.t < t_end:
            self.step(rw)

    def names(self) -> list[str]:
        return [e for _, e in self.events]


def swipe(rig: Rig, x_from: float, x_to: float, seconds: float = 0.3,
          y: float = CHEST_Y) -> None:
    """Rest, sweep the right wrist horizontally, rest again."""
    rig.hold(0.2, (x_from, y))
    rig.move(seconds, (x_from, y), (x_to, y))
    rig.hold(0.3, (x_to, y))


def test_fast_sweep_image_plus_x_is_prev_and_minus_x_is_next() -> None:
    rig = Rig()
    swipe(rig, LEFT_X, RIGHT_X)
    assert rig.names() == ["prev"]   # image +x = person's left; person_right_is_next

    rig = Rig()
    swipe(rig, RIGHT_X, LEFT_X)
    assert rig.names() == ["next"]


def test_slow_drift_does_not_fire() -> None:
    rig = Rig()
    swipe(rig, LEFT_X, RIGHT_X, seconds=2.0)
    assert rig.names() == []


def test_cooldown_blocks_second_swipe_then_allows_later_one() -> None:
    rig = Rig()
    swipe(rig, LEFT_X, RIGHT_X)
    assert rig.names() == ["prev"]
    t_fire = rig.events[0][0]

    # Valid swipe back starting 0.5 s after the first fired: inside cooldown.
    rig.hold_until(t_fire + 0.5, (RIGHT_X, CHEST_Y))
    rig.move(0.3, (RIGHT_X, CHEST_Y), (LEFT_X, CHEST_Y))
    assert rig.names() == ["prev"]

    # Another valid swipe starting 1.2 s after the first fired: allowed.
    rig.hold_until(t_fire + 1.2, (LEFT_X, CHEST_Y))
    rig.move(0.3, (LEFT_X, CHEST_Y), (RIGHT_X, CHEST_Y))
    rig.hold(0.3, (RIGHT_X, CHEST_Y))
    assert rig.names() == ["prev", "prev"]
    assert rig.events[1][0] > t_fire + 1.2


@pytest.mark.parametrize("track_kw", [{"facing": "back"}, {"facing": "side"}, {"ok": False}])
def test_ignored_when_not_frontal_or_not_tracked(track_kw: dict) -> None:
    rig = Rig(**track_kw)
    swipe(rig, LEFT_X, RIGHT_X)
    assert rig.names() == []


def test_low_wrist_visibility_does_not_fire() -> None:
    rig = Rig(rw_vis=0.3)
    swipe(rig, LEFT_X, RIGHT_X)
    assert rig.names() == []


def test_sweep_above_head_does_not_fire() -> None:
    rig = Rig()
    swipe(rig, LEFT_X, RIGHT_X, y=ABOVE_HEAD_Y)
    assert rig.names() == []

    # Starts and ends at chest height but arcs above the nose mid-sweep.
    rig = Rig()
    rig.hold(0.2, (LEFT_X, CHEST_Y))
    mid = ((LEFT_X + RIGHT_X) / 2, ABOVE_HEAD_Y)
    rig.move(0.15, (LEFT_X, CHEST_Y), mid)
    rig.move(0.15, mid, (RIGHT_X, CHEST_Y))
    rig.hold(0.3, (RIGHT_X, CHEST_Y))
    assert rig.names() == []


@pytest.mark.parametrize("mirror_x, right_is_next, x_from, x_to, expected", [
    (False, True, LEFT_X, RIGHT_X, "prev"),
    (True, True, LEFT_X, RIGHT_X, "next"),
    (True, True, RIGHT_X, LEFT_X, "prev"),
    (False, False, LEFT_X, RIGHT_X, "next"),
    (True, False, LEFT_X, RIGHT_X, "prev"),
])
def test_direction_flags(mirror_x: bool, right_is_next: bool,
                         x_from: float, x_to: float, expected: str) -> None:
    rig = Rig(load_cfg(mirror_x=mirror_x, person_right_is_next=right_is_next))
    swipe(rig, x_from, x_to)
    assert rig.names() == [expected]


def test_both_hands_above_head_toggles_gallery_once_and_rearms() -> None:
    up_l, up_r = (720.0, ABOVE_HEAD_Y), (560.0, ABOVE_HEAD_Y)
    rig = Rig()
    rig.hold(1.2, up_r, up_l)
    assert rig.names() == ["toggle_gallery"]

    rig.hold(1.0, up_r, up_l)            # still holding: no repeat
    assert rig.names() == ["toggle_gallery"]

    rig.hold(0.3, (560.0, CHEST_Y), (720.0, CHEST_Y))   # drop hands -> re-arm
    rig.hold(1.2, up_r, up_l)
    assert rig.names() == ["toggle_gallery", "toggle_gallery"]


def test_swipe_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig()
    with caplog.at_level(logging.INFO, logger="fitcast.gestures"):
        swipe(rig, LEFT_X, RIGHT_X)
    msgs = [r.getMessage() for r in caplog.records if r.name == "fitcast.gestures"]
    assert len(msgs) == 1
    assert "prev" in msgs[0] and "wrist=right" in msgs[0] and "velocity=" in msgs[0]


def test_reset_clears_cooldown() -> None:
    rig = Rig()
    swipe(rig, LEFT_X, RIGHT_X)
    rig.det.reset()
    swipe(rig, RIGHT_X, LEFT_X)
    assert rig.names() == ["prev", "next"]
