"""Tests for fitcast.sizing using synthetic front-facing skeletons with known pixel geometry."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fitcast.sizing import SizeEstimator, recommend_size
from fitcast.types import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, TrackResult,
)

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())["sizing"]

H, W = 720, 1280
HEIGHT_CM = 180.0
PPC = 3.0                                              # px per cm the geometry is built with
CX, NOSE_Y, SH_Y, HIP_Y, KNEE_Y = 640.0, 100.0, 250.0, 400.0, 495.0
ANKLE_Y = NOSE_Y + CFG["nose_to_ankle_ratio"] * HEIGHT_CM * PPC   # 575.2
HIP_HALF = 45.0                                        # px, hip/knee/ankle x offset from center


def make_track(shoulder_cm: float = 40.0, *, ankles: bool = True, facing: str = "front",
               chest_cm: float | None = None) -> TrackResult:
    """Front-facing skeleton centered at CX. `chest_cm` draws a rectangular clothes mask that wide."""
    t = TrackResult.empty(H, W)
    t.ok, t.facing = True, facing

    def put(i: int, x: float, y: float, vis: float = 1.0) -> None:
        t.joints[i, :2] = (x, y)
        t.visibility[i] = vis

    half = shoulder_cm * PPC / 2
    put(NOSE, CX, NOSE_Y)
    put(L_SHOULDER, CX + half, SH_Y)       # person's left appears on the image right
    put(R_SHOULDER, CX - half, SH_Y)
    put(L_HIP, CX + HIP_HALF, HIP_Y)
    put(R_HIP, CX - HIP_HALF, HIP_Y)
    put(L_KNEE, CX + HIP_HALF, KNEE_Y)
    put(R_KNEE, CX - HIP_HALF, KNEE_Y)
    if ankles:
        put(L_ANKLE, CX + HIP_HALF, ANKLE_Y)
        put(R_ANKLE, CX - HIP_HALF, ANKLE_Y)
    if chest_cm is not None:
        w = round(chest_cm * PPC)
        x0 = round(CX - w / 2)
        t.clothes_mask[280:341, x0:x0 + w + 1] = 255   # columns x0..x0+w -> extent of w px
    return t


def test_known_geometry_with_ankles() -> None:
    est = SizeEstimator(CFG, height_cm=HEIGHT_CM)
    est.update(make_track(40.0, chest_cm=36.0))
    m = est.measurements()
    assert m is not None and m.n_frames == 1
    assert m.px_per_cm == pytest.approx(PPC, rel=1e-4)
    assert m.shoulder_cm == pytest.approx(40.0, rel=0.02)
    assert m.chest_cm == pytest.approx(36.0, abs=1e-3)
    assert m.torso_cm == pytest.approx((HIP_Y - SH_Y) / PPC, rel=1e-4)
    leg_px = (KNEE_Y - HIP_Y) + (ANKLE_Y - KNEE_Y)
    assert m.inseam_cm == pytest.approx(leg_px / PPC, rel=1e-4)
    size_text, measure_text = est.recommendation()
    assert "estimate" in size_text
    factor = float(CFG.get("chest_circumference_factor", 1.0))
    assert measure_text == f"Shoulder 40 cm  Chest {36.0 * factor:.0f} cm  Torso 50 cm  Inseam 58 cm"


def test_no_ankles_uses_torso_scale_and_no_inseam() -> None:
    est = SizeEstimator(CFG, height_cm=HEIGHT_CM)
    est.update(make_track(40.0, ankles=False))         # empty mask -> chest fallback too
    m = est.measurements()
    assert m is not None
    expected_ppc = (HIP_Y - SH_Y) / (HEIGHT_CM * CFG["shoulder_to_hip_ratio"])
    assert m.px_per_cm == pytest.approx(expected_ppc, rel=1e-4)
    assert m.px_per_cm != pytest.approx(PPC, rel=0.01)  # proves the fallback branch ran
    assert m.inseam_cm is None
    assert m.chest_cm == pytest.approx(1.15 * m.shoulder_cm, rel=1e-4)
    assert est.recommendation()[1].endswith("Inseam --")


def test_non_front_and_not_ok_frames_ignored() -> None:
    est = SizeEstimator(CFG, height_cm=HEIGHT_CM)
    est.update(make_track(facing="side"))
    est.update(make_track(facing="back"))
    lost = make_track()
    lost.ok = False
    est.update(lost)
    assert est.measurements() is None
    assert est.recommendation() == ("Size: --", "Stand facing the mirror")
    est.update(make_track())
    assert est.measurements().n_frames == 1


def test_recommend_size_charts() -> None:
    mens, womens = CFG["charts"]["mens"], CFG["charts"]["womens"]
    assert recommend_size(90, mens) == "S"
    assert recommend_size(102, mens) == "M"
    assert recommend_size(140, mens) == "XXL"
    assert recommend_size(92, womens) == "M"
    assert recommend_size(45, mens, measure="shoulder") == "M"


def test_averaging_alternating_shoulders() -> None:
    est = SizeEstimator(CFG, height_cm=HEIGHT_CM)
    for i in range(30):
        est.update(make_track(38.0 if i % 2 else 42.0))
    m = est.measurements()
    assert m.n_frames == 30
    assert m.shoulder_cm == pytest.approx(40.0, rel=1e-4)


def test_ring_buffer_and_set_height_reset() -> None:
    est = SizeEstimator({**CFG, "average_frames": 4}, height_cm=HEIGHT_CM)
    for _ in range(4):
        est.update(make_track(50.0))
    for _ in range(4):
        est.update(make_track(40.0))
    m = est.measurements()
    assert m.n_frames == 4 and m.shoulder_cm == pytest.approx(40.0, rel=1e-4)
    est.set_height(170.0)
    assert est.measurements() is None
