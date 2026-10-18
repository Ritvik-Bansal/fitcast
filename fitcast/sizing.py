"""Size recommendation: body measurements from front-facing frames, mapped to a size chart."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .types import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, TrackResult,
)

log = logging.getLogger(__name__)

# Fallbacks for keys not yet in config.yaml `sizing:` (proposed to the lead for addition).
_DEFAULTS = {
    "min_visibility": 0.5,                 # landmark counts as visible at or above this
    "chest_fallback_shoulder_frac": 1.15,  # chest width = this * shoulder width when the mask is empty
    "chest_circumference_factor": 1.0,     # frontal chest width -> chart chest value
}
NO_DATA_TEXT = ("Size: --", "Stand facing the mirror")


@dataclass
class Measurements:
    shoulder_cm: float
    chest_cm: float           # frontal width of the clothes mask in the chest band
    torso_cm: float           # shoulder-center to hip-center
    inseam_cm: float | None   # None when ankles/knees are not visible
    px_per_cm: float
    n_frames: int
    ankles: bool = False      # scale came from nose->ankle (True) or the shoulder->hip fallback (False)


def recommend_size(value_cm: float, table: dict, measure: str | None = None) -> str:
    """Return the first size in `table["sizes"]` whose upper bound for `measure` is >= `value_cm`.

    `table` is one entry of `sizing.charts`; `measure` defaults to `table["key"]`.
    Values past the last bound map to the last (largest) size.
    """
    measure = measure or table["key"]
    if measure == "combined":
        measure = "chest"                 # single-measure callers of a combined chart use the chest column
    for size, bound in zip(table["sizes"], table[measure]):
        if bound >= value_cm:
            return size
    return table["sizes"][-1]


class SizeEstimator:
    """Accumulates front-facing TrackResults and turns averaged measurements into a size estimate."""

    def __init__(self, cfg: dict, height_cm: float | None = None) -> None:
        self._min_frontal = float(cfg.get("min_frontal_ratio", 0.0))
        self._min_ankle_frames = int(cfg.get("min_ankle_frames", 10))
        self._label_hold_s = float(cfg.get("label_hold_s", 0.0))
        self._shown_size: str | None = None
        self._cand_size: str | None = None
        self._cand_since = 0.0
        """`cfg` is `config["sizing"]`; `height_cm` defaults to `cfg["height_cm"]`."""
        self.cfg = cfg
        self._min_vis = float(cfg.get("min_visibility", _DEFAULTS["min_visibility"]))
        self._chest_frac = float(cfg.get("chest_fallback_shoulder_frac",
                                         _DEFAULTS["chest_fallback_shoulder_frac"]))
        self._circ_factor = float(cfg.get("chest_circumference_factor",
                                          _DEFAULTS["chest_circumference_factor"]))
        for name, chart in cfg["charts"].items():
            cols = ("shoulder", "chest") if chart["key"] == "combined" else (chart["key"],)
            if any(len(chart[c]) != len(chart["sizes"]) for c in cols):
                raise ValueError(f"sizing.charts.{name}: '{chart['key']}' bounds and 'sizes' differ in length")
        self.height_cm = float(cfg["height_cm"] if height_cm is None else height_cm)
        self._buf: deque[Measurements] = deque(maxlen=int(cfg["average_frames"]))
        self._warned: set[str] = set()

    def set_height(self, height_cm: float) -> None:
        """Set the person's height (cm) and restart averaging."""
        self.height_cm = float(height_cm)
        self.reset()

    def reset(self) -> None:
        self._shown_size = self._cand_size = None
        """Drop all accumulated frames (new person / new height)."""
        self._buf.clear()
        self._warned.clear()

    def update(self, track: TrackResult) -> None:
        """Accumulate one frame; ignored unless `track.ok` and the person faces front."""
        if not (track.ok and track.facing == "front"):
            return
        # skip partially turned frames (facing keeps "front" through the hysteresis window)
        if track.ref_shoulder_width > 0 and track.shoulder_width < self._min_frontal * track.ref_shoulder_width:
            return
        m = self._measure(track)
        if m is not None:
            self._buf.append(m)

    def measurements(self) -> Measurements | None:
        """Mean measurements over the last `average_frames` accepted frames, or None before any."""
        buf = list(self._buf)
        if not buf:
            return None
        # never mix the two scale methods (they differ by up to 2x): prefer ankle-based frames when enough exist
        with_ankles = [m for m in buf if m.ankles]
        if len(with_ankles) >= self._min_ankle_frames:
            buf = with_ankles
        else:
            buf = [m for m in buf if not m.ankles] or buf
        inseams = [m.inseam_cm for m in buf if m.inseam_cm is not None]
        return Measurements(
            shoulder_cm=fmean(m.shoulder_cm for m in buf),
            chest_cm=fmean(m.chest_cm for m in buf),
            torso_cm=fmean(m.torso_cm for m in buf),
            inseam_cm=fmean(inseams) if inseams else None,
            px_per_cm=fmean(m.px_per_cm for m in buf),
            n_frames=len(buf),
            ankles=all(m.ankles for m in buf),
        )

    def _stable_size(self, size: str) -> str:
        """Displayed size: a new candidate must persist `label_hold_s` seconds before it replaces the shown one."""
        import time
        now = time.perf_counter()
        if size == self._shown_size:
            self._cand_size, self._cand_since = size, now
            return size
        if size != self._cand_size:
            self._cand_size, self._cand_since = size, now
        if self._shown_size is None or now - self._cand_since >= self._label_hold_s:
            self._shown_size = size
        return self._shown_size

    def recommendation(self, size_table: str | None = None) -> tuple[str, str]:
        """(size_text, measure_text) for the wall panel; `size_table` comes from the current outfit."""
        m = self.measurements()
        if m is None:
            return NO_DATA_TEXT
        table = self._table(size_table)
        chest = m.chest_cm * self._circ_factor
        shoulder = m.shoulder_cm * float(self.cfg.get("shoulder_landmark_factor", 1.0))
        key = table["key"]
        if key == "combined":
            sizes = table["sizes"]
            i_s = sizes.index(recommend_size(shoulder, table, "shoulder"))
            i_c = sizes.index(recommend_size(chest, table, "chest"))
            size = sizes[int(round((i_s + i_c) / 2))]
            size_text = f"Size {self._stable_size(size)} (estimate)"
        else:
            value = {"shoulder": shoulder, "chest": chest, "torso": m.torso_cm, "inseam": m.inseam_cm}.get(key)
            size_text = "Size: --" if value is None else f"Size {self._stable_size(recommend_size(value, table))} (estimate)"
        inseam = "--" if m.inseam_cm is None else f"{m.inseam_cm:.0f} cm"
        measure_text = (f"Shoulder {m.shoulder_cm:.0f} cm  Chest {chest:.0f} cm  "
                        f"Torso {m.torso_cm:.0f} cm  Inseam {inseam}")
        return size_text, measure_text

    # ---- internals -------------------------------------------------------------------------

    def _table(self, name: str | None) -> dict:
        charts, default = self.cfg["charts"], self.cfg["default_table"]
        name = name or default
        if name not in charts:
            self._warn_once(f"table:{name}", "sizing: unknown size table '%s', using '%s'", name, default)
            name = default
        return charts[name]

    def _warn_once(self, key: str, msg: str, *args: object) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    def _measure(self, track: TrackResult) -> Measurements | None:
        xy = track.joints[:, :2].astype(np.float64)
        vis = track.visibility >= self._min_vis
        if not vis[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all():
            return None
        h, w = track.clothes_mask.shape[:2]
        core = xy[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
        if (core < 0).any() or (core[:, 0] >= w).any() or (core[:, 1] >= h).any():
            return None                      # torso must be fully inside the frame to measure it
        sh_c = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
        hip_c = (xy[L_HIP] + xy[R_HIP]) / 2
        torso_px = float(np.linalg.norm(hip_c - sh_c))
        shoulder_px = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
        if torso_px <= 0:
            return None

        ankles_seen = bool(vis[L_ANKLE] and vis[R_ANKLE] and vis[NOSE])
        if ankles_seen:
            ankle_c = (xy[L_ANKLE] + xy[R_ANKLE]) / 2
            px_per_cm = float(np.linalg.norm(ankle_c - xy[NOSE])) / (
                self.height_cm * self.cfg["nose_to_ankle_ratio"])
        else:
            px_per_cm = torso_px / (self.height_cm * self.cfg["shoulder_to_hip_ratio"])
        if px_per_cm <= 0:
            return None

        chest_px = self._chest_px(track.clothes_mask, sh_c[1], hip_c[1], shoulder_px)
        inseam_px = self._inseam_px(xy, vis) if ankles_seen else None
        return Measurements(
            shoulder_cm=shoulder_px / px_per_cm,
            chest_cm=chest_px / px_per_cm,
            torso_cm=torso_px / px_per_cm,
            inseam_cm=None if inseam_px is None else inseam_px / px_per_cm,
            px_per_cm=px_per_cm,
            n_frames=1,
            ankles=ankles_seen,
        )

    def _chest_px(self, mask: np.ndarray, sh_y: float, hip_y: float, shoulder_px: float) -> float:
        """Horizontal extent of the mask over the chest band rows; shoulder-based fallback if empty."""
        f0, f1 = self.cfg["chest_mask_band_frac"]
        span = hip_y - sh_y
        h = mask.shape[0]
        r0 = min(max(round(sh_y + f0 * span), 0), h)
        r1 = min(max(round(sh_y + f1 * span) + 1, 0), h)
        if r1 > r0:
            band = mask[r0:r1] > 0
            rows = band.any(axis=1)
            if rows.any():
                # narrowest occupied row in the band = torso width below the sleeves, above the hem
                first = band.argmax(axis=1)
                last = band.shape[1] - 1 - band[:, ::-1].argmax(axis=1)
                widths = (last - first)[rows]
                return float(np.percentile(widths, float(self.cfg.get("chest_row_percentile", 25))))
        self._warn_once("chest", "sizing: clothes mask empty in chest band, using %.2f x shoulder width",
                        self._chest_frac)
        return self._chest_frac * shoulder_px

    @staticmethod
    def _inseam_px(xy: np.ndarray, vis: np.ndarray) -> float | None:
        """Mean hip->knee->ankle polyline length over sides whose knee and ankle are visible."""
        legs = [float(np.linalg.norm(xy[k] - xy[hp]) + np.linalg.norm(xy[a] - xy[k]))
                for hp, k, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
                if vis[k] and vis[a]]
        return fmean(legs) if legs else None
