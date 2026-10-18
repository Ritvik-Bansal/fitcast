"""Mask clipping, head blackout (always last), brightness/gamma, tracking-loss fade, wall gallery panel."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .types import GalleryState, TrackResult


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Compositor:
    def __init__(self, cfg: dict, size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.W, self.H = size
        self.gallery_layer: np.ndarray | None = None
        self.out = np.zeros((self.H, self.W, 3), np.uint8)
        self._lit = np.zeros((self.H, self.W), np.uint8)
        self._lit3 = np.zeros((self.H, self.W, 3), np.uint8)
        self._lut = self._make_lut(float(cfg["brightness"]), float(cfg["gamma"]))
        self.fade_s = float(cfg["fade_ms"]) / 1000.0
        self._fade = 0.0            # 0 = dark, 1 = full; ramps with tracking state
        self._held = np.zeros((self.H, self.W, 3), np.uint8)   # last good output, faded out on tracking loss
        self._last_t: float | None = None
        g = cfg["gallery"]
        self.gallery_enabled = bool(g["enabled"])
        self.rect = [int(x) for x in g["rect"]]
        self.g = g
        self._thumb_cache: dict[tuple[int, int], np.ndarray] = {}
        self.k_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(g["min_gap_to_person_px"]) + 1,) * 2)

    @staticmethod
    def _make_lut(brightness: float, gamma: float) -> np.ndarray:
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.clip((x ** (1.0 / max(gamma, 1e-3))) * brightness, 0.0, 1.0)
        return (y * 255.0 + 0.5).astype(np.uint8)

    @property
    def fade(self) -> float:
        """Current tracking-loss fade factor (1 = fully lit, 0 = black)."""
        return float(self._fade)

    def set_brightness_gamma(self, brightness: float, gamma: float) -> None:
        self._lut = self._make_lut(brightness, gamma)

    def use_separate_gallery_layer(self, enabled: bool = True) -> None:
        """Draw the gallery into `self.gallery_layer` (camera space BGR) instead of the lit frame."""
        self.gallery_layer = np.zeros((self.H, self.W, 3), np.uint8) if enabled else None

    def set_gallery_rect(self, rect: list[int] | tuple[int, int, int, int]) -> None:
        self.rect = [int(x) for x in rect]

    # -- main ----------------------------------------------------------------------------------
    def compose(self, rgba: np.ndarray, track: TrackResult, gallery: GalleryState, now: float | None = None
                ) -> np.ndarray:
        """RGBA garment layer + TrackResult -> BGR frame for the projector (camera space).

        Light only where alpha AND clothes mask agree; head box forced black as the final step.
        """
        now = time.perf_counter() if now is None else now
        if self._last_t is None:
            self._fade = 1.0 if track.ok else 0.0
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_t)
        self._last_t = now
        step = dt / self.fade_s if self.fade_s > 0 else 1.0
        self._fade = min(1.0, self._fade + step) if track.ok else max(0.0, self._fade - step)

        out = self.out
        if self.gallery_layer is not None:
            self.gallery_layer.fill(0)
        if not track.ok:
            # tracking lost: fade the last good frame to black (it already has its head box blacked out)
            if self._fade > 0.0:
                cv2.convertScaleAbs(self._held, dst=out, alpha=self._fade, beta=0)
            else:
                out.fill(0)
            return out

        # premultiply: colour * alpha * mask  (black = no light)
        alpha = rgba[:, :, 3]
        cv2.bitwise_and(alpha, track.clothes_mask, dst=self._lit)
        cv2.merge((self._lit, self._lit, self._lit), dst=self._lit3)
        cv2.multiply(rgba[:, :, :3], self._lit3, dst=out, scale=1.0 / 255.0)
        cv2.LUT(out, self._lut, dst=out)

        if self.gallery_enabled and gallery.visible and gallery.outfits:
            # the gallery lives on the wall: with `gallery_layer` set it is returned separately so the caller can warp
            # it with the wall-distance mapping while the garments use the person's distance
            self._draw_gallery(out if self.gallery_layer is None else self.gallery_layer, gallery, track, now)

        # SAFETY: head blackout is always the last operation on the lit frame (and on the gallery layer).
        x0, y0, x1, y1 = track.head_box
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
            if self.gallery_layer is not None:
                self.gallery_layer[y0:y1, x0:x1] = 0
        np.copyto(self._held, out)
        if self._fade < 1.0:
            cv2.convertScaleAbs(out, dst=out, alpha=self._fade, beta=0)
        return out

    # -- gallery -------------------------------------------------------------------------------
    def _thumb(self, gallery: GalleryState, idx: int, size: int) -> np.ndarray:
        key = (idx, size)
        t = self._thumb_cache.get(key)
        if t is None:
            t = cv2.resize(gallery.outfits[idx].thumb, (size, size), interpolation=cv2.INTER_AREA)
            self._thumb_cache[key] = t
        return t

    def _draw_gallery(self, out: np.ndarray, gal: GalleryState, track: TrackResult, now: float) -> None:
        g = self.g
        x0, y0, x1, y1 = self.rect
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            return
        panel = out[y0:y1, x0:x1]
        panel[:] = g["bg_color"]
        pw, ph = x1 - x0, y1 - y0
        n = len(gal.outfits)
        ts, scale = int(g["thumb_size"]), float(g["current_scale"])
        big = int(ts * scale)
        vertical = ph >= pw
        spacing = int(ts * 1.25)
        # animated virtual index
        t = _ease((now - gal.changed_at) / (float(g["anim_ms"]) / 1000.0)) if g["anim_ms"] > 0 else 1.0
        d = gal.index - gal.prev_index
        if abs(d) > n / 2:  # wrap-around: take the short way
            d -= np.sign(d) * n
        vidx = gal.prev_index + d * t
        accent = tuple(int(c) for c in g["accent_color"])
        text_h = int(70 * float(g["font_scale"]) / 0.55)
        # centre of the strip (leave room for text at the bottom)
        cx, cy = pw // 2, (ph - text_h) // 2
        half_vis = int(g["thumbs_visible"]) // 2
        for k in range(-half_vis - 1, half_vis + 2):
            i = (gal.index + k) % n
            off = (i - vidx)
            if off > n / 2:
                off -= n
            if off < -n / 2:
                off += n
            if abs(off) > half_vis + 0.999:
                continue
            is_cur = (i == gal.index)
            s = big if is_cur else ts
            px, py = (cx, int(cy + off * spacing)) if vertical else (int(cx + off * spacing), cy)
            tx0, ty0 = px - s // 2, py - s // 2
            tx1, ty1 = tx0 + s, ty0 + s
            if tx0 < 0 or ty0 < 0 or tx1 > pw or ty1 > ph - text_h:
                continue
            panel[ty0:ty1, tx0:tx1] = self._thumb(gal, i, s)
            if is_cur:
                cv2.rectangle(panel, (tx0 - 3, ty0 - 3), (tx1 + 2, ty1 + 2), accent, 3)
        fs = float(g["font_scale"])
        lines = [gal.current.name, gal.size_text, gal.measure_text, "Size is an estimate"]
        y = ph - text_h + int(18 * fs / 0.55)
        for li, line in enumerate(lines):
            if not line:
                continue
            col = accent if li == 1 else (235, 235, 235)
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs if li < 2 else fs * 0.75, col, 1, cv2.LINE_AA)
            y += int(20 * fs / 0.55)
        # never light the person: clear panel pixels on/near the person mask
        if track.ok:
            body = track.person_mask if track.person_mask is not None else track.clothes_mask
            near = cv2.dilate(body[y0:y1, x0:x1], self.k_gap)
            panel[near > 0] = 0
