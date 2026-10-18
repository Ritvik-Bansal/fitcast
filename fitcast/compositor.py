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
        self.head_fill = int(min(max(int(cfg.get('head_fill_level', 0)), 0), 64))   # <= 25% of full white
        self.lit_px = 0
        self.rect_right: list[int] | None = None
        self.rect_msg: list[int] | None = None
        self.msg_lines: list[str] = []
        self.msg_until = 0.0
        self.msg_qr: np.ndarray | None = None
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

    def set_message_rect(self, rect: list[int] | tuple[int, int, int, int] | None) -> None:
        """Where the stylist's answer is written (camera px, on the wall between the strips)."""
        self.rect_msg = None if rect is None else [int(x) for x in rect]

    def set_message(self, lines: list[str], now: float, duration_s: float, qr: np.ndarray | None = None) -> None:
        """Show `lines` (and an optional QR image, BGR square) in the message panel until `now + duration_s`."""
        self.msg_lines = [str(x) for x in lines][:6]
        self.msg_until = now + float(duration_s)
        self.msg_qr = qr

    def _draw_message(self, out: np.ndarray, track: TrackResult, now: float) -> None:
        if self.rect_msg is None or not self.msg_lines or now > self.msg_until:
            return
        g = self.g
        x0, y0, x1, y1 = self.rect_msg
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
        if x1 - x0 < 60 or y1 - y0 < 30:
            return
        panel = out[y0:y1, x0:x1]
        panel[:] = g["bg_color"]
        fs = float(g["font_scale"])
        accent = tuple(int(c) for c in g["accent_color"])
        step = int(22 * fs / 0.55)
        text_w = x1 - x0
        if self.msg_qr is not None:
            q = (y1 - y0) - 8
            if q >= 24:
                qr = cv2.resize(self.msg_qr, (q, q), interpolation=cv2.INTER_NEAREST)
                panel[4:4 + q, text_w - q - 4:text_w - 4] = qr
                text_w -= q + 12
        max_chars = max(8, int(text_w / (11 * fs / 0.55)))
        y = int(18 * fs / 0.55)
        for li, line in enumerate(self.msg_lines):
            if y > (y1 - y0) - 4:
                break
            cv2.putText(panel, line[:max_chars], (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs if li == 0 else fs * 0.85,
                        accent if li == 0 else (235, 235, 235), 1, cv2.LINE_AA)
            y += step
        if track.ok:   # never light the person (the head is in this band when the beam top is low)
            body = track.person_mask if track.person_mask is not None else track.clothes_mask
            near = cv2.dilate(body[y0:y1, x0:x1], self.k_gap)
            panel[near > 0] = 0
            hx0, hy0, hx1, hy1 = track.head_box
            panel[max(0, hy0 - y0):max(0, hy1 - y0), max(0, hx0 - x0):max(0, hx1 - x0)] = 0

    def set_gallery_rects(self, left: list[int] | tuple[int, int, int, int], right: list[int] | tuple[int, int, int, int] | None) -> None:
        """Two strips: bottoms in `left`, tops in `right` (None = single strip in `left` following the last swipe)."""
        self.rect = [int(x) for x in left]
        self.rect_right = None if right is None else [int(x) for x in right]

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
            target = out if self.gallery_layer is None else self.gallery_layer
            if self.rect_right is None:
                self._draw_gallery(target, gallery, track, now)
            else:
                self._draw_gallery(target, gallery, track, now, self.rect, "bottoms")
                self._draw_gallery(target, gallery, track, now, self.rect_right, "tops")
            self._draw_message(target, track, now)

        self.lit_px = int(cv2.countNonZero(self._lit))
        # SAFETY: the head box is always the last operation on the lit frame: a flat dim fill (`head_fill_level`,
        # default 0 = black, hard cap 25%), never garment pixels. The gallery layer stays black there.
        x0, y0, x1, y1 = track.head_box
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = self.head_fill
            if self.gallery_layer is not None:
                self.gallery_layer[y0:y1, x0:x1] = 0
        np.copyto(self._held, out)
        if self._fade < 1.0:
            cv2.convertScaleAbs(out, dst=out, alpha=self._fade, beta=0)
        return out

    # -- gallery -------------------------------------------------------------------------------
    def _thumb(self, gallery: GalleryState, idx: int, size: int, strip: str | None = None) -> np.ndarray:
        strip = gallery.strip if strip is None else strip
        key = (strip, idx, size)
        t = self._thumb_cache.get(key)
        if t is None:
            o = gallery.outfits[idx]
            part = o.top_front if strip == "tops" else o.bottom_front if strip == "bottoms" else None
            if part is None:
                t = cv2.resize(o.thumb, (size, size), interpolation=cv2.INTER_AREA)
            else:
                # one garment part over the panel background, letterboxed into a square
                bg = np.array(self.g["bg_color"], np.float32)
                h, w = part.shape[:2]
                sc = (size - 8) / max(h, w)
                rs = cv2.resize(part, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
                a = rs[:, :, 3:4].astype(np.float32) / 255.0
                tile = np.empty((size, size, 3), np.float32)
                tile[:] = bg
                y0, x0 = (size - rs.shape[0]) // 2, (size - rs.shape[1]) // 2
                roi = tile[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]]
                roi[:] = rs[:, :, :3].astype(np.float32) * a + roi * (1.0 - a)
                t = tile.astype(np.uint8)
            self._thumb_cache[key] = t
        return t

    def _draw_gallery(self, out: np.ndarray, gal: GalleryState, track: TrackResult, now: float,
                      rect: list[int] | None = None, strip: str | None = None) -> None:
        g = self.g
        x0, y0, x1, y1 = self.rect if rect is None else rect
        strip = gal.strip if strip is None else strip
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            return
        panel = out[y0:y1, x0:x1]
        panel[:] = g["bg_color"]
        pw, ph = x1 - x0, y1 - y0
        items = gal.bottom_cycle if (strip == "bottoms" and gal.bottom_cycle) else list(range(len(gal.outfits)))
        n = len(items)
        ts, scale = int(g["thumb_size"]), float(g["current_scale"])
        big = int(ts * scale)
        vertical = ph >= pw
        spacing = int(ts * 1.25)
        # animated virtual index (the strip follows whichever part was swiped last)
        if strip == "bottoms":
            cur = gal.bottom_index if gal.bottom_index >= 0 else gal.index
            prev = gal.prev_index if gal.strip == "bottoms" else cur        # animate only the strip that was swiped
            if gal.bottom_cycle:                                           # positions within the distinct-bottoms cycle
                cur = items.index(cur) if cur in items else 0
                prev = items.index(prev) if prev in items else cur
        else:
            cur = gal.index
            prev = gal.prev_index if gal.strip != "bottoms" else cur
        t = _ease((now - gal.changed_at) / (float(g["anim_ms"]) / 1000.0)) if g["anim_ms"] > 0 else 1.0
        d = cur - prev
        if abs(d) > n / 2:  # wrap-around: take the short way
            d -= np.sign(d) * n
        vidx = prev + d * t
        accent = tuple(int(c) for c in g["accent_color"])
        text_h = int(70 * float(g["font_scale"]) / 0.55)
        # centre of the strip (leave room for text at the bottom)
        cx, cy = pw // 2, (ph - text_h) // 2
        half_vis = int(g["thumbs_visible"]) // 2
        for k in range(-half_vis - 1, half_vis + 2):
            i = (cur + k) % n
            off = (i - vidx)
            if off > n / 2:
                off -= n
            if off < -n / 2:
                off += n
            if abs(off) > half_vis + 0.999:
                continue
            is_cur = (i == cur)
            s = big if is_cur else ts
            px, py = (cx, int(cy + off * spacing)) if vertical else (int(cx + off * spacing), cy)
            tx0, ty0 = px - s // 2, py - s // 2
            tx1, ty1 = tx0 + s, ty0 + s
            if tx0 < 0 or ty0 < 0 or tx1 > pw or ty1 > ph - text_h:
                continue
            panel[ty0:ty1, tx0:tx1] = self._thumb(gal, items[i], s, strip)
            if is_cur:
                cv2.rectangle(panel, (tx0 - 3, ty0 - 3), (tx1 + 2, ty1 + 2), accent, 3)
        fs = float(g["font_scale"])
        if strip != "outfits":
            cv2.putText(panel, strip.upper(), (8, int(22 * fs / 0.55)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, accent, 1, cv2.LINE_AA)
        if strip == "tops":
            lines = [gal.current.name.split(" + ")[0], gal.size_text, gal.measure_text, "Size is an estimate"]
        elif strip == "bottoms":
            lines = [gal.current.name.split(" + ")[-1], "", "", ""]
        else:
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
