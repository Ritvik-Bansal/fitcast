"""Compose the projector frame and the laptop debug frame.

The garment layer is built once per frame in camera pixels (premultiplied BGRA,
only inside a dirty box), masked to the person, then:
  - projector: warped through the camera->projector homography onto pure black,
    with small UI (name + price by the hip, mode, checkout ring / receipt);
  - debug: composited over the camera frame, mirrored, with keypoints and a HUD.
Black in the projector frame means no light lands on the person.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .state import CHARGING, FAILED, PROCESSING, SUCCESS
from .warp import Layer, SideAssigner, ViewWarper

VISA_BLUE = (113, 31, 26)
VISA_GOLD = (0, 182, 247)
WHITE = (255, 255, 255)
GREEN = (90, 220, 90)
RED = (60, 60, 235)
CYAN = (230, 210, 80)
FONT = cv2.FONT_HERSHEY_DUPLEX
SKELETON = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


def _text(img, text, org, scale, color, thick=1, align="left"):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if align == "right":
        x -= tw
    elif align == "center":
        x -= tw // 2
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw, th


def _checkmark(img, center, size, color, thick):
    cx, cy = center
    pts = np.array([[cx - size, cy], [cx - size * 0.35, cy + size * 0.6], [cx + size, cy - size * 0.7]], np.int32)
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, cfg, catalog):
        self.cfg = cfg
        self.size = (cfg.camera_width, cfg.camera_height)
        self.proj_size = (cfg.projector_width, cfg.projector_height)
        self.views = [Layer.empty(self.size), Layer.empty(self.size)]
        self.final = Layer.empty(self.size)
        self.side = SideAssigner()
        self.presence = 0.0
        self.last_t: float | None = None
        self.last_body = None
        self.start_t = time.monotonic()
        self.modes: dict[str, str] = {}
        self.timings: dict[str, float] = {}
        t0 = time.perf_counter()
        for g in catalog.garments:
            for v in g.views.values():
                v.warper = ViewWarper(v, g.category, g.sleeve_length, cfg)
        print(f"[render] prepared {len(catalog.garments)} garments in {(time.perf_counter() - t0) * 1000:.0f} ms")

    # -- garment layer ----------------------------------------------------
    def build_layer(self, body, state, views: list[tuple[str, float]], mask: np.ndarray | None, now: float) -> Layer:
        cfg = self.cfg
        dt = 0.033 if self.last_t is None else max(0.0, min(0.2, now - self.last_t))
        self.last_t = now
        t = time.perf_counter()
        if body is not None:
            self.presence = min(1.0, self.presence + dt * 1000.0 / max(1.0, cfg.fade_in_ms))
            self.last_body = body
            names = self.side.assign(body.kp)
            for layer in self.views:
                layer.clear()
            for (view, _), layer in zip(views, self.views):
                for category in ("pants", "shirt"):
                    g = state.garment(category)
                    if g is not None:
                        self.modes[category] = g.views[view].warper.render(layer, body, names) or "-"
            self._combine(views)
            self._apply_mask(mask, body)
        else:
            self.presence = max(0.0, self.presence - dt * 1000.0 / max(1.0, cfg.fade_out_ms))
            if self.presence <= 0.0:
                self.final.clear()
        self.timings["warp"] = (time.perf_counter() - t) * 1000
        return self.final

    def _combine(self, views) -> None:
        self.final.clear()
        if len(views) == 1:
            a = self.views[0]
            if not a.is_empty:
                x0, y0, x1, y1 = a.roi
                self.final.img[y0:y1, x0:x1] = a.img[y0:y1, x0:x1]
                self.final.roi = list(a.roi)
            return
        (_, w0), (_, w1) = views
        a, b = self.views
        boxes = [l.roi for l in (a, b) if not l.is_empty]
        if not boxes:
            return
        x0, y0 = min(r[0] for r in boxes), min(r[1] for r in boxes)
        x1, y1 = max(r[2] for r in boxes), max(r[3] for r in boxes)
        mix = a.img[y0:y1, x0:x1].astype(np.float32) * w0 + b.img[y0:y1, x0:x1].astype(np.float32) * w1
        self.final.img[y0:y1, x0:x1] = np.clip(mix, 0, 255).astype(np.uint8)
        self.final.roi = [x0, y0, x1, y1]

    def _apply_mask(self, mask: np.ndarray | None, body) -> None:
        cfg = self.cfg
        if self.final.is_empty:
            return
        x0, y0, x1, y1 = self.final.roi
        if mask is not None:
            if mask.shape[:2] != self.final.img.shape[:2]:
                mask = cv2.resize(mask, (self.final.img.shape[1], self.final.img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = mask[y0:y1, x0:x1]
            if cfg.mask_dilate_px > 0:
                k = 2 * cfg.mask_dilate_px + 1
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            if cfg.mask_feather_px > 0:
                k = 2 * cfg.mask_feather_px + 1
                m = cv2.GaussianBlur(m, (k, k), 0)
            m = m.copy()
        else:
            m = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        # Never light the head, whatever the segmenter says.
        kp = body.kp
        head = np.mean([kp[k] for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")], axis=0)
        ear_d = float(np.linalg.norm(kp["left_ear"] - kp["right_ear"]))
        sh_d = float(np.linalg.norm(kp["left_shoulder"] - kp["right_shoulder"]))
        torso = float(np.linalg.norm((kp["left_shoulder"] + kp["right_shoulder"]) / 2 - (kp["left_hip"] + kp["right_hip"]) / 2))
        r = cfg.head_exclusion_scale * max(0.65 * ear_d, 0.3 * sh_d, 0.18 * torso, 12.0)
        cv2.circle(m, (int(head[0] - x0), int(head[1] - y0)), int(r), 0, -1, cv2.LINE_AA)
        f = m.astype(np.float32) * (1.0 / 255.0)
        roi = self.final.img[y0:y1, x0:x1]
        roi[...] = (roi.astype(np.float32) * f[..., None]).astype(np.uint8)

    # -- projector -----------------------------------------------------------
    def projector_frame(self, calib, state, gestures, now: float) -> np.ndarray:
        cfg = self.cfg
        t = time.perf_counter()
        pw, ph = self.proj_size
        out = np.zeros((ph, pw, 3), np.uint8)
        if (now - self.start_t) * 1000 < cfg.startup_blank_ms:
            self.timings["proj"] = (time.perf_counter() - t) * 1000
            return out
        layer = self.final
        if not layer.is_empty and self.presence > 0:
            x0, y0, x1, y1 = layer.roi
            src = layer.img[y0:y1, x0:x1, :3]
            gain = cfg.garment_brightness * self.presence
            corners = calib.cam_to_proj([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            bx0, by0 = np.floor(corners.min(axis=0)).astype(int)
            bx1, by1 = np.ceil(corners.max(axis=0)).astype(int) + 1
            bx0, by0 = max(0, bx0), max(0, by0)
            bx1, by1 = min(pw, bx1), min(ph, by1)
            if bx1 > bx0 and by1 > by0 and (bx1 - bx0) * (by1 - by0) <= 4 * pw * ph:
                T_src = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
                T_dst = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], np.float64)
                M = T_dst @ calib.H @ T_src
                warped = cv2.warpPerspective(src, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if gain < 0.999:
                    warped = cv2.convertScaleAbs(warped, alpha=gain)
                out[by0:by1, bx0:bx1] = warped
        body = self.last_body if self.presence > 0 else None
        if body is not None:
            self._draw_ui(out, lambda p: calib.cam_to_proj([p])[0], body, state, gestures, now,
                          scale=1.3 * cfg.ui_scale, mirrored=False)
        self.timings["proj"] = (time.perf_counter() - t) * 1000
        return out

    # -- UI shared by projector and debug ------------------------------------
    def _draw_ui(self, img, to_px, body, state, gestures, now, scale: float, mirrored: bool) -> None:
        kp = body.kp
        names = self.side.assign(kp)
        il_hip, ir_hip = kp[names["il_hip"]], kp[names["ir_hip"]]
        s_mid = (kp["left_shoulder"] + kp["right_shoulder"]) / 2
        h_mid = (il_hip + ir_hip) / 2
        torso_px = float(np.linalg.norm(to_px(s_mid) - to_px(h_mid)))
        hip_w = ir_hip - il_hip
        # label beside the hip, on the side with more room in the output image
        right_pt = to_px(ir_hip + hip_w * 0.9)
        left_pt = to_px(il_hip - hip_w * 0.9)
        H, W = img.shape[:2]
        if mirrored:
            right_pt, left_pt = left_pt, right_pt
        anchor, align = (right_pt, "left") if W - right_pt[0] > left_pt[0] else (left_pt, "right")
        g = state.active
        if g is not None and state.checkout not in (SUCCESS, FAILED, PROCESSING):
            lines = [(f"[{state.mode.upper()}]", VISA_GOLD, 0.55), (g.name, WHITE, 0.75), (g.price_text, WHITE, 0.75)]
            y = anchor[1] - 0.5 * 34 * scale
            for text, color, s in lines:
                _text(img, text, (anchor[0], y), s * scale, color, max(1, int(round(scale))), align)
                y += 34 * scale
            if state.toast:
                _text(img, state.toast.text, (anchor[0], anchor[1] - 2.2 * 34 * scale), 0.85 * scale, CYAN,
                      max(1, int(round(scale))), align)
        chest = to_px(s_mid * 0.6 + h_mid * 0.4)
        if state.checkout == CHARGING and gestures.palm_pos is not None:
            c = to_px(gestures.palm_pos)
            r = int(46 * scale)
            cv2.ellipse(img, (int(c[0]), int(c[1] - r * 0.4)), (r, r), -90, 0, 360 * state.checkout_progress,
                        VISA_GOLD, max(3, int(8 * scale)), cv2.LINE_AA)
            _text(img, "HOLD TO BUY", (c[0], c[1] - r * 1.8), 0.6 * scale, VISA_GOLD, max(1, int(scale)), "center")
        elif state.checkout == PROCESSING:
            r = int(40 * scale)
            a0 = (now * 360.0) % 360
            cv2.ellipse(img, (int(chest[0]), int(chest[1])), (r, r), a0, 0, 270, VISA_GOLD, max(3, int(7 * scale)), cv2.LINE_AA)
            _text(img, "PROCESSING", (chest[0], chest[1] + r + 34 * scale), 0.7 * scale, WHITE, max(1, int(scale)), "center")
        elif state.checkout in (SUCCESS, FAILED):
            self._receipt(img, chest, max(280.0, min(560.0, 1.4 * torso_px)) * self.cfg.ui_scale, state)

    def _receipt(self, img, center, width: float, state) -> None:
        g, res = state.checkout_garment, state.checkout_result
        ok = state.checkout == SUCCESS
        w, h = int(width), int(width * 0.62)
        x0, y0 = int(center[0] - w / 2), int(center[1] - h / 2)
        s = width / 300.0
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), VISA_BLUE if ok else (40, 40, 160), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0 + h - int(12 * s)), (x0 + w, y0 + h), VISA_GOLD, -1)
        if ok:
            _checkmark(img, (x0 + int(34 * s), y0 + int(40 * s)), 16 * s, GREEN, max(2, int(5 * s)))
            _text(img, "PAID", (x0 + int(64 * s), y0 + int(52 * s)), 1.05 * s, WHITE, max(1, int(2 * s)))
        else:
            _text(img, "DECLINED", (x0 + int(18 * s), y0 + int(52 * s)), 1.0 * s, WHITE, max(1, int(2 * s)))
        _text(img, "VISA", (x0 + w - int(16 * s), y0 + int(48 * s)), 0.9 * s, VISA_GOLD, max(1, int(2 * s)), "right")
        if g is not None:
            _text(img, g.name, (x0 + int(18 * s), y0 + int(100 * s)), 0.6 * s, WHITE, max(1, int(1.4 * s)))
            _text(img, g.price_text, (x0 + int(18 * s), y0 + int(138 * s)), 0.85 * s, WHITE, max(1, int(2 * s)))
        msg = "Visa **** 1111" if ok else (res.message[:28] if res else "")
        _text(img, msg, (x0 + w - int(16 * s), y0 + int(138 * s)), 0.45 * s, WHITE, 1, "right")

    # -- debug ------------------------------------------------------------------
    def debug_frame(self, frame: np.ndarray, body, state, gestures, hud: list[str],
                    footer: str, now: float) -> np.ndarray:
        t = time.perf_counter()
        out = frame.copy()
        layer = self.final
        if not layer.is_empty and self.presence > 0:
            x0, y0, x1, y1 = layer.roi
            prem = layer.img[y0:y1, x0:x1].astype(np.float32)
            if self.presence < 1.0:
                prem *= self.presence
            base = out[y0:y1, x0:x1].astype(np.float32)
            out[y0:y1, x0:x1] = np.clip(base * (1.0 - prem[..., 3:4] / 255.0) + prem[..., :3], 0, 255).astype(np.uint8)
        if state.debug_overlays and body is not None:
            for a, b in SKELETON:
                if a in body.kp and b in body.kp:
                    cv2.line(out, tuple(int(v) for v in body.kp[a]), tuple(int(v) for v in body.kp[b]), CYAN, 2, cv2.LINE_AA)
            for name, p in body.kp.items():
                col = GREEN if body.vis.get(name, 0) >= self.cfg.min_visibility else RED
                cv2.circle(out, (int(p[0]), int(p[1])), 4, col, -1, cv2.LINE_AA)
        out = cv2.flip(out, 1)
        W = out.shape[1]
        body_ui = self.last_body if self.presence > 0 else None
        if body_ui is not None:
            self._draw_ui(out, lambda p: np.array([W - 1 - p[0], p[1]]), body_ui, state, gestures, now,
                          scale=0.8, mirrored=True)
        if state.debug_overlays:
            lines = hud
        else:
            lines = hud[:1]
        if lines:
            box_h = 10 + 22 * len(lines)
            sub = out[0:box_h, 0:min(W, 900)]
            sub[...] = (sub.astype(np.float32) * 0.35).astype(np.uint8)
            for i, line in enumerate(lines):
                _text(out, line, (10, 24 + 22 * i), 0.5, WHITE, 1)
        if footer:
            _text(out, footer, (10, out.shape[0] - 12), 0.48, VISA_GOLD, 1)
        self.timings["debug"] = (time.perf_counter() - t) * 1000
        return out
