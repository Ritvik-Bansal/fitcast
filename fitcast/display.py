"""Projector window (fullscreen on the second display) + operator debug window and overlay drawing."""
from __future__ import annotations

import logging
import subprocess

import cv2
import numpy as np

from .types import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, R_ANKLE, R_ELBOW, R_HIP, R_KNEE,
                    R_SHOULDER, R_WRIST, TrackResult)

log = logging.getLogger("fitcast.display")

BONES = ((L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW),
         (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE),
         (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE))


def detect_displays() -> list[dict]:
    """Screens as top-left-origin dicts {x, y, w, h, scale, main}. Uses AppKit when available."""
    screens: list[dict] = []
    try:
        import AppKit  # type: ignore
        all_s = AppKit.NSScreen.screens()
        main_h = all_s[0].frame().size.height
        for i, s in enumerate(all_s):
            fr = s.frame()
            # Cocoa: origin bottom-left, y up. Convert to top-left, y down (what cv2.moveWindow expects).
            screens.append({"x": int(fr.origin.x), "y": int(main_h - (fr.origin.y + fr.size.height)),
                            "w": int(fr.size.width), "h": int(fr.size.height),
                            "scale": float(s.backingScaleFactor()), "main": i == 0})
    except Exception as e:  # pragma: no cover
        log.warning("AppKit screen detection unavailable (%s); falling back to system_profiler", e)
        try:
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True,
                                 timeout=10).stdout
            w = h = 0
            for ln in out.splitlines():
                if "Resolution:" in ln:
                    parts = ln.split(":")[1].split()
                    w, h = int(parts[0]), int(parts[2])
                    screens.append({"x": 0 if not screens else screens[-1]["x"] + screens[-1]["w"], "y": 0,
                                    "w": w, "h": h, "scale": 1.0, "main": not screens})
        except Exception:
            pass
    for s in screens:
        log.info("display: %s", s)
    return screens


class Display:
    """Owns the two OpenCV windows. Must be used from the main thread (macOS Cocoa requirement)."""

    def __init__(self, cfg: dict, projector_size: tuple[int, int], projector_xy: tuple[int | None, int | None],
                 headless: bool = False) -> None:
        self.cfg = cfg
        self.pw, self.ph = projector_size
        self.headless = headless
        self.proj_name = cfg["projector_window"]
        self.dbg_name = cfg["debug_window"]
        self.fullscreen = False
        self.debug_open = False
        self.proj_open = False
        self.origin = self._resolve_origin(projector_xy)

    def _resolve_origin(self, override: tuple[int | None, int | None]) -> tuple[int, int]:
        screens = detect_displays() if not self.headless else []
        second = next((s for s in screens if not s["main"]), None)
        if second is None and screens:
            log.warning("Only one display detected. Connect the projector over HDMI as an EXTENDED display "
                        "(System Settings > Displays, not mirrored). Using the main display for now.")
        x = override[0] if override[0] is not None else (second["x"] if second else 0)
        y = override[1] if override[1] is not None else (second["y"] if second else 0)
        return int(x), int(y)

    # -- windows -------------------------------------------------------------------------------
    def open_projector(self, fullscreen: bool = True) -> None:
        if self.headless:
            return
        cv2.namedWindow(self.proj_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.proj_name, *self.origin)
        cv2.resizeWindow(self.proj_name, self.pw, self.ph)
        self.proj_open = True
        if fullscreen:
            self.set_fullscreen(True)
        log.info("projector window at %s fullscreen=%s", self.origin, fullscreen)

    def set_fullscreen(self, on: bool) -> None:
        if self.headless or not self.proj_open:
            return
        cv2.moveWindow(self.proj_name, *self.origin)
        cv2.setWindowProperty(self.proj_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
        self.fullscreen = on

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.fullscreen)

    def open_debug(self) -> None:
        if self.headless:
            return
        cv2.namedWindow(self.dbg_name, cv2.WINDOW_NORMAL)
        self.debug_open = True

    def show_projector(self, img: np.ndarray) -> None:
        if self.proj_open:
            cv2.imshow(self.proj_name, img)

    def show_debug(self, img: np.ndarray) -> None:
        if self.debug_open:
            cv2.imshow(self.dbg_name, img)

    def key(self, wait_ms: int = 1) -> int:
        """Pump the GUI event loop and return the pressed key (-1 for none)."""
        if self.headless:
            return -1
        return cv2.waitKey(wait_ms) & 0xFFFF

    def close(self) -> None:
        if not self.headless:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------------------------
def draw_debug(frame: np.ndarray, track: TrackResult, info: dict, min_vis: float = 0.5,
               composite: np.ndarray | None = None, show_overlay: bool = True) -> np.ndarray:
    """Camera frame + skeleton, mask tint, head box and status text. Optionally side-by-side with the composite."""
    vis = frame.copy()
    if show_overlay and track.ok:
        tinted = cv2.addWeighted(vis, 0.55, np.full_like(vis, (0, 110, 0)), 0.45, 0)
        cv2.copyTo(tinted, track.clothes_mask, vis)
        j, v = track.joints, track.visibility
        for a, b in BONES:
            if v[a] >= min_vis and v[b] >= min_vis:
                cv2.line(vis, (int(j[a, 0]), int(j[a, 1])), (int(j[b, 0]), int(j[b, 1])), (0, 255, 255), 2)
        for i in range(33):
            if v[i] >= min_vis:
                cv2.circle(vis, (int(j[i, 0]), int(j[i, 1])), 3, (0, 0, 255), -1)
        x0, y0, x1, y1 = track.head_box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), 2)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 1)
    if show_overlay:
        lines = [f"FPS {info.get('fps', 0):.1f}  latency {info.get('latency_ms', 0):.0f} ms  "
                 f"pose {info.get('pose_ms', 0):.1f} seg {info.get('seg_ms', 0):.1f} render {info.get('render_ms', 0):.1f} ms",
                 f"facing {track.facing if track.ok else '--'}  ratio {info.get('ratio', 0):.2f}  "
                 f"mask {info.get('mask_mode', '?')}{' (fallback)' if info.get('fallback') else ''}",
                 f"outfit {info.get('outfit', '?')}   calib {info.get('calib', '?')}   {info.get('extra', '')}"]
        y = 22
        for ln in lines:
            cv2.putText(vis, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        if not track.ok:
            cv2.putText(vis, "NO PERSON", (8, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    if composite is not None:
        vis = np.hstack([vis, composite])
    return vis
