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
                            "scale": float(s.backingScaleFactor()), "main": i == 0,
                            "name": str(s.localizedName()), "cocoa_index": i})
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


def cocoa_place_borderless(title: str, cocoa_index: int) -> tuple[int, int, int, int] | None:
    """Make the OpenCV Cocoa window `title` a borderless window covering NSScreen `cocoa_index`, above the
    menu bar. OpenCV's own WND_PROP_FULLSCREEN uses the MAIN screen's frame, so it never lands on a projector.
    Returns the resulting window frame (x, y, w, h in Cocoa points) or None when AppKit is unavailable."""
    try:
        import AppKit  # type: ignore
    except Exception:  # pragma: no cover
        return None
    app = AppKit.NSApplication.sharedApplication()
    win = next((w for w in app.windows() if str(w.title()) == title), None)
    screens = AppKit.NSScreen.screens()
    if win is None or cocoa_index >= len(screens):
        return None
    fr = screens[cocoa_index].frame()
    win.setStyleMask_(AppKit.NSWindowStyleMaskBorderless)
    win.setHasShadow_(False)
    win.setBackgroundColor_(AppKit.NSColor.blackColor())
    win.setLevel_(AppKit.NSScreenSaverWindowLevel)          # above the menu bar / dock of that display
    win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                               | AppKit.NSWindowCollectionBehaviorStationary)
    win.setFrame_display_(fr, True)
    win.orderFrontRegardless()
    got = win.frame()
    return int(got.origin.x), int(got.origin.y), int(got.size.width), int(got.size.height)


class CocoaProjector:
    """Borderless NSWindow on one NSScreen whose layer contents are set to a CGImage each frame.
    ~2 ms per 1080p frame versus ~19 ms for cv2.imshow through OpenCV's Cocoa window."""

    def __init__(self, cocoa_index: int, size: tuple[int, int]) -> None:
        import AppKit, Quartz  # type: ignore
        self._AppKit, self._Q = AppKit, Quartz
        fr = AppKit.NSScreen.screens()[cocoa_index].frame()
        self.win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            fr, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        self.win.setLevel_(AppKit.NSScreenSaverWindowLevel)
        self.win.setBackgroundColor_(AppKit.NSColor.blackColor())
        self.win.setHasShadow_(False)
        self.win.setIgnoresMouseEvents_(True)
        self.win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                        | AppKit.NSWindowCollectionBehaviorStationary)
        self.view = AppKit.NSView.alloc().initWithFrame_(((0, 0), (fr.size.width, fr.size.height)))
        self.view.setWantsLayer_(True)
        self.win.setContentView_(self.view)
        self.view.layer().setBackgroundColor_(Quartz.CGColorCreateGenericRGB(0, 0, 0, 1))
        self.view.layer().setContentsGravity_("resize")
        self.win.orderFrontRegardless()
        self.w, self.h = size
        self._rgba = [np.zeros((self.h, self.w, 4), np.uint8) for _ in range(2)]
        self._flip = 0
        self._cs = Quartz.CGColorSpaceCreateDeviceRGB()
        self.frame = (int(fr.origin.x), int(fr.origin.y), int(fr.size.width), int(fr.size.height))
        # the window-server commit blocks ~10 ms; do it on a worker so the tracking loop never waits for it
        import threading
        self._cv = threading.Condition()
        self._pending: bytes | None = None
        self._quit = False
        self._worker = threading.Thread(target=self._run, name="projector", daemon=True)
        self._worker.start()
        self.commit_ms = 0.0

    def show(self, bgr: np.ndarray) -> None:
        """Hand the frame to the commit thread (latest frame wins; never queues)."""
        self._flip ^= 1
        buf = self._rgba[self._flip]
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA, dst=buf)
        data = buf.tobytes()
        with self._cv:
            self._pending = data
            self._cv.notify()

    def _run(self) -> None:
        import time as _t
        Q = self._Q
        while True:
            with self._cv:
                while self._pending is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    return
                data, self._pending = self._pending, None
            t0 = _t.perf_counter()
            provider = Q.CGDataProviderCreateWithData(None, data, len(data), None)
            img = Q.CGImageCreate(self.w, self.h, 8, 32, self.w * 4, self._cs,
                                  Q.kCGImageAlphaNoneSkipLast | Q.kCGBitmapByteOrderDefault,
                                  provider, None, False, Q.kCGRenderingIntentDefault)
            Q.CATransaction.begin()
            Q.CATransaction.setDisableActions_(True)
            self.view.layer().setContents_(img)
            Q.CATransaction.commit()
            Q.CATransaction.flush()
            self.commit_ms = (_t.perf_counter() - t0) * 1e3

    def close(self) -> None:
        with self._cv:
            self._quit = True
            self._cv.notify()
        self._worker.join(timeout=1.0)
        try:
            self.win.orderOut_(None)
            self.win.close()
        except Exception:  # pragma: no cover
            pass


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
        self.second: dict | None = None
        self.origin = self._resolve_origin(projector_xy)
        self._black = np.zeros((self.ph, self.pw, 3), np.uint8)

    def _resolve_origin(self, override: tuple[int | None, int | None]) -> tuple[int, int]:
        screens = detect_displays() if not self.headless else []
        second = next((s for s in screens if not s["main"]), None)
        self.second = second
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
        self.cocoa: CocoaProjector | None = None
        if fullscreen and self.second is not None and self.cfg.get("projector_sink", "cocoa") == "cocoa":
            try:
                self.cocoa = CocoaProjector(self.second["cocoa_index"], (self.pw, self.ph))
                self.proj_open = self.fullscreen = True
                log.info("projector: CoreGraphics layer window on '%s' frame %s", self.second.get("name"), self.cocoa.frame)
                return
            except Exception as e:  # pragma: no cover
                log.warning("CocoaProjector failed (%s); falling back to the OpenCV window", e)
                self.cocoa = None
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
        if on:
            cv2.imshow(self.proj_name, self._black)      # realise the NSWindow before touching it
            cv2.waitKey(1)
            placed = None
            if self.second is not None and "cocoa_index" in self.second:
                placed = cocoa_place_borderless(self.proj_name, self.second["cocoa_index"])
            if placed is not None:
                log.info("projector window placed borderless on '%s' at cocoa frame %s",
                         self.second.get("name", "?"), placed)
            else:
                log.warning("AppKit placement unavailable; falling back to OpenCV fullscreen (main screen only)")
                cv2.setWindowProperty(self.proj_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.proj_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.proj_name, self.pw, self.ph)
        self.fullscreen = on

    def toggle_fullscreen(self) -> None:
        if getattr(self, "cocoa", None) is not None:
            return                      # the CoreGraphics window is always exactly the projector screen
        self.set_fullscreen(not self.fullscreen)

    def open_debug(self) -> None:
        if self.headless:
            return
        cv2.namedWindow(self.dbg_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.dbg_name, 40, 60)                    # on the laptop screen, not under the projector window
        try:
            cv2.setWindowProperty(self.dbg_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        self.debug_open = True

    def show_projector(self, img: np.ndarray) -> None:
        if not self.proj_open:
            return
        if getattr(self, "cocoa", None) is not None:
            self.cocoa.show(img)
        else:
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
        if getattr(self, "cocoa", None) is not None:
            self.cocoa.close()
        if not self.headless:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------------------------
def draw_debug(frame: np.ndarray, track: TrackResult, info: dict, min_vis: float = 0.5,
               composite: np.ndarray | None = None, show_overlay: bool = True, mask_tint: bool = True) -> np.ndarray:
    """Camera frame + skeleton, mask outline/tint, head box and status text. Optionally side-by-side with the composite."""
    vis = frame.copy()
    if show_overlay and track.ok:
        if mask_tint:
            tinted = cv2.addWeighted(vis, 0.55, np.full_like(vis, (0, 110, 0)), 0.45, 0)
            cv2.copyTo(tinted, track.clothes_mask, vis)
        else:   # outline only, so the real projected light on the clothes stays visible in recordings
            cnts, _ = cv2.findContours(track.clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
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
