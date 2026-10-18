"""Windows: the fullscreen projector output on the second display, plus key input.

The projector window is tried as a cv2 window first (moveWindow to the second
display, then fullscreen) and verified with getWindowImageRect. If it did not
land on the projector, it is reopened with pygame in a child process, because
pygame (SDL) and cv2 highgui both drive the macOS NSApplication and clash when
they share one process. Frames reach the child through shared memory.

The calibration tool opens the projector through this same class, so whatever
offset the window ends up with is baked into the homography.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import queue
import subprocess
import sys
import threading
from multiprocessing import shared_memory

import cv2
import numpy as np

from .config import ROOT

PROJECTOR_WINDOW = "Fitcast projector"

_CV2_KEYS = {
    63234: "left", 63235: "right", 63232: "up", 63233: "down",       # macOS Cocoa
    65361: "left", 65363: "right", 65362: "up", 65364: "down",       # GTK
    2424832: "left", 2555904: "right", 2490368: "up", 2621440: "down",  # Windows
    2: "left", 3: "right", 0: "up", 1: "down",
    13: "enter", 10: "enter", 27: "esc", 32: "space", 9: "tab", 8: "backspace", 127: "backspace",
}


def cv2_key_name(code: int) -> str | None:
    if code is None or code < 0:
        return None
    if code in _CV2_KEYS:
        return _CV2_KEYS[code]
    low = code & 0xFF
    if 32 < low < 127:
        return chr(low).lower()
    return None


def list_displays() -> list[dict]:
    """Active displays in global points: [{'x','y','w','h','main','id'}], main first."""
    if sys.platform != "darwin":
        return []
    try:
        cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics")
                                     or "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    except OSError:
        return []

    class CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    class CGSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

    class CGRect(ctypes.Structure):
        _fields_ = [("origin", CGPoint), ("size", CGSize)]

    cg.CGGetActiveDisplayList.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
    cg.CGDisplayBounds.restype = CGRect
    cg.CGMainDisplayID.restype = ctypes.c_uint32
    ids = (ctypes.c_uint32 * 16)()
    count = ctypes.c_uint32(0)
    if cg.CGGetActiveDisplayList(16, ids, ctypes.byref(count)) != 0:
        return []
    main = cg.CGMainDisplayID()
    out = []
    for i in range(count.value):
        r = cg.CGDisplayBounds(ids[i])
        out.append({"id": int(ids[i]), "x": int(r.origin.x), "y": int(r.origin.y),
                    "w": int(r.size.width), "h": int(r.size.height), "main": ids[i] == main})
    out.sort(key=lambda d: not d["main"])
    return out


def pick_display(cfg) -> dict | None:
    displays = list_displays()
    if cfg.projector_display >= 0:
        if cfg.projector_display < len(displays):
            return {**displays[cfg.projector_display], "index": cfg.projector_display}
        print(f"[display] projector_display={cfg.projector_display} but only {len(displays)} display(s) found")
        return None
    for i, d in enumerate(displays):
        if not d["main"]:
            return {**d, "index": i}
    if cfg.projector_allow_main_display and displays:
        return {**displays[0], "index": 0}
    return None


class ProjectorWindow:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = (cfg.projector_width, cfg.projector_height)
        self.backend: str | None = None
        self.fullscreen = True
        self.error = ""
        self.display = pick_display(cfg)
        if self.display is None:
            self.error = ("no second display found; plug in the projector as an extended display "
                          "(or set projector_allow_main_display: true to test on the laptop screen)")
            return
        d = self.display
        if (d["w"], d["h"]) != self.size:
            print(f"[display] target display is {d['w']}x{d['h']} points, projector frame is "
                  f"{self.size[0]}x{self.size[1]}; set the projector to its native 1920x1080 mode")
        want = cfg.projector_backend
        if want in ("auto", "cv2"):
            if self._open_cv2() or want == "cv2":
                return
            print("[display] cv2 window did not land on the projector; switching to pygame")
            cv2.destroyWindow(PROJECTOR_WINDOW)
            cv2.waitKey(1)
        self._open_pygame()

    @property
    def ok(self) -> bool:
        return self.backend is not None

    def geometry(self) -> dict:
        d = self.display or {}
        return {"backend": self.backend, "display": {k: d.get(k) for k in ("x", "y", "w", "h", "index")},
                "size": list(self.size)}

    # -- cv2 ----------------------------------------------------------------
    def _open_cv2(self) -> bool:
        d = self.display
        black = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        cv2.namedWindow(PROJECTOR_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(PROJECTOR_WINDOW, black)
        cv2.waitKey(1)
        cv2.moveWindow(PROJECTOR_WINDOW, d["x"] + 40, d["y"] + 40)
        cv2.waitKey(1)
        cv2.setWindowProperty(PROJECTOR_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for _ in range(12):
            cv2.imshow(PROJECTOR_WINDOW, black)
            cv2.waitKey(25)
        self.backend = "cv2"
        try:
            x, y, w, h = cv2.getWindowImageRect(PROJECTOR_WINDOW)
        except cv2.error:
            return True
        cx = x + w / 2
        main_h = next((m["h"] for m in list_displays() if m["main"]), d["h"])
        cy_top, cy_flip = y + h / 2, main_h - (y + h / 2)
        in_x = d["x"] <= cx < d["x"] + d["w"]
        in_y = d["y"] <= cy_top < d["y"] + d["h"] or d["y"] <= cy_flip < d["y"] + d["h"]
        print(f"[display] cv2 projector window rect {x},{y},{w}x{h}; target display {d['x']},{d['y']} {d['w']}x{d['h']}")
        if in_x and in_y:
            return True
        self.backend = None
        return False

    # -- pygame (separate process) -----------------------------------------
    def _open_pygame(self) -> None:
        d = self.display
        w, h = self.size
        self._slot_bytes = w * h * 3
        self._shm = shared_memory.SharedMemory(create=True, size=64 + 2 * self._slot_bytes)
        self._header = np.ndarray((2,), np.uint64, buffer=self._shm.buf[:16])
        self._header[:] = 0
        self._slots = [np.ndarray((h, w, 3), np.uint8, buffer=self._shm.buf[64 + i * self._slot_bytes:64 + (i + 1) * self._slot_bytes])
                       for i in range(2)]
        for slot in self._slots:
            slot[:] = 0
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "tryon.projector_child", self._shm.name, str(w), str(h),
             str(d["x"]), str(d["y"]), str(d.get("index", 1))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(ROOT))
        self._keys: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_child, daemon=True).start()
        self.backend = "pygame"

    def _read_child(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if line and line != "ready":
                self._keys.put(line)

    def _send(self, cmd: str) -> None:
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    # -- common ---------------------------------------------------------------
    def show(self, img: np.ndarray) -> None:
        if self.backend == "cv2":
            cv2.imshow(PROJECTOR_WINDOW, img)
        elif self.backend == "pygame":
            slot = 1 - int(self._header[1])
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=self._slots[slot])
            self._header[1] = slot
            self._header[0] += 1

    def poll_keys(self) -> list[str]:
        out = []
        if self.backend == "pygame":
            while not self._keys.empty():
                out.append(self._keys.get_nowait())
            if self._proc.poll() is not None:
                self.error = "projector window process exited"
                self.backend = None
        return out

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.backend == "cv2":
            cv2.setWindowProperty(PROJECTOR_WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif self.backend == "pygame":
            self._send("fullscreen")

    def close(self) -> None:
        if self.backend == "cv2":
            cv2.destroyWindow(PROJECTOR_WINDOW)
        elif self.backend == "pygame":
            self._send("quit")
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        if hasattr(self, "_shm"):
            self._header = None
            self._slots = []
            self._shm.close()
            self._shm.unlink()
            del self._shm
        self.backend = None
