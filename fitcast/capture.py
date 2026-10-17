"""Threaded camera / video-file source that holds only the latest frame."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("fitcast.capture")

_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}


def parse_source(s: str | int) -> int | str:
    """'0' -> 0 (camera index), anything else -> path string."""
    if isinstance(s, int):
        return s
    return int(s) if str(s).strip().isdigit() else str(s)


def list_cameras(max_index: int = 5) -> list[str]:
    """Print camera devices (macOS system_profiler names + OpenCV index probe)."""
    lines: list[str] = []
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10).stdout
        names = [ln.strip().rstrip(":") for ln in out.splitlines() if ln.startswith("    ") and not ln.startswith("      ")]
        lines.append("macOS camera devices: " + (", ".join(names) if names else "(none reported)"))
    except Exception as e:  # pragma: no cover - platform specific
        lines.append(f"system_profiler unavailable: {e}")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, f = cap.read()
            shape = f.shape if ok else None
            lines.append(f"index {i}: open, frame {shape}, backend {cap.getBackendName()}")
        else:
            lines.append(f"index {i}: not available")
        cap.release()
    lines.append("If every index is 'not available' on macOS, grant Camera access to your terminal app in "
                 "System Settings > Privacy & Security > Camera, then retry.")
    for ln in lines:
        print(ln)
    return lines


class FrameSource:
    """Background capture thread. `read()` returns the most recent frame only; nothing is queued.

    Frames are resized to (cfg.width, cfg.height) = camera space. Video files loop and,
    when `video_realtime` is set, play at their native fps.
    """

    def __init__(self, cfg: dict, source: int | str | None = None, sync: bool = False) -> None:
        self.cfg = cfg
        self.sync = sync        # True: no thread, read() decodes the next frame synchronously (offline review runs)
        self.source = parse_source(cfg["source"] if source is None else source)
        self.size = (int(cfg["width"]), int(cfg["height"]))
        self.interp = _INTERP[cfg.get("resize_interpolation", "area")]
        self.is_file = isinstance(self.source, str)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self.cap: cv2.VideoCapture | None = None
        self.native_fps = float(cfg.get("fps", 30))
        self.frames_read = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        """Open the device/file and log what we actually got. Raises RuntimeError with a fix-it hint."""
        if self.is_file and not Path(self.source).exists():
            raise RuntimeError(f"Video file not found: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            hint = ("Grant Camera access to your terminal app (System Settings > Privacy & Security > Camera) "
                    "and check `python main.py --list-cameras`." if not self.is_file else "Check the path / codec.")
            raise RuntimeError(f"Could not open source {self.source!r}. {hint}")
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.get("fps", 30)))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.native_fps = fps
        log.info("source %r opened: native %dx%d @ %.1f fps (backend %s) -> camera space %dx%d",
                 self.source, w, h, fps, cap.getBackendName(), *self.size)
        self.cap = cap

    def start(self) -> "FrameSource":
        if self.cap is None:
            self.open()
        if not self.sync:
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    # -- data --------------------------------------------------------------
    def read(self) -> tuple[np.ndarray, float, int] | None:
        """Latest (frame BGR camera-space, capture timestamp perf_counter, sequence number) or None."""
        if self.sync:
            return self._read_sync()
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._ts, self._seq

    def _read_sync(self) -> tuple[np.ndarray, float, int] | None:
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file and self.cfg.get("video_loop", True):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=self.interp)
        self._seq += 1
        self.frames_read += 1
        self._frame, self._ts = frame, time.perf_counter()
        return frame, self._ts, self._seq

    def _run(self) -> None:
        assert self.cap is not None
        period = 1.0 / self.native_fps if (self.is_file and self.cfg.get("video_realtime", True)) else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.cfg.get("video_loop", True):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                log.error("source %r stopped delivering frames", self.source)
                break
            ts = time.perf_counter()
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                frame = cv2.resize(frame, self.size, interpolation=self.interp)
            with self._lock:
                self._frame, self._ts, self._seq = frame, ts, self._seq + 1
            self.frames_read += 1
            if period:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
