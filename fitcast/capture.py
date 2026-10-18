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
    """'0' -> 0 (camera index), 'auto' -> first camera that streams non-black frames, else a path string."""
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if s == "auto":
        return pick_camera()
    if s.lower().startswith("name:"):
        return camera_index_by_name(s[5:])
    return int(s) if s.isdigit() else s


def camera_index_by_name(substring: str) -> int:
    """OpenCV index of the AVFoundation video device whose name contains `substring` (case-insensitive).
    OpenCV's AVFoundation backend enumerates devices in the same order as AVCaptureDevice."""
    try:
        import AVFoundation as AV  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"camera by name needs pyobjc-framework-AVFoundation ({e}); use a numeric --source") from e
    names = [str(d.localizedName()) for d in AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)]
    for i, n in enumerate(names):
        if substring.lower() in n.lower():
            log.info("camera by name %r -> index %d (%s); all devices: %s", substring, i, n, names)
            return i
    raise RuntimeError(f"No camera named like {substring!r}. Devices: {names}")


def pick_camera(max_index: int = 4, probe_s: float = 1.5, min_brightness: float = 5.0) -> int:
    """First camera index that delivers non-black frames (an idle/locked Continuity Camera opens fine but
    streams black or nothing, and the built-in camera's index moves when the iPhone comes and goes)."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        t0 = time.perf_counter()
        last = None
        while time.perf_counter() - t0 < probe_s:
            ok, f = cap.read()
            if ok:
                last = f
        cap.release()
        if last is not None and float(last.mean()) > min_brightness:
            log.info("camera auto-select: index %d streams (mean brightness %.0f)", i, float(last.mean()))
            return i
        log.info("camera auto-select: index %d skipped (%s)", i, "no frames" if last is None else "black frames")
    raise RuntimeError("No camera delivers frames. Unlock/wake the iPhone or check Camera permission for the terminal app.")


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
            for _ in range(5):          # skip warm-up frames before judging brightness
                ok, f = cap.read()
            shape = f.shape if ok else None
            note = ""
            if ok and float(f.mean()) < 3.0:
                note = "  <- frames are BLACK (iPhone locked / covered / Continuity Camera idle?)"
            lines.append(f"index {i}: open, frame {shape}, backend {cap.getBackendName()}{note}")
        else:
            lines.append(f"index {i}: not available")
        cap.release()
    lines.append("Note: when an iPhone is nearby it usually takes index 0 and the built-in camera moves to index 1.")
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
        self.source_spec = cfg["source"] if source is None else source   # keep 'auto' so reopen() can re-pick
        self.source = parse_source(self.source_spec)
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

    def reopen(self) -> bool:
        """Camera stopped delivering: release it, re-resolve the source ('auto' re-probes) and restart the thread."""
        log.warning("reopening camera source %r", self.source_spec)
        self.stop()
        try:
            self.source = parse_source(self.source_spec)
            self.cap = None
            self.open()
        except RuntimeError as e:
            log.error("reopen failed: %s", e)
            return False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        if not self.sync:
            self._thread.start()
        return True

    @property
    def alive(self) -> bool:
        return self.sync or self._thread.is_alive()

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
