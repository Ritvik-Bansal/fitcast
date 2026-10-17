"""Threaded camera reader that only ever hands out the newest frame.

A background thread keeps reading so the driver buffer never fills up; the main
loop and the inference worker both pull the latest frame and its capture time.
A video file can stand in for the camera (camera_source) and is played back at
its own frame rate, which is how recorded sessions are replayed.
"""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.size = (cfg.camera_width, cfg.camera_height)
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._idx = -1
        self._running = False
        self._thread: threading.Thread | None = None
        self.is_file = bool(cfg.camera_source)
        self.fps = float(cfg.camera_fps)
        self.read_fps = 0.0
        self.error = ""

    def start(self) -> "Camera":
        cfg = self.cfg
        if self.is_file:
            self.cap = cv2.VideoCapture(str(cfg.path(cfg.camera_source)))
            if not self.cap.isOpened():
                sys.exit(f"[camera] could not open video file {cfg.camera_source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or float(cfg.camera_fps)
        else:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(cfg.camera_index, backend)
            if not self.cap.isOpened():
                sys.exit(
                    f"[camera] could not open camera index {cfg.camera_index}. "
                    "Check camera_index in config.yaml, and on macOS give your terminal app "
                    "camera access in System Settings > Privacy & Security > Camera."
                )
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            sys.exit("[camera] opened but returned no frames")
        native = (frame.shape[1], frame.shape[0])
        if native != self.size:
            print(f"[camera] source is {native[0]}x{native[1]}, resizing to {self.size[0]}x{self.size[1]}")
        self._publish(frame)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        return self

    def _publish(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        with self._cond:
            self._frame = frame
            self._ts = time.monotonic()
            self._idx += 1
            self._cond.notify_all()

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.fps)
        next_t = time.monotonic()
        count, t0 = 0, time.monotonic()
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.is_file and self.cfg.video_loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_file:
                    self.error = "video ended"
                    self._running = False
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.005)
                continue
            if self.is_file:
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()
            self._publish(frame)
            count += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.read_fps = count / (now - t0)
                count, t0 = 0, now

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._cond:
            return self._frame, self._ts, self._idx

    def wait_for_new(self, last_idx: int, timeout: float = 0.1) -> tuple[np.ndarray | None, float, int]:
        with self._cond:
            if self._idx == last_idx and self._running:
                self._cond.wait(timeout)
            return self._frame, self._ts, self._idx

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()
