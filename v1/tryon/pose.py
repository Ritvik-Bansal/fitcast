"""MediaPipe pose + person segmentation + a front/back face cue, on a worker thread.

The worker always processes the newest camera frame, so a slow model makes the
pose update rate drop but never stalls rendering. Keypoints come back in camera
pixel coordinates using MediaPipe's own left/right labels; the warp and the
orientation code decide how much to trust those labels.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

LANDMARKS = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}
WORLD_POINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


@dataclass
class PoseResult:
    t: float
    frame_idx: int
    size: tuple[int, int]
    present: bool
    kp: dict[str, np.ndarray] = field(default_factory=dict)
    vis: dict[str, float] = field(default_factory=dict)
    world: dict[str, np.ndarray] = field(default_factory=dict)
    mask: np.ndarray | None = None
    mask_t: float = 0.0
    face_score: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def world_z(self) -> dict[str, float]:
        return {k: float(v[2]) for k, v in self.world.items()}


def _mp_image(rgb: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class PoseTracker:
    def __init__(self, cfg, camera=None):
        self.cfg = cfg
        self.camera = camera
        base = mp_tasks.BaseOptions
        self.mask_source = cfg.mask_source
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(cfg.path(cfg.pose_model))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg.min_pose_confidence,
            min_pose_presence_confidence=cfg.min_pose_confidence,
            min_tracking_confidence=cfg.min_pose_confidence,
            output_segmentation_masks=self.mask_source == "pose",
        ))
        self.segmenter = None
        if self.mask_source == "segmenter":
            self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=base(model_asset_path=str(cfg.path(cfg.segmenter_model))),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            ))
        self.face = None
        if cfg.face_detect_every_n > 0 and cfg.path(cfg.face_model).exists():
            self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=base(model_asset_path=str(cfg.path(cfg.face_model))),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.3,
            ))
        self.seg_every = max(1, cfg.seg_every_n_frames) if cfg.seg_every_n_frames > 0 else 1
        self._auto_seg = cfg.seg_every_n_frames <= 0
        self._last_ts_ms = -1
        self._count = 0
        self._last_mask: np.ndarray | None = None
        self._last_mask_t = 0.0
        self._ema_ms = 0.0
        self._lock = threading.Lock()
        self._result: PoseResult | None = None
        self._seq = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self.infer_fps = 0.0

    # -- threading -------------------------------------------------------
    def start(self) -> "PoseTracker":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="pose", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        last_idx = -1
        n, t0 = 0, time.monotonic()
        while self._running:
            frame, ts, idx = self.camera.wait_for_new(last_idx, 0.2)
            if frame is None or idx == last_idx:
                continue
            last_idx = idx
            res = self.process(frame, ts, idx)
            with self._lock:
                self._result = res
                self._seq += 1
            n += 1
            now = time.monotonic()
            if now - t0 >= 1.0:
                self.infer_fps = n / (now - t0)
                n, t0 = 0, now

    def latest(self) -> tuple[PoseResult | None, int]:
        with self._lock:
            return self._result, self._seq

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.landmarker.close()
        if self.segmenter:
            self.segmenter.close()
        if self.face:
            self.face.close()

    # -- one frame -------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, ts: float, idx: int) -> PoseResult:
        t_start = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        ts_ms = int(ts * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timings: dict[str, float] = {}

        t = time.perf_counter()
        out = self.landmarker.detect_for_video(_mp_image(rgb), ts_ms)
        timings["pose"] = (time.perf_counter() - t) * 1000
        res = PoseResult(t=ts, frame_idx=idx, size=(w, h), present=bool(out.pose_landmarks))

        if res.present:
            lms = out.pose_landmarks[0]
            for name, i in LANDMARKS.items():
                lm = lms[i]
                res.kp[name] = np.array([lm.x * w, lm.y * h], np.float32)
                vis = lm.visibility if lm.visibility is not None else 1.0
                pres = lm.presence if lm.presence is not None else 1.0
                res.vis[name] = float(min(vis, pres))
            if out.pose_world_landmarks:
                wl = out.pose_world_landmarks[0]
                for name in WORLD_POINTS:
                    p = wl[LANDMARKS[name]]
                    res.world[name] = np.array([p.x, p.y, p.z], np.float32)

        # person mask
        t = time.perf_counter()
        if self.mask_source == "pose":
            if out.segmentation_masks:
                m = out.segmentation_masks[0].numpy_view()
                m = m[..., 0] if m.ndim == 3 else m
                self._last_mask = ((m > self.cfg.seg_threshold) * 255).astype(np.uint8)
                if self._last_mask.shape[:2] != (h, w):
                    self._last_mask = cv2.resize(self._last_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                self._last_mask_t = ts
        elif self.segmenter is not None and self._count % self.seg_every == 0:
            sw = self.cfg.seg_input_width
            sh = max(1, int(round(sw * h / w)))
            small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
            seg = self.segmenter.segment_for_video(_mp_image(small), ts_ms)
            conf = seg.confidence_masks[0].numpy_view()
            conf = conf[..., 0] if conf.ndim == 3 else conf
            conf8 = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
            big = cv2.resize(conf8, (w, h), interpolation=cv2.INTER_LINEAR)
            self._last_mask = np.where(big > int(self.cfg.seg_threshold * 255), 255, 0).astype(np.uint8)
            self._last_mask_t = ts
        timings["seg"] = (time.perf_counter() - t) * 1000
        res.mask = self._last_mask
        res.mask_t = self._last_mask_t

        # front/back cue: is there a face in the head crop?
        t = time.perf_counter()
        if res.present and self.face is not None and self._count % self.cfg.face_detect_every_n == 0:
            res.face_score = self._face_score(rgb, res)
        timings["face"] = (time.perf_counter() - t) * 1000

        self._count += 1
        total = (time.perf_counter() - t_start) * 1000
        timings["infer"] = total
        res.timings = timings
        if self._count > 10:  # skip model warm-up frames
            self._ema_ms = total if self._ema_ms == 0 else 0.9 * self._ema_ms + 0.1 * total
        if self._auto_seg and self.segmenter is not None and self._count > 30:
            budget = 1000.0 / max(1.0, self.cfg.camera_fps)
            if self.seg_every == 1 and self._ema_ms > 0.9 * budget:
                self.seg_every = 2
                print(f"[pose] inference {self._ema_ms:.1f} ms > budget, segmenting every other frame")
            elif self.seg_every == 2 and self._ema_ms < 0.6 * budget:
                self.seg_every = 1
        return res

    def _face_score(self, rgb: np.ndarray, res: PoseResult) -> float | None:
        h, w = rgb.shape[:2]
        pts = [res.kp[k] for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
        center = np.mean(pts, axis=0)
        ear_d = float(np.linalg.norm(res.kp["left_ear"] - res.kp["right_ear"]))
        sh_d = float(np.linalg.norm(res.kp["left_shoulder"] - res.kp["right_shoulder"]))
        size = max(2.8 * ear_d, 0.6 * sh_d, 48.0)
        x0, y0 = int(center[0] - size / 2), int(center[1] - size / 2)
        x1, y1 = int(x0 + size), int(y0 + size)
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1c - x0c < 16 or y1c - y0c < 16:
            return None
        crop = cv2.resize(rgb[y0c:y1c, x0c:x1c], (192, 192), interpolation=cv2.INTER_LINEAR)
        det = self.face.detect(_mp_image(crop))
        scores = [d.categories[0].score for d in det.detections if d.categories]
        return float(max(scores)) if scores else 0.0
