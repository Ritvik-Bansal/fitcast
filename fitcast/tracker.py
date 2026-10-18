"""Pose, clothes mask, head box, facing. Outputs TrackResult in camera pixel space.

Backends: MediaPipe PoseLandmarker (default) or rtmlib RTMPose (COCO-17 mapped to the 33-point layout).
The clothes mask always comes from the MediaPipe selfie_multiclass segmenter (class 4 = clothes).
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from . import geometry as geo
from .types import (FACE_IDX, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_ELBOW,
                    R_HIP, R_KNEE, R_SHOULDER, R_WRIST, L_EYE, R_EYE, TrackResult)

log = logging.getLogger("fitcast.tracker")

MASK_MODES = ("clothes", "skeleton", "intersect")

# COCO-17 (rtmlib) index -> MediaPipe-33 index
COCO_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
              11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}

LIMBS = ((L_SHOULDER, L_ELBOW, "upper"), (L_ELBOW, L_WRIST, "fore"),
         (R_SHOULDER, R_ELBOW, "upper"), (R_ELBOW, R_WRIST, "fore"),
         (L_HIP, L_KNEE, "thigh"), (L_KNEE, L_ANKLE, "shin"),
         (R_HIP, R_KNEE, "thigh"), (R_KNEE, R_ANKLE, "shin"))


def _ksize(px: int) -> tuple[int, int]:
    px = max(1, int(px))
    return (2 * px + 1, 2 * px + 1)


def box_dilate(src: np.ndarray, k: tuple[int, int], dst: np.ndarray, tmp: np.ndarray) -> np.ndarray:
    """Square dilation of a 0/255 mask via an integral-image box filter (O(1) per pixel, ~10x faster than
    cv2.dilate with a large kernel). Any 255 inside the window -> 255."""
    cv2.blur(src, k, dst=tmp)
    cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY, dst=dst)
    return dst


def box_erode(src: np.ndarray, k: tuple[int, int], dst: np.ndarray, tmp: np.ndarray) -> np.ndarray:
    """Square erosion of a 0/255 mask via a box filter: window must be entirely 255."""
    cv2.blur(src, k, dst=tmp)
    cv2.threshold(tmp, 254, 255, cv2.THRESH_BINARY, dst=dst)
    return dst


# ---------------------------------------------------------------------------------------------
# Facing
# ---------------------------------------------------------------------------------------------
class FacingClassifier:
    """front / back / side from shoulder order and normalised shoulder width, with hysteresis.

    The frontal reference is the MAX width over a sliding window of `ref_window_frames` (self-healing:
    a bystander walking past the lens or a bad first second expires instead of poisoning the session).
    Two widths are tracked: normalised by torso length when both hips are inside the frame (distance
    invariant), raw px otherwise; each has its own window.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.side_ratio = float(cfg["side_ratio"])
        self.hold = int(cfg["hold_frames"])
        self.window = int(cfg["ref_window_frames"])
        self.face_thr = float(cfg["face_vis_back_threshold"])
        self.reset()

    def reset(self) -> None:
        self.state = "front"
        self._cand = "front"
        self._count = 0
        self._ring_norm = np.zeros(self.window, np.float32)
        self._ring_px = np.zeros(self.window, np.float32)
        self._i_norm = 0
        self._i_px = 0
        self.ref_norm = 0.0
        self.ref_px = 0.0
        self.ratio = 1.0

    def update(self, joints: np.ndarray, vis: np.ndarray, min_vis: float, hips_in_frame: bool = True) -> str:
        """Feed one frame of (33,3) joints + visibility; returns the (hysteresis-filtered) state."""
        ls, rs = joints[L_SHOULDER, :2], joints[R_SHOULDER, :2]
        sw = float(np.hypot(*(ls - rs)))
        hips_ok = hips_in_frame and vis[L_HIP] >= min_vis and vis[R_HIP] >= min_vis
        self._ring_px[self._i_px % self.window] = sw
        self._i_px += 1
        self.ref_px = float(self._ring_px[: min(self._i_px, self.window)].max())
        if hips_ok:
            norm = geo.torso_length(joints)
            width = sw / norm if norm > 1e-3 else sw
            self._ring_norm[self._i_norm % self.window] = width
            self._i_norm += 1
            self.ref_norm = float(self._ring_norm[: min(self._i_norm, self.window)].max())
            ref = self.ref_norm
        else:
            width, ref = sw, self.ref_px
        self.ratio = width / ref if ref > 1e-6 else 1.0
        if self.ratio < self.side_ratio:
            cand = "side"
        else:
            # person's LEFT shoulder (11) appears image-RIGHT when facing the camera
            cand = "front" if ls[0] > rs[0] else "back"
            face_vis = float(min(vis[NOSE], vis[L_EYE], vis[R_EYE]))
            if cand == "front" and face_vis < self.face_thr:
                cand = "back"
        if cand == self._cand:
            self._count += 1
        else:
            self._cand, self._count = cand, 1
        if self._cand != self.state and self._count >= self.hold:
            self.state = self._cand
        return self.state


# ---------------------------------------------------------------------------------------------
# Geometry masks
# ---------------------------------------------------------------------------------------------
def draw_skeleton_mask(out: np.ndarray, joints: np.ndarray, vis: np.ndarray, sw: float, cfg: dict,
                       min_vis: float) -> np.ndarray:
    """Fill `out` (H,W uint8) with thick torso polygon + limb capsules sized from shoulder width."""
    out.fill(0)
    if vis[L_SHOULDER] < min_vis or vis[R_SHOULDER] < min_vis:
        return out
    if vis[L_HIP] >= min_vis and vis[R_HIP] >= min_vis:
        quad = geo.torso_quad(joints, sw, cfg["torso_margin_frac"], cfg["neck_raise_frac"], cfg["hem_drop_frac"],
                              min_width=cfg["min_torso_width_frac"] * sw)
        cv2.fillConvexPoly(out, np.round(quad).astype(np.int32), 255)
    widths = {"upper": cfg["upper_arm_width_frac"], "fore": cfg["forearm_width_frac"],
              "thigh": cfg["thigh_width_frac"], "shin": cfg["shin_width_frac"]}
    for a, b, kind in LIMBS:
        if vis[a] < min_vis or vis[b] < min_vis:
            continue
        w = 0.5 * widths[kind] * sw
        quad = geo.limb_quad(joints[a, :2], joints[b, :2], w, w * 0.9)
        cv2.fillConvexPoly(out, np.round(quad).astype(np.int32), 255)
        cv2.circle(out, (int(joints[a, 0]), int(joints[a, 1])), int(w), 255, -1)
        cv2.circle(out, (int(joints[b, 0]), int(joints[b, 1])), int(w * 0.9), 255, -1)
    return out


def compute_head_box(joints: np.ndarray, vis: np.ndarray, sw: float, cfg: dict, size: tuple[int, int],
                     person_mask: np.ndarray | None = None, hair_face_mask: np.ndarray | None = None
                     ) -> tuple[int, int, int, int]:
    """Padded head box (x0,y0,x1,y1) covering face landmarks + margins, extended by any person/hair pixels
    above the shoulder line within a window around the face. Bottom sits just above the shoulders."""
    W, H = size
    face = joints[list(FACE_IDX), :2]
    fx0, fy0 = face.min(axis=0)
    fx1, fy1 = face.max(axis=0)
    m_side = max(cfg["side_margin_frac"], cfg["min_margin_frac"]) * sw
    m_top = max(cfg["top_margin_frac"], cfg["min_margin_frac"]) * sw
    m_min = cfg["min_margin_frac"] * sw
    sh_y = float(min(joints[L_SHOULDER, 1], joints[R_SHOULDER, 1]))
    x0, x1 = fx0 - m_side, fx1 + m_side
    y0 = fy0 - m_top
    y1 = max(sh_y - cfg["bottom_gap_frac"] * sw, fy1 + m_min)
    # union with segmentation evidence above the shoulder line, near the face
    cx = 0.5 * (fx0 + fx1)
    win = cfg["search_window_frac"] * sw
    row1 = int(np.clip(sh_y, 0, H))
    cx0, cx1 = int(np.clip(cx - win, 0, W)), int(np.clip(cx + win, 0, W))
    for m in (person_mask, hair_face_mask):
        if m is None or row1 <= 0 or cx1 <= cx0:
            continue
        sub = m[:row1, cx0:cx1]
        ys, xs = np.nonzero(sub)
        if len(ys):
            x0 = min(x0, cx0 + xs.min() - m_min)
            x1 = max(x1, cx0 + xs.max() + m_min)
            y0 = min(y0, ys.min() - m_min)
    x0, y0 = int(np.clip(np.floor(x0), 0, W)), int(np.clip(np.floor(y0), 0, H))
    x1, y1 = int(np.clip(np.ceil(x1), 0, W)), int(np.clip(np.ceil(y1), 0, H))
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------
class MediaPipeBackend:
    """PoseLandmarker (VIDEO mode, with person segmentation) + selfie multiclass ImageSegmenter."""

    def __init__(self, cfg: dict) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        for key in ("pose_model", "seg_model"):
            from pathlib import Path
            if not Path(cfg[key]).exists():
                raise RuntimeError(f"Model missing: {cfg[key]}. Run `.venv/bin/python scripts/download_models.py`.")
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg["pose_model"]),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=cfg["min_pose_detection_confidence"],
            min_pose_presence_confidence=cfg["min_pose_presence_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
            output_segmentation_masks=True))
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=cfg["seg_model"]),
            running_mode=vision.RunningMode.VIDEO, output_category_mask=False, output_confidence_masks=True))
        self.seg_thr = float(cfg.get("seg_confidence_threshold", 0.5))
        self.has_person_mask = True
        self._rgb: np.ndarray | None = None
        self._last_ms = -1

    def _image(self, frame_bgr: np.ndarray):
        if self._rgb is None or self._rgb.shape != frame_bgr.shape:
            self._rgb = np.empty_like(frame_bgr)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=self._rgb)

    def timestamp_ms(self, ts: float) -> int:
        """Strictly increasing integer ms for MediaPipe VIDEO mode."""
        ms = int(ts * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms

    def detect(self, frame_bgr: np.ndarray, ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """-> joints (33,3) px, visibility (33,), person mask float32 (H,W) or None. Empty arrays when no pose."""
        h, w = frame_bgr.shape[:2]
        self._img = self._image(frame_bgr)
        r = self.pose.detect_for_video(self._img, ms)
        if not r.pose_landmarks:
            return np.zeros((33, 3), np.float32), np.zeros((33,), np.float32), None
        lm = r.pose_landmarks[0]
        joints = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], np.float32)
        vis = np.array([p.visibility for p in lm], np.float32)
        pm = r.segmentation_masks[0].numpy_view() if r.segmentation_masks else None
        if pm is not None and pm.ndim == 3:
            pm = pm[:, :, 0]
        return joints, vis, pm

    def segment(self, frame_bgr: np.ndarray, ms: int) -> np.ndarray:
        """Category mask (H,W) uint8 with selfie_multiclass labels; reuses the image built in detect()."""
        return self.segment_image(self._img, ms)

    def segment_image(self, img, ms: int) -> np.ndarray:
        """Segment a prepared mp.Image (used by the worker thread). Returns a (H,W) uint8 label map built from
        the bilinear confidence masks (smooth edges; the category mask is nearest-neighbour and blocky).
        Labels: 1 hair, 2 body-skin, 3 face-skin, 4 clothes (clothes wins ties)."""
        r = self.seg.segment_for_video(img, ms)
        conf = r.confidence_masks
        def m(i):
            a = conf[i].numpy_view()
            return a[:, :, 0] if a.ndim == 3 else a
        cat = np.zeros(m(4).shape, np.uint8)
        for label in (2, 1, 3, 4):           # later labels overwrite earlier ones
            cat[m(label) > self.seg_thr] = label
        return cat

    def image_copy(self, frame_bgr: np.ndarray, buf: tuple | None = None) -> tuple:
        """(mp.Image, rgb buffer) with its own RGB memory, safe to hand to the worker thread.
        Pass the previous tuple for that slot back in to reuse its buffer (double-buffering)."""
        rgb = None if buf is None else buf[1]
        if rgb is None or rgb.shape != frame_bgr.shape:
            rgb = np.empty_like(frame_bgr)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=rgb)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), rgb

    def close(self) -> None:
        self.pose.close()
        self.seg.close()


class RtmlibBackend(MediaPipeBackend):
    """RTMPose (COCO-17) via rtmlib/onnxruntime for pose; MediaPipe segmenter for the clothes mask."""

    def __init__(self, cfg_mp: dict, cfg_rtm: dict) -> None:
        super().__init__(cfg_mp)      # we still need the segmenter (and its image conversion)
        self.pose.close()
        from pathlib import Path
        for key in ("det_model", "pose_model"):
            if not Path(cfg_rtm[key]).exists():
                raise RuntimeError(f"Model missing: {cfg_rtm[key]}. Run `scripts/download_models.py --rtm`.")
        from rtmlib import Body
        self.body = Body(det=cfg_rtm["det_model"], det_input_size=tuple(cfg_rtm["det_input_size"]),
                         pose=cfg_rtm["pose_model"], pose_input_size=tuple(cfg_rtm["pose_input_size"]),
                         backend="onnxruntime", device="cpu")
        self.min_score = float(cfg_rtm["min_score"])
        self.has_person_mask = False

    def detect(self, frame_bgr: np.ndarray, ms: int):
        self._img = self._image(frame_bgr)
        kps, scores = self.body(frame_bgr)
        joints = np.zeros((33, 3), np.float32)
        vis = np.zeros((33,), np.float32)
        if len(kps) == 0:
            return joints, vis, None
        i = int(np.argmax(scores.mean(axis=1)))
        kp, sc = kps[i], scores[i]
        if sc.mean() < self.min_score:
            return joints, vis, None
        for c, m in COCO_TO_MP.items():
            joints[m, :2] = kp[c]
            vis[m] = sc[c]
        # fill the remaining face points from the eyes/nose so FACE_IDX bounding boxes work
        for m in (1, 3, 4, 6, 9, 10):
            src = 2 if m in (1, 3) else 5 if m in (4, 6) else 0
            joints[m, :2] = joints[src, :2]
            vis[m] = vis[src]
        return joints, vis, None

    def close(self) -> None:
        self.seg.close()


class SegWorker(threading.Thread):
    """Runs the clothes segmenter off the main thread. One request slot, one result slot; never queues."""

    def __init__(self, backend: MediaPipeBackend) -> None:
        super().__init__(name="segmenter", daemon=True)
        self.backend = backend
        self._cv = threading.Condition()
        self._req: tuple | None = None
        self._result: np.ndarray | None = None
        self._quit = False
        self.busy = False
        self.failed = False
        self.last_ms = 0.0

    def submit(self, img, ms: int, rect: tuple[int, int, int, int] | None = None) -> bool:
        """Hand a frame (+ the camera-space rect it was cropped from) to the worker; False when still busy."""
        with self._cv:
            if self.busy:
                return False
            self._req, self.busy = (img, ms, rect), True
            self._cv.notify()
        return True

    def latest(self) -> tuple[np.ndarray, tuple | None] | None:
        """Newest finished (label map copy, crop rect) or None if nothing new since the last call."""
        with self._cv:
            r, self._result = self._result, None
        return r

    def stop(self) -> None:
        with self._cv:
            self._quit = True
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while self._req is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    return
                img, ms, rect = self._req
                self._req = None
            t0 = time.perf_counter()
            try:
                cat = (self.backend.segment_image(img, ms).copy(), rect)
            except Exception:  # noqa: BLE001 - must never die silently
                log.exception("segmenter worker crashed; tracker switches to synchronous segmentation")
                with self._cv:
                    self.failed, self.busy = True, False
                return
            self.last_ms = (time.perf_counter() - t0) * 1e3
            with self._cv:
                self._result, self.busy = cat, False


# ---------------------------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------------------------
class Tracker:
    """Turns a camera frame into a TrackResult. Owns the backends, mask post-processing and facing state."""

    def __init__(self, cfg: dict, size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.W, self.H = size
        self.min_vis = float(cfg["min_visibility"])
        self.margin = int(cfg["frame_margin_px"])
        self.seg_every = max(1, int(cfg["seg_every_n_frames"]))
        self.seg_async = bool(cfg.get("seg_async", True))
        backend = cfg.get("backend", "mediapipe")
        if backend == "rtmlib":
            self.backend = RtmlibBackend(cfg["mediapipe"], cfg["rtmlib"])
        elif backend == "mediapipe":
            self.backend = MediaPipeBackend(cfg["mediapipe"])
        else:
            raise RuntimeError(f"Unknown tracker.backend {backend!r}; use mediapipe or rtmlib")
        log.info("tracker backend: %s (segmenter %s)", backend, "async" if self.seg_async else "sync")
        self.worker = SegWorker(self.backend) if self.seg_async else None
        if self.worker is not None:
            self.worker.start()
        mcfg = cfg["mask"]
        self.mask_mode = mcfg["mode"]
        if self.mask_mode not in MASK_MODES:
            raise RuntimeError(f"mask.mode must be one of {MASK_MODES}")
        self.clothes_class = int(mcfg["clothes_class"])
        self.k_close = _ksize(mcfg["close_kernel_px"])
        self.k_erode = _ksize(mcfg["erode_px"])
        self.k_gate = _ksize(mcfg["person_gate_dilate_px"])
        self.k_skel = _ksize(mcfg["skeleton"]["dilate_px"])
        self.gate_thr = float(mcfg["person_gate_threshold"])
        self.area_jump = float(mcfg["area_jump_fallback"])
        self.jump_frames = int(mcfg.get("area_jump_frames", 2))
        self._jump_streak = 0
        self.seg_crop_margin = float(mcfg.get("seg_crop_margin_frac", 0.25))
        self._seg_rect: tuple[int, int, int, int] | None = None
        self.torso_fill = bool(mcfg["torso_fill"])
        self.fallback_hold = int(mcfg["fallback_hold_frames"])
        self.facing = FacingClassifier(cfg["facing"])
        self.head_cfg = cfg["head"]
        # reusable buffers
        self._cat = np.zeros((self.H, self.W), np.uint8)
        self._clothes = np.zeros((self.H, self.W), np.uint8)
        self._skel = np.zeros((self.H, self.W), np.uint8)
        self._gate = np.zeros((self.H, self.W), np.uint8)
        self._tmp = np.zeros((self.H, self.W), np.uint8)
        self._tmp2 = np.zeros((self.H, self.W), np.uint8)
        self._hairface = np.zeros((self.H, self.W), np.uint8)
        self._pf = np.zeros((self.H, self.W), np.float32)
        self._out = [np.zeros((self.H, self.W), np.uint8), np.zeros((self.H, self.W), np.uint8)]
        self._rgb_buf = [None, None]
        self._flip = 0
        self.hair_class = int(mcfg["hair_class"])
        self.face_class = int(mcfg["face_class"])
        self.skin_class = int(mcfg["body_skin_class"])
        self.side_width_floor = float(cfg["head"]["side_width_floor_frac"])
        self.torso_fill_hem_frac = float(mcfg["torso_fill_hem_frac"])
        self._n = 0
        self._prev_area = -1.0
        self._fallback_left = 0
        self.fallback_active = False
        self.last_person_mask: np.ndarray | None = None
        self.timings: dict[str, float] = {}
        self.roi: tuple[int, int, int, int] | None = None   # x0,y0,x1,y1 camera px: models only see this crop
        self._person_full = np.zeros((self.H, self.W), np.float32)
        self._cat_roi: np.ndarray | None = None

    # -- controls --------------------------------------------------------------------------
    def cycle_mask_mode(self) -> str:
        self.mask_mode = MASK_MODES[(MASK_MODES.index(self.mask_mode) + 1) % len(MASK_MODES)]
        log.info("mask mode -> %s", self.mask_mode)
        return self.mask_mode

    def reset_reference(self) -> None:
        self.facing.reset()

    def _person_crop(self, joints: np.ndarray, vis: np.ndarray) -> tuple[int, int, int, int]:
        """Camera-space rect around the detected person (+margin), clamped to the frame/ROI, even-aligned."""
        pts = joints[vis > 0.2, :2] if (vis > 0.2).any() else joints[:, :2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        mx, my = self.seg_crop_margin * max(x1 - x0, 1), self.seg_crop_margin * max(y1 - y0, 1)
        bx0, by0, bx1, by1 = (0, 0, self.W, self.H) if self.roi is None else self.roi
        x0, y0 = int(max(bx0, x0 - mx)), int(max(by0, y0 - my))
        x1, y1 = int(min(bx1, x1 + mx)), int(min(by1, y1 + my))
        x0, y0 = x0 - (x0 % 2), y0 - (y0 % 2)
        if x1 - x0 < 64 or y1 - y0 < 64:
            return (bx0, by0, bx1, by1)
        return (x0, y0, x1, y1)

    def set_roi(self, rect: tuple[int, int, int, int] | None, margin_frac: float = 0.0) -> None:
        """Restrict pose + segmentation to a camera-space rectangle (e.g. the projector's footprint): bystanders
        outside the beam are never tracked and the subject gets more model resolution. None = full frame."""
        if rect is None:
            self.roi = None
            return
        x0, y0, x1, y1 = rect
        mx, my = margin_frac * (x1 - x0), margin_frac * (y1 - y0)
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(self.W, x1 + mx)), int(min(self.H, y1 + my))
        x0, y0 = x0 - (x0 % 2), y0 - (y0 % 2)                 # even offsets keep colour conversions aligned
        if x1 - x0 < 64 or y1 - y0 < 64:
            log.warning("tracker ROI %s too small; ignoring", rect)
            self.roi = None
            return
        self.roi = (x0, y0, x1, y1)
        log.info("tracker ROI set to %s (%dx%d of %dx%d)", self.roi, x1 - x0, y1 - y0, self.W, self.H)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=1.0)
        self.backend.close()

    # -- main --------------------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, ts: float) -> TrackResult:
        """Run pose (+ segmentation every N frames) and build the TrackResult for this frame."""
        t0 = time.perf_counter()
        ms = self.backend.timestamp_ms(ts)
        roi = self.roi
        model_frame = frame_bgr if roi is None else np.ascontiguousarray(frame_bgr[roi[1]:roi[3], roi[0]:roi[2]])
        joints, vis, person = self.backend.detect(model_frame, ms)
        if roi is not None:
            joints[:, 0] += roi[0]
            joints[:, 1] += roi[1]
            if person is not None:
                self._person_full.fill(0.0)
                self._person_full[roi[1]:roi[3], roi[0]:roi[2]] = person
                person = self._person_full
        t1 = time.perf_counter()
        self._n += 1
        ok = bool(vis[L_SHOULDER] >= self.min_vis and vis[R_SHOULDER] >= self.min_vis and
                  np.isfinite(joints).all() and vis.max() > 0)
        if not ok:
            self._prev_area = -1.0
            self.timings = {"pose_ms": (t1 - t0) * 1e3, "seg_ms": 0.0, "mask_ms": 0.0}
            return TrackResult.empty(self.H, self.W, ts)
        # zero visibility for landmarks outside the frame (knees below the frame still report 0.6+)
        inside = ((joints[:, 0] >= self.margin) & (joints[:, 0] <= self.W - self.margin) &
                  (joints[:, 1] >= self.margin) & (joints[:, 1] <= self.H - self.margin))
        hips_in_frame = bool(inside[L_HIP] and inside[R_HIP])
        inside[[L_HIP, R_HIP]] = True      # hips off-frame still anchor the torso quad (clipped by the frame)
        vis = np.where(inside, vis, 0.0).astype(np.float32)

        # segmentation (every N frames; reuse previous category mask otherwise)
        seg_ms = 0.0
        if self.worker is not None and self.worker.failed:
            log.error("segmenter worker failed; continuing with synchronous segmentation (slower)")
            self.worker = None
        # segmenter input: a tight crop around the detected person (2x the mask resolution of a full frame)
        seg_rect = self._person_crop(joints, vis)
        def _store_cat(cat_small: np.ndarray, rect: tuple | None) -> None:
            self._cat.fill(0)
            if rect is None:
                np.copyto(self._cat, cat_small)
            else:
                self._cat[rect[1]:rect[3], rect[0]:rect[2]] = cat_small
        if self.worker is not None:
            if (self._n - 1) % self.seg_every == 0:
                self._flip ^= 1
                crop = np.ascontiguousarray(frame_bgr[seg_rect[1]:seg_rect[3], seg_rect[0]:seg_rect[2]])
                self._rgb_buf[self._flip] = self.backend.image_copy(crop, self._rgb_buf[self._flip])
                self.worker.submit(self._rgb_buf[self._flip][0], ms, seg_rect)
            res = self.worker.latest()
            if res is not None:
                _store_cat(*res)
                seg_ms = self.worker.last_ms
        elif (self._n - 1) % self.seg_every == 0:
            crop = np.ascontiguousarray(frame_bgr[seg_rect[1]:seg_rect[3], seg_rect[0]:seg_rect[2]])
            self.backend._img = self.backend._image(crop)
            _store_cat(self.backend.segment(crop, ms), seg_rect)
        t2 = time.perf_counter()

        sw = geo.shoulder_width(joints)
        facing = self.facing.update(joints, vis, self.min_vis, hips_in_frame)
        ref_px = self.facing.ref_px or sw
        sw_for_size = max(sw, self.side_width_floor * ref_px)

        # person gate: pose segmentation (dilated) or the skeleton when the backend has none
        skel_cfg = self.cfg["mask"]["skeleton"]
        draw_skeleton_mask(self._skel, joints, vis, sw_for_size, skel_cfg, self.min_vis)
        if person is not None:
            cv2.threshold(person, self.gate_thr, 255, cv2.THRESH_BINARY, dst=self._pf)
            np.copyto(self._tmp, self._pf, casting="unsafe")
            box_dilate(self._tmp, self.k_gate, self._gate, self._tmp2)
            self.last_person_mask = self._gate
        else:
            box_dilate(self._skel, self.k_skel, self._gate, self._tmp2)
            self.last_person_mask = None

        # clothes mask: class 4, gated, closed, eroded
        np.equal(self._cat, self.clothes_class, out=self._clothes.view(bool))
        self._clothes *= 255
        cv2.bitwise_and(self._clothes, self._gate, dst=self._clothes)
        box_dilate(self._clothes, self.k_close, self._tmp, self._tmp2)        # close = dilate ...
        box_erode(self._tmp, self.k_close, self._clothes, self._tmp2)         # ... then erode
        box_erode(self._clothes, self.k_erode, self._tmp, self._tmp2)
        np.copyto(self._clothes, self._tmp)
        area = float(cv2.countNonZero(self._clothes))

        # automatic fallback when the clothes area jumps frame to frame (projection confusing the segmenter).
        # A reference of 0 means "no mask yet" (person just appeared / first async result), not a jump.
        # The jump must persist `area_jump_frames` consecutive frames so one segmenter glitch does not flip the mask.
        if self._prev_area > 0 and area > 0 and abs(area - self._prev_area) / self._prev_area > self.area_jump:
            self._jump_streak += 1
            if self._jump_streak >= self.jump_frames:
                self._fallback_left = self.fallback_hold
        else:
            self._jump_streak = 0
        self._prev_area = area
        self.fallback_active = self._fallback_left > 0
        if self._fallback_left > 0:
            self._fallback_left -= 1

        mode = self.mask_mode
        if mode == "intersect" and self.fallback_active:
            mode = "skeleton"
        mask = self._out[self._n & 1]          # alternate two output buffers: no per-frame allocation
        if mode == "clothes":
            np.copyto(mask, self._clothes)
        elif mode == "skeleton":
            if person is not None:          # fallback stays inside the person's silhouette (never fatter than the body)
                np.copyto(self._tmp, self._pf, casting="unsafe")
                cv2.bitwise_and(self._skel, self._tmp, dst=mask)
            else:
                np.copyto(mask, self._skel)
        else:
            box_dilate(self._skel, self.k_skel, self._tmp, self._tmp2)
            cv2.bitwise_and(self._clothes, self._tmp, dst=mask)
            if self.torso_fill and person is not None and vis[L_HIP] >= self.min_vis and vis[R_HIP] >= self.min_vis:
                # torso polygon ∩ (eroded) person mask minus segmenter skin: fills segmenter holes on fabric only
                self._tmp.fill(0)
                core = geo.torso_quad(joints, sw_for_size, 0.0, 0.0, skel_cfg["hem_drop_frac"] * self.torso_fill_hem_frac,
                                      min_width=skel_cfg["min_torso_width_frac"] * sw_for_size)
                cv2.fillConvexPoly(self._tmp, np.round(core).astype(np.int32), 255)
                np.copyto(self._tmp2, self._pf, casting="unsafe")              # undilated person mask
                box_erode(self._tmp2, self.k_erode, self._hairface, self._tmp2)  # (hairface buffer reused as scratch)
                cv2.bitwise_and(self._tmp, self._hairface, dst=self._tmp)
                np.equal(self._cat, self.skin_class, out=self._tmp2.view(bool))
                self._tmp2 *= 255
                cv2.bitwise_not(self._tmp2, dst=self._tmp2)
                cv2.bitwise_and(self._tmp, self._tmp2, dst=self._tmp)
                cv2.bitwise_or(mask, self._tmp, dst=mask)

        # head box: landmarks + person pixels + hair/face-skin pixels above the shoulders
        hairface = None
        if self.head_cfg.get("use_person_mask", True):
            np.equal(self._cat, self.hair_class, out=self._hairface.view(bool))
            np.equal(self._cat, self.face_class, out=self._tmp.view(bool))
            cv2.bitwise_or(self._hairface, self._tmp, dst=self._hairface)
            self._hairface *= 255
            hairface = self._hairface
        head = compute_head_box(joints, vis, sw_for_size, self.head_cfg, (self.W, self.H),
                                self.last_person_mask if self.head_cfg.get("use_person_mask", True) else None,
                                hairface)
        t3 = time.perf_counter()
        self.timings = {"pose_ms": (t1 - t0) * 1e3, "seg_ms": seg_ms if self.worker is not None else (t2 - t1) * 1e3,
                        "mask_ms": (t3 - t2) * 1e3}
        return TrackResult(ok=True, joints=joints, visibility=vis, clothes_mask=mask, head_box=head,
                           facing=facing, timestamp=ts, shoulder_width=sw, ref_shoulder_width=ref_px,
                           person_mask=self.last_person_mask)
