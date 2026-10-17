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

    Width is normalised by torso length when hips are visible so stepping toward the camera does
    not change the ratio. Reference = running max captured during warm-up, then updated slowly.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.side_ratio = float(cfg["side_ratio"])
        self.hold = int(cfg["hold_frames"])
        self.rate = float(cfg["ref_update_rate"])
        self.decay = float(cfg.get("ref_decay_rate", 0.0005))
        self.warmup = int(cfg.get("ref_warmup_frames", 30))
        self.face_thr = float(cfg["face_vis_back_threshold"])
        self.reset()

    def reset(self) -> None:
        self.state = "front"
        self._cand = "front"
        self._count = 0
        self.ref = 0.0          # normalised reference width
        self.ref_px = 0.0       # reference shoulder width in px (for sizing / rendering)
        self._n = 0
        self.ratio = 1.0

    def update(self, joints: np.ndarray, vis: np.ndarray, min_vis: float) -> str:
        """Feed one frame of (33,3) joints + visibility; returns the (hysteresis-filtered) state."""
        ls, rs = joints[L_SHOULDER, :2], joints[R_SHOULDER, :2]
        sw = float(np.hypot(*(ls - rs)))
        hips_ok = vis[L_HIP] >= min_vis and vis[R_HIP] >= min_vis
        norm = geo.torso_length(joints) if hips_ok else 1.0
        width = sw / norm if norm > 1e-3 else sw
        self._n += 1
        # reference: fast running max during warm-up, then slow updates
        if self.ref <= 0 or self._n <= self.warmup:
            if width > self.ref:
                self.ref, self.ref_px = width, sw
        elif width > self.ref:
            self.ref += self.rate * (width - self.ref)
            self.ref_px += self.rate * (sw - self.ref_px)
        elif self.state != "side":
            self.ref -= self.decay * (self.ref - width)
            self.ref_px -= self.decay * (self.ref_px - sw)
        self.ratio = width / self.ref if self.ref > 1e-6 else 1.0
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
        quad = geo.torso_quad(joints, sw, cfg["torso_margin_frac"], 0.05, cfg["hem_drop_frac"],
                              min_width=cfg.get("min_torso_width_frac", 0.7) * sw)
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
    win = 0.75 * sw
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
            running_mode=vision.RunningMode.VIDEO, output_category_mask=True, output_confidence_masks=False))
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
        """Segment a prepared mp.Image (used by the worker thread). Returns a (H,W) uint8 view."""
        r = self.seg.segment_for_video(img, ms)
        cat = r.category_mask.numpy_view()
        return cat[:, :, 0] if cat.ndim == 3 else cat

    def image_copy(self, frame_bgr: np.ndarray):
        """A standalone mp.Image (own RGB buffer) safe to hand to another thread."""
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

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
        self.last_ms = 0.0

    def submit(self, img, ms: int) -> bool:
        """Hand a frame to the worker; returns False (frame skipped) when it is still busy."""
        with self._cv:
            if self.busy:
                return False
            self._req, self.busy = (img, ms), True
            self._cv.notify()
        return True

    def latest(self) -> np.ndarray | None:
        """Newest finished category mask (a private copy) or None if nothing new since the last call."""
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
                img, ms = self._req
                self._req = None
            t0 = time.perf_counter()
            cat = self.backend.segment_image(img, ms).copy()
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
        self.torso_fill = bool(mcfg.get("torso_fill", True))
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
        self._n = 0
        self._prev_area = -1.0
        self._fallback_left = 0
        self.fallback_active = False
        self.last_person_mask: np.ndarray | None = None
        self.timings: dict[str, float] = {}

    # -- controls --------------------------------------------------------------------------
    def cycle_mask_mode(self) -> str:
        self.mask_mode = MASK_MODES[(MASK_MODES.index(self.mask_mode) + 1) % len(MASK_MODES)]
        log.info("mask mode -> %s", self.mask_mode)
        return self.mask_mode

    def reset_reference(self) -> None:
        self.facing.reset()

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
        joints, vis, person = self.backend.detect(frame_bgr, ms)
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
        vis = np.where(inside, vis, 0.0).astype(np.float32)

        # segmentation (every N frames; reuse previous category mask otherwise)
        seg_ms = 0.0
        if self.worker is not None:
            if (self._n - 1) % self.seg_every == 0:
                self.worker.submit(self.backend.image_copy(frame_bgr), ms)
            cat = self.worker.latest()
            if cat is not None:
                np.copyto(self._cat, cat)
                seg_ms = self.worker.last_ms
        elif (self._n - 1) % self.seg_every == 0:
            np.copyto(self._cat, self.backend.segment(frame_bgr, ms))
        t2 = time.perf_counter()

        sw = geo.shoulder_width(joints)
        facing = self.facing.update(joints, vis, self.min_vis)
        ref_px = self.facing.ref_px or sw
        sw_for_size = max(sw, 0.7 * ref_px)

        # person gate: pose segmentation (dilated) or the skeleton when the backend has none
        skel_cfg = self.cfg["mask"]["skeleton"]
        draw_skeleton_mask(self._skel, joints, vis, sw_for_size, skel_cfg, self.min_vis)
        if person is not None:
            cv2.threshold(person, self.gate_thr, 255, cv2.THRESH_BINARY, dst=self._pf)
            self._pf.astype(np.uint8, copy=False)
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
        if self._prev_area > 0 and area > 0 and abs(area - self._prev_area) / self._prev_area > self.area_jump:
            self._fallback_left = self.fallback_hold
        self._prev_area = area
        self.fallback_active = self._fallback_left > 0
        if self._fallback_left > 0:
            self._fallback_left -= 1

        mode = self.mask_mode
        if mode == "intersect" and self.fallback_active:
            mode = "skeleton"
        if mode == "clothes":
            mask = self._clothes.copy()
        elif mode == "skeleton":
            mask = self._skel.copy()
        else:
            box_dilate(self._skel, self.k_skel, self._tmp, self._tmp2)
            mask = cv2.bitwise_and(self._clothes, self._tmp)
            if self.torso_fill and person is not None and vis[L_HIP] >= self.min_vis and vis[R_HIP] >= self.min_vis:
                # torso polygon ∩ (eroded) person mask: the torso is always fabric, so fill segmenter holes there
                self._tmp.fill(0)
                core = geo.torso_quad(joints, sw_for_size, 0.0, 0.0, skel_cfg["hem_drop_frac"] * 0.5,
                                      min_width=skel_cfg.get("min_torso_width_frac", 0.7) * sw_for_size)
                cv2.fillConvexPoly(self._tmp, np.round(core).astype(np.int32), 255)
                np.copyto(self._tmp2, self._pf, casting="unsafe")              # undilated person mask
                box_erode(self._tmp2, self.k_erode, self._hairface, self._tmp2)  # (hairface buffer reused as scratch)
                cv2.bitwise_and(self._tmp, self._hairface, dst=self._tmp)
                cv2.bitwise_or(mask, self._tmp, dst=mask)

        # head box: landmarks + person pixels + hair/face-skin pixels above the shoulders
        hairface = None
        if self.head_cfg.get("use_person_mask", True):
            np.equal(self._cat, 1, out=self._hairface.view(bool))
            np.equal(self._cat, 3, out=self._tmp.view(bool))
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
                           facing=facing, timestamp=ts, shoulder_width=sw, ref_shoulder_width=ref_px)
