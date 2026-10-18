"""Shared dataclasses. Only the lead edits this file (see CLAUDE.md)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MediaPipe Pose 33-point indices used across modules.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
FACE_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


@dataclass
class TrackResult:
    ok: bool
    joints: np.ndarray                      # (33, 3) float32: x, y camera px, z relative depth
    visibility: np.ndarray                  # (33,) float32 in [0, 1]
    clothes_mask: np.ndarray                # (H, W) uint8 0/255
    head_box: tuple[int, int, int, int]     # x0, y0, x1, y1 camera px, padded
    facing: str                             # "front" | "back" | "side"
    timestamp: float                        # perf_counter at capture
    shoulder_width: float = 0.0
    ref_shoulder_width: float = 0.0
    person_mask: np.ndarray | None = None   # (H, W) uint8 0/255 dilated whole-person mask when the backend has one
    facing_ratio: float = 1.0               # normalised shoulder width / frontal reference (1 = squarely frontal)
    # 3D sensor path (iPhone LiDAR); None on the webcam path
    depth: np.ndarray | None = None         # (H, W) float32 metres, camera space (0 = invalid)
    joints3d: np.ndarray | None = None      # (33, 3) float32 metres, OpenCV camera coords (+x right, +y down, +z forward); NaN = unknown
    K_cam: np.ndarray | None = None         # (3, 3) float64 intrinsics of the camera space image

    @staticmethod
    def empty(h: int, w: int, timestamp: float = 0.0) -> "TrackResult":
        """A 'no person' result with correctly shaped zero arrays."""
        return TrackResult(
            ok=False,
            joints=np.zeros((33, 3), np.float32),
            visibility=np.zeros((33,), np.float32),
            clothes_mask=np.zeros((h, w), np.uint8),
            head_box=(0, 0, 0, 0),
            facing="front",
            timestamp=timestamp,
        )


@dataclass
class Outfit:
    id: str
    name: str
    sleeve: str                                   # "short" | "long" | "none"
    top_front: np.ndarray | None                  # RGBA uint8
    top_back: np.ndarray | None
    sleeve_tex: np.ndarray | None
    bottom_front: np.ndarray | None
    bottom_back: np.ndarray | None
    thumb: np.ndarray                             # BGR uint8
    size_table: str = "mens"                      # "mens" | "womens"
    top_body_region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass
class GalleryState:
    outfits: list[Outfit] = field(default_factory=list)
    index: int = 0
    prev_index: int = 0
    changed_at: float = 0.0
    visible: bool = True
    size_text: str = ""
    measure_text: str = ""
    bottom_index: int = -1              # -1 = bottoms follow `index` (whole outfit); else mix-and-match
    strip: str = "outfits"              # wall strip shows "outfits" | "tops" | "bottoms"
    bottom_cycle: list = field(default_factory=list)   # outfit indices with distinct bottoms (bottom swipes step through these)
    _mix: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def current(self) -> Outfit:
        """The outfit to render: `index` whole, or its top combined with `bottom_index`'s bottoms."""
        top = self.outfits[self.index]
        if self.bottom_index < 0 or self.bottom_index == self.index:
            return top
        key = (self.index, self.bottom_index)
        o = self._mix.get(key)
        if o is None:
            bot = self.outfits[self.bottom_index]
            o = Outfit(id=f"{top.id}+{bot.id}", name=f"{top.name.split(' + ')[0]} + {bot.name.split(' + ')[-1]}",
                       sleeve=top.sleeve, top_front=top.top_front, top_back=top.top_back, sleeve_tex=top.sleeve_tex,
                       bottom_front=bot.bottom_front, bottom_back=bot.bottom_back, thumb=top.thumb,
                       size_table=top.size_table, top_body_region=top.top_body_region)
            self._mix[key] = o
        return o
