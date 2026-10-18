# FitCast — projection-mapped virtual try-on

Single Python 3.11 process, multithreaded, no server, no network at runtime.
Person in all-white clothes stands on a floor mark; camera (iPhone Continuity
or webcam) mounted on the projector tracks them; projector paints garments on
the clothes. Mirror faces the person.

**Safety rule that overrides everything: never project light onto the head.
`compositor.compose()` forces `track.head_box` to black as its LAST operation,
every frame, no exceptions.**

## Pipeline

```
CaptureThread -> Tracker -> Smoother -> Renderer -> Compositor -> Calibration.to_projector -> Display
  (latest       (pose,     (One Euro   (RGBA in    (mask, head    (homography warp to      (fullscreen on
   frame only)   mask,      + predict)  camera px)   blackout,      1920x1080)               projector +
                 head,                               gallery)                                debug window)
                 facing)
```

Camera space = the processed frame size (`capture.width` x `capture.height`,
default 1280x720). Everything up to `to_projector()` lives in camera pixels.

## Interface contract (do not drift from this)

All shared types live in `fitcast/types.py`.

```python
@dataclass
class TrackResult:
    ok: bool                          # False when no person detected
    joints: np.ndarray                # (33, 3) float32, x,y camera pixels, z relative depth
    visibility: np.ndarray            # (33,) float32 in [0, 1]; 0 when landmark is outside the frame
    clothes_mask: np.ndarray          # (H, W) uint8 0/255, camera space, already post-processed
    head_box: tuple[int,int,int,int]  # x0, y0, x1, y1 camera pixels, padded (covers hats)
    facing: str                       # "front" | "back" | "side"
    timestamp: float                  # time.perf_counter() of the capture
    shoulder_width: float             # px, current
    ref_shoulder_width: float         # px, reference frontal width (running max)
    person_mask: np.ndarray | None    # (H, W) uint8 dilated whole-person mask (None for backends without one)
    facing_ratio: float               # normalised shoulder width / frontal reference (1.0 = squarely frontal)

@dataclass
class Outfit:
    id: str; name: str; sleeve: str   # sleeve: "short" | "long" | "none"
    top_front, top_back, sleeve_tex, bottom_front, bottom_back: np.ndarray | None   # RGBA uint8
    thumb: np.ndarray                 # BGR uint8
    size_table: str                   # "mens" | "womens"
    top_body_region: tuple[float,float,float,float]   # x0,y0,x1,y1 normalized in top image

@dataclass
class GalleryState:
    outfits: list[Outfit]; index: int; prev_index: int; changed_at: float
    visible: bool; size_text: str; measure_text: str

renderer.draw(track: TrackResult, outfit: Outfit) -> np.ndarray      # (H, W, 4) RGBA camera space
compositor.compose(rgba, track, gallery_state, now=None) -> np.ndarray   # (H, W, 3) BGR camera space; `now` = perf_counter (tests)
calibration.to_projector(img_cam) -> np.ndarray                      # (1080, 1920, 3) BGR
calibration.load(path=None, allow_simulated=False) -> bool           # simulated files only for video-file sources
gestures.update(track) -> list[str]                                  # events: "next" | "prev" | "toggle_gallery"
sizing.update(track) -> None ; sizing.recommendation(size_table=None) -> tuple[str, str]  # (size_text, measure_text)
```

Landmark indices follow MediaPipe Pose 33-point order. Other backends map to
it inside `tracker.py`. Key indices: 0 nose, 2/5 eyes, 7/8 ears, 11/12
shoulders (L/R), 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28
ankles. "L" = the person's left, which appears on the image RIGHT when facing
the camera.

## Conventions

- Every tunable lives in `config.yaml`; modules receive their sub-dict. No magic numbers in code.
- Images are BGR uint8 (OpenCV) unless the name says RGBA. Masks are uint8 0/255.
- No per-frame allocation in hot loops where a reusable buffer works.
- No silent failures: missing camera/model/calibration logs a clear fix-it message.
- Type hints and a docstring on every public function. Modules stay small.
- `pytest tests/` must pass before any commit. Run: `.venv/bin/python -m pytest -q`.
- Python: `.venv/bin/python` (3.11, created with `uv venv --python 3.11 .venv`).
- Test clips: `assets/test/white_turn.mp4`, `assets/test/pink_turn.mp4` (4K, 25 fps, cropped above the knees).
- Only the lead edits this file and `fitcast/types.py`. Builders propose changes in their report.
- Record non-obvious choices in `DECISIONS.md`, milestone checks in `PROGRESS.md`.


## 3D sensor path (iPhone 17 Pro LiDAR) — see docs/BUILD_SPEC_3D.md

`--sensor webcam` (default) keeps everything above. `--sensor iphone` replaces capture+tracker with
`fitcast/sensors/iphone.py` and replaces `Calibration.to_projector` with `fitcast/render3d.py`.
TrackResult gains `depth`, `joints3d`, `K_cam` (None on the webcam path). Camera space is still the
sensor's colour image (downscaled to `capture.width x height`); depth is resampled to it.

### Wire protocol (phone -> Mac), `fitcast/sensors/protocol.py`
TCP stream, little-endian. Message = fixed header + JSON header + payload:
```
u32 magic = 0x46435331 ('FCS1')   u8 version = 1   u8 msg_type (1 frame, 2 info, 3 ping)
u16 json_len                       u32 payload_len
json[json_len]                     payload[payload_len]
```
Frame JSON: `ts` (ARFrame timestamp, s), `send_ts` (phone monotonic, s), `img_w`, `img_h`, `K` = [fx, fy, cx, cy]
for the captured image, `depth_w`, `depth_h`, `rgb_w`, `rgb_h`, `joint_names` (19 Vision names),
`joints2d` [[x, y, conf] x 19] in captured-image pixels (NaN when missing), `joints3d` [[X, Y, Z] x 19] metres in
ARKit camera coords (+x right, +y up, -z forward; NaN when no valid depth),
`blobs`: list of {name, dtype, shape, offset, nbytes} describing the payload:
`depth` float16 (depth_h, depth_w) metres; `conf` uint8 (depth_h, depth_w) 0..2 (ARKit confidence);
`seg` uint8 (seg_h, seg_w) 0/255 person segmentation (optional); `rgb` JPEG bytes (optional).
Depth intrinsics = K scaled by depth_w/img_w, depth_h/img_h. The Mac converts ARKit camera coords to OpenCV
(y and z negated). Transport: the phone is a TCP server on `sensor.port` (7777); the Mac connects over
`pymobiledevice3` usbmux port forwarding (USB) or a Wi-Fi IP.

### Vision 19 -> MediaPipe 33 joint mapping
nose 0, left_eye 2, right_eye 5, left_ear 7, right_ear 8, left_shoulder 11, right_shoulder 12, left_elbow 13,
right_elbow 14, left_wrist 15, right_wrist 16, left_hip 23, right_hip 24, left_knee 25, right_knee 26,
left_ankle 27, right_ankle 28. `neck` and `root` are not mapped. Remaining face indices copy the nearest eye/nose
(as the rtmlib backend does). Confidence -> visibility.

### Module contracts
```python
# fitcast/depthmask.py
class DepthBodyMask:
    def __init__(self, cfg: dict, depth_size: tuple[int,int], cam_size: tuple[int,int]) -> None   # cfg = config["depthmask"]
    def learn_background(self, depth_frames: list[np.ndarray], conf_frames: list[np.ndarray]) -> None   # median of the empty scene
    @property ready(self) -> bool
    def compute(self, depth: np.ndarray, conf: np.ndarray, rgb_cam: np.ndarray,
                person_seg: np.ndarray | None = None, skeleton_mask: np.ndarray | None = None) -> np.ndarray  # (H,W) uint8 camera res

# fitcast/render3d.py
class DepthMeshProjector:
    def __init__(self, cfg: dict, cam_size: tuple[int,int], proj_size: tuple[int,int],
                 K_cam: np.ndarray, K_p: np.ndarray, dist_p: np.ndarray, R: np.ndarray, t: np.ndarray) -> None  # cfg = config["render3d"]
    def render(self, image_cam: np.ndarray, depth_cam: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray
        # image_cam (H,W,3) BGR camera space; depth_cam (H,W) float32 metres (0 = invalid) at camera res; -> (ph,pw,3) BGR
    def set_extrinsics(self, R, t) -> None

# fitcast/calib3d.py (+ scripts/calibrate3d.py)
@dataclass
class Calibration3D: K_cam, dist_cam, K_p, dist_p, R, t, cam_size, proj_size, reproj_error_px, n_points, timestamp, notes
def save_calibration3d(path, c) / load_calibration3d(path) -> Calibration3D
def gray_code_patterns(proj_size) -> list[np.ndarray]           # + white + black, OpenCV GrayCodePattern
def decode_gray_code(captures: list[np.ndarray], white, black, proj_size, cfg) -> (proj_xy (H,W,2) float32, valid (H,W) bool)
def solve_projector(cam_pts3d (N,3), proj_pts2d (N,2), proj_size, throw_ratio_guess, cfg) -> Calibration3D-ish + error

# fitcast/sensors/iphone.py
class IPhoneSensor:   # background thread, latest frame only
    def __init__(self, cfg: dict, cam_size: tuple[int,int]) -> None   # cfg = config["sensor"]
    def start(self) -> "IPhoneSensor"; def stop(self) -> None
    def read(self) -> tuple[np.ndarray, float, int] | None            # (frame_bgr camera space, timestamp, seq) like FrameSource
    def track(self) -> TrackResult | None                             # latest decoded TrackResult (depth, joints3d, K_cam filled)
```
Only the lead edits this file and `fitcast/types.py`.
