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
compositor.compose(rgba, track, gallery_state) -> np.ndarray         # (H, W, 3) BGR camera space
calibration.to_projector(img_cam) -> np.ndarray                      # (1080, 1920, 3) BGR
gestures.update(track) -> list[str]                                  # events: "next" | "prev" | "toggle_gallery"
sizing.update(track) -> None ; sizing.recommendation() -> tuple[str, str]  # (size_text, measure_text)
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
