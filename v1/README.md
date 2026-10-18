# Fitcast — a projector that dresses you

Stand on the taped spot in a baggy white T-shirt and white pants. A webcam next to
the projector lens finds your body, a garment image is warped onto your torso and
limbs, and the projector paints it onto you. Swipe a hand to change the garment,
raise a hand to switch between shirt and pants, and turn around: the projection
switches between the front, side and back photos of the garment. Hold your palm up
to buy what you're wearing through a (mock) Visa checkout. A laptop window shows
the camera feed with the garment composited on: a magic mirror that doubles as the
fallback demo if projection alignment fails.

DubHacks 2026, Visa track.

## Quick start

This version lives in `v1/` of the repo; run everything from there.

```bash
cd v1
./setup.sh                        # Python 3.11 venv, dependencies, MediaPipe models, placeholder garments
source .venv/bin/activate

python tools/make_placeholders.py # 3 shirts + 2 pants as labelled shapes (setup.sh already ran this)
python -m tryon                   # webcam only: laptop magic mirror (no calib.json -> debug-only mode)

python tools/calibrate.py         # projector connected: solve camera->projector, writes calib.json
python -m tryon                   # now also fullscreen on the projector
```

Useful flags for `python -m tryon`:

| flag | what |
|---|---|
| `--source 1` / `--source clip.mp4` | other camera index, or replay a recorded video instead of the camera |
| `--record clip.mp4` | save the raw camera feed so a session can be replayed and tuned later |
| `--no-projector` | laptop window only, even when calib.json exists |
| `--headless --duration 10` | no windows, quit after 10 s (testing) |
| `--snapshot out/shot` | on exit, save the last debug and projector frames |
| `--keys "1:right,2:space"` | scripted key presses at given seconds (testing) |
| `--config other.yaml` | another config file |

## Controls

| gesture | keyboard | action |
|---|---|---|
| swipe a hand sideways, above the hips | `←` / `→` | previous / next garment in the active layer |
| raise a hand above your nose for 0.5 s | `space` | toggle active layer: shirt ↔ pants |
| hold a hand still between shoulder and nose height for 1.5 s (a gold ring fills) | `enter` | buy the active garment (mock Visa checkout) |
| | `c` | reload calib.json (and open the projector window if it wasn't) |
| | `d` | toggle debug overlays (keypoints, HUD) |
| | `f` | toggle fullscreen (projector window, or the laptop window when not projecting) |
| | `q` / `esc` | quit |

Gestures only fire while you face the camera and are not mid-turn. Swiping toward
your own left means "next" (`swipe_left_is_next`). Both the shirt and the pants are
always drawn; the mode only decides which one a swipe or a purchase acts on.

## Demo checklist

1. **Dark corner.** Pick the darkest corner you can get, with a plain light wall. Put a
   soft lamp to the side of the wearer so the camera still sees them. The projection
   must beat the ambient light.
2. **Projector + camera.** Put the Epson on a table or stand 2.5–3 m from the spot, at
   chest height. Tape the webcam (or iPhone, through Continuity Camera) so it touches
   the projector lens and points the same way. The closer they are, the less
   parallax.
3. **macOS display settings.** Set the projector to an *extended* display, not mirrored,
   at **1920×1080 "Default for display"** (not a scaled mode). Turn off sleep and
   screen saver, plug the Mac in, and turn on Do Not Disturb.
4. **Tape the spot.** From the spot, the camera must see you from head to ankles. Check
   this in the laptop window (`python -m tryon`).
5. **Calibrate.** Run `python tools/calibrate.py` and hold a white foam board flat at the
   spot, at chest height, facing the projector. Use `+`/`-` and the arrow keys until the
   whole projected checkerboard sits on the board, then press `space`. Step in: a green
   dot should land on your nose for 10 s. Press `q` to save, or `r` to retry. If the
   board won't detect, darken the room or press `g` for the dot-grid fallback (keep the
   board still).
6. **Run** `python -m tryon`. The HUD should read about 30 fps for render, camera and
   pose. Calibrate again if anything moves.
7. **Wearer.** Baggy white T-shirt and white pants. If the garment is too narrow or too
   wide, adjust `torso_width_scale` / `garment_scale` / `pants_width_scale`.
8. **Rehearse.** Swipe, raise a hand, hold a palm up, and turn slowly through a full
   circle. With the placeholders, the big `L` marker must stay on the wearer's left
   in every view, including the back.
9. **Fallback.** If the projection is off, recalibrate and press `c`. Or run the laptop
   magic mirror: `d` hides the overlays and `f` makes it fullscreen.
10. **Record a rehearsal.** Run `python -m tryon --record rehearsal.mp4`, then tune
    thresholds offline with `--source rehearsal.mp4`.

## How it works

```
capture thread ──newest frame──▶ pose worker thread ──PoseResult──▶ main thread (one iteration per camera frame)
 (1280×720@30)                   MediaPipe pose (full)              One Euro smoothing + 60 ms look-ahead
                                 selfie segmenter (256 px)          orientation (front/left/back/right)
                                 face detector on head crop         gestures → app state → checkout thread
                                                                     warp garment in camera px → mask
                                                                     ├─ homography → projector frame (black + garment + UI)
                                                                     └─ composite → mirrored laptop frame + HUD
```

- **Pipeline.** Inference runs on its own thread (MediaPipe releases the GIL). The
  renderer draws one frame per camera frame from the latest pose, predicted forward
  by `projector_latency_ms`, so a slow model lowers the pose rate but never the
  projector rate. Measured on an M5 Pro: pose 8–10 ms, segmenter 5 ms, warp 3–5 ms
  for shirt + pants, projector warp 1 ms. Render, camera and pose all run at 30 fps.
- **Warp.** The shirt torso is a 7-triangle mesh over the collar, shoulders, armpits,
  hems and a torso centre, with one affine per triangle. Adjacent triangles agree on
  their shared edges, so the torso bends without seams. Sleeves are rigid segments
  that pivot at the armhole centre and follow the elbow: a short sleeve is one
  segment, a long sleeve is two hinged at the elbow. They can follow the arm anywhere
  without folding inside out. Pants are a waist piece (everything above the crotch,
  mapped through the waist corners and crotch) plus a thigh and a shin segment per
  leg, hinged on the crotch line and at the knee. The crotch keeps the garment photo's
  own rise. Every garment pixel belongs to exactly one
  piece. Pixels outside the anchor mesh go to the nearest piece, so collar tabs and
  hems are extrapolated instead of cropped. The layer is built in camera pixels,
  multiplied by the person mask, and warped into projector pixels through the
  homography.
- **No mirrored back.** Garment anchors are matched to body joints by which side of
  the image they are on, not by MediaPipe's left/right labels. MediaPipe often labels
  a person facing away as if they faced the camera; that cannot flip the back view.
- **Degenerate cases.** Side views are narrow strips, so they use a 2-point similarity
  transform (shoulder-mid → hem-mid). The front/back mesh falls back to the same
  transform when two or more torso triangles fold or stretch more than 4×. Limbs are
  clamped to `max_limb_stretch`. A hidden joint makes the limb hang at its photo angle.
- **Orientation.** The image shoulder width over torso length gives |cos yaw|, and the
  world-z order of the shoulders gives the sign. MediaPipe's label swap makes each
  measurement ambiguous by 180°, so the estimator keeps the candidate closest to the
  previous yaw (people turn continuously). A face detector on the head crop re-anchors
  front vs back whenever the shoulders are square to the camera. Buckets use 15° of
  hysteresis, a 250 ms hold, then a 150 ms crossfade.
- **Safety.** A circle around the head is always black in the projector frame, whatever
  the segmenter says. The first 1.5 s is blank so the camera exposure settles.
- **Checkout.** `tryon/checkout.py:submit_payment()` is the single swap point. Without
  credentials it posts a CyberSource-shaped `/pts/v2/payments` request to a local mock
  server, which answers like CyberSource after 0.9 s. With `VISA_API_KEY`,
  `VISA_SHARED_SECRET` and `VISA_MERCHANT_ID` in `.env`, it signs the same request
  with CyberSource HTTP Signature auth and sends it to `apitest.cybersource.com`. It
  runs on a worker thread, and every attempt is appended to `transactions.jsonl`.

## Shooting and annotating real garments

1. **Hang it.** Put the garment on a hanger against a plain wall that contrasts with it.
   Light it evenly and avoid hard shadows. Pin or tape the sleeves out at about 30–45°
   from the body: the sleeve is warped separately from the torso, so it shouldn't lie
   flat against it. Hang pants by the waist with the legs slightly apart.
2. **Four angles.** Shoot straight on from chest height, filling the frame with the
   garment (1000–2000 px tall is plenty). Turn the hanger 90° between shots:
   - `front.png`: the front.
   - `left.png`: the garment's own left side (the wearer's left arm) toward the camera.
   - `back.png`: the back.
   - `right.png`: the garment's right side toward the camera.

   Only `front.png` is required. A missing back reuses the front, and a missing side is
   mirrored from the other side.
3. **Remove the background with rembg.** Install it in its own venv, because it pulls in
   `opencv-python-headless`, which would clash with this project's OpenCV:
   ```bash
   python3.11 -m venv ~/rembg-venv && ~/rembg-venv/bin/pip install "rembg[cpu,cli]"
   ~/rembg-venv/bin/rembg i photo_front.jpg assets/garments/my_tee/front.png
   ~/rembg-venv/bin/rembg p photos/ cutouts/          # or a whole folder
   ```
   Open the PNGs in Preview and check that the background is transparent.
4. **Annotate.**
   ```bash
   python tools/annotate.py assets/garments/my_tee --category shirt --sleeve short --name "My Tee" --price 30
   python tools/annotate.py assets/garments/my_jeans --category pants --name "My Jeans" --price 60
   ```
   Click each anchor as it's prompted. Keys: `s` skip, `u` undo, `[`/`]` switch view,
   `w` write, `q` write and quit.
   - Left/right always mean the **wearer's** left/right. On `front.png` the wearer's
     left is on the image's right; on `back.png` it is on the image's left.
   - Click the centre of each cuff, knee and ankle. Widths are measured from the PNG's
     alpha.
   - Shirt anchors are `left/right_shoulder` (the outer shoulder seam), `left/right_armpit`,
     `left/right_hem` (bottom corners), `left/right_sleeve_end` and `collar` (centre of the
     neckline).
   - Pants anchors are `left/right_waist` (top corners of the waistband),
     `left/right_knee`, `left/right_ankle` and `crotch`.
   - Side views only need the shoulders/collar and hems (shirts), or the waist and
     ankles (pants).
5. **Order the catalog.** Garments appear alphabetically by folder name, or in the
   order of `catalog:` in config.yaml. `meta.json` holds `name`, `price`, `category`
   (`shirt` or `pants`) and `sleeve_length` (`short` or `long`). Delete the placeholder
   folders once you have real ones, or list only the real ones in `catalog:`.

## Visa / CyberSource

Put credentials in `.env` (git-ignored) to switch from the mock to the CyberSource
sandbox:

```
VISA_API_KEY=<key id>
VISA_SHARED_SECRET=<base64 shared secret>
VISA_MERCHANT_ID=<merchant id>
# VISA_HOST=apitest.cybersource.com   (default)
```

The payment uses the public Visa test card 4111 1111 1111 1111. Without all three
values the app uses the mock and says so on startup (`checkout provider: mock`).
To demo a decline, set `checkout_mock_decline_rate: 1.0`.

## Config knobs

All in `config.yaml`, loaded into one dataclass (`tryon/config.py`):

<!-- CONFIG_TABLE -->
| knob | default | what it does |
|---|---|---|
| **camera** | | |
| `camera_index` | `0` | OpenCV camera index (0 is usually the first camera; Continuity Camera can shift it) |
| `camera_source` | `""` | path to a video file to replay instead of the camera ("" = use camera_index) |
| `camera_width` | `1280` | capture width in px (all pixel thresholds assume this ~720p frame) |
| `camera_height` | `720` | capture height in px |
| `camera_fps` | `30` | requested camera frame rate; also the inference budget |
| `video_loop` | `true` | loop camera_source video files |
| **models + inference** | | |
| `pose_model` | `models/pose_landmarker_full.task` | pose model (full ~8 ms on M-series CPU; lite ~6 ms) |
| `segmenter_model` | `models/selfie_segmenter.tflite` | person segmenter used when mask_source is segmenter |
| `face_model` | `models/blaze_face_short_range.tflite` | face detector run on the head crop as the front/back cue |
| `mask_source` | `segmenter` | where the person mask comes from: segmenter | pose | none |
| `seg_input_width` | `256` | segmenter input width in px (frame is downscaled to this first) |
| `seg_threshold` | `0.5` | person-probability cutoff for the binary mask |
| `seg_every_n_frames` | `0` | run the segmenter every N frames (0 = auto: 1, or 2 if inference misses the budget) |
| `min_pose_confidence` | `0.5` | MediaPipe detection/presence/tracking confidence |
| `min_visibility` | `0.5` | keypoint visibility below this is treated as missing |
| `face_detect_every_n` | `3` | run the head-crop face detector every N pose frames (0 = off) |
| **smoothing + latency** | | |
| `one_euro_min_cutoff` | `1.0` | One Euro min cutoff (Hz): lower = steadier when standing still |
| `one_euro_beta` | `0.02` | One Euro speed coefficient (per 720p px/s): higher = less lag when moving |
| `one_euro_d_cutoff` | `1.0` | One Euro derivative cutoff (Hz) |
| `projector_latency_ms` | `60` | look-ahead to cancel projector + pipeline lag (constant-velocity prediction) |
| `max_prediction_px` | `30` | cap on the look-ahead displacement (720p px) |
| `lost_reset_ms` | `500` | forget smoothing history after the person is gone this long |
| **garments + fit** | | |
| `garments_dir` | `assets/garments` | folder of garment folders |
| `catalog` | `[]` | garment folder names in swipe order ([] = every folder, alphabetical) |
| `garment_max_size` | `1024` | garment PNGs are downscaled so their long side is at most this |
| `garment_scale` | `1.0` | overall garment scale about the torso centre |
| `torso_width_scale` | `1.15` | shirts: widen shoulders/hips, since real shirts are wider than the skeleton |
| `armpit_drop` | `0.30` | shirts: armpit = this fraction of the way from shoulder to hip |
| `hem_drop` | `0.05` | shirts: hem sits this fraction of torso length below the hip joints |
| `collar_raise` | `0.08` | shirts: collar = shoulder midpoint raised by this fraction of torso length |
| `pants_width_scale` | `1.35` | pants: waist width relative to the hip-joint spacing |
| `waist_raise` | `0.15` | pants: waistband sits this fraction of torso length above the hip joints |
| `crotch_drop` | `0.0` | pants: extra crotch drop (fraction of thigh length) beyond the garment photo's own rise |
| `long_sleeve_elbow_frac` | `0.47` | long sleeves: where the elbow hinge sits along the sleeve (0 = armhole, 1 = cuff) |
| `max_limb_stretch` | `2.5` | sleeves/legs never stretch beyond this many times their photo length |
| **mask + light** | | |
| `mask_dilate_px` | `8` | grow the person mask so loose clothing edges still get garment |
| `mask_feather_px` | `7` | soften the mask edge (px) |
| `head_exclusion_scale` | `1.4` | size of the never-project circle over the head (always on, eye safety) |
| `garment_brightness` | `0.9` | projector brightness of the garment (lower if the camera exposure hunts) |
| `fade_out_ms` | `300` | garment fade-out when the person is lost |
| `fade_in_ms` | `150` | garment fade-in when the person appears |
| `startup_blank_ms` | `1500` | project nothing for this long at start so camera exposure settles |
| **orientation** | | |
| `orientation_hysteresis_deg` | `15` | extra degrees past a bucket edge before the view may change |
| `orientation_hold_ms` | `250` | the new view must persist this long before switching |
| `view_crossfade_ms` | `150` | crossfade between front/side/back images |
| `shoulder_ratio_ref` | `0.70` | shoulder width / torso length when square to the camera (auto-adapts) |
| `orientation_z_sign` | `1.0` | flip to -1 if left/right side views come out swapped |
| `orientation_z_deadband` | `0.06` | ignore world-z shoulder differences smaller than this (torso lengths) |
| `face_front_threshold` | `0.5` | face cue above this = facing the camera |
| `face_back_threshold` | `0.2` | face cue below this = facing away |
| **gestures** | | |
| `gestures_front_only` | `true` | only accept gestures while facing the camera |
| `swipe_speed_px_s` | `1000` | swipe: wrist speed relative to torso (720p px/s) |
| `swipe_min_distance_px` | `110` | swipe: minimum sideways travel (720p px) |
| `swipe_window_ms` | `160` | swipe: time window the speed is measured over |
| `swipe_debounce_ms` | `800` | swipe: minimum time between swipes |
| `swipe_left_is_next` | `true` | swipe toward the wearer's own left = next garment (false: toward right = next) |
| `raise_hold_ms` | `500` | raise hand: wrist above nose this long toggles shirt/pants |
| `raise_debounce_ms` | `1500` | raise hand: minimum time between toggles |
| `palm_hold_ms` | `1500` | palm hold: wrist still between shoulder and nose this long = checkout |
| `palm_still_speed_px_s` | `140` | palm hold: wrist counts as still below this speed (720p px/s) |
| `gesture_cooldown_ms` | `600` | no gesture fires within this long after any other gesture |
| `turn_rate_suppress` | `0.8` | block gestures while the body turns faster than this (shoulder-ratio units/s) |
| **projector + windows** | | |
| `calib_path` | `calib.json` | camera->projector homography file written by tools/calibrate.py |
| `projector_width` | `1920` | projector frame width |
| `projector_height` | `1080` | projector frame height |
| `projector_display` | `-1` | display index for the projector (-1 = first non-main display) |
| `projector_backend` | `auto` | auto (cv2, falling back to pygame) | cv2 | pygame |
| `projector_allow_main_display` | `false` | allow the projector window on the laptop screen (testing without a projector) |
| `debug_window_width` | `1280` | laptop debug window width |
| `debug_every_n_when_projecting` | `2` | while projecting, refresh the laptop window every Nth frame (macOS window redraws are slow) |
| `show_debug_overlays` | `true` | start with keypoints/HUD visible (toggle with d) |
| `ui_scale` | `1.0` | size of projected text, ring and receipt |
| **checkout** | | |
| `currency` | `USD` | currency sent with payments |
| `checkout_display_ms` | `3000` | how long the paid/declined receipt stays up |
| `checkout_mock_latency_ms` | `900` | simulated processor latency of the mock Visa endpoint |
| `checkout_mock_decline_rate` | `0.0` | fraction of mock payments that decline (to demo the failure path) |
| `transactions_path` | `transactions.jsonl` | every payment attempt is appended here |
| `env_path` | `.env` | where VISA_API_KEY / VISA_SHARED_SECRET / VISA_MERCHANT_ID are read from |
| **perf** | | |
| `profile_print_after_s` | `5` | print one per-stage timing summary to the console after this many seconds |

## Tests

```bash
python -m unittest discover -s tests -v      # 189 tests, ~40 s, no camera or projector needed
```

The suite covers:
- synthetic skeletons: turning both ways with and without MediaPipe's label swap,
  reappearing side-on or facing away, swipes, raise hand, palm hold, return strokes
  and tracker glitches
- every garment × view × awkward pose through the warp (crossed arms, overhead, side-on,
  half out of frame, wild keypoints), back-view mirroring, seams and holes, and warp
  speed (shirt + pants about 2.6 ms)
- renderer checks: pure-black background, head exclusion, fades, crossfades
- homography recovery from a synthetic camera view of the projected checkerboard
- the checkout mock and the CyberSource signing
- a full headless app run on a video generated from `tests/data/pose.jpg`
  (MediaPipe's public pose test image)

## Repo layout

Paths are relative to `v1/`.

```
tryon/
  __main__.py        app loop (python -m tryon)
  config.py          every tunable in one dataclass, loaded from config.yaml
  capture.py         threaded camera reader, always the newest frame (or a video file)
  pose.py            MediaPipe pose + segmenter + head-crop face cue, on a worker thread
  smoothing.py       One Euro filter per keypoint + latency look-ahead
  calibration.py     homography, checkerboard / dot patterns, calib.json load/save
  garments.py        catalog, view PNGs, anchors, meta
  warp.py            garment view -> body warp (torso mesh + limb segments + similarity fallback)
  orientation.py     front/left/back/right with hysteresis, hold and crossfade
  gestures.py        swipe, raise hand, palm hold
  state.py           mode, catalog indices, checkout state machine
  render.py          projector frame + mirrored debug frame + projected UI
  checkout.py        mock Visa endpoint + CyberSource-ready payment call, transactions.jsonl
  display.py         projector window on the second display (cv2, pygame fallback), key mapping
  projector_child.py pygame projector window process
tools/
  calibrate.py       standalone calibration tool
  make_placeholders.py  placeholder garments
  annotate.py        click anchors onto garment PNGs
assets/garments/<id>/{front,left,back,right}.png + anchors.json + meta.json
tests/               unittest suite
config.yaml  calib.json (made by calibrate.py)  requirements.txt  setup.sh
```

## Judgment calls (things I had to guess)

- **Person mask.** The mask comes from MediaPipe's `selfie_segmenter` (binary person
  mask, 256 px input, about 5 ms). I benchmarked `selfie_multiclass_256x256` and dropped
  it: it took 19 ms and was weak at full-body distance. `mask_source: pose` uses the pose
  model's own mask instead (nearly free), and `none` turns masking off.
- **Head exclusion is always on.** The spec asked for nothing on the face. Rather than
  trust the segmenter for eye safety, a landmark-based circle over the head is always
  black.
- **Front vs back.** A BlazeFace (short-range) face detector runs on an upscaled head
  crop every third frame. MediaPipe's own face visibility stays high even from behind,
  so it is only a fallback when the detector is off. "No face" only counts as facing
  away once the detector has seen this wearer's face at least once. That way, a
  detector that is out of range at the taped spot can't flip a front-facing person to
  the back view; turning is still tracked by continuity. The thresholds
  (`face_front_threshold`, `face_back_threshold`) were not tuned on a real wearer.
- **The MediaPipe left/right swap.** It is handled twice. The warp matches anchors by
  image side, so the back view can't mirror. The orientation code keeps whichever of
  yaw / yaw+180 continues the turn. `orientation_z_sign` flips the side views if
  left/right ever come out swapped on the real camera.
- **Sleeves and legs are rigid hinged segments,** not quads sharing the torso's armpit
  point. The quad version folds inside out as soon as an arm goes above the shoulder
  or hangs straight down. The torso is the piecewise-affine triangle mesh from the spec.
- **Short sleeves end at the elbow,** as specified. Most real tees end mid-bicep, so
  shoot and annotate accordingly, or the sleeve will look stretched.
- **Fit constants are educated guesses:** `hem_drop` 0.05, `waist_raise` 0.15,
  `pants_width_scale` 1.35 and `long_sleeve_elbow_frac` 0.47. Tune them on the real
  wearer.
- **Crotch position.** The spec put the crotch at "hip midpoint shifted down".
  Pinning it there squashed the waist region of any garment whose photo had a deeper
  rise, and opened a hole above the crotch. The crotch now keeps the garment photo's
  own rise (crotch depth ÷ waist-to-ankle length), and `crotch_drop` (default 0) only
  adds an extra drop.
- **Gesture zones.** "Raise hand" is a wrist above the nose. "Palm hold" is a wrist held
  still *between* shoulder and nose height, so the two can't trigger together.
  "Swipe toward your own left = next" is a coin flip, and `swipe_left_is_next`
  reverses it. Swipe speed is measured relative to the torso, one-frame keypoint jumps
  are ignored, and all gestures are blocked while turning or facing away.
- **Projector window.** cv2 is tried first, as specified, then checked with
  `getWindowImageRect`. If it didn't land on the projector, pygame takes over in a
  *separate process* (`tryon/projector_child.py`). OpenCV 5 bundles its own SDL2, and
  two SDL copies in one process trigger Cocoa class clashes. Measured on the laptop
  screen, the pygame path left more headroom (22 ms loop vs 29 ms). If the HUD shows the
  render fps dropping below 30 on the projector, set `projector_backend: pygame`.
- **Laptop window throttling.** On macOS every `cv2.waitKey` costs about 12 ms, and
  redrawing two windows about 25 ms. While projecting, the laptop window therefore
  refreshes every other frame (`debug_every_n_when_projecting`). The projector frame
  still updates every frame.
- **Calibration extras.** `+`/`-` and the arrow keys size and move the projected board
  so it fits on the foam board, and `g` runs a dot-grid fallback. The spec only had
  space/r/q.
- **CyberSource.** The real path signs requests with HTTP Signature auth (HmacSHA256 over
  `host date (request-target) digest v-c-merchant-id` with the base64-decoded secret).
  CyberSource's SDKs write `(request-target)` and parts of its docs write
  `request-target`, so on a 401 it retries once with the other spelling. It also needs
  `VISA_MERCHANT_ID`, which the spec didn't mention. It has **not** been run against
  the sandbox, because there were no credentials.
- **Mock endpoint.** The mock is a real local HTTP server, so the request/response
  path matches the real one. Two macOS quirks are handled in code: `HTTPServer`'s
  reverse-DNS lookup on bind, and urllib's system-proxy lookup. Each stalled for 30+ s
  on venue wifi.
- **Dependencies are pinned** to what pip resolved on Oct 17 2026: mediapipe 1.1.1 and
  opencv-contrib 5.0. mediapipe needs `opencv-contrib-python`, so plain `opencv-python`
  must not be installed alongside it.
- **The laptop window is a true mirror,** so text printed on a garment reads backwards
  there, as it would in a real mirror. The projector frame is never flipped.
- **View naming.** `left.png` is the wearer's *left* side facing the camera, which is what
  the camera sees after the wearer turns 90° to their right.

## TODO

- [ ] Shoot, cut out (rembg) and annotate the real demo garments, then remove the placeholders from `catalog:`.
- [ ] Calibrate on site with the real projector; check `projector_backend` (cv2 vs pygame) by watching the HUD fps.
- [ ] Record a rehearsal (`--record`) of the real wearer turning and gesturing, then tune `face_*_threshold`, `orientation_z_sign`, `swipe_speed_px_s` and `palm_still_speed_px_s` on it.
- [ ] Tune the fit constants (`torso_width_scale`, `hem_drop`, `pants_width_scale`, `waist_raise`, `crotch_drop`) on the white outfit.
- [ ] Run one CyberSource sandbox payment with real keys in `.env` and confirm the signature format is accepted.
- [ ] The homography is exact only at the calibration plane; a person leaning forward or back drifts slightly. A two-depth calibration could fix that.
- [ ] Camera auto-exposure can't be locked through OpenCV's AVFoundation backend; if the projected garment makes exposure pump, lower `garment_brightness` or lock exposure in a camera app first.
