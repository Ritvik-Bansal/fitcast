# FitCast — projection-mapped virtual try-on

A person in plain white clothes stands on a floor mark. A camera on top of the projector tracks
their body; the projector paints shirts, pants and full outfits onto the white fabric. A mirror in
front of them shows the result. Swipe an arm to change outfit; a gallery and a size estimate are
projected on the wall beside them.

Everything runs locally in one Python process. No network at runtime.

**Safety:** the head box is forced to black on every frame, after everything else. Never disable it.

## 1. Setup from a clean Mac (Apple Silicon, macOS 26)

```bash
# 1. Python 3.11 via uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this repo> && cd rad-AR
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# optional RTMPose backend (must be --no-deps, see requirements-rtm.txt)
uv pip install --python .venv/bin/python --no-deps -r requirements-rtm.txt

# 2. models (one-time download, ~30 MB; add --rtm for the RTMPose ONNX files)
.venv/bin/python scripts/download_models.py --rtm

# 3. demo outfits (procedural, zero downloads)
.venv/bin/python scripts/prepare_garments.py --demo

# 4. tests + an offline run on a test clip
.venv/bin/python -m pytest -q
.venv/bin/python main.py --source assets/test/white_turn.mp4 --no-projector
```

Camera permission: the first time you open a camera, macOS asks to allow the terminal app
(Terminal / iTerm / VS Code). If `--list-cameras` prints `not available` for every index, grant it in
System Settings > Privacy & Security > Camera and restart the terminal.

Verified versions (2026-10-17): mediapipe 1.1.1, opencv-contrib-python 5.0.0.93, numpy 2.4.6,
onnxruntime 1.31.0 (CoreML EP present), rtmlib 0.0.16, Python 3.11.15.

## 2. Hardware

- **Projector** Epson EpiqVision Flex CO-FH02 (1920x1080). Connect over HDMI. In System Settings >
  Displays make it an **extended** display (not mirrored), 1920x1080, placed to the right of the
  laptop screen. `main.py` detects the second display and opens the projector window there; override
  with `--projector-x/--projector-y` if the arrangement differs. Press `f` to toggle fullscreen.
- **Camera** iPhone via Continuity Camera (appears as a normal device) or any webcam. Mount it
  **on top of the projector, pointing the same way**, as close to the lens as possible: the system
  uses a planar homography, so camera/projector parallax is what limits alignment.
  `python main.py --list-cameras` prints what OpenCV can open. Processing runs at 1280x720.
- **Person** stands on a taped floor mark ~2.5-3.5 m from the projector, facing a full-length mirror.

## 3. Calibration

Short version (what worked on the real rig):

```bash
.venv/bin/python scripts/calibrate.py --preview 40          # aim: all 9 dots inside the camera view, nobody in frame
.venv/bin/python scripts/calibrate.py --auto                # markers one at a time, no clicks; prints reprojection + VERIFY
.venv/bin/python scripts/calibrate.py --refine              # person stands on the mark: probe discs measure the depth offset
.venv/bin/python main.py --demo
```

Rules: camera as close to the projector lens as possible, pointing the same way; nothing may move after
`--auto`; `--refine` must be redone whenever the standing mark or the camera moves. Keys `[` / `]` nudge all
garments up/down live if the collar sits wrong for a particular person.

### Full walkthrough

```bash
.venv/bin/python scripts/calibrate.py --source 0            # live
.venv/bin/python scripts/calibrate.py --simulate            # no hardware: synthetic rig -> calibration_sim.yaml, must print VERIFY PASS
# calibration_sim.yaml is marked simulated and is REFUSED for live cameras; pass --calibration calibration_sim.yaml for clip runs only
```

Live walkthrough (someone holds a large white board at the floor mark, or the person stands there):

1. The projector shows black; the tool grabs a dark frame.
2. It projects a 3x3 grid of bright circles and grabs a lit frame.
3. Markers are detected on the difference image. If the count is wrong, press `m` and click the
   markers in the debug window in the printed order (`u` undoes a click).
4. The homography is solved; the **reprojection error** is printed. Good: **< 2 px** (RMS, in
   projector pixels). 2-6 px: usable, check the board was flat and facing the camera. > 6 px: redo it.
5. Click two corners in the debug window to define the **wall gallery rectangle** (a strip on the
   wall beside the person, where no one stands).
6. Verification: a second grid is projected and re-detected; **VERIFY PASS** means every point lands
   within `calibration.verify_tolerance_px` (6 px) of where it should.
7. `s` saves `calibration.yaml` (timestamp, error, gallery rect, camera size). `main.py` loads it at
   start and `r` reloads it while running.

What "good" looks like live: `python main.py --source 0 --mask-mode skeleton` projects a solid color
onto the body; edges should sit on the clothing edges within ~1-2 cm at the floor mark. If the color
slides when the person steps forward/back that is parallax: move the camera closer to the lens.

## 4. Room setup

- Tape an X on the floor; everything is calibrated for that plane.
- Full-length mirror facing the person, outside the projector beam.
- Dark, matte backdrop behind the person (a black curtain). The segmenter is trained on people, but a
  light wall at the same brightness as a white shirt costs mask quality.
- Dim lights over the projection area; ambient light anywhere else is fine. 3000 lumens is enough
  for a lit room if the beam area is not under a spotlight.
- Nothing bright in the camera's view that moves (screens, windows).

## 5. Adding an outfit

An outfit is a folder under `assets/garments/<id>/` with `meta.yaml` and RGBA PNGs:
`top_front.png`, `top_back.png`, `sleeve.png`, `bottom_front.png`, `bottom_back.png`, `thumb.png`.

- From a VITON-HD or DressCode folder: `scripts/prepare_garments.py --import <dir>`.
- From product photos on white: put them in a folder and run the same command; the background is
  removed by thresholding (`garments.white_bg_threshold`).
- Missing backs are synthesized (mirror + collar/graphic inpaint), sleeves are sampled from the
  fabric, colors are boosted (`garments.saturation_boost`, `brightness_boost`) because projected color
  on white fabric looks washed out. Near-black garments are warned about: black does not project.
- `meta.yaml` fields: `name`, `category`, `sleeve` (short/long/none), `size_table` (mens/womens),
  `parts`, `top_body_region` (normalised x0,y0,x1,y1 of the torso in the top image, excluding sleeves).

## 6. Running

```bash
.venv/bin/python main.py --source 0 --demo --height 178
```

Keys (debug window focused): `←/→` or `n/p` outfit, `c` re-capture the frontal reference and
sizing (stand facing the mirror), `m` cycle mask mode (clothes / skeleton / intersect), `d` toggle
overlay, `f` toggle projector fullscreen, `g` toggle gallery panel, `r` reload `calibration.yaml`
(projector recalibration itself is `scripts/calibrate.py`), `q` quit.

Gestures: swipe an arm sideways at chest height (more than 1.2 shoulder widths within 0.4 s) to
change outfit; raise both hands above the head for 1 s to toggle the gallery. Gestures are ignored
while facing sideways or away. `gestures.person_right_is_next` flips the direction mapping if the
mirror makes it feel wrong.

`--demo`: fullscreen projector, outfits auto-cycle every 8 s after 20 s without a swipe, and tracking
loss fades to black and back in.

### 60-second demo script

1. (0-10 s) "Plain white tee and pants. The projector paints clothes on them, tracked at 30 fps."
   Person steps on the mark; outfit appears. Point at the mirror.
2. (10-25 s) "Swipe to change." Two swipes. Gallery highlight moves on the wall.
3. (25-40 s) "Turn around." Back design appears; side view narrows. "The face is always black: we
   never put projector light in anyone's eyes."
4. (40-50 s) "It also measures you." Point at the size panel: shoulder, chest, size estimate.
5. (50-60 s) "All local, one laptop, any camera, any projector: calibration takes one minute."

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Mask flickers at the edges | raise `tracker.mask.temporal_alpha` smoothing (lower value = more smoothing), raise `erode_px`; press `m` to compare modes |
| Mask disappears when the garment appears | expected degradation: `intersect` mode auto-falls back to the skeleton mask on big area jumps (`area_jump_fallback`); set `mask.mode: skeleton` if the room light is low |
| Garment sits beside the body | calibration: re-run `calibrate.py`; check camera resolution matches `calibration.yaml` (it is rescaled, but a moved camera is not); camera must not move after calibration |
| Lag / garment trails the body | check FPS in the debug window; raise `tracker.seg_every_n_frames`; raise `smoothing.predict_ms` (50-80), lower `min_cutoff` only if jitter is fine |
| Jitter | lower `smoothing.min_cutoff` (0.5), lower `beta` |
| Wrong camera | `capture.source` accepts `name:MacBook` / `name:iPhone` (AVFoundation device name) or `auto`; numeric indices shuffle when the iPhone comes and goes |
| Garment sits low / high on everyone | re-run `calibrate.py --refine` on the mark; then `[` / `]` live nudge; make sure the camera has not moved |
| Garment too big / flickers between shapes | the mask fell back to skeleton polygons: check `(fallback)` in the debug text; raise `mask.area_jump_fallback` or `area_jump_frames` |
| Window on the wrong display | `--projector-x 1512 --projector-y 0` (origin of the second display in points), `f` toggles fullscreen |
| Washed-out colors | dim the lights over the beam, raise `compositor.brightness` up to 1.0 and `garments.saturation_boost`, regenerate garments; avoid pastel/black garments |
| Light on the face | should be impossible: `compositor.compose` blacks out `head_box` last. If the box looks small, raise `tracker.head.top_margin_frac` / `side_margin_frac` |
| `not authorized to capture video` | grant Camera permission to the terminal app |
| Projector goes black mid-demo, log says "no new camera frame" | the camera stalled (Continuity Camera sleep, cable); the blackout is deliberate. Wake the phone / replug; the app exits after `capture.stall_exit_s` |
| `SIMULATED calibration ... Refusing` | you ran `calibrate.py --simulate` and pointed `--calibration` at it; run the live calibration instead |
| MediaPipe timestamp error | frames must have increasing timestamps; `Tracker` guarantees this; if you wrote a new source, use `time.perf_counter()` |

## 8. Measured performance (this MacBook Pro, Apple Silicon, 1280x720)

See `PROGRESS.md` for the raw logs. Summary:

| Stage | ms / frame |
|---|---|
| MediaPipe pose (full, with person mask) | 6.4 |
| MediaPipe selfie segmenter (every 2nd frame) | 17.4 |
| rtmlib RTMPose-s lightweight (alt backend) | 16 |
| rtmlib balanced (too slow, not used) | 71 |
| 4K decode + resize (capture thread) | 1.5 + 6 |
| Mask post-processing (box-filter morphology) | ~3 |
| Render + compose + projector warp | ~7.5 |
| **Main-loop step, live-equivalent (threaded capture, segmenter in worker thread)** | **25.5 mean / 33.6 p95 -> ~39 fps capacity** |
| Offline review run (sync 4K decode + review video encode included) | 33 -> 30 fps |
| 2-minute soak, looping white clip, final code | 30.1 -> 33.2 fps, 3986 frames, no crash |

The test clips are 25 fps, so an offline run can never show more than 25 fps achieved; the
"capacity" number is what the loop can sustain with a 30/60 fps camera.

## 9. 3D path: iPhone 17 Pro LiDAR (depth-correct projection)

The 2D path maps camera pixels to projector pixels with one homography, exact at one depth only. The 3D path
(`--sensor iphone`) streams LiDAR depth, person segmentation and Vision body joints from the phone and renders
each camera pixel through its true 3D position (`fitcast/render3d.py`), after a full projector calibration
(`scripts/calibrate3d.py`). Spec: `docs/BUILD_SPEC_3D.md`.

### Setup (once)
1. Phone: iPhone 17 Pro, USB cable, unlocked, "Trust This Computer", Developer Mode on. Clamp it on top of the
   projector, lens next to the projector lens, LANDSCAPE with the USB-C port on the right. Disable auto-lock.
2. Mac: `scripts/ios_deploy.sh` builds `ios/FitCastSensor`, installs and launches it (signing: your Apple ID team).
3. In the app, tab "Capability": wait for the 10 s fps measurement; it must say `sceneDepth` present and ~60 fps.
   Tab "Stream": the joint overlay must sit on the body; if not, change the orientation picker (should be `.up`).
4. Mac receiver check: `.venv/bin/python -m fitcast.sensors.fake_phone` is the offline stand-in;
   live: `main.py --sensor iphone --no-projector` shows camera + depth-mask overlay and sensor fps/latency in the status line.

### Calibration (every time the rig moves)
1. Put two boxes or foam boards at the floor mark, at different angles, in front of the wall. Empty beam.
2. `.venv/bin/python scripts/calibrate3d.py --sensor iphone` projects 44 Gray-code patterns + white/black (about
   30 s), lifts every decoded pixel with LiDAR depth and solves the projector intrinsics and pose.
   Expected: reprojection < 1.5 px RMS, tens of thousands of correspondences, `calibration3d.yaml` written.
   Simulated reference: `--simulate` gives 0.41 px / focal 0.013% / 0.003 deg / 0.4 mm.
3. Verification pattern (projected at the end): grid lines must sit on the boxes' edges within ~1 cm.
4. Remove the boxes. `main.py --sensor iphone --demo`. First 2 s: nobody in the beam (depth background is
   learned; key `b` relearns). Then the person steps on the mark.

### Measuring instead of guessing
- `scripts/accuracy_check.py --sensor iphone [--turn] --report`: projected body outline vs depth silhouette in mm
  (targets: still median < 10 mm, p95 < 20 mm; slow turn median < 25 mm).
- `scripts/latency_probe.py --method loopback-camera`: true light-to-light latency; `--method video` for a
  slow-motion recording. `main.py`'s "latency" number is capture-to-display only.

### Demo-day checklist (2 minutes)
1. Projector on, extended display, no keystone. Phone clamped, cable in, unlocked, app running, "Stream" tab.
2. `scripts/calibrate3d.py --sensor iphone` with the boxes (only if the rig moved since last time).
3. Boxes out, beam empty, `main.py --sensor iphone --demo --height <cm>`; wait for "depth background learned".
4. Person on the mark: check collar and hem, swipe once each way, turn once. `k` shows the skeleton on the body
   if anything looks off; `[`/`]` nudges; `b` relearns the background if someone walked through.
5. Fallback if the phone misbehaves: `main.py --sensor webcam` with the 2D calibration (`calibrate.py --auto`, `--refine`).

## Stylist (Gemini, optional)

Ask for an outfit in plain language and the booth dresses you with the closest catalogue pieces, says one sentence and
lists where to buy similar items.

```
echo "AIza..." > secrets/gemini.key            # or export GEMINI_API_KEY=...
.venv/bin/python main.py --stylist
# press v, speak ("interview outfit under 80 dollars"), press v again
# or: echo "black jacket and blue jeans" > out/ask.txt
.venv/bin/python scripts/stylist_cli.py "casual plaid shirt look"      # test without camera / projector
```
Settings live under `stylist:` in `config.yaml` (models, microphone device, speech, message time). The demo itself stays
offline; only this feature talks to the network. Shopping prices are model estimates unless `grounding: true` and the
key has Google Search quota.
