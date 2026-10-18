# Progress log

Each milestone: what was run, what was observed.

## M0 Environment (2026-10-17)
- `uv venv --python 3.11 .venv`; installed mediapipe 1.1.1, opencv-contrib-python 5.0.0.93, numpy 2.4.6, onnxruntime 1.31.0 (CoreML EP available), pyyaml, rtmlib 0.0.16. All import.
- Clips: both 4096x2160 @ 25 fps; white_turn 586 frames / 23.4 s, pink_turn 310 frames / 12.4 s.
- MediaPipe on 1280x720: pose_full 6.0 ms/frame (6.4 ms with person mask), pose_lite 5.0 ms, selfie_multiclass 17.4 ms. rtmlib lightweight 16 ms, balanced 71 ms.
- 4K decode 1.0-1.5 ms; resize INTER_AREA 6 ms, INTER_LINEAR 0.4 ms.
- Camera: this terminal process is NOT authorized for the camera (`OpenCV: not authorized to capture video`). Hardware milestones need System Settings > Privacy & Security > Camera granted to the terminal app.
- Only one display attached (built-in 1512x982 pt @2x); projector fullscreen placement cannot be verified in this session.

## M1 Scaffold + core modules (2026-10-17)
- Written: CLAUDE.md (contract), fitcast/types.py, config.yaml, requirements*.txt, scripts/download_models.py (all three MediaPipe models + two rtmlib ONNX verified downloading, HTTP 200), .claude/agents/{researcher,builder,reviewer}.md.
- Core modules by lead: capture (thread, latest-frame-only, sync mode for offline runs), smoothing (vectorised One Euro + prediction + mask EMA), geometry, tracker (MediaPipe + rtmlib backends, facing, masks, head box), renderer, compositor, display, main.
- Builders (parallel): gestures (+16 tests), sizing (+6 tests); garments and calibration in flight.
- Ran tracker standalone on both clips (`scratch run_tracker.py`): 22 ms/frame mean (pose 6 + seg 17/2 + mask), facing run-lengths white: front164 side133 back126 side78 front85; pink: front90 side54 back65 side49 front52 — no flicker. Head box covers the cap in every sampled frame. Background "clothes" blobs removed by the person gate.
- Display smoke test: two Cocoa windows open, fullscreen property toggles 0 -> 1.
- `pytest -q`: 44 passed (facing, renderer, compositor incl. head blackout, smoothing, gestures, sizing).

## M2 Offline loop (2026-10-17)
- `main.py --source assets/test/white_turn.mp4 --no-window --record out/white_review.mp4` and the same for pink_turn: review videos (debug | composite, 2560x720) written to `out/`.
- Observed on sampled frames: skeleton, mask tint, head box (covers the cap), red tee warped onto the torso + sleeves in camera space, gallery panel with 5 thumbs and the size text on the left strip.
- Fixed on the way: headless runs looped forever (now single pass unless `--loop`); boolean-index mask tint and float compose were 9 ms (now cv2 ops).

## M3 Full body rendering (2026-10-17)
- Front/side/back on both clips. Facing run-lengths (white): front164 side133 back126 side77 front86; (pink): front90 side54 back65 side49 front52. No flicker.
- Back view uses top_back (pink clip: graphic tee shows the star+FC on the front, plain purple on the back). Winding handled by mapping image-left shoulder to texture TL for both views.
- Side view: torso quad widened to `side_min_width_frac` 0.85 of the reference shoulder width and the mask gets a (person-mask ∩ torso-polygon) fill because the selfie segmenter drops chunks of the shirt side on white-on-gray. Before the fill the composite had black wedges in side frames; after it the side frames are solid.
- Legs: knees/ankles are outside the frame on both clips -> visibility forced to 0 by the frame-margin rule -> leg quads skipped (unit test `test_legs_skipped_when_not_visible`), waist band drawn under the hem.
- Mask fallback bug found with the async segmenter (first result looked like a 400% jump, fallback stuck forever); fixed, fallback now fires on 15 of 586 white frames.

## M4 Gestures + gallery (2026-10-17)
- Gesture detector: 16 unit tests (swipe fires / slow drift ignored / cooldown / back-side ignored / above-head ignored / mirror flags / raise-hands toggle). Every event is logged with timestamp, wrist, travel and velocity; `main.py` prints the gesture log at exit.
- No swipe clip exists in the repo; on both turning clips the detector fired zero events (no false positives during a full turn with arms swinging). Real swipes still need a recorded clip or the live rig.
- Gallery: 5 thumbs, current enlarged + outlined, 200 ms eased slide, name + size + "Size is an estimate"; pixels on/near the person mask are cleared (`test_gallery_panel_never_overlaps_person`).

## M5 Calibration tool (2026-10-17)
- `scripts/calibrate.py --simulate`: reprojection error 0.085 px RMS over 9 markers, VERIFY PASS (max 0.09 px, tolerance 6 px), recovered H within 0.22 projector px of the truth. 7 unit tests incl. round trip, detection within 1.5 px, rescale across camera sizes, letterbox fallback without calibration.
- Live mode (camera + projector windows, manual click fallback, gallery-rect clicks, verification grid) written but NOT run on hardware in this session.

## M7 Sizing (2026-10-17)
- Chest = narrowest row of the clothes mask in the 0.25-0.60 torso band (below sleeves, above hem) x 2.65 circumference factor. Only squarely frontal frames (shoulder width >= 0.9 x reference) are accumulated.
- White clip with the default 175 cm height: "Size S, Shoulder 35 cm, Chest 92 cm, Torso 56 cm, Inseam --", stable through the clip. Ankles are not visible on either clip so the shoulder->hip fallback scale is used; real validation needs the full-body clip.

## M8 Demo mode (2026-10-17)
- `--demo`: fullscreen projector, auto-cycle after `demo.idle_s` without a gesture every `demo.auto_cycle_s`, tracking loss fades the last good frame to black over `compositor.fade_ms` and fades back in (unit test `test_fade_on_tracking_loss`). Auto-cycle verified headlessly on pink_turn with idle_s=2, auto_cycle_s=1: outfit index advanced at t=1.99, 3.02, 4.02 ... 9.08 s (8 changes in 9.9 s).

## Performance (2026-10-17, this MacBook Pro)
- Live-equivalent step (threaded capture, no recording, overlay off), white clip, 225 frames: mean 25.5 ms, p50 25.1, p95 33.6 -> 39 fps capacity. Breakdown: pose 6.5 ms, segmenter 19 ms every 2nd frame in a worker thread (hidden), mask post 3 ms, render+compose+warp 7.5 ms.
- Offline review run (sync 4K decode + INTER_AREA resize 8 ms + mp4 encode): 33 ms/frame -> 30 fps.
- cProfile before optimisation: ellipse `cv2.dilate` x3 = 8.4 ms, boolean-index tint 5 ms, float compose 4 ms; replaced by box-filter morphology, cv2.copyTo tint, cv2.multiply.

## M6 Live hardware, partial (2026-10-17)
- Camera permission became available during the session. `main.py --list-cameras`: "MacBook Pro Camera", index 0 opens (AVFoundation). iPhone Continuity Camera not present (phone not nearby).
- `main.py --source 0 --no-projector --max-seconds 8`: built-in camera delivers 1280x720 @ 30 fps natively; 226 frames in 8.2 s = 27.6 fps achieved with the debug window on (loop 35 ms: pose 11 ms because the full detector re-runs every frame when nobody is in view). No person in view (camera faced the ceiling), projector output stayed black, no crash.
- Projector + person-on-mark + calibration still NOT exercised: needs the Epson on HDMI as an extended display and someone in frame.
- rtmlib backend end to end: `--backend rtmlib` on white_turn, 150 frames: pose 15.8 ms, loop 42 ms (23.7 fps capacity, offline incl. decode); COCO-17 -> MediaPipe-33 mapping renders the cyan tee + head box correctly (out/white_rtmlib.mp4).

## Soak (2026-10-17)
- `main.py --source assets/test/white_turn.mp4 --no-window --loop --max-seconds 120 --outfit 6`: 3830 frames in 120.0 s = 31.9 fps, ~6.5 loops of the clip (MediaPipe timestamps stay monotonic across loops), no crash, no error log lines. Loop 31.3 ms incl. sync 4K decode; pose 6.4 ms, seg 9.2 ms every 2nd frame (worker thread), render+compose+warp 8.5 ms.

## Review round 1 (2026-10-17)
- Reviewer (fresh-eyes agent) ran pytest, simulated calibration, both clips, and its own head-safety probe over every frame of both clips: 0 head-box violations on 896 frames, 0 lit face/hair pixels, zero light in a 20 px band above/beside the box, >=109 px margin above the topmost face landmark (cap covered). Verdict PASS with 5 MAJORs.
- Fixed all 5 MAJORs: (1) facing reference poisoned by hidden hips -> dual references + jump rule, tests `test_hidden_hips_do_not_poison_the_facing_reference`, `test_side_first_session_recovers...`; (2) `d` key collided with 'next' -> key map fixed + test; (3) SegWorker crash was silent -> logged with traceback, tracker falls back to synchronous segmentation, test; (4) simulated calibration.yaml at the repo root was being loaded -> separate `calibration_sim.yaml`, `simulated: true` marker, refused for cameras (verified live: "Refusing to use it with a real camera"), test; (5) stale camera frame held the last garment on the projector -> blank after 500 ms, exit after 15 s.
- Fixed MINORs: side-state texture by shoulder order, torso fill minus skin class, gallery cleared against the person mask, head box grown by the prediction shift (test), demo idle timer resets on keyboard changes, predict horizon = measured latency, per-frame allocations (accumulateWeighted, threshold into buffers, double-buffered mask/RGB, precomputed kernel, bounded stats), hard-coded tunables moved to config (`taper`, `bottom_knee_frac`, `neck_raise_frac`, `search_window_frac`, `side_width_floor_frac`, class ids, `ref_warmup_frames`, `ref_jump_frac`), CLAUDE.md contract updated (person_mask, now=, size_table=, allow_simulated).
- Not done from the review: Gray-code Level 2 (stretch), inseam is still hip->knee->ankle (labelled as an estimate), mask lag vs. predicted joints is documented not compensated, `_DEFAULTS` dicts in sizing/calibration/gestures remain (config has every key now).
- After fixes: 62 tests pass; white 586 frames 30.8 fps, pink 310 frames 30.3 fps (offline incl. decode); live camera 8 s 29.5 fps; rtmlib 25 fps.

## Final soak on shipped code (2026-10-17, commit after review fixes)
- `main.py --source assets/test/white_turn.mp4 --no-window --loop --max-seconds 120 --outfit 6 --calibration calibration_sim.yaml`: 3986 frames in 120.0 s = 33.2 fps, no crash, no error lines. Loop 30.1 ms incl. sync 4K decode; pose 6.3 ms, seg 9.1 ms every 2nd frame (worker), render+compose+warp 7.8 ms, latency 18 ms.
- Demo auto-cycle re-verified after the idle-timer fix: changes at 1.99, 3.0, 4.01 ... 9.08 s with idle 2 s / cycle 1 s.
- `mp.Image` verified to copy its input (double-buffer handoff is race-free).
- 62 tests pass.

## Live hardware session (2026-10-17 evening, Epson CO-FH02 on HDMI + MacBook camera)
- OpenCV fullscreen landed on the laptop (macOS Cocoa fullscreen = main screen only). Fixed with a borderless NSWindow on the EPSON NSScreen (AppKit), later replaced by a CoreGraphics layer window with a background commit thread (projector commit no longer blocks the loop).
- Camera indices shuffle when the iPhone comes and goes; added `name:MacBook`, `auto` and automatic reopen after 3 s without frames (observed one stall ~8 s after opening, recovered).
- Marker calibration in a lit room: fixed threshold failed; channel-max difference + adaptive threshold + disc-shape filter + one-marker-at-a-time median captures -> 9/9 markers, reprojection 2.31 px RMS, verify max 1.60 px.
- Depth refinement (`calibrate.py --refine`): probe discs projected on the tracked chest/belly through H, offset measured by the camera and folded into H. Pass 1 error (-7, +98) px (the garment sat ~10 cm low, as the team saw), pass 2 6.4 px, pass 3 4.5 px.
- Tracker now runs on the projector-footprint ROI only (bystanders beside the rig had hijacked tracking and one refine run; that run was undone exactly).
- Fit tuning with the person on the mark (runs 5-6): segmenter on a person crop with bilinear confidence masks (no blocky edges), fallback mask clipped to the silhouette and needing 2 consecutive jumps (ended the flicker + "too big" look), hem at the hips, neckline at the neck base, head box bottom at chin + 0.30 SW (0 violations on the clips), combined shoulder/chest sizing -> steady "M / shoulder 42 / chest 93" for the test subject. Gesture cap at 12 sw/s and 2.6 sw travel rejects landmark jumps seen during turns.
- Run 6: 1350 frames / 45 s = 29.8 fps live with projector + debug window + recording, latency ~25 ms.
- Multi-person: pose model returns up to 3 people; the tracker keeps the one whose torso centre is inside the projector footprint polygon (bystanders at the frame edges hijacked tracking and one refine run before this). Pose cost unchanged offline (7.2 ms for 1/2/3), ~13 ms live with two people in view.
- Refine hardened: waits for 10 still frames, probe blob area must be 0.3-3x the drawn disc, chest/belly probes must agree within 30 px, nothing is saved when a pass is rejected. Converged 74 -> 23 -> 7.8 px at the mark (run A), 12 px on a repeat.
- Depth model (two-point refine) implemented, but the subject only moved 10-15% in distance so the slope was noise (-2.4 px/px); cleared. `--depth-point` now requires a >= 25% torso-length change. The base mapping at the mark is what the demo relies on; co-locating the camera with the projector (iPhone on the projector) is the real fix for depth.
- Run 7 (19:04): 1302 frames / 46.8 s = 27.8 fps, 12 swipes, garment visibly on the shirt/trousers in the camera frames; size panel reads XL/shoulder 58 cm for the second subject because `sizing.height_cm` is still the 175 cm default.

## State of play before the 3D upgrade (2026-10-17 19:20)
- 2D system: offline verified; live verified at the mark after `--auto` + `--refine`; every live failure today was mechanical (camera moved 3x, people in the beam during calibration). Skeleton-on-projector overlay (key k) shows the transfer error directly.
- Decision (user): proceed with docs/BUILD_SPEC_3D.md (iPhone 17 Pro LiDAR, full projector calibration, depth-correct GPU render) while keeping `--sensor webcam`.
- Toolchain: Xcode 26.2 + xcodegen 2.44.1 + Swift 6.2.3 present; ios/project.yml has signing team KQ3W39U2QQ and ARKit/camera keys; iPhone 17 Pro visible to devicectl but "unavailable" (needs USB + trust + Developer Mode).
- Plan: researcher verifies ARKit/Vision/usbmux/OpenCV APIs; builders in parallel for (a) GPU depth-mesh projector `fitcast/render3d.py`, (b) depth body mask `fitcast/depthmask.py`, (c) Mac receiver + wire protocol `fitcast/sensors/iphone.py`, (d) `scripts/calibrate3d.py` with a synthetic test rig; iOS app builder after the research; reviewer on hardware data.

## 3D path, software milestones (2026-10-17 evening)
- Research (iOS 26.2 SDK headers, Apple docs, OpenCV binding): recorded in DECISIONS.md. Phone is iPhone 17 Pro on iOS 27.2 beta; Mac Xcode 26.2. Device deploy untested until the phone is on USB.
- `fitcast/render3d.py` DepthMeshProjector (ModernGL, offscreen OpenGL 4.1 on the M4 Pro): planar case matches the analytic homography, occlusion and depth-gap tests pass; 1080p render 4.6 ms (mesh_step 1), 3.6 ms (step 2).
- `fitcast/depthmask.py`: background depth model + 3D box + guided upsampling + seg/skeleton fusion; IoU 0.999 synthetic, projected colour changes the mask by 0.01%; 3.5 ms typical.
- `fitcast/sensors/{protocol,iphone,fake_phone}.py`: FCS1 protocol, usbmux/TCP receiver, Vision->MediaPipe mapping, ping/pong latency; loopback at 60 fps passes.
- `fitcast/calib3d.py` + `scripts/calibrate3d.py`: Gray code (vectorised decoder, 25 ms at 720p), LiDAR lift, calibrateCamera + PnP RANSAC; simulated rig (wall + two boxes, 5 mm depth noise): reprojection 0.41 px, focal 0.013%, rotation 0.003 deg, translation 0.4 mm. A full-matrix SVD in the DLT refit allocated tens of GB (OOM, exit 137) -> replaced by a 12x12 eigen solve (1.3 GB peak).
- `main.py --sensor iphone`: phone joints + depth body mask -> `Tracker.process_external` (same mask/head/facing logic), depth-correct GPU projection when calibration3d.yaml exists, else the 2D homography; key `b` relearns the depth background. Loopback against the fake phone: 400 frames at 42-47 fps with recording, pose cost 0.3 ms.
- iOS app `ios/FitCastSensor` (11 Swift files): ARKit session with semantics fallback chain, 60 fps 4:3 format, Vision one-in-flight pose with orientation picker, 5x5 median LiDAR lift, VTPixelTransfer + VideoToolbox JPEG, NWListener TCP with one-frame queue, ping echo, info message, capability tab with 10 s fps measurement. Device build clean; Swift-emitted bytes decode in the Python reader; orientation un-rotation verified with Vision on macOS (0.00 px). Not yet run on the phone.
- Open: device deploy (iOS 27 beta vs Xcode 26.2), on-device capability numbers, usbmux throughput, real Gray-code capture with the phone, accuracy_check.py, latency measurement.

## 3D path on the device (2026-10-17 20:15-20:25)
- `scripts/ios_deploy.sh`: BUILD SUCCEEDED and install OK on iOS 27.2 with Xcode 26.2; first launch denied until the user lifted a Screen Time restriction and trusted the developer profile.
- Capability report (info message): semantics `sceneDepth+personSegmentationWithDepth`, 1920x1440 @ 60, measured 60 fps, depth/seg 256x192, JPEG hardware encoder, thermal nominal. Full table in DECISIONS.md.
- Receiver over USB: 60.0 fps, ~15.5 MB/s, 0 drops / 2710 frames, RTT 1-4 ms, capture->Mac 41-56 ms. Depth 100% valid, median 3.07 m.
- Vision joints: none detected in the first probe because the phone was hand-held in portrait at arm's length (image sideways); needs the landscape mount at 2-3 m.
- First on-device joint overlay (phone landscape, hand-held at 0.5 m): Vision joints on eyes/nose/ears/shoulders, ARKit segmentation hugs head/hoodie/hand; 13 visible joints, 3D shoulder at z = 0.48 m. Orientation `.up` confirmed. Next: mount on the projector, person at 2-3 m.
- Mounted phone (20:40): joints correct at 3.8 m (23 visible), depth 100% valid, camera sees the whole projection. ARKit delivers 30.8 fps in the darkened room (low-light frame-rate halving; thermal "fair"), capture->Mac 45-77 ms. Focus was unlocked by the operator (fx drifting 891-895): must be locked before calibrate3d. Vision face confidences at distance tripped the weak "back" hint -> `face_vis_back_threshold` set to 0.

## Review round 2 (3D path) and fixes (2026-10-17 ~21:00)
- Reviewer verdict: modules PASS, main.py integration FAIL with two BLOCKERs: (1) tracking loss on the 3D path fell through to the 2D homography with the fading held frame; (2) depth background was only collected while a person was tracked. Both fixed: loss now dims the cached last projector frame (projector space, never the 2D map), background is learned only from frames with nobody tracked and the app says so until it is.
- MAJORs fixed: depth mask now runs at the sensor's depth resolution with K scaled (morphology radii mean depth px again; learn 37 ms instead of 720 ms); head box forced to an opaque surface at body depth before the GPU render (depth-less heads were transparent to wall content); 2D calibration no longer loaded/applied on the phone path when calibration3d exists; projector K_cam from calibration3d with a drift warning; key `r` crash on the sensor source; single-lock `IPhoneSensor.latest()` (image/track/conf/seg from the same frame); NEAREST image sampling in render3d (LINEAR bled light across the head-box edge); planarity gate in calibrate3d (5-95% depth span >= 0.35 m or refuse); moderngl/glcontext/pymobiledevice3 declared; iphone camera space 960x720 (square pixels); default-dict keys moved into config.yaml.
- Not yet done from the review: phone-side person selection by footprint (largest bbox today), `pose_ts` into the prediction horizon, allocation-free `_publish`, duplicate facing/head work in the receiver, live calibrate3d verification measured in px/mm (visual only).
- New test `tests/test_iphone_path.py` drives `App.step` against the fake phone with an empty-scene phase: background ready < 1.6 s, 3D path used, head box black, 0 calls to the 2D mapping on loss.

## 2026-10-17 (evening) — webcam-only self-calibration, live on the Epson + MacBook camera
- Pivot: no LiDAR, no phone. MacBook camera (1920x1080 native, 1.5x centre crop -> 1280x720 camera space) at the projector.
- Self-calibration (file trigger `out/calibrate.now`, key `t`, SIGUSR1, or T-pose): 24 dim discs (head band skipped) -> wall
  homography; 15 discs aimed at the body -> similarity correction; then a Gray-code dense scan (46 patterns at level 170,
  head blacked out through the disc-pass mapping) -> body plane from ~2.5k points (rms 2.2-3.7 px), wall plane, and
  upper/lower body planes split at the hips (`D_upper`/`D_lower`, feathered warp). Green dot mesh shown for 3 s as the check.
- Live numbers (room lights on, 4 m): wall rms 1.1-1.7 px, body rms 2.2-3.7 px, residual median 2-5 px (~0.3-0.7 cm).
- Mask: `silhouette` mode = segmenter non-background minus zoned skin/face/hair, unioned with the skeleton capsules and
  clipped to capsules+0.5 SW; 6-frame hold. Baggy grey sweatpants on a white wall are now fully covered without flicker.
- Head box bottom at the chin (+0.15 SW); search window 0.4 SW. Too-close person filter (sw > 320 px ignored).
- Camera: black-frame stall detection + reopen; stall_exit 600 s. Debug view dumped to `out/live_debug.jpg` every 45 frames.
- Tests: 117+ passing (`tests/test_selfcal.py` covers the two-plane rig and the split warp).
- Open: second-distance T-pose (two-plane depth model) not yet exercised live; gesture reach threshold still generous;
  the dense scan cannot decode clipped white fabric (now excluded) so torso points come mostly from the disc pass + arms.
- Garments: 10 new outfits (14-23) from real Wikimedia Commons photos (tees, madras shirt, overshirts, sweater, flared /
  grey / baggy jeans, track pants, khaki trousers) + procedural hoodie, jersey, cargo, chinos, sweatpants.
  `scripts/fetch_garments.py` (download + cut-out + write), `scripts/procedural_garments.py`; licences in
  `assets/garments/SOURCES.md` (three photos are CC BY-SA: outfits 14, 15, 16, 21 - fine for the demo with attribution).
- Mix-and-match: zone swipes (above the waist = top, below = bottoms), two wall strips (bottoms left, tops right),
  swipes only while standing still and one-handed.
- Stylist (Gemini): `--stylist` or `stylist.enabled`; key in `secrets/gemini.key` (gitignored) or GEMINI_API_KEY.
  Press `v` to talk (again to send; ffmpeg avfoundation mic) or write a question to `out/ask.txt`. Reply = one top +
  one bottom from the catalogue (set live), one spoken sentence (macOS `say`) and 2-4 shopping suggestions shown in
  the message band at the top of the beam. Verified: typed question 5-7 s, spoken question (synthesized speech) 7 s,
  end-to-end on the test clip. Model `gemini-3.8-flash` (2.5 models closed to new keys); search grounding is over
  quota on the free key, so prices are the model's estimates ("Prices approximate" on the wall).
- Stylist v2 (2026-10-18): wake word "Fitbot" via offline Vosk (fitcast/wakeword.py, ffmpeg mic pipe, VAD end of
  utterance), request audio + rough transcript to Gemini (gemini-3.8-flash, free tier; flash-lite fallback), curated
  verified shop links per garment (assets/shop_links.yaml) merged first into every answer, QR of the first link on the
  wall, links saved to out/last_links.txt and optionally sent by iMessage. Verified end to end from a synthesized
  "Fitbot, find me a casual outfit with a hoodie and sweatpants" recording (wake heard, 3.5 s request, answer in 8.5 s).
