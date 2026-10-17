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
- `--demo`: fullscreen projector, auto-cycle after `demo.idle_s` without a gesture every `demo.auto_cycle_s`, tracking loss fades the last good frame to black over `compositor.fade_ms` and fades back in (unit test `test_fade_on_tracking_loss`). Auto-cycle verified headlessly with short timers (see below).

## Performance (2026-10-17, this MacBook Pro)
- Live-equivalent step (threaded capture, no recording, overlay off), white clip, 225 frames: mean 25.5 ms, p50 25.1, p95 33.6 -> 39 fps capacity. Breakdown: pose 6.5 ms, segmenter 19 ms every 2nd frame in a worker thread (hidden), mask post 3 ms, render+compose+warp 7.5 ms.
- Offline review run (sync 4K decode + INTER_AREA resize 8 ms + mp4 encode): 33 ms/frame -> 30 fps.
- cProfile before optimisation: ellipse `cv2.dilate` x3 = 8.4 ms, boolean-index tint 5 ms, float compose 4 ms; replaced by box-filter morphology, cv2.copyTo tint, cv2.multiply.
