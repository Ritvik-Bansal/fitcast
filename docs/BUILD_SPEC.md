# Build spec: Projection Mapped Virtual Try-On ("FitCast")

(Condensed copy of the build document the project was created from. Reviewers check the code against this.)

## 1. What we are building
Person in all-white clothing stands on a taped floor mark. Camera tracks the body in real time. Projector paints virtual outfits onto the white clothes. Mirror faces the person.
Features, priority order:
1. Live projection mapping: garment locked to the body at 30+ FPS, low jitter/lag.
2. Front, back and side views: back design when turned around, side rendering when sideways.
3. Gesture control: swipe an arm left/right to cycle outfits. No controller.
4. Wall gallery: projector area on the wall beside the person shows a carousel of 5 thumbnails, current highlighted; swiping moves it.
5. Size recommendation: shoulder width, chest width, torso length, inseam in real units -> XS..XXL on the wall panel.
Safety rule overriding everything: never project bright light onto the face/eyes. Black out the head region every frame.

## 2. Hardware
MacBook Pro Apple Silicon macOS 26 (no CUDA). Epson CO-FH02 1920x1080 over HDMI as a second display. iPhone Continuity Camera mounted on the projector (any webcam must work). Python 3.11 venv, pinned requirements. Partially lit room. Check which packages install on Apple Silicon before writing tracking code.

## 3. Architecture
Single process, multithreaded, no web server, no network at runtime.
CaptureThread (latest frame only) -> Tracker (pose, clothes mask, face box, facing, gestures) -> Smoother (One Euro per coordinate + 1-frame velocity prediction) -> Renderer (garment layers warped in CAMERA space, RGBA) -> Compositor (clip to clothes mask, black out head, add wall panel) -> Calibration (warp camera -> projector 1920x1080) -> Output (borderless fullscreen OpenCV window on the projector).
Operator debug window: camera feed with skeleton, mask overlay, FPS, latency estimate, facing, outfit, calibration status.
Repository layout: CLAUDE.md, README.md, requirements.txt, config.yaml (every tunable), main.py, fitcast/{capture,tracker,smoothing,garments,renderer,compositor,calibration,sizing,gestures,display}.py, scripts/{prepare_garments,calibrate,record_clip}.py, assets/{garments,models}, tests/.
Interface contract: TrackResult(ok, joints (33,3), visibility (33,), clothes_mask (H,W) uint8, head_box, facing front|back|side, timestamp); renderer.draw(track, outfit) -> RGBA; compositor.compose(rgba, track, gallery_state) -> BGR; calibration.to_projector(img) -> (1080,1920,3). MediaPipe 33-point landmark order; other backends map to it in tracker.py.

## 4. Module specs
4.1 Capture: background thread holding only the most recent frame, never a queue. Request 1280x720 @30/60, log the actual values. --source camera index or video path; video loops at real-time speed. --list-cameras.
4.2 Tracking: MediaPipe Tasks PoseLandmarker VIDEO mode, full model, downloaded into assets/models by a script. rtmlib backend selectable in config; benchmark both, record FPS in README. Clothes mask: ImageSegmenter selfie_multiclass_256x256 class 4 only, upsample, threshold, small morphological close, 3-5 px erosion. Fallback skeleton mask from pose (thick torso polygon, limb capsules sized from shoulder width); mask_mode clothes|skeleton|intersect, default intersect with automatic fallback to skeleton when the clothes area changes >40% frame to frame. Head box from nose/eyes/ears/shoulders with >=25% shoulder-width margin, down to just above the shoulders, always black. Facing: left shoulder (11) appears RIGHT of right shoulder (12) when facing the camera; ratio = width / reference frontal width (captured at session start, running max); side when ratio < 0.55 else front/back from shoulder order confirmed by nose/eye visibility; 6-frame hysteresis.
4.3 Smoothing: One Euro on every x,y; min_cutoff 1.0, beta 0.05, d_cutoff 1.0 in config. Predict forward by pipeline latency (predict_ms default 50), clamped. Temporal exponential blend of the clothes mask.
4.4 Garments: folder per outfit with meta.yaml, top_front/top_back/sleeve/bottom_front/bottom_back/thumb PNGs. prepare_garments.py: VITON-HD and DressCode import, plain product images (white background removal), back synthesis (mirror + inpaint collar/graphic), sleeve texture sampling, saturation/brightness boost (configurable), low-luminance warning, >=12 procedural demo outfits with zero downloads.
4.5 Renderer: camera space RGBA, per-part warps. Torso quad from shoulders/hips expanded by 12% of shoulder width and raised at the neckline, getPerspectiveTransform + warpPerspective; back uses top_back (left/right swapped in image space). Sleeves: upper arm + forearm quads, short sleeves stop at the elbow. Bottoms: hip-knee and knee-ankle quads per leg + waist band, texture split left/right. Side view: narrowed front garment. Draw order bottoms, torso, sleeves. Feathered edges. All geometry in config.
4.6 Compositor: multiply RGBA by the mask; outside = pure black. Force head box black (not optional). Global brightness/gamma. Wall gallery panel in a configurable camera-space rectangle: 5 thumbnails, current enlarged+outlined, outfit name, size recommendation, ~200 ms animation. Never overlap the person's mask.
4.7 Gestures: wrist relative to shoulder centre normalised by shoulder width over ~0.4 s; swipe = >1.2 shoulder widths horizontally within the window, between hip and head height, velocity above threshold. Mirror-aware direction mapping as a config flag. 1.0 s cooldown. Ignore while facing back/side. Optional raise-both-hands 1 s toggles the gallery. Keyboard fallbacks: arrows outfit, c recalibrate, m mask mode, d debug overlay, f fullscreen, q quit.
4.8 Calibration: Level 1 planar homography: project 4-9 markers, detect on a difference image (blob), manual click correction, findHomography RANSAC, save calibration.yaml with timestamp + reprojection error, operator defines the wall panel rectangle, verification grid within 6 px pass/fail. Simulated mode for unit tests (<1 px). to_projector = warpPerspective(img, H, (1920,1080)). Level 2 (Gray code) optional stretch.
4.9 Sizing: operator enters height; pixels per cm from nose-to-ankle; shoulder, chest (mask width mid-torso), torso, inseam averaged over 60 frames facing front; size chart XS..XXL mens/womens in config; UI states it is an estimate.
4.10 Display: borderless fullscreen OpenCV window on the second display, detect layout, --projector-x/y overrides; separate debug window; show FPS and end-to-end latency.

## 5. Performance
30 FPS or better end to end; segmentation may run every 2nd frame; no heavy per-frame allocations; profile with cProfile if missed and document.

## 6. Process
Milestones with PROGRESS.md checks: 1 scaffold, 2 offline loop on the test clips (annotated videos in out/), 3 full body rendering + facing, 4 gestures + gallery (log detected gestures), 5 calibration tool with simulated mode (<1 px), 6 live hardware (solid colour on the mask first), 7 sizing, 8 polish + --demo (fullscreen, auto-cycle every 8 s after 20 s idle, fade out/in on tracking loss). Tests: homography round trip, warp onto synthetic skeleton, facing classifier incl. hysteresis, gesture detector (fires/ignores drift/cooldown), head blackout always zero, garment prep RGBA + dimensions.

## 8. Rules
Local only. Every tunable in config.yaml. No silent failures. Never project onto the head (test explicitly). Type hints, docstrings, small modules. Do not invent APIs. Commit after each milestone. Before done: full test suite + 2-minute offline run without crashes, report average FPS.

## 10. Test clips
assets/test/white_turn.mp4 (white tee, dark cap, khaki pants, light gray background, full turn with clear side profile) is the primary clip; pink_turn.mp4 confirms colour independence. Both cropped above the knees: do not use them to verify pants/legs/sizing; leg parts must skip cleanly when knees/ankles are not visible. For each clip produce an annotated review video in out/ with skeleton, mask, facing label and composite side by side.
