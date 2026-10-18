# Upgrade spec: true 3D projection mapping with iPhone 17 Pro LiDAR

(Condensed copy of the upgrade document. Keep `--sensor webcam` working; new path is `--sensor iphone`.)

## 1. Hardware
iPhone 17 Pro (rear LiDAR + wide camera) rigidly mounted on top of the Epson, pointing the same way; recalibrate if bumped. Epson CO-FH02 1920x1080 extended display. Phone on USB (Wi-Fi fallback). Disable Center Stage / Portrait / Studio Light / auto-lock on the phone; disable auto keystone on the Epson after calibration.

## 2. Verify capabilities on the device first
Diagnostic iOS screen printing: `ARWorldTrackingConfiguration.supportsFrameSemantics` for `.sceneDepth`, `.smoothedSceneDepth`, `.personSegmentationWithDepth` and combinations; `ARBodyTrackingConfiguration.isSupported` and its supported semantics; `videoFormats` (resolution, fps; pick 60 fps); measured delivered fps over 10 s. Record in DECISIONS.md.
Preferred: one configuration with LiDAR sceneDepth + person segmentation at 60 fps; 2D joints from Vision (`VNDetectHumanBodyPoseRequest` / `DetectHumanBodyPoseRequest`) on `capturedImage`, lifted to metric 3D with LiDAR depth (median of a small neighbourhood, confidence-weighted, invalid depth ignored). Alternative: `ARBodyTrackingConfiguration` with scene depth if supported; compare on the same recording with numbers.

## 3. iOS sensor app `FitCastSensor` (ios/)
SwiftUI; sensing + streaming only. Per frame: ARFrame timestamp + monotonic send time, camera intrinsics + image resolution, depth map (256x192 Float16 m) + confidence (UInt8), person segmentation mask, 2D joints (captured-image px, confidences) + lifted 3D joints (camera coords, m), downscaled RGB (e.g. 960x720 JPEG/HEVC; not every frame if bandwidth is tight). Transport: USB first (usbmux / iPhone USB network), Wi-Fi WebSocket fallback; binary framing with a version byte; never block the ARKit delegate (copy and release the ARFrame). On-phone overlay: fps, depth validity %, connection status.

## 4. Mac receiver `fitcast/sensors/iphone.py`
Background thread, latest frame only, same TrackResult contract + `depth` (m, camera space), `joints3d` (m), `K_cam`. Vision joints mapped to the 33-point convention (document the mapping).

## 5. Body mask that ignores the projector
Background depth model of the empty scene (median over 2 s). Body = pixels closer than background by a threshold (e.g. 8 cm) inside a 3D box around the mark, morphology-cleaned. Fuse with ARKit person segmentation and the skeleton mask (depth decides membership, RGB masks refine edges). Edge-aware (joint bilateral / guided) upsampling of the depth mask to camera resolution. Then intersect with the clothes region as before. Head blackout stays mandatory and last.

## 6. Full projector-camera calibration with LiDAR (`scripts/calibrate3d.py`)
Non-planar target (boxes/foam boards at angles + wall) at the mark. Project a Gray code sequence (OpenCV `structured_light.GrayCodePattern`, 1920x1080, + white/black refs); capture several frames per pattern with depth; decode projector (col,row) per camera pixel; lift decoded pixels to 3D via LiDAR (high confidence, stable decode; depth->RGB intrinsics scaling); solve `cv2.calibrateCamera` with `CALIB_USE_INTRINSIC_GUESS` (initial f from throw ratio), refine `[R|t]` with iterative `solvePnP`, RANSAC outliers. Target reprojection < 1.5 px; second target configuration + joint solve if high. Verification: project a 3D test pattern (box edges, wall grid from measured depth) and confirm within 1 cm with the camera; save before/after to out/. Save `calibration3d.yaml` with timestamp, errors, settings.

## 7. Depth-correct rendering
Keep the camera-space renderer. Replace the homography warp: X = depth(u,v) K_cam^-1 [u v 1]; p = K_p (R X + t) + distortion. Implement as a GPU mesh render (triangle grid over the depth map, skip triangles across depth discontinuities, camera image as texture, render from the projector viewpoint) with ModernGL or Metal at 60 fps; benchmark. Depth test handles occlusion. Fill tiny holes only, never across real depth gaps.

## 8. Latency compensation
Measure true end-to-end latency (frame counter / flashing patch filmed in slow motion); helper script + documented method. Kalman or One Euro + velocity prediction on 3D joints and per-limb rigid mesh motion. Report static and moving accuracy separately.

## 9. Optional
SMPL-X fit to 3D joints + depth, UV-textured garments from the projector viewpoint; adopt only if it measurably improves fit at acceptable fps.

## 10. Accuracy verification (`scripts/accuracy_check.py`)
Project a thin outline of the predicted body boundary; measure the distance to the depth-based silhouette in mm along the boundary; mean / median / p95, still and slow turn. Targets: still median < 10 mm, p95 < 20 mm; slow movement median < 25 mm.

## 11. Process
1 capability check (stop and report if a core assumption fails); 2 iOS app streaming at 60 fps (fps + latency measured); 3 depth-based mask ignoring projected patterns; 4 3D calibration with verification; 5 depth-correct GPU rendering (solid colour first); 6 latency measurement + prediction; 7 accuracy numbers; 8 README runbook + 2-minute demo-day checklist. Researcher verifies Apple/OpenCV APIs before coding; builders per module; reviewer runs on hardware data. Never claim a number not measured.

## 12. Final report
Capabilities + design chosen; measured sensor fps, Mac render fps, end-to-end latency, calibration reprojection error, static/moving accuracy in mm; what is not done; on-site steps in order with expected numbers.
