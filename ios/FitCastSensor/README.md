# FitCastSensor (iOS)

Sensing and streaming only: ARKit LiDAR depth + confidence, person segmentation, Vision 2D body pose lifted to 3D
with LiDAR, downscaled JPEG, streamed as FCS1 (CLAUDE.md "3D sensor path", `fitcast/sensors/protocol.py`) to the Mac.

## Build and run

```
cd ios && xcodegen generate
xcodebuild -project RadAR.xcodeproj -scheme FitCastSensor -destination 'generic/platform=iOS' \
  -configuration Debug build -allowProvisioningUpdates
```
Or open `ios/RadAR.xcodeproj`, scheme FitCastSensor, pick the iPhone, Run. Team 7JCAPWSS7Z, iOS 18+.

1. Mount the phone LANDSCAPE with the USB-C port on the RIGHT, rear camera next to the projector lens.
2. Launch. Allow camera (and local network if you use Wi-Fi). The ARSession starts at once and keeps running
   on both tabs; the screen never auto-locks while the app is open.
3. Capability tab: report of the device checks. After 10 s of frames it adds `measured_fps_10s` and writes
   `Documents/capability_report.json` (Files app -> On My iPhone -> FitCastSensor, or Finder file sharing).
4. Mac: `python main.py --sensor iphone` (usbmux, port 7777), or `sensor.usbmux: false` + `sensor.host` = phone IP.
   The phone is the TCP server; the newest client replaces the previous one.

## Stream tab

Preview (about 12 fps, only while the tab is visible) with the 19 joints drawn in captured-image pixels:
green = joint has a LiDAR 3D position, yellow = 2D only. The overlay uses exactly the numbers sent on the wire,
so if the dots sit on the body, the Mac gets correct `joints2d`.

Status line: ARKit fps, pose fps, depth valid % (finite depth with confidence >= 1), client / listener, MB/s sent,
thermal state, `fx` (watch it stop changing once focus is locked), video format, active frame semantics, delegate
and pack step times (ms), focus state + method, dropped frames (capture slots full / network replaced a pending
frame), pings answered, last error.

Controls:
- **Orientation** (up / down / left / right): image orientation passed to Vision. `up` is correct for the
  landscape, USB-C-right mount. Other values are un-rotated back to captured-image pixels before sending; pick
  the one where the overlay lands on the body if the mount changes.
- **RGB every N frames** (1..4): the `rgb` JPEG blob (960 px wide, same aspect as the captured image, quality
  0.6) is sent on every Nth frame; depth, confidence, segmentation and joints go every frame.
- **Autofocus (off = locked)**: focus changes move the intrinsics. Default: continuous AF for 3 s after the first
  frame, then locked. Lock uses `AVCaptureDevice.focusMode = .locked` on ARKit's configurable primary camera
  (keeps the current lens position); if ARKit exposes no device, it re-runs with `isAutoFocusEnabled = false`.
  The method shows in the status line and in the report (`focus_lock`).
- **Person segmentation**: on = `[.sceneDepth, .personSegmentationWithDepth]` and the `seg` blob; off = re-runs
  with `.sceneDepth` only and omits `seg` (less phone compute).

## Pipeline (60 fps budget)

- `session(_:didUpdate:)` on queue `ar.delegate`: copies depth (Float32 -> Float16 with vImage), confidence,
  segmentation and intrinsics into one of 3 reusable slots, and does one synchronous VTPixelTransferSession
  downscale of `capturedImage` into a pooled 420f IOSurface buffer (only when RGB is due, Vision is idle or the
  preview is on). No ARFrame or ARKit pixel buffer leaves the callback. All slots busy = frame dropped, not queued.
- Queue `pack`: VideoToolbox JPEG (hardware, RealTime), builds the FCS1 message, hands it to the server.
- Vision (`DetectHumanBodyPoseRequest`) runs on the downscaled copy, one request in flight, frames skipped
  while busy; the largest-bounding-box person wins. Its joints are lifted with that same frame's depth snapshot
  (5x5 median, confidence 2 preferred, >= 1 accepted) and tagged `pose_ts` = that ARFrame's timestamp. Every frame
  carries the newest pose.
- Network queue: `NWListener` TCP :7777, no-delay; at most one frame waits behind the send in flight (a newer
  frame replaces it). Pings are echoed at once with `phone_ts` = CACurrentMediaTime(); an info message (the
  capability report + `device`, `fps`, `img_w`, `img_h`) is sent on connect.

## Wire fields

Frame JSON: `ts`, `send_ts`, `img_w`, `img_h`, `K` [fx, fy, cx, cy] (captured-image px), `depth_w`, `depth_h`
(0 when no depth), `rgb_w`, `rgb_h`, `joint_names` (19 names, order below), `joints2d` [[x, y, conf]],
`joints3d` [[X, Y, Z]] ARKit camera coords (+x right, +y up, -z forward), `seq`, `pose_ts` (null before the
first pose), `fps`, `pose_fps`, `depth_valid_frac`, `thermal_state`, `focus_locked`, `orientation`, `dropped`.
Missing joints are `[null, null, 0]` / `[null, null, null]` (protocol.py reads null as NaN).
Joint order: nose, left_eye, right_eye, left_ear, right_ear, neck, left_shoulder, right_shoulder, left_elbow,
right_elbow, left_wrist, right_wrist, root, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
Blobs (payload order): `depth` float16 [h, w] m, `conf` uint8 [h, w] 0..2, `seg` uint8 [h, w] 0/255 (optional),
`rgb` jpeg (optional).

## Byte-level check (no device needed)

`Wire.swift`, `FrameMessage.swift`, `Joints.swift` and `Util.swift` are Foundation-only. Compile them with a
small `main.swift` using `swiftc` on the Mac, write a few messages to a file and decode it with
`fitcast.sensors.protocol.FrameReader` + `fitcast.sensors.iphone.map_vision_joints`.

## Files

`FitCastSensorApp.swift` (app + main-actor model), `ContentView.swift` (tabs, preview overlay),
`SensorController.swift` (ARSession, semantics choice, focus, delegate), `FramePacker.swift` (slot pool, copies,
downscaler, JPEG, preview), `PoseEstimator.swift` (Vision + lift), `Joints.swift` (names, orientation, lift),
`Capability.swift` (report), `Transport.swift` (TCP server), `FrameMessage.swift` + `Wire.swift` (FCS1), `Util.swift`.
