// Owns the ARSession. The delegate (queue "ar.delegate") only copies: depth F32 -> F16, confidence,
// segmentation, intrinsics, timestamps and one synchronous downscale of capturedImage; then it hands a
// FrameSlot to the serial "pack" queue. No ARFrame (or its pixel buffers) outlives the delegate call.

import ARKit
import AVFoundation
import Foundation
import QuartzCore

/// User-controlled settings, read from the capture queue.
struct StreamSettings {
    var orientation: CaptureOrientation = .up
    var rgbEvery = 1                // send the rgb JPEG every Nth frame (1..4)
    var autofocus = true            // false = focus locked (default: locked 3 s after start)
    var segmentation = true         // person segmentation semantics + `seg` blob
    var previewOn = false           // Stream tab visible
}

/// Numbers for the status line (copied to the UI a few times per second).
struct StreamStatus {
    var arFps = 0.0, poseFps = 0.0, depthValid = 0.0, mbps = 0.0
    var connected = false, listening = false, peer = ""
    var thermal = "nominal", fx: Float = 0, semantics = "", format = ""
    var dropped = 0, netDropped = 0, pings = 0
    var focusLocked = false, focusMethod = ""
    var delegateMs = 0.0, packMs = 0.0
    var message = ""
}

final class SensorController: NSObject, ARSessionDelegate, @unchecked Sendable {
    static let port: UInt16 = 7777
    static let rgbWidth = 960
    static let jpegQuality = 0.6
    static let focusLockDelay = 3.0
    static let fpsMeasureSeconds = 10.0
    static let slotCount = 3

    let session = ARSession()
    let server = SensorServer(port: SensorController.port)
    let pose = PoseEstimator()
    let previewer = Previewer()
    let settings = Locked(StreamSettings())
    let report = Locked<[String: Any]>([:])
    /// Called on the main queue when the 10 s fps measurement finishes (report complete).
    var onReportUpdated: (() -> Void)?

    private let delegateQueue = DispatchQueue(label: "ar.delegate", qos: .userInteractive)
    private let packQueue = DispatchQueue(label: "pack", qos: .userInitiated)
    private let pool = SlotPool(count: SensorController.slotCount)
    private let downscaler = Downscaler(targetWidth: SensorController.rgbWidth)
    private let encoder = JPEGEncoder(quality: SensorController.jpegQuality)
    private let arRate = RateCounter(window: 1.0)

    // capture-queue state
    private var seq = 0
    private var dropped = Locked<Int>(0)
    private var firstFrameSeen = false
    private var measureStart = 0.0
    private var measureCount = 0
    private var measureDone = false
    private var depthMissingFrames = 0
    private var lastDepthValid = Locked<Double>(0)
    private var lastFx = Locked<Float>(0)
    private let delegateMs = Locked<Double>(0)    // EMA of the delegate callback duration
    private let packMs = Locked<Double>(0)        // EMA of the pack step (JPEG + message)

    // main-queue state
    private var config: ARWorldTrackingConfiguration?
    private let modeL = Locked<ARConfiguration.FrameSemantics>([])   // read by the delegate queue too
    private var mode: ARConfiguration.FrameSemantics { modeL.get() }
    private var userTouchedFocus = false
    private var depthOnlyForced = false
    private let focusState = Locked<(locked: Bool, method: String)>((false, ""))
    private let message = Locked<String>("")

    // MARK: lifecycle (main queue)

    func start() {
        var r = Capability.staticReport()
        r["status"] = "measuring delivered fps (\(Int(Self.fpsMeasureSeconds)) s)"
        report.set(r)
        Capability.write(r)
        session.delegate = self
        session.delegateQueue = delegateQueue
        server.infoProvider = { [weak self] in
            guard let self else { return nil }
            var info = self.report.get()
            info["device"] = info["device"] ?? deviceModelIdentifier()
            info["fps"] = self.arRate.rate
            if let f = info["video_format"] as? [String: Any] {
                info["img_w"] = f["w"]
                info["img_h"] = f["h"]
            }
            return FrameMessage.info(report: info)
        }
        server.start()
        runSession(segmentation: settings.get().segmentation)
    }

    /// Restart the 10 s fps measurement (Capability tab button).
    func remeasure() {
        delegateQueue.async { self.measureDone = false; self.measureStart = 0; self.measureCount = 0 }
        report.mutate { $0["status"] = "measuring delivered fps (\(Int(Self.fpsMeasureSeconds)) s)" }
    }

    /// Semantics for the requested mode; every candidate is checked with supportsFrameSemantics on the EXACT set
    /// before it can be assigned (an unsupported set raises an uncatchable ObjC exception).
    private func chooseSemantics(segmentation: Bool, forceDepthOnly: Bool) -> (ARConfiguration.FrameSemantics, String) {
        let W = ARWorldTrackingConfiguration.self
        let preferred: ARConfiguration.FrameSemantics = [.sceneDepth, .personSegmentationWithDepth]
        if forceDepthOnly && W.supportsFrameSemantics(.sceneDepth) {
            return (.sceneDepth, "personSegmentationWithDepth alone delivered no sceneDepth; switched to sceneDepth only")
        }
        if !segmentation && W.supportsFrameSemantics(.sceneDepth) { return (.sceneDepth, "") }
        if W.supportsFrameSemantics(preferred) { return (preferred, "") }
        if W.supportsFrameSemantics(.personSegmentationWithDepth) {
            return (.personSegmentationWithDepth, "combined set unsupported; using personSegmentationWithDepth alone (expecting sceneDepth)")
        }
        if W.supportsFrameSemantics(.sceneDepth) {
            return (.sceneDepth, "person segmentation unsupported with world tracking; sceneDepth only")
        }
        return ([], "no LiDAR scene depth on this device")
    }

    /// Build the configuration and run (main queue). Without reset options tracking continues across re-runs.
    private func runSession(segmentation: Bool, forceDepthOnly: Bool = false) {
        guard ARWorldTrackingConfiguration.isSupported else {
            message.set("ARWorldTrackingConfiguration is not supported on this device")
            return
        }
        if forceDepthOnly { depthOnlyForced = true }
        let (chosen, note) = chooseSemantics(segmentation: segmentation, forceDepthOnly: depthOnlyForced)
        let W = ARWorldTrackingConfiguration.self
        let cfg = ARWorldTrackingConfiguration()
        if let f = Capability.pickVideoFormat(ARWorldTrackingConfiguration.supportedVideoFormats) { cfg.videoFormat = f }
        cfg.isLightEstimationEnabled = false
        cfg.isAutoFocusEnabled = true
        if focusState.get().locked && focusState.get().method == "arkit" { cfg.isAutoFocusEnabled = false }
        if W.supportsFrameSemantics(chosen) { cfg.frameSemantics = chosen }
        config = cfg
        modeL.set(cfg.frameSemantics)
        delegateQueue.async { self.depthMissingFrames = 0 }
        session.run(cfg)
        report.mutate { r in
            r["semantics"] = Capability.semanticsName(cfg.frameSemantics)
            r["semantics_note"] = note
            r["video_format"] = Capability.describe(cfg.videoFormat)
        }
        if !note.isEmpty { message.set(note) }
        if focusState.get().locked && focusState.get().method == "avcapture" {
            // A re-run may reset the capture device: re-apply the lock shortly after.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.applyFocus(locked: true) }
        }
    }

    // MARK: settings (main queue)

    func setSegmentation(_ on: Bool) {
        settings.mutate { $0.segmentation = on }
        if chooseSemantics(segmentation: on, forceDepthOnly: depthOnlyForced).0 != mode { runSession(segmentation: on) }
    }

    func setAutofocus(_ on: Bool, byUser: Bool = true) {
        if byUser { userTouchedFocus = true }
        settings.mutate { $0.autofocus = on }
        applyFocus(locked: !on)
    }

    /// Lock: AVCaptureDevice `.locked` at the current lens position when ARKit exposes the device, else
    /// re-run with `isAutoFocusEnabled = false`. Unlock: continuous AF.
    private func applyFocus(locked: Bool) {
        if let dev = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera {
            do {
                try dev.lockForConfiguration()
                let mode: AVCaptureDevice.FocusMode = locked ? .locked : .continuousAutoFocus
                if dev.isFocusModeSupported(mode) { dev.focusMode = mode }
                dev.unlockForConfiguration()
                focusState.set((locked, "avcapture"))
                report.mutate { $0["focus_lock"] = ["method": "avcapture", "locked": locked, "lens_position": dev.lensPosition] }
                return
            } catch {
                message.set("focus: lockForConfiguration failed (\(error)); using ARKit isAutoFocusEnabled")
            }
        }
        focusState.set((locked, "arkit"))
        report.mutate { $0["focus_lock"] = ["method": "arkit", "locked": locked] }
        if let cfg = config, cfg.isAutoFocusEnabled == locked {
            cfg.isAutoFocusEnabled = !locked
            session.run(cfg)
        }
    }

    // MARK: status (main queue)

    func status() -> StreamStatus {
        var s = StreamStatus()
        s.arFps = arRate.rate
        s.poseFps = pose.rate.rate
        s.depthValid = lastDepthValid.get()
        s.mbps = server.sentBytes.bytesPerSecond / 1e6
        let st = server.state.get()
        s.connected = st.connected; s.listening = st.listening; s.peer = st.peer
        s.thermal = thermalStateString()
        s.fx = lastFx.get()
        s.semantics = Capability.semanticsName(mode)
        if let f = config?.videoFormat {
            s.format = "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height))@\(f.framesPerSecond)"
        }
        s.dropped = dropped.get()
        s.netDropped = server.droppedFrames.get()
        s.pings = server.pings.get()
        let fs = focusState.get()
        s.focusLocked = fs.locked; s.focusMethod = fs.method
        s.delegateMs = delegateMs.get(); s.packMs = packMs.get()
        s.message = [message.get(), st.error, pose.lastError.get()].filter { !$0.isEmpty }.joined(separator: " | ")
        return s
    }

    // MARK: ARSessionDelegate (delegate queue)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = CACurrentMediaTime()
        arRate.tick(now)
        seq += 1
        let cfg = settings.get()
        if !firstFrameSeen { firstFrameSeen = true; firstFrame(frame) }
        measureFps(now)

        guard let slot = pool.acquire() else { dropped.mutate { $0 += 1 }; return }
        var m = FrameMeta()
        m.ts = frame.timestamp
        m.seq = seq
        let res = frame.camera.imageResolution
        m.imgW = Int(res.width); m.imgH = Int(res.height)
        let K = frame.camera.intrinsics
        m.K = [K[0][0], K[1][1], K[2][0], K[2][1]]
        lastFx.set(m.K[0])
        m.orientation = cfg.orientation.rawValue
        m.thermal = thermalStateString()
        m.focusLocked = !cfg.autofocus

        // Vision: claim the estimator now so its depth snapshot can be filled from this frame.
        let visionThisFrame = pose.tryBegin()
        if let sd = frame.sceneDepth {
            depthMissingFrames = 0
            if let (w, h) = CaptureCopy.depth(sd.depthMap, into: &slot.depth16, f32: visionThisFrame ? pose.snapshot : nil) {
                slot.hasDepth = true
                m.depthW = w; m.depthH = h
                if let cm = sd.confidenceMap, let (cw, ch) = CaptureCopy.plane8(cm, into: &slot.conf), cw == w, ch == h {
                    slot.hasConf = true
                    if visionThisFrame {
                        pose.snapshot.conf.withUnsafeMutableBufferPointer { d in
                            slot.conf.withUnsafeBufferPointer { s in _ = memcpy(d.baseAddress!, s.baseAddress!, w * h) }
                        }
                        pose.snapshot.hasConf = true
                    }
                } else if visionThisFrame { pose.snapshot.hasConf = false }
            } else if visionThisFrame { pose.snapshot.resize(0, 0) }
        } else {
            if visionThisFrame { pose.snapshot.resize(0, 0) }
            depthMissingFrames += 1
            if depthMissingFrames == 30, mode.contains(.personSegmentationWithDepth), !mode.contains(.sceneDepth) {
                DispatchQueue.main.async { self.runSession(segmentation: false, forceDepthOnly: true) }
            }
        }
        if cfg.segmentation, let seg = frame.segmentationBuffer, let (w, h) = CaptureCopy.plane8(seg, into: &slot.seg) {
            slot.hasSeg = true
            m.segW = w; m.segH = h
        }

        let rgbDue = (seq % max(1, cfg.rgbEvery)) == 0
        var small: CVPixelBuffer?
        if rgbDue || visionThisFrame || cfg.previewOn {
            small = downscaler.downscale(frame.capturedImage)
        }
        m.rgbW = downscaler.size.w; m.rgbH = downscaler.size.h
        if visionThisFrame {
            if let small {
                pose.run(buffer: small, orientation: cfg.orientation, ts: frame.timestamp, imgW: m.imgW, imgH: m.imgH, K: m.K)
            } else {
                pose.cancelClaim()
            }
        }
        slot.rgbBuffer = rgbDue ? small : nil
        slot.viewBuffer = cfg.previewOn ? small : nil
        slot.meta = m
        packQueue.async { self.pack(slot) }
        let dt = (CACurrentMediaTime() - now) * 1e3
        delegateMs.mutate { $0 = $0 == 0 ? dt : 0.95 * $0 + 0.05 * dt }
    }

    private func firstFrame(_ frame: ARFrame) {
        var probe: [String: Any] = [
            "captured_image": ["w": CVPixelBufferGetWidth(frame.capturedImage), "h": CVPixelBufferGetHeight(frame.capturedImage),
                               "format": fourCC(CVPixelBufferGetPixelFormatType(frame.capturedImage))],
            "image_resolution": [Int(frame.camera.imageResolution.width), Int(frame.camera.imageResolution.height)],
            "intrinsics_fx_fy_cx_cy": [frame.camera.intrinsics[0][0], frame.camera.intrinsics[1][1],
                                       frame.camera.intrinsics[2][0], frame.camera.intrinsics[2][1]].map { Double($0) },
            "scene_depth_present": frame.sceneDepth != nil,
            "timestamp_minus_CACurrentMediaTime_s": frame.timestamp - CACurrentMediaTime(),
        ]
        if let sd = frame.sceneDepth {
            probe["depth"] = ["w": CVPixelBufferGetWidth(sd.depthMap), "h": CVPixelBufferGetHeight(sd.depthMap),
                              "format": fourCC(CVPixelBufferGetPixelFormatType(sd.depthMap))]
            if let c = sd.confidenceMap {
                probe["confidence"] = ["w": CVPixelBufferGetWidth(c), "h": CVPixelBufferGetHeight(c),
                                       "format": fourCC(CVPixelBufferGetPixelFormatType(c))]
            }
        }
        if let s = frame.segmentationBuffer {
            probe["segmentation"] = ["w": CVPixelBufferGetWidth(s), "h": CVPixelBufferGetHeight(s),
                                     "format": fourCC(CVPixelBufferGetPixelFormatType(s))]
        }
        report.mutate { $0["first_frame"] = probe }
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.focusLockDelay) {
            if !self.userTouchedFocus { self.setAutofocus(false, byUser: false) }
            self.onReportUpdated?()
        }
    }

    private func measureFps(_ now: Double) {
        guard !measureDone else { return }
        if measureStart == 0 { measureStart = now; measureCount = 0; return }
        measureCount += 1
        let span = now - measureStart
        if span >= Self.fpsMeasureSeconds {
            measureDone = true
            let fps = Double(measureCount) / span
            report.mutate { r in
                r["measured_fps_10s"] = (fps * 100).rounded() / 100
                r["measured_frames"] = measureCount
                r["status"] = "complete"
                r["thermal_state"] = thermalStateString()
            }
            DispatchQueue.main.async {
                Capability.write(self.report.get())
                self.onReportUpdated?()
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        message.set("ARSession failed: \(error.localizedDescription); restarting in 1 s")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.runSession(segmentation: self.settings.get().segmentation) }
    }

    func sessionWasInterrupted(_ session: ARSession) { message.set("ARSession interrupted") }
    func sessionInterruptionEnded(_ session: ARSession) { message.set("") }

    // MARK: pack queue

    private func pack(_ slot: FrameSlot) {
        let t0 = CACurrentMediaTime()
        defer {
            pool.release(slot)
            let dt = (CACurrentMediaTime() - t0) * 1e3
            packMs.mutate { $0 = $0 == 0 ? dt : 0.95 * $0 + 0.05 * dt }
        }
        var m = slot.meta
        if let v = slot.viewBuffer { previewer.offer(v) }
        m.fps = arRate.rate
        m.poseFps = pose.rate.rate
        m.depthValidFrac = slot.hasDepth ? CaptureCopy.validFraction(depth16: slot.depth16, conf: slot.hasConf ? slot.conf : nil) : 0
        lastDepthValid.set(m.depthValidFrac)
        m.droppedFrames = dropped.get() + server.droppedFrames.get()
        if !slot.hasDepth { m.depthW = 0; m.depthH = 0 }
        if !slot.hasSeg { m.segW = 0; m.segH = 0 }
        guard server.isConnected else { return }
        var jpeg: Data?
        if let rgb = slot.rgbBuffer { jpeg = encoder.encode(rgb) }
        let sample = pose.current
        m.sendTs = CACurrentMediaTime()
        let msg: Data? = slot.depth16.withUnsafeBytes { d in
            slot.conf.withUnsafeBytes { c in
                slot.seg.withUnsafeBytes { s in
                    if let jpeg {
                        return jpeg.withUnsafeBytes { r in
                            FrameMessage.build(meta: m, pose: sample, depth16: slot.hasDepth ? d : nil,
                                               conf: slot.hasConf ? c : nil, seg: slot.hasSeg ? s : nil, jpeg: r)
                        }
                    }
                    return FrameMessage.build(meta: m, pose: sample, depth16: slot.hasDepth ? d : nil,
                                              conf: slot.hasConf ? c : nil, seg: slot.hasSeg ? s : nil, jpeg: nil)
                }
            }
        }
        if let msg { server.sendFrame(msg) }
    }
}
