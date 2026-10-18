// Device capability checks (BUILD_SPEC_3D.md section 2): frame semantics, body tracking, video formats,
// VideoToolbox encoders, plus what the running session actually delivers (filled in by SensorController).

import ARKit
import AVFoundation
import Foundation
import UIKit
import VideoToolbox

enum Capability {
    /// Semantics sets probed with `supportsFrameSemantics` (exact combined sets; never assigned unchecked).
    static let probes: [(String, ARConfiguration.FrameSemantics)] = [
        ("sceneDepth", .sceneDepth),
        ("smoothedSceneDepth", .smoothedSceneDepth),
        ("personSegmentation", .personSegmentation),
        ("personSegmentationWithDepth", .personSegmentationWithDepth),
        ("bodyDetection", .bodyDetection),
        ("sceneDepth+personSegmentationWithDepth", [.sceneDepth, .personSegmentationWithDepth]),
        ("smoothedSceneDepth+personSegmentationWithDepth", [.smoothedSceneDepth, .personSegmentationWithDepth]),
        ("sceneDepth+personSegmentation", [.sceneDepth, .personSegmentation]),
        ("sceneDepth+bodyDetection", [.sceneDepth, .bodyDetection]),
        ("sceneDepth+personSegmentationWithDepth+bodyDetection", [.sceneDepth, .personSegmentationWithDepth, .bodyDetection]),
    ]

    static func semanticsName(_ s: ARConfiguration.FrameSemantics) -> String {
        var parts: [String] = []
        if s.contains(.sceneDepth) { parts.append("sceneDepth") }
        if s.contains(.smoothedSceneDepth) { parts.append("smoothedSceneDepth") }
        if s.contains(.personSegmentationWithDepth) { parts.append("personSegmentationWithDepth") }
        else if s.contains(.personSegmentation) { parts.append("personSegmentation") }
        if s.contains(.bodyDetection) { parts.append("bodyDetection") }
        return parts.isEmpty ? "none" : parts.joined(separator: "+")
    }

    static func describe(_ f: ARConfiguration.VideoFormat) -> [String: Any] {
        [
            "w": Int(f.imageResolution.width), "h": Int(f.imageResolution.height), "fps": f.framesPerSecond,
            "device_type": f.captureDeviceType.rawValue,
            "position": f.captureDevicePosition == .back ? "back" : (f.captureDevicePosition == .front ? "front" : "unspecified"),
            "hdr": f.isVideoHDRSupported,
            "high_res_capture": f.isRecommendedForHighResolutionFrameCapturing,
        ]
    }

    /// Back wide-angle 4:3 at 60 fps (closest to 1920x1440), else the highest-fps back format, else the default.
    static func pickVideoFormat(_ formats: [ARConfiguration.VideoFormat]) -> ARConfiguration.VideoFormat? {
        let back = formats.filter { $0.captureDevicePosition == .back }
        func is43(_ f: ARConfiguration.VideoFormat) -> Bool { abs(f.imageResolution.width / f.imageResolution.height - 4.0 / 3.0) < 0.01 }
        func dist(_ f: ARConfiguration.VideoFormat) -> Double { abs(Double(f.imageResolution.width) - 1920) }
        let ideal = back.filter { $0.framesPerSecond == 60 && $0.captureDeviceType == .builtInWideAngleCamera && is43($0) }
        if let f = ideal.min(by: { dist($0) < dist($1) }) { return f }
        let pool = back.isEmpty ? formats : back
        return pool.max { ($0.framesPerSecond, is43($0) ? 1 : 0) < ($1.framesPerSecond, is43($1) ? 1 : 0) }
    }

    static func videoEncoders() -> [[String: Any]] {
        var list: CFArray?
        guard VTCopyVideoEncoderList(nil, &list) == noErr, let arr = list as? [[String: Any]] else { return [] }
        return arr.map { e in
            [
                "codec": (e[kVTVideoEncoderList_CodecName as String] as? String) ?? "?",
                "encoder": (e[kVTVideoEncoderList_EncoderName as String] as? String) ?? "?",
                "codec_type": fourCC((e[kVTVideoEncoderList_CodecType as String] as? NSNumber)?.uint32Value ?? 0),
                "hardware": (e[kVTVideoEncoderList_IsHardwareAccelerated as String] as? Bool) ?? false,
            ]
        }
    }

    /// Static part of the report (no session needed).
    static func staticReport() -> [String: Any] {
        var world: [String: Bool] = [:]
        var body: [String: Bool] = [:]
        for (name, s) in probes {
            world[name] = ARWorldTrackingConfiguration.supportsFrameSemantics(s)
            body[name] = ARBodyTrackingConfiguration.isSupported && ARBodyTrackingConfiguration.supportsFrameSemantics(s)
        }
        let worldFormats = ARWorldTrackingConfiguration.supportedVideoFormats
        let bodyFormats = ARBodyTrackingConfiguration.isSupported ? ARBodyTrackingConfiguration.supportedVideoFormats : []
        let encoders = videoEncoders()
        return [
            "device": deviceModelIdentifier(),
            "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "app": "FitCastSensor",
            "protocol_version": Int(Wire.version),
            "world_tracking_supported": ARWorldTrackingConfiguration.isSupported,
            "world_tracking_semantics": world,
            "body_tracking_supported": ARBodyTrackingConfiguration.isSupported,
            "body_tracking_semantics": body,
            "video_formats": worldFormats.map(describe),
            "body_tracking_video_formats": bodyFormats.map(describe),
            "video_encoders": encoders,
            "jpeg_hardware_encoder": encoders.contains { ($0["codec_type"] as? String) == "jpeg" && ($0["hardware"] as? Bool) == true },
            "thermal_state": thermalStateString(),
        ]
    }

    /// Pretty JSON for the UI / Documents file.
    static func prettyJSON(_ report: [String: Any]) -> String {
        let r = sanitizeJSON(report)
        guard JSONSerialization.isValidJSONObject(r),
              let d = try? JSONSerialization.data(withJSONObject: r, options: [.prettyPrinted, .sortedKeys])
        else { return "\(r)" }
        return String(decoding: d, as: UTF8.self)
    }

    /// Writes Documents/capability_report.json; returns the path or an error string.
    @discardableResult
    static func write(_ r: [String: Any]) -> String {
        do {
            let dir = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let url = dir.appendingPathComponent("capability_report.json")
            try Data(prettyJSON(r).utf8).write(to: url, options: .atomic)
            return url.path
        } catch {
            return "write failed: \(error)"
        }
    }
}
