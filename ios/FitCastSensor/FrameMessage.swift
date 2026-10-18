// Builds the FCS1 messages the Mac decodes (fitcast/sensors/protocol.py FrameReader, iphone.py). Foundation only.

import Foundation

/// Per-frame scalar fields of the frame JSON (CLAUDE.md "Frame JSON" + a few diagnostics).
struct FrameMeta {
    var ts: Double = 0                 // ARFrame.timestamp (s, host clock)
    var sendTs: Double = 0             // CACurrentMediaTime() when the message was built
    var imgW = 0, imgH = 0             // frame.camera.imageResolution
    var K: [Float] = [0, 0, 0, 0]      // fx, fy, cx, cy in imgW x imgH pixels
    var depthW = 0, depthH = 0         // 0 when no scene depth
    var segW = 0, segH = 0
    var rgbW = 0, rgbH = 0             // size of the rgb JPEG (sent even on frames without one)
    var seq = 0
    var fps: Double = 0                // delivered ARKit fps
    var poseFps: Double = 0
    var depthValidFrac: Double = 0     // finite depth > 0 with confidence >= 1
    var thermal = "nominal"
    var focusLocked = false
    var orientation = "up"
    var droppedFrames = 0
}

/// Replace non-finite numbers with NSNull: JSONSerialization raises an (uncatchable) exception on NaN / inf.
func sanitizeJSON(_ v: Any) -> Any {
    switch v {
    case let d as [String: Any]: return d.mapValues(sanitizeJSON)
    case let a as [Any]: return a.map(sanitizeJSON)
    case let n as NSNumber:
        if CFGetTypeID(n) == CFBooleanGetTypeID() { return n }
        return n.doubleValue.isFinite ? n : NSNull()
    case let x as Double: return x.isFinite ? x : NSNull()
    case let x as Float: return x.isFinite ? x : NSNull()
    default: return v
    }
}

enum FrameMessage {
    /// Frame message (msg_type 1). Blobs in payload order depth, conf, seg, rgb; nil buffers are omitted.
    static func build(meta: FrameMeta, pose: PoseSample,
                      depth16: UnsafeRawBufferPointer?, conf: UnsafeRawBufferPointer?,
                      seg: UnsafeRawBufferPointer?, jpeg: UnsafeRawBufferPointer?) -> Data? {
        var j = JSONWriter(reserve: 3072)
        j.beginObject()
        j.field("ts", meta.ts)
        j.field("send_ts", meta.sendTs)
        j.field("img_w", meta.imgW)
        j.field("img_h", meta.imgH)
        j.key("K"); j.floatArray(meta.K)
        j.field("depth_w", meta.depthW)
        j.field("depth_h", meta.depthH)
        j.field("rgb_w", meta.rgbW)
        j.field("rgb_h", meta.rgbH)
        j.key("joint_names"); j.stringArray(Joints.names)
        j.key("joints2d")
        j.beginArray()
        for row in pose.joints2d {
            j.beginArray(); j.float(row[0]); j.float(row[1]); j.float(row[2].isFinite ? row[2] : 0); j.endArray()
        }
        j.endArray()
        j.key("joints3d")
        j.beginArray()
        for row in pose.joints3d { j.beginArray(); j.float(row[0]); j.float(row[1]); j.float(row[2]); j.endArray() }
        j.endArray()
        j.field("seq", meta.seq)
        j.optionalField("pose_ts", pose.poseTs.isFinite ? pose.poseTs : nil)
        j.field("fps", meta.fps)
        j.field("pose_fps", meta.poseFps)
        j.field("depth_valid_frac", meta.depthValidFrac)
        j.field("thermal_state", meta.thermal)
        j.field("focus_locked", meta.focusLocked)
        j.field("orientation", meta.orientation)
        j.field("dropped", meta.droppedFrames)

        var blobs: [Wire.Blob] = []
        if let d = depth16, meta.depthW > 0 {
            blobs.append(.init(name: "depth", dtype: "float16", shape: [meta.depthH, meta.depthW], bytes: d))
        }
        if let c = conf, meta.depthW > 0 {
            blobs.append(.init(name: "conf", dtype: "uint8", shape: [meta.depthH, meta.depthW], bytes: c))
        }
        if let s = seg, meta.segW > 0 {
            blobs.append(.init(name: "seg", dtype: "uint8", shape: [meta.segH, meta.segW], bytes: s))
        }
        if let r = jpeg, r.count > 0 {
            blobs.append(.init(name: "rgb", dtype: "jpeg", shape: [r.count], bytes: r))
        }
        return Wire.encode(msgType: Wire.msgFrame, jsonBody: &j, blobs: blobs)
    }

    /// Info message (msg_type 2): the capability report plus the fake phone's keys. Drops the bulky
    /// format/encoder lists if the JSON would not fit the u16 length field.
    static func info(report: [String: Any]) -> Data? {
        var r = (sanitizeJSON(report) as? [String: Any]) ?? [:]
        r["blobs"] = [Any]()
        for dropKey in [nil, "video_formats", "body_tracking_video_formats", "video_encoders"] as [String?] {
            if let k = dropKey { r[k] = "omitted (info message size limit)" }
            guard JSONSerialization.isValidJSONObject(r),
                  let js = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]) else { return nil }
            if js.count <= Wire.maxJSON { return Wire.encodeRaw(msgType: Wire.msgInfo, json: js) }
        }
        return nil
    }

    /// Ping echo (msg_type 3): the Mac's JSON ({"mac_ts", "id"}) with every key kept plus `phone_ts`
    /// (CACurrentMediaTime, the clock of ts / send_ts). `phone_t` is added as an alias.
    static func pong(request json: Data, phoneTs: Double) -> Data? {
        var obj = ((try? JSONSerialization.jsonObject(with: json)) as? [String: Any]) ?? [:]
        obj.removeValue(forKey: "blobs")
        obj["phone_ts"] = phoneTs
        obj["phone_t"] = phoneTs
        obj["blobs"] = [Any]()
        let clean = (sanitizeJSON(obj) as? [String: Any]) ?? [:]
        guard JSONSerialization.isValidJSONObject(clean),
              let js = try? JSONSerialization.data(withJSONObject: clean) else { return nil }
        return Wire.encodeRaw(msgType: Wire.msgPing, json: js)
    }
}
