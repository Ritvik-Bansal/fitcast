// Capture-side copies (delegate queue), the downscaler, the JPEG encoder and the pack queue that turns one
// FrameSlot into an FCS1 message. Nothing here keeps a reference to an ARFrame or its pixel buffers.

import Accelerate
import CoreImage
import CoreMedia
import CoreVideo
import Foundation
import QuartzCore
import VideoToolbox

/// Everything the pack queue needs from one ARFrame, copied on the delegate queue. Reused (see SlotPool).
final class FrameSlot: @unchecked Sendable {
    var meta = FrameMeta()
    var depth16: [UInt16] = []
    var conf: [UInt8] = []
    var seg: [UInt8] = []
    var hasDepth = false, hasConf = false, hasSeg = false
    var rgbBuffer: CVPixelBuffer?      // downscaled copy to JPEG-encode this frame (nil = no rgb this frame)
    var viewBuffer: CVPixelBuffer?     // any downscaled copy of this frame (preview)

    func reset() {
        hasDepth = false; hasConf = false; hasSeg = false
        rgbBuffer = nil; viewBuffer = nil
    }
}

/// Fixed set of reusable slots; when all are busy the capture queue drops the frame instead of queueing.
final class SlotPool: @unchecked Sendable {
    private let free: Locked<[FrameSlot]>
    init(count: Int) { free = Locked((0 ..< count).map { _ in FrameSlot() }) }
    func acquire() -> FrameSlot? { free.mutate { $0.popLast() } }
    func release(_ s: FrameSlot) { s.reset(); free.mutate { $0.append(s) } }
}

enum CaptureCopy {
    /// depthMap (DepthFloat32, or DepthFloat16 just in case) -> tight Float16 bits in `out`, and optionally
    /// tight Float32 metres into `f32`. Returns (w, h) or nil for an unexpected pixel format.
    static func depth(_ pb: CVPixelBuffer, into out: inout [UInt16], f32: DepthSnapshot?) -> (Int, Int)? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        guard fmt == kCVPixelFormatType_DepthFloat32 || fmt == kCVPixelFormatType_DepthFloat16 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        if out.count != w * h { out = [UInt16](repeating: 0, count: w * h) }
        f32?.resize(w, h)
        var src = vImage_Buffer(data: base, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: rowBytes)
        if fmt == kCVPixelFormatType_DepthFloat32 {
            out.withUnsafeMutableBytes { o in
                var dst = vImage_Buffer(data: o.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w),
                                        rowBytes: w * 2)
                _ = vImageConvert_PlanarFtoPlanar16F(&src, &dst, vImage_Flags(kvImageNoFlags))
            }
            if let s = f32 {
                s.depth.withUnsafeMutableBytes { o in
                    for y in 0 ..< h { memcpy(o.baseAddress! + y * w * 4, base + y * rowBytes, w * 4) }
                }
            }
        } else {
            out.withUnsafeMutableBytes { o in
                for y in 0 ..< h { memcpy(o.baseAddress! + y * w * 2, base + y * rowBytes, w * 2) }
            }
            if let s = f32 {
                s.depth.withUnsafeMutableBytes { o in
                    var dst = vImage_Buffer(data: o.baseAddress, height: vImagePixelCount(h),
                                            width: vImagePixelCount(w), rowBytes: w * 4)
                    _ = vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
                }
            }
        }
        return (w, h)
    }

    /// One-component 8-bit buffer (confidence, segmentation) -> tight bytes. Returns (w, h) or nil.
    static func plane8(_ pb: CVPixelBuffer, into out: inout [UInt8]) -> (Int, Int)? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard CVPixelBufferGetPlaneCount(pb) == 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb), CVPixelBufferGetBytesPerRow(pb) >= w else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        if out.count != w * h { out = [UInt8](repeating: 0, count: w * h) }
        out.withUnsafeMutableBytes { o in
            if rowBytes == w { memcpy(o.baseAddress!, base, w * h) } else {
                for y in 0 ..< h { memcpy(o.baseAddress! + y * w, base + y * rowBytes, w) }
            }
        }
        return (w, h)
    }

    /// Fraction of depth pixels with a finite positive value (and confidence >= 1 when confidence exists).
    static func validFraction(depth16: [UInt16], conf: [UInt8]?) -> Double {
        guard !depth16.isEmpty else { return 0 }
        var n = 0
        let useConf = conf != nil && conf!.count == depth16.count
        depth16.withUnsafeBufferPointer { d in
            if useConf {
                conf!.withUnsafeBufferPointer { c in
                    for i in 0 ..< d.count {
                        let b = d[i]
                        // positive, non-zero, exponent not all ones (inf / NaN)
                        if b & 0x8000 == 0, b != 0, b & 0x7C00 != 0x7C00, c[i] >= 1 { n += 1 }
                    }
                }
            } else {
                for b in d where b & 0x8000 == 0 && b != 0 && b & 0x7C00 != 0x7C00 { n += 1 }
            }
        }
        return Double(n) / Double(depth16.count)
    }
}

/// capturedImage -> reusable downscaled 420f IOSurface buffers (same aspect ratio, `targetWidth` wide).
final class Downscaler: @unchecked Sendable {
    let targetWidth: Int
    private var session: VTPixelTransferSession?
    private var pool: CVPixelBufferPool?
    private(set) var size = (w: 0, h: 0)
    private var poolKey = (0, 0)
    private let maxBuffers: Int

    init(targetWidth: Int = 960, maxBuffers: Int = 6) {
        self.targetWidth = targetWidth
        self.maxBuffers = maxBuffers
    }

    /// Synchronous scale (delegate queue). nil when the pool is exhausted or the transfer fails.
    func downscale(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let sw = CVPixelBufferGetWidth(src), sh = CVPixelBufferGetHeight(src)
        if poolKey != (sw, sh) { configure(sw, sh) }
        guard let pool, let session else { return nil }
        var out: CVPixelBuffer?
        let aux = [kCVPixelBufferPoolAllocationThresholdKey as String: maxBuffers] as CFDictionary
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pool, aux, &out) == kCVReturnSuccess,
              let dst = out else { return nil }
        guard VTPixelTransferSessionTransferImage(session, from: src, to: dst) == noErr else { return nil }
        return dst
    }

    private func configure(_ sw: Int, _ sh: Int) {
        poolKey = (sw, sh)
        let w = targetWidth
        let h = max(2, Int((Double(targetWidth) * Double(sh) / Double(max(sw, 1)) / 2).rounded()) * 2)
        size = (w, h)
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        let poolAttrs: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 3]
        var p: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, poolAttrs as CFDictionary, attrs as CFDictionary, &p)
        pool = p
        if session == nil {
            var s: VTPixelTransferSession?
            VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &s)
            session = s
        }
    }
}

/// Hardware JPEG through VTCompressionSession (one complete JPEG per frame). Pack queue only.
final class JPEGEncoder {
    let quality: Double
    private var session: VTCompressionSession?
    private var sessionSize = (0, 0)
    private var result: Data?
    private(set) var lastError: OSStatus = noErr
    private var pts: Int64 = 0

    init(quality: Double = 0.6) { self.quality = quality }

    /// Blocks until the JPEG is out (fine on the pack queue; never call from the ARKit delegate).
    func encode(_ pb: CVPixelBuffer) -> Data? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        if session == nil || sessionSize != (w, h) { makeSession(w, h) }
        guard let session else { return nil }
        result = nil
        pts += 1
        let t = CMTime(value: pts, timescale: 60)
        let st = VTCompressionSessionEncodeFrame(session, imageBuffer: pb, presentationTimeStamp: t,
                                                 duration: .invalid, frameProperties: nil, infoFlagsOut: nil) {
            [weak self] status, _, sample in
            guard let self else { return }
            guard status == noErr, let sample, let bb = CMSampleBufferGetDataBuffer(sample) else {
                self.lastError = status
                return
            }
            let n = CMBlockBufferGetDataLength(bb)
            var d = Data(count: n)
            let ok = d.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(bb, atOffset: 0, dataLength: n,
                                                                           destination: $0.baseAddress!) }
            if ok == kCMBlockBufferNoErr { self.result = d }
        }
        guard st == noErr else { lastError = st; return nil }
        VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: t)
        return result
    }

    private func makeSession(_ w: Int, _ h: Int) {
        if let s = session { VTCompressionSessionInvalidate(s) }
        session = nil
        sessionSize = (w, h)
        var s: VTCompressionSession?
        let st = VTCompressionSessionCreate(allocator: nil, width: Int32(w), height: Int32(h),
                                            codecType: kCMVideoCodecType_JPEG, encoderSpecification: nil,
                                            imageBufferAttributes: nil, compressedDataAllocator: nil,
                                            outputCallback: nil, refcon: nil, compressionSessionOut: &s)
        guard st == noErr, let s else { lastError = st; return }
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(s, key: kVTCompressionPropertyKey_Quality, value: NSNumber(value: quality))
        VTCompressionSessionPrepareToEncodeFrames(s)
        session = s
    }
}

/// Preview images for the Stream tab (CIContext on its own queue, at most one in flight).
final class Previewer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "preview")
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let busy = Locked<Bool>(false)
    private var last = 0.0
    let interval: Double
    let image = Locked<CGImage?>(nil)

    init(interval: Double = 1.0 / 12.0) { self.interval = interval }

    func offer(_ pb: CVPixelBuffer) {
        let now = CACurrentMediaTime()
        guard now - last >= interval, busy.mutate({ b -> Bool in if b { return false }; b = true; return true })
        else { return }
        last = now
        queue.async {
            defer { self.busy.set(false) }
            let ci = CIImage(cvPixelBuffer: pb)
            if let cg = self.context.createCGImage(ci, from: ci.extent) { self.image.set(cg) }
        }
    }
}
