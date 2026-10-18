// Joint naming, orientation un-rotation and LiDAR lift. Foundation only (unit-tested on macOS against the
// Python receiver; see README.md "Byte-level check").

import Foundation

enum Joints {
    /// The 19 Vision joints in wire order, canonical snake_case names (CLAUDE.md "Vision 19 -> MediaPipe 33").
    static let names: [String] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "root", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    static let count = 19

    /// Bones for the on-phone overlay (indices into `names`).
    static let bones: [(Int, Int)] = [
        (5, 12), (6, 7), (13, 14), (6, 8), (8, 10), (7, 9), (9, 11), (13, 15), (15, 17), (14, 16), (16, 18),
        (6, 13), (7, 14), (0, 5),
    ]

    /// Window for the depth median: (2r+1)^2 depth pixels around the joint.
    static let liftRadius = 2
    /// Use only confidence-2 depth pixels when at least this many are in the window; otherwise accept confidence >= 1.
    static let liftMinHighConf = 6
}

/// Image orientation handed to Vision (same raw values as CGImagePropertyOrientation for the four rotations).
enum CaptureOrientation: String, CaseIterable, Identifiable {
    case up, down, left, right
    var id: String { rawValue }

    /// CGImagePropertyOrientation raw value (EXIF): up 1, down 3, left 8, right 6.
    var exif: UInt32 {
        switch self {
        case .up: return 1
        case .down: return 3
        case .left: return 8
        case .right: return 6
        }
    }

    /// True when the upright image is the raw buffer rotated by 90 degrees (width and height swap).
    var swapsAxes: Bool { self == .left || self == .right }

    /// Upright normalized point (upper-left origin, as Vision reports it after `orientation`) -> raw buffer
    /// normalized point (upper-left origin). EXIF semantics: `.right` (6) = the raw image must be rotated 90 deg
    /// clockwise to look upright, `.left` (8) = 90 deg counter-clockwise, `.down` (3) = 180 deg.
    func rawNormalized(fromUpright p: (x: Double, y: Double)) -> (x: Double, y: Double) {
        switch self {
        case .up: return p
        case .down: return (1 - p.x, 1 - p.y)
        case .right: return (p.y, 1 - p.x)    // raw (u, v) -> upright (1 - v, u)
        case .left: return (1 - p.y, p.x)     // raw (u, v) -> upright (v, 1 - u)
        }
    }
}

/// A private copy of one frame's depth (+ confidence) for lifting joints found on that frame.
final class DepthSnapshot: @unchecked Sendable {
    private(set) var width = 0
    private(set) var height = 0
    var depth: [Float] = []           // metres, row-major, tight; 0 / NaN = invalid
    var conf: [UInt8] = []            // ARKit confidence 0..2, same size, or empty when unavailable
    var hasConf = false

    func resize(_ w: Int, _ h: Int) {
        if w != width || h != height || depth.count != w * h {
            width = w; height = h
            depth = [Float](repeating: 0, count: w * h)
            conf = [UInt8](repeating: 0, count: w * h)
        }
    }

    /// Median depth (m) over the window around depth pixel (ix, iy); nil when no valid pixel.
    func medianDepth(ix: Int, iy: Int, scratch: inout [Float]) -> Float? {
        let r = Joints.liftRadius
        scratch.removeAll(keepingCapacity: true)
        var low: [Float] = []
        for y in max(0, iy - r) ... min(height - 1, iy + r) {
            for x in max(0, ix - r) ... min(width - 1, ix + r) {
                let i = y * width + x
                let d = depth[i]
                guard d.isFinite, d > 0 else { continue }
                let c: UInt8 = hasConf ? conf[i] : 2
                if c >= 2 { scratch.append(d) } else if c >= 1 { low.append(d) }
            }
        }
        if scratch.count < Joints.liftMinHighConf { scratch.append(contentsOf: low) }
        guard !scratch.isEmpty else { return nil }
        scratch.sort()
        let n = scratch.count
        return n % 2 == 1 ? scratch[n / 2] : 0.5 * (scratch[n / 2 - 1] + scratch[n / 2])
    }

    /// Captured-image pixel (u, v) -> ARKit camera coords [X, -Y, -Z] (m; +x right, +y up, -z forward), or nil.
    /// K = [fx, fy, cx, cy] in captured-image pixels; depth intrinsics = K scaled by depth/img size (CLAUDE.md).
    func lift(u: Float, v: Float, imgW: Int, imgH: Int, K: [Float], scratch: inout [Float]) -> [Float]? {
        guard width > 0, height > 0, imgW > 0, imgH > 0, K.count == 4, u.isFinite, v.isFinite else { return nil }
        let sx = Float(width) / Float(imgW), sy = Float(height) / Float(imgH)
        let ud = u * sx, vd = v * sy
        let ix = Int(ud.rounded(.down)), iy = Int(vd.rounded(.down))
        guard ix >= 0, iy >= 0, ix < width, iy < height else { return nil }
        guard let d = medianDepth(ix: ix, iy: iy, scratch: &scratch) else { return nil }
        let fx = K[0] * sx, fy = K[1] * sy, cx = K[2] * sx, cy = K[3] * sy
        let X = (ud - cx) * d / fx, Y = (vd - cy) * d / fy
        return [X, -Y, -d]
    }
}

/// One pose estimate: joints in captured-image pixels, tagged with the ARFrame timestamp they came from.
struct PoseSample: Sendable {
    var poseTs: Double
    var joints2d: [[Float]]        // 19 x [x, y, conf]; x/y NaN when missing (written as null)
    var joints3d: [[Float]]        // 19 x [X, Y, Z] ARKit camera coords; NaN when missing
    var imgW: Int
    var imgH: Int

    static func empty(imgW: Int = 0, imgH: Int = 0) -> PoseSample {
        PoseSample(poseTs: .nan,
                   joints2d: Array(repeating: [.nan, .nan, 0], count: Joints.count),
                   joints3d: Array(repeating: [.nan, .nan, .nan], count: Joints.count),
                   imgW: imgW, imgH: imgH)
    }
}
