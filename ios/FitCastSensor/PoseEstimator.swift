// 2D body pose with the iOS 18 Vision API on a private downscaled copy of the captured image, lifted to metric
// 3D with that frame's LiDAR depth. One request in flight; callers drop frames while `isBusy`.

import CoreVideo
import Foundation
import ImageIO
import QuartzCore
import Vision

final class PoseEstimator: @unchecked Sendable {
    private let busy = Locked<Bool>(false)
    private let latest = Locked<PoseSample>(PoseSample.empty())
    let rate = RateCounter(window: 1.0)
    let lastError = Locked<String>("")
    /// Depth of the frame being analysed (written by the capture queue only while not busy).
    let snapshot = DepthSnapshot()
    private let request = DetectHumanBodyPoseRequest()

    /// Map Vision's joint names to wire order once.
    private static let order: [HumanBodyPoseObservation.JointName] = [
        .nose, .leftEye, .rightEye, .leftEar, .rightEar, .neck, .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow, .leftWrist, .rightWrist, .root, .leftHip, .rightHip,
        .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
    ]

    var isBusy: Bool { busy.get() }
    var current: PoseSample { latest.get() }

    /// Claim the estimator (capture queue). Returns false when a request is still running.
    func tryBegin() -> Bool { busy.mutate { b in if b { return false }; b = true; return true } }

    /// Give back a claim from `tryBegin` without running (no image this frame).
    func cancelClaim() { busy.set(false) }

    /// Run on `buffer` (a private copy, same field of view as the captured image). `snapshot` must already hold
    /// this frame's depth. `imgW/imgH/K` describe the captured image; `ts` is the ARFrame timestamp.
    func run(buffer: CVPixelBuffer, orientation: CaptureOrientation, ts: Double, imgW: Int, imgH: Int, K: [Float]) {
        let req = request
        let exif = CGImagePropertyOrientation(rawValue: orientation.exif) ?? .up
        Task.detached(priority: .userInitiated) { [self] in
            defer { self.busy.set(false) }
            do {
                let obs = try await req.perform(on: buffer, orientation: exif)
                let sample = self.convert(obs, orientation: orientation, ts: ts, imgW: imgW, imgH: imgH, K: K)
                self.latest.set(sample)
                self.rate.tick()
            } catch {
                self.lastError.set("vision: \(error)")
            }
        }
    }

    private func convert(_ obs: [HumanBodyPoseObservation], orientation: CaptureOrientation, ts: Double,
                         imgW: Int, imgH: Int, K: [Float]) -> PoseSample {
        var s = PoseSample.empty(imgW: imgW, imgH: imgH)
        s.poseTs = ts
        // Largest joint bounding box (normalized upright coordinates) wins.
        var best: [HumanBodyPoseObservation.JointName: Joint]?
        var bestArea = -1.0
        for o in obs {
            let joints = o.allJoints()
            var x0 = 1.0, y0 = 1.0, x1 = 0.0, y1 = 0.0, n = 0
            for (_, j) in joints where j.confidence > 0 {
                x0 = min(x0, j.location.x); x1 = max(x1, j.location.x)
                y0 = min(y0, j.location.y); y1 = max(y1, j.location.y)
                n += 1
            }
            let area = n >= 2 ? (x1 - x0) * (y1 - y0) : 0
            if area > bestArea { bestArea = area; best = joints }
        }
        guard let joints = best else { return s }
        var scratch: [Float] = []
        scratch.reserveCapacity(32)
        let size = CGSize(width: imgW, height: imgH)
        for (i, name) in Self.order.enumerated() {
            guard let j = joints[name], j.confidence > 0 else { continue }
            let u: Double, v: Double
            if orientation == .up {
                let p = j.location.toImageCoordinates(size, origin: .upperLeft)
                u = Double(p.x); v = Double(p.y)
            } else {
                // Vision reports normalized points in the upright image (lower-left origin): un-rotate to raw.
                let raw = orientation.rawNormalized(fromUpright: (Double(j.location.x), 1 - Double(j.location.y)))
                u = raw.x * Double(imgW); v = raw.y * Double(imgH)
            }
            s.joints2d[i] = [Float(u), Float(v), j.confidence]
            if let p3 = snapshot.lift(u: Float(u), v: Float(v), imgW: imgW, imgH: imgH, K: K, scratch: &scratch) {
                s.joints3d[i] = p3
            }
        }
        return s
    }
}
