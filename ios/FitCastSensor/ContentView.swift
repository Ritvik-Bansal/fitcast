import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            CapabilityView().tabItem { Label("Capability", systemImage: "checklist") }
            StreamView().tabItem { Label("Stream", systemImage: "dot.radiowaves.left.and.right") }
        }
    }
}

/// Capability report (selectable text); written to Documents/capability_report.json when complete.
struct CapabilityView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Capability report").font(.headline)
                Spacer()
                Button("Re-measure fps") { model.remeasure() }
            }
            if !model.reportPath.isEmpty {
                Text("Saved: \(model.reportPath)").font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            ScrollView {
                Text(model.reportText)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
    }
}

/// Live preview with the joint overlay, status line and stream controls.
struct StreamView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            PreviewWithJoints(image: model.preview, pose: model.pose)
                .aspectRatio(4.0 / 3.0, contentMode: .fit)
                .background(Color.black)
            VStack(alignment: .leading, spacing: 8) {
                statusText
                Picker("Orientation", selection: $model.orientation) {
                    ForEach(CaptureOrientation.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Stepper("RGB every \(model.rgbEvery) frame\(model.rgbEvery == 1 ? "" : "s")", value: $model.rgbEvery, in: 1 ... 4)
                Toggle("Autofocus (off = locked)", isOn: Binding(get: { model.autofocus }, set: { model.userSetAutofocus($0) }))
                Toggle("Person segmentation", isOn: Binding(get: { model.segmentation }, set: { model.userSetSegmentation($0) }))
                Spacer(minLength: 0)
            }
            .frame(maxWidth: 330)
        }
        .padding(8)
        .onAppear { model.setPreview(true) }
        .onDisappear { model.setPreview(false) }
    }

    private var statusText: some View {
        let s = model.status
        return VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "AR %.1f fps   pose %.1f fps   depth %.0f%%", s.arFps, s.poseFps, s.depthValid * 100))
            Text(s.connected ? "client \(s.peer)" : (s.listening ? "listening on :\(SensorController.port)" : "listener down"))
                .foregroundStyle(s.connected ? .green : .orange)
            Text(String(format: "%.1f MB/s   thermal %@   fx %.1f", s.mbps, s.thermal, s.fx))
            Text("\(s.format)  \(s.semantics)")
            Text(String(format: "delegate %.2f ms   pack %.2f ms", s.delegateMs, s.packMs))
            Text("focus \(s.focusLocked ? "locked" : "auto") (\(s.focusMethod))  dropped \(s.dropped)/\(s.netDropped)  pings \(s.pings)")
            if !s.message.isEmpty { Text(s.message).foregroundStyle(.red).lineLimit(3) }
        }
        .font(.system(.caption, design: .monospaced))
    }
}

/// Downscaled camera frame (raw sensor orientation, which is upright for the landscape-right UI) and the
/// 19 joints drawn in captured-image pixels scaled to the view.
struct PreviewWithJoints: View {
    let image: CGImage?
    let pose: PoseSample

    var body: some View {
        GeometryReader { geo in
            ZStack {
                if let image {
                    Image(decorative: image, scale: 1).resizable().frame(width: geo.size.width, height: geo.size.height)
                } else {
                    Text("waiting for camera...").foregroundStyle(.white)
                }
                Canvas { ctx, size in
                    guard pose.imgW > 0, pose.imgH > 0 else { return }
                    let sx = size.width / CGFloat(pose.imgW), sy = size.height / CGFloat(pose.imgH)
                    func pt(_ i: Int) -> CGPoint? {
                        let r = pose.joints2d[i]
                        guard r[0].isFinite, r[1].isFinite else { return nil }
                        return CGPoint(x: CGFloat(r[0]) * sx, y: CGFloat(r[1]) * sy)
                    }
                    var path = Path()
                    for (a, b) in Joints.bones { if let p = pt(a), let q = pt(b) { path.move(to: p); path.addLine(to: q) } }
                    ctx.stroke(path, with: .color(.cyan), lineWidth: 2)
                    for i in 0 ..< Joints.count {
                        guard let p = pt(i) else { continue }
                        let has3d = pose.joints3d[i][2].isFinite
                        let r: CGFloat = 4
                        ctx.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                                 with: .color(has3d ? .green : .yellow))
                    }
                    if let nose = pt(0) {
                        ctx.draw(Text("nose").font(.caption2).foregroundColor(.white), at: CGPoint(x: nose.x, y: nose.y - 10))
                    }
                }
            }
        }
    }
}
