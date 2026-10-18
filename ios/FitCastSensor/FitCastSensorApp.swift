import SwiftUI
import UIKit

@main
struct FitCastSensorApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .onAppear {
                    UIApplication.shared.isIdleTimerDisabled = true
                    model.start()
                }
        }
    }
}

/// Main-actor bridge between SwiftUI and SensorController (polls status a few times per second).
@MainActor
final class AppModel: ObservableObject {
    let sensor = SensorController()
    @Published var status = StreamStatus()
    @Published var reportText = "running checks..."
    @Published var reportPath = ""
    @Published var preview: CGImage?
    @Published var pose = PoseSample.empty()
    @Published var orientation: CaptureOrientation = .up { didSet { sensor.settings.mutate { $0.orientation = orientation } } }
    @Published var rgbEvery = 1 { didSet { sensor.settings.mutate { $0.rgbEvery = rgbEvery } } }
    @Published var autofocus = true
    @Published var segmentation = true

    private var started = false
    private var timer: Timer?

    func start() {
        guard !started else { return }
        started = true
        sensor.onReportUpdated = { [weak self] in Task { @MainActor in self?.refreshReport() } }
        sensor.start()
        refreshReport()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private var tickCount = 0
    private func tick() {
        tickCount += 1
        let previewOn = sensor.settings.get().previewOn
        if previewOn {
            preview = sensor.previewer.image.get()
            pose = sensor.pose.current
        }
        if tickCount % 3 == 0 {
            status = sensor.status()
            let locked = status.focusLocked
            if autofocus == locked { autofocus = !locked }      // reflect the automatic 3 s lock
        }
        if tickCount % 20 == 0 { refreshReport() }
    }

    func refreshReport() {
        let r = sensor.report.get()
        reportText = Capability.prettyJSON(r)
        if (r["status"] as? String) == "complete", reportPath.isEmpty { reportPath = Capability.write(r) }
    }

    func setPreview(_ on: Bool) { sensor.settings.mutate { $0.previewOn = on } }
    func userSetAutofocus(_ on: Bool) { autofocus = on; sensor.setAutofocus(on) }
    func userSetSegmentation(_ on: Bool) { segmentation = on; sensor.setSegmentation(on) }
    func remeasure() { sensor.remeasure(); reportPath = "" }
}
