import ARKit
import SwiftUI

struct ContentView: View {
    private let hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    var body: some View {
        VStack(spacing: 16) {
            Text("RadAR")
                .font(.largeTitle.bold())
            Label(
                hasLiDAR ? "LiDAR available" : "No LiDAR on this device",
                systemImage: hasLiDAR ? "checkmark.circle.fill" : "xmark.octagon.fill"
            )
            .foregroundStyle(hasLiDAR ? .green : .red)
        }
        .padding()
    }
}
