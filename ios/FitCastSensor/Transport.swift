// TCP server for the Mac (NWListener on `port`). One client at a time (the newest connection wins).
// Frames: at most one queued behind the send in flight (a newer frame replaces the pending one).
// Pings (msg_type 3) are answered immediately; an info message (msg_type 2) is sent on connect.

import Foundation
import Network
import QuartzCore

final class SensorServer: @unchecked Sendable {
    let port: UInt16
    private let queue = DispatchQueue(label: "net")
    private var listener: NWListener?
    private var conn: NWConnection?
    private var connReady = false
    private var inFlight = false
    private var pending: Data?
    private var reader = Wire.Reader()
    private var restartScheduled = false

    /// Called on the network queue when a client becomes ready; returns the info message to send first.
    var infoProvider: (() -> Data?)?
    /// Snapshot for the UI.
    let state = Locked<(connected: Bool, peer: String, listening: Bool, error: String)>((false, "", false, ""))
    let sentBytes = ByteRate(window: 1.0)
    let droppedFrames = Locked<Int>(0)
    let pings = Locked<Int>(0)

    init(port: UInt16) { self.port = port }

    /// Start listening (idempotent; restarts itself after failures).
    func start() { queue.async { self.startListener() } }

    private func startListener() {
        restartScheduled = false
        listener?.cancel()
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        let params = NWParameters(tls: nil, tcp: tcp)
        params.allowLocalEndpointReuse = true
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        do {
            let l = try NWListener(using: params, on: nwPort)
            l.stateUpdateHandler = { [weak self] st in self?.listenerState(st) }
            l.newConnectionHandler = { [weak self] c in self?.adopt(c) }
            listener = l
            l.start(queue: queue)
        } catch {
            state.mutate { $0.error = "listener: \(error)"; $0.listening = false }
            scheduleRestart()
        }
    }

    private func listenerState(_ st: NWListener.State) {
        switch st {
        case .ready: state.mutate { $0.listening = true; $0.error = "" }
        case .failed(let e):
            state.mutate { $0.listening = false; $0.error = "listener failed: \(e)" }
            scheduleRestart()
        case .cancelled: state.mutate { $0.listening = false }
        case .waiting(let e): state.mutate { $0.error = "listener waiting: \(e)" }
        default: break
        }
    }

    private func scheduleRestart() {
        guard !restartScheduled else { return }
        restartScheduled = true
        queue.asyncAfter(deadline: .now() + 1.0) { [weak self] in self?.startListener() }
    }

    private func adopt(_ c: NWConnection) {
        dropConnection()                               // newest client wins
        conn = c
        connReady = false
        reader = Wire.Reader()
        let peer = "\(c.endpoint)"
        c.stateUpdateHandler = { [weak self, weak c] st in
            guard let self, let c, c === self.conn else { return }
            switch st {
            case .ready:
                self.connReady = true
                self.state.mutate { $0.connected = true; $0.peer = peer }
                if let info = self.infoProvider?() { c.send(content: info, completion: .contentProcessed { _ in }) }
                self.receive(on: c)
            case .failed, .cancelled:
                self.dropConnection()
            default: break
            }
        }
        c.start(queue: queue)
    }

    private func dropConnection() {
        if let c = conn { c.stateUpdateHandler = nil; c.cancel() }
        conn = nil
        connReady = false
        inFlight = false
        pending = nil
        state.mutate { $0.connected = false; $0.peer = "" }
    }

    private func receive(on c: NWConnection) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self, weak c] data, _, isComplete, error in
            guard let self, let c, c === self.conn else { return }
            if let data, !data.isEmpty {
                for (type, json) in self.reader.feed(data) where type == Wire.msgPing {
                    if let pong = FrameMessage.pong(request: json, phoneTs: CACurrentMediaTime()) {
                        c.send(content: pong, completion: .contentProcessed { _ in })
                        self.pings.mutate { $0 += 1 }
                    }
                }
                if self.reader.broken {
                    self.state.mutate { $0.error = "bad bytes from client; dropped it" }
                    self.dropConnection()
                    return
                }
            }
            if isComplete || error != nil { self.dropConnection(); return }
            self.receive(on: c)
        }
    }

    /// Queue a frame message (any thread). Replaces the pending frame if a send is still in flight.
    func sendFrame(_ d: Data) {
        queue.async {
            guard self.conn != nil, self.connReady else { return }
            if self.inFlight {
                if self.pending != nil { self.droppedFrames.mutate { $0 += 1 } }
                self.pending = d
            } else {
                self.transmit(d)
            }
        }
    }

    private func transmit(_ d: Data) {
        guard let c = conn else { return }
        inFlight = true
        let n = d.count
        c.send(content: d, completion: .contentProcessed { [weak self, weak c] err in
            guard let self else { return }
            // completion runs on `queue`
            guard let c, c === self.conn else { return }
            self.inFlight = false
            if err != nil { self.dropConnection(); return }
            self.sentBytes.add(n)
            if let p = self.pending { self.pending = nil; self.transmit(p) }
        })
    }

    var isConnected: Bool { state.get().connected }
}
