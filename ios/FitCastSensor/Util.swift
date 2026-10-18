// Small thread-safety and statistics helpers shared by the capture, pack, vision and network queues.

import Foundation
import QuartzCore

/// A value guarded by a lock (readers/writers on different queues).
final class Locked<T>: @unchecked Sendable {
    private var value: T
    private let lock = NSLock()
    init(_ v: T) { value = v }
    func get() -> T { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: T) { lock.lock(); value = v; lock.unlock() }
    @discardableResult
    func mutate<R>(_ f: (inout T) -> R) -> R { lock.lock(); defer { lock.unlock() }; return f(&value) }
}

/// Events-per-second over a sliding window (call `tick` per event, `rate` from any queue).
final class RateCounter: @unchecked Sendable {
    private var times: [Double] = []
    private var head = 0
    private let window: Double
    private let lock = NSLock()
    init(window: Double = 1.0) { self.window = window; times.reserveCapacity(256) }

    func tick(_ t: Double = CACurrentMediaTime()) {
        lock.lock(); defer { lock.unlock() }
        times.append(t)
        while head < times.count, t - times[head] > window { head += 1 }
        if head > 512 { times.removeFirst(head); head = 0 }
    }

    var rate: Double {
        lock.lock(); defer { lock.unlock() }
        let now = CACurrentMediaTime()
        var i = head
        while i < times.count, now - times[i] > window { i += 1 }
        let n = times.count - i
        guard n >= 2 else { return 0 }
        let span = times[times.count - 1] - times[i]
        return span > 0 ? Double(n - 1) / span : 0
    }
}

/// Bytes-per-second over a sliding window.
final class ByteRate: @unchecked Sendable {
    private var log: [(Double, Int)] = []
    private let window: Double
    private let lock = NSLock()
    init(window: Double = 1.0) { self.window = window }
    func add(_ n: Int, _ t: Double = CACurrentMediaTime()) {
        lock.lock(); defer { lock.unlock() }
        log.append((t, n))
        if let first = log.first, t - first.0 > window * 2 { log.removeAll { t - $0.0 > window } }
    }
    var bytesPerSecond: Double {
        lock.lock(); defer { lock.unlock() }
        let now = CACurrentMediaTime()
        return Double(log.reduce(0) { now - $1.0 <= window ? $0 + $1.1 : $0 }) / window
    }
}

/// OSType -> printable four-character code ("420f", "fdep", ...).
func fourCC(_ t: OSType) -> String {
    let bytes = [UInt8((t >> 24) & 0xFF), UInt8((t >> 16) & 0xFF), UInt8((t >> 8) & 0xFF), UInt8(t & 0xFF)]
    if bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) { return String(decoding: bytes, as: UTF8.self) }
    return String(t)
}

/// ProcessInfo thermal state as a short string ("nominal" | "fair" | "serious" | "critical").
func thermalStateString(_ s: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// Hardware model identifier, e.g. "iPhone18,1".
func deviceModelIdentifier() -> String {
    var u = utsname()
    uname(&u)
    return withUnsafeBytes(of: &u.machine) { raw in
        String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
    }
}
