// FCS1 wire format (phone -> Mac). Foundation only, so it also compiles on macOS for the byte-level test
// against fitcast/sensors/protocol.py (FrameReader). See CLAUDE.md "3D sensor path" / protocol.py.
//
//   u32 magic = 0x46435331   u8 version = 1   u8 msg_type (1 frame, 2 info, 3 ping)
//   u16 json_len             u32 payload_len  json[json_len]  payload[payload_len]     (little-endian)

import Foundation

enum Wire {
    static let magic: UInt32 = 0x4643_5331
    static let version: UInt8 = 1
    static let headerSize = 12
    static let msgFrame: UInt8 = 1
    static let msgInfo: UInt8 = 2
    static let msgPing: UInt8 = 3
    static let maxJSON = 0xFFFF
    static let maxPayload = 64 << 20

    /// One payload blob. `bytes` must stay valid until `encode` returns.
    struct Blob {
        let name: String
        let dtype: String          // "float16" | "uint8" | "jpeg"
        let shape: [Int]
        let bytes: UnsafeRawBufferPointer
    }

    /// Full message: fixed header + JSON (the caller's `json` object body plus the generated `blobs` table) + payload.
    /// `jsonBody` is a JSONWriter whose object is still OPEN (no closing brace); `encode` appends `blobs` and closes it.
    /// Returns nil when the JSON would exceed the u16 length field.
    static func encode(msgType: UInt8, jsonBody: inout JSONWriter, blobs: [Blob]) -> Data? {
        var offset = 0
        jsonBody.key("blobs")
        jsonBody.beginArray()
        for b in blobs {
            jsonBody.beginObject()
            jsonBody.field("name", b.name)
            jsonBody.field("dtype", b.dtype)
            jsonBody.key("shape"); jsonBody.intArray(b.shape)
            jsonBody.field("offset", offset)
            jsonBody.field("nbytes", b.bytes.count)
            jsonBody.endObject()
            offset += b.bytes.count
        }
        jsonBody.endArray()
        jsonBody.endObject()
        let js = jsonBody.bytes
        guard js.count <= maxJSON, offset <= maxPayload else { return nil }
        var d = Data(capacity: headerSize + js.count + offset)
        appendLE(&d, magic)
        d.append(version)
        d.append(msgType)
        appendLE(&d, UInt16(js.count))
        appendLE(&d, UInt32(offset))
        d.append(contentsOf: js)
        for b in blobs where b.bytes.count > 0 {
            d.append(b.bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), count: b.bytes.count)
        }
        return d
    }

    /// Message from an already serialised JSON object (no blobs; the object must already contain "blobs": []).
    static func encodeRaw(msgType: UInt8, json: Data) -> Data? {
        guard json.count <= maxJSON else { return nil }
        var d = Data(capacity: headerSize + json.count)
        appendLE(&d, magic)
        d.append(version)
        d.append(msgType)
        appendLE(&d, UInt16(json.count))
        appendLE(&d, UInt32(0))
        d.append(json)
        return d
    }

    static func appendLE<T: FixedWidthInteger>(_ d: inout Data, _ v: T) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { d.append(contentsOf: $0) }
    }

    /// Incremental decoder for messages arriving from the Mac (pings). Returns (msgType, json bytes) per message.
    struct Reader {
        private var buf = Data()
        private(set) var broken = false

        mutating func feed(_ data: Data) -> [(UInt8, Data)] {
            guard !broken else { return [] }
            buf.append(data)
            var out: [(UInt8, Data)] = []
            var pos = buf.startIndex
            while buf.endIndex - pos >= headerSize {
                let magicIn = readLE(UInt32.self, at: pos)
                let ver = buf[pos + 4], mtype = buf[pos + 5]
                let jlen = Int(readLE(UInt16.self, at: pos + 6)), plen = Int(readLE(UInt32.self, at: pos + 8))
                if magicIn != magic || ver != version || plen > maxPayload { broken = true; return out }
                let total = headerSize + jlen + plen
                if buf.endIndex - pos < total { break }
                let j0 = pos + headerSize
                out.append((mtype, Data(buf[j0 ..< j0 + jlen])))
                pos += total
            }
            buf = Data(buf[pos...])
            return out
        }

        private func readLE<T: FixedWidthInteger>(_: T.Type, at i: Data.Index) -> T {
            var v: T = 0
            for k in 0 ..< MemoryLayout<T>.size { v |= T(buf[i + k]) << (8 * k) }
            return v
        }
    }
}

/// Minimal allocation-light JSON writer. Non-finite numbers are written as `null` (protocol.py's
/// as_float_array maps null -> NaN); Float32 values use Float's shortest round-trip text.
struct JSONWriter {
    private(set) var bytes: [UInt8] = []
    private var counts: [Int] = []          // values written so far in each open container
    private var afterKey = false            // the next value completes a `"key":` pair

    init(reserve: Int = 4096) { bytes.reserveCapacity(reserve) }

    private mutating func raw(_ s: String) { bytes.append(contentsOf: s.utf8) }

    /// Comma handling for a new element of the innermost container.
    private mutating func element() {
        guard !counts.isEmpty else { return }
        if counts[counts.count - 1] > 0 { bytes.append(UInt8(ascii: ",")) }
        counts[counts.count - 1] += 1
    }
    private mutating func valuePrefix() {
        if afterKey { afterKey = false } else { element() }
    }

    mutating func beginObject() { valuePrefix(); bytes.append(UInt8(ascii: "{")); counts.append(0) }
    mutating func endObject() { counts.removeLast(); bytes.append(UInt8(ascii: "}")) }
    mutating func beginArray() { valuePrefix(); bytes.append(UInt8(ascii: "[")); counts.append(0) }
    mutating func endArray() { counts.removeLast(); bytes.append(UInt8(ascii: "]")) }

    /// Writes `"k":`; the next value written belongs to it.
    mutating func key(_ k: String) {
        element()
        quoted(k)
        bytes.append(UInt8(ascii: ":"))
        afterKey = true
    }

    private mutating func quoted(_ s: String) {
        bytes.append(UInt8(ascii: "\""))
        for u in s.utf8 {
            switch u {
            case UInt8(ascii: "\""): raw("\\\"")
            case UInt8(ascii: "\\"): raw("\\\\")
            case 0x0A: raw("\\n")
            case 0x0D: raw("\\r")
            case 0x09: raw("\\t")
            case 0 ..< 0x20: raw(String(format: "\\u%04x", Int(u)))
            default: bytes.append(u)
            }
        }
        bytes.append(UInt8(ascii: "\""))
    }

    mutating func string(_ s: String) { valuePrefix(); quoted(s) }
    mutating func null() { valuePrefix(); raw("null") }
    mutating func bool(_ v: Bool) { valuePrefix(); raw(v ? "true" : "false") }
    mutating func int(_ v: Int) { valuePrefix(); raw(String(v)) }
    mutating func double(_ v: Double) { if v.isFinite { valuePrefix(); raw(v.description) } else { null() } }
    mutating func float(_ v: Float) { if v.isFinite { valuePrefix(); raw(v.description) } else { null() } }

    mutating func intArray(_ a: [Int]) { beginArray(); for v in a { int(v) }; endArray() }
    mutating func floatArray(_ a: [Float]) { beginArray(); for v in a { float(v) }; endArray() }
    mutating func stringArray(_ a: [String]) { beginArray(); for v in a { string(v) }; endArray() }

    mutating func field(_ k: String, _ v: String) { key(k); string(v) }
    mutating func field(_ k: String, _ v: Int) { key(k); int(v) }
    mutating func field(_ k: String, _ v: Double) { key(k); double(v) }
    mutating func field(_ k: String, _ v: Float) { key(k); float(v) }
    mutating func field(_ k: String, _ v: Bool) { key(k); bool(v) }
    mutating func optionalField(_ k: String, _ v: Double?) { key(k); if let v { double(v) } else { null() } }
}
