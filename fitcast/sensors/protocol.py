"""FitCast sensor wire protocol (phone <-> Mac), see CLAUDE.md "3D sensor path".

TCP stream, little-endian. Each message = 12-byte fixed header + JSON header + binary payload:

    u32 magic = 0x46435331 ('FCS1')   u8 version = 1   u8 msg_type (1 frame, 2 info, 3 ping)
    u16 json_len                       u32 payload_len
    json[json_len]                     payload[payload_len]

The JSON header lists the payload blobs as {name, dtype, shape, offset, nbytes}. Numeric dtypes are numpy
names ("float16", "uint8", ...); dtype "jpeg" (the `rgb` blob) or "bytes" means opaque bytes.

Ping (msg_type 3): the Mac sends {"mac_ts": perf_counter, "id": n}; the phone echoes the same JSON with
"phone_ts" (its monotonic clock, the clock of `send_ts` / ARFrame.timestamp) added.
"""
from __future__ import annotations

import json
import logging
import struct
from typing import Any, Iterator

import numpy as np

log = logging.getLogger("fitcast.sensors.protocol")

MAGIC = 0x46435331
VERSION = 1
MSG_FRAME, MSG_INFO, MSG_PING = 1, 2, 3
KNOWN_TYPES = (MSG_FRAME, MSG_INFO, MSG_PING)
HEADER = struct.Struct("<IBBHI")            # magic, version, msg_type, json_len, payload_len
RAW_DTYPES = ("jpeg", "bytes")
NUM_DTYPES = ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float16", "float32", "float64")
MAX_PAYLOAD = 64 << 20                      # protocol sanity bound: larger lengths mean a corrupt stream
TYPE_KEY = "_msg_type"                      # decoded headers carry their msg_type under this key

Message = tuple[dict, dict[str, Any]]


class ProtocolError(RuntimeError):
    """The byte stream is not a valid FCS1 stream (bad magic/version/lengths/JSON/blob table)."""


def encode_frame(header: dict, blobs: dict[str, bytes | np.ndarray] | None = None,
                 msg_type: int = MSG_FRAME) -> bytes:
    """Reference encoder: serialise `header` (+ an auto-generated `blobs` table) and `blobs` into one message.

    numpy arrays are sent C-contiguous little-endian with their dtype name and shape; bytes-like values
    are opaque (dtype "jpeg" for the `rgb` blob, "bytes" otherwise).
    """
    descs, parts, off = [], [], 0
    for name, v in (blobs or {}).items():
        if isinstance(v, (bytes, bytearray, memoryview)):
            data = bytes(v)
            desc = {"dtype": "jpeg" if name == "rgb" else "bytes", "shape": [len(data)]}
        else:
            a = np.ascontiguousarray(v)
            if a.dtype.name not in NUM_DTYPES:
                raise ValueError(f"blob {name!r}: unsupported dtype {a.dtype}")
            data = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
            desc = {"dtype": a.dtype.name, "shape": list(a.shape)}
        descs.append({"name": name, **desc, "offset": off, "nbytes": len(data)})
        parts.append(data)
        off += len(data)
    h = {k: v for k, v in header.items() if k != TYPE_KEY}
    h["blobs"] = descs
    js = json.dumps(h, separators=(",", ":")).encode()
    if len(js) > 0xFFFF:
        raise ValueError(f"JSON header too long ({len(js)} > 65535 bytes)")
    payload = b"".join(parts)
    return HEADER.pack(MAGIC, VERSION, msg_type, len(js), len(payload)) + js + payload


def decode_blobs(header: dict, payload: bytes) -> dict[str, Any]:
    """Blobs described by header["blobs"] -> {name: np.ndarray (read-only view) | bytes}. Raises ProtocolError."""
    out: dict[str, Any] = {}
    for b in header.get("blobs") or []:
        try:
            name, dtype, off, n = str(b["name"]), str(b["dtype"]), int(b["offset"]), int(b["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"bad blob descriptor {b!r}") from e
        if off < 0 or n < 0 or off + n > len(payload):
            raise ProtocolError(f"blob {name!r} [{off}, {off + n}) outside payload of {len(payload)} bytes")
        if dtype in RAW_DTYPES:
            out[name] = payload[off:off + n]
            continue
        if dtype not in NUM_DTYPES:
            raise ProtocolError(f"blob {name!r}: unknown dtype {dtype!r}")
        dt = np.dtype(dtype).newbyteorder("<")
        shape = tuple(int(s) for s in b.get("shape", ()))
        count = int(np.prod(shape)) if shape else n // dt.itemsize
        if count * dt.itemsize != n:
            raise ProtocolError(f"blob {name!r}: shape {shape} x {dtype} != {n} bytes")
        out[name] = np.frombuffer(payload, dt, count=count, offset=off).reshape(shape or (count,))
    return out


class FrameReader:
    """Incremental FCS1 decoder. Push bytes with `feed()`, or give it a socket/stream and call
    `read_available()` / iterate. Yields (header, blobs); header[TYPE_KEY] is the msg_type.

    Unknown msg types are skipped (logged once per type). Any framing error raises ProtocolError and
    leaves the reader broken (the stream is desynchronised: reconnect).
    """

    def __init__(self, stream: Any = None, recv_bytes: int = 1 << 20, max_payload: int = MAX_PAYLOAD) -> None:
        self.stream = stream
        self.recv_bytes = int(recv_bytes)
        self.max_payload = int(max_payload)
        self._buf = bytearray()
        self._broken: ProtocolError | None = None
        self._skipped: set[int] = set()
        self.bytes_in = 0

    def _fail(self, msg: str) -> None:
        self._broken = ProtocolError(msg)
        raise self._broken

    def feed(self, data: bytes) -> list[Message]:
        """Append bytes and return every message completed by them (possibly none)."""
        if self._broken is not None:
            raise self._broken
        buf = self._buf
        buf += data
        self.bytes_in += len(data)
        out: list[Message] = []
        pos = 0
        while len(buf) - pos >= 4:
            if int.from_bytes(buf[pos:pos + 4], "little") != MAGIC:
                self._fail(f"bad magic 0x{int.from_bytes(buf[pos:pos + 4], 'little'):08x} (expected 0x{MAGIC:08x})")
            if len(buf) - pos < HEADER.size:
                break
            _, ver, mtype, jlen, plen = HEADER.unpack_from(buf, pos)
            if ver != VERSION:
                self._fail(f"unsupported protocol version {ver} (expected {VERSION})")
            if plen > self.max_payload:
                self._fail(f"payload_len {plen} exceeds {self.max_payload} bytes")
            total = HEADER.size + jlen + plen
            if len(buf) - pos < total:
                break
            j0 = pos + HEADER.size
            if mtype in KNOWN_TYPES:
                try:
                    header = json.loads(bytes(buf[j0:j0 + jlen]))
                except ValueError as e:
                    self._fail(f"bad JSON header: {e}")
                if not isinstance(header, dict):
                    self._fail("JSON header is not an object")
                try:
                    blobs = decode_blobs(header, bytes(buf[j0 + jlen:pos + total]))
                except ProtocolError as e:
                    self._broken = e
                    raise
                header[TYPE_KEY] = mtype
                out.append((header, blobs))
            elif mtype not in self._skipped:
                self._skipped.add(mtype)
                log.warning("skipping unknown sensor msg_type %d", mtype)
            pos += total
        del buf[:pos]
        return out

    def read_available(self) -> list[Message]:
        """One recv()/read() from the stream, then `feed()`. [] on socket timeout; EOFError when closed."""
        try:
            data = self.stream.recv(self.recv_bytes) if hasattr(self.stream, "recv") else self.stream.read(self.recv_bytes)
        except TimeoutError:            # socket.timeout is an alias of TimeoutError (3.10+)
            return []
        if not data:
            raise EOFError("sensor stream closed")
        return self.feed(data)

    def __iter__(self) -> Iterator[Message]:
        """Yield messages until the stream ends."""
        while True:
            try:
                msgs = self.read_available()
            except EOFError:
                return
            yield from msgs
