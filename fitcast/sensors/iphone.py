"""iPhone LiDAR sensor receiver: replaces FrameSource + Tracker on the `--sensor iphone` path.

Connects to the FitCastSensor app (a TCP server on the phone, `sensor.port`) either through usbmux (USB) or
a plain TCP host (Wi-Fi / the fake phone). A background thread decodes FCS1 messages (protocol.py) and keeps
only the latest frame: `read()` mirrors FrameSource.read(), `track()` returns the matching TrackResult with
`depth`, `joints3d` and `K_cam` filled.

usbmux (verified against the installed pymobiledevice3 11.34.1): `pymobiledevice3.usbmux.select_device(udid=None,
connection_type=None)` and `MuxDevice.connect(port)` are COROUTINES; `connect` returns the usbmuxd socket.socket,
already connected to the device port and left NON-BLOCKING (we switch it back to blocking + timeout). Older
releases had synchronous versions; both are handled.

Clocks: the phone's `ts` (ARFrame.timestamp) and `send_ts` are phone-monotonic seconds and are NOT comparable
with the Mac's perf_counter. A ping (msg_type 3) every `ping_interval_s` estimates the clock offset from the
lowest-RTT exchange of the last `ping_window` pings; until the first pong the latency stats are None.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import math
import re
import socket
import threading
import time
from typing import Any

import cv2
import numpy as np

from .. import geometry as geo
from ..tracker import FacingClassifier, _ksize, box_dilate, compute_head_box
from ..types import L_EYE, L_HIP, L_SHOULDER, NOSE, R_EYE, R_HIP, R_SHOULDER, TrackResult
from .protocol import MSG_FRAME, MSG_INFO, MSG_PING, TYPE_KEY, FrameReader, ProtocolError, encode_frame

log = logging.getLogger("fitcast.sensors.iphone")

# Defaults for sensor keys not (yet) in config.yaml; config values always win. Proposed for config.yaml `sensor:`.
DEFAULTS: dict[str, Any] = {
    "udid": None,               # usbmux: pick this device (None = first USB device)
    "reconnect_s": 1.0,         # wait between connection attempts
    "recv_timeout_s": 0.05,     # socket timeout of the reader loop (bounds stop() and ping latency)
    "recv_bytes": 1 << 20,      # bytes per recv() call
    "max_payload_mb": 64,       # larger payload_len = corrupt stream
    "ping_interval_s": 1.0,
    "ping_window": 8,           # clock offset from the lowest-RTT ping among the last N
    "stats_window_s": 1.0,      # fps / bytes-per-second window
    "depth_interp": "nearest",  # nearest | linear (linear invents depths across edges and invalid pixels)
    "min_visibility": None,     # None = tracker.min_visibility (Vision confidences run lower than MediaPipe's)
    "synth_head_frac": 0.5,     # no face joint at all (back view): head centre this many shoulder widths above the shoulders
}

_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}

# Vision 19 joints (canonical snake_case names, CLAUDE.md) -> MediaPipe 33 index; neck/root are not mapped.
VISION_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                 "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle", "neck", "root")
VISION_TO_MP = {"nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
                "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
                "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
                "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}
# VNHumanBodyPoseObservation.JointName raw values, accepted in case the app sends those instead
_VISION_ALIASES = {"head_joint": "nose", "neck_1_joint": "neck", "left_shoulder_1_joint": "left_shoulder",
                   "right_shoulder_1_joint": "right_shoulder", "left_forearm_joint": "left_elbow",
                   "right_forearm_joint": "right_elbow", "left_hand_joint": "left_wrist",
                   "right_hand_joint": "right_wrist", "left_up_leg_joint": "left_hip", "right_up_leg_joint": "right_hip",
                   "left_leg_joint": "left_knee", "right_leg_joint": "right_knee", "left_foot_joint": "left_ankle",
                   "right_foot_joint": "right_ankle", "left_eye_joint": "left_eye", "right_eye_joint": "right_eye",
                   "left_ear_joint": "left_ear", "right_ear_joint": "right_ear", "root_joint": "root"}
# MediaPipe-only face points copy their nearest mapped point (as RtmlibBackend does)
_FACE_COPIES = {1: L_EYE, 3: L_EYE, 4: R_EYE, 6: R_EYE, 9: NOSE, 10: NOSE}
_FACE_MAPPED = (NOSE, L_EYE, R_EYE, 7, 8)


def canonical_joint_name(name: str) -> str:
    """'leftShoulder' / 'left_shoulder_1_joint' / 'LEFT_SHOULDER' -> 'left_shoulder' (unknown names pass through)."""
    n = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", str(name)).lower()
    return _VISION_ALIASES.get(n, n)


def build_mapping(names: list[str] | tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Joint-name list -> (source indices, MediaPipe indices, unknown names). Unmapped names are dropped."""
    src, dst, unknown = [], [], []
    for i, n in enumerate(names):
        c = canonical_joint_name(n)
        if c in VISION_TO_MP:
            src.append(i)
            dst.append(VISION_TO_MP[c])
        elif c not in ("neck", "root"):
            unknown.append(str(n))
    return np.array(src, np.intp), np.array(dst, np.intp), unknown


def as_float_array(rows: Any, cols: int, n: int) -> np.ndarray:
    """JSON joint rows -> (n, cols) float64; null / "nan" / NaN / missing -> NaN."""
    out = np.full((n, cols), np.nan)
    if rows is None:
        return out
    a = np.array(rows, dtype=np.float64).reshape(-1, cols)[:n]
    out[: len(a)] = a
    return out


def arkit_to_opencv(p: np.ndarray) -> np.ndarray:
    """ARKit camera coords (+x right, +y up, -z forward) -> OpenCV (+x right, +y down, +z forward)."""
    return np.asarray(p, np.float64) * np.array([1.0, -1.0, -1.0])


def map_vision_joints(header: dict, cam_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame header -> (joints (33,3) float32 camera px with z=0, visibility (33,), joints3d (33,3) float32 metres
    OpenCV coords). Missing joints: position 0 / visibility 0 / joints3d NaN. Face points 1,3,4,6,9,10 copy
    their nearest eye/nose (2D only; joints3d keeps NaN there)."""
    names = header.get("joint_names") or VISION_JOINTS
    src, dst, _ = build_mapping(names)
    n = len(names)
    j2 = as_float_array(header.get("joints2d"), 3, n)
    j3 = as_float_array(header.get("joints3d"), 3, n)
    W, H = cam_size
    sx, sy = W / float(header["img_w"]), H / float(header["img_h"])
    joints = np.zeros((33, 3), np.float32)
    vis = np.zeros((33,), np.float32)
    joints3d = np.full((33, 3), np.nan, np.float32)
    xy, conf = j2[src, :2], j2[src, 2]
    good = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
    joints[dst[good], 0] = xy[good, 0] * sx
    joints[dst[good], 1] = xy[good, 1] * sy
    vis[dst[good]] = np.clip(conf[good], 0.0, 1.0)
    joints3d[dst] = arkit_to_opencv(j3[src])
    for m, s in _FACE_COPIES.items():
        joints[m], vis[m] = joints[s], vis[s]
    return joints, vis, joints3d


def scale_intrinsics(K: list[float], img_size: tuple[float, float], cam_size: tuple[int, int]) -> np.ndarray:
    """[fx, fy, cx, cy] of the captured image -> (3,3) float64 K for camera space (plain per-axis scaling,
    the same convention CLAUDE.md uses for the depth intrinsics)."""
    sx, sy = cam_size[0] / float(img_size[0]), cam_size[1] / float(img_size[1])
    fx, fy, cx, cy = (float(v) for v in K)
    return np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], np.float64)


def _connect_usbmux(port: int, udid: str | None, timeout: float) -> socket.socket:
    """TCP connection to `port` on the USB-attached iPhone through usbmuxd (pymobiledevice3)."""
    from pymobiledevice3 import usbmux
    from pymobiledevice3.exceptions import PyMobileDevice3Exception

    async def _go() -> socket.socket:
        dev = usbmux.select_device(udid=udid)
        dev = await dev if inspect.isawaitable(dev) else dev
        if dev is None:
            raise RuntimeError("No iPhone visible to usbmuxd. Plug it in over USB, unlock it, tap 'Trust', or set "
                               "sensor.usbmux: false and sensor.host to the phone's Wi-Fi IP.")
        s = dev.connect(port)
        return await s if inspect.isawaitable(s) else s

    try:
        sock = asyncio.run(asyncio.wait_for(_go(), timeout))
    except usbmux.ConnectionFailedToUsbmuxdError as e:      # subclass of ConnectionFailedError: test it first
        raise RuntimeError("cannot reach usbmuxd (macOS runs it by default; reconnect the USB cable)") from e
    except usbmux.ConnectionFailedError as e:
        raise RuntimeError(f"iPhone found but nothing listens on port {port}: open the FitCastSensor app") from e
    except PyMobileDevice3Exception as e:
        raise RuntimeError(f"usbmux error ({type(e).__name__}: {e})") from e
    sock.setblocking(True)                                   # pymobiledevice3 leaves it non-blocking
    return sock


class IPhoneSensor:
    """Background receiver for the FitCastSensor stream; latest frame only (nothing is queued).

    Raw pieces for the depth-mask module, updated together with each TrackResult:
    `last_depth_raw` / `last_conf_raw` / `last_seg_raw` at sensor resolution and `last_conf` / `last_seg`
    resampled (nearest / bilinear + threshold) to camera space; `phone_ts` / `send_ts` of the latest frame;
    `info` = the last info message. `stats` holds fps, bytes_per_s, rtt_ms, clock_offset_s, latency_ms
    (phone send -> Mac receive), capture_latency_ms (ARFrame -> Mac receive), frames, dropped, reconnects.
    """

    def __init__(self, cfg: dict, cam_size: tuple[int, int], tracker_cfg: dict | None = None) -> None:
        self.cfg = {**DEFAULTS, **cfg}
        tcfg = tracker_cfg if tracker_cfg is not None else cfg.get("tracker_cfg")
        if tcfg is None:
            raise RuntimeError("IPhoneSensor needs the tracker config: IPhoneSensor(cfg, size, tracker_cfg=config['tracker'])")
        self.cam_size = (int(cam_size[0]), int(cam_size[1]))
        self.min_vis = float(self.cfg["min_visibility"] if self.cfg["min_visibility"] is not None
                             else tcfg["min_visibility"])
        self.margin = int(tcfg["frame_margin_px"])
        self.head_cfg = tcfg["head"]
        self.side_width_floor = float(tcfg["head"]["side_width_floor_frac"])
        self.k_gate = _ksize(tcfg["mask"]["person_gate_dilate_px"])
        self.facing = FacingClassifier(tcfg["facing"])
        self.synth_head_frac = float(self.cfg["synth_head_frac"])
        self.depth_interp = _INTERP[self.cfg["depth_interp"]]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._latest: tuple[np.ndarray, float, int] | None = None
        self._track: TrackResult | None = None
        self._image: np.ndarray | None = None      # last decoded rgb (camera space)
        self._seq = 0
        self._mapping_key: tuple | None = None
        self._warned: set[str] = set()
        self._pings: collections.deque = collections.deque(maxlen=int(self.cfg["ping_window"]))
        self._ping_id = 0
        self._recv_log: collections.deque = collections.deque()   # (t, bytes) within stats_window_s
        self._frame_times: collections.deque = collections.deque()   # published frame times within stats_window_s
        self.offset_s: float | None = None          # phone clock - Mac perf_counter
        self.phone_ts = self.send_ts = math.nan
        self.info: dict = {}
        self.last_depth_raw = self.last_conf_raw = self.last_seg_raw = None
        self.last_conf: np.ndarray | None = None
        self.last_seg: np.ndarray | None = None
        self.stats: dict[str, Any] = {"connected": False, "frames": 0, "dropped": 0, "reconnects": 0, "fps": 0.0,
                                      "bytes_per_s": 0.0, "rtt_ms": None, "clock_offset_s": None, "latency_ms": None,
                                      "capture_latency_ms": None, "process_ms": 0.0}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "IPhoneSensor":
        """Start the background connect/receive thread (reconnects on its own)."""
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="iphone-sensor", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the thread and close the connection."""
        self._stop.set()
        self._close_sock()
        if self._thread is not None:
            self._thread.join(timeout=float(self.cfg["connect_timeout_s"]) + 1.0)

    def reopen(self) -> bool:
        """FrameSource compatibility for the main loop's stall handling: drop the connection (the thread reconnects)."""
        self._close_sock()
        return True

    @property
    def alive(self) -> bool:
        """True while the receive thread runs."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def native_fps(self) -> float:
        """Measured receive fps (FrameSource compatibility)."""
        return float(self.stats["fps"])

    def _close_sock(self) -> None:
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    # -- data --------------------------------------------------------------
    def read(self) -> tuple[np.ndarray, float, int] | None:
        """Latest (frame BGR camera space, Mac receive perf_counter, seq) or None before the first frame."""
        with self._lock:
            return self._latest

    def track(self) -> TrackResult | None:
        """Latest TrackResult (ok=False when nobody is tracked; depth / K_cam are filled either way)."""
        with self._lock:
            return self._track

    # -- connection --------------------------------------------------------
    def _connect(self) -> socket.socket:
        port, timeout = int(self.cfg["port"]), float(self.cfg["connect_timeout_s"])
        if self.cfg["usbmux"]:
            sock = _connect_usbmux(port, self.cfg["udid"], timeout)
            where = f"usbmux port {port}"
        else:
            sock = socket.create_connection((self.cfg["host"], port), timeout=timeout)
            where = f"{self.cfg['host']}:{port}"
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass                                   # usbmux hands us an AF_UNIX socket
        sock.settimeout(float(self.cfg["recv_timeout_s"]))
        log.info("iPhone sensor connected (%s)", where)
        return sock

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                self._sock = sock = self._connect()
            except (OSError, RuntimeError, asyncio.TimeoutError) as e:
                failures += 1
                (log.warning if failures == 1 else log.debug)("iPhone sensor not reachable (%s); retrying every %.1f s",
                                                              e, float(self.cfg["reconnect_s"]))
                self._stop.wait(float(self.cfg["reconnect_s"]))
                continue
            failures = 0
            self.stats["connected"] = True
            try:
                self._serve(sock)
            except (OSError, EOFError, ProtocolError) as e:
                if not self._stop.is_set():
                    log.warning("iPhone sensor connection lost: %s", e)
            except Exception:  # noqa: BLE001 - the receive thread must never die silently
                log.exception("iPhone sensor receive thread error; reconnecting")
            finally:
                self.stats["connected"] = False
                self._close_sock()
            if not self._stop.is_set():
                self.stats["reconnects"] += 1
                self._stop.wait(float(self.cfg["reconnect_s"]))

    def _serve(self, sock: socket.socket) -> None:
        reader = FrameReader(sock, int(self.cfg["recv_bytes"]), int(float(self.cfg["max_payload_mb"]) * (1 << 20)))
        interval = float(self.cfg["ping_interval_s"])
        next_ping = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_ping:
                self._ping_id += 1
                sock.sendall(encode_frame({"mac_ts": now, "id": self._ping_id}, msg_type=MSG_PING))
                next_ping = now + interval
            n0 = reader.bytes_in
            msgs = reader.read_available()
            t_recv = time.perf_counter()
            self._account(t_recv, reader.bytes_in - n0)
            frames = []
            for h, b in msgs:
                kind = h[TYPE_KEY]
                if kind == MSG_FRAME:
                    frames.append((h, b))
                elif kind == MSG_PING:
                    self._on_pong(h, t_recv)
                elif kind == MSG_INFO:
                    self.info = h
                    log.info("iPhone sensor info: %s", {k: v for k, v in h.items() if k not in ("blobs", TYPE_KEY)})
            if frames:
                self.stats["frames"] += len(frames)
                for _ in frames:
                    self._fps_log(t_recv)
                self.stats["dropped"] += len(frames) - 1
                # newest frame wins; its image is the newest rgb in the batch (frames may omit rgb)
                rgb = next((b["rgb"] for _, b in reversed(frames) if "rgb" in b), None)
                self._publish(frames[-1][0], frames[-1][1], rgb, t_recv)

    # -- stats -------------------------------------------------------------
    def _account(self, t: float, nbytes: int) -> None:
        if nbytes <= 0:
            return
        q, win = self._recv_log, float(self.cfg["stats_window_s"])
        q.append((t, nbytes))
        while q and t - q[0][0] > win:
            q.popleft()
        self.stats["bytes_per_s"] = sum(n for _, n in q) / win

    def _on_pong(self, h: dict, t_recv: float) -> None:
        if "phone_ts" not in h or "mac_ts" not in h:
            return
        rtt = t_recv - float(h["mac_ts"])
        self._pings.append((rtt, float(h["phone_ts"]) - (float(h["mac_ts"]) + 0.5 * rtt)))
        self.offset_s = min(self._pings)[1]
        self.stats["rtt_ms"] = rtt * 1e3
        self.stats["clock_offset_s"] = self.offset_s

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, *args)

    # -- frame -> TrackResult ----------------------------------------------
    def _publish(self, h: dict, blobs: dict, rgb: bytes | None, t_recv: float) -> None:
        t0 = time.perf_counter()
        W, H = self.cam_size
        image = self._image
        if rgb is not None:
            img = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._warn_once("jpeg", "iPhone sensor: undecodable rgb JPEG (keeping the previous image)")
            else:
                if img.shape[1] != W or img.shape[0] != H:
                    shrink = img.shape[1] > W
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
                image = self._image = img
        if image is None:
            self._warn_once("norgb", "iPhone sensor: no rgb blob received yet; read() serves a black image")
            image = self._image = np.zeros((H, W, 3), np.uint8)

        depth_raw = blobs.get("depth")
        depth = None
        if depth_raw is not None:
            depth_raw = depth_raw.astype(np.float32)
            depth_raw[~(np.isfinite(depth_raw) & (depth_raw > 0))] = 0.0
            depth = cv2.resize(depth_raw, (W, H), interpolation=self.depth_interp)
        conf_raw = blobs.get("conf")
        conf = None if conf_raw is None else cv2.resize(np.ascontiguousarray(conf_raw), (W, H),
                                                          interpolation=cv2.INTER_NEAREST)
        seg_raw = blobs.get("seg")
        seg = None
        if seg_raw is not None:
            seg = cv2.resize(np.ascontiguousarray(seg_raw), (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY, dst=seg)
        K_cam = scale_intrinsics(h["K"], (h["img_w"], h["img_h"]), self.cam_size) if "K" in h else None

        self._log_mapping(h)
        track = self._build_track(h, seg, t_recv)
        track.depth, track.K_cam = depth, K_cam

        phone_ts, send_ts = float(h.get("ts", math.nan)), float(h.get("send_ts", math.nan))
        if self.offset_s is not None:
            self.stats["latency_ms"] = (t_recv - (send_ts - self.offset_s)) * 1e3
            self.stats["capture_latency_ms"] = (t_recv - (phone_ts - self.offset_s)) * 1e3
        with self._lock:
            self._seq += 1
            self._latest = (image, t_recv, self._seq)
            self._track = track
            self.phone_ts, self.send_ts = phone_ts, send_ts
            self.last_depth_raw, self.last_conf_raw, self.last_seg_raw = depth_raw, conf_raw, seg_raw
            self.last_conf, self.last_seg = conf, seg
        self.stats["process_ms"] = (time.perf_counter() - t0) * 1e3

    def _fps_log(self, t: float) -> None:
        """Received frame-message rate over the last `stats_window_s`."""
        q, win = self._frame_times, float(self.cfg["stats_window_s"])
        q.append(t)
        while q and t - q[0] > win:
            q.popleft()
        span = q[-1] - q[0]
        self.stats["fps"] = (len(q) - 1) / span if span > 0 else 0.0

    def _log_mapping(self, h: dict) -> None:
        names = tuple(h.get("joint_names") or ())
        if names == self._mapping_key:
            return
        self._mapping_key = names
        if not names:
            self._warn_once("nonames", "iPhone sensor: frame without joint_names; assuming the canonical Vision order")
            names = VISION_JOINTS
        src, dst, unknown = build_mapping(names)
        log.info("Vision -> MediaPipe joint mapping: %s (neck/root unmapped; face 1,3/4,6/9,10 copy eye/eye/nose)",
                 ", ".join(f"{names[s]}->{d}" for s, d in zip(src, dst)))
        if unknown:
            log.warning("iPhone sensor: unknown joint names ignored: %s", unknown)

    def _fill_face(self, joints: np.ndarray, vis: np.ndarray) -> None:
        """Make every FACE_IDX point usable by compute_head_box: missing face points take the centroid of the
        present ones; with no face point at all (back view) they sit `synth_head_frac` shoulder widths above
        the shoulder centre along the torso axis. Filled points keep visibility 0."""
        present = [i for i in _FACE_MAPPED if vis[i] > 0]
        if present:
            c = joints[present, :2].mean(axis=0)
        else:
            sc = 0.5 * (joints[L_SHOULDER, :2] + joints[R_SHOULDER, :2])
            up = np.array([0.0, -1.0], np.float32)
            if vis[L_HIP] > 0 and vis[R_HIP] > 0:
                up = geo.unit(sc - 0.5 * (joints[L_HIP, :2] + joints[R_HIP, :2]))
            c = sc + up * self.synth_head_frac * geo.shoulder_width(joints)
        for i in _FACE_MAPPED:
            if vis[i] <= 0:
                joints[i, :2] = c
        for m, s in _FACE_COPIES.items():
            joints[m], vis[m] = joints[s], vis[s]

    def _build_track(self, h: dict, seg: np.ndarray | None, ts: float) -> TrackResult:
        """Same semantics as Tracker.process(): off-frame visibility, facing, shoulder widths, head box."""
        W, H = self.cam_size
        joints, vis, joints3d = map_vision_joints(h, self.cam_size)
        ok = bool(vis[L_SHOULDER] >= self.min_vis and vis[R_SHOULDER] >= self.min_vis)
        if not ok:
            t = TrackResult.empty(H, W, ts)
            t.joints3d = joints3d
            return t
        inside = ((joints[:, 0] >= self.margin) & (joints[:, 0] <= W - self.margin) &
                  (joints[:, 1] >= self.margin) & (joints[:, 1] <= H - self.margin))
        hips_in_frame = bool(inside[L_HIP] and inside[R_HIP])
        inside[[L_HIP, R_HIP]] = True
        vis = np.where(inside, vis, 0.0).astype(np.float32)
        self._fill_face(joints, vis)
        sw = geo.shoulder_width(joints)
        facing = self.facing.update(joints, vis, self.min_vis, hips_in_frame)
        ref_px = self.facing.ref_px or sw
        sw_for_size = max(sw, self.side_width_floor * ref_px)
        person = None
        if seg is not None:
            person = np.empty_like(seg)
            box_dilate(seg, self.k_gate, person, np.empty_like(seg))
        use_pm = self.head_cfg.get("use_person_mask", True)
        head = compute_head_box(joints, vis, sw_for_size, self.head_cfg, (W, H), person if use_pm else None)
        # PLACEHOLDER clothes mask: the whole ARKit person segmentation (fitcast/depthmask.py replaces it)
        clothes = seg.copy() if seg is not None else np.zeros((H, W), np.uint8)
        return TrackResult(ok=True, joints=joints, visibility=vis, clothes_mask=clothes, head_box=head,
                           facing=facing, timestamp=ts, shoulder_width=sw, ref_shoulder_width=ref_px,
                           person_mask=person, facing_ratio=float(self.facing.ratio), joints3d=joints3d)
