"""iPhone sensor path: FCS1 protocol, Vision->MediaPipe mapping, ARKit->OpenCV, loopback against the fake phone."""
from __future__ import annotations

import io
import json
import math
import threading
import time

import cv2
import numpy as np
import pytest

from fitcast.sensors.fake_phone import FakePhone
from fitcast.sensors.iphone import (VISION_JOINTS, VISION_TO_MP, IPhoneSensor, arkit_to_opencv, map_vision_joints,
                                    scale_intrinsics)
from fitcast.sensors.protocol import (HEADER, MAGIC, MSG_FRAME, MSG_PING, TYPE_KEY, FrameReader, ProtocolError,
                                      encode_frame)
from fitcast.types import L_SHOULDER, R_SHOULDER


def _sample() -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 4.0, (192, 256)).astype(np.float16)
    conf = rng.integers(0, 3, (192, 256), dtype=np.uint8)
    seg = (rng.random((192, 256)) > 0.5).astype(np.uint8) * 255
    ok, jpg = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))
    header = {"ts": 12.5, "send_ts": 12.51, "img_w": 1920, "img_h": 1440, "K": [1500.0, 1500.0, 960.0, 720.0],
              "joint_names": list(VISION_JOINTS), "joints2d": [[1.0, 2.0, 0.5]] * 19}
    return header, {"depth": depth, "conf": conf, "seg": seg, "rgb": jpg.tobytes()}


def _check(h: dict, b: dict, header: dict, blobs: dict) -> None:
    assert h[TYPE_KEY] == MSG_FRAME
    assert h["ts"] == header["ts"] and h["K"] == header["K"] and h["joints2d"] == header["joints2d"]
    for k in ("depth", "conf", "seg"):
        assert b[k].dtype == blobs[k].dtype and b[k].shape == blobs[k].shape
        np.testing.assert_array_equal(b[k], blobs[k])
    assert b["rgb"] == blobs["rgb"]
    assert cv2.imdecode(np.frombuffer(b["rgb"], np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)


def test_round_trip() -> None:
    header, blobs = _sample()
    msg = encode_frame(header, blobs)
    magic, ver, mtype, jlen, plen = HEADER.unpack_from(msg)
    assert (magic, ver, mtype) == (MAGIC, 1, MSG_FRAME) and len(msg) == HEADER.size + jlen + plen
    assert msg[:4] == b"1SCF"                       # little-endian 'FCS1'
    js = json.loads(msg[HEADER.size:HEADER.size + jlen])
    assert {d["name"]: d["dtype"] for d in js["blobs"]} == {"depth": "float16", "conf": "uint8", "seg": "uint8",
                                                           "rgb": "jpeg"}
    out = FrameReader().feed(msg)
    assert len(out) == 1
    _check(*out[0], header, blobs)


def test_partial_reads_and_stream_iteration() -> None:
    header, blobs = _sample()
    ping = encode_frame({"mac_ts": 1.0, "id": 1}, msg_type=MSG_PING)
    unknown = encode_frame({"x": 1}, msg_type=9)           # unknown msg types are skipped
    data = encode_frame(header, blobs) + unknown + ping + encode_frame(header, blobs)
    r = FrameReader()
    msgs = []
    for i in range(0, len(data), 7):
        msgs += r.feed(data[i:i + 7])
    assert [h[TYPE_KEY] for h, _ in msgs] == [MSG_FRAME, MSG_PING, MSG_FRAME]
    _check(*msgs[0], header, blobs)
    _check(*msgs[2], header, blobs)
    assert msgs[1][0]["mac_ts"] == 1.0
    # file-like stream source
    assert [h[TYPE_KEY] for h, _ in FrameReader(io.BytesIO(data), recv_bytes=1000)] == [MSG_FRAME, MSG_PING, MSG_FRAME]


def test_corrupt_stream_raises() -> None:
    header, blobs = _sample()
    msg = bytearray(encode_frame(header, blobs))
    bad = bytearray(msg)
    bad[0] ^= 0xFF
    with pytest.raises(ProtocolError):
        FrameReader().feed(bytes(bad[:7]))              # magic is checked as soon as 4 bytes are in
    r = FrameReader()
    r.feed(bytes(msg))
    with pytest.raises(ProtocolError):
        r.feed(b"\x00" * 16)                            # garbage after a valid message
    with pytest.raises(ProtocolError):
        r.feed(bytes(msg))                              # reader stays broken
    wrong_ver = bytearray(msg)
    wrong_ver[4] = 2
    with pytest.raises(ProtocolError):
        FrameReader().feed(bytes(wrong_ver))
    h2 = dict(header, blobs=[{"name": "depth", "dtype": "float16", "shape": [192, 256], "offset": 0, "nbytes": 999}])
    js = json.dumps(h2).encode()
    forged = HEADER.pack(MAGIC, 1, MSG_FRAME, len(js), 999) + js + b"\x00" * 999
    with pytest.raises(ProtocolError):
        FrameReader().feed(forged)                      # shape x dtype disagrees with nbytes


def test_joint_mapping_and_scaling() -> None:
    names = list(reversed(VISION_JOINTS))               # mapping is by name, not position
    rng = np.random.default_rng(1)
    xy = rng.uniform(100, 1400, (19, 2))
    conf = rng.uniform(0.2, 0.95, 19)
    rows = [[float(x), float(y), float(c)] for (x, y), c in zip(xy, conf)]
    miss = names.index("right_knee")
    rows[miss] = [None, None, None]                     # Swift sends null for missing joints
    j3 = rng.uniform(-1, 1, (19, 3))
    j3[:, 2] = -2.0
    header = {"img_w": 1920, "img_h": 1440, "joint_names": names, "joints2d": rows, "joints3d": j3.tolist()}
    joints, vis, joints3d = map_vision_joints(header, (1280, 720))
    sx, sy = 1280 / 1920, 720 / 1440
    for i, n in enumerate(names):
        if n in ("neck", "root"):
            continue
        m = VISION_TO_MP[n]
        np.testing.assert_allclose(joints3d[m], j3[i] * [1, -1, -1], rtol=1e-6)
        if i == miss:
            assert vis[m] == 0 and joints[m, 0] == 0
            continue
        np.testing.assert_allclose(joints[m], [xy[i, 0] * sx, xy[i, 1] * sy, 0.0], rtol=1e-5)
        assert vis[m] == pytest.approx(conf[i])
    assert {VISION_TO_MP[n] for n in VISION_TO_MP} == {0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
    for m, src in {1: 2, 3: 2, 4: 5, 6: 5, 9: 0, 10: 0}.items():         # face copies
        np.testing.assert_array_equal(joints[m], joints[src])
    unmapped = [i for i in range(33) if i not in set(VISION_TO_MP.values()) | {1, 3, 4, 6, 9, 10}]
    assert (vis[unmapped] == 0).all() and np.isnan(joints3d[unmapped]).all()
    # camelCase / Vision raw-value names resolve to the same indices
    alt = dict(header, joint_names=["leftShoulder" if n == "left_shoulder" else
                                    "right_shoulder_1_joint" if n == "right_shoulder" else n for n in names])
    j_alt, v_alt, _ = map_vision_joints(alt, (1280, 720))
    np.testing.assert_array_equal(j_alt[[L_SHOULDER, R_SHOULDER]], joints[[L_SHOULDER, R_SHOULDER]])
    K = scale_intrinsics([1500, 1400, 960, 720], (1920, 1440), (1280, 720))
    np.testing.assert_allclose(K, [[1000, 0, 640], [0, 700, 360], [0, 0, 1]])


def test_arkit_to_opencv_signs() -> None:
    # ARKit: a point 2 m in front, 0.5 m right and 0.3 m up is (0.5, 0.3, -2)
    p = arkit_to_opencv(np.array([[0.5, 0.3, -2.0]]))[0]
    assert p[0] == 0.5 and p[1] == -0.3 and p[2] == 2.0       # OpenCV: right +x, up = -y, forward +z
    K = scale_intrinsics([1500, 1500, 960, 720], (1920, 1440), (1920, 1440))
    u, v, w = K @ p
    assert u / w > 960 and v / w < 720                         # right of and above the principal point


def _wait(cond, timeout: float) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_loopback_fake_phone(config: dict) -> None:
    phone = FakePhone(port=0, fps=60).start()
    cfg = {**config["sensor"], "usbmux": False, "host": "127.0.0.1", "port": phone.port}
    sensor = IPhoneSensor(cfg, (1280, 720), tracker_cfg=config["tracker"]).start()
    try:
        assert _wait(lambda: sensor.read() is not None, 2.0), "no frame within 2 s"
        t0 = time.perf_counter()
        seqs, tracks = [], []
        while time.perf_counter() - t0 < 1.2:
            item = sensor.read()
            if not seqs or item[2] != seqs[-1]:
                seqs.append(item[2])
                frame, ts, seq = item
                assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
                tracks.append(sensor.track())
            time.sleep(0.002)
        assert all(b > a for a, b in zip(seqs, seqs[1:])) and len(seqs) >= 40
        for t in tracks:
            assert t.ok and t.depth.shape == (720, 1280) and t.depth.dtype == np.float32
            assert t.K_cam.shape == (3, 3) and t.clothes_mask.shape == (720, 1280)
            mapped = sorted(VISION_TO_MP.values())
            assert np.isfinite(t.joints3d[mapped]).all() and (t.joints3d[mapped, 2] > 0).all()
            assert t.facing == "front" and t.shoulder_width > 0
            x0, y0, x1, y1 = t.head_box
            assert x0 < t.joints[0, 0] < x1 and y0 < t.joints[0, 1] < y1
        last = tracks[-1]
        valid = last.depth[last.depth > 0]
        assert valid.min() == pytest.approx(2.2, abs=0.01) and valid.max() == pytest.approx(3.0, abs=0.01)
        assert last.depth[0, 0] == 0.0                              # invalid LiDAR pixels -> 0
        st = dict(sensor.stats)
        assert st["fps"] >= 50, st
        assert st["bytes_per_s"] > 0 and st["connected"]
        # ping/pong: the fake phone clock is perf_counter + 1000 s, so the offset must be recovered
        assert _wait(lambda: sensor.stats["latency_ms"] is not None, 1.0)
        assert sensor.stats["clock_offset_s"] == pytest.approx(phone.clock_offset_s, abs=0.01)
        assert -5.0 < sensor.stats["latency_ms"] < 50.0, sensor.stats
        assert phone.pings_answered >= 1 and sensor.last_conf.shape == (720, 1280)
        assert sensor.last_depth_raw.shape == (192, 256)
    finally:
        sensor.stop()
        phone.stop()
    assert not sensor.alive
    assert not [t for t in threading.enumerate() if t.name in ("iphone-sensor", "fake-phone")]


def test_back_view_head_box_without_face_joints(config: dict) -> None:
    """Vision drops every face joint when the person turns away: the head box must still cover the head."""
    phone = FakePhone(port=0)
    header, _ = FrameReader().feed(phone.frame_message(0.0, 0))[0]
    phone.stop()
    for i, n in enumerate(header["joint_names"]):
        if n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            header["joints2d"][i] = [math.nan, math.nan, math.nan]
    sensor = IPhoneSensor({**config["sensor"], "usbmux": False}, (1280, 720), tracker_cfg=config["tracker"])
    t = sensor._build_track(header, None, 0.0)
    true_nose = phone.project(phone.joints_arkit(0.0), phone.img_size)[0] * [1280 / 1920, 720 / 1440]
    x0, y0, x1, y1 = t.head_box
    assert t.ok and x0 < true_nose[0] < x1 and y0 < true_nose[1] < y1
    assert y1 < t.joints[L_SHOULDER, 1]
