"""Fake FitCastSensor phone: a TCP server streaming synthetic FCS1 frames (tests, and `main.py --sensor iphone
--sensor-host 127.0.0.1` without hardware).

Scene: a wall plane at `wall_m` and a frontal stick-figure person at `person_m` (ARKit camera coords: +x right,
+y up, -z forward) swaying sideways and swinging the arms. Each frame carries the CLAUDE.md frame JSON plus
`depth` float16, `conf` uint8, `seg` uint8 and (every `rgb_every` frames) an `rgb` JPEG. Answers pings
(msg_type 3) by echoing them with `phone_ts`. The phone clock is perf_counter + `clock_offset_s`, so the
receiver's ping-based offset estimate is exercised.

    python -m fitcast.sensors.fake_phone [--port 7777] [--fps 60]
"""
from __future__ import annotations

import argparse
import logging
import math
import select
import socket
import threading
import time

import cv2
import numpy as np

from .iphone import VISION_JOINTS
from .protocol import MSG_FRAME, MSG_INFO, MSG_PING, TYPE_KEY, FrameReader, ProtocolError, encode_frame

log = logging.getLogger("fitcast.sensors.fake_phone")

# Frontal pose, person-relative metres (x toward the person's LEFT = camera +x, y up from the hip centre)
_POSE = {"nose": (0.0, 0.62), "left_eye": (0.035, 0.66), "right_eye": (-0.035, 0.66), "left_ear": (0.075, 0.64),
         "right_ear": (-0.075, 0.64), "neck": (0.0, 0.50), "left_shoulder": (0.20, 0.48),
         "right_shoulder": (-0.20, 0.48), "left_elbow": (0.24, 0.20), "right_elbow": (-0.24, 0.20),
         "left_wrist": (0.26, -0.05), "right_wrist": (-0.26, -0.05), "root": (0.0, 0.0), "left_hip": (0.12, 0.0),
         "right_hip": (-0.12, 0.0), "left_knee": (0.12, -0.45), "right_knee": (-0.12, -0.45),
         "left_ankle": (0.12, -0.88), "right_ankle": (-0.12, -0.88)}
_BONES = (("neck", "root"), ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
          ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
          ("right_elbow", "right_wrist"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
          ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("left_shoulder", "left_hip"),
          ("right_shoulder", "right_hip"))


class FakePhone:
    """Single-client FCS1 server. `start()` binds (port 0 = ephemeral; read `.port`) and serves in a thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, fps: float = 60.0,
                 img_size: tuple[int, int] = (1920, 1440), depth_size: tuple[int, int] = (256, 192),
                 rgb_size: tuple[int, int] = (640, 480), rgb_every: int = 1, jpeg_quality: int = 80,
                 focal_px: float = 1500.0, wall_m: float = 3.0, person_m: float = 2.2, hip_y_m: float = -0.1,
                 body_width_m: float = 0.16, clock_offset_s: float = 1000.0, send_seg: bool = True,
                 person_after_s: float = 0.0) -> None:
        self.host, self.fps = host, float(fps)
        self.person_after_s = float(person_after_s)      # empty scene (wall only, no joints) until then
        self._t_start = time.perf_counter()
        self.img_size, self.depth_size, self.rgb_size = img_size, depth_size, rgb_size
        self.rgb_every, self.jpeg_quality = max(1, int(rgb_every)), int(jpeg_quality)
        self.K = [focal_px, focal_px, img_size[0] / 2.0, img_size[1] / 2.0]
        self.wall_m, self.person_m, self.hip_y_m, self.body_width_m = wall_m, person_m, hip_y_m, body_width_m
        self.clock_offset_s, self.send_seg = clock_offset_s, send_seg
        self._srv = socket.create_server((host, port))
        self._srv.settimeout(0.1)
        self.port = self._srv.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="fake-phone", daemon=True)
        self.frames_sent = 0
        self.pings_answered = 0
        self._wall = np.full((depth_size[1], depth_size[0]), wall_m, np.float16)
        self._rgb_bg = np.full((rgb_size[1], rgb_size[0], 3), 90, np.uint8)

    def clock(self) -> float:
        """The fake phone's monotonic clock (deliberately not the Mac's perf_counter)."""
        return time.perf_counter() + self.clock_offset_s

    def start(self) -> "FakePhone":
        """Serve in a background thread."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and close the listening socket."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._srv.close()

    # -- scene ---------------------------------------------------------------
    def joints_arkit(self, t: float) -> np.ndarray:
        """(19, 3) joints in ARKit camera coords at time t (s), in VISION_JOINTS order."""
        sway = 0.3 * math.sin(2 * math.pi * 0.25 * t)
        swing = 0.12 * math.sin(2 * math.pi * 0.5 * t)
        out = np.zeros((len(VISION_JOINTS), 3))
        for i, n in enumerate(VISION_JOINTS):
            x, y = _POSE[n]
            if n.endswith(("elbow", "wrist")):
                x += math.copysign(swing * (2 if "wrist" in n else 1), x)
            out[i] = (x + sway, y + self.hip_y_m, -self.person_m)
        return out

    def project(self, p: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """ARKit camera points (N,3) -> pixels (N,2) in an image of `size` with the same FOV as the captured image."""
        fx, fy, cx, cy = self.K
        s = np.array([size[0] / self.img_size[0], size[1] / self.img_size[1]])
        z = -p[:, 2]
        uv = np.stack([fx * p[:, 0] / z + cx, cy - fy * p[:, 1] / z], axis=1)
        return uv * s

    def _silhouette(self, uv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Stick-figure person mask (uint8 0/255) at `size` from joint pixels in that image."""
        m = np.zeros((size[1], size[0]), np.uint8)
        idx = {n: i for i, n in enumerate(VISION_JOINTS)}
        px_per_m = self.K[0] * size[0] / self.img_size[0] / self.person_m
        thick = max(2, int(self.body_width_m * px_per_m))
        for a, b in _BONES:
            pa, pb = uv[idx[a]], uv[idx[b]]
            cv2.line(m, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 255, thick)
        torso = np.array([uv[idx[k]] for k in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")], np.int32)
        cv2.fillConvexPoly(m, torso, 255)
        head = uv[idx["nose"]]
        cv2.circle(m, (int(head[0]), int(head[1] - 0.03 * px_per_m)), int(0.11 * px_per_m), 255, -1)
        return m

    def frame_message(self, t: float, n: int) -> bytes:
        """Encode synthetic frame number `n` at scene time `t`."""
        j3 = self.joints_arkit(t)
        uv = self.project(j3, self.img_size)
        dw, dh = self.depth_size
        present = (time.perf_counter() - self._t_start) >= self.person_after_s
        seg = self._silhouette(self.project(j3, self.depth_size), self.depth_size) if present else np.zeros((dh, dw), np.uint8)
        depth = self._wall.copy()
        depth[seg > 0] = self.person_m
        conf = np.full((dh, dw), 2, np.uint8)
        depth[:4, :4], conf[:4, :4] = np.nan, 0            # a few invalid pixels, like real LiDAR
        blobs: dict = {"depth": depth, "conf": conf}
        if self.send_seg:
            blobs["seg"] = seg
        if n % self.rgb_every == 0:
            img = self._rgb_bg.copy()
            if present:
                img[self._silhouette(self.project(j3, self.rgb_size), self.rgb_size) > 0] = 235
            cv2.putText(img, f"fake phone #{n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            blobs["rgb"] = jpg.tobytes()
        now = self.clock()
        header = {"ts": now, "send_ts": now, "img_w": self.img_size[0], "img_h": self.img_size[1], "K": self.K,
                  "depth_w": dw, "depth_h": dh, "rgb_w": self.rgb_size[0], "rgb_h": self.rgb_size[1],
                  "joint_names": list(VISION_JOINTS),
                  "joints2d": [[float(u), float(v), 0.9] for u, v in uv] if present else [[None, None, 0.0]] * len(uv),
                  "joints3d": j3.tolist() if present else [[None, None, None]] * len(uv)}
        return encode_frame(header, blobs, MSG_FRAME)

    # -- serving ---------------------------------------------------------------
    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            log.info("fake phone: client %s connected", addr)
            try:
                self._stream(conn)
            except (OSError, ProtocolError) as e:
                log.info("fake phone: client gone (%s)", e)
            finally:
                conn.close()

    def _stream(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setblocking(True)
        conn.sendall(encode_frame({"device": "FakePhone", "fps": self.fps, "img_w": self.img_size[0],
                                   "img_h": self.img_size[1]}, msg_type=MSG_INFO))
        reader = FrameReader()
        period = 1.0 / self.fps
        t0 = next_t = time.perf_counter()
        n = 0
        while not self._stop.is_set():
            wait = max(0.0, next_t - time.perf_counter())
            readable, _, _ = select.select([conn], [], [], wait)
            if readable:
                data = conn.recv(65536)
                if not data:
                    return
                for h, _ in reader.feed(data):
                    if h[TYPE_KEY] == MSG_PING:
                        echo = {k: v for k, v in h.items() if k not in ("blobs", TYPE_KEY)}
                        conn.sendall(encode_frame({**echo, "phone_ts": self.clock()}, msg_type=MSG_PING))
                        self.pings_answered += 1
                continue
            conn.sendall(self.frame_message(time.perf_counter() - t0, n))
            n += 1
            self.frames_sent += 1
            next_t += period
            if next_t < time.perf_counter() - period:      # fell behind: do not burst
                next_t = time.perf_counter()


def main() -> None:
    """CLI: serve synthetic frames until Ctrl-C."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="default: config.yaml sensor.port")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--rgb-every", type=int, default=1, help="send the rgb JPEG every Nth frame")
    ap.add_argument("--rgb-size", type=int, nargs=2, default=(960, 720), metavar=("W", "H"))
    ap.add_argument("--no-seg", action="store_true", help="omit the person segmentation blob")
    ap.add_argument("--person-after", type=float, default=0.0, help="empty scene for this many seconds, then the person appears")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = a.port
    if port is None:
        import yaml
        from pathlib import Path
        with open(Path(__file__).resolve().parents[2] / "config.yaml") as f:
            port = int(yaml.safe_load(f)["sensor"]["port"])
    phone = FakePhone(a.host, port, a.fps, rgb_size=tuple(a.rgb_size), rgb_every=a.rgb_every,
                      send_seg=not a.no_seg, person_after_s=a.person_after).start()
    log.info("fake phone serving on %s:%d at %.0f fps (Ctrl-C to stop)", a.host, phone.port, a.fps)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        phone.stop()


if __name__ == "__main__":
    main()
