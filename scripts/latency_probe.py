#!/usr/bin/env python
"""Measure TRUE end-to-end latency: light change in the world -> light change on the projector.

Method 1, loopback-camera (automatic, no second phone):
  .venv/bin/python scripts/latency_probe.py --method loopback-camera --source name:iPhone
  .venv/bin/python scripts/latency_probe.py --method loopback-camera --sensor iphone
Aim the tracking camera at the projection (or at a white wall lit by it). The projector alternates full-frame
white/black; the camera's mean brightness in an ROI (default: centred 20% box) is sampled with its capture
timestamp.
  Phase 1 (open loop): white for --period s, black for --period s, --count cycles. Per edge:
      delay = camera-observed transition time - display hand-off time
            = projector display lag + camera exposure (~1/2..1 frame) + USB/Continuity/Wi-Fi transport
              + driver buffering + capture thread pickup.
    It does NOT include tracking/render compute: it is the LOWER BOUND of the system latency.
    --sensor iphone also reports the same edges on the ARFrame clock (receive time - capture_latency_ms), which
    removes transport and leaves ~projector lag + exposure.
  Phase 2 (round trip): the patch turns on, the script waits until the camera sees it, turns it off, waits
    until the camera sees black, ... Per edge it reports
      seen - handoff          = same terms as phase 1, but on single frames (no sub-frame interpolation)
      next_handoff - handoff  = full loop: the above + the script's reaction (sampler poll + GUI pump), i.e.
                                what main.py would achieve with zero tracking/render compute.

Method 2, video (slow-motion video from a second phone, independent of the tracking camera):
  a) .venv/bin/python scripts/latency_probe.py --method video        # flashes + writes out/flash_log.json
     Film at 240 fps BEFORE starting, framing BOTH the projection and the Mac screen's "FitCast Debug" window
     (it shows the same flashing patch + a frame counter).
  b) .venv/bin/python scripts/latency_probe.py --method video clip.mov --roi x,y,w,h --ref-roi x,y,w,h \
         --log out/flash_log.json [--video-fps 240]
     --roi = the projection in video pixels, --ref-roi = the Mac-screen patch. The video clock is aligned to the
     log by the FIRST dark->bright transition of the Mac-screen ROI; delay_i = projection_on_i - (log_on_i +
     offset). Assumption: the Mac screen shows the patch with a constant (small, unmeasured) lag, so the delays are
     projector lag relative to the Mac screen. Uncertainty +-1 video frame (+-4.2 ms at 240 fps). Without
     --ref-roi the projection's own first transition is used: the first delay is 0 by construction and only the
     variation (jitter) is meaningful.

--estimate-from-main prints end_to_end ~= camera_exposure_and_transport + main_latency + projector_lag and fills
the measured terms (--main-latency-ms from main.py's "last latency", --from-json a loopback result).

Every run writes out/latency_<timestamp>.json (raw times + summary). Tunables: config.yaml `latency:` (optional).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fitcast import latency as L  # noqa: E402

log = logging.getLogger("fitcast.latency_probe")

FORMULA = "end_to_end ~= camera_exposure_and_transport + main_latency + projector_lag"
MONITOR_SIZE = (480, 270)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line (see the module docstring for the methods)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--method", choices=["loopback-camera", "video"],
                    help="loopback-camera: tracking camera watches the projection; video: slow-mo video analysis")
    ap.add_argument("video", nargs="?", help="--method video: slow-motion clip to analyse (omit to run the flashes "
                                             "and write --log)")
    ap.add_argument("--source", help="camera index, 'auto' or 'name:<substring>' as in main.py (default: config)")
    ap.add_argument("--sensor", choices=["webcam", "iphone"], help="webcam (default) or the iPhone LiDAR sensor app")
    ap.add_argument("--sensor-host", help="iphone sensor: connect to this IP instead of USB (usbmux)")
    ap.add_argument("--projector-x", type=int, help="override projector window x origin")
    ap.add_argument("--projector-y", type=int, help="override projector window y origin")
    ap.add_argument("--period", type=float, help="seconds white, then the same black (default latency.period_s)")
    ap.add_argument("--count", type=int, help="flash cycles per phase (default latency.count)")
    ap.add_argument("--roi", help="x,y,w,h: loopback = camera-space px (default centred latency.roi_frac box); "
                                  "video = the projection in video px")
    ap.add_argument("--ref-roi", help="video: x,y,w,h of the Mac-screen patch (clock alignment)")
    ap.add_argument("--log", default=None, help="flash log json (default out/flash_log.json)")
    ap.add_argument("--video-fps", type=float, help="video: capture fps when the container reports the playback rate")
    ap.add_argument("--max-delay", type=float, help="seconds; later observations are not matched (latency.max_delay_s)")
    ap.add_argument("--no-aim", action="store_true", help="loopback: skip the aiming preview")
    ap.add_argument("--skip-roundtrip", action="store_true", help="loopback: phase 1 only")
    ap.add_argument("--out", help="result json path (default out/latency_<timestamp>.json)")
    ap.add_argument("--estimate-from-main", action="store_true",
                    help="print the end-to-end formula and fill the measured terms")
    ap.add_argument("--main-latency-ms", type=float, help="main.py 'last latency' (capture -> display hand-off)")
    ap.add_argument("--from-json", help="a previous out/latency_*.json loopback result for --estimate-from-main")
    ap.add_argument("--headless", action="store_true", help="no windows (only meaningful with --simulate)")
    ap.add_argument("--simulate", type=float, metavar="MS",
                    help="loopback without hardware: a fake camera sees the patch MS ms after hand-off")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def load_config(path: str) -> dict:
    """config.yaml as a dict (exits with a message when missing)."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"config not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def default_out(lcfg: dict, name: str | None = None) -> Path:
    """out/latency_<timestamp>.json (or out/<name>)."""
    d = Path(lcfg["out_dir"])
    if not d.is_absolute():
        d = ROOT / d
    return d / (name or f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json")


def write_json(path: Path, data: dict) -> None:
    """Write `data` (numpy arrays -> lists, NaN -> null)."""
    def conv(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(data), indent=1))
    print(f"wrote {path}")


def fmt(s: dict) -> str:
    """One-line ms summary."""
    if not s.get("n"):
        return f"no matches (0/{s.get('n_triggers', 0)})"
    return (f"mean {s['mean_ms']:.1f}  median {s['median_ms']:.1f}  p95 {s['p95_ms']:.1f}  "
            f"min {s['min_ms']:.1f}  max {s['max_ms']:.1f} ms  (n {s['n']}/{s['n_triggers']})")


# ---------------------------------------------------------------------------------------------
# Patch display (projector + Mac-screen monitor)
# ---------------------------------------------------------------------------------------------
class PatchShower:
    """Puts the full-frame white/black patch on the projector and a copy + counter on the Mac-screen monitor
    window. `show()` returns the hand-off time = perf_counter right after Display.show_projector() returned
    (with the cocoa sink that is the enqueue to the commit thread; the window-server commit follows a few ms later
    and is not timestamped). History of (handoff, on) feeds the simulated camera."""

    def __init__(self, display: Any, proj_size: tuple[int, int]) -> None:
        self.display = display
        pw, ph = proj_size
        self.white = np.full((ph, pw, 3), 255, np.uint8)
        self.black = np.zeros((ph, pw, 3), np.uint8)
        self.mon = np.zeros((MONITOR_SIZE[1], MONITOR_SIZE[0], 3), np.uint8)
        self.history: list[tuple[float, bool]] = []
        self.counter = 0
        self.status = ""
        self.quit = False
        self._lock = threading.Lock()

    def show(self, on: bool) -> tuple[float, float]:
        """Patch on/off. Returns (projector hand-off time, Mac-monitor shown-and-pumped time)."""
        self.display.show_projector(self.white if on else self.black)
        t = time.perf_counter()
        with self._lock:
            self.history.append((t, on))
        self.counter += 1
        self.draw_monitor(on)
        t_mon = time.perf_counter()
        self.pump()
        return t, t_mon

    def draw_monitor(self, on: bool, extra: np.ndarray | None = None) -> None:
        """Mac-screen window: the same patch (reference for the video method) + frame counter + status."""
        import cv2
        if extra is not None:
            img = cv2.resize(extra, MONITOR_SIZE)
        else:
            self.mon[:] = 255 if on else 0
            img = self.mon
        c = (0, 0, 0) if on and extra is None else (255, 255, 255)
        cv2.putText(img, f"#{self.counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 2, cv2.LINE_AA)
        if self.status:
            cv2.putText(img, self.status, (10, MONITOR_SIZE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        self.display.show_debug(img)

    def pump(self, wait_ms: int = 1) -> int:
        """Run the GUI event loop once; q / Esc sets `quit`."""
        k = self.display.key(wait_ms)
        if k in (ord("q"), 27):
            self.quit = True
        return k

    def on_fraction(self, a: float, b: float) -> float:
        """Fraction of [a, b] during which the patch was on (simulated camera exposure)."""
        if b <= a:
            return 0.0
        with self._lock:
            h = list(self.history)
        state, t_prev, acc = False, a, 0.0
        for t, on in h:
            if t <= a:
                state = on
                continue
            if t >= b:
                break
            acc += (t - t_prev) if state else 0.0
            state, t_prev = on, t
        acc += (b - t_prev) if state else 0.0
        return acc / (b - a)


def make_display(cfg: dict, args: argparse.Namespace) -> tuple[Any, tuple[int, int]]:
    """fitcast.display.Display placed like main.py does: fullscreen projector window on the second display."""
    from fitcast.display import Display
    proj = (int(cfg["calibration"]["projector_width"]), int(cfg["calibration"]["projector_height"]))
    display = Display(cfg["display"], proj, (args.projector_x, args.projector_y), headless=args.headless)
    if not args.headless:
        display.open_projector(fullscreen=display.second is not None)
        display.open_debug()
    return display, proj


# ---------------------------------------------------------------------------------------------
# Camera side
# ---------------------------------------------------------------------------------------------
class SimulatedCamera:
    """Fake camera for --simulate: frames at `fps`, each integrating the patch over its exposure window
    [t - delay - 1/fps, t - delay] (so a detected edge lands at delay + half a frame after hand-off)."""

    def __init__(self, shower: PatchShower, delay_s: float, fps: float = 60.0, size: tuple[int, int] = (160, 90),
                 noise: float = 2.0) -> None:
        self.shower, self.delay, self.T, self.size, self.noise = shower, delay_s, 1.0 / fps, size, noise
        self.stats: dict = {}
        self._latest: tuple[np.ndarray, float, int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sim-camera", daemon=True)
        self._rng = np.random.default_rng(0)

    def start(self) -> "SimulatedCamera":
        """Start producing frames."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the frame thread."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def read(self) -> tuple[np.ndarray, float, int] | None:
        """Latest (frame, timestamp, seq) like FrameSource.read()."""
        return self._latest

    def _run(self) -> None:
        seq, nxt = 0, time.perf_counter()
        w, h = self.size
        while not self._stop.is_set():
            nxt += self.T
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            t = time.perf_counter()
            f = self.shower.on_fraction(t - self.delay - self.T, t - self.delay)
            v = 20.0 + 200.0 * f + self._rng.normal(0.0, self.noise)
            seq += 1
            self._latest = (np.full((h, w, 3), np.clip(v, 0, 255), np.uint8), t, seq)


class Sampler:
    """Background thread polling `source.read()` every `poll_s`; records (timestamp, ROI brightness,
    capture_latency_ms or NaN) once per new frame. The main thread's GUI pump (~12 ms) would miss frames."""

    def __init__(self, source: Any, poll_s: float, latency_fn: Callable[[], float | None] | None = None) -> None:
        self.source, self.poll_s, self.latency_fn = source, float(poll_s), latency_fn
        self.roi: tuple[int, int, int, int] | None = None
        self.frame: np.ndarray | None = None
        self.t: list[float] = []
        self.b: list[float] = []
        self.cl: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="latency-sampler", daemon=True)

    def start(self) -> "Sampler":
        """Start polling."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            r = self.source.read()
            if r is not None and r[2] != last:
                frame, ts, last = r
                self.frame = frame
                if self.roi is not None:
                    b = L.frame_brightness(frame, self.roi)
                    cl = self.latency_fn() if self.latency_fn else None
                    with self._lock:
                        self.t.append(float(ts))
                        self.b.append(b)
                        self.cl.append(float(cl) if cl is not None else float("nan"))
            time.sleep(self.poll_s)

    def count(self) -> int:
        """Samples recorded so far."""
        with self._lock:
            return len(self.t)

    def since(self, i: int) -> tuple[list[float], list[float]]:
        """(times, brightness) of samples i.. (for online round-trip detection)."""
        with self._lock:
            return self.t[i:], self.b[i:]

    def arrays(self, i0: int = 0, i1: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(times, brightness, capture_latency_ms) of samples i0..i1 as float64 arrays."""
        with self._lock:
            sl = slice(i0, i1)
            return (np.asarray(self.t[sl], np.float64), np.asarray(self.b[sl], np.float64),
                    np.asarray(self.cl[sl], np.float64))


def open_camera(cfg: dict, args: argparse.Namespace, shower: PatchShower) -> tuple[Any, Callable | None, str]:
    """(source with .read()/.stop(), capture-latency getter or None, description)."""
    size = (int(cfg["capture"]["width"]), int(cfg["capture"]["height"]))
    if args.simulate is not None:
        return SimulatedCamera(shower, args.simulate / 1e3).start(), None, f"simulated camera, {args.simulate:.0f} ms"
    kind = args.sensor or (cfg.get("sensor") or {}).get("kind", "webcam")
    if kind == "iphone":
        from fitcast.sensors.iphone import IPhoneSensor
        scfg = dict(cfg["sensor"])
        if args.sensor_host:
            scfg["usbmux"], scfg["host"] = False, args.sensor_host
        src = IPhoneSensor(scfg, size, tracker_cfg=cfg["tracker"]).start()
        return src, (lambda: src.stats.get("capture_latency_ms")), "iphone sensor"
    from fitcast.capture import FrameSource
    src = FrameSource(dict(cfg["capture"]), args.source).start()
    if src.is_file:
        src.stop()
        raise RuntimeError("loopback-camera needs a live camera (--source index / name:... / auto), not a video file")
    return src, None, f"camera {src.source!r}"


def wait_first_frame(source: Any, shower: PatchShower, timeout_s: float = 15.0) -> np.ndarray:
    """Block (pumping the GUI) until the source delivers a frame."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = source.read()
        if r is not None:
            return r[0]
        shower.pump(10)
        time.sleep(0.01)
    raise RuntimeError("no camera frame within %.0f s. Check the camera / Camera permission (main.py --list-cameras) "
                       "or that the FitCastSensor app is running and connected." % timeout_s)


def aim(sampler: Sampler, shower: PatchShower, roi: tuple[int, int, int, int]) -> None:
    """Project white and show the camera preview with the ROI box until space/enter (q aborts)."""
    import cv2
    print("AIM: point the camera at the projection (or a white wall lit by it) so the green ROI box is inside the\n"
          "     projected area. Lock exposure if you can (auto exposure fights full-frame flashes).\n"
          "     Press SPACE or ENTER in the 'FitCast Debug' window to start, q to abort.")
    shower.display.show_projector(shower.white)
    x, y, w, h = roi
    while not shower.quit:
        f = sampler.frame
        if f is not None:
            vis = f.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            shower.status = f"ROI mean {L.frame_brightness(f, roi):.0f}   SPACE = start"
            shower.draw_monitor(True, vis)
        k = shower.pump(15)
        if k in (32, 13, 10):
            break
    shower.status = ""


def settle(shower: PatchShower, seconds: float) -> None:
    """Black patch for `seconds` while pumping the GUI."""
    shower.show(False)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end and not shower.quit:
        shower.pump(10)


def run_open_loop(shower: PatchShower, period_s: float, count: int) -> tuple[list[float], list[float], dict]:
    """Phase 1 / video flashes: follow flash_schedule; returns hand-off times (on, off) and Mac-monitor times."""
    sched = L.flash_schedule(period_s, count, t0=time.perf_counter() + 0.05)
    t_on: list[float] = []
    t_off: list[float] = []
    mon: dict = {"on": [], "off": []}
    events = sorted([(a, True) for a, _ in sched] + [(b, False) for _, b in sched])
    for i, (t_at, on) in enumerate(events):
        while not shower.quit:
            dt = t_at - time.perf_counter()
            if dt <= 0:
                break
            if dt > 0.02:
                shower.pump(5)
            else:
                time.sleep(max(0.0, dt - 0.0005))
        if shower.quit:
            break
        shower.status = f"cycle {i // 2 + 1}/{count}"
        t, tm = shower.show(on)
        (t_on if on else t_off).append(t)
        mon["on" if on else "off"].append(tm)
    shower.show(False)
    return t_on, t_off, mon


def run_round_trip(shower: PatchShower, sampler: Sampler, count: int, level: float, hyst: float,
                   timeout_s: float) -> list[dict]:
    """Phase 2: toggle the patch as soon as the camera sees the previous toggle. One dict per edge."""
    edges: list[dict] = []
    on = True
    i0 = sampler.count()
    t_h, _ = shower.show(on)
    while len(edges) < 2 * count and not shower.quit:
        ts, bs = sampler.since(i0)
        seen = None
        for k, (t, b) in enumerate(zip(ts, bs)):
            if t > t_h and ((on and b >= level + hyst) or (not on and b <= level - hyst)):
                seen = t
                i0 += k + 1
                break
        else:
            i0 += len(ts)
        now = time.perf_counter()
        if seen is None and now - t_h < timeout_s:
            if now - shower.history[-1][0] > 0.03:
                shower.pump(1)
            else:
                time.sleep(0.0005)
            continue
        shower.status = f"round trip {len(edges) // 2 + 1}/{count}"
        t_next, _ = shower.show(not on)
        edges.append({"on": on, "handoff": t_h, "seen": seen, "next_handoff": t_next,
                      "timeout": seen is None})
        on, t_h = not on, t_next
    shower.show(False)
    return edges


def analyse_phase1(t: np.ndarray, b: np.ndarray, cl: np.ndarray, t_on: list[float], t_off: list[float],
                   lcfg: dict) -> dict:
    """Edges of the sampled signal matched to the hand-off times."""
    kw = {"hysteresis_frac": float(lcfg["hysteresis_frac"]), "pct": float(lcfg["robust_pct"])}
    lo, hi = L.robust_range(b, float(lcfg["robust_pct"]))
    rises = L.detect_transitions(b, t, edge="rising", **kw)
    falls = L.detect_transitions(b, t, edge="falling", **kw)
    md = float(lcfg["max_delay_s"])
    d_on, s_on = L.match_delays(np.asarray(t_on), rises, md)
    d_off, s_off = L.match_delays(np.asarray(t_off), falls, md)
    res: dict[str, Any] = {
        "robust_low": lo, "robust_high": hi, "contrast": hi - lo, "rises": rises, "falls": falls,
        "delays_on_s": d_on, "delays_off_s": d_off, "summary_on": s_on, "summary_off": s_off,
        "summary": L.summarize(np.concatenate([d_on, d_off]), n_triggers=len(t_on) + len(t_off))}
    if np.isfinite(cl).any():          # iphone: same edges on the ARFrame clock (receive - capture latency)
        ok = np.isfinite(cl)
        tc, bc = t[ok] - cl[ok] / 1e3, b[ok]
        dc_on, _ = L.match_delays(np.asarray(t_on), L.detect_transitions(bc, tc, edge="rising", **kw), md)
        dc_off, _ = L.match_delays(np.asarray(t_off), L.detect_transitions(bc, tc, edge="falling", **kw), md)
        res["arframe_clock"] = {"delays_on_s": dc_on, "delays_off_s": dc_off,
                                "summary": L.summarize(np.concatenate([dc_on, dc_off]), len(t_on) + len(t_off)),
                                "capture_latency_ms_median": float(np.nanmedian(cl))}
    return res


def run_loopback(cfg: dict, lcfg: dict, args: argparse.Namespace) -> int:
    """Method 1: phase 1 (open loop) + phase 2 (round trip) with the tracking camera watching the projection."""
    display, proj = make_display(cfg, args)
    shower = PatchShower(display, proj)
    source = None
    sampler = None
    try:
        source, lat_fn, desc = open_camera(cfg, args, shower)
        sampler = Sampler(source, float(lcfg["poll_s"]), lat_fn).start()
        frame = wait_first_frame(source, shower)
        H, W = frame.shape[:2]
        roi = L.parse_roi(args.roi) if args.roi else L.centre_roi(W, H, float(lcfg["roi_frac"]))
        print(f"camera: {desc}, frames {W}x{H}, ROI {roi}; projector {proj[0]}x{proj[1]} "
              f"({'second display' if display.second else 'NO second display: main screen'})")
        if not (args.no_aim or args.headless or args.simulate is not None):
            aim(sampler, shower, roi)
        if shower.quit:
            print("aborted")
            return 1
        sampler.roi = roi
        period, count = float(lcfg["period_s"]), int(lcfg["count"])
        print(f"PHASE 1 (open loop): {count} cycles x {2 * period:.2f} s. Keep the camera and the room still.")
        settle(shower, float(lcfg["settle_s"]))
        i0 = sampler.count()
        t_on, t_off, mon = run_open_loop(shower, period, count)
        time.sleep(float(lcfg["max_delay_s"]))
        t, b, cl = sampler.arrays(i0)
        p1 = analyse_phase1(t, b, cl, t_on, t_off, lcfg)
        result: dict[str, Any] = {
            "method": "loopback-camera", "source": desc, "roi": roi, "camera_size": [W, H], "config": lcfg,
            "clock": "time.perf_counter seconds", "handoff_on": t_on, "handoff_off": t_off, "monitor_shown": mon,
            "samples": {"t": t, "brightness": b, "capture_latency_ms": cl}, "phase1": p1,
            "phase1_includes": "projector display lag + camera exposure + transport + driver buffering + capture "
                               "pickup (NOT tracking/render compute): lower bound of the system latency"}
        print(f"\nPHASE 1 (hand-off -> camera sees it), {len(t)} camera samples, contrast {p1['contrast']:.0f}")
        print(f"  dark->bright: {fmt(p1['summary_on'])}")
        print(f"  bright->dark: {fmt(p1['summary_off'])}")
        print(f"  all edges   : {fmt(p1['summary'])}")
        if "arframe_clock" in p1:
            print(f"  ARFrame clock (removes transport ~{p1['arframe_clock']['capture_latency_ms_median']:.0f} ms): "
                  f"{fmt(p1['arframe_clock']['summary'])}")
        code = 0
        if p1["contrast"] < float(lcfg["min_contrast"]):
            print(f"\nFIX: the camera barely sees the flash (contrast {p1['contrast']:.1f} < {lcfg['min_contrast']}). "
                  "Aim the camera at the projection, move --roi inside the projected area, dim the room, lock "
                  "exposure. Skipping the round trip.")
            code = 2
        elif not args.skip_roundtrip and not shower.quit:
            lo, hi = p1["robust_low"], p1["robust_high"]
            level, hyst = 0.5 * (lo + hi), float(lcfg["hysteresis_frac"]) * (hi - lo)
            print(f"\nPHASE 2 (round trip): {count} cycles, threshold {level:.0f} +- {hyst:.0f}")
            settle(shower, float(lcfg["settle_s"]) / 2)
            edges = run_round_trip(shower, sampler, count, level, hyst, float(lcfg["loop_timeout_s"]))
            seen = np.array([e["seen"] - e["handoff"] for e in edges if not e["timeout"]], np.float64)
            loop = np.array([e["next_handoff"] - e["handoff"] for e in edges if not e["timeout"]], np.float64)
            react = loop - seen
            result["phase2"] = {"edges": edges, "summary_seen": L.summarize(seen, len(edges)),
                                "summary_loop": L.summarize(loop, len(edges)),
                                "summary_reaction": L.summarize(react, len(edges)),
                                "timeouts": int(sum(e["timeout"] for e in edges))}
            result["phase2_includes"] = {
                "seen": "hand-off -> first camera frame past the threshold (phase-1 terms, frame-quantised)",
                "loop": "hand-off -> next hand-off = seen + script reaction (sampler poll + GUI pump): the loop "
                        "main.py would close with zero tracking/render compute",
                "reaction": "loop - seen"}
            p2 = result["phase2"]
            print(f"  seen - handoff (projector+camera+transport): {fmt(p2['summary_seen'])}")
            print(f"  full loop (+ script reaction)              : {fmt(p2['summary_loop'])}")
            print(f"  reaction only                              : {fmt(p2['summary_reaction'])}")
            if p2["timeouts"]:
                print(f"  {p2['timeouts']} edges timed out (> {lcfg['loop_timeout_s']} s)")
        cap_ms = (p1["arframe_clock"]["capture_latency_ms_median"] if "arframe_clock" in p1 else None)
        result["estimate"] = print_estimate(p1["summary"].get("median_ms"), args.main_latency_ms, cap_ms)
        write_json(Path(args.out) if args.out else default_out(lcfg), result)
        return code
    finally:
        if sampler is not None:
            sampler.stop()
        if source is not None:
            source.stop()
        display.close()


# ---------------------------------------------------------------------------------------------
# Method 2: slow-motion video
# ---------------------------------------------------------------------------------------------
VIDEO_HOWTO = """HOW TO (video method):
 1. Second phone: Camera > Slo-Mo, 240 fps. Frame BOTH the projection and the Mac screen's 'FitCast Debug'
    window (it flashes with the projector and shows a frame counter). Steady the phone; keep both in focus.
 2. Start filming, THEN run:  latency_probe.py --method video        (writes out/flash_log.json)
 3. Stop filming after the last flash. AirDrop the clip to the Mac.
 4. Find the two ROIs in video pixels (e.g. open a frame in Preview / ffmpeg -i clip.mov -frames:v 1 f.png):
      latency_probe.py --method video clip.mov --roi x,y,w,h --ref-roi x,y,w,h --log out/flash_log.json
    If the clip's fps metadata is the playback rate (30) rather than 240, add --video-fps 240.
 Result = projector lag relative to the Mac screen, +-1 video frame (+-4.2 ms at 240 fps)."""


def run_flash_log(cfg: dict, lcfg: dict, args: argparse.Namespace) -> int:
    """Method 2a: flash projector + Mac monitor on the schedule and write the hand-off log."""
    print(VIDEO_HOWTO)
    display, proj = make_display(cfg, args)
    shower = PatchShower(display, proj)
    try:
        period, count = float(lcfg["period_s"]), int(lcfg["count"])
        print(f"flashing {count} cycles x {2 * period:.2f} s (q aborts)")
        settle(shower, float(lcfg["settle_s"]))
        t_on, t_off, mon = run_open_loop(shower, period, count)
        data = {"clock": "time.perf_counter seconds", "period_s": period, "count": count, "t_on": t_on,
                "t_off": t_off, "monitor_shown": mon, "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "projector_size": list(proj), "second_display": bool(display.second)}
        write_json(Path(args.log) if args.log else default_out(lcfg, "flash_log.json"), data)
        write_json(Path(args.out) if args.out else default_out(lcfg),
                   {"method": "video-flash-log", "config": lcfg, **data})
        return 0 if not shower.quit else 1
    finally:
        display.close()


def analyse_video(lcfg: dict, args: argparse.Namespace) -> int:
    """Method 2b: transitions in the projection ROI vs the flash log, clocks aligned by the first transition."""
    if not args.roi:
        sys.exit("--method video FILE needs --roi x,y,w,h (the projection in video pixels)\n\n" + VIDEO_HOWTO)
    log_path = Path(args.log) if args.log else default_out(lcfg, "flash_log.json")
    if not log_path.exists():
        sys.exit(f"flash log not found: {log_path} (run --method video without a FILE first)")
    flog = json.loads(log_path.read_text())
    log_on, log_off = np.asarray(flog["t_on"], np.float64), np.asarray(flog["t_off"], np.float64)
    period = float(flog.get("period_s", lcfg["period_s"]))
    roi = L.parse_roi(args.roi)
    t, b = L.roi_brightness(args.video, roi, fps=args.video_fps)
    fps = (1.0 / (t[1] - t[0])) if t.size > 1 else float("nan")
    kw = {"hysteresis_frac": float(lcfg["hysteresis_frac"]), "pct": float(lcfg["robust_pct"])}
    p_on = L.detect_transitions(b, t, edge="rising", **kw)
    p_off = L.detect_transitions(b, t, edge="falling", **kw)
    md = float(lcfg["max_delay_s"])
    tol = -min(0.25 * period, md / 2)           # aligned clocks: tolerate a projection edge slightly BEFORE the ref
    result: dict[str, Any] = {"method": "video", "video": str(args.video), "video_fps": fps, "roi": roi,
                              "log": str(log_path), "config": lcfg, "samples": {"t": t, "brightness": b},
                              "projection_on": p_on, "projection_off": p_off,
                              "uncertainty_ms": 1e3 / fps if fps == fps else None}
    print(f"video {args.video}: {t.size} frames @ {fps:.1f} fps, projection ROI {roi}: "
          f"{p_on.size} rising / {p_off.size} falling edges (log: {log_on.size} flashes)")
    scale = L.time_scale_ratio(p_on, 2 * period)
    result["time_scale_ratio"] = scale
    if scale is not None and abs(scale - 1.0) > 0.05:
        print(f"WARNING: video edges are {scale:.2f}x the logged spacing. The fps metadata is probably the playback "
              f"rate; rerun with --video-fps {fps * scale:.0f}.")
    if args.ref_roi:
        rroi = L.parse_roi(args.ref_roi)
        _, rb = L.roi_brightness(args.video, rroi, fps=args.video_fps)
        r_on = L.detect_transitions(rb, t, edge="rising", **kw)
        r_off = L.detect_transitions(rb, t, edge="falling", **kw)
        offset = L.first_transition_offset(r_on, log_on)
        result.update(ref_roi=rroi, ref_on=r_on, ref_off=r_off, aligned_by="first Mac-screen (ref ROI) transition")
        d_dir_on, s_dir_on = L.match_delays(r_on, p_on, md, tol)
        d_dir_off, s_dir_off = L.match_delays(r_off, p_off, md, tol)
        result["direct_vs_ref"] = {"delays_on_s": d_dir_on, "delays_off_s": d_dir_off, "summary_on": s_dir_on,
                                   "summary_off": s_dir_off}
    else:
        offset = L.first_transition_offset(p_on, log_on)
        result["aligned_by"] = "first projection transition (no --ref-roi): first delay is 0 by construction"
        print("WARNING: no --ref-roi: aligned by the projection's own first edge, so these numbers are JITTER "
              "around the first flash, not absolute latency. Add --ref-roi on the Mac-screen patch.")
    if offset is None:
        print("FIX: no dark->bright transition found. Check the ROIs (video pixels) and that the clip covers the run.")
        write_json(Path(args.out) if args.out else default_out(lcfg), result)
        return 2
    d_on, s_on = L.match_delays(log_on + offset, p_on, md, tol)
    d_off, s_off = L.match_delays(log_off + offset, p_off, md, tol)
    result.update(offset_s=offset, delays_on_s=d_on, delays_off_s=d_off, summary_on=s_on, summary_off=s_off,
                  summary=L.summarize(np.concatenate([d_on, d_off]), log_on.size + log_off.size))
    unc = result["uncertainty_ms"]
    print(f"aligned: {result['aligned_by']} (offset {offset:.3f} s); +-1 video frame = +-{unc:.1f} ms")
    print(f"  dark->bright vs log: {fmt(s_on)}")
    print(f"  bright->dark vs log: {fmt(s_off)}")
    print(f"  all edges          : {fmt(result['summary'])}")
    if "direct_vs_ref" in result:
        print(f"  direct projection - Mac screen, per flash: on {fmt(result['direct_vs_ref']['summary_on'])}")
        print(f"                                             off {fmt(result['direct_vs_ref']['summary_off'])}")
        print("  (= projector lag minus the Mac screen's own lag, which this method cannot see)")
    write_json(Path(args.out) if args.out else default_out(lcfg), result)
    return 0


# ---------------------------------------------------------------------------------------------
# Estimate from main.py numbers
# ---------------------------------------------------------------------------------------------
def print_estimate(loopback_ms: float | None, main_ms: float | None, capture_ms: float | None = None) -> dict:
    """Print FORMULA with the measured terms filled in; returns them as a dict."""
    print(f"\n{FORMULA}")
    print("  camera_exposure_and_transport + projector_lag = loopback phase-1 median "
          f"({'%.1f ms' % loopback_ms if loopback_ms is not None else 'not measured: run --method loopback-camera'})")
    print("      (measured together; they only split on the iPhone path: transport ~ capture_latency_ms"
          f"{' = %.1f ms' % capture_ms if capture_ms is not None else ''}, the ARFrame-clock number ~ projector lag"
          " + exposure)")
    print("  main_latency = main.py 'last latency' / debug overlay 'latency' (capture timestamp -> display hand-off) "
          f"({'%.1f ms' % main_ms if main_ms is not None else 'pass --main-latency-ms'})")
    print("      IPhoneSensor stats capture_latency_ms (ARFrame -> Mac receive) is the transport term; main.py's "
          "latency starts at receive, so it does not double count")
    total = loopback_ms + main_ms if loopback_ms is not None and main_ms is not None else None
    if total is not None:
        print(f"  => end_to_end ~= {loopback_ms:.1f} + {main_ms:.1f} = {total:.1f} ms")
    return {"formula": FORMULA, "loopback_phase1_median_ms": loopback_ms, "main_latency_ms": main_ms,
            "capture_latency_ms": capture_ms, "end_to_end_ms": total}


def estimate_standalone(args: argparse.Namespace) -> int:
    """--estimate-from-main without a measurement run (fills from --from-json / --main-latency-ms)."""
    lb = cap = None
    if args.from_json:
        d = json.loads(Path(args.from_json).read_text())
        p1 = d.get("phase1") or {}
        lb = (p1.get("summary") or {}).get("median_ms")
        cap = (p1.get("arframe_clock") or {}).get("capture_latency_ms_median")
    print_estimate(lb, args.main_latency_ms, cap)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    lcfg = L.latency_cfg(cfg)
    for k, a in (("period_s", args.period), ("count", args.count), ("max_delay_s", args.max_delay)):
        if a is not None:
            lcfg[k] = a
    if args.method is None:
        if args.estimate_from_main:
            return estimate_standalone(args)
        print("choose --method loopback-camera | video, or --estimate-from-main (see --help)")
        return 1
    if args.headless and args.simulate is None and not (args.method == "video" and args.video):
        sys.exit("--headless only makes sense with --simulate (nothing would be projected)")
    try:
        if args.method == "loopback-camera":
            return run_loopback(cfg, lcfg, args)
        if args.video:
            return analyse_video(lcfg, args)
        return run_flash_log(cfg, lcfg, args)
    except RuntimeError as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
