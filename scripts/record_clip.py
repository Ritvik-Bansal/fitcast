#!/usr/bin/env python
"""Record a test clip from a camera for offline development.

  .venv/bin/python scripts/record_clip.py --source 0 --seconds 20 --out assets/test/full_body.mp4
Shows a preview window; press q to stop early. Records at the camera-space size from config.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitcast.capture import FrameSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default="0")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["capture"]
    src = FrameSource(cfg, a.source).start()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fps = src.native_fps if 1 < src.native_fps <= 120 else float(cfg["fps"])
    w, h = int(cfg["width"]), int(cfg["height"])
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0 = time.perf_counter()
    last = -1
    n = 0
    try:
        while time.perf_counter() - t0 < a.seconds:
            item = src.read()
            if item is None or item[2] == last:
                if not a.no_preview and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue
            frame, _, last = item
            writer.write(frame)
            n += 1
            if not a.no_preview:
                cv2.putText(frame, f"REC {time.perf_counter() - t0:4.1f}s  q=stop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("record_clip", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        writer.release()
        src.stop()
        cv2.destroyAllWindows()
    print(f"wrote {n} frames @ {fps:.1f} fps to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
