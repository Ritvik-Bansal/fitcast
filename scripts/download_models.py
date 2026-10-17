#!/usr/bin/env python
"""Download tracking models into assets/models/ (run once, needs network).

MediaPipe: pose_landmarker_full/lite (.task) + selfie_multiclass_256x256 (.tflite).
rtmlib (optional backend): YOLOX-tiny detector + RTMPose-s, shipped as zips of one .onnx.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "models"

MP = "https://storage.googleapis.com/mediapipe-models"
MM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
FILES = {
    "pose_landmarker_full.task": f"{MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_lite.task": f"{MP}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "selfie_multiclass_256x256.tflite": f"{MP}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
RTM_ZIPS = {
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx": f"{MM}/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx": f"{MM}/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        if r.status != 200:
            raise RuntimeError(f"{url} -> HTTP {r.status}")
        return r.read()


def main(with_rtm: bool) -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out = DEST / name
        if out.exists():
            print(f"ok   {name} ({out.stat().st_size} bytes)")
            continue
        print(f"get  {url}")
        out.write_bytes(fetch(url))
        print(f"ok   {name} ({out.stat().st_size} bytes)")
    if with_rtm:
        for name, url in RTM_ZIPS.items():
            out = DEST / name
            if out.exists():
                print(f"ok   {name}")
                continue
            print(f"get  {url}")
            z = zipfile.ZipFile(io.BytesIO(fetch(url)))
            member = next(m for m in z.namelist() if m.endswith(".onnx"))
            out.write_bytes(z.read(member))
            print(f"ok   {name} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(with_rtm="--rtm" in sys.argv))
