#!/usr/bin/env bash
# Fitcast setup: Python 3.11 venv, dependencies, MediaPipe model files, placeholder garments.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.11 not found. Install it (brew install python@3.11) or run: PYTHON=/path/to/python3.11 ./setup.sh"
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
  echo "$PY is $("$PY" --version 2>&1); this project needs Python 3.11 (the pinned wheels are built for it)."
  echo "Install it (brew install python@3.11) or run: PYTHON=/path/to/python3.11 ./setup.sh"
  exit 1
fi
echo "==> using $PY ($("$PY" --version))"

if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
fi
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "==> dependencies installed"

mkdir -p models
BASE=https://storage.googleapis.com/mediapipe-models
fetch() {
  local url="$1" out="models/$(basename "$1")"
  # every model is >200 KB; anything smaller is a truncated download or an error page
  if [ -f "$out" ] && [ "$(wc -c < "$out")" -gt 200000 ]; then echo "    have $out"; return; fi
  echo "    downloading $out"
  curl -fsSL --retry 3 -o "$out.part" "$url"
  if [ "$(wc -c < "$out.part")" -le 200000 ]; then
    echo "    $out looks truncated; delete models/ and rerun"; rm -f "$out.part"; exit 1
  fi
  mv "$out.part" "$out"
}
echo "==> models"
fetch "$BASE/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
fetch "$BASE/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
fetch "$BASE/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
fetch "$BASE/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

if [ ! -d assets/garments ] || [ -z "$(ls -A assets/garments 2>/dev/null)" ]; then
  echo "==> placeholder garments"
  .venv/bin/python tools/make_placeholders.py
fi

.venv/bin/python - <<'PY'
import cv2, mediapipe, numpy, scipy, yaml  # noqa: F401
print(f"==> import check ok (mediapipe {mediapipe.__version__}, opencv {cv2.__version__})")
PY
# pygame separately: it bundles its own SDL2, as does OpenCV, and the two warn when loaded together
.venv/bin/python -c "import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'; import pygame; print(f'==> pygame {pygame.version.ver} ok')"
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run the mirror:       python -m tryon"
echo "Calibrate projector:  python tools/calibrate.py"
