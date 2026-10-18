"""End-to-end: replay a synthetic video through `python -m tryon --headless` with scripted keys."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = ROOT / ".venv" / "bin" / "python"
POSE = HERE / "data" / "pose.jpg"
H_TRUE = np.array([[1.5, 0.05, 120.0], [-0.04, 1.5, -80.0], [1.5e-5, 1.0e-5, 1.0]])


def make_video(path: Path, seconds=6, fps=30):
    img = cv2.imread(str(POSE))
    assert img is not None, POSE
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    assert w.isOpened()
    n = seconds * fps
    for i in range(n):
        ph = 2 * np.pi * i / n
        sc = 1.04 + 0.06 * np.sin(ph)               # slow scale change
        tx = 640 - 500 * sc + 70 * np.sin(ph)         # slow translation
        ty = 380 - 333 * sc + 25 * np.cos(ph)
        M = np.array([[sc, 0, tx], [0, sc, ty]], np.float64)
        frame = cv2.warpAffine(img, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w.write(frame)
    w.release()


@unittest.skipUnless(PY.exists() and POSE.exists() and (ROOT / "models" / "pose_landmarker_full.task").exists(),
                     "venv, models or test image missing")
class TestPipelineVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.video = tmp / "in.mp4"
        make_video(cls.video)
        calib = tmp / "calib.json"
        calib.write_text(json.dumps({"H": H_TRUE.tolist(), "camera_size": [1280, 720], "projector_size": [1920, 1080],
                                     "error_px": 0.3, "method": "checkerboard"}))
        cls.tx = tmp / "transactions.jsonl"
        cfg = tmp / "cfg.yaml"
        cfg.write_text("\n".join([
            f"calib_path: {calib}",
            f"transactions_path: {cls.tx}",
            f"env_path: {tmp / 'no.env'}",
            "startup_blank_ms: 0",
            "profile_print_after_s: 4",
            "",
        ]))
        cls.snap = tmp / "snap"
        env = {k: v for k, v in os.environ.items() if not k.startswith("VISA_")}
        cls.proc = subprocess.run(
            [str(PY), "-m", "tryon", "--source", str(cls.video), "--headless", "--duration", "6",
             "--snapshot", str(cls.snap), "--config", str(cfg), "--keys", "1:right,2:space,3:enter,4:left"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)
        cls.out = cls.proc.stdout + "\n" + cls.proc.stderr
        print("\n----- app output -----\n" + cls.out[-3000:] + "\n----------------------")

    @classmethod
    def tearDownClass(cls):
        keep = os.environ.get("FITCAST_KEEP_SNAPSHOTS")
        if keep:
            import shutil
            Path(keep).mkdir(parents=True, exist_ok=True)
            for p in Path(cls.tmp.name).glob("snap_*.png"):
                shutil.copy(p, keep)
        cls.tmp.cleanup()

    def test_exit_code_and_no_traceback(self):
        self.assertEqual(self.proc.returncode, 0, self.out[-2000:])
        self.assertNotIn("Traceback", self.out)

    def test_calibration_loaded_and_projecting(self):
        self.assertIn("[app] calibration loaded", self.out)
        self.assertNotIn("DEBUG ONLY", self.out.split("[perf]")[0].replace("DEBUG ONLY: no calib.json", ""))

    def test_perf_line_fps(self):
        m = re.findall(r"\[perf\] render ([\d.]+) fps, camera ([\d.]+) fps, pose ([\d.]+) fps \| (.*)", self.out)
        self.assertTrue(m, "no [perf] line")
        render_fps = float(m[0][0])
        print(f"\n[perf-line] {m[0]}")
        self.assertGreaterEqual(render_fps, 25.0)

    def test_checkout_authorized(self):
        self.assertIn("[checkout] AUTHORIZED", self.out)
        self.assertTrue(self.tx.exists(), "transactions file not written to the temp path")
        rows = [json.loads(l) for l in self.tx.read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertFalse((ROOT / "transactions.jsonl").exists(), "app wrote transactions.jsonl into the repo root")

    def test_snapshots_exist_and_valid(self):
        for suffix, shape in (("debug", (720, 1280, 3)), ("projector", (1080, 1920, 3))):
            p = Path(f"{self.snap}_{suffix}.png")
            self.assertTrue(p.exists(), p)
            img = cv2.imread(str(p))
            self.assertEqual(img.shape, shape)

    def test_projector_snapshot_is_black_background(self):
        img = cv2.imread(f"{self.snap}_projector.png")
        nb = img.max(axis=2) > 0
        self.assertGreater(nb.sum(), 1000, "nothing was projected")
        self.assertLess(nb.mean(), 0.5)
        # the garment lands in the part of the projector frame that the known H maps the body to (not at the corners)
        ys, xs = np.nonzero(nb)
        self.assertGreater(xs.min(), 100)

    def test_debug_snapshot_has_garment_on_body(self):
        """Garment pixels (non-photo colours) should overlap the person's torso region in the debug (mirrored) frame."""
        dbg = cv2.imread(f"{self.snap}_debug.png")
        vid = cv2.VideoCapture(str(self.video))
        vid.set(cv2.CAP_PROP_POS_FRAMES, 150)
        ok, ref = vid.read()
        self.assertTrue(ok)
        ref = cv2.flip(ref, 1)
        diff = np.abs(dbg.astype(int) - ref.astype(int)).max(axis=2)
        # ignore the HUD strip at the top and footer
        diff[:170] = 0
        diff[-40:] = 0
        changed = diff > 40
        self.assertGreater(changed.sum(), 3000, "debug frame shows no garment overlay")


if __name__ == "__main__":
    unittest.main()
