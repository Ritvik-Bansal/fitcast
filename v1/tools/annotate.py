"""Click anchor points on garment PNGs and write anchors.json (and meta.json).

    python tools/annotate.py assets/garments/my_tee --category shirt --sleeve short --name "My Tee" --price 30
    python tools/annotate.py assets/garments/my_jeans --category pants
    python tools/annotate.py assets/garments/my_tee --view back        # redo one view

The window walks through each anchor of each view; click where it goes.
"left"/"right" always mean the WEARER's left/right: on front.png the wearer's
left is on the image's right; on back.png it is on the image's left. Click the
centre of each sleeve cuff, knee and ankle (the warp measures widths from the
PNG's alpha). Side views only need the shoulder/collar and hems (shirts) or
waist and ankles (pants); skip the rest with s.

Keys: click = set   s = skip   u / backspace = undo   [ ] = previous / next view
      c = clear this view   w = write   q = write and quit   esc = quit without writing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tryon.display import cv2_key_name  # noqa: E402
from tryon.garments import ANCHORS, VIEWS  # noqa: E402

WINDOW = "Fitcast anchors"
SIDE_VIEW_ANCHORS = {"shirt": ("collar", "left_shoulder", "right_shoulder", "left_hem", "right_hem"),
                     "pants": ("left_waist", "right_waist", "left_ankle", "right_ankle")}


def hint(view: str, name: str) -> str:
    if view == "front" and name.startswith(("left_", "right_")):
        side = "image RIGHT" if name.startswith("left_") else "image LEFT"
        return f"(wearer's {name.split('_')[0]} = {side} on the front view)"
    if view == "back" and name.startswith(("left_", "right_")):
        side = "image LEFT" if name.startswith("left_") else "image RIGHT"
        return f"(wearer's {name.split('_')[0]} = {side} on the back view)"
    if name.endswith(("sleeve_end", "knee", "ankle")):
        return "(click the centre of the opening)"
    return ""


class Annotator:
    def __init__(self, folder: Path, category: str, views: list[str]):
        self.folder = folder
        self.category = category
        self.views = views
        path = folder / "anchors.json"
        self.anchors = json.loads(path.read_text()) if path.exists() else {}
        self.vi = 0
        self.ai = 0
        self.dirty = False
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.load_view()

    @property
    def view(self) -> str:
        return self.views[self.vi]

    def names(self) -> tuple:
        if self.view in ("left", "right"):
            return SIDE_VIEW_ANCHORS[self.category]
        return ANCHORS[self.category]

    def load_view(self) -> None:
        img = cv2.imread(str(self.folder / f"{self.view}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"cannot read {self.folder / (self.view + '.png')}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        self.img = img
        self.scale = min(1.0, 900 / img.shape[0], 1400 / img.shape[1])
        h, w = int(img.shape[0] * self.scale), int(img.shape[1] * self.scale)
        cv2.resizeWindow(WINDOW, w, h + 60)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        tile = ((np.indices((h, w)).sum(axis=0) // 16) % 2).astype(np.float32)
        bg = (150 + 50 * tile)[..., None].repeat(3, axis=2)
        a = small[..., 3:4].astype(np.float32) / 255
        self.base = (small[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        self.anchors.setdefault(self.view, {})
        self.ai = next((i for i, n in enumerate(self.names()) if n not in self.anchors[self.view]), len(self.names()))

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.ai >= len(self.names()):
            return
        y -= 60
        if y < 0:
            return
        name = self.names()[self.ai]
        self.anchors[self.view][name] = [round(x / self.scale, 1), round(y / self.scale, 1)]
        self.dirty = True
        self.ai += 1

    def draw(self) -> np.ndarray:
        out = np.zeros((self.base.shape[0] + 60, self.base.shape[1], 3), np.uint8)
        out[60:] = self.base
        for name, (x, y) in self.anchors[self.view].items():
            p = (int(x * self.scale), int(y * self.scale) + 60)
            col = (0, 220, 255) if name.startswith("left_") else (255, 0, 255) if name.startswith("right_") else (0, 255, 0)
            cv2.circle(out, p, 6, col, -1, cv2.LINE_AA)
            cv2.putText(out, name, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        names = self.names()
        if self.ai < len(names):
            prompt = f"[{self.view}] click: {names[self.ai]} {hint(self.view, names[self.ai])}"
        else:
            prompt = f"[{self.view}] done - ] next view, w write, q write+quit"
        cv2.putText(out, prompt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "s skip  u undo  [ ] view  c clear  w write  q write+quit  esc abort",
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        return out

    def write(self) -> None:
        clean = {v: pts for v, pts in self.anchors.items() if pts}
        (self.folder / "anchors.json").write_text(json.dumps(clean, indent=2))
        self.dirty = False
        print(f"[annotate] wrote {self.folder / 'anchors.json'}")

    def run(self) -> None:
        while True:
            cv2.imshow(WINDOW, self.draw())
            k = cv2_key_name(cv2.waitKeyEx(30))
            if k is None:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            if k == "esc":
                if self.dirty:
                    print("[annotate] quit without writing")
                break
            if k == "q":
                self.write()
                break
            if k == "w":
                self.write()
            elif k == "s":
                self.ai = min(self.ai + 1, len(self.names()))
            elif k in ("u", "backspace") or k == "\x08":
                if self.ai > 0:
                    self.ai -= 1
                    self.anchors[self.view].pop(self.names()[self.ai], None)
                    self.dirty = True
            elif k == "c":
                self.anchors[self.view] = {}
                self.ai = 0
                self.dirty = True
            elif k in ("]", "tab", "right"):
                self.vi = (self.vi + 1) % len(self.views)
                self.load_view()
            elif k in ("[", "left"):
                self.vi = (self.vi - 1) % len(self.views)
                self.load_view()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="garment folder containing front.png (and back/left/right.png)")
    ap.add_argument("--view", choices=VIEWS, help="only annotate this view")
    ap.add_argument("--category", choices=("shirt", "pants"))
    ap.add_argument("--sleeve", choices=("short", "long"))
    ap.add_argument("--name")
    ap.add_argument("--price", type=float)
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"{folder} is not a folder")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key, val in (("category", args.category), ("sleeve_length", args.sleeve), ("name", args.name), ("price", args.price)):
        if val is not None:
            meta[key] = val
    meta.setdefault("name", folder.name.replace("_", " ").title())
    meta.setdefault("price", 0.0)
    if "category" not in meta:
        sys.exit("first run needs --category shirt|pants (stored in meta.json)")
    meta.setdefault("sleeve_length", "short")
    views = [args.view] if args.view else [v for v in VIEWS if (folder / f"{v}.png").exists()]
    if not views:
        sys.exit(f"no view PNGs (front/back/left/right.png) in {folder}")
    meta_path.write_text(json.dumps(meta, indent=2))
    Annotator(folder, meta["category"], views).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
