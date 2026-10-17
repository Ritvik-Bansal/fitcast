"""Generate placeholder garments so the app runs before any real photos exist.

Three shirts (two short-sleeve, one long-sleeve) and two pants, each with
front/left/back/right PNGs (transparent background), anchors.json and meta.json.
Every view carries a big label and the view name, and an "L" / "R" on the
wearer's left / right, so a mirrored back view is obvious at a glance.

    python tools/make_placeholders.py            # writes assets/garments/
    python tools/make_placeholders.py --force    # also overwrite non-placeholder folders
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FONT = cv2.FONT_HERSHEY_DUPLEX

SHIRTS = [
    ("shirt_01_crimson_tee", "Crimson Tee", 34.00, "short", (50, 40, 205), "CRIMSON", None),
    ("shirt_02_ocean_longsleeve", "Ocean Long Sleeve", 48.00, "long", (205, 120, 30), "OCEAN", None),
    ("shirt_03_lime_stripe_tee", "Lime Stripe Tee", 29.00, "short", (70, 205, 120), "LIME", (40, 120, 60)),
]
PANTS = [
    ("pants_01_indigo_jeans", "Indigo Jeans", 68.00, (140, 70, 40), "INDIGO"),
    ("pants_02_khaki_chinos", "Khaki Chinos", 54.00, (120, 180, 215), "KHAKI"),
]


def _shade(color, f):
    return tuple(int(max(0, min(255, c * f))) for c in color)


def _label(img, text, center, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(center[0] - tw / 2), int(center[1] + th / 2)), FONT, scale, color, thick, cv2.LINE_AA)


def _fill(img, poly, color):
    pts = np.round(np.asarray(poly)).astype(np.int32)
    cv2.fillPoly(img, [pts], (*color, 255), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (*_shade(color, 0.55), 255), 6, cv2.LINE_AA)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _fit_text_scale(text, max_w, base=2.6, thick=7):
    s = base
    while s > 0.6 and cv2.getTextSize(text, FONT, s, thick)[0][0] > max_w:
        s -= 0.1
    return s


# --------------------------------------------------------------------------- shirts
def shirt_front_back(color, label, stripe, long_sleeve: bool, back: bool):
    W, H = (1150, 1000) if long_sleeve else (900, 950)
    cx, top = W / 2, 110
    neck_l, neck_r = np.array([cx - 75, top]), np.array([cx + 75, top])
    collar = np.array([cx, top + (14 if back else 45)])
    sh = {"il": np.array([cx - 195, top + 40]), "ir": np.array([cx + 195, top + 40])}
    ap = {"il": np.array([cx - 170, top + 250]), "ir": np.array([cx + 170, top + 250])}
    hem = {"il": np.array([cx - 165, H - 50]), "ir": np.array([cx + 165, H - 50])}
    ang = np.radians(66 if long_sleeve else 48)
    length = 560 if long_sleeve else 210
    half = 52 if long_sleeve else 75
    sleeve = {}
    for side, sgn in (("il", -1), ("ir", 1)):
        root = (sh[side] + ap[side]) / 2
        u = np.array([sgn * np.cos(ang), np.sin(ang)])
        n = np.array([-u[1], u[0]])
        if np.dot(n, sh[side] - ap[side]) < 0:
            n = -n
        end = root + u * length
        sleeve[side] = (end, end + n * half, end - n * half)   # center, shoulder-side, armpit-side
    img = np.zeros((H, W, 4), np.uint8)
    for side in ("il", "ir"):
        end, ca, cb = sleeve[side]
        _fill(img, [sh[side], ca, cb, ap[side]], _shade(color, 0.92))
    body = [neck_l, collar, neck_r, sh["ir"], ap["ir"], hem["ir"], hem["il"], ap["il"], sh["il"]]
    _fill(img, body, color)
    if stripe is not None:
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.round(np.asarray(body)).astype(np.int32)], 255)
        for y in range(int(top + 120), H - 60, 90):
            band = np.zeros_like(mask)
            cv2.rectangle(band, (0, y), (W, y + 34), 255, -1)
            img[(band > 0) & (mask > 0)] = (*stripe, 255)
    # neckline trim
    cv2.polylines(img, [np.round(np.array([neck_l, collar, neck_r])).astype(np.int32)], False,
                  (*_shade(color, 0.45), 255), 14, cv2.LINE_AA)
    # wearer's left is image-right on the front, image-left on the back
    wl, wr = ("il", "ir") if back else ("ir", "il")
    if not back:
        p = sh[wl] + np.array([-95 if wl == "ir" else 25, 95])
        cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[0] + 70), int(p[1] + 80)), (*_shade(color, 0.7), 255), 5)
    s = _fit_text_scale(label, 300)
    _label(img, label, (cx, top + 330), s, (255, 255, 255, 255), 7)
    _label(img, "BACK" if back else "FRONT", (cx, top + 450), 1.6, (255, 255, 255, 255), 5)
    _label(img, "L", hem[wl] + np.array([45 if wl == "il" else -45, -60]), 1.8, (255, 255, 255, 255), 6)
    _label(img, "R", hem[wr] + np.array([45 if wr == "il" else -45, -60]), 1.8, (20, 20, 20, 255), 6)
    anchors = {
        "collar": collar,
        "left_shoulder": sh[wl], "right_shoulder": sh[wr],
        "left_armpit": ap[wl], "right_armpit": ap[wr],
        "left_hem": hem[wl], "right_hem": hem[wr],
        "left_sleeve_end": sleeve[wl][0], "right_sleeve_end": sleeve[wr][0],
    }
    return img, anchors


def shirt_side(color, label, long_sleeve: bool, side: str):
    """Profile strip. side='left' shows the wearer's left; they face image-left."""
    W, H = 440, 1000 if long_sleeve else 950
    top = 110
    front_x, back_x = 110, 330
    if side == "right":
        front_x, back_x = back_x, front_x
    img = np.zeros((H, W, 4), np.uint8)
    xs = sorted([front_x, back_x])
    body = [[xs[0] + 20, top + 20], [xs[1] - 20, top + 20], [xs[1], top + 70], [xs[1] + 8, H - 50], [xs[0] - 8, H - 50], [xs[0], top + 70]]
    _fill(img, body, color)
    cxs = (front_x + back_x) / 2
    sl_len = 760 if long_sleeve else 260
    sleeve = [[cxs - 70, top + 40], [cxs + 70, top + 40], [cxs + 62, top + 40 + sl_len], [cxs - 62, top + 40 + sl_len]]
    _fill(img, sleeve, _shade(color, 0.85))
    s = _fit_text_scale(label, 200, 1.6, 5)
    _label(img, label, (cxs, min(top + 120 + sl_len, H - 110)), s, (255, 255, 255, 255), 5)
    _label(img, "L" if side == "left" else "R", (cxs, top + 200), 3.2, (255, 255, 255, 255), 9)
    _label(img, "SIDE", (cxs, top + 290), 1.1, (255, 255, 255, 255), 3)
    top_pt = np.array([cxs, top + 30])
    hem_pts = (np.array([xs[0], H - 50]), np.array([xs[1], H - 50]))
    near = side
    far = "right" if side == "left" else "left"
    anchors = {
        "collar": top_pt, f"{near}_shoulder": top_pt, f"{far}_shoulder": top_pt,
        f"{near}_armpit": np.array([cxs, top + 250]), f"{far}_armpit": np.array([cxs, top + 250]),
        "left_hem": hem_pts[0], "right_hem": hem_pts[1],
        f"{near}_sleeve_end": np.array([cxs, top + 40 + sl_len]),
    }
    return img, anchors


# --------------------------------------------------------------------------- pants
def pants_front_back(color, label, back: bool):
    W, H = 800, 1180
    cx = W / 2
    top = 60
    waist = {"il": np.array([cx - 150, top]), "ir": np.array([cx + 150, top])}
    crotch = np.array([cx, 330])
    outline = [waist["il"], waist["ir"], [cx + 168, 270], [cx + 188, 660], [cx + 182, H - 60], [cx + 28, H - 60],
               [cx + 30, 660], crotch, [cx - 30, 660], [cx - 28, H - 60], [cx - 182, H - 60], [cx - 188, 660], [cx - 168, 270]]
    img = np.zeros((H, W, 4), np.uint8)
    _fill(img, outline, color)
    cv2.rectangle(img, (int(cx - 150), top), (int(cx + 150), top + 55), (*_shade(color, 0.7), 255), -1)
    for x in (cx - 110, cx - 30, cx + 30, cx + 110):
        cv2.rectangle(img, (int(x - 6), top), (int(x + 6), top + 55), (*_shade(color, 0.5), 255), -1)
    if back:
        for x in (cx - 95, cx + 95):
            cv2.rectangle(img, (int(x - 55), 130), (int(x + 55), 240), (*_shade(color, 0.75), 255), 6)
    else:
        cv2.line(img, (int(cx), top + 55), (int(cx), 330), (*_shade(color, 0.55), 255), 5, cv2.LINE_AA)
    knee = {"il": np.array([cx - 109, 660]), "ir": np.array([cx + 109, 660])}
    ankle = {"il": np.array([cx - 105, H - 70]), "ir": np.array([cx + 105, H - 70])}
    s = _fit_text_scale(label, 330, 2.2, 6)
    _label(img, label, (cx, 300), s, (255, 255, 255, 255), 6)
    _label(img, "BACK" if back else "FRONT", (cx, 470), 1.2, (255, 255, 255, 255), 4)
    wl, wr = ("il", "ir") if back else ("ir", "il")
    _label(img, "L", knee[wl] + np.array([0, 120]), 2.0, (255, 255, 255, 255), 6)
    _label(img, "R", knee[wr] + np.array([0, 120]), 2.0, (20, 20, 20, 255), 6)
    anchors = {
        "left_waist": waist[wl], "right_waist": waist[wr], "crotch": crotch,
        "left_knee": knee[wl], "right_knee": knee[wr], "left_ankle": ankle[wl], "right_ankle": ankle[wr],
    }
    return img, anchors


def pants_side(color, label, side: str):
    W, H = 400, 1180
    img = np.zeros((H, W, 4), np.uint8)
    outline = [[110, 60], [290, 60], [300, 400], [270, 660], [262, H - 60], [138, H - 60], [130, 660], [100, 400]]
    _fill(img, outline, color)
    cv2.rectangle(img, (110, 60), (290, 115), (*_shade(color, 0.7), 255), -1)
    _label(img, "L" if side == "left" else "R", (200, 330), 3.0, (255, 255, 255, 255), 9)
    _label(img, "SIDE", (200, 430), 1.1, (255, 255, 255, 255), 3)
    s = _fit_text_scale(label, 220, 1.4, 4)
    _label(img, label, (200, 560), s, (255, 255, 255, 255), 4)
    anchors = {"left_waist": np.array([200, 60]), "right_waist": np.array([200, 60]),
               "left_knee": np.array([200, 660]), "right_knee": np.array([200, 660]),
               "left_ankle": np.array([200, H - 70]), "right_ankle": np.array([200, H - 70]),
               "crotch": np.array([200, 330])}
    return img, anchors


# --------------------------------------------------------------------------- write
def write_garment(root: Path, gid: str, meta: dict, views: dict, force: bool) -> bool:
    folder = root / gid
    if folder.exists() and not (folder / ".placeholder").exists() and not force:
        print(f"  skip {gid}: folder exists and is not a placeholder (use --force)")
        return False
    folder.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for view, (img, pts) in views.items():
        cv2.imwrite(str(folder / f"{view}.png"), img)
        anchors[view] = {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in pts.items()}
    (folder / "anchors.json").write_text(json.dumps(anchors, indent=2))
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    (folder / ".placeholder").write_text("generated by tools/make_placeholders.py\n")
    print(f"  wrote {gid}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "assets" / "garments"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[placeholders] writing to {root}")
    for gid, name, price, sleeve, color, label, stripe in SHIRTS:
        long_s = sleeve == "long"
        views = {
            "front": shirt_front_back(color, label, stripe, long_s, back=False),
            "back": shirt_front_back(color, label, stripe, long_s, back=True),
            "left": shirt_side(color, label, long_s, "left"),
            "right": shirt_side(color, label, long_s, "right"),
        }
        write_garment(root, gid, {"name": name, "price": price, "category": "shirt", "sleeve_length": sleeve}, views, args.force)
    for gid, name, price, color, label in PANTS:
        views = {
            "front": pants_front_back(color, label, back=False),
            "back": pants_front_back(color, label, back=True),
            "left": pants_side(color, label, "left"),
            "right": pants_side(color, label, "right"),
        }
        write_garment(root, gid, {"name": name, "price": price, "category": "pants", "sleeve_length": "short"}, views, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
