"""Photo-like procedural garments for scripts/fetch_garments.py: the pieces no free photo was found for.

Unlike the flat demo outfits in prepare_garments.py these add a knit/woven fabric texture (heather
fibres, twill, mesh, brushed plaid), soft fold shading, edge shading (volume of a ghost-mannequin shot),
seams with dashed top-stitching, ribbing, pockets and hardware. Each builder takes the canvas (w, h) and
returns (RGBA uint8, front-only detail mask 0/255, top_body_region or None) in the same layout as the
demo garments (tops: TEE_BODY torso; bottoms: PANTS-style legs stretched over each half).
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from prepare_garments import LONG_SLEEVE, PANTS_STRETCH, TANK_BODY, TEE_BODY, TEE_BODY_X, _Draw, _lerp, _mirror, _shade

Garment = tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None]

# Look tunables (asset-generation only; not runtime config).
FOLD_SIGMA = 0.07           # fold shading blob size, fraction of the canvas width
FOLD_AMP = 0.09             # +- brightness of the fold shading
EDGE_DARK = 0.22            # darkening at the silhouette edge (volume)
EDGE_FALLOFF = 0.035        # ... decays over this fraction of the width
CREASE_AMP = 0.16           # strength of drawn creases
STITCH_DASH = 0.011         # dash length of top-stitching, fraction of the width

HEATHER_GREY = (178, 176, 172)      # BGR
HEATHER_DARK = (120, 118, 115)
SKY = (232, 184, 110)
ROYAL = (170, 70, 20)
OLIVE = (92, 124, 118)
KHAKI = (150, 178, 196)
FLANNEL_RED = (52, 48, 200)
NAVY = (110, 50, 25)
CREAM = (215, 235, 245)
BRASS = (60, 150, 190)
WHITE = (245, 245, 245)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _smooth_noise(h: int, w: int, sigma_px: float, seed: int) -> np.ndarray:
    """Zero-mean, unit-ish amplitude low-frequency noise (float32 HxW)."""
    n = _rng(seed).normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(1.0, sigma_px))
    return n / max(1e-6, float(np.abs(n).max()))


def _heather(w: int, h: int, base, seed: int, amp: float = 26.0) -> np.ndarray:
    """Heathered knit: vertical-ish fibre streaks + light/dark flecks around `base`."""
    r = _rng(seed)
    fib = r.normal(0, 1, (h, w)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (1, 7), 0) * amp
    fleck = (r.random((h, w)) < 0.025).astype(np.float32) * r.choice([-1.0, 1.0], (h, w)) * amp * 1.6
    knit = ((np.arange(w) % 3 == 0).astype(np.float32) * -6.0)[None, :]
    img = np.asarray(base, np.float32)[None, None] + (fib + fleck + knit)[..., None]
    return img.clip(0, 255).astype(np.uint8)


def _twill(w: int, h: int, base, seed: int, amp: float = 9.0, step_px: int = 3, noise: float = 7.0) -> np.ndarray:
    """Woven twill: diagonal ridges + yarn noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    ridge = np.sin((xx + yy) * (2 * np.pi / (2 * step_px))).astype(np.float32) * amp
    n = cv2.GaussianBlur(_rng(seed).normal(0, noise, (h, w)).astype(np.float32), (0, 0), 0.8)
    slub = cv2.GaussianBlur(_rng(seed + 1).normal(0, 1, (h, w)).astype(np.float32), (1, 31), 0) * 8.0
    img = np.asarray(base, np.float32)[None, None] + (ridge + n + slub)[..., None]
    return img.clip(0, 255).astype(np.uint8)


def _mesh(w: int, h: int, base, seed: int) -> np.ndarray:
    """Athletic mesh: staggered grid of small darker holes over a slightly noisy base."""
    img = _twill(w, h, base, seed, amp=2.0, noise=4.0).astype(np.float32)
    hole = np.zeros((h, w), np.float32)
    s = max(4, round(w / 85))
    for row, y in enumerate(range(0, h + s, s)):
        for x in range((row % 2) * s // 2, w + s, s):
            cv2.circle(hole, (x, y), max(1, s // 4), 1.0, -1, cv2.LINE_AA)
    img *= (1.0 - 0.22 * hole)[..., None]
    return img.clip(0, 255).astype(np.uint8)


def _flannel(w: int, h: int, seed: int) -> np.ndarray:
    """Brushed red/navy/cream tartan flannel with a soft nap."""
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = FLANNEL_RED
    p = 0.22 * w

    def bands(v, width, off):
        return (((v - off) % p) < width).astype(np.float32)
    for col, op, width, off in ((NAVY, 0.55, 0.09 * w, 0.0), (NAVY, 0.35, 0.025 * w, 0.13 * w),
                                (CREAM, 0.55, 0.008 * w, 0.165 * w)):
        m = np.minimum(1.0, op * (bands(x, width, off)[None, :] + bands(y, width, off)[:, None]))[..., None]
        img = img * (1 - m) + np.asarray(col, np.float32) * m
    nap = cv2.GaussianBlur(_rng(seed).normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 14.0
    img = cv2.GaussianBlur(img, (0, 0), 0.9) + nap[..., None]
    return img.clip(0, 255).astype(np.uint8)


def _stitch(d: _Draw, pts, color, px: float = 2.0, detail: bool = False) -> None:
    """Dashed top-stitching along a polyline (normalized coordinates)."""
    p = d.p(pts).astype(np.float32)
    dash = max(3.0, STITCH_DASH * d.w)
    t = max(1, d.t(px))
    for a, b in zip(p[:-1], p[1:]):
        L = float(np.hypot(*(b - a)))
        n = max(1, int(L // dash))
        for i in range(0, n, 2):
            s, e = a + (b - a) * (i / n), a + (b - a) * (min(n, i + 1) / n)
            cv2.line(d.img, tuple(int(v) for v in s), tuple(int(v) for v in e), tuple(int(c) for c in color), t, cv2.LINE_AA)
            if detail:
                cv2.line(d.detail, tuple(int(v) for v in s), tuple(int(v) for v in e), 255, t, cv2.LINE_AA)


def _rib(d: _Draw, quad, color, period: float = 0.012, horizontal_ribs: bool = False) -> None:
    """Ribbed knit band (cuff/waistband): fill `quad` and add fine rib lines across it."""
    d.poly(quad, img=color, alpha=255)
    mask = np.zeros((d.h, d.w), np.uint8)
    cv2.fillPoly(mask, [d.p(quad)], 255, cv2.LINE_AA)
    yy, xx = np.mgrid[0:d.h, 0:d.w]
    coord = yy if horizontal_ribs else xx
    per = max(3.0, period * d.w)
    ribs = (0.5 + 0.5 * np.sin(coord * 2 * np.pi / per)).astype(np.float32)
    shade = (1.0 - 0.16 * ribs) * (mask.astype(np.float32) / 255.0) + (1.0 - mask.astype(np.float32) / 255.0)
    d.img = (d.img.astype(np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)


def _crease(map_: np.ndarray, d: _Draw, pts, px: float, sign: float) -> None:
    cv2.polylines(map_, [d.p(pts)], False, float(sign), d.t(px), cv2.LINE_AA)


def _finish(d: _Draw, seed: int, creases: list[tuple[list, float, float]] = ()) -> np.ndarray:
    """Fold shading + silhouette edge shading + creases ((pts, width_px, +1 light / -1 dark)); returns RGBA."""
    h, w = d.h, d.w
    shade = 1.0 + FOLD_AMP * _smooth_noise(h, w, FOLD_SIGMA * w, seed)
    solid = (d.alpha > 127).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(solid, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    shade *= 1.0 - EDGE_DARK * np.exp(-dist / (EDGE_FALLOFF * w))
    cm = np.zeros((h, w), np.float32)
    for pts, px, sign in creases:
        _crease(cm, d, pts, px, sign)
    shade *= 1.0 + CREASE_AMP * cv2.GaussianBlur(cm, (0, 0), 0.012 * w)
    img = (d.img.astype(np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)
    a = cv2.GaussianBlur(d.alpha, (5, 5), 1.0)
    return np.dstack([img, np.minimum(d.alpha, a)])


BOTH = (lambda p: p, _mirror)


# ----------------------------------------------------------------------------- tops

def hoodie(w: int, h: int) -> Garment:
    """Heather-grey pullover hoodie: hood laid behind the neck, drawcords, kangaroo pocket, ribbed cuffs/hem."""
    base = HEATHER_GREY
    d = _Draw(_heather(w, h, base, seed=11))
    d.poly(TEE_BODY, alpha=255)
    for m in BOTH:
        d.poly(m(LONG_SLEEVE), alpha=255)
    seam = _shade(base, 0.7)
    # hood: lies flat around the neck; the opening is a soft dark hollow, not a flat disc
    d.ellipse((0.5, 0.085), (0.150, 0.085), img=_shade(base, 0.93), alpha=255)
    d.ellipse((0.5, 0.085), (0.150, 0.085), px=2, img=seam)
    hollow = np.zeros((d.h, d.w), np.uint8)
    cv2.ellipse(hollow, tuple(int(v) for v in d.p([(0.5, 0.075)])[0]), (round(0.095 * d.w), round(0.06 * d.h)),
                0, 0, 360, 255, -1, cv2.LINE_AA)
    depth = cv2.GaussianBlur(hollow.astype(np.float32) / 255.0, (0, 0), 0.03 * d.w)
    d.img = (d.img.astype(np.float32) * (1.0 - 0.55 * depth[..., None])).clip(0, 255).astype(np.uint8)
    d.ellipse((0.5, 0.075), (0.105, 0.068), px=3, img=_shade(base, 1.05))  # hood edge hem
    # drawcords with metal eyelets and aglets
    for x0, x1 in ((0.455, 0.445), (0.545, 0.56)):
        d.circle((x0, 0.185), 0.010, img=BRASS, detail=True)
        d.circle((x0, 0.185), 0.005, img=_shade(base, 0.4), detail=True)
        d.line([(x0, 0.19), ((x0 + x1) / 2 + 0.004, 0.28), (x1, 0.37)], 6, img=WHITE, detail=True)
        d.line([(x1, 0.37), (x1, 0.40)], 7, img=(150, 150, 150), detail=True)
    # kangaroo pocket
    pocket = [(0.35, 0.62), (0.65, 0.62), (0.70, 0.875), (0.30, 0.875)]
    d.poly(pocket, img=_shade(base, 1.04), detail=True)
    d.line(pocket, 2, img=seam, closed=True, detail=True)
    _stitch(d, [(0.36, 0.635), (0.64, 0.635)], seam, detail=True)
    for m in BOTH:
        d.line(m([(0.35, 0.62), (0.318, 0.75), (0.305, 0.87)]), 5, img=_shade(base, 0.6), detail=True)
        _stitch(d, m([(0.365, 0.64), (0.335, 0.75), (0.322, 0.86)]), seam, detail=True)
        # raglan-ish shoulder seam and ribbed cuffs
        d.line(m([(0.33, 0.10), (0.25, 0.12)]), 2, img=seam)
        a, b = LONG_SLEEVE[3], LONG_SLEEVE[4]
        _rib(d, m([a, b, _lerp(b, LONG_SLEEVE[5], 0.11), _lerp(a, LONG_SLEEVE[2], 0.11)]), _shade(base, 0.92))
    _rib(d, [(0.268, 0.89), (0.732, 0.89), (0.73, 0.96), (0.27, 0.96)], _shade(base, 0.92))
    creases = [([(0.40, 0.30), (0.38, 0.55)], 6, -1), ([(0.61, 0.28), (0.63, 0.52)], 6, -1),
               ([(0.30, 0.45), (0.33, 0.60)], 5, 1)]
    rgba = _finish(d, seed=12, creases=creases)
    return rgba, d.detail, d.region(*TEE_BODY_X)


def jersey(w: int, h: int) -> Garment:
    """Sky-blue mesh basketball jersey: deep V-neck/armhole trim, side panels, tackle-twill number."""
    d = _Draw(_mesh(w, h, SKY, seed=21))
    d.poly(TANK_BODY, alpha=255)
    neck = [(0.42, 0.03), (0.5, 0.20), (0.58, 0.03)]
    d.poly([(0.415, 0.0), (0.5, 0.19), (0.585, 0.0)], alpha=0)
    d.line(neck, 16, img=ROYAL)
    d.line(neck, 3, img=WHITE)
    arm = [TANK_BODY[0], TANK_BODY[11], TANK_BODY[10], TANK_BODY[9]]
    for m in BOTH:
        d.line(m(arm), 16, img=ROYAL)
        d.line(m([(x + 0.012, y) for x, y in arm]), 3, img=WHITE)
        d.poly(m([(0.252, 0.36), (0.30, 0.36), (0.30, 0.94), (0.272, 0.94)]), img=_shade(ROYAL, 1.2))
        d.line(m([(0.30, 0.36), (0.30, 0.94)]), 2, img=WHITE)
    d.text("TEAM", (0.5, 0.30), 1.55, 7, img=ROYAL, outline=WHITE, detail=True)
    d.text("23", (0.5, 0.52), 4.2, 16, img=WHITE, outline=ROYAL, detail=True)
    _stitch(d, [(0.28, 0.93), (0.72, 0.93)], _shade(SKY, 0.65))
    rgba = _finish(d, seed=22, creases=[([(0.40, 0.65), (0.39, 0.90)], 5, -1), ([(0.62, 0.62), (0.64, 0.88)], 5, -1)])
    return rgba, d.detail, d.region(*TEE_BODY_X)


def flannel(w: int, h: int) -> Garment:
    """Red tartan flannel overshirt: point collar, button placket, flap chest pockets, curved shirttail."""
    d = _Draw(_flannel(w, h, seed=31))
    body = TEE_BODY[:3] + [(0.27, 0.93), (0.36, 0.975), (0.5, 0.95), (0.64, 0.975), (0.73, 0.93)] + TEE_BODY[5:]
    d.poly(body, alpha=255)
    for m in BOTH:
        d.poly(m(LONG_SLEEVE), alpha=255)
    dark = _shade(FLANNEL_RED, 0.45)
    d.ellipse((0.5, 0.035), (0.09, 0.045), alpha=0)
    collar = [(0.405, 0.025), (0.50, 0.085), (0.45, 0.17), (0.345, 0.07)]
    for cl in (collar, _mirror(collar)):
        d.poly(cl, img=_shade(FLANNEL_RED, 1.08), alpha=255, detail=True)
        d.line(cl, 2, img=dark, closed=True)
        _stitch(d, [_lerp(cl[3], cl[2], 0.1), _lerp(cl[2], cl[1], 0.1)], dark)
    d.poly([(0.478, 0.09), (0.522, 0.09), (0.522, 0.955), (0.478, 0.955)], detail=True)
    for x in (0.478, 0.522):
        d.line([(x, 0.10), (x, 0.95)], 2, img=dark)
    for y in np.linspace(0.14, 0.86, 7):
        d.circle((0.5, y), 0.010, img=CREAM, detail=True)
        d.circle((0.5, y), 0.010, px=1, img=_shade(CREAM, 0.6), detail=True)
    for m in BOTH:
        pocket = m([(0.57, 0.22), (0.69, 0.22), (0.69, 0.36), (0.63, 0.375), (0.57, 0.36)])
        d.poly(pocket, detail=True)
        d.line(pocket, 2, img=dark, closed=True, detail=True)
        flap = m([(0.565, 0.21), (0.695, 0.21), (0.695, 0.255), (0.63, 0.27), (0.565, 0.255)])
        d.poly(flap, img=_shade(FLANNEL_RED, 1.05), detail=True)
        d.line(flap, 2, img=dark, closed=True, detail=True)
        d.circle(m([(0.63, 0.245)])[0], 0.008, img=CREAM, detail=True)
        d.line(m([(0.33, 0.07), (0.255, 0.11)]), 2, img=dark)  # shoulder seam
        a, b = LONG_SLEEVE[3], LONG_SLEEVE[4]
        cuff = m([a, b, _lerp(b, LONG_SLEEVE[5], 0.10), _lerp(a, LONG_SLEEVE[2], 0.10)])
        d.poly(cuff, img=_shade(FLANNEL_RED, 0.92))
        d.line(cuff, 2, img=dark, closed=True)
    creases = [([(0.40, 0.42), (0.37, 0.70)], 6, -1), ([(0.61, 0.45), (0.64, 0.72)], 6, -1),
               ([(0.30, 0.50), (0.29, 0.80)], 4, 1)]
    rgba = _finish(d, seed=32, creases=creases)
    return rgba, d.detail, d.region(*TEE_BODY_X)


# ----------------------------------------------------------------------------- bottoms

def _legs(hip_out: float, hem_out: float, hem_in: float, crotch_y: float = 0.37, waist: float = 0.22) -> list:
    """Pants polygon in PANTS coordinates (stretched by PANTS_STRETCH): outer hip/hem x, inner hem x."""
    return [(waist, 0.03), (1 - waist, 0.03), (1 - hip_out, 0.30), (1 - hem_out, 0.97), (1 - hem_in, 0.97),
            (0.5, crotch_y), (hem_in, 0.97), (hem_out, 0.97), (hip_out, 0.30)]


def _waistband(d: _Draw, color, dark, loops: bool = True) -> None:
    d.poly([(0.215, 0.025), (0.785, 0.025), (0.787, 0.085), (0.213, 0.085)], img=color)
    d.line([(0.213, 0.085), (0.787, 0.085)], 3, img=dark)
    _stitch(d, [(0.22, 0.033), (0.78, 0.033)], dark)
    _stitch(d, [(0.22, 0.077), (0.78, 0.077)], dark)
    if loops:
        for x in (0.25, 0.37, 0.63, 0.75):
            loop = [(x - 0.011, 0.02), (x + 0.011, 0.02), (x + 0.011, 0.10), (x - 0.011, 0.10)]
            d.poly(loop, img=_shade(color, 1.06))
            d.line(loop, 2, img=dark, closed=True)


def _fly(d: _Draw, dark, button) -> None:
    d.circle((0.5, 0.055), 0.016, img=button, detail=True)
    d.line([(0.5, 0.09), (0.5, 0.33)], 2, img=dark)
    _stitch(d, [(0.535, 0.09), (0.535, 0.28), (0.525, 0.315), (0.502, 0.33)], dark, detail=True)


def cargo(w: int, h: int) -> Garment:
    """Olive cotton-twill cargo pants: wide straight legs, bellows thigh pockets with snap flaps."""
    base = OLIVE
    d = _Draw(_twill(w, h, base, seed=41), sx=PANTS_STRETCH)
    d.poly(_legs(0.20, 0.235, 0.47), alpha=255)
    dark = _shade(base, 0.5)
    _waistband(d, _shade(base, 0.96), dark)
    _fly(d, dark, (90, 90, 90))
    for m in BOTH:
        d.line(m([(0.33, 0.085), (0.30, 0.16), (0.212, 0.205)]), 3, img=dark)  # slant pocket
        _stitch(d, m([(0.345, 0.09), (0.315, 0.17), (0.215, 0.222)]), dark)
        pocket = m([(0.21, 0.40), (0.33, 0.40), (0.335, 0.60), (0.212, 0.60)])
        d.poly(pocket, img=_shade(base, 1.06), detail=False)
        d.line(pocket, 3, img=dark, closed=True)
        _stitch(d, m([(0.215, 0.59), (0.33, 0.59)]), dark)
        d.line(m([(0.27, 0.43), (0.27, 0.59)]), 3, img=_shade(base, 0.75))  # bellows pleat
        flap = m([(0.205, 0.38), (0.34, 0.38), (0.34, 0.435), (0.205, 0.435)])
        d.poly(flap, img=_shade(base, 1.02))
        d.line(flap, 3, img=dark, closed=True)
        _stitch(d, m([(0.21, 0.427), (0.335, 0.427)]), dark)
        d.circle(m([(0.272, 0.415)])[0], 0.011, img=(80, 80, 80))
        d.line(m([(0.205, 0.64), (0.49, 0.64)]), 2, img=_shade(base, 0.7))  # knee seam
        _stitch(d, m([(0.21, 0.65), (0.485, 0.65)]), dark)
        _stitch(d, m([(0.22, 0.945), (0.47, 0.945)]), dark)
        d.line(m([(0.20, 0.30), (0.215, 0.965)]), 2, img=_shade(base, 0.7))  # outseam
    creases = []
    for m in BOTH:
        creases += [(m([(0.40, 0.42), (0.36, 0.52)]), 5, -1), (m([(0.30, 0.70), (0.42, 0.74)]), 6, -1),
                    (m([(0.28, 0.80), (0.44, 0.83)]), 5, -1), (m([(0.33, 0.25), (0.45, 0.33)]), 6, 1)]
    return _finish(d, seed=42, creases=creases), d.detail, None


def chinos(w: int, h: int) -> Garment:
    """Khaki chinos: slim-straight legs, slant pockets, pressed centre creases."""
    base = KHAKI
    d = _Draw(_twill(w, h, base, seed=51, amp=5.0, step_px=2, noise=5.0), sx=PANTS_STRETCH)
    d.poly(_legs(0.205, 0.275, 0.455, crotch_y=0.36), alpha=255)
    dark = _shade(base, 0.6)
    _waistband(d, base, dark)
    _fly(d, dark, _shade(base, 0.55))
    creases = []
    for m in BOTH:
        d.line(m([(0.31, 0.085), (0.212, 0.23)]), 3, img=dark)
        _stitch(d, m([(0.325, 0.088), (0.216, 0.25)]), dark)
        _stitch(d, m([(0.28, 0.945), (0.45, 0.945)]), dark)
        creases += [(m([(0.335, 0.12), (0.358, 0.965)]), 2, 1), (m([(0.343, 0.12), (0.366, 0.965)]), 2, -1),
                    (m([(0.40, 0.38), (0.33, 0.45)]), 5, -1), (m([(0.30, 0.86), (0.44, 0.88)]), 4, -1)]
    return _finish(d, seed=52, creases=creases), d.detail, None


def sweatpants(w: int, h: int) -> Garment:
    """Heather-grey fleece joggers: elastic ribbed waistband with drawcord, tapered legs, ribbed ankle cuffs."""
    base = HEATHER_GREY
    d = _Draw(_heather(w, h, base, seed=61), sx=PANTS_STRETCH)
    d.poly(_legs(0.19, 0.30, 0.445, crotch_y=0.40, waist=0.21), alpha=255)
    dark = _shade(base, 0.62)
    _rib(d, [(0.205, 0.02), (0.795, 0.02), (0.797, 0.105), (0.203, 0.105)], _shade(base, 0.95), period=0.008)
    for y in (0.045, 0.07):  # elastic channels
        d.line([(0.21, y), (0.79, y)], 2, img=_shade(base, 0.8))
    for x0, x1 in ((0.485, 0.47), (0.515, 0.535)):
        d.circle((x0, 0.06), 0.007, img=_shade(base, 0.5), detail=True)
        d.line([(x0, 0.065), (x1, 0.20)], 5, img=WHITE, detail=True)
        d.line([(x1, 0.20), (x1, 0.225)], 6, img=(150, 150, 150), detail=True)
    creases = []
    for m in BOTH:
        d.line(m([(0.25, 0.13), (0.215, 0.27)]), 3, img=dark)  # side pocket opening
        cuff = m([(0.30, 0.89), (0.445, 0.89), (0.445, 0.97), (0.30, 0.97)])
        _rib(d, cuff, _shade(base, 0.93), period=0.008)
        d.line(m([(0.30, 0.89), (0.445, 0.89)]), 2, img=dark)
        # bunching above the cuff and at the knee
        for k, y in enumerate((0.80, 0.84, 0.875)):
            creases.append((m([(0.29, y), (0.37, y + 0.012), (0.45, y - 0.005)]), 5, -1 if k % 2 == 0 else 1))
        creases += [(m([(0.25, 0.55), (0.38, 0.60)]), 7, -1), (m([(0.40, 0.42), (0.34, 0.50)]), 6, -1)]
    return _finish(d, seed=62, creases=creases), d.detail, None


# key -> (kind, builder); description for SOURCES.md
GARMENTS: dict[str, tuple[str, Callable[[int, int], Garment]]] = {
    "hoodie_heather": ("top", hoodie),
    "jersey_basketball": ("top", jersey),
    "flannel_red": ("top", flannel),
    "cargo_olive": ("bottom", cargo),
    "chinos_khaki": ("bottom", chinos),
    "sweatpants_heather": ("bottom", sweatpants),
}
DESCRIPTIONS: dict[str, str] = {
    "hoodie_heather": "heather-grey pullover hoodie, kangaroo pocket, drawcords, ribbed cuffs/hem",
    "jersey_basketball": "sky-blue mesh basketball jersey with royal trim and number",
    "flannel_red": "red/navy tartan flannel overshirt with flap pockets",
    "cargo_olive": "olive twill cargo pants, wide legs, bellows pockets",
    "chinos_khaki": "khaki slim chinos with pressed creases",
    "sweatpants_heather": "heather-grey joggers, ribbed waistband and cuffs, drawcord",
}
