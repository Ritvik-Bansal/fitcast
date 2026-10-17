"""Warp one garment view onto body keypoints, in camera pixel coordinates.

Shirts: the torso is a 7-triangle mesh over collar, shoulders, armpits, hems and
a torso centre; each triangle has its own affine, so adjacent triangles agree on
shared edges and the torso bends without seams. Sleeves are rigid "limbs" that
pivot at the armhole centre and follow the arm (one segment for short sleeves,
two hinged at the elbow for long ones), so they can never flip inside out.
Pants: the waist is one triangle (both waist corners + crotch), each leg is a
thigh limb and a shin limb hinged at the knee.

Every opaque garment pixel belongs to exactly one piece. Pixels outside the
anchor mesh (collar tabs, hem flare) are assigned to the nearest piece, so the
garment outline is extrapolated by that piece's transform instead of cropped.
Each piece's premultiplied patch is forward-warped with cv2.warpAffine into its
destination box. Pieces of one region add together (seamless because their
transforms agree on shared edges); regions then stack with "over", so sleeves
sit over the torso and the waistband over the legs.

Left/right: garment anchors are matched to body joints by which side of the
image they are on, not by MediaPipe's labels. The back view therefore never
mirrors, even when MediaPipe swaps left and right on a person facing away.

Side views (narrow strips) and degenerate poses use a 2-point similarity
transform (shoulder-mid/hem-mid or waist-mid/ankle-mid) instead of the mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SIDE_PAIRS_SHIRT = ("shoulder", "armpit", "hem", "sleeve_end")
SIDE_PAIRS_PANTS = ("waist", "knee", "ankle")
SHIRT_TRIS = [("Sl", "C", "Al"), ("C", "Sr", "Ar"), ("C", "Ar", "Al"),
              ("Al", "Ar", "T"), ("Ar", "Hr", "T"), ("Hr", "Hl", "T"), ("Hl", "Al", "T")]


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _tri_area(a, b, c) -> float:
    return 0.5 * float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def limb_affine(ps, es, pd, ed, across: float, min_along: float, max_along: float) -> np.ndarray:
    """Affine taking src segment ps->es onto dst pd->ed, scaling width by `across`."""
    ps, es, pd, ed = (np.asarray(p, np.float64) for p in (ps, es, pd, ed))
    vs, vd = es - ps, ed - pd
    ls, ld = float(np.linalg.norm(vs)), float(np.linalg.norm(vd))
    us, ud = _unit(vs), _unit(vd)
    along = np.clip(ld / max(ls, 1e-6), min_along, max_along)
    rs = np.stack([us, _perp(us)])          # rows: src axis frame
    rd = np.stack([ud, _perp(ud)], axis=1)  # cols: dst axis frame
    m = rd @ np.diag([along, across]) @ rs
    t = pd - m @ ps
    return np.hstack([m, t[:, None]])


def similarity_affine(s0, s1, d0, d1) -> np.ndarray:
    """Rotation + uniform scale + translation taking s0->d0 and s1->d1."""
    s0, s1, d0, d1 = (np.asarray(p, np.float64) for p in (s0, s1, d0, d1))
    vs, vd = s1 - s0, d1 - d0
    ls = max(float(np.linalg.norm(vs)), 1e-6)
    scale = float(np.linalg.norm(vd)) / ls
    ang = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    m = np.array([[c, -s], [s, c]])
    t = d0 - m @ s0
    return np.hstack([m, t[:, None]])


@dataclass
class Piece:
    key: str               # triangle index "t0".. or limb name
    region: str
    x0: int
    y0: int
    patch: np.ndarray      # premultiplied BGRA uint8, zero outside the piece


@dataclass
class Layer:
    """Premultiplied BGRA garment layer in camera pixels plus its dirty box."""
    img: np.ndarray
    roi: list = field(default_factory=lambda: [0, 0, 0, 0])  # x0, y0, x1, y1 (empty when x1<=x0)

    @classmethod
    def empty(cls, size: tuple[int, int]) -> "Layer":
        return cls(np.zeros((size[1], size[0], 4), np.uint8), [0, 0, 0, 0])

    def clear(self) -> None:
        x0, y0, x1, y1 = self.roi
        if x1 > x0 and y1 > y0:
            self.img[y0:y1, x0:x1] = 0
        self.roi = [0, 0, 0, 0]

    def grow(self, x0, y0, x1, y1) -> None:
        if self.roi[2] <= self.roi[0]:
            self.roi = [x0, y0, x1, y1]
        else:
            self.roi = [min(self.roi[0], x0), min(self.roi[1], y0), max(self.roi[2], x1), max(self.roi[3], y1)]

    @property
    def is_empty(self) -> bool:
        return self.roi[2] <= self.roi[0] or self.roi[3] <= self.roi[1]


def premultiply(bgra: np.ndarray) -> np.ndarray:
    a = bgra[..., 3:4].astype(np.uint16)
    out = bgra.copy()
    out[..., :3] = ((bgra[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    return out


def over_into(dst: np.ndarray, src: np.ndarray) -> None:
    """dst = src over dst, both premultiplied BGRA uint8, in place."""
    a = src[..., 3:4].astype(np.float32) * (1.0 / 255.0)
    dst[...] = np.clip(src.astype(np.float32) + dst.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


class SideAssigner:
    """Sticky image-left/image-right assignment for the arm and leg chains."""

    def __init__(self, margin_px: float = 6.0):
        self.margin = margin_px
        self.upper_swap = False   # True: MediaPipe 'left' chain is on image-left
        self.lower_swap = False

    def _decide(self, current: bool, x_left: float, x_right: float) -> bool:
        d = x_left - x_right   # mp-left minus mp-right
        if d < -self.margin:
            return True
        if d > self.margin:
            return False
        return current

    def assign(self, kp: dict) -> dict:
        self.upper_swap = self._decide(self.upper_swap, kp["left_shoulder"][0], kp["right_shoulder"][0])
        self.lower_swap = self._decide(self.lower_swap, kp["left_hip"][0], kp["right_hip"][0])
        out = {}
        for part, swap in (("shoulder", self.upper_swap), ("elbow", self.upper_swap), ("wrist", self.upper_swap),
                           ("hip", self.lower_swap), ("knee", self.lower_swap), ("ankle", self.lower_swap)):
            il, ir = ("left", "right") if swap else ("right", "left")
            out["il_" + part] = f"{il}_{part}"
            out["ir_" + part] = f"{ir}_{part}"
        return out


class ViewWarper:
    """Precomputed pieces for one garment view, and the per-frame warp."""

    def __init__(self, view, category: str, sleeve_length: str, cfg):
        self.cfg = cfg
        self.category = category
        self.long_sleeve = sleeve_length == "long"
        self.view_name = view.name
        img = view.image
        self.h, self.w = img.shape[:2]
        self.premul = premultiply(img)
        self.alpha = img[..., 3]
        self.src = self._image_side_anchors(view.anchors)
        self.pieces: list[Piece] = []
        self.mode = "similarity"
        self.sim_src = self._similarity_axis_src(view.anchors)
        if view.name in ("front", "back") and self._mesh_ready():
            try:
                self._build_mesh()
                self.mode = "mesh"
            except ValueError as e:
                print(f"[warp] {view.name}: mesh build failed ({e}); using similarity")
                self.pieces = []

    # -- source geometry -------------------------------------------------
    def _image_side_anchors(self, anchors: dict) -> dict:
        """Rename wearer-left/right anchors to image-left (l) / image-right (r)."""
        pairs = SIDE_PAIRS_SHIRT if self.category == "shirt" else SIDE_PAIRS_PANTS
        ref = "shoulder" if self.category == "shirt" else "waist"
        flip = False
        if f"left_{ref}" in anchors and f"right_{ref}" in anchors:
            flip = anchors[f"left_{ref}"][0] < anchors[f"right_{ref}"][0]
        out = {}
        for p in pairs:
            a, b = anchors.get(f"left_{p}"), anchors.get(f"right_{p}")
            il, ir = (a, b) if flip else (b, a)
            if il is not None:
                out[p + "_l"] = np.asarray(il, np.float64)
            if ir is not None:
                out[p + "_r"] = np.asarray(ir, np.float64)
        for k in ("collar", "crotch"):
            if k in anchors:
                out[k] = np.asarray(anchors[k], np.float64)
        return out

    def _mesh_ready(self) -> bool:
        s = self.src
        if self.category == "shirt":
            need = ["shoulder_l", "shoulder_r", "armpit_l", "armpit_r", "hem_l", "hem_r", "collar",
                    "sleeve_end_l", "sleeve_end_r"]
        else:
            need = ["waist_l", "waist_r", "crotch", "knee_l", "knee_r", "ankle_l", "ankle_r"]
        if any(k not in s for k in need):
            return False
        # Narrow strips (side photos) have near-collinear anchors: refuse the mesh.
        if self.category == "shirt":
            width = np.linalg.norm(s["shoulder_l"] - s["shoulder_r"])
            height = np.linalg.norm((s["shoulder_l"] + s["shoulder_r"]) / 2 - (s["hem_l"] + s["hem_r"]) / 2)
        else:
            width = np.linalg.norm(s["waist_l"] - s["waist_r"])
            height = np.linalg.norm((s["waist_l"] + s["waist_r"]) / 2 - (s["ankle_l"] + s["ankle_r"]) / 2)
        return width > 0.25 * height and height > 10

    def _similarity_axis_src(self, anchors: dict):
        def mean(names):
            pts = [anchors[n] for n in names if n in anchors]
            return np.mean(pts, axis=0).astype(np.float64) if pts else None
        if self.category == "shirt":
            top = mean(["left_shoulder", "right_shoulder"])
            if top is None:
                top = mean(["collar"])
            bot = mean(["left_hem", "right_hem"])
        else:
            top = mean(["left_waist", "right_waist"])
            bot = mean(["left_ankle", "right_ankle"])
            if bot is None:
                bot = mean(["left_knee", "right_knee"])
        if top is None or bot is None or np.linalg.norm(bot - top) < 10:
            # No usable anchors: span the opaque pixels top to bottom.
            ys, xs = np.nonzero(self.alpha > 20)
            if len(ys) == 0:
                return np.array([self.w / 2, 0.0]), np.array([self.w / 2, float(self.h)])
            cx = float(xs.mean())
            top = np.array([cx, float(ys.min())])
            bot = np.array([cx, float(ys.max())])
        return top, bot

    def _measure_halfwidth(self, p, direction, max_d) -> float | None:
        """March from p along direction until the garment alpha ends."""
        d = _unit(direction)
        steps = np.arange(1, int(max_d) + 1, dtype=np.float64)
        xs = np.round(p[0] + d[0] * steps).astype(int)
        ys = np.round(p[1] + d[1] * steps).astype(int)
        ok = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h)
        if not ok.any():
            return None
        inside = self.alpha[ys[ok], xs[ok]] > 127
        if not inside.any():
            return None
        if inside.all():
            return float(max_d)
        return float(np.argmin(inside) + 1)

    def _build_mesh(self) -> None:
        s = self.src
        labels = np.full((self.h, self.w), -1, np.int32)
        opaque = self.alpha > 8
        self.keys: list[str] = []
        self.regions: list[str] = []
        self.dilate_keys: set[str] = set()

        def add_poly(key, region, poly, dilate=False):
            idx = len(self.keys)
            self.keys.append(key)
            self.regions.append(region)
            if dilate:
                self.dilate_keys.add(key)
            cv2.fillPoly(labels, [np.round(np.asarray(poly)).astype(np.int32)], idx)
            return idx

        if self.category == "shirt":
            s["C"] = s["collar"]
            s["Sl"], s["Sr"] = s["shoulder_l"], s["shoulder_r"]
            s["Al"], s["Ar"] = s["armpit_l"], s["armpit_r"]
            s["Hl"], s["Hr"] = s["hem_l"], s["hem_r"]
            s["T"] = (s["Al"] + s["Ar"] + s["Hl"] + s["Hr"]) / 4.0
            self.src_scale = float(np.linalg.norm((s["Sl"] + s["Sr"]) / 2 - (s["Hl"] + s["Hr"]) / 2))
            # sleeves first so the torso triangles win on the shared armhole edge
            self.sleeve_geo = {}
            for side in ("l", "r"):
                S, A, E = s["S" + side], s["A" + side], s["sleeve_end_" + side]
                root = (S + A) / 2
                axis = _unit(E - root)
                n = _perp(axis)
                if np.dot(n, S - A) < 0:
                    n = -n                      # n points to the shoulder side of the cuff
                root_len = float(np.linalg.norm(S - A))
                wa = self._measure_halfwidth(E, n, root_len)
                wb = self._measure_halfwidth(E, -n, root_len)
                if wa is None or wb is None or min(wa, wb) < 0.1 * root_len:
                    # anchor not at the cuff centre: split whatever width was found evenly
                    total = (wa or 0.0) + (wb or 0.0)
                    wa = wb = total / 2 if total > 0.3 * root_len else 0.4 * root_len
                m = 0.12 * root_len + 4
                ca = E + n * (wa + m) + axis * m
                cb = E - n * (wb + m) + axis * m
                # extend the root edge outward a little so sleeve-cap pixels belong to the sleeve
                Sx = S + n * m * 0.5
                poly = [Sx, A, cb, ca]
                self.sleeve_geo[side] = (root, E)
                if self.long_sleeve:
                    f = self.cfg.long_sleeve_elbow_frac
                    K = root + (E - root) * f
                    kA, kB = K + n * (wa + m) * 1.25, K - n * (wb + m) * 1.25
                    add_poly("sleeve_up_" + side, "sleeve_up_" + side, [Sx, A, kB, kA], dilate=True)
                    add_poly("sleeve_lo_" + side, "sleeve_lo_" + side, [kA, kB, cb, ca], dilate=True)
                    self.sleeve_geo[side] = (root, E, K)
                else:
                    add_poly("sleeve_" + side, "sleeve_" + side, poly, dilate=True)
            for i, tri in enumerate(SHIRT_TRIS):
                pts = [s[k] for k in tri]
                if abs(_tri_area(*pts)) < 4.0:
                    raise ValueError(f"degenerate source triangle {tri}")
                add_poly(f"t{i}", "torso", pts)
        else:
            # Everything above the crotch line is the "waist" piece (one affine through the
            # waist corners and crotch); below it, each leg is a thigh and a shin segment
            # hinged on the crotch line and at the knee.
            Wl, Wr, X = s["waist_l"], s["waist_r"], s["crotch"]
            self.src_scale = float(np.linalg.norm(Wl - Wr))
            across = _unit(Wr - Wl)
            down = _perp(across)
            if np.dot(down, X - (Wl + Wr) / 2) < 0:
                down = -down
            w_mid = (Wl + Wr) / 2
            self.src_len = float(np.dot((s["ankle_l"] + s["ankle_r"]) / 2 - w_mid, down))
            if self.src_len < 10:
                raise ValueError("ankles are not below the waist")
            self.rise_frac = float(np.dot(X - w_mid, down)) / self.src_len
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            side_val = (xx - X[0]) * across[0] + (yy - X[1]) * across[1]
            below_crotch = ((xx - X[0]) * down[0] + (yy - X[1]) * down[1]) > 0
            self.leg_geo = {}
            for side, W, on_side in (("l", Wl, side_val <= 0), ("r", Wr, side_val > 0)):
                K, N = s["knee_" + side], s["ankle_" + side]
                P = X + across * (np.dot(W - X, across) / 2)   # thigh root: mid-leg on the crotch line
                self.leg_geo[side] = (P, K, N)
                v = _unit(N - P)
                below_knee = ((xx - K[0]) * v[0] + (yy - K[1]) * v[1]) > 0
                leg = on_side & below_crotch
                for key, sel in (("thigh_" + side, leg & ~below_knee), ("shin_" + side, leg & below_knee)):
                    labels[sel] = len(self.keys)
                    self.keys.append(key)
                    self.regions.append(key)
                    self.dilate_keys.add(key)
            labels[~below_crotch] = len(self.keys)
            self.keys.append("waist")
            self.regions.append("waist")
            self.dilate_keys.add("waist")

        # Opaque pixels left unlabeled go to the nearest labeled piece.
        lab = np.where(opaque, labels, -1)
        if (lab >= 0).sum() == 0:
            raise ValueError("no garment pixels inside the anchor mesh")
        unl = opaque & (lab < 0)
        if unl.any():
            src = (lab < 0).astype(np.uint8)
            _, nearest = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            zero_ids = lab.ravel()[np.flatnonzero(src.ravel() == 0)]
            lut = np.concatenate([[-1], zero_ids]).astype(np.int32)
            lab = np.where(unl, lut[nearest], lab)
        lab[~opaque] = -1
        self.labels = lab
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for i, key in enumerate(self.keys):
            m = (lab == i).astype(np.uint8)
            if key in self.dilate_keys:
                m = cv2.dilate(m, kernel) & opaque.astype(np.uint8)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            x0, x1 = max(0, xs.min() - 2), min(self.w, xs.max() + 3)
            y0, y1 = max(0, ys.min() - 2), min(self.h, ys.max() + 3)
            patch = self.premul[y0:y1, x0:x1] * m[y0:y1, x0:x1, None]
            self.pieces.append(Piece(key, self.regions[i], int(x0), int(y0), np.ascontiguousarray(patch)))

    # -- body targets ----------------------------------------------------
    def targets(self, body, names: dict) -> dict | None:
        cfg = self.cfg
        kp = body.kp
        g = lambda n: np.asarray(kp[names[n]], np.float64)  # noqa: E731
        try:
            Sl, Sr, Hl, Hr = g("il_shoulder"), g("ir_shoulder"), g("il_hip"), g("ir_hip")
        except KeyError:
            return None
        s_mid, h_mid = (Sl + Sr) / 2, (Hl + Hr) / 2
        up = s_mid - h_mid
        torso_len = float(np.linalg.norm(up))
        if torso_len < 10:
            return None
        u = up / torso_len
        k = cfg.garment_scale
        center = (s_mid + h_mid) / 2
        sc = lambda p: center + (p - center) * k  # noqa: E731
        t = {"torso_len": torso_len * k, "s_mid": sc(s_mid), "h_mid": sc(h_mid), "up": u}
        if self.category == "shirt":
            ws = cfg.torso_width_scale
            Sl2, Sr2 = s_mid + (Sl - s_mid) * ws, s_mid + (Sr - s_mid) * ws
            Hl2, Hr2 = h_mid + (Hl - h_mid) * ws, h_mid + (Hr - h_mid) * ws
            t["Sl"], t["Sr"] = sc(Sl2), sc(Sr2)
            t["Al"] = sc(Sl2 + cfg.armpit_drop * (Hl2 - Sl2))
            t["Ar"] = sc(Sr2 + cfg.armpit_drop * (Hr2 - Sr2))
            t["Hl"] = sc(Hl2 - u * cfg.hem_drop * torso_len)
            t["Hr"] = sc(Hr2 - u * cfg.hem_drop * torso_len)
            t["C"] = sc(s_mid + u * cfg.collar_raise * torso_len)
            t["T"] = (t["Al"] + t["Ar"] + t["Hl"] + t["Hr"]) / 4.0
            t["top"], t["bot"] = (t["Sl"] + t["Sr"]) / 2, (t["Hl"] + t["Hr"]) / 2
            for side in ("l", "r"):
                for part in ("elbow", "wrist"):
                    name = names[f"i{side}_{part}"]
                    if body.ok(name, cfg.min_visibility):
                        t[f"{part}_{side}"] = sc(g(f"i{side}_{part}"))
        else:
            wsc = cfg.pants_width_scale
            Wl = h_mid + (Hl - h_mid) * wsc + u * cfg.waist_raise * torso_len
            Wr = h_mid + (Hr - h_mid) * wsc + u * cfg.waist_raise * torso_len
            t["Wl"], t["Wr"] = sc(Wl), sc(Wr)
            for side in ("l", "r"):
                for part in ("knee", "ankle"):
                    name = names[f"i{side}_{part}"]
                    if body.ok(name, cfg.min_visibility * 0.6):
                        t[f"{part}_{side}"] = sc(g(f"i{side}_{part}"))
            # Missing ankles: extend the thigh; missing knees: hang straight down.
            for side, H in (("l", Hl), ("r", Hr)):
                if f"knee_{side}" not in t:
                    t[f"knee_{side}"] = sc(H - u * 0.95 * torso_len)
                if f"ankle_{side}" not in t:
                    kn = t[f"knee_{side}"]
                    t[f"ankle_{side}"] = kn + (kn - sc(H)) * 0.95
            t["top"] = (t["Wl"] + t["Wr"]) / 2
            t["bot"] = (t["ankle_l"] + t["ankle_r"]) / 2
            # crotch: the garment's own rise (crotch depth / waist-to-ankle length), plus an optional extra drop
            t["len"] = max(float(np.dot(t["bot"] - t["top"], -u)), 0.5 * torso_len * k)
            thigh = 0.5 * (np.linalg.norm(t["knee_l"] - sc(Hl)) + np.linalg.norm(t["knee_r"] - sc(Hr)))
            rise = getattr(self, "rise_frac", 0.25)
            t["X"] = t["top"] - u * (rise * t["len"] + cfg.crotch_drop * thigh)
        return t

    # -- per-frame transforms --------------------------------------------
    def _piece_affines(self, t: dict) -> dict | None:
        cfg = self.cfg
        s = self.src
        out = {}
        if self.category == "shirt":
            body_scale = t["torso_len"] / max(self.src_scale, 1e-6)
            bad = 0
            for i, tri in enumerate(SHIRT_TRIS):
                src = np.float32([s[k] for k in tri])
                dst = np.float32([t[k] for k in tri])
                a_s, a_d = _tri_area(*src), _tri_area(*dst)
                A = cv2.getAffineTransform(src, dst)
                sv = np.linalg.svd(A[:, :2], compute_uv=False) / body_scale
                if (a_s * a_d <= 0 or abs(a_d) < 0.02 * abs(a_s) * body_scale ** 2
                        or sv.max() > 4.0 or sv.min() < 0.04):
                    bad += 1
                out[f"t{i}"] = A
            if bad >= 2:
                return None
            for side in ("l", "r"):
                root_s, E_s = self.sleeve_geo[side][0], self.sleeve_geo[side][1]
                Sd, Ad = t["S" + side], t["A" + side]
                root_d = (Sd + Ad) / 2
                rest = self._rest_dir(t, E_s - root_s)
                end_key = "wrist_" + side if self.long_sleeve else "elbow_" + side
                lo, hi = 0.25, cfg.max_limb_stretch
                if self.long_sleeve:
                    K_s = self.sleeve_geo[side][2]
                    elbow = t.get("elbow_" + side, root_d + rest * np.linalg.norm(K_s - root_s) * body_scale)
                    wrist = t.get(end_key, elbow + rest * np.linalg.norm(E_s - K_s) * body_scale)
                    out["sleeve_up_" + side] = limb_affine(root_s, K_s, root_d, elbow, body_scale, lo * body_scale, hi * body_scale)
                    out["sleeve_lo_" + side] = limb_affine(K_s, E_s, elbow, wrist, body_scale, lo * body_scale, hi * body_scale)
                else:
                    end = t.get(end_key, root_d + rest * np.linalg.norm(E_s - root_s) * body_scale)
                    out["sleeve_" + side] = limb_affine(root_s, E_s, root_d, end, body_scale, lo * body_scale, hi * body_scale)
        else:
            Wl, Wr, X = s["waist_l"], s["waist_r"], s["crotch"]
            src = np.float32([Wl, Wr, X])
            dst = np.float32([t["Wl"], t["Wr"], t["X"]])
            a_s, a_d = _tri_area(*src), _tri_area(*dst)
            # vertical scale (waist-to-ankle) is stable while turning; width shrinks side-on
            scale_v = t["len"] / max(self.src_len, 1e-6)
            if a_s * a_d <= 0 or abs(a_d) < 0.03 * abs(a_s) * scale_v ** 2:
                return None
            A = cv2.getAffineTransform(src, dst)
            out["waist"] = A
            width = float(np.linalg.norm(t["Wl"] - t["Wr"])) / max(self.src_scale, 1e-6)
            across = float(np.clip(width, 0.5 * scale_v, 1.5 * scale_v))
            lo, hi = 0.25 * scale_v, cfg.max_limb_stretch * scale_v
            for side in ("l", "r"):
                P_s, K_s, N_s = self.leg_geo[side]
                P_d = A[:, :2] @ P_s + A[:, 2]   # thigh root moves with the waist piece: no gap at the hip
                out["thigh_" + side] = limb_affine(P_s, K_s, P_d, t["knee_" + side], across, lo, hi)
                out["shin_" + side] = limb_affine(K_s, N_s, t["knee_" + side], t["ankle_" + side], across, lo, hi)
        return out

    def _rest_dir(self, t: dict, src_vec: np.ndarray) -> np.ndarray:
        """Direction a limb hangs when its end joint is not visible: src angle, rotated with the torso."""
        src_up = -_unit(self.sim_src[1] - self.sim_src[0])
        ang = np.arctan2(t["up"][1], t["up"][0]) - np.arctan2(src_up[1], src_up[0])
        c, s_ = np.cos(ang), np.sin(ang)
        v = _unit(src_vec)
        return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])

    def similarity(self, t: dict) -> np.ndarray:
        return similarity_affine(self.sim_src[0], self.sim_src[1], t["top"], t["bot"])

    # -- rendering -------------------------------------------------------
    def render(self, layer: Layer, body, names: dict) -> str | None:
        """Draw this view onto `layer` (over whatever is there). Returns the mode used."""
        t = self.targets(body, names)
        if t is None:
            return None
        size = (layer.img.shape[1], layer.img.shape[0])
        affines = self._piece_affines(t) if self.mode == "mesh" else None
        if affines is None:
            A = self.similarity(t)
            buf = _warp_patch(self.premul, 0, 0, A, size)
            if buf is None:
                return None
            (bx0, by0, bx1, by1), w = buf
            over_into(layer.img[by0:by1, bx0:bx1], w)
            layer.grow(bx0, by0, bx1, by1)
            return "similarity"
        # Torso triangles add (they share edges exactly); hinged limb segments stack with "over",
        # upper segments on top so a bent elbow or knee overlaps cleanly.
        if self.category == "shirt":
            order = ["torso", "sleeve_lo_l", "sleeve_lo_r", "sleeve_up_l", "sleeve_up_r", "sleeve_l", "sleeve_r"]
        else:
            order = ["shin_l", "shin_r", "thigh_l", "thigh_r", "waist"]
        for region in order:
            warped = []
            for p in self.pieces:
                if p.region != region or p.key not in affines:
                    continue
                r = _warp_patch(p.patch, p.x0, p.y0, affines[p.key], size)
                if r is not None:
                    warped.append(r)
            if not warped:
                continue
            x0 = min(b[0][0] for b in warped)
            y0 = min(b[0][1] for b in warped)
            x1 = max(b[0][2] for b in warped)
            y1 = max(b[0][3] for b in warped)
            acc = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            for (bx0, by0, bx1, by1), w in warped:
                sub = acc[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
                cv2.add(sub, w, dst=sub)
            over_into(layer.img[y0:y1, x0:x1], acc)
            layer.grow(x0, y0, x1, y1)
        return "mesh"


def _warp_patch(patch: np.ndarray, px0: int, py0: int, A: np.ndarray, size: tuple[int, int]):
    """Forward-warp a source patch (located at px0,py0 in the garment image) by A."""
    h, w = patch.shape[:2]
    A = np.asarray(A, np.float64)
    # affine from patch-local coords to destination
    Al = A.copy()
    Al[:, 2] = A[:, 2] + A[:, :2] @ np.array([px0, py0], np.float64)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64)
    pts = corners @ Al.T
    if not np.all(np.isfinite(pts)):
        return None
    x0 = max(0, int(np.floor(pts[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - 1)
    x1 = min(size[0], int(np.ceil(pts[:, 0].max())) + 2)
    y1 = min(size[1], int(np.ceil(pts[:, 1].max())) + 2)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    if (x1 - x0) * (y1 - y0) > 4 * size[0] * size[1]:
        return None
    T = Al.copy()
    T[:, 2] -= [x0, y0]
    out = cv2.warpAffine(patch, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return (x0, y0, x1, y1), out
