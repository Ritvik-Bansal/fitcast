#!/usr/bin/env python
"""Turn any online garment (product page or image URL, or a saved image file) into a FitCast outfit folder.

    .venv/bin/python scripts/garment_from_url.py <URL|file> [--part top|bottom|auto] [--name "Black Bomber"]
        [--sleeve short|long|none] [--pair-with <outfit folder>] [--out assets/garments/90_black_bomber]

Steps:
  1. fetch     browser User-Agent + retries. An HTML page yields its product image (og:image / twitter:image
               meta, JSON-LD "image", then the largest <img> by declared size); direct image URLs, file:// URLs
               and local paths are used as they are. Sites that block scripts (HTTP 403/503, captcha pages)
               get a clear "save the image and pass the file path" message.
  2. cut out   scripts/fetch_garments.garment_rgba (near-white / border-colour threshold -> cv2.grabCut ->
               crop); images that already have transparency keep their alpha. Dark garments are lifted with
               a gamma (the projector can only add light). layout_top / layout_bottom fit the canvas and
               add the 2 px feather.
  3. classify  --part auto: title / URL keywords, else the cut-out aspect ratio. Sleeve from keywords,
               else the shape estimate of scripts/prepare_garments.estimate_sleeve.
  4. pair      the other part comes from an existing real-photo outfit (first folder 14..89 that has it,
               procedural garments skipped; --pair-with overrides). Its PNGs are copied as they are
               (already colour-boosted).
  5. write     prepare_garments.write_outfit (colour boost, synthesized back, sleeve tile, thumb, meta.yaml)
               or fetch_garments_bottoms.write_pair for a new bottom; the downloaded image is kept as
               `source.<ext>`; a line is appended to SOURCES.md under "## From URL"; the folder is loaded
               back with fitcast.garments.load_outfit and printed.

Default output: assets/garments/<next number >= 90>_<slug>. An existing folder is only replaced when it was
made by this script (it holds a source.* file) or with --force.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import logging
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fitcast.garments import ensure_rgba, load_outfit  # noqa: E402
from fetch_garments import BG_DIST, PAIRS as PHOTO_PAIRS, garment_rgba, layout_bottom, layout_top, torso_region  # noqa: E402
from prepare_garments import _hex, _median_color, estimate_sleeve, load_config, mean_luminance, write_outfit  # noqa: E402
import procedural_garments as pgen  # noqa: E402

log = logging.getLogger("fitcast.garment_from_url")

# Tunables; a `garments.from_url:` mapping in config.yaml overrides any of them.
URL_DEFAULTS: dict = {
    "timeout_s": 30.0,
    "retries": 3,
    "retry_backoff_s": 1.5,        # sleep = backoff * attempt number
    "min_image_px": 200,           # candidates whose shorter side is below this are icons, not product shots
    "max_candidates": 8,           # page images tried before giving up
    "transparent_frac": 0.02,      # an image with at least this share of alpha < 128 is already cut out
    "bottom_aspect": 1.3,          # cut-out height/width at or above this is a bottom when no keyword decides
    "name_max_chars": 28,
    "dark_gamma_min": 0.35,        # strongest lift for garments darker than garments.min_mean_luminance
    "edge_erode_px": 2,            # alpha erosion at photo resolution: drops background fringe (white halo)
    "first_prefix": 90,            # default output folders are numbered from here
    "pair_min_prefix": 14,         # first real-photo outfit folder
    "pair_max_prefix": 89,         # folders above this are outputs of this script, never partners
}

SOURCES_HEADING = "## From URL"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/129.0.0.0 Safari/537.36")
TOOL_UA = "FitCastGarmentFromURL/1.0 (offline asset prep for a local projection-mapping demo)"
ACCEPT = "text/html,application/xhtml+xml,image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
BLOCK_CODES = (401, 403, 429, 503)
BLOCK_MARKERS = ("robot check", "captcha", "access denied", "just a moment", "are you a human",
                 "pardon our interruption", "attention required", "request blocked", "bot detection")

TOP_WORDS = ("tee", "tshirt", "shirt", "hoodie", "jacket", "sweater", "jersey", "polo", "sweatshirt", "blouse",
             "cardigan", "tank", "top", "coat", "bomber", "overshirt", "pullover", "crewneck", "fleece", "vest",
             "parka", "windbreaker", "henley", "camisole", "anorak", "longsleeve")
BOTTOM_WORDS = ("jeans", "pant", "pants", "trouser", "trousers", "jogger", "joggers", "shorts", "cargo", "cargos",
                "chino", "chinos", "sweatpants", "trackpants", "leggings", "skirt", "slacks", "bottoms")
SLEEVE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (     # first rule with a hit wins
    ("short", ("short sleeve", "short sleeved", "shortsleeve")),
    ("long", ("long sleeve", "long sleeved", "longsleeve")),
    ("none", ("sleeveless", "tank", "singlet", "vest", "camisole", "jersey")),
    ("long", ("hoodie", "sweater", "sweatshirt", "jacket", "coat", "cardigan", "bomber", "fleece", "pullover",
              "parka", "overshirt", "windbreaker", "anorak", "long")),
)
IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp|avif|gif|bmp|tiff?)$", re.I)


class BlockedError(RuntimeError):
    """The site refused the download (HTTP 401/403/429/503 or a captcha / bot-check page)."""


def _blocked_msg(where: str, why: str) -> str:
    return (f"{where} blocked the download ({why}). Shops often refuse scripts: open the page in a browser, "
            f"save the product image (right-click > Save Image As...) and run "
            f"`.venv/bin/python scripts/garment_from_url.py ~/Downloads/<image>.jpg --name \"...\"` instead.")


def url_cfg(cfg: dict) -> dict:
    """URL_DEFAULTS completed with any `from_url:` overrides in `cfg` (config["garments"])."""
    return {**URL_DEFAULTS, **(cfg.get("from_url") or {})}


# ----------------------------------------------------------------------------- fetch

@dataclass
class Fetched:
    data: bytes
    url: str                 # final URL (after redirects); file:// URI for local files
    content_type: str = ""


def fetch_bytes(src: str, ucfg: dict, referer: str | None = None) -> Fetched:
    """Read a local path / file:// URL, or GET an http(s) URL with retries.

    Raises BlockedError for 401/403/429/503 after the retries, RuntimeError for other failures."""
    p = Path(src).expanduser()
    if "://" not in src:
        if not p.is_file():
            raise RuntimeError(f"{src}: no such file (and not a URL)")
        return Fetched(p.read_bytes(), p.resolve().as_uri())
    parsed = urllib.parse.urlparse(src)
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
        if not path.is_file():
            raise RuntimeError(f"{src}: no such file")
        return Fetched(path.read_bytes(), src)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"{src}: unsupported URL scheme {parsed.scheme!r}")
    retries = int(ucfg["retries"])
    err: str = ""
    blocked: str = ""
    for attempt in range(retries):
        ua = BROWSER_UA if attempt % 2 == 0 else TOOL_UA   # some hosts (Wikimedia) prefer an honest tool UA
        headers = {"User-Agent": ua, "Accept": ACCEPT, "Accept-Language": "en-US,en;q=0.9"}
        if referer:
            headers["Referer"] = referer
        try:
            with urllib.request.urlopen(urllib.request.Request(src, headers=headers),
                                        timeout=float(ucfg["timeout_s"])) as r:
                data = r.read()
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    data = gzip.decompress(data)
                return Fetched(data, r.geturl(), r.headers.get("Content-Type", "") or "")
        except urllib.error.HTTPError as e:
            if e.code not in BLOCK_CODES:
                raise RuntimeError(f"{src}: HTTP {e.code} {e.reason}") from e
            blocked = f"HTTP {e.code}"
            log.warning("%s: HTTP %d (attempt %d/%d)", parsed.netloc, e.code, attempt + 1, retries)
        except (urllib.error.URLError, OSError, ValueError) as e:
            err = str(e)
            log.warning("%s: %s (attempt %d/%d)", parsed.netloc, e, attempt + 1, retries)
        if attempt + 1 < retries:
            time.sleep(float(ucfg["retry_backoff_s"]) * (attempt + 1))
    if blocked:
        raise BlockedError(_blocked_msg(parsed.netloc, blocked))
    raise RuntimeError(f"download failed after {retries} tries: {src}: {err}")


def decode_image(data: bytes) -> np.ndarray | None:
    """BGR or BGRA uint8 image from encoded bytes (JPEG/PNG/WebP/AVIF/...); None when OpenCV cannot decode it."""
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = ensure_rgba(img) if (img.dtype != np.uint8 or img.ndim == 2 or img.shape[2] != 3) else img
    return img


def _looks_like_html(f: Fetched) -> bool:
    head = f.data[:2048].lstrip().lower()
    return "html" in f.content_type.lower() or head.startswith((b"<!doctype", b"<html", b"<head", b"<meta", b"<"))


# ----------------------------------------------------------------------------- HTML

class _PageParser(HTMLParser):
    """Collects meta tags, <title>, JSON-LD blocks, <link rel=image_src> and <img> candidates."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.ld: list[str] = []
        self.links: list[str] = []
        self.imgs: list[tuple[int, int, str]] = []   # (declared area, order, url)
        self._in_title = False
        self._in_ld = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (d.get("property") or d.get("name") or d.get("itemprop") or "").lower()
            if key and d.get("content") and key not in self.meta:
                self.meta[key] = d["content"].strip()
        elif tag == "title":
            self._in_title = True
        elif tag == "script" and "ld+json" in d.get("type", "").lower():
            self._in_ld, self._buf = True, []
        elif tag == "link" and "image_src" in d.get("rel", "").lower() and d.get("href"):
            self.links.append(d["href"])
        elif tag == "img":
            self._img(d)

    def _img(self, d: dict[str, str]) -> None:
        def num(v: str) -> int:
            m = re.match(r"\s*(\d+)", v or "")
            return int(m.group(1)) if m else 0
        area = num(d.get("width", "")) * num(d.get("height", ""))
        dyn = d.get("data-a-dynamic-image")          # Amazon: {"url": [w, h], ...}
        if dyn:
            try:
                for u, wh in json.loads(dyn).items():
                    self.imgs.append((int(wh[0]) * int(wh[1]), len(self.imgs), u))
            except (ValueError, TypeError, IndexError, AttributeError):
                pass
        best_srcset = ""
        for part in (d.get("srcset") or d.get("data-srcset") or "").split(","):
            bits = part.strip().split()
            if len(bits) >= 1 and bits[0]:
                w = num(bits[1]) if len(bits) > 1 and bits[1].endswith("w") else 0
                if w * w > area or not best_srcset:
                    best_srcset, area = bits[0], max(area, w * w)
        url = d.get("data-old-hires") or d.get("data-zoom-image") or d.get("data-src") or best_srcset or d.get("src", "")
        if url and not url.startswith("data:"):
            self.imgs.append((area, len(self.imgs), url))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ld.append("".join(self._buf))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._in_ld:
            self._buf.append(data)


def _ld_images(obj, out: list[str], product_first: list[str]) -> None:
    """Walk a JSON-LD value; image URLs of Product nodes go to `product_first`, the rest to `out`."""
    if isinstance(obj, list):
        for v in obj:
            _ld_images(v, out, product_first)
        return
    if not isinstance(obj, dict):
        return
    types = obj.get("@type", "")
    is_product = "product" in (" ".join(types) if isinstance(types, list) else str(types)).lower()
    img = obj.get("image")
    urls: list[str] = []
    for v in (img if isinstance(img, list) else [img]):
        if isinstance(v, str):
            urls.append(v)
        elif isinstance(v, dict):
            u = v.get("contentUrl") or v.get("url")
            if isinstance(u, str):
                urls.append(u)
    (product_first if is_product else out).extend(urls)
    for k, v in obj.items():
        if k != "image" and isinstance(v, (dict, list)):
            _ld_images(v, out, product_first)


@dataclass
class Page:
    title: str
    candidates: list[str] = field(default_factory=list)   # absolute image URLs, best first


def parse_page(html_text: str, base_url: str) -> Page:
    """Product title and ordered image candidates of an HTML page (URLs resolved against `base_url`)."""
    p = _PageParser()
    try:
        p.feed(html_text)
        p.close()
    except (AssertionError, ValueError) as e:      # malformed markup: keep what was parsed
        log.warning("HTML parse stopped early: %s", e)
    m = p.meta
    cands = [m[k] for k in ("og:image:secure_url", "og:image", "og:image:url", "twitter:image",
                            "twitter:image:src") if m.get(k)]
    ld_other: list[str] = []
    ld_product: list[str] = []
    for block in p.ld:
        try:
            _ld_images(json.loads(block), ld_other, ld_product)
        except ValueError:
            continue
    cands += ld_product + ld_other + p.links
    if m.get("image"):                              # itemprop="image"
        cands.append(m["image"])
    cands += [u for _, _, u in sorted(p.imgs, key=lambda t: (-t[0], t[1]))]
    seen: set[str] = set()
    out = []
    for c in cands:
        u = urllib.parse.urljoin(base_url, c.strip())
        if u.startswith("//"):
            u = "https:" + u
        if u not in seen and not u.lower().split("?")[0].endswith(".svg"):
            seen.add(u)
            out.append(u)
    title = m.get("og:title") or m.get("twitter:title") or p.title
    return Page(re.sub(r"\s+", " ", title).strip(), out)


# ----------------------------------------------------------------------------- resolve

@dataclass
class Source:
    image: np.ndarray        # BGR or BGRA uint8
    data: bytes              # encoded image bytes as downloaded
    image_url: str
    page_url: str | None     # None for direct image URLs / files
    title: str


def resolve_image(src: str, ucfg: dict) -> Source:
    """Fetch `src`; return the product image (directly, or the best candidate of an HTML page).

    Raises BlockedError when the site refuses, RuntimeError when no usable image is found."""
    f = fetch_bytes(src, ucfg)
    img = decode_image(f.data)
    if img is not None:
        return Source(img, f.data, f.url, None, "")
    if not _looks_like_html(f):
        raise RuntimeError(f"{src}: not an image OpenCV can decode (SVG/HEIC are not supported) and not an "
                           "HTML page; convert it to JPEG/PNG and pass the file path")
    m = re.search(r"charset=([\w-]+)", f.content_type)
    text = f.data.decode(m.group(1) if m else "utf-8", errors="replace")
    page = parse_page(text, f.url)
    host = urllib.parse.urlparse(f.url).netloc or src
    if any(k in page.title.lower() for k in BLOCK_MARKERS) or (
            not page.candidates and any(k in text.lower() for k in BLOCK_MARKERS)):
        raise BlockedError(_blocked_msg(host, f"bot-check page {page.title[:60]!r}"))
    if not page.candidates:
        raise RuntimeError(f"{host}: no product image found on the page (no og:image / JSON-LD / <img>); "
                           "the page is probably built by JavaScript. Save the image in a browser and pass the "
                           "file path instead.")
    log.info("page %r: %d image candidates", page.title[:70], len(page.candidates))
    min_px = int(ucfg["min_image_px"])
    for cand in page.candidates[:int(ucfg["max_candidates"])]:
        try:
            fi = fetch_bytes(cand, ucfg, referer=f.url)
        except BlockedError as e:
            raise BlockedError(f"image host: {e}") from e
        except RuntimeError as e:
            log.warning("candidate failed: %s", e)
            continue
        im = decode_image(fi.data)
        if im is None:
            log.warning("candidate %s is not a decodable image; next", cand[:120])
            continue
        if min(im.shape[:2]) < min_px:
            log.info("candidate %s is only %dx%d; next", cand[:120], im.shape[1], im.shape[0])
            continue
        log.info("product image %s (%dx%d)", fi.url[:140], im.shape[1], im.shape[0])
        return Source(im, fi.data, fi.url, f.url, page.title)
    raise RuntimeError(f"{host}: none of the first {ucfg['max_candidates']} page images was a usable product "
                       f"photo (>= {min_px} px). Save the image in a browser and pass the file path instead.")


# ----------------------------------------------------------------------------- cut-out

def cut_garment(img: np.ndarray, cfg: dict, ucfg: dict, white: int | None = None, bg_dist: float | None = None,
                gamma: float | None = None) -> np.ndarray:
    """Tightly cropped RGBA garment. Transparent PNGs keep their alpha; others go through
    fetch_garments.garment_rgba. `gamma` None = auto (lift dark garments). Raises ValueError when no
    garment separates from the background."""
    rgba = ensure_rgba(img)
    a = rgba[..., 3]
    if float((a < 128).mean()) >= float(ucfg["transparent_frac"]):
        ys, xs = np.nonzero(a > 127)
        if not len(ys):
            raise ValueError("image is fully transparent")
        out = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        log.info("image already has transparency; using its alpha")
    else:
        out = garment_rgba(np.ascontiguousarray(rgba[..., :3]),
                           int(white if white is not None else cfg["white_bg_threshold"]),
                           bg_dist=float(BG_DIST if bg_dist is None else bg_dist))
        px = int(ucfg["edge_erode_px"])
        if px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
            out[..., 3] = cv2.erode(out[..., 3], k)
    if gamma is None:
        lum = mean_luminance(out, cfg)
        target = float(cfg["min_mean_luminance"])
        gamma = 1.0
        if 0 < lum < target:   # solve (lum/255)^g = target/255, capped so blacks do not turn flat grey
            gamma = max(float(ucfg["dark_gamma_min"]), float(np.log(target / 255.0) / np.log(lum / 255.0)))
            log.info("dark garment (mean luminance %.0f < %.0f): gamma %.2f so it shows when projected",
                     lum, target, gamma)
    if gamma != 1.0:
        lut = (255.0 * (np.arange(256) / 255.0) ** gamma).round().astype(np.uint8)
        out = np.dstack([cv2.LUT(np.ascontiguousarray(out[..., :3]), lut), out[..., 3]])
    return out


# ----------------------------------------------------------------------------- classify / name

def _words(text: str) -> str:
    """Lower-case text with URL punctuation turned into spaces ("t-shirt" -> "t shirt")."""
    t = urllib.parse.unquote(text or "").lower()
    return " " + re.sub(r"[^a-z0-9']+", " ", t) + " "


def _hits(words: str, vocab: tuple[str, ...]) -> int:
    return sum(len(re.findall(rf"(?<![a-z]){re.escape(w)}(?![a-z])", words)) for w in vocab)


def classify_part(text: str, rgba: np.ndarray | None, ucfg: dict) -> str:
    """'top' or 'bottom' from title/URL keywords; ties or no keywords fall back to the cut-out aspect ratio."""
    w = _words(text)
    top, bottom = _hits(w, TOP_WORDS), _hits(w, BOTTOM_WORDS)
    if top != bottom:
        part = "top" if top > bottom else "bottom"
        log.info("part: %s (keywords top=%d bottom=%d)", part, top, bottom)
        return part
    if rgba is None:
        return "top"
    h, wd = rgba.shape[:2]
    part = "bottom" if h / max(1, wd) >= float(ucfg["bottom_aspect"]) else "top"
    log.info("part: %s (no keyword decided; cut-out aspect h/w %.2f)", part, h / max(1, wd))
    return part


def sleeve_from_text(text: str) -> str | None:
    """'short' | 'long' | 'none' from keywords, None when no keyword says."""
    w = _words(text)
    for sleeve, vocab in SLEEVE_RULES:
        if _hits(w, vocab):
            return sleeve
    return None


def clean_name(title: str, max_chars: int) -> str:
    """Short product name from a page title: site prefix/suffix and gender words dropped, cut at a word
    boundary to `max_chars`."""
    t = re.sub(r"\s+", " ", title or "").strip()
    t = re.sub(r"^(amazon\.[a-z.]+|file)\s*:\s*", "", t, flags=re.I)
    t = re.split(r"\s[|–—-]\s|:\s|\.\s", t)[0]
    t = IMAGE_EXT.sub("", t)
    t = re.sub(r"(?i)\b(men's|women's|mens|womens|men|women|unisex|kids')\s*", "", t)
    t = re.sub(r"[_]+", " ", t)
    return truncate(t, max_chars)


def truncate(text: str, max_chars: int) -> str:
    """Whitespace-collapsed `text` cut at a word boundary to at most `max_chars`."""
    t = re.sub(r"\s+", " ", text or "").strip(" ,.-")
    out = ""
    for word in t.split(" "):
        cand = f"{out} {word}".strip()
        if len(cand) > max_chars:
            break
        out = cand
    return out or t[:max_chars].strip()


def name_from_url(url: str, max_chars: int) -> str:
    """Readable name from an image URL / file name (Commons thumbnail paths keep the file name last)."""
    stem = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
    stem = re.sub(r"^\d+px-", "", IMAGE_EXT.sub("", stem))
    return clean_name(re.sub(r"[_-]+", " ", stem), max_chars)


def slug(text: str) -> str:
    """Folder-safe lower-case slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "item"


# ----------------------------------------------------------------------------- pairing / output

def _prefix(name: str) -> int | None:
    m = re.match(r"(\d+)_", name)
    return int(m.group(1)) if m else None


def _procedural(folder_name: str, need: str) -> bool:
    for oid, _, _, top_key, bottom_key in PHOTO_PAIRS:
        if oid == folder_name:
            return (top_key if need == "top" else bottom_key) in pgen.GARMENTS
    return False


def _has_part(folder: Path, need: str) -> bool:
    part = "top_front" if need == "top" else "bottom_front"
    try:
        meta = yaml.safe_load((folder / "meta.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(meta, dict) and part in (meta.get("parts") or []) and (folder / f"{part}.png").is_file()


def find_partner(garments_dir: Path, need: str, ucfg: dict, pair_with: str | None = None,
                 exclude: Path | None = None) -> Path:
    """Folder that supplies the `need` ('top' | 'bottom') part. `pair_with` (path or folder name) wins;
    otherwise the first real-photo outfit folder numbered pair_min_prefix..pair_max_prefix."""
    if pair_with:
        p = Path(pair_with).expanduser()
        cand = p if p.is_dir() else garments_dir / pair_with
        if not cand.is_dir():
            raise FileNotFoundError(f"--pair-with {pair_with}: no such outfit folder (looked in {garments_dir})")
        if not _has_part(cand, need):
            raise ValueError(f"--pair-with {cand.name} has no {need} (needs {need}_front in meta.yaml parts)")
        return cand
    lo, hi = int(ucfg["pair_min_prefix"]), int(ucfg["pair_max_prefix"])
    for folder in sorted(p for p in garments_dir.iterdir() if p.is_dir()):
        n = _prefix(folder.name)
        if n is None or not lo <= n <= hi or (exclude and folder.resolve() == exclude.resolve()):
            continue
        if _has_part(folder, need) and not _procedural(folder.name, need):
            return folder
    raise FileNotFoundError(f"no outfit folder {lo}..{hi} in {garments_dir} has a real-photo {need}; "
                            f"run scripts/fetch_garments.py or pass --pair-with <folder>")


def piece_name(outfit_name: str, part: str) -> str:
    """The `part` half of an "<Top> + <Bottom>" outfit name."""
    bits = [b.strip() for b in outfit_name.split(" + ")]
    return bits[0] if part == "top" or len(bits) == 1 else bits[-1]


def default_out(garments_dir: Path, piece: str, ucfg: dict) -> Path:
    """`<garments_dir>/<n>_<slug>`: an existing folder with the same slug numbered >= first_prefix is reused,
    otherwise the next free number >= first_prefix."""
    first = int(ucfg["first_prefix"])
    s = slug(piece)
    used = []
    for p in garments_dir.iterdir() if garments_dir.is_dir() else []:
        n = _prefix(p.name)
        if n is not None and n >= first:
            if p.name == f"{n:02d}_{s}":
                return p
            used.append(n)
    return garments_dir / f"{max(used, default=first - 1) + 1:02d}_{s}"


def _check_out(folder: Path, force: bool) -> None:
    if folder.exists() and not force and not any(folder.glob("source.*")):
        raise FileExistsError(f"{folder} exists and was not made by this script (no source.* file); "
                              "pick another --out or pass --force to replace it")


def _ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return ".avif"
    if data[:3] == b"GIF":
        return ".gif"
    return ".img"


def write_from_url_outfit(folder: Path, rgba: np.ndarray, part: str, piece: str, sleeve: str | None,
                          partner: Path, cfg: dict, size_table: str | None = None) -> str:
    """Lay out the fetched piece, write the outfit folder with the partner's other part; returns a summary."""
    partner_meta = yaml.safe_load((partner / "meta.yaml").read_text())
    other = piece_name(str(partner_meta.get("name", partner.name)), "bottom" if part == "top" else "top")
    name = f"{piece} + {other}" if part == "top" else f"{other} + {piece}"
    if part == "top":
        canvas = layout_top(rgba, cfg)
        region = torso_region(canvas, cfg)
        if sleeve is None:
            sleeve = estimate_sleeve(canvas, cfg)
            log.info("sleeve: %s (shape estimate; pass --sleeve to override)", sleeve)
        meta = dict(name=name, category="outfit", sleeve=sleeve,
                    size_table=size_table or partner_meta.get("size_table", "mens"),
                    colors=[_hex(_median_color(canvas, cfg))])
        line = write_outfit(folder.parent, folder.name, meta, cfg, top=canvas, region=region)
        out_meta = yaml.safe_load((folder / "meta.yaml").read_text())
        for p in ("bottom_front", "bottom_back"):
            if p in (partner_meta.get("parts") or []) and (partner / f"{p}.png").is_file():
                shutil.copyfile(partner / f"{p}.png", folder / f"{p}.png")
                out_meta["parts"].append(p)
        out_meta["colors"] = out_meta["colors"] + list(partner_meta.get("colors", []))[-1:]
        (folder / "meta.yaml").write_text(yaml.safe_dump(out_meta, sort_keys=False))
        return f"{folder.name:<22} parts={','.join(out_meta['parts'])} {line.split('lum ', 1)[-1]} bottom<-{partner.name}"
    from fetch_garments_bottoms import write_pair   # imported lazily: only the bottom path needs it
    if sleeve is not None and sleeve != partner_meta.get("sleeve"):
        log.warning("--sleeve %s ignored: a fetched bottom keeps the sleeve of its top (%s, from %s)",
                    sleeve, partner_meta.get("sleeve"), partner.name)
    line = write_pair(folder.parent, folder.name, name, partner, layout_bottom(rgba, cfg), None, cfg)
    if size_table:
        out_meta = yaml.safe_load((folder / "meta.yaml").read_text())
        out_meta["size_table"] = size_table
        (folder / "meta.yaml").write_text(yaml.safe_dump(out_meta, sort_keys=False))
    return line


def append_source(sources_md: Path, line: str, key: str | None = None) -> None:
    """Append `line` at the end of the "## From URL" section of SOURCES.md (section created if absent);
    earlier lines of that section containing `key` (the folder) are replaced."""
    text = sources_md.read_text() if sources_md.is_file() else "# Garment sources\n"
    lines = text.rstrip("\n").split("\n")
    if SOURCES_HEADING in lines:
        i = lines.index(SOURCES_HEADING) + 1
        while i < len(lines) and not lines[i].startswith("## "):
            if key and lines[i].startswith("- ") and key in lines[i]:
                del lines[i]
                continue
            i += 1
        while i > 0 and not lines[i - 1].strip():
            i -= 1
        lines.insert(i, line)
    else:
        lines += ["", SOURCES_HEADING, "",
                  "Made by `scripts/garment_from_url.py`. Shop images are the retailer's copyright: local demo use only.",
                  "", line]
    sources_md.write_text("\n".join(lines) + "\n")


def describe(folder: Path, cfg: dict) -> str:
    """Load the folder with fitcast.garments.load_outfit and return a human-readable verification."""
    o = load_outfit(folder, cfg)
    parts = {k: getattr(o, k) for k in ("top_front", "top_back", "sleeve_tex", "bottom_front", "bottom_back")}
    sizes = ", ".join(f"{k} {v.shape[1]}x{v.shape[0]}" for k, v in parts.items() if v is not None)
    return (f"loaded {o.id}: {o.name!r}\n  sleeve={o.sleeve} size_table={o.size_table} "
            f"top_body_region={tuple(round(v, 3) for v in o.top_body_region)}\n  {sizes}, "
            f"thumb {o.thumb.shape[1]}x{o.thumb.shape[0]}")


def make_outfit_from_url(src: str, cfg: dict, *, part: str = "auto", name: str | None = None,
                         sleeve: str | None = None, pair_with: str | None = None, out: Path | None = None,
                         garments_dir: Path | None = None, sources_md: Path | None = None,
                         white: int | None = None, bg_dist: float | None = None, gamma: float | None = None,
                         size_table: str | None = None, credit: str | None = None, force: bool = False) -> Path:
    """Fetch -> cut out -> classify -> pair -> write -> verify. Returns the written outfit folder.

    `cfg` is config["garments"] completed with fitcast.garments.DEFAULTS (prepare_garments.load_config)."""
    ucfg = url_cfg(cfg)
    gdir = Path(garments_dir or cfg["dir"])
    gdir = gdir if gdir.is_absolute() else ROOT / gdir
    source = resolve_image(src, ucfg)
    rgba = cut_garment(source.image, cfg, ucfg, white, bg_dist, gamma)
    text = " ".join(filter(None, [source.title, urllib.parse.urlparse(source.page_url or "").path,
                                  urllib.parse.urlparse(source.image_url).path, name or ""]))
    if part == "auto":
        part = classify_part(text, rgba, ucfg)
    if sleeve is None and part == "top":
        sleeve = sleeve_from_text(text)
        if sleeve:
            log.info("sleeve: %s (keyword)", sleeve)
    if size_table is None and re.search(r"(?<![a-z])wom[ae]n", _words(text)):
        size_table = "womens"
    max_chars = int(ucfg["name_max_chars"])
    piece = (truncate(name, max_chars) if name else
             clean_name(source.title, max_chars) or name_from_url(source.image_url, max_chars)
             or f"Fetched {part.title()}")
    folder = Path(out) if out else default_out(gdir, piece, ucfg)
    folder = folder if folder.is_absolute() else ROOT / folder
    partner = find_partner(gdir, "bottom" if part == "top" else "top", ucfg, pair_with, exclude=folder)
    if folder.resolve() == partner.resolve():
        raise ValueError(f"--out {folder} is the partner folder itself")
    _check_out(folder, force)
    log.info("%s %r -> %s (partner %s)", part, piece, folder, partner.name)
    print(write_from_url_outfit(folder, rgba, part, piece, sleeve, partner, cfg, size_table))
    (folder / f"source{_ext(source.data)}").write_bytes(source.data)
    shown = src if "://" in src else str(Path(src).expanduser().resolve())
    try:
        rel = folder.resolve().relative_to(gdir.resolve())
    except ValueError:
        rel = folder
    entry = (f"- {_dt.date.today().isoformat()}: `{rel}` {part} \"{piece}\" from <{shown}>"
             + (f" (image <{source.image_url}>)" if source.page_url and source.image_url != shown else "")
             + f"; paired with `{partner.name}`" + (f"; {credit}" if credit else ""))
    append_source(Path(sources_md) if sources_md else gdir / "SOURCES.md", entry, key=f"`{rel}`")
    print(describe(folder, cfg))
    return folder


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (2 = the site blocked the download)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="product page URL, direct image URL, file:// URL or local image path")
    ap.add_argument("--part", choices=("top", "bottom", "auto"), default="auto")
    ap.add_argument("--name", help="piece name (default: page title / file name, <= name_max_chars)")
    ap.add_argument("--sleeve", choices=("short", "long", "none"), help="tops only (default: keywords, else shape)")
    ap.add_argument("--pair-with", help="outfit folder (name or path) supplying the other part")
    ap.add_argument("--out", type=Path, help="output outfit folder (default: <garments dir>/<n >= 90>_<slug>)")
    ap.add_argument("--white", type=int, help="near-white background threshold (default garments.white_bg_threshold; "
                                              "~250 for a white garment on white)")
    ap.add_argument("--bg-dist", type=float, help=f"Lab distance to the border colour counted as background "
                                                  f"(default {BG_DIST}; 0 = threshold only)")
    ap.add_argument("--gamma", type=float, help="tone curve for the fetched piece (default: auto-lift dark garments)")
    ap.add_argument("--size-table", choices=("mens", "womens"))
    ap.add_argument("--credit", help="license / author text added to the SOURCES.md line")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--garments-dir", type=Path, help="default: config garments.dir")
    ap.add_argument("--force", action="store_true", help="replace an existing --out folder not made by this script")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    try:
        make_outfit_from_url(args.src, cfg, part=args.part, name=args.name, sleeve=args.sleeve,
                             pair_with=args.pair_with, out=args.out, garments_dir=args.garments_dir,
                             white=args.white, bg_dist=args.bg_dist, gamma=args.gamma,
                             size_table=args.size_table, credit=args.credit, force=args.force)
    except BlockedError as e:
        log.error("%s", e)
        return 2
    except ValueError as e:
        log.error("%s. Product shots on a plain light background cut out best; try --white 250 for a white "
                  "garment, --bg-dist 0, or another image of the item", e)
        return 1
    except (RuntimeError, OSError, cv2.error) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
