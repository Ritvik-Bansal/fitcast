"""Shopping cart for the booth: the top and bottoms currently on the person, with a purchase link each.

Stored locally in `out/cart.json` (a list of entries); the web page in `scripts/webapp.py` lists it.
Links: curated `assets/shop_links.yaml` by garment name, then the product page recorded by the fetch
scripts (`assets/garments/_raw*/sources.json`), then a shopping search for the name.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from pathlib import Path

from .types import Outfit

log = logging.getLogger("fitcast.cart")

CART_FILE = Path("out/cart.json")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class LinkBook:
    """Garment name -> purchase URL, built once from the curated links and every sources.json."""

    def __init__(self, links_file: str | Path = "assets/shop_links.yaml", garments_dir: str | Path = "assets/garments") -> None:
        self.curated: dict[str, list[dict]] = {}
        try:
            from .stylist import load_shop_links
            self.curated = load_shop_links(links_file)
        except Exception as e:  # noqa: BLE001
            log.warning("shop links unavailable: %s", e)
        self.sources: list[tuple[str, str]] = []      # (searchable text, page url)
        for f in Path(garments_dir).glob("_raw*/sources.json"):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            items = d.items() if isinstance(d, dict) else enumerate(d)
            for key, v in items:
                if not isinstance(v, dict):
                    continue
                url = v.get("page_url") or v.get("url") or ""
                text = " ".join(str(v.get(k, "")) for k in ("title", "desc", "brand", "name")) + " " + str(key)
                if url:
                    self.sources.append((_slug(text), url))

    def url_for(self, name: str) -> str:
        """Best link for a garment name (as shown in the gallery)."""
        for e in self.curated.get(name, []):
            if e.get("url"):
                return e["url"]
        words = [w for w in _slug(name).split() if len(w) > 2]
        best, best_n = "", 0
        for text, url in self.sources:
            n = sum(1 for w in words if w in text)
            if n > best_n:
                best, best_n = url, n
        if best and best_n >= max(1, len(words) - 1):
            return best
        return "https://www.google.com/search?tbm=shop&q=" + urllib.parse.quote_plus(name)


def load_cart(path: Path = CART_FILE) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except ValueError:
        return []


def add_to_cart(outfit: Outfit, links: LinkBook, path: Path = CART_FILE) -> list[dict]:
    """Append the current top and bottoms (one entry each) and return the new entries."""
    cart = load_cart(path)
    top, bottom = outfit.name.split(" + ")[0], outfit.name.split(" + ")[-1]
    stamp = time.strftime("%H:%M:%S")
    new = []
    for part, name in (("top", top), ("bottom", bottom)):
        if name:
            new.append({"time": stamp, "part": part, "name": name, "url": links.url_for(name)})
    cart.extend(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cart, indent=1))
    log.info("cart: added %s", " | ".join(f"{e['name']} -> {e['url']}" for e in new))
    return new
