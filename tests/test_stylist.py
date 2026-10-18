"""Stylist: catalogue ids, reply parsing and the wall message panel (no network)."""
import time

import numpy as np
import yaml

from fitcast.compositor import Compositor
from fitcast.stylist import GeminiStylist
from fitcast.types import GalleryState, Outfit, TrackResult


def _outfit(i, name, bottom=True):
    rgba = np.zeros((64, 48, 4), np.uint8)
    rgba[8:56, 8:40] = (200, 50, 50, 255)
    return Outfit(id=f"{i:02d}", name=name, sleeve="short", top_front=rgba, top_back=rgba, sleeve_tex=None,
                  bottom_front=rgba if bottom else None, bottom_back=rgba if bottom else None,
                  thumb=np.zeros((160, 160, 3), np.uint8))


def test_catalog_and_parse():
    outs = [_outfit(0, "Red Tee + Jeans"), _outfit(1, "Blue Shirt + Jeans"), _outfit(2, "Hoodie + Chinos")]
    st = GeminiStylist({"models": ["x"], "links_file": "assets/shop_links.yaml"}, outs, "key")
    cat = st.catalog()
    assert "T0: Red Tee" in cat and "T2: Hoodie" in cat
    assert "B0: Jeans" in cat and "B1: Chinos" in cat and "B2" not in cat          # distinct bottoms only
    r = st.parse('```json\n{"heard": "interview", "top_id": "T1", "bottom_id": "B1", "say": "ok", '
                 '"shopping": [{"item": "Oxford shirt", "store": "Uniqlo", "price_usd": 29.9, "url": "u"}]}\n```', "q")
    assert (r.top_index, r.bottom_index) == (1, 2) and r.question == "interview"
    assert r.shopping[0]["url"].startswith("http") and any(x["store"] == "Uniqlo" for x in r.shopping)   # curated first, model extra kept
    r2 = st.parse('{"top_id": "T9", "bottom_id": null, "say": "", "shopping": []}', "q")
    assert r2.top_index is None and r2.bottom_index is None


def test_message_panel_draws_and_expires():
    cfg = yaml.safe_load(open("config.yaml"))["compositor"]
    comp = Compositor(cfg, (320, 180))
    comp.set_gallery_rects([4, 20, 60, 160], [260, 20, 316, 160])
    comp.set_message_rect([70, 20, 250, 60])
    outs = [_outfit(0, "Red Tee + Jeans")]
    gal = GalleryState(outfits=outs)
    j = np.zeros((33, 3), np.float32); j[:, 0] = 160; j[:, 1] = 120
    tr = TrackResult(ok=True, joints=j, visibility=np.ones(33, np.float32), clothes_mask=np.zeros((180, 320), np.uint8),
                     head_box=(150, 100, 170, 115), facing="front", timestamp=1.0, shoulder_width=20.0,
                     ref_shoulder_width=20.0, person_mask=np.zeros((180, 320), np.uint8), facing_ratio=1.0)
    rgba = np.zeros((180, 320, 4), np.uint8)
    comp.set_message(["Interview look", "Blue Shirt + Chinos"], now=1.0, duration_s=5.0)
    comp.compose(rgba, tr, gal, now=1.5)
    layer = comp.gallery_layer if comp.gallery_layer is not None else comp.out
    assert layer[20:60, 70:250].max() > 0
    comp.compose(rgba, tr, gal, now=7.0)
    layer = comp.gallery_layer if comp.gallery_layer is not None else comp.out
    assert layer[20:60, 70:250].max() == 0
