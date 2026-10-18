"""Cart: link lookup and the entries written for the current outfit."""
import json

import numpy as np

from fitcast.cart import LinkBook, add_to_cart, load_cart
from fitcast.types import Outfit


def test_cart_adds_top_and_bottom_with_links(tmp_path):
    links = LinkBook("assets/shop_links.yaml", "assets/garments")
    o = Outfit(id="x", name="Heather Hoodie + Jeans", sleeve="long", top_front=np.zeros((4, 4, 4), np.uint8), top_back=None,
               sleeve_tex=None, bottom_front=np.zeros((4, 4, 4), np.uint8), bottom_back=None, thumb=np.zeros((8, 8, 3), np.uint8))
    path = tmp_path / "cart.json"
    new = add_to_cart(o, links, path)
    assert [e["part"] for e in new] == ["top", "bottom"]
    assert all(e["url"].startswith("http") for e in new)
    assert "champion" in new[0]["url"] or "amazon" in new[0]["url"]     # curated link for the hoodie
    assert load_cart(path) == json.loads(path.read_text()) and len(load_cart(path)) == 2
    assert links.url_for("Something Nobody Sells").startswith("https://www.google.com/search")
