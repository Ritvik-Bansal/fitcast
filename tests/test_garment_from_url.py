"""scripts/garment_from_url.py: page parsing, fetching (local http.server + file paths), classification and an
end-to-end outfit build into tmp_path (no network)."""
from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("garment_from_url", ROOT / "scripts" / "garment_from_url.py")
gfu = importlib.util.module_from_spec(_spec)
sys.modules["garment_from_url"] = gfu
_spec.loader.exec_module(gfu)
import prepare_garments as pg  # noqa: E402  (scripts/ is on sys.path after the import above)
from fitcast.garments import load_outfit  # noqa: E402

FAST = {"retries": 2, "retry_backoff_s": 0.0, "timeout_s": 5.0}


@pytest.fixture()
def cfg(config) -> dict:
    c = pg.garment_cfg(dict(config["garments"]))
    c["from_url"] = FAST
    return c


def tee_image(color=(40, 40, 200)) -> np.ndarray:
    """Short-sleeve T shape on a white background (BGR)."""
    img = np.full((560, 480, 3), 255, np.uint8)
    cv2.rectangle(img, (150, 110), (330, 500), color, -1)           # torso
    cv2.fillConvexPoly(img, np.array([[150, 110], [60, 200], [100, 250], [150, 200]]), color)
    cv2.fillConvexPoly(img, np.array([[330, 110], [420, 200], [380, 250], [330, 200]]), color)
    cv2.ellipse(img, (240, 110), (40, 22), 0, 0, 180, (255, 255, 255), -1)   # neckline
    return img


def pants_image(color=(150, 90, 40)) -> np.ndarray:
    """Two legs joined at the waist, gap between them (BGR, tall)."""
    img = np.full((640, 360, 3), 255, np.uint8)
    cv2.rectangle(img, (90, 40), (270, 200), color, -1)
    cv2.fillConvexPoly(img, np.array([[90, 200], [178, 200], [170, 600], [80, 600]]), color)
    cv2.fillConvexPoly(img, np.array([[182, 200], [270, 200], [280, 600], [190, 600]]), color)
    return img


@pytest.fixture()
def site(tmp_path):
    """Local http.server over tmp_path/site on a random port; yields (base_url, site_dir)."""
    d = tmp_path / "site"
    d.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/blocked"):
                self.send_error(403, "Forbidden")
                return
            super().do_GET()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(d)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", d
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def garments(tmp_path, cfg) -> Path:
    """tmp garments dir holding one complete partner outfit (14_partner) made with write_outfit."""
    g = tmp_path / "garments"
    g.mkdir()
    top = cv2.cvtColor(tee_image((60, 160, 60)), cv2.COLOR_BGR2BGRA)
    top[..., 3] = np.where(top[..., :3].min(axis=2) < 250, 255, 0)
    bot = cv2.cvtColor(pants_image(), cv2.COLOR_BGR2BGRA)
    bot[..., 3] = np.where(bot[..., :3].min(axis=2) < 250, 255, 0)
    meta = dict(name="Green Tee + Brown Pants", category="outfit", sleeve="short", size_table="mens",
                colors=["#00FF00", "#663300"])
    pg.write_outfit(g, "14_partner", meta, cfg, top=top, bottom=bot)
    (g / "SOURCES.md").write_text("# Garment sources\n\n## Bottoms (other)\n\n| a | b |\n")
    return g


# ----------------------------------------------------------------------------- parsing

def test_parse_page_prefers_og_and_resolves_relative():
    html = """<html><head><title>Red Cotton Tee | Shop</title>
      <meta property="og:image" content="/img/tee.png"><meta name="twitter:image" content="tw.jpg"></head>
      <body><img src="small.jpg" width="10" height="10"></body></html>"""
    p = gfu.parse_page(html, "http://shop.test/p/red-tee")
    assert p.title == "Red Cotton Tee | Shop"
    assert p.candidates[0] == "http://shop.test/img/tee.png"
    assert p.candidates[1] == "http://shop.test/p/tw.jpg"


def test_parse_page_jsonld_then_largest_img():
    ld = {"@context": "https://schema.org", "@graph": [
        {"@type": "WebPage", "image": "https://cdn.test/banner.jpg"},
        {"@type": "Product", "name": "Jeans", "image": [{"@type": "ImageObject", "contentUrl": "/jeans.webp"}]}]}
    html = (f"<title>Jeans</title><script type='application/ld+json'>{json.dumps(ld)}</script>"
            "<script type='application/ld+json'>{broken</script>"
            "<img src='a.jpg' width='100' height='100'><img src='b.jpg' width='800' height='1000'>"
            "<img src='c.jpg' srcset='c1.jpg 300w, c2.jpg 1200w'>"
            "<img id='landingImage' src='x.jpg' data-a-dynamic-image='{\"https://m.test/big.jpg\": [1500, 1500]}'>")
    p = gfu.parse_page(html, "https://shop.test/x/")
    assert p.candidates[:2] == ["https://shop.test/jeans.webp", "https://cdn.test/banner.jpg"]
    rest = p.candidates[2:]
    assert rest.index("https://m.test/big.jpg") < rest.index("https://shop.test/x/c2.jpg") \
        < rest.index("https://shop.test/x/b.jpg") < rest.index("https://shop.test/x/a.jpg")


def test_classify_and_sleeve_keywords():
    u = {"bottom_aspect": 1.3}
    assert gfu.classify_part("Men's Slim Fit Jeans", None, u) == "bottom"
    assert gfu.classify_part("/products/heavyweight-t-shirt-black", None, u) == "top"
    assert gfu.classify_part("Short Sleeve Polo", None, u) == "top"            # "short sleeve" is not "shorts"
    assert gfu.classify_part("Cargo Shorts", None, u) == "bottom"
    tall, wide = np.zeros((300, 100, 4), np.uint8), np.zeros((100, 120, 4), np.uint8)
    assert gfu.classify_part("item 123", tall, u) == "bottom"
    assert gfu.classify_part("item 123", wide, u) == "top"
    assert gfu.sleeve_from_text("Basketball Jersey") == "none"
    assert gfu.sleeve_from_text("Tank Top") == "none"
    assert gfu.sleeve_from_text("Zip Hoodie") == "long"
    assert gfu.sleeve_from_text("Long Sleeve Jersey Tee") == "long"
    assert gfu.sleeve_from_text("Crew Tee") is None


def test_clean_name():
    assert gfu.clean_name("Amazon.com: Hanes Men's Beefy-T Short Sleeve T-Shirt: Clothing", 28) == \
        "Hanes Beefy-T Short Sleeve"
    assert gfu.clean_name("Men's Crew Neck T-Shirt | UNIQLO US", 28) == "Crew Neck T-Shirt"
    assert len(gfu.clean_name("A very long product title that goes on and on forever", 28)) <= 28
    assert gfu.name_from_url("https://upload.test/thumb/a/ab/Red_T-shirt.jpg/800px-Red_T-shirt.jpg", 28) == "Red T shirt"


# ----------------------------------------------------------------------------- fetch

def test_resolve_from_html_over_http(site, cfg):
    base, d = site
    cv2.imwrite(str(d / "tee.png"), tee_image())
    cv2.imwrite(str(d / "icon.png"), np.zeros((32, 32, 3), np.uint8))
    (d / "p.html").write_text("<html><head><title>Red Cotton Tee | Shop</title>"
                              "<meta property='og:image' content='icon.png'></head>"
                              "<body><img src='tee.png' width='480' height='560'></body></html>")
    s = gfu.resolve_image(f"{base}/p.html", gfu.url_cfg(cfg))
    assert s.image_url == f"{base}/tee.png"            # icon skipped (too small)
    assert s.title == "Red Cotton Tee | Shop" and s.image.shape[:2] == (560, 480)


def test_resolve_direct_file_and_file_url(tmp_path, cfg):
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), tee_image())
    for src in (str(p), p.as_uri()):
        s = gfu.resolve_image(src, gfu.url_cfg(cfg))
        assert s.page_url is None and s.image.shape[:2] == (560, 480)


def test_blocked_site_message(site, cfg):
    base, d = site
    with pytest.raises(gfu.BlockedError, match="save the product image"):
        gfu.fetch_bytes(f"{base}/blocked/item", gfu.url_cfg(cfg))
    (d / "captcha.html").write_text("<html><head><title>Amazon.com Robot Check</title></head>"
                                    "<body><img src='captcha.jpg'></body></html>")
    with pytest.raises(gfu.BlockedError, match="bot-check"):
        gfu.resolve_image(f"{base}/captcha.html", gfu.url_cfg(cfg))


# ----------------------------------------------------------------------------- end to end

def test_top_from_page_builds_loadable_outfit(site, garments, cfg, capsys):
    base, d = site
    cv2.imwrite(str(d / "tee.png"), tee_image())
    (d / "red-cotton-tee.html").write_text("<html><head><title>Red Cotton Tee | Shop</title>"
                                           "<meta property='og:image' content='tee.png'></head></html>")
    url = f"{base}/red-cotton-tee.html"
    folder = gfu.make_outfit_from_url(url, cfg, garments_dir=garments)
    assert folder == garments / "90_red_cotton_tee"
    o = load_outfit(folder, cfg)
    assert o.name == "Red Cotton Tee + Brown Pants"
    assert o.sleeve == "short"
    tw, th = cfg["top_size"]
    assert o.top_front.shape == (th, tw, 4) and o.top_back is not None and o.sleeve_tex is not None
    assert o.bottom_front is not None and o.bottom_back is not None
    assert np.array_equal(o.bottom_front, cv2.imread(str(garments / "14_partner/bottom_front.png"), -1))
    x0, y0, x1, y1 = o.top_body_region
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and (x1 - x0) < 0.8     # sleeves excluded from the torso
    red = o.top_front[o.top_front[..., 3] == 255][:, :3].mean(axis=0)
    assert red[2] > 150 and red[0] < 100                                       # BGR: the red tee survived
    meta = yaml.safe_load((folder / "meta.yaml").read_text())
    assert meta["parts"] == ["top_front", "top_back", "sleeve", "bottom_front", "bottom_back"]
    assert (folder / "source.png").is_file()
    src_md = (garments / "SOURCES.md").read_text()
    assert gfu.SOURCES_HEADING in src_md and url in src_md and "90_red_cotton_tee" in src_md
    out = capsys.readouterr().out
    assert "loaded 90_red_cotton_tee" in out and "top_body_region" in out
    # second run: same slug reuses the folder; SOURCES.md keeps one heading and one line per folder
    gfu.make_outfit_from_url(url, cfg, garments_dir=garments)
    gfu.make_outfit_from_url(str(d / "tee.png"), cfg, garments_dir=garments, name="Other Tee")
    text = (garments / "SOURCES.md").read_text()
    assert text.count(gfu.SOURCES_HEADING) == 1 and text.count(url) == 1 and "91_other_tee" in text
    assert text.index("## Bottoms (other)") < text.index(gfu.SOURCES_HEADING)
    assert sorted(p.name for p in garments.iterdir() if p.is_dir()) == ["14_partner", "90_red_cotton_tee",
                                                                         "91_other_tee"]


def test_bottom_from_file_pairs_existing_top(tmp_path, garments, cfg):
    p = tmp_path / "item_8812.jpg"
    cv2.imwrite(str(p), pants_image((60, 60, 180)))
    out = tmp_path / "outfits" / "my_pants"
    folder = gfu.make_outfit_from_url(str(p), cfg, name="Red Wide Pants", out=out, pair_with="14_partner",
                                      garments_dir=garments, sources_md=tmp_path / "S.md")
    assert folder == out
    o = load_outfit(folder, cfg)
    assert o.name == "Green Tee + Red Wide Pants"        # part from the aspect ratio (no keyword in the file name)
    bw, bh = cfg["bottom_size"]
    assert o.bottom_front.shape == (bh, bw, 4)
    assert np.array_equal(o.top_front, cv2.imread(str(garments / "14_partner/top_front.png"), -1))
    assert o.sleeve == "short"
    assert "my_pants" in (tmp_path / "S.md").read_text()


def test_refuses_to_replace_foreign_folder(tmp_path, garments, cfg):
    p = tmp_path / "tee.png"
    cv2.imwrite(str(p), tee_image())
    with pytest.raises(FileExistsError):
        gfu.make_outfit_from_url(str(p), cfg, out=_foreign(garments), garments_dir=garments,
                                 sources_md=tmp_path / "S.md")
    assert (garments / "20_someone_elses" / "meta.yaml").read_text() == "name: x\n"


def _foreign(garments: Path) -> Path:
    d = garments / "20_someone_elses"
    d.mkdir()
    (d / "meta.yaml").write_text("name: x\n")
    return d


def test_cli_exit_code_on_block(site, cfg, tmp_path):
    base, _ = site
    cfg_path = tmp_path / "config.yaml"
    full = yaml.safe_load((ROOT / "config.yaml").read_text())
    full["garments"]["from_url"] = FAST
    cfg_path.write_text(yaml.safe_dump(full))
    assert gfu.main([f"{base}/blocked/x", "--config", str(cfg_path), "--garments-dir", str(tmp_path)]) == 2


# ----------------------------------------------------------------------------- on-model / cropped photo gate

def test_page_skips_rejected_candidates(site, cfg):
    base, d = site
    cv2.imwrite(str(d / "model.png"), np.full((600, 500, 3), 128, np.uint8))
    cv2.imwrite(str(d / "tee.png"), tee_image())
    (d / "p.html").write_text("<meta property='og:image' content='model.png'><img src='tee.png'>")
    check = lambda im: "on-model photo (9% skin)" if im.shape[:2] == (600, 500) else None  # noqa: E731
    s = gfu.resolve_image(f"{base}/p.html", gfu.url_cfg(cfg), check)
    assert s.image_url.endswith("/tee.png")
    (d / "q.html").write_text("<meta property='og:image' content='model.png'>")
    with pytest.raises(gfu.UnusableImageError, match="product-only shot"):
        gfu.resolve_image(f"{base}/q.html", gfu.url_cfg(cfg), check)
    with pytest.raises(gfu.UnusableImageError):
        gfu.resolve_image(str(d / "model.png"), gfu.url_cfg(cfg), check)


def test_edge_backstop_rejects_garment_cut_by_frame(cfg):
    u = gfu.url_cfg(cfg)
    cropped = np.full((500, 400, 3), 255, np.uint8)
    cv2.rectangle(cropped, (60, 80), (340, 499), (40, 40, 200), -1)          # runs off the bottom edge
    assert "runs off" in gfu.reject_reason(cropped, cfg, u, None)
    assert gfu.reject_reason(tee_image(), cfg, u, None) is None
    assert gfu.reject_reason(pants_image(), cfg, u, None) is None
    barred = tee_image()
    barred[-12:] = 150                                                         # museum credit bar, separate blob
    assert gfu.reject_reason(barred, cfg, u, None) is None


def test_skin_detector_on_flat_garment(config, cfg):
    pytest.importorskip("mediapipe")
    model = ROOT / config["tracker"]["mediapipe"]["seg_model"]
    if not model.is_file():
        pytest.skip(f"{model} missing")
    det = gfu.ModelShotDetector(model, gfu.url_cfg(cfg))
    try:
        assert det.skin_frac(tee_image()) < gfu.url_cfg(cfg)["model_skin_frac"]
    finally:
        det.close()
    assert gfu.ModelShotDetector(ROOT / "nope.tflite", gfu.url_cfg(cfg)).skin_frac(tee_image()) is None


def test_cli_exit_code_on_model_only(tmp_path, cfg):
    p = tmp_path / "cropped.png"
    img = np.full((500, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 80), (340, 499), (40, 40, 200), -1)
    cv2.imwrite(str(p), img)
    assert gfu.main([str(p), "--garments-dir", str(tmp_path)]) == 3
