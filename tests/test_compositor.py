import numpy as np

from fitcast.compositor import Compositor
from fitcast.types import GalleryState, Outfit
from tests.conftest import make_track


def _gallery(n=3) -> GalleryState:
    outs = [Outfit(id=f"o{i}", name=f"Outfit {i}", sleeve="short", top_front=None, top_back=None, sleeve_tex=None,
                   bottom_front=None, bottom_back=None, thumb=np.full((160, 160, 3), 200, np.uint8)) for i in range(n)]
    return GalleryState(outfits=outs, index=0, prev_index=0, changed_at=0.0, visible=True,
                        size_text="Size M (estimate)", measure_text="Shoulder 46 cm")


def test_head_box_always_black_even_with_full_light(config):
    comp = Compositor(config["compositor"], (1280, 720))
    tr = make_track(mask_full=False)
    tr.clothes_mask[300:600, 500:780] = 255               # torso lit
    rgba = np.full((720, 1280, 4), 255, np.uint8)
    x0, y0, x1, y1 = tr.head_box
    # gallery panel deliberately covering the head box to prove the blackout is applied last
    comp.set_gallery_rect((x0 - 50, 0, x1 + 50, y1 + 20))
    out = comp.compose(rgba, tr, _gallery(), now=10.0)
    assert out.shape == (720, 1280, 3)
    assert out[y0:y1, x0:x1].max() == 0
    assert out[300:600, 500:780].min() > 0                # garment lit
    assert out[0:y1, x0 - 50:x0].max() > 0                # panel drawn beside the head box


def test_no_light_outside_mask(config):
    comp = Compositor(config["compositor"], (1280, 720))
    comp.gallery_enabled = False
    tr = make_track(mask_full=False)
    tr.clothes_mask[300:400, 500:700] = 255
    rgba = np.full((720, 1280, 4), 255, np.uint8)
    out = comp.compose(rgba, tr, _gallery(), now=10.0)
    assert out[300:400, 500:700].min() > 0
    outside = out.copy()
    outside[300:400, 500:700] = 0
    assert outside.max() == 0


def test_fade_on_tracking_loss(config):
    cfg = dict(config["compositor"], fade_ms=1000)
    comp = Compositor(cfg, (1280, 720))
    comp.gallery_enabled = False
    rgba = np.full((720, 1280, 4), 255, np.uint8)
    tr = make_track()
    comp.compose(rgba, tr, _gallery(), now=0.0)
    full = comp.compose(rgba, tr, _gallery(), now=2.0).copy()
    assert full[400, 640].max() == 255
    lost = make_track(ok=False, mask_full=False)
    mid = comp.compose(rgba, lost, _gallery(), now=2.5)   # half way through the fade-out
    assert 0 < mid[400, 640].max() < 255
    x0, y0, x1, y1 = tr.head_box
    assert mid[y0:y1, x0:x1].max() == 0                   # the held frame keeps its head blackout
    dark = comp.compose(rgba, lost, _gallery(), now=3.5)
    assert dark.max() == 0
    back = comp.compose(rgba, tr, _gallery(), now=3.6)
    assert 0 < back[400, 640].max() < 255, "fade back in should be partial right after recovery"


def test_gallery_panel_never_overlaps_person(config):
    comp = Compositor(config["compositor"], (1280, 720))
    comp.set_gallery_rect((20, 40, 300, 700))
    tr = make_track(mask_full=False)
    tr.clothes_mask[100:600, 100:250] = 255              # person standing inside the panel rect
    rgba = np.zeros((720, 1280, 4), np.uint8)
    out = comp.compose(rgba, tr, _gallery(), now=10.0)
    assert out[100:600, 100:250].max() == 0
    assert out[40:700, 20:300].max() > 0                  # but the panel is drawn elsewhere
