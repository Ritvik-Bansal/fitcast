import numpy as np

from fitcast.renderer import Renderer
from fitcast.types import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, Outfit
from tests.conftest import frontal_joints, make_track


def _outfit(with_bottom=True) -> Outfit:
    top = np.zeros((640, 512, 4), np.uint8)
    top[:, :, :] = (0, 0, 255, 255)        # opaque red (BGRA)
    back = np.zeros_like(top)
    back[:, :, :] = (255, 0, 0, 255)       # opaque blue
    sleeve = np.full((256, 256, 4), (0, 255, 0, 255), np.uint8)
    bottom = np.full((768, 512, 4), (255, 255, 0, 255), np.uint8) if with_bottom else None
    thumb = np.zeros((160, 160, 3), np.uint8)
    return Outfit(id="t", name="t", sleeve="long", top_front=top, top_back=back, sleeve_tex=sleeve,
                  bottom_front=bottom, bottom_back=bottom, thumb=thumb, top_body_region=(0.2, 0.0, 0.8, 1.0))


def test_torso_warp_inside_expected_bbox(config):
    r = Renderer(config["renderer"], (1280, 720))
    tr = make_track()
    rgba = r.draw(tr, _outfit(with_bottom=False))
    assert rgba.shape == (720, 1280, 4) and rgba.dtype == np.uint8
    a = rgba[:, :, 3]
    assert a.max() > 0, "garment should be drawn"
    j = tr.joints
    ys, xs = np.nonzero(a)
    sw = tr.shoulder_width
    # torso + sleeves live between the wrists horizontally and from above the shoulders to below the hips
    assert xs.min() >= j[R_SHOULDER, 0] - 0.6 * sw and xs.max() <= j[L_SHOULDER, 0] + 0.6 * sw
    assert ys.min() >= j[L_SHOULDER, 1] - 0.5 * sw and ys.max() <= j[L_HIP, 1] + 0.6 * sw
    # torso centre is red
    cx, cy = int((j[L_SHOULDER, 0] + j[R_HIP, 0]) / 2), int((j[L_SHOULDER, 1] + j[L_HIP, 1]) / 2)
    assert tuple(rgba[cy, cx, :3]) == (0, 0, 255)
    assert r.parts_drawn == 5  # torso + 2 upper arms + 2 forearms


def test_back_uses_back_texture(config):
    r = Renderer(config["renderer"], (1280, 720))
    j = frontal_joints()
    j[[L_SHOULDER, R_SHOULDER], 0] = j[[R_SHOULDER, L_SHOULDER], 0]
    j[[L_HIP, R_HIP], 0] = j[[R_HIP, L_HIP], 0]
    tr = make_track(j, facing="back")
    rgba = r.draw(tr, _outfit(with_bottom=False))
    cy = int((j[L_SHOULDER, 1] + j[L_HIP, 1]) / 2)
    assert tuple(rgba[cy, 640, :3]) == (255, 0, 0)


def test_legs_skipped_when_not_visible(config):
    r = Renderer(config["renderer"], (1280, 720))
    tr = make_track()
    tr.visibility[25:] = 0.0            # knees + ankles unseen (clips cropped above the knee)
    rgba = r.draw(tr, _outfit())
    assert r.parts_drawn == 6           # waist band + torso + 4 sleeves, no leg quads
    assert rgba[:, :, 3][int(tr.joints[L_HIP, 1]) + 200:, :].max() == 0


def test_full_body_draws_legs(config):
    r = Renderer(config["renderer"], (1280, 720))
    tr = make_track()
    r.draw(tr, _outfit())
    assert r.parts_drawn == 10          # band + 4 leg quads + torso + 4 sleeves


def test_no_person_is_blank(config):
    r = Renderer(config["renderer"], (1280, 720))
    tr = make_track(ok=False)
    assert r.draw(tr, _outfit()).max() == 0
