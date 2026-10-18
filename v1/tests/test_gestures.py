import unittest

import numpy as np

import synth
from tryon.config import load_config
from tryon.gestures import GestureDetector
from tryon.orientation import OrientationEstimator

DT = 1 / 30
HIP_Y = 400.0
# geometry of synth person at hip_y=400: nose ~185, shoulders 255, hips 400
NOSE_Y = HIP_Y - 0.74 * synth.SCALE
SH_Y = HIP_Y - 0.50 * synth.SCALE


def smooth(a, b, u):
    u = min(1.0, max(0.0, u))
    return a + (b - a) * u


class Sim:
    """Feed scripted frames; collect (t, event). script(t) -> dict(cx, wrists=(lx, rx) points)."""

    def __init__(self, cfg=None, orientation=None, checkout_ready=True, **cfgov):
        self.cfg = cfg or load_config(overrides=cfgov)
        self.det = GestureDetector(self.cfg)
        self.ori = orientation or synth_fake()
        self.ready = checkout_ready
        self.events = []
        self.t = 0.0
        self.i = 0
        self.progress = []

    def frame(self, cx=640.0, wrists=None, vis_overrides=None, present=True):
        res = synth.make_pose(self.t, 0.0, cx=cx, hip_y=HIP_Y, wrists=wrists, frame_idx=self.i,
                              present=present, vis_overrides=vis_overrides)
        ev = self.det.update(res, self.ori, checkout_ready=self.ready)
        for e in ev:
            self.events.append((self.t, e))
        self.progress.append(self.det.palm_progress)
        self.t += DT
        self.i += 1
        return ev

    def run(self, dur, fn):
        """fn(local_t) -> kwargs for frame()"""
        t0 = self.t
        n = int(round(dur / DT))
        for k in range(n):
            self.frame(**fn(k * DT))

    def names(self):
        return [e for _, e in self.events]


def synth_fake(bucket="front"):
    return synth.FakeOrientation(bucket)


REST = {"left_wrist": (640 + 0.25 * synth.SCALE, HIP_Y + 6), "right_wrist": (640 - 0.25 * synth.SCALE, HIP_Y + 6)}


def rest_fn(_t):
    return dict(wrists=dict(REST))


def hand_at(x, y, cx=640.0):
    w = dict(REST)
    w["left_wrist"] = (x, y)
    return dict(cx=cx, wrists=w)


def swipe_fn(x0, x1, y, t_start, dur):
    def fn(t):
        return hand_at(smooth(x0, x1, (t - t_start) / dur), y)
    return fn


class SwipeTests(unittest.TestCase):
    def test_swipe_right_in_image_is_next(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
        self.assertEqual(s.names(), ["next"])

    def test_swipe_left_in_image_is_prev(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(760, 560, 330, 0.0, 0.15))
        self.assertEqual(s.names(), ["prev"])

    def test_swipe_direction_flag_inverts(self):
        s = Sim(swipe_left_is_next=False)
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
        self.assertEqual(s.names(), ["prev"])

    def test_right_hand_swipe_works_too(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            w = dict(REST)
            w["right_wrist"] = (smooth(720, 520, t / 0.15), 330)
            return dict(wrists=w)
        s.run(1.5, fn)
        self.assertEqual(s.names(), ["prev"])

    def test_slow_movement_nothing(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(3.0, swipe_fn(500, 800, 330, 0.0, 1.2))   # 250 px/s
        self.assertEqual(s.names(), [])

    def test_medium_speed_below_threshold_nothing(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(2.0, swipe_fn(500, 800, 330, 0.0, 0.40))  # 750 px/s
        self.assertEqual(s.names(), [])

    def test_swipe_below_hip_line_nothing(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, HIP_Y + 20, 0.0, 0.15))
        self.assertEqual(s.names(), [])

    def test_short_fast_flick_below_min_distance(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(600, 680, 330, 0.0, 0.07))  # 80px, fast
        self.assertEqual(s.names(), [])

    def test_mostly_vertical_fast_motion_nothing(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            u = t / 0.15
            return hand_at(smooth(600, 740, u), smooth(250, 395, u))
        s.run(1.5, fn)
        self.assertEqual(s.names(), [])

    def test_two_swipes_within_debounce_one_event(self):
        s = Sim()
        s.run(0.5, rest_fn)
        # swipe, pause 100 ms, second swipe in the same direction (jump back off-screen is hidden by
        # keeping the hand out of view via low visibility)
        s.run(0.15 + 0.01, swipe_fn(540, 740, 330, 0.0, 0.15))

        def fn2(t):
            return hand_at(smooth(540, 740, (t - 0.25) / 0.15), 330) if t >= 0.25 else hand_at(540, 330)
        s.run(1.5, fn2)
        self.assertEqual(len(s.names()), 1, s.events)

    def test_two_swipes_apart_two_events(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.0, swipe_fn(540, 740, 330, 0.0, 0.15))
        # hand drops (below hip -> not a swipe), then raise & swipe again later
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(540, 740, 330, 0.0, 0.15))
        self.assertEqual(s.names(), ["next", "next"], s.events)

    def test_return_stroke_after_swipe_no_opposite_event(self):
        for gap in (0.0, 0.3, 0.5):
            with self.subTest(gap=gap):
                s = Sim()
                s.run(0.5, rest_fn)
                s.run(0.3, swipe_fn(540, 740, 330, 0.0, 0.15))
                s.run(gap, lambda t: hand_at(740, 330)) if gap else None
                s.run(1.5, swipe_fn(740, 540, 330, 0.0, 0.15))
                self.assertEqual(s.names(), ["next"], s.events)

    def test_whole_body_translation_no_swipe(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            cx = smooth(500, 800, t / 0.25)   # 1200 px/s sideways
            w = {k: (v[0] - 640 + cx, 330) for k, v in REST.items()}
            return dict(cx=cx, wrists=w)
        s.run(1.0, fn)
        self.assertEqual(s.names(), [])

    def test_body_translation_hands_up_no_swipe_even_with_tracking_lag(self):
        """Torso moves, wrists follow (same displacement)."""
        s = Sim()
        s.run(0.5, lambda t: dict(cx=400, wrists={k: (v[0] - 640 + 400, 330) for k, v in REST.items()}))

        def fn(t):
            cx = smooth(400, 900, t / 0.4)
            return dict(cx=cx, wrists={k: (v[0] - 640 + cx, 330) for k, v in REST.items()})
        s.run(1.0, fn)
        self.assertEqual(s.names(), [])

    def test_swipe_blocked_when_not_front(self):
        for b in ("left", "back", "right"):
            s = Sim(orientation=synth_fake(b))
            s.run(0.5, rest_fn)
            s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
            self.assertEqual(s.names(), [], b)
            self.assertEqual(s.det.blocked, f"facing {b}")

    def test_swipe_blocked_when_turning(self):
        s = Sim(orientation=synth.FakeOrientation("front", turning=True))
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
        self.assertEqual(s.names(), [])

    def test_swipe_during_crossfade_blocked(self):
        s = Sim(orientation=synth.FakeOrientation("front", transition=("left", "front", 0.0)))
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
        self.assertEqual(s.names(), [])

    def test_gestures_front_only_false_allows_other_buckets(self):
        s = Sim(orientation=synth_fake("back"), gestures_front_only=False)
        s.run(0.5, rest_fn)
        s.run(1.5, swipe_fn(560, 760, 330, 0.0, 0.15))
        self.assertEqual(s.names(), ["next"])

    def test_swipe_at_30fps_jittery_timestamps(self):
        """Real cameras jitter; swipe at ~2x threshold must still be detected."""
        s = Sim()
        s.run(0.5, rest_fn)
        rng = np.random.default_rng(3)
        t0 = s.t
        n = 0
        while s.t - t0 < 1.5:
            local = s.t - t0
            s.frame(**hand_at(smooth(540, 760, local / 0.15), 330))
            s.t += rng.uniform(-0.006, 0.006)
            n += 1
        self.assertEqual(s.names(), ["next"])

    def test_minimum_detectable_swipe_speed_probe(self):
        """Tuning probe (720p, ~2.5 m): a brisk natural hand wave of ~250 px in 250 ms (1000 px/s)
        should arguably register.  Reports whether it does."""
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.0, swipe_fn(520, 770, 330, 0.0, 0.25))
        print(f"[tuning] 250px/250ms swipe events: {s.names()}")


class RaiseTests(unittest.TestCase):
    RAISED_Y = NOSE_Y - 60

    def up(self):
        return lambda t: hand_at(640 + 90, self.RAISED_Y)

    def test_raise_600ms_one_toggle(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(0.6 + 0.1, self.up())
        s.run(0.5, rest_fn)
        self.assertEqual(s.names(), ["toggle_mode"])

    def test_raise_short_no_toggle(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(0.3, self.up())
        s.run(0.5, rest_fn)
        self.assertEqual(s.names(), [])

    def test_raise_held_3s_still_one(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(3.0, self.up())
        self.assertEqual(s.names(), ["toggle_mode"])

    def test_raise_progress_ramps(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(0.3, self.up())
        self.assertTrue(0.3 < s.det.raise_progress < 1.0, s.det.raise_progress)

    def test_lower_and_raise_again_after_debounce(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(0.7, self.up())
        s.run(1.2, rest_fn)                       # lowered; total since last toggle > raise_debounce (1500)
        s.run(0.4, rest_fn)
        s.run(0.7, self.up())
        s.run(0.3, rest_fn)
        self.assertEqual(s.names(), ["toggle_mode", "toggle_mode"], s.events)

    def test_raise_again_inside_debounce_ignored(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(0.6, self.up())
        s.run(0.2, rest_fn)
        s.run(0.6, self.up())     # 0.8 s after first toggle: < 1500 ms debounce
        self.assertEqual(s.names(), ["toggle_mode"], s.events)

    def test_raise_blocked_when_back(self):
        s = Sim(orientation=synth_fake("back"))
        s.run(0.5, rest_fn)
        s.run(1.0, self.up())
        self.assertEqual(s.names(), [])

    def test_low_visibility_wrist_does_not_toggle(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            d = hand_at(640 + 90, self.RAISED_Y)
            d["vis_overrides"] = {"left_wrist": 0.2}
            return d
        s.run(1.0, fn)
        self.assertEqual(s.names(), [])

    def test_raised_hand_is_not_a_swipe(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            return hand_at(640 + 90 + 40 * np.sin(2 * np.pi * 1.0 * t), self.RAISED_Y)
        s.run(0.8, fn)
        self.assertEqual([e for e in s.names() if e != "toggle_mode"], [])


class PalmTests(unittest.TestCase):
    PALM_Y = (NOSE_Y + SH_Y) / 2 + 10

    def palm(self):
        return lambda t: hand_at(640 + 100, self.PALM_Y)

    def test_hold_1p6s_one_checkout_progress_ramps(self):
        s = Sim()
        s.run(0.5, rest_fn)
        n0 = len(s.progress)
        s.run(1.6, self.palm())
        s.run(0.3, self.palm())
        self.assertEqual(s.names(), ["checkout"], s.events)
        prog = s.progress[n0:]
        self.assertLess(prog[2], 0.3)
        self.assertGreater(max(prog), 0.9)   # progress is zeroed on the frame the event fires
        # monotone non-decreasing until the event
        k = prog.index(max(prog))
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(prog[:k], prog[1:k + 1])), prog[:k + 1])

    def test_no_second_checkout_while_still_held(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(5.0, self.palm())
        self.assertEqual(s.names(), ["checkout"])

    def test_checkout_rearms_after_hand_leaves_zone(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(2.0, self.palm())
        s.run(1.0, rest_fn)
        s.run(2.2, self.palm())
        self.assertEqual(s.names(), ["checkout", "checkout"], s.events)

    def test_moving_wrist_resets(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.0, self.palm())
        self.assertGreater(s.det.palm_progress, 0.4)
        # one full wave period inside the zone (peak ~1500 px/s would be a swipe, so keep amplitude small)
        s.run(0.6, lambda t: hand_at(640 + 100 + 60 * np.sin(2 * np.pi * t / 0.6), self.PALM_Y))
        self.assertLess(s.det.palm_progress, 0.4)
        s.run(0.5, self.palm())
        self.assertEqual(s.names(), [])

    def test_not_ready_no_event(self):
        s = Sim(checkout_ready=False)
        s.run(0.5, rest_fn)
        s.run(3.0, self.palm())
        self.assertEqual(s.names(), [])
        self.assertEqual(s.det.palm_progress, 0.0)

    def test_hand_below_shoulder_no_checkout(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(3.0, lambda t: hand_at(740, SH_Y + 40))
        self.assertEqual(s.names(), [])

    def test_palm_blocked_when_not_front(self):
        s = Sim(orientation=synth_fake("right"))
        s.run(0.5, rest_fn)
        s.run(3.0, self.palm())
        self.assertEqual(s.names(), [])

    def test_palm_with_noise_still_counts_as_still(self):
        """~2 px keypoint jitter (raw, unsmoothed) must stay under palm_still_speed_px_s."""
        s = Sim()
        rng = np.random.default_rng(5)
        s.run(0.5, rest_fn)
        s.run(2.5, lambda t: hand_at(740 + rng.normal(0, 1.5), self.PALM_Y + rng.normal(0, 1.5)))
        self.assertEqual(s.names(), ["checkout"], "palm hold fails with 1.5px jitter (speed over 0.2 s window)")

    def test_palm_with_tracking_dropout(self):
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            d = hand_at(740, self.PALM_Y)
            if 0.8 < t < 0.9:
                d["vis_overrides"] = {"left_wrist": 0.1}
            return d
        s.run(2.2, fn)
        self.assertEqual(s.names(), ["checkout"], s.events)

    def test_person_lost_resets(self):
        s = Sim()
        s.run(0.5, rest_fn)
        s.run(1.0, self.palm())
        s.run(0.2, lambda t: dict(present=False))
        self.assertEqual(s.det.palm_progress, 0.0)
        self.assertEqual(s.det.blocked, "no person")

    def test_raise_then_palm_interplay(self):
        """Hand rising through the palm zone to above the nose must not check out."""
        s = Sim()
        s.run(0.5, rest_fn)

        def fn(t):
            return hand_at(740, smooth(HIP_Y - 20, NOSE_Y - 60, t / 1.0))
        s.run(1.0, fn)
        self.assertNotIn("checkout", s.names())


class RealOrientationIntegration(unittest.TestCase):
    def test_swipe_with_real_estimator_front(self):
        cfg = load_config()
        ori = OrientationEstimator(cfg)
        det = GestureDetector(cfg)
        ev = []
        for i in range(90):
            t = i / 30
            x = smooth(560, 760, (t - 1.5) / 0.15)
            res = synth.make_pose(t, 0.0, wrists={"left_wrist": (x, 330), "right_wrist": REST["right_wrist"]}, frame_idx=i)
            ori.update(res)
            ev += det.update(res, ori, True)
        self.assertEqual(ev, ["next"])


if __name__ == "__main__":
    unittest.main()
