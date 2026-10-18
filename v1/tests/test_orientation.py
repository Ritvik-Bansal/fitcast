import unittest

import numpy as np

import synth
from tryon.config import load_config
from tryon.orientation import OrientationEstimator


def run_yaws(est, seq, **kw):
    """Feed (t, yaw) pairs; return list of (t, bucket) change points + per-frame buckets."""
    changes, buckets = [], []
    prev = est.bucket
    for i, (t, yaw) in enumerate(seq):
        est.update(synth.make_pose(t, yaw, frame_idx=i, **kw))
        buckets.append(est.bucket)
        if est.bucket != prev:
            changes.append((t, est.bucket))
            prev = est.bucket
    return changes, buckets


def _with_tail(seq, n=30):
    t_end, y_end = seq[-1]
    return seq + [(t_end + (i + 1) / 30, y_end) for i in range(n)]


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(overrides={})

    def _check_turn(self, direction, swap, rev_s=4.0):
        est = OrientationEstimator(self.cfg)
        start = 0.0
        seq = list(synth.yaw_sweep(rev_s, direction, start=start))
        t_end = seq[-1][0]
        seq += [(t_end + (i + 1) / 30, start) for i in range(30)]  # smoothing lags the true yaw ~10 deg
        # settle at the start yaw first for 0.5 s
        pre = [(-0.5 + i / 30, start) for i in range(15)]
        changes, _ = run_yaws(est, pre + seq, mp_swap_when_back=swap)
        names = [b for _, b in changes]
        if direction > 0:
            expect = ["left", "back", "right", "front"]
        else:
            expect = ["right", "back", "left", "front"]
        self.assertEqual(names, expect, f"dir={direction} swap={swap} changes={changes}")

    def test_cw_with_mp_swap(self):
        self._check_turn(+1, True)

    def test_cw_no_swap(self):
        self._check_turn(+1, False)

    def test_ccw_with_mp_swap(self):
        self._check_turn(-1, True)

    def test_ccw_no_swap(self):
        self._check_turn(-1, False)

    def test_fast_turn_1s(self):
        est = OrientationEstimator(self.cfg)
        pre = [(-0.5 + i / 30, 0.0) for i in range(15)]
        changes, _ = run_yaws(est, pre + _with_tail(list(synth.yaw_sweep(1.5, +1))), mp_swap_when_back=True)
        names = [b for _, b in changes]
        self.assertEqual(names, ["left", "back", "right", "front"], changes)

    def test_two_revolutions_no_drift(self):
        est = OrientationEstimator(self.cfg)
        pre = [(-0.5 + i / 30, 0.0) for i in range(15)]
        seq = _with_tail(list(synth.yaw_sweep(4.0, +1, total_deg=720)))
        changes, _ = run_yaws(est, pre + seq, mp_swap_when_back=True)
        self.assertEqual([b for _, b in changes], ["left", "back", "right", "front"] * 2, changes)

    def test_never_mirrored_back_view(self):
        """Holding yaw 110 -> must be 'left' (wearer's left side to camera), 250 -> 'right'."""
        for swap in (False, True):
            for yaw, exp in ((90, "left"), (270, "right"), (180, "back")):
                est = OrientationEstimator(self.cfg)
                # approach continuously from front
                path = np.linspace(0, yaw if yaw <= 180 else yaw - 360, 60) % 360
                seq = [(i / 30, y) for i, y in enumerate(path)]
                seq += [(2.0 + i / 30, yaw) for i in range(30)]
                run_yaws(est, seq, mp_swap_when_back=swap)
                self.assertEqual(est.bucket, exp, f"yaw={yaw} swap={swap}")


class StillTests(unittest.TestCase):
    def test_standing_front_with_noise(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        rng = np.random.default_rng(1)
        changes = []
        for i in range(300):
            est.update(synth.make_pose(i / 30, 0.0, noise_px=2.0, rng=rng, frame_idx=i))
            if est.bucket != "front" or est.transition is not None:
                changes.append(i)
        self.assertEqual(changes, [])

    def test_standing_front_noise_world_z(self):
        # shoulder-ratio noise bigger (4 px) and slight body yaw jitter
        cfg = load_config()
        est = OrientationEstimator(cfg)
        rng = np.random.default_rng(2)
        for i in range(300):
            est.update(synth.make_pose(i / 30, rng.normal(0, 8), noise_px=4.0, rng=rng, frame_idx=i))
            self.assertEqual(est.bucket, "front", f"frame {i}")


class HysteresisTests(unittest.TestCase):
    def test_oscillation_around_boundary_no_flicker(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        # settle at 0
        seq = [(i / 30, 0.0) for i in range(15)]
        # oscillate 45 +/- 5 (inside 45+15 hysteresis) at 1.5 Hz
        t0 = 0.5
        for i in range(180):
            seq.append((t0 + i / 30, 45 + 5 * np.sin(2 * np.pi * 1.5 * i / 30)))
        changes, _ = run_yaws(est, seq)
        self.assertEqual(changes, [], changes)

    def test_oscillation_plus_minus_50_around_45_never_flickers(self):
        """+/-50 deg of swing around 45 deg (i.e. -5..95)."""
        cfg = load_config()
        est = OrientationEstimator(cfg)
        seq = [(i / 30, 0.0) for i in range(15)]
        t0 = 0.5
        for i in range(240):
            seq.append((t0 + i / 30, 45 + 50 * np.sin(2 * np.pi * 1.0 * i / 30)))
        changes, buckets = run_yaws(est, seq)
        # each excursion past 60+ deg lasts long enough to switch once; make sure no
        # rapid back-and-forth (<250 ms) switching happens.
        for (t1, _), (t2, _) in zip(changes, changes[1:]):
            self.assertGreaterEqual(t2 - t1, 0.25, changes)

    def test_short_excursion_does_not_switch(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        seq = [(i / 30, 0.0) for i in range(20)]
        t = 20 / 30
        # 150 ms jump to 90 (5 frames = 133 ms at 30 fps), then back
        for i in range(5):
            seq.append((t + i / 30, 90.0))
        t += 5 / 30
        for i in range(30):
            seq.append((t + i / 30, 0.0))
        changes, _ = run_yaws(est, seq)
        self.assertEqual(changes, [])

    def test_sustained_excursion_does_switch(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        seq = [(i / 30, 0.0) for i in range(20)]
        seq += [(20 / 30 + i / 30, 90.0) for i in range(30)]
        changes, _ = run_yaws(est, seq)
        self.assertEqual([b for _, b in changes], ["left"])


class CrossfadeTests(unittest.TestCase):
    def test_view_weights(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        seq = [(i / 30, 0.0) for i in range(20)] + [(20 / 30 + i / 30, 90.0) for i in range(40)]
        changes, _ = run_yaws(est, seq)
        self.assertEqual(len(changes), 1)
        t0 = changes[0][0]
        vw = est.view_weights(t0 + 0.05)
        self.assertEqual(len(vw), 2)
        self.assertAlmostEqual(sum(w for _, w in vw), 1.0)
        self.assertEqual([n for n, _ in vw], ["front", "left"])
        self.assertAlmostEqual(vw[1][1], 0.05 * 1000 / cfg.view_crossfade_ms, places=3)
        vw = est.view_weights(t0 + cfg.view_crossfade_ms / 1000.0 + 0.01)
        self.assertEqual(vw, [("left", 1.0)])
        self.assertIsNone(est.transition)

    def test_no_transition_single_view(self):
        est = OrientationEstimator(load_config())
        est.update(synth.make_pose(0.0, 0.0))
        self.assertEqual(est.view_weights(0.0), [("front", 1.0)])


class LostReappearTests(unittest.TestCase):
    def _lost_then(self, yaw, swap, n=10, **kw):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        for i in range(30):
            est.update(synth.make_pose(i / 30, 0.0, frame_idx=i))
        self.assertEqual(est.bucket, "front")
        t = 2.5  # gap > 1 s
        first = None
        for i in range(n):
            est.update(synth.make_pose(t + i / 30, yaw, mp_swap_when_back=swap, frame_idx=100 + i, **kw))
            if first is None:
                first = est.bucket
        return est, first

    def test_reappears_facing_back_mp_swapped(self):
        est, first = self._lost_then(180.0, True)
        self.assertEqual(first, "back")
        self.assertEqual(est.bucket, "back")

    def test_reappears_facing_back_unswapped_settles(self):
        est, first = self._lost_then(180.0, False, n=60)
        self.assertEqual(est.bucket, "back")

    # Regression: the first frame after a reset must seed the smoothed shoulder ratio / dz
    # instead of blending with stale state (used to pick the wrong bucket on reappearance).
    def test_reappears_facing_back_unswapped_first_frame(self):
        est, first = self._lost_then(180.0, False, n=1)
        self.assertEqual(first, "back")

    def test_reappears_left_side_first_frame(self):
        est, first = self._lost_then(100.0, False, n=1)
        self.assertEqual(first, "left")

    # Regression: a wrong 180-degree branch chosen at init would be kept by the continuity rule
    # while the person stays side-on (no hemisphere re-anchor) -> mirrored view.
    def test_reappears_side_view_settles(self):
        # (not exactly 90/270: with no face cue the front/back pick at c==0 is a coin flip on cos(90deg)'s sign)
        est, first = self._lost_then(100.0, False, n=60)
        self.assertEqual(est.bucket, "left")
        est, first = self._lost_then(260.0, False, n=60)
        self.assertEqual(est.bucket, "right")

    def test_reappears_facing_back_then_stays(self):
        est, _ = self._lost_then(180.0, True)
        for i in range(60):
            est.update(synth.make_pose(3.0 + i / 30, 180.0, mp_swap_when_back=True, frame_idx=200 + i))
            self.assertEqual(est.bucket, "back")


class TurningTests(unittest.TestCase):
    def test_turning_true_during_turn_false_when_still(self):
        cfg = load_config()
        est = OrientationEstimator(cfg)
        for i in range(20):
            est.update(synth.make_pose(i / 30, 0.0, frame_idx=i))
        self.assertFalse(est.turning)
        saw = False
        for i, (t, yaw) in enumerate(synth.yaw_sweep(4.0, +1, total_deg=100)):
            est.update(synth.make_pose(1.0 + t, yaw, frame_idx=100 + i))
            saw |= est.turning
        self.assertTrue(saw)
        # stop at 90 for 2 s
        for i in range(60):
            est.update(synth.make_pose(4.0 + i / 30, 100.0, frame_idx=300 + i))
        self.assertFalse(est.turning)

    def test_turning_during_whole_slow_turn(self):
        """A 2 s/rev turn: turning should be True for most of the turn (gestures suppressed)."""
        cfg = load_config()
        est = OrientationEstimator(cfg)
        for i in range(20):
            est.update(synth.make_pose(i / 30, 0.0, frame_idx=i))
        flags = []
        for i, (t, yaw) in enumerate(synth.yaw_sweep(2.0, +1, total_deg=120)):
            est.update(synth.make_pose(1.0 + t, yaw, frame_idx=100 + i))
            flags.append(est.turning)
        self.assertGreater(sum(flags) / len(flags), 0.5)

    def test_peak_turn_rate_4s_per_rev_vs_threshold(self):
        """Tuning probe: a relaxed 4 s/rev turn never exceeds turn_rate_suppress (1.5)."""
        cfg = load_config()
        est = OrientationEstimator(cfg)
        peak = 0.0
        for i, (t, yaw) in enumerate(synth.yaw_sweep(4.0, +1)):
            est.update(synth.make_pose(t, yaw, frame_idx=i))
            peak = max(peak, est.turn_rate)
        print(f"[tuning] peak turn_rate at 4 s/rev = {peak:.2f} (threshold {cfg.turn_rate_suppress})")
        self.assertGreater(peak, 0.5)


if __name__ == "__main__":
    unittest.main()


class FaceDetectorOutOfRangeTests(unittest.TestCase):
    """A face detector that never fires (wearer too far away) must not flip a front-facing
    person to the back view: absence of a face only counts once a face has been seen."""

    def test_front_stays_front_when_detector_never_sees_a_face(self):
        est = OrientationEstimator(load_config())
        for i in range(90):
            est.update(synth.make_pose(i / 30, 0.0, face_score=0.0, frame_idx=i))
        self.assertEqual(est.bucket, "front")
        self.assertFalse(est.face_validated)

    def test_back_detected_once_detector_has_seen_the_face(self):
        est = OrientationEstimator(load_config())
        for i in range(30):
            est.update(synth.make_pose(i / 30, 0.0, face_score=0.9, frame_idx=i))
        self.assertTrue(est.face_validated)
        t = 2.5  # person leaves and comes back facing away (MediaPipe labels as if facing front)
        for i in range(30):
            est.update(synth.make_pose(t + i / 30, 180.0, mp_swap_when_back=True, face_score=0.0, frame_idx=100 + i))
        self.assertEqual(est.bucket, "back")
