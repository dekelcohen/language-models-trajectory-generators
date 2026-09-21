"""Unit tests for the tracking evaluation harness (``tests/tracking_eval.py``).

Two families:

* **Metric math** on synthetic frame samples with known error, known occlusion windows,
  known spikes and known lost frames - including the negative controls that matter: a
  tracker which reports ``lost`` every frame must score *badly*, not perfectly.
* **The seed-offset correction.** A perfect tracker - one whose prediction is exactly the
  surface point the ground-truth pose implies - must score ~0. Without that check the
  harness would quietly measure a constant surface-vs-origin bias and every provider
  comparison built on it would be meaningless.

Plus one end-to-end run of the whole harness on the simulator-free
:class:`tracking_eval.SyntheticSphereScene`, so the runner, the real
:class:`tracking.session.TrackingSession` and the reporting path are all exercised.

Run: python -m pytest tests\\test_tracking_eval.py -q
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tracking_eval as te  # noqa: E402
from sim_adapter import transforms  # noqa: E402


def make_samples(errors, lost=None, occluded=None, latencies=None, base=(0.0, 0.4, 0.2),
                 direction=(1.0, 0.0, 0.0)):
    """Frame samples whose measured error is exactly ``errors[i]`` metres.

    The expected point walks along a straight line so that inter-frame prediction jumps are
    a known quantity too.
    """
    base = np.asarray(base, dtype=float)
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    samples = []
    for i, err in enumerate(errors):
        expected = base + np.array([0.0, 0.0, 0.01]) * i
        is_lost = bool(lost[i]) if lost is not None else False
        sample = te.FrameSample(
            frame=i, obj="cube", expected=expected,
            predicted=None if is_lost else expected + unit * float(err),
            lost=is_lost,
            occluded=bool(occluded[i]) if occluded is not None else False,
            latency_ms=float(latencies[i]) if latencies is not None else 1.0,
        )
        samples.append(sample)
    return samples


class TestSeedOffsetCorrection(unittest.TestCase):
    """The harness must measure tracking error, not the constant seed bias."""

    SEED_POINTS = np.array([
        [0.02, 0.44, 0.21],
        [0.05, 0.42, 0.19],
        [0.01, 0.46, 0.23],
    ])

    def setUp(self):
        self.origin = np.array([0.0, 0.40, 0.17])
        self.quat = transforms.quat_from_euler([0.0, 0.0, 0.3])
        self.offset = te.capture_seed_offset(self.SEED_POINTS, self.origin, self.quat)

    def test_offset_is_the_surface_bias(self):
        self.assertTrue(self.offset.rotating)
        np.testing.assert_allclose(self.offset.expected(self.origin, self.quat),
                                   self.SEED_POINTS.mean(axis=0), atol=1e-9)
        self.assertGreater(float(np.linalg.norm(self.offset.local)), 0.02,
                           "the fixture must carry a real bias or the test proves nothing")

    def test_perfect_tracker_scores_zero(self):
        """Ground-truth-derived predictions must score ~0, including while rotating."""
        samples = []
        for i in range(20):
            position = self.origin + np.array([0.01, 0.0, 0.004]) * i
            quat = transforms.quat_from_euler([0.0, 0.0, 0.3 + 0.05 * i])
            expected = self.offset.expected(position, quat)
            samples.append(te.FrameSample(frame=i, obj="cube", expected=expected,
                                          predicted=expected.copy(), lost=False))
        metrics = te.compute_metrics(samples)
        self.assertLess(metrics["median_l2_m"], 1e-9)
        self.assertLess(metrics["max_l2_m"], 1e-9)
        self.assertEqual(metrics["pct_lost"], 0.0)
        self.assertEqual(metrics["n_valid_frames"], 20)

    def test_uncorrected_scoring_measures_the_bias_instead(self):
        """Negative control: scoring against the body origin reports the constant offset."""
        bias = float(np.linalg.norm(self.offset.local))
        samples = []
        for i in range(10):
            position = self.origin + np.array([0.01, 0.0, 0.0]) * i
            quat = transforms.quat_from_euler([0.0, 0.0, 0.3 + 0.05 * i])
            samples.append(te.FrameSample(frame=i, obj="cube", expected=position,
                                          predicted=self.offset.expected(position, quat),
                                          lost=False))
        self.assertAlmostEqual(te.compute_metrics(samples)["median_l2_m"], bias, places=9)

    def test_world_frame_offset_is_wrong_once_the_object_rotates(self):
        """The documented limitation of a no-orientation scene driver, pinned down."""
        flat = te.capture_seed_offset(self.SEED_POINTS, self.origin, None)
        self.assertFalse(flat.rotating)
        turned = transforms.quat_from_euler([0.0, 0.0, 1.2])
        np.testing.assert_allclose(flat.expected(self.origin, turned),
                                   self.SEED_POINTS.mean(axis=0), atol=1e-9)
        self.assertGreater(
            float(np.linalg.norm(flat.expected(self.origin, turned)
                                 - self.offset.expected(self.origin, turned))), 0.01)


class TestMetrics(unittest.TestCase):
    def test_known_constant_error(self):
        metrics = te.compute_metrics(make_samples([0.02] * 10))
        self.assertAlmostEqual(metrics["median_l2_m"], 0.02, places=9)
        self.assertAlmostEqual(metrics["p95_l2_m"], 0.02, places=9)
        self.assertAlmostEqual(metrics["max_l2_m"], 0.02, places=9)
        self.assertEqual(metrics["n_frames"], 10)
        self.assertEqual(metrics["n_valid_frames"], 10)
        self.assertEqual(metrics["pct_lost"], 0.0)

    def test_percentiles_follow_the_distribution(self):
        errors = [0.01] * 19 + [0.5]
        metrics = te.compute_metrics(make_samples(errors))
        self.assertAlmostEqual(metrics["median_l2_m"], 0.01, places=9)
        self.assertGreater(metrics["p95_l2_m"], 0.02, "the tail must lift p95 above the median")
        self.assertLess(metrics["p95_l2_m"], 0.5)
        self.assertAlmostEqual(metrics["max_l2_m"], 0.5, places=9)

    def test_a_tracker_that_always_gives_up_scores_badly(self):
        """The central anti-flattering guarantee of the harness."""
        metrics = te.compute_metrics(make_samples([0.0] * 12, lost=[True] * 12))
        self.assertEqual(metrics["pct_lost"], 100.0)
        self.assertEqual(metrics["n_valid_frames"], 0)
        self.assertAlmostEqual(metrics["median_l2_m"], te.LOST_PENALTY_M, places=9)
        self.assertIsNone(metrics["median_l2_valid_m"])

    def test_lost_frames_are_penalised_not_dropped(self):
        lost = [False] * 5 + [True] * 5
        metrics = te.compute_metrics(make_samples([0.01] * 10, lost=lost))
        self.assertEqual(metrics["pct_lost"], 50.0)
        self.assertEqual(metrics["n_valid_frames"], 5)
        self.assertAlmostEqual(metrics["median_l2_valid_m"], 0.01, places=9)
        self.assertAlmostEqual(metrics["median_l2_m"], (0.01 + te.LOST_PENALTY_M) / 2.0, places=9)
        self.assertGreater(metrics["median_l2_m"], metrics["median_l2_valid_m"],
                           "the penalty must drag the headline median up")
        self.assertAlmostEqual(metrics["max_l2_m"], te.LOST_PENALTY_M, places=9)

    def test_dropped_frames_count_as_lost(self):
        samples = make_samples([0.01] * 4)
        samples[2].dropped = True
        samples[2].lost = True
        samples[2].predicted = None
        metrics = te.compute_metrics(samples)
        self.assertEqual(metrics["n_dropped_frames"], 1)
        self.assertEqual(metrics["pct_lost"], 25.0)

    def test_occlusion_window_error(self):
        errors = [0.01] * 5 + [0.2] * 4 + [0.01] * 5
        occluded = [False] * 5 + [True] * 4 + [False] * 5
        metrics = te.compute_metrics(make_samples(errors, occluded=occluded))
        self.assertEqual(metrics["n_occluded_frames"], 4)
        self.assertAlmostEqual(metrics["err_during_occlusion_m"], 0.2, places=9)
        self.assertAlmostEqual(metrics["median_l2_m"], 0.01, places=9)

    def test_frames_to_recover(self):
        errors = [0.01] * 3 + [0.3] * 3 + [0.3, 0.2, 0.01, 0.01]
        occluded = [False] * 3 + [True] * 3 + [False] * 4
        metrics = te.compute_metrics(make_samples(errors, occluded=occluded))
        self.assertEqual(metrics["frames_to_recover"], 2)

    def test_recovery_is_none_without_occlusion_or_recovery(self):
        self.assertIsNone(te.compute_metrics(make_samples([0.01] * 5))["frames_to_recover"])
        errors = [0.01] * 2 + [0.4] * 2 + [0.4] * 3
        occluded = [False] * 2 + [True] * 2 + [False] * 3
        self.assertIsNone(
            te.compute_metrics(make_samples(errors, occluded=occluded))["frames_to_recover"])

    def test_max_jump_detects_a_spike(self):
        samples = make_samples([0.0] * 8)
        samples[4].predicted = np.asarray(samples[4].expected) + np.array([0.0, 0.0, 0.6])
        metrics = te.compute_metrics(samples)
        self.assertGreater(metrics["max_jump_m"], 0.55)

    def test_max_jump_ignores_gaps_around_lost_frames(self):
        """A lost frame breaks the chain; the resume must not be scored as a jump."""
        lost = [False, False, True, False, False]
        samples = make_samples([0.0] * 5, lost=lost)
        samples[3].predicted = np.asarray(samples[3].expected) + np.array([0.0, 0.0, 0.9])
        samples[4].predicted = np.asarray(samples[4].expected) + np.array([0.0, 0.0, 0.9])
        self.assertLess(te.compute_metrics(samples)["max_jump_m"], 0.05)

    def test_latency_metrics(self):
        latencies = [10.0] * 19 + [100.0]
        metrics = te.compute_metrics(make_samples([0.01] * 20, latencies=latencies))
        self.assertAlmostEqual(metrics["latency_ms_mean"], 14.5, places=3)
        self.assertGreater(metrics["latency_ms_p95"], 10.0, "the slow frame must show in p95")
        self.assertLessEqual(metrics["latency_ms_p95"], 100.0)

    def test_scoring_never_raises(self):
        self.assertEqual(te.compute_metrics([])["n_frames"], 0)
        broken = make_samples([0.01] * 3)
        broken[1].predicted = np.array([np.nan, np.nan, np.nan])
        metrics = te.compute_metrics(broken)
        self.assertNotIn("error", metrics)
        self.assertEqual(metrics["n_valid_frames"], 2)

    def test_peak_vram_is_optional(self):
        value = te.peak_vram_mb()
        self.assertTrue(value is None or isinstance(value, float))
        te.reset_vram_peak()


class TestAggregate(unittest.TestCase):
    def test_means_central_tendency_and_worst_tail(self):
        per_object = {
            "a": te.compute_metrics(make_samples([0.01] * 10)),
            "b": te.compute_metrics(make_samples([0.03] * 10)),
        }
        agg = te.aggregate_metrics(per_object)
        self.assertEqual(agg["n_objects"], 2)
        self.assertEqual(agg["n_frames"], 20)
        self.assertAlmostEqual(agg["median_l2_m"], 0.02, places=6)
        self.assertAlmostEqual(agg["max_l2_m"], 0.03, places=6)

    def test_empty_aggregate(self):
        self.assertEqual(te.aggregate_metrics({})["n_objects"], 0)


class TestOutputHelpers(unittest.TestCase):
    def _payload(self, label, median):
        metrics = te.compute_metrics(make_samples([median] * 6))
        return {"schema": te.SCHEMA, "config": {"label": label}, "peak_vram_mb": None,
                "objects": {"cube": metrics}, "aggregate": te.aggregate_metrics({"cube": metrics})}

    def test_write_report_round_trip(self):
        payload = self._payload("template+depth_fusion", 0.02)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__))) as tmp:
            path = te.write_report(payload, os.path.join(tmp, "nested", "run.json"))
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["config"]["label"], "template+depth_fusion")

    def test_comparison_table(self):
        table = te.render_comparison_table([self._payload("a+depth_fusion", 0.02),
                                            self._payload("b+triangulate", 0.05)])
        self.assertIn("a+depth_fusion", table)
        self.assertIn("b+triangulate", table)
        self.assertIn("median_m", table)
        self.assertEqual(len(set(len(line) for line in table.splitlines())), 1,
                         "table columns must line up")

    def test_comparison_table_with_no_results(self):
        self.assertEqual(te.render_comparison_table([]), "(no results)")


class TestHarnessEndToEnd(unittest.TestCase):
    """The runner + the real TrackingSession on the simulator-free synthetic scene."""

    @classmethod
    def setUpClass(cls):
        cls.scene = te.SyntheticSphereScene(n_frames=18, occlusion=(8, 13))
        cls.payload = te.run_eval(cls.scene, tracker_provider="template",
                                  tracker3d="depth_fusion", cameras=("head", "wrist"))

    def test_run_is_self_describing_and_serialisable(self):
        cfg = self.payload["config"]
        self.assertEqual(self.payload["schema"], te.SCHEMA)
        self.assertNotIn("error", self.payload)
        self.assertEqual(cfg["tracker_provider"], "template")
        self.assertEqual(cfg["tracker3d"], "depth_fusion")
        self.assertEqual(cfg["cameras"], ["head", "wrist"])
        self.assertEqual(cfg["scene"], "synthetic_sphere")
        self.assertIn("image_size", cfg)
        json.dumps(self.payload)      # must not raise: numpy has to be out of the payload

    def test_metrics_are_plausible(self):
        metrics = self.payload["objects"]["ball"]
        self.assertEqual(metrics["n_frames"], 18)
        self.assertEqual(self.payload["session"]["errors"], 0)
        self.assertLess(metrics["pct_lost"], 20.0)
        self.assertLess(metrics["median_l2_m"], 0.05,
                        f"synthetic scene should track tightly, got {metrics['median_l2_m']}")
        self.assertGreater(metrics["n_occluded_frames"], 0,
                           "the scene must actually exercise an occlusion window")
        self.assertLess(metrics["max_jump_m"], 0.2)

    def test_seed_offset_matches_the_sphere_radius(self):
        offset = self.payload["seed_offsets"]["ball"]["local"]
        self.assertAlmostEqual(float(np.linalg.norm(offset)), self.scene.radius, places=2)


if __name__ == "__main__":
    unittest.main()
