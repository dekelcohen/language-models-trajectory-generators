"""Unit tests for the CoTracker3 point-tracker provider.

The logic tests drive :class:`providers.trackers.cotracker_tracker.CoTrackerTracker` with a
*fake* predictor injected through the ``model`` kwarg, so they exercise the real buffering,
staleness, visibility and re-seed code with no download and no GPU. Only the one test that
touches the real checkpoint is skipped when the model is not cached.

Run:  python -m pytest tests\\test_cotracker_tracker.py -q
Real model (downloads once into TORCH_HOME, ~97 MB):
      set COTRACKER_REAL_TEST=1 && python -m pytest tests\\test_cotracker_tracker.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from providers.trackers.base import PointTracker, TrackResult  # noqa: E402
from providers.trackers.factory import SUPPORTED, get_tracker  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_TORCH = False

if HAVE_TORCH:
    from providers.trackers import cotracker_tracker as ct

WIDTH, HEIGHT = 64, 48
STEP = 4  # small window so the tests stay fast; the real model uses step=8


def frame(value=0):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :, 0] = value
    return img


class FakeCoTracker:
    """Stands in for ``CoTrackerOnlinePredictor``.

    Mirrors the real contract: ``is_first_step=True`` stores the queries and returns
    ``(None, None)``; every later call returns ``(tracks, visibility)`` shaped
    ``(B, T, N, 2)`` / ``(B, T, N)`` for the chunk it was given.
    """

    def __init__(self, step=STEP, dx=1.0, visibility=None, dtype=None):
        self.step = step
        self.dx = dx
        self.visibility = visibility
        self.dtype = dtype or torch.bool
        self.queries = None
        self.calls = []
        self.first_step_calls = 0

    def __call__(self, video_chunk=None, is_first_step=False, queries=None, **kwargs):
        self.calls.append({"is_first_step": is_first_step, "T": int(video_chunk.shape[1])})
        if is_first_step:
            self.queries = queries.clone()
            self.first_step_calls += 1
            return (None, None)
        seed = self.queries[:, :, 1:]                      # (B, N, 2)
        n = seed.shape[1]
        t = int(video_chunk.shape[1])
        offsets = torch.arange(t, dtype=torch.float32).reshape(1, t, 1, 1) * self.dx
        tracks = seed[:, None] + torch.cat([offsets, torch.zeros_like(offsets)], dim=-1)
        if self.visibility is None:
            vis = torch.ones((1, t, n), dtype=self.dtype)
        else:
            vis = torch.as_tensor(self.visibility, dtype=self.dtype).reshape(1, 1, n).repeat(1, t, 1)
        return (tracks, vis)


def make_tracker(**kwargs):
    model = kwargs.pop("model", None) or FakeCoTracker(**{k: kwargs.pop(k) for k in
                                                          ("dx", "visibility", "dtype")
                                                          if k in kwargs})
    tracker = ct.CoTrackerTracker(model=model, step=STEP, **kwargs)
    return tracker, model


@unittest.skipUnless(HAVE_TORCH, "torch is not installed in this environment")
class TestCoTrackerConformance(unittest.TestCase):
    def test_is_a_point_tracker(self):
        tracker, _ = make_tracker()
        self.assertIsInstance(tracker, PointTracker)
        self.assertEqual(tracker.name, "cotracker")
        self.assertFalse(tracker.initialised)
        self.assertEqual(tracker.window, 2 * STEP)

    def test_update_before_init_is_empty_not_a_crash(self):
        tracker, _ = make_tracker()
        result = tracker.update(frame())
        self.assertIsInstance(result, TrackResult)
        self.assertEqual(result.n_visible, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_init_rejects_points_outside_the_image(self):
        tracker, _ = make_tracker()
        with self.assertRaises(ValueError):
            tracker.init(frame(), [[-5.0, -5.0], [WIDTH + 10.0, 10.0]])

    def test_off_image_seed_keeps_its_slot_and_is_never_visible(self):
        # Row j must stay seed point j: the 3D lift maps rows to template ids by position.
        tracker, _ = make_tracker()
        tracker.init(frame(), [[10.0, 10.0], [WIDTH + 10.0, 10.0], [20.0, 20.0]])
        self.assertEqual(tracker.n_points, 3)
        results = [tracker.update(frame(i)) for i in range(1, 2 * STEP + 2)]
        for result in results:
            self.assertEqual(len(result.points), 3)
            self.assertFalse(result.visible[1])
            self.assertEqual(result.scores[1], 0.0)
        self.assertTrue(results[-1].visible[0] and results[-1].visible[2])

    def test_reset_clears_state(self):
        tracker, _ = make_tracker()
        tracker.init(frame(), [[10.0, 10.0]])
        tracker.update(frame(1))
        tracker.reset()
        self.assertFalse(tracker.initialised)
        self.assertEqual(tracker.n_points, 0)
        self.assertEqual(tracker.update(frame(2)).n_visible, 0)

    def test_factory_registration(self):
        self.assertIn("cotracker", SUPPORTED)
        tracker = get_tracker("cotracker", model=FakeCoTracker(), step=STEP)
        self.assertIsInstance(tracker, ct.CoTrackerTracker)

    def test_grayscale_and_float_frames_are_accepted(self):
        tracker, _ = make_tracker()
        gray = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        tracker.init(gray, [[8.0, 8.0]])
        result = tracker.update(gray + 5.0)
        self.assertEqual(len(result.points), 1)

    def test_frame_size_change_is_a_clear_error(self):
        tracker, _ = make_tracker()
        tracker.init(frame(), [[10.0, 10.0]])
        with self.assertRaises(ValueError) as ctx:
            tracker.update(np.zeros((HEIGHT * 2, WIDTH, 3), dtype=np.uint8))
        self.assertIn("re-seed", str(ctx.exception))


@unittest.skipUnless(HAVE_TORCH, "torch is not installed in this environment")
class TestWindowStaleness(unittest.TestCase):
    """The online model only emits every ``step`` frames; we return the newest result."""

    def test_warm_start_gives_a_fresh_prediction_on_the_first_update(self):
        tracker, model = make_tracker(warm_start=True)
        tracker.init(frame(), [[10.0, 20.0]])
        result = tracker.update(frame(1))
        self.assertEqual(result.meta["stale_frames"], 0)
        self.assertTrue(result.meta["fresh"])
        self.assertEqual(model.first_step_calls, 1)
        # Every model call must see a full window, as the online model requires.
        self.assertTrue(all(c["T"] == 2 * STEP for c in model.calls))

    def test_staleness_rises_between_flushes_and_resets_on_one(self):
        tracker, model = make_tracker(warm_start=True)
        tracker.init(frame(), [[10.0, 20.0]])
        staleness = [tracker.update(frame(i)).meta["stale_frames"] for i in range(1, 2 * STEP + 1)]
        # frame 1 flushes, then step-1 stale frames, then a flush again, ...
        self.assertEqual(staleness[:STEP], [0] + list(range(1, STEP)))
        self.assertEqual(staleness[STEP], 0)
        self.assertEqual(model.first_step_calls, 1)
        self.assertEqual(tracker._flushes, len([s for s in staleness if s == 0]))

    def test_cold_start_waits_a_full_window_then_flushes(self):
        tracker, model = make_tracker(warm_start=False)
        tracker.init(frame(), [[10.0, 20.0]])
        staleness = [tracker.update(frame(i)).meta["stale_frames"] for i in range(1, 2 * STEP + 1)]
        # The window holds the seed frame plus 2*step-1 updates before it is full.
        self.assertEqual(staleness[:2 * STEP - 2], list(range(1, 2 * STEP - 1)))
        self.assertEqual(staleness[2 * STEP - 2], 0)
        self.assertEqual(staleness[-1], 1)
        self.assertEqual(model.first_step_calls, 1)

    def test_stale_results_are_repeated_not_extrapolated(self):
        tracker, _ = make_tracker(warm_start=True, dx=2.0)
        tracker.init(frame(), [[10.0, 20.0]])
        fresh = tracker.update(frame(1))
        stale = [tracker.update(frame(i)) for i in range(2, STEP + 1)]
        for result in stale:
            np.testing.assert_allclose(result.points, fresh.points)

    def test_scores_decay_with_staleness(self):
        tracker, _ = make_tracker(warm_start=True, stale_decay=0.1)
        tracker.init(frame(), [[10.0, 20.0]])
        confidences = [tracker.update(frame(i)).confidence for i in range(1, STEP + 1)]
        self.assertAlmostEqual(confidences[0], 1.0)
        for older, newer in zip(confidences, confidences[1:]):
            self.assertLess(newer, older)
        self.assertAlmostEqual(confidences[-1], 1.0 - 0.1 * (STEP - 1), places=6)

    def test_meta_carries_the_window_contract(self):
        tracker, _ = make_tracker()
        tracker.init(frame(), [[10.0, 20.0]])
        meta = tracker.update(frame(1)).meta
        self.assertEqual(meta["provider"], "cotracker")
        self.assertEqual((meta["window"], meta["step"]), (2 * STEP, STEP))
        self.assertEqual(meta["flushes"], 1)
        self.assertIn("device", meta)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed in this environment")
class TestVisibility(unittest.TestCase):
    def test_boolean_visibility_maps_to_lost(self):
        tracker, _ = make_tracker(visibility=[True, False], dtype=torch.bool)
        tracker.init(frame(), [[10.0, 20.0], [30.0, 20.0]])
        result = tracker.update(frame(1))
        np.testing.assert_array_equal(result.visible, [True, False])
        self.assertEqual(result.scores[1], 0.0)
        self.assertEqual(result.n_visible, 1)

    def test_soft_visibility_is_thresholded_into_scores(self):
        tracker, _ = make_tracker(visibility=[0.9, 0.2], dtype=torch.float32,
                                  vis_threshold=0.5)
        tracker.init(frame(), [[10.0, 20.0], [30.0, 20.0]])
        result = tracker.update(frame(1))
        np.testing.assert_array_equal(result.visible, [True, False])
        self.assertAlmostEqual(result.scores[0], 0.9)
        self.assertAlmostEqual(result.scores[1], 0.0)
        self.assertAlmostEqual(result.confidence, 0.9)

    def test_all_points_lost_gives_zero_confidence(self):
        tracker, _ = make_tracker(visibility=[False, False], dtype=torch.bool)
        tracker.init(frame(), [[10.0, 20.0], [30.0, 20.0]])
        result = tracker.update(frame(1))
        self.assertEqual(result.n_visible, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_point_leaving_the_frame_is_lost_even_when_model_says_visible(self):
        # dx pushes the tracked point off the right edge within one window.
        tracker, _ = make_tracker(dx=float(WIDTH))
        tracker.init(frame(), [[10.0, 20.0]])
        result = tracker.update(frame(1))
        self.assertFalse(bool(result.visible[0]))
        self.assertEqual(result.scores[0], 0.0)
        self.assertEqual(result.confidence, 0.0)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed in this environment")
class TestReseed(unittest.TestCase):
    def test_reseed_restarts_the_online_state_and_keeps_the_model(self):
        tracker, model = make_tracker(warm_start=True)
        tracker.init(frame(), [[10.0, 20.0]])
        tracker.update(frame(1))
        tracker.update(frame(2))
        self.assertEqual(model.first_step_calls, 1)

        tracker.init(frame(3), [[40.0, 30.0], [41.0, 31.0]])
        self.assertEqual(tracker.n_points, 2)
        self.assertEqual(tracker._flushes, 0)
        self.assertEqual(tracker._stale, 0)
        result = tracker.update(frame(4))
        self.assertEqual(model.first_step_calls, 2)       # online state re-initialised
        self.assertIs(tracker._model, model)              # weights were NOT reloaded
        self.assertEqual(result.meta["stale_frames"], 0)  # warm start: fresh immediately
        np.testing.assert_allclose(result.points[:, 1], [30.0, 31.0])

    def test_reseed_queries_follow_the_new_points(self):
        tracker, model = make_tracker(warm_start=True, dx=0.0)
        tracker.init(frame(), [[10.0, 20.0]])
        tracker.update(frame(1))
        tracker.init(frame(2), [[25.0, 15.0]])
        result = tracker.update(frame(3))
        np.testing.assert_allclose(result.points[0], [25.0, 15.0])
        np.testing.assert_allclose(model.queries[0, 0].numpy(), [0.0, 25.0, 15.0])

    def test_reseed_before_any_flush_is_harmless(self):
        tracker, model = make_tracker(warm_start=False)
        tracker.init(frame(), [[10.0, 20.0]])
        tracker.update(frame(1))
        tracker.init(frame(2), [[12.0, 22.0]])
        result = tracker.update(frame(3))
        self.assertEqual(model.first_step_calls, 0)       # window not full yet
        self.assertEqual(result.meta["stale_frames"], 1)
        np.testing.assert_allclose(result.points[0], [12.0, 22.0])


@unittest.skipUnless(HAVE_TORCH, "torch is not installed in this environment")
class TestModelLoading(unittest.TestCase):
    def test_download_failure_names_torch_home(self):
        original = ct._hub_load
        ct._hub_load = lambda variant: (_ for _ in ()).throw(OSError("no route to host"))
        try:
            with self.assertRaises(RuntimeError) as err:
                ct.load_cotracker_model(device="cpu")
        finally:
            ct._hub_load = original
        message = str(err.exception)
        self.assertIn("TORCH_HOME", message)
        self.assertIn("--tracker-provider template", message)
        self.assertIn("no route to host", message)

    def test_offline_variant_is_refused(self):
        with self.assertRaises(ValueError):
            ct.load_cotracker_model(variant="cotracker3_offline", device="cpu")

    def test_device_selection_never_picks_a_second_gpu(self):
        self.assertEqual(ct.resolve_device("cuda:1"), "cuda:0")
        self.assertEqual(ct.resolve_device("CUDA"), "cuda:0")
        self.assertEqual(ct.resolve_device("cpu"), "cpu")
        self.assertIn(ct.resolve_device(None), ("cpu", "cuda:0"))

    def test_ensure_torch_home_sets_the_env(self):
        previous = os.environ.get("TORCH_HOME")
        try:
            resolved = ct.ensure_torch_home(ct.DEFAULT_TORCH_HOME)
            self.assertEqual(os.environ["TORCH_HOME"], resolved)
            self.assertTrue(os.path.isdir(resolved))
        finally:
            if previous is None:
                os.environ.pop("TORCH_HOME", None)
            else:
                os.environ["TORCH_HOME"] = previous


def _real_model_available():
    """True only when the caller opted in *and* torch can be imported."""
    if not HAVE_TORCH or os.environ.get("COTRACKER_REAL_TEST", "").lower() in ("", "0", "false"):
        return False
    return True


@unittest.skipUnless(_real_model_available(),
                     "set COTRACKER_REAL_TEST=1 (needs torch + a cached/downloadable model)")
class TestRealModel(unittest.TestCase):
    """End-to-end on the actual checkpoint: a white square translating across a frame."""

    def test_tracks_a_moving_square(self):
        tracker = ct.CoTrackerTracker()
        size, h, w = 12, 240, 320
        y0, x0 = 100, 60

        def synth(step):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[::16, :, :] = 40  # a little background texture
            x = x0 + 4 * step
            img[y0:y0 + size, x:x + size] = 255
            return img, (x + size / 2.0, y0 + size / 2.0)

        first, centre = synth(0)
        tracker.init(first, [[centre[0], centre[1]]], obj_id="square")
        last, expected = None, None
        for step in range(1, 20):
            img, expected = synth(step)
            last = tracker.update(img)
        self.assertTrue(bool(last.visible[0]))
        self.assertLessEqual(last.meta["stale_frames"], tracker.step)
        # Allow the staleness lag (the last flush may be a few frames old) plus slack.
        tolerance = 4 * last.meta["stale_frames"] + 8
        self.assertLess(abs(last.points[0][0] - expected[0]), tolerance)
        self.assertLess(abs(last.points[0][1] - expected[1]), tolerance)


if __name__ == "__main__":
    unittest.main()
