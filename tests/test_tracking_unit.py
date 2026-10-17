"""Unit tests for the rollout tracking subsystem - no simulator, no LLM.

Covers the parts that are pure functions of their inputs:
  * pixel <-> world round-trips and the depth/occlusion visibility test
  * weighted multi-camera fusion and robust centroids
  * the cross-camera re-seed policy (unseeded / low-confidence / disagreement)
  * the monitor return contract and the built-in invariants
  * the point-tracker providers on synthetic images

Run: python -m pytest tests\\test_tracking_unit.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from providers.trackers.factory import get_tracker  # noqa: E402
from sim_adapter import camera_math  # noqa: E402
from tracking import geometry, monitors  # noqa: E402
from tracking.health import CamHealthState, decide_reseeds, score_camera, temporal_continuity  # noqa: E402
from tracking.monitor import MonitorRunner, compile_monitor, normalise  # noqa: E402
from tracking.types import (  # noqa: E402
    STATUS_ABORT,
    STATUS_OK,
    STATUS_RECORD,
    STATUS_WARN,
    CameraView,
    CamTrack,
    GripperState,
    TrackedObjectState,
    TrackFrameReport,
)

WIDTH = HEIGHT = 128
NEAR, FAR = 0.01, 100.0


def make_view(name="head", eye=(0.0, -1.0, 0.5), target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0),
              depth_fill=None):
    """A CameraView with real GL view/projection matrices and a flat depth buffer."""
    view_matrix = camera_math.gl_view_matrix(eye, target, up)
    projection = camera_math.gl_projection_matrix(60.0, 1.0, NEAR, FAR)
    depth = np.full((HEIGHT, WIDTH), FAR if depth_fill is None else depth_fill, dtype=np.float32)
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return CameraView(name=name, rgb=rgb, depth=depth,
                      view_matrix=geometry.mat4(view_matrix),
                      projection_matrix=geometry.mat4(projection),
                      near=NEAR, far=FAR, position=list(eye))


def paint_depth(view, world_pos, radius=6):
    """Write the true metric depth of ``world_pos`` into a disc of the depth buffer."""
    pixel, z_eye = geometry.project_world_to_pixel(view, world_pos)
    assert pixel is not None
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    ys, xs = np.ogrid[:view.height, :view.width]
    mask = (xs - x) ** 2 + (ys - y) ** 2 <= radius ** 2
    view.depth[mask] = z_eye
    return pixel, z_eye


class TestGeometry(unittest.TestCase):
    def test_project_deproject_round_trip(self):
        view = make_view()
        world = np.array([0.05, 0.1, 0.2])
        pixel, z_eye = geometry.project_world_to_pixel(view, world)
        self.assertIsNotNone(pixel)
        back = geometry.deproject_pixel_to_world(view, pixel, z_eye)
        np.testing.assert_allclose(back, world, atol=2e-3)

    def test_point_behind_camera_is_rejected(self):
        view = make_view(eye=(0.0, -1.0, 0.5), target=(0.0, 0.0, 0.5))
        pixel, _z = geometry.project_world_to_pixel(view, np.array([0.0, -2.0, 0.5]))
        self.assertIsNone(pixel)

    def test_visibility_requires_matching_depth(self):
        view = make_view()
        world = np.array([0.0, 0.0, 0.1])
        paint_depth(view, world)
        visible, pixel, detail = geometry.is_visible(view, world)
        self.assertTrue(visible, detail)
        self.assertIsNotNone(pixel)

    def test_occluder_in_front_blocks_visibility(self):
        view = make_view()
        world = np.array([0.0, 0.0, 0.1])
        pixel, z_eye = paint_depth(view, world)
        # Something much closer to the camera now covers that pixel.
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        view.depth[y - 3:y + 4, x - 3:x + 4] = z_eye - 0.3
        visible, _pixel, detail = geometry.is_visible(view, world)
        self.assertFalse(visible)
        self.assertIn("occluded", detail)

    def test_out_of_frame_is_not_visible(self):
        view = make_view()
        visible, _pixel, detail = geometry.is_visible(view, np.array([5.0, 0.0, 0.1]))
        self.assertFalse(visible)
        self.assertEqual(detail, "out_of_frame")

    def test_robust_centroid_rejects_an_outlier(self):
        pts = np.array([[0.0, 0.0, 0.1], [0.01, 0.0, 0.1], [0.0, 0.01, 0.1], [3.0, 3.0, 3.0]])
        valid = np.ones(4, dtype=bool)
        centroid, used = geometry.robust_centroid(pts, valid)
        self.assertEqual(used, 3)
        self.assertLess(float(np.linalg.norm(centroid - np.array([0.0033, 0.0033, 0.1]))), 0.02)

    def test_fusion_weights_the_closer_camera_more(self):
        near_pt = np.array([0.0, 0.0, 0.0])
        far_pt = np.array([0.10, 0.0, 0.0])
        fused, disagreement, used = geometry.fuse_world_points({
            "wrist": (near_pt, geometry.camera_weight(1.0, 0.2, 4)),
            "head": (far_pt, geometry.camera_weight(1.0, 1.5, 4)),
        })
        self.assertEqual(sorted(used), ["head", "wrist"])
        self.assertAlmostEqual(disagreement, 0.10, places=6)
        # The wrist camera is 7.5x closer, so the fused point sits near its estimate.
        self.assertLess(fused[0], 0.02)

    def test_fusion_ignores_zero_weight_cameras(self):
        fused, disagreement, used = geometry.fuse_world_points({
            "head": (np.array([1.0, 1.0, 1.0]), 0.0),
            "wrist": (np.array([0.0, 0.0, 0.0]), 1.0),
        })
        self.assertEqual(used, ["wrist"])
        self.assertIsNone(disagreement)
        np.testing.assert_allclose(fused, np.zeros(3))

    def test_points_to_world_marks_invalid_depth(self):
        view = make_view()
        world = np.array([0.0, 0.0, 0.1])
        pixel, _z = paint_depth(view, world, radius=3)
        pts = np.array([pixel, [1.0, 1.0]])
        world_pts, valid = geometry.points_to_world(view, pts, np.ones(2, dtype=bool))
        self.assertTrue(valid[0])
        # The second point looks at the far plane, which is outside the valid depth range.
        self.assertFalse(valid[1])
        np.testing.assert_allclose(world_pts[0], world, atol=5e-3)


class TestHealthPolicy(unittest.TestCase):
    def _track(self, cam, seeded=True, conf=0.9, world=(0.0, 0.0, 0.1), health=0.9):
        t = CamTrack(cam=cam, seeded=seeded, confidence=conf, depth_valid=True,
                     world_point=None if world is None else np.array(world, dtype=float),
                     visible=np.ones(3, dtype=bool), points_2d=np.zeros((3, 2)))
        t.health = health
        return t

    def test_temporal_continuity_penalises_a_jump(self):
        self.assertAlmostEqual(temporal_continuity(np.zeros(3), np.zeros(3)), 1.0)
        self.assertEqual(temporal_continuity(np.zeros(3), np.array([10.0, 0, 0])), 0.0)

    def test_score_is_zero_without_a_world_point(self):
        track = self._track("head", world=None)
        self.assertEqual(score_camera(track, CamHealthState()), 0.0)

    def test_unseeded_camera_is_reseeded_from_the_fused_estimate(self):
        tracks = {"head": self._track("head"), "wrist": self._track("wrist", seeded=False, conf=0.0, world=None, health=0.0)}
        states = {c: CamHealthState() for c in tracks}
        decisions = decide_reseeds("mug", tracks, states, frame_idx=0, fused_world_point=np.array([0.0, 0.0, 0.1]))
        self.assertEqual([(cam, reason) for cam, reason, _d, _p in decisions], [("wrist", "unseeded")])

    def test_low_confidence_camera_is_reseeded_from_the_healthy_one(self):
        tracks = {"head": self._track("head", conf=0.1, health=0.1),
                  "wrist": self._track("wrist", conf=0.95, health=0.95)}
        states = {c: CamHealthState() for c in tracks}
        decisions = []
        for frame in range(config.track_reseed_patience):
            decisions = decide_reseeds("mug", tracks, states, frame_idx=frame)
        self.assertTrue(any(cam == "head" and reason == "low_confidence" for cam, reason, _d, _p in decisions))
        self.assertTrue(all(donor == "wrist" for _c, _r, donor, _p in decisions))

    def test_cooldown_suppresses_repeated_reseeds(self):
        tracks = {"head": self._track("head", conf=0.1, health=0.1),
                  "wrist": self._track("wrist", conf=0.95, health=0.95)}
        states = {c: CamHealthState() for c in tracks}
        states["head"].last_reseed_frame = 0
        for frame in range(config.track_reseed_patience):
            decisions = decide_reseeds("mug", tracks, states, frame_idx=frame)
        self.assertEqual(decisions, [])

    def test_disagreement_reseeds_the_less_healthy_camera(self):
        tracks = {"head": self._track("head", conf=0.4, world=(0.5, 0.5, 0.5), health=0.2),
                  "wrist": self._track("wrist", conf=0.95, world=(0.0, 0.0, 0.1), health=0.9)}
        states = {c: CamHealthState() for c in tracks}
        decisions = decide_reseeds("mug", tracks, states, frame_idx=3)
        self.assertEqual(len(decisions), 1)
        cam, reason, donor, point = decisions[0]
        self.assertEqual((cam, donor), ("head", "wrist"))
        self.assertIn("disagreement", reason)
        np.testing.assert_allclose(point, [0.0, 0.0, 0.1])

    def test_equally_healthy_cameras_are_not_reseeded_on_disagreement(self):
        tracks = {"head": self._track("head", world=(0.5, 0.5, 0.5), health=0.9),
                  "wrist": self._track("wrist", world=(0.0, 0.0, 0.1), health=0.9)}
        states = {c: CamHealthState() for c in tracks}
        self.assertEqual(decide_reseeds("mug", tracks, states, frame_idx=3), [])


def make_report(dist=None, lost=False, frame_idx=0, gripper=(0.0, 0.0, 0.2), name="mug",
                disagreement=None):
    """A minimal frame report with the object ``dist`` metres from the gripper."""
    report = TrackFrameReport(frame_idx=frame_idx)
    report.gripper = GripperState(world_pos=np.array(gripper, dtype=float))
    world = None if lost or dist is None else np.array(gripper, dtype=float) + np.array([dist, 0.0, 0.0])
    report.objects[name] = TrackedObjectState(name=name, world_point=world, lost=lost,
                                              disagreement=disagreement)
    return report


class TestMonitorContract(unittest.TestCase):
    def test_normalise_accepts_the_documented_shapes(self):
        self.assertEqual(normalise(None).status, STATUS_OK)
        self.assertEqual(normalise({"status": "abort", "reason": "x"}).status, STATUS_ABORT)
        self.assertEqual(normalise("record").status, STATUS_RECORD)
        self.assertEqual(normalise(False).status, STATUS_ABORT)
        self.assertEqual(normalise(True).status, STATUS_OK)

    def test_unknown_status_is_downgraded_to_warn(self):
        self.assertEqual(normalise({"status": "explode"}).status, STATUS_WARN)
        self.assertEqual(normalise(3.5).status, STATUS_WARN)

    def test_a_raising_monitor_cannot_break_the_rollout(self):
        def boom(_state):
            raise ZeroDivisionError("bad monitor")

        runner = MonitorRunner(boom)
        result = runner(make_report(0.01))
        self.assertEqual(result.status, STATUS_WARN)
        self.assertEqual(runner.error_count, 1)

    def test_runner_remembers_the_worst_status(self):
        statuses = iter([STATUS_OK, STATUS_ABORT, STATUS_OK])
        runner = MonitorRunner(lambda _s: {"status": next(statuses), "reason": "r"})
        for _ in range(3):
            runner(make_report(0.01))
        self.assertEqual(runner.worst.status, STATUS_ABORT)

    def test_compile_monitor_from_source(self):
        fn = compile_monitor("def check(state):\n    return {'status': 'abort', 'reason': 'nope'}\n")
        self.assertEqual(normalise(fn(make_report(0.01))).status, STATUS_ABORT)

    def test_from_spec_builds_a_builtin_by_name(self):
        runner = MonitorRunner.from_spec({"builtin": "attached_to_gripper",
                                          "kwargs": {"name": "mug", "max_dist": 0.05,
                                                     "grace_frames": 1}})
        runner(make_report(0.01))                      # arms the invariant
        self.assertEqual(runner(make_report(0.5)).status, STATUS_ABORT)

    def test_from_spec_defaults_to_the_default_monitor(self):
        runner = MonitorRunner.from_spec(None)
        self.assertEqual(runner(make_report(0.01)).status, STATUS_OK)


class TestBuiltinMonitors(unittest.TestCase):
    def test_attached_to_gripper_arms_only_after_the_grasp(self):
        monitor = monitors.attached_to_gripper("mug", max_dist=0.05, grace_frames=1)
        # Far away while approaching - not a failure, the invariant is not armed yet.
        self.assertEqual(normalise(monitor(make_report(0.4))).status, STATUS_OK)
        self.assertEqual(normalise(monitor(make_report(0.01))).status, STATUS_OK)   # grasped
        self.assertEqual(normalise(monitor(make_report(0.4))).status, STATUS_ABORT)  # dropped

    def test_attached_to_gripper_tolerates_a_single_bad_frame(self):
        monitor = monitors.attached_to_gripper("mug", max_dist=0.05, grace_frames=3)
        monitor(make_report(0.01))
        self.assertEqual(normalise(monitor(make_report(0.4))).status, STATUS_WARN)
        self.assertEqual(normalise(monitor(make_report(0.01))).status, STATUS_OK)

    def test_attached_to_gripper_ignores_missing_measurements(self):
        monitor = monitors.attached_to_gripper("mug", max_dist=0.05, grace_frames=1)
        monitor(make_report(0.01))
        self.assertEqual(normalise(monitor(make_report(lost=True))).status, STATUS_OK)

    def test_object_not_lost_escalates_after_patience(self):
        monitor = monitors.object_not_lost("mug", patience=2)
        monitor(make_report(0.05))                       # seen once
        self.assertEqual(normalise(monitor(make_report(lost=True))).status, STATUS_WARN)
        self.assertEqual(normalise(monitor(make_report(lost=True))).status, STATUS_RECORD)

    def test_object_not_lost_warns_on_camera_disagreement(self):
        monitor = monitors.object_not_lost("mug")
        result = normalise(monitor(make_report(0.05, disagreement=config.track_disagree_m + 0.1)))
        self.assertEqual(result.status, STATUS_WARN)
        self.assertIn("disagree", result.reason)

    def test_stays_within(self):
        monitor = monitors.stays_within("mug", ((-1, -1, 0), (1, 1, 1)), grace_frames=1)
        self.assertEqual(normalise(monitor(make_report(0.0, gripper=(0.0, 0.0, 0.5)))).status, STATUS_OK)
        self.assertEqual(normalise(monitor(make_report(0.0, gripper=(0.0, 0.0, 5.0)))).status, STATUS_ABORT)

    def test_moved_at_least_flags_a_door_that_never_opens(self):
        monitor = monitors.moved_at_least("mug", 0.05, after_frames=2)
        for frame in range(4):
            result = normalise(monitor(make_report(0.0, frame_idx=frame)))
        self.assertEqual(result.status, STATUS_RECORD)

    def test_combine_takes_the_worst_status(self):
        monitor = monitors.combine(lambda _s: {"status": STATUS_WARN, "reason": "a"},
                                   lambda _s: {"status": STATUS_ABORT, "reason": "b"})
        result = normalise(monitor(make_report(0.01)))
        self.assertEqual(result.status, STATUS_ABORT)
        self.assertIn("b", result.reason)

    def test_default_monitor_discovers_targets_and_aborts_on_a_drop(self):
        monitor = monitors.default_monitor(max_dist=0.05)
        self.assertEqual(normalise(monitor(make_report(0.01))).status, STATUS_OK)
        for _ in range(config.track_attach_grace_frames):
            result = normalise(monitor(make_report(0.5)))
        self.assertEqual(result.status, STATUS_ABORT)


def synthetic_frame(size=96, square=None):
    """A textured background with a bright square - something a tracker can lock onto."""
    rng = np.random.RandomState(0)
    img = (rng.rand(size, size, 3) * 60).astype(np.uint8)
    if square is not None:
        cx, cy, half = square
        img[cy - half:cy + half, cx - half:cx + half] = 240
        # Break the square's symmetry so template matching has a unique peak.
        img[cy - half:cy, cx - half:cx] = 120
    return img


class TestTrackerProviders(unittest.TestCase):
    def test_factory_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            get_tracker("does-not-exist")

    def test_remote_provider_is_lazy_and_fails_loudly_without_a_server(self):
        # autolaunch=False: the default would start a gpu_server for a local URL.
        tracker = get_tracker("remote", url="tcp://127.0.0.1:1", timeout=2, autolaunch=False)  # no I/O yet
        with self.assertRaises(ConnectionError):
            tracker.init(synthetic_frame(square=(40, 40, 10)), [[40, 40]])

    def test_template_tracker_follows_a_moving_patch(self):
        tracker = get_tracker("template", patch_half=8)
        frame = synthetic_frame(square=(40, 40, 10))
        tracker.init(frame, [[40, 40]])
        for step in range(1, 6):
            moved = synthetic_frame(square=(40 + 3 * step, 40, 10))
            result = tracker.update(moved)
            self.assertEqual(result.n_visible, 1)
        np.testing.assert_allclose(result.points[0], [55.0, 40.0], atol=2.0)
        self.assertGreater(result.confidence, 0.5)

    def test_template_tracker_reports_loss_when_the_patch_disappears(self):
        tracker = get_tracker("template", patch_half=8)
        tracker.init(synthetic_frame(square=(40, 40, 10)), [[40, 40]])
        rng = np.random.RandomState(1)
        for _ in range(3):
            result = tracker.update((rng.rand(96, 96, 3) * 255).astype(np.uint8))
        self.assertEqual(result.n_visible, 0)
        self.assertEqual(result.confidence, 0.0)

    def test_template_tracker_refuses_a_featureless_point(self):
        flat = np.zeros((96, 96, 3), dtype=np.uint8)
        tracker = get_tracker("template", patch_half=8)
        with self.assertRaises(ValueError):
            tracker.init(flat, [[40, 40]])

    def test_reset_requires_reinit(self):
        tracker = get_tracker("template", patch_half=8)
        tracker.init(synthetic_frame(square=(40, 40, 10)), [[40, 40]])
        tracker.reset()
        self.assertFalse(tracker.initialised)
        self.assertEqual(tracker.update(synthetic_frame(square=(40, 40, 10))).n_visible, 0)

    def test_tracker_tracks_several_points_independently(self):
        tracker = get_tracker("template", patch_half=6)
        frame = synthetic_frame(size=128, square=(30, 30, 8))
        frame[70:86, 70:86] = 200
        frame[70:78, 70:78] = 90
        tracker.init(frame, [[30, 30], [78, 78]])
        self.assertEqual(tracker.n_points, 2)
        result = tracker.update(frame)
        self.assertEqual(result.n_visible, 2)

    def test_untrackable_seeds_keep_their_slot(self):
        # One output row per seed, in seed order: an edge / featureless seed becomes a
        # never-visible slot instead of being dropped (which would shift every later
        # correspondence onto the wrong template point).
        tracker = get_tracker("template", patch_half=6)
        frame = synthetic_frame(size=128, square=(30, 30, 8))
        frame[70:86, 70:86] = 200
        frame[70:78, 70:78] = 90
        frame[5:40, 95:125] = 50                            # flat: no NCC template
        seeds = [[30, 30], [2, 2], [110, 20], [78, 78]]     # [1] off-edge, [2] featureless
        tracker.init(frame, seeds)
        self.assertEqual(tracker.n_points, 4)
        for _ in range(3):
            result = tracker.update(frame)
            self.assertEqual(len(result.points), 4)
            np.testing.assert_array_equal(result.visible, [True, False, False, True])
        np.testing.assert_allclose(result.points[3], [78, 78], atol=1.0)


class TestCSRTProvider(unittest.TestCase):
    def test_csrt_is_either_usable_or_explains_the_missing_package(self):
        from providers.trackers import csrt_tracker

        if not csrt_tracker.csrt_available():
            with self.assertRaises(RuntimeError) as ctx:
                get_tracker("csrt")
            self.assertIn("opencv-contrib-python", str(ctx.exception))
            self.skipTest("opencv-contrib-python is not installed")
        tracker = get_tracker("csrt", patch_half=8)
        frame = synthetic_frame(square=(40, 40, 10))
        tracker.init(frame, [[40, 40]])
        result = tracker.update(synthetic_frame(square=(46, 40, 10)))
        self.assertEqual(result.n_visible, 1)


class _FakeWindowAnchor:
    """A windowed tracker stand-in: exact points every ``step`` frames, the old ones between."""

    name = "fake_anchor"
    keeps_memory_through_occlusion = True

    def __init__(self, truth, step=4):
        self.truth, self.step = truth, step
        self.t = 0
        self._flushes = 0
        self.inits = 0
        self.n_points = 0
        self.last = None

    def init(self, frame, points, obj_id=None):
        self.inits += 1
        self.n_points = len(points)
        self.last = np.asarray(points, dtype=float)

    def reset(self):
        pass

    def update(self, frame):
        from providers.trackers.base import TrackResult
        self.t += 1
        stale = self.t % self.step
        if stale == 0:
            self._flushes += 1
            self.last = self.truth(self.t)
        n = len(self.last)
        return TrackResult(points=self.last.copy(), visible=np.ones(n, bool), scores=np.ones(n),
                           meta={"stale_frames": stale, "flushes": self._flushes})


class TestHybridTracker(unittest.TestCase):
    SEEDS = np.array([[40.0, 40.0], [60.0, 45.0], [50.0, 70.0], [70.0, 65.0]])

    @staticmethod
    def frame_at(t):
        rng = np.random.RandomState(3)
        tex = (rng.rand(128, 128, 3) * 255).astype(np.uint8)
        import cv2 as cv
        tex = cv.GaussianBlur(tex, (5, 5), 1.0)
        return np.roll(tex, shift=2 * t, axis=1)       # 2 px/frame to the right

    def make(self, motion_model="similarity"):
        from providers.trackers.hybrid_tracker import HybridTracker
        truth = lambda t: self.SEEDS + [2.0 * t, 0.0]   # noqa: E731
        anchor = _FakeWindowAnchor(truth)
        tracker = HybridTracker(anchor=anchor, bridge="klt", motion_model=motion_model)
        tracker.init(self.frame_at(0), self.SEEDS)
        return tracker, anchor, truth

    def test_bridge_measures_the_frames_between_anchor_flushes(self):
        for model in ("similarity", "points"):
            tracker, anchor, truth = self.make(model)
            for t in range(1, 10):
                result = tracker.update(self.frame_at(t))
                # Every frame is a current measurement, not a stale window.
                self.assertEqual(result.meta["stale_frames"], 0, (model, t))
                self.assertEqual(result.meta["source"], "anchor" if t % 4 == 0 else "bridge")
                np.testing.assert_allclose(result.points, truth(t), atol=0.5, err_msg=f"{model} t={t}")
                self.assertTrue(np.all(result.visible))
            # Flushes are forwarded for the latency model; the anchor is never re-initialised.
            self.assertEqual(tracker._flushes, anchor._flushes)
            self.assertEqual(result.meta["flushes"], 2)
            self.assertEqual(anchor.inits, 1)

    def test_factory_and_memory_flag(self):
        from providers.trackers.factory import SUPPORTED, uses_gpu_server
        self.assertIn("hybrid", SUPPORTED)
        self.assertTrue(uses_gpu_server("hybrid_remote"))
        self.assertFalse(uses_gpu_server("hybrid"))
        tracker, _anchor, _truth = self.make()
        self.assertTrue(tracker.keeps_memory_through_occlusion)
        self.assertFalse(get_tracker("klt").keeps_memory_through_occlusion)


class TestMemoryTrackerReseedPolicy(unittest.TestCase):
    """A tracker that keeps memory through occlusion (CoTracker) is not reset on occlusion."""

    def holds(self, tracker, reason, seeded=True):
        from types import SimpleNamespace

        from tracking.session import TrackingSession
        session = SimpleNamespace(memory_holds={})
        target = SimpleNamespace(trackers={"head": tracker}, seeded={"head": seeded})
        return TrackingSession._memory_holds(session, target, "head", reason), session.memory_holds

    def test_occlusion_reasons_are_skipped_only_for_memory_trackers(self):
        memory = _FakeWindowAnchor(lambda t: TestHybridTracker.SEEDS)
        for reason in ("lost", "low_confidence", "reacquire"):
            held, counts = self.holds(memory, reason)
            self.assertTrue(held, reason)
            self.assertEqual(counts, {reason: 1})
            self.assertFalse(self.holds(get_tracker("klt"), reason)[0])
        # Verified repairs and first seeding still go through.
        for reason in ("unseeded", "jumped", "disagreement", "pose_correction", "reacquire_roma"):
            self.assertFalse(self.holds(memory, reason)[0], reason)
        self.assertFalse(self.holds(memory, "lost", seeded=False)[0])

    def test_policy_can_be_disabled(self):
        memory = _FakeWindowAnchor(lambda t: TestHybridTracker.SEEDS)
        saved = config.track_memory_skip_reseed_reasons
        config.track_memory_skip_reseed_reasons = ()
        try:
            self.assertFalse(self.holds(memory, "lost")[0])
        finally:
            config.track_memory_skip_reseed_reasons = saved


class _FakeRomaClient:
    """Stands in for GpuClient: maps the reference seed pixels to ``points_b``."""

    def __init__(self, points_b, overlap):
        self.points_b, self.overlap, self.calls = points_b, overlap, 0

    def call(self, module, method, **kw):
        assert (module, method) == ("romav2", "match")
        self.calls += 1
        n = len(kw["points_a"])
        return {"match_s": 0.01}, {"points_b": self.points_b[:n].copy(),
                                   "overlap": np.full(n, self.overlap)}


class TestRomaRedetector(unittest.TestCase):
    # Fronto-parallel camera looking at the plane y=0, 1 m away, so depth is constant.
    SEEDS = np.array([[x, 0.0, z] for x in (-0.05, 0.0, 0.05) for z in (-0.04, 0.04)])
    SHIFT = np.array([0.03, 0.0, 0.02])

    def setUp(self):
        from tracking.types import ObjectPose
        self.view = make_view(eye=(0.0, -1.0, 0.0), target=(0.0, 0.0, 0.0), depth_fill=1.0)
        self.identity = ObjectPose(R=np.eye(3), t=np.zeros(3))
        self.moved_px = np.array([geometry.project_world_to_pixel(self.view, p + self.SHIFT)[0]
                                  for p in self.SEEDS])

    def redetect(self, overlap=0.9, max_dev_m=0.2):
        from tracking.roma_redetect import RomaRedetector
        ref = RomaRedetector.capture(self.view, self.identity, self.SEEDS, frame=0)
        self.assertIsNotNone(ref)
        self.assertEqual(len(ref.indices), len(self.SEEDS))
        roma = RomaRedetector(client=_FakeRomaClient(self.moved_px, overlap))
        out = roma.redetect({"head": self.view}, {"head": ref}, self.SEEDS,
                            [("last", self.identity, 0.0)], max_dev_m=max_dev_m)
        return roma, out

    def test_hit_recovers_the_moved_pose(self):
        roma, (R, t, info) = self.redetect()
        self.assertIsNotNone(R, info)
        np.testing.assert_allclose(t, self.SHIFT, atol=5e-3)
        self.assertEqual(roma.summary()["hits"], 1)

    def test_low_overlap_is_a_miss(self):
        _roma, (R, _t, info) = self.redetect(overlap=0.1)
        self.assertIsNone(R)
        self.assertIn("no rigid consensus", info["why"])

    def test_implausible_jump_is_rejected(self):
        _roma, (R, _t, info) = self.redetect(max_dev_m=0.01)
        self.assertIsNone(R)
        self.assertIn("implausible jump", info["why"])


class TestTrackingComponentsCli(unittest.TestCase):
    def parse(self, *argv):
        import argparse
        from tracking import session as ts
        parser = argparse.ArgumentParser()
        ts.add_tracking_args(parser)
        return parser.parse_args(list(argv))

    def test_cotracker_and_roma(self):
        args = self.parse("--tracking", "cotracker", "roma")
        self.assertTrue(args.tracking)
        self.assertEqual(args.tracker_provider, "remote")
        self.assertTrue(args.track_roma)

    def test_bare_tracking_keeps_defaults(self):
        args = self.parse("--tracking")
        self.assertTrue(args.tracking)
        self.assertFalse(args.track_roma)

    def test_two_trackers_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.parse("--tracking", "cotracker", "ncc")


if __name__ == "__main__":
    unittest.main()
