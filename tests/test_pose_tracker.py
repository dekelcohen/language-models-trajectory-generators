"""Unit tests for the point-tracking pose pipeline (``pose_tracker`` + ``RigidBodyKF``).

No simulator: two synthetic OpenGL cameras whose depth buffers are painted exactly at the
pixels of known world points, so every 3D reading is ground truth unless a test corrupts it.

Run: python -m pytest tests/test_pose_tracker.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_tracker3d import make_track, make_view, project_into  # noqa: E402

from providers.tracker3d import get_tracker3d  # noqa: E402
from providers.tracker3d.pose_tracker import PoseTracker3D, epipolar_distance  # noqa: E402
from tracking import geometry  # noqa: E402
from sim_adapter.transforms import exp_so3  # noqa: E402
from tracking.kalman import RigidBodyKF  # noqa: E402
from tracking.monitors import attached_to_gripper, object_not_lost  # noqa: E402
from tracking.types import GripperState, ObjectPose, TrackedObjectState, TrackFrameReport  # noqa: E402

BACKGROUND_M = 3.0

# Spread, non-collinear points (~6 cm extent) so rotation is observable.
TEMPLATE = np.array([
    [0.00, 0.40, 0.15],
    [0.06, 0.38, 0.19],
    [-0.05, 0.43, 0.12],
    [0.02, 0.45, 0.24],
    [-0.03, 0.36, 0.20],
], dtype=float)
CENTROID = TEMPLATE.mean(axis=0)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def posed(R=np.eye(3), shift=(0.0, 0.0, 0.0)):
    """Template rotated about its centroid by ``R`` and shifted."""
    return (TEMPLATE - CENTROID) @ R.T + CENTROID + np.asarray(shift, dtype=float)


def views_with(points):
    """Two cameras whose depth buffers image exactly ``points`` (3x3 patches)."""
    out = {}
    for name, eye in (("head", (0.9, 1.1, 0.8)), ("wrist", (-0.7, 1.0, 0.6))):
        view = make_view(name, eye)
        view.depth = np.full_like(view.depth, BACKGROUND_M)
        for p in points:
            px, z = geometry.project_world_to_pixel(view, p)
            x, y = int(round(px[0])), int(round(px[1]))
            view.depth[y - 1:y + 2, x - 1:x + 2] = z
        out[name] = view
    return out


def tracks_for(views, points, stale=0, only=None):
    tracks = {}
    for cam, view in views.items():
        if only is not None and cam not in only:
            continue
        track = make_track(cam, project_into(view, points))
        track.stale_frames = stale
        tracks[cam] = track
    return tracks


class TestRigidBodyKF(unittest.TestCase):
    def test_learns_constant_velocity(self):
        kf = RigidBodyKF()
        v = np.array([0.01, 0.0, -0.005])
        for k in range(15):
            kf.predict()
            kf.update(v * k, np.eye(3))
        np.testing.assert_allclose(kf.velocity, v, atol=1.5e-3)
        kf.predict()
        np.testing.assert_allclose(kf.position, v * 15, atol=3e-3)

    def test_single_outlier_is_rejected_but_consistent_jump_reinitialises(self):
        kf = RigidBodyKF()
        for _ in range(5):
            kf.predict()
            kf.update(np.zeros(3), np.eye(3))
        kf.predict()
        first = kf.update(np.array([0.3, 0.0, 0.0]), np.eye(3))
        self.assertFalse(first.accepted)
        kf.predict()
        second = kf.update(np.array([0.3, 0.0, 0.0]), np.eye(3))
        self.assertTrue(second.accepted and second.reset)   # a real drop is not denied forever
        np.testing.assert_allclose(kf.position, [0.3, 0.0, 0.0], atol=1e-9)

    def test_goes_stale_after_max_coast(self):
        kf = RigidBodyKF()
        kf.update(np.zeros(3), np.eye(3))
        for _ in range(kf.cfg.max_coast + 1):
            kf.predict()
        self.assertTrue(kf.stale)


class TestEpipolarDistance(unittest.TestCase):
    def test_true_correspondence_is_on_the_line_and_offset_is_not(self):
        views = views_with(TEMPLATE)
        pa = project_into(views["head"], TEMPLATE[:1])[0]
        pb = project_into(views["wrist"], TEMPLATE[:1])[0]
        self.assertLess(epipolar_distance(views["head"], pa, views["wrist"], pb), 0.05)
        wrong = project_into(views["wrist"], TEMPLATE[:1] + [0.0, 0.0, 0.05])[0]
        self.assertGreater(epipolar_distance(views["head"], pa, views["wrist"], wrong), 2.0)


class TestPoseTrackerModes(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseTracker3D()

    def lift(self, points, tracks=None, commit=True):
        views = views_with(points)
        tracks = tracks_for(views, points) if tracks is None else tracks(views)
        return self.tracker.lift("obj", views, tracks, seed_world_points=TEMPLATE,
                                 commit=commit)

    def test_green_recovers_rotation_and_translation(self):
        # Gradual motion (2 deg + 1 mm per frame): the output is Kalman-filtered, so a
        # one-frame 20 deg jump would (correctly) be partly smoothed.
        for k in range(11):
            R = rot_z(2.0 * k)
            shift = (0.001 * k, 0.0, 0.0)
            res = self.lift(posed(R, shift))
        self.assertEqual(res.meta["mode"], "green")
        self.assertTrue(res.meta["rotation_observable"])
        self.assertAlmostEqual(res.pose.rotation_deg, 20.0, delta=1.0)
        np.testing.assert_allclose(res.pose.apply(TEMPLATE), posed(R, shift), atol=3e-3)
        np.testing.assert_allclose(res.world_point, CENTROID + shift, atol=3e-3)
        self.assertFalse(res.predicted)

    def test_tight_cluster_is_translation_only(self):
        tight = CENTROID + (TEMPLATE - CENTROID) * 0.1        # < 1 cm extent
        tracker = PoseTracker3D()
        views = views_with(tight)
        res = tracker.lift("obj", views, tracks_for(views, tight), seed_world_points=tight)
        self.assertEqual(res.meta["mode"], "green")
        self.assertFalse(res.meta["rotation_observable"])
        self.assertAlmostEqual(res.pose.rotation_deg, 0.0, delta=1e-6)

    def test_points_slipping_onto_background_are_rejected_by_the_jump_gate(self):
        for k in range(4):
            self.lift(posed(shift=(0.005 * k, 0.0, 0.0)))
        truth = posed(shift=(0.02, 0.0, 0.0))

        def slipped(views):
            tracks = tracks_for(views, truth)
            # head points 0-1 now sample the background (depth 3 m): a >1 m 3D jump.
            px = tracks["head"].points_2d.copy()
            px[:2] += [25.0, 25.0]
            tracks["head"].points_2d = px
            return tracks

        res = self.lift(truth, tracks=slipped)
        self.assertIn(res.meta["mode"], ("green", "yellow"))
        self.assertGreaterEqual(res.meta["rejected_jump"].get("head", 0), 2)
        np.testing.assert_allclose(res.world_point, truth.mean(axis=0), atol=5e-3)

    def test_no_measurement_predicts_and_asks_to_reacquire(self):
        for k in range(5):
            self.lift(posed(shift=(0.01 * k, 0.0, 0.0)))
        res = self.lift(TEMPLATE, tracks=lambda views: {})
        self.assertEqual(res.meta["mode"], "red")
        self.assertTrue(res.predicted and res.meta["reacquire"])
        self.assertEqual(res.pose.source, "predicted")
        # the prediction carries the motion on (constant velocity ~1 cm/frame)
        self.assertGreater(res.world_point[0], CENTROID[0] + 0.045)

    def test_red_turns_black_after_black_after_frames(self):
        tracker = PoseTracker3D(black_after=2)
        views = views_with(TEMPLATE)
        tracker.lift("obj", views, tracks_for(views, TEMPLATE), seed_world_points=TEMPLATE)
        modes = [tracker.lift("obj", views, {}, seed_world_points=TEMPLATE).meta["mode"]
                 for _ in range(4)]
        self.assertEqual(modes, ["red", "red", "black", "black"])

    def test_stale_windowed_tracker_coasts_without_escalating(self):
        for k in range(4):
            self.lift(posed(shift=(0.01 * k, 0.0, 0.0)))
        stale = lambda views: tracks_for(views, TEMPLATE, stale=3)  # noqa: E731
        res = self.lift(TEMPLATE, tracks=stale)
        self.assertEqual(res.meta["mode"], "coast")
        self.assertTrue(res.predicted)
        self.assertNotIn("reacquire", res.meta)
        self.assertEqual(self.tracker._objects["obj"].unmeasured_frames, 0)
        # predicted forward, not frozen at the stale points
        self.assertGreater(res.world_point[0], CENTROID[0] + 0.035)

    def test_yellow_when_cameras_disagree_and_one_fits(self):
        for _ in range(3):
            self.lift(TEMPLATE)
        truth = TEMPLATE

        def wrist_scrambled(views):
            tracks = tracks_for(views, truth)
            # wrist identities shuffled: each point reads another point's position -> the
            # fused rigid fit breaks, the head-only fit is still perfect.
            tracks["wrist"].points_2d = tracks["wrist"].points_2d[[1, 2, 3, 4, 0]]
            return tracks

        tracker = PoseTracker3D(epipolar_px=1e6, jump_m=1.0, fuse_tol_m=1.0)
        views = views_with(truth)
        tracker.lift("obj", views, tracks_for(views, truth), seed_world_points=TEMPLATE)
        res = tracker.lift("obj", views, wrist_scrambled(views), seed_world_points=TEMPLATE)
        self.assertEqual(res.meta["mode"], "yellow")
        self.assertEqual(res.used_cams, ["head"])
        self.assertEqual(res.meta["correct_cams"], ["wrist"])

    def test_dry_run_does_not_advance_the_filter(self):
        for k in range(3):
            self.lift(posed(shift=(0.01 * k, 0.0, 0.0)))
        kf = self.tracker._objects["obj"].kf
        before = (kf.x.copy(), kf.P.copy())
        self.lift(posed(shift=(0.2, 0.0, 0.0)), commit=False)
        after = self.tracker._objects["obj"].kf
        np.testing.assert_array_equal(after.x, before[0])
        np.testing.assert_array_equal(after.P, before[1])

    def test_factory_and_never_raises(self):
        provider = get_tracker3d("pose_tracker")
        self.assertEqual(provider.name, "pose_tracker")
        views = views_with(TEMPLATE)
        self.assertFalse(provider.lift("obj", views, {}).ok)             # no template
        self.assertFalse(provider.lift("obj", views, {}, seed_world_points=TEMPLATE).ok)


class TestMonitorsIgnorePredictions(unittest.TestCase):
    def frame(self, obj_pos, grip_pos, predicted=False, mode="green", idx=0):
        rep = TrackFrameReport(frame_idx=idx)
        rep.gripper = GripperState(world_pos=np.asarray(grip_pos, dtype=float))
        state = TrackedObjectState(name="cube", world_point=np.asarray(obj_pos, dtype=float),
                                   lost=False)
        state.predicted = predicted
        state.pose = ObjectPose(R=np.eye(3), t=np.zeros(3),
                                source="predicted" if predicted else "measured", mode=mode)
        rep.objects["cube"] = state
        return rep

    def test_fk_coast_is_not_evidence_of_attachment(self):
        mon = attached_to_gripper("cube", max_dist=0.05, grace_frames=2)
        g = [0.0, 0.0, 0.5]
        self.assertEqual(mon(self.frame(g, g))["status"], "ok")               # armed
        self.assertEqual(mon(self.frame([0.3, 0, 0.1], g))["status"], "warn")  # dropped
        # an FK-coasted frame would put it back in the hand - it must not reset the count
        self.assertTrue(mon(self.frame(g, g, predicted=True, mode="black"))["predicted"])
        self.assertEqual(mon(self.frame([0.3, 0, 0.1], g))["status"], "abort")

    def test_red_prediction_counts_as_unseen_but_coast_does_not(self):
        mon = object_not_lost("cube", patience=2)
        g = [0.0, 0.0, 0.5]
        mon(self.frame(g, g))
        self.assertEqual(mon(self.frame(g, g, predicted=True, mode="coast"))["status"], "ok")
        self.assertEqual(mon(self.frame(g, g, predicted=True, mode="red"))["status"], "warn")
        self.assertEqual(mon(self.frame(g, g, predicted=True, mode="red"))["status"], "record")


if __name__ == "__main__":
    unittest.main()
