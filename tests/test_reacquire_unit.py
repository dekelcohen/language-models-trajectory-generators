"""Unit tests: search-based re-acquisition, pose injection, coherence-scaled twist (§13)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_pose_tracker import CENTROID, TEMPLATE, posed, rot_z, tracks_for, views_with  # noqa: E402
from providers.tracker3d.pose_tracker import PoseTracker3D  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.kalman import KFConfig, RigidBodyKF  # noqa: E402
from tracking.reacquire import capture_bank, kabsch, ransac_pose, verify_pose  # noqa: E402
from tracking.types import ObjectPose  # noqa: E402

GRID = np.array([[x, y, z] for x in (-0.04, 0.0, 0.04) for y in (-0.03, 0.03)
                 for z in (0.0, 0.02)], dtype=float) + [0.0, 0.4, 0.15]


class TestKabschRansac(unittest.TestCase):
    def setUp(self):
        self.R, self.t = rot_z(25.0), np.array([0.05, -0.02, 0.01])
        self.truth = GRID @ self.R.T + self.t

    def test_kabsch_recovers_rigid_motion(self):
        R, t = kabsch(GRID, self.truth)
        np.testing.assert_allclose(R, self.R, atol=1e-9)
        np.testing.assert_allclose(t, self.t, atol=1e-9)

    def test_ransac_finds_pose_among_wrong_candidates(self):
        rng = np.random.default_rng(1)
        cands = {}
        for i, p in enumerate(self.truth):
            wrong = p + rng.uniform(-0.2, 0.2, size=(2, 3))
            cands[i] = np.vstack([wrong, p[None]]) if i % 4 else wrong   # 1/4 have no true peak
        fit = ransac_pose(GRID, cands, 400, 0.01, 6)
        self.assertIsNotNone(fit)
        R, t, inl, rmse = fit
        np.testing.assert_allclose(GRID @ R.T + t, self.truth, atol=1e-6)
        self.assertEqual(len(inl), sum(1 for i in range(len(GRID)) if i % 4))
        self.assertLess(rmse, 1e-6)

    def test_three_agreeing_points_are_not_enough_with_min_inliers_four(self):
        # Any 3 non-degenerate candidates fit *some* rigid pose - the measured door wrong-lock.
        cands = {i: self.truth[i][None] for i in range(3)}
        rng = np.random.default_rng(2)
        cands.update({i: GRID[i][None] + rng.uniform(0.1, 0.3, size=(1, 3))
                      for i in range(3, len(GRID))})
        self.assertIsNone(ransac_pose(GRID, cands, 300, 0.01, 4))


class TestVerifyPose(unittest.TestCase):
    def test_true_pose_verifies_and_shifted_pose_does_not(self):
        views = views_with(TEMPLATE)
        rng = np.random.default_rng(3)
        for view in views.values():
            view.rgb = rng.integers(0, 255, size=view.rgb.shape, dtype=np.uint8)
        eye = ObjectPose(R=np.eye(3), t=np.zeros(3), source="measured", mode="green")
        banks = {c: capture_bank(v, eye, TEMPLATE, 0, half=4, angles=(0,))
                 for c, v in views.items()}
        self.assertGreaterEqual(sum(len(b.templates) for b in banks.values()), 6)
        ok, chk = verify_pose(views, banks, TEMPLATE, np.eye(3), np.zeros(3))
        self.assertEqual(ok, chk)
        self.assertGreater(ok, 0)
        # 1.5 cm off: projections leave the depth-verified patches -> nothing verifies
        ok2, _chk2 = verify_pose(views, banks, TEMPLATE, np.eye(3), np.array([0.015, 0.0, 0.0]))
        self.assertEqual(ok2, 0)


class TestTemplateExtension(unittest.TestCase):
    def test_new_face_points_measure_the_pose_and_keep_the_centroid(self):
        from test_pose_tracker import make_track, project_into
        tr = PoseTracker3D()
        views = views_with(TEMPLATE)
        tr.lift("obj", views, tracks_for(views, TEMPLATE), seed_world_points=TEMPLATE)
        new = TEMPLATE[:3] + [0.0, 0.0, -0.03]
        self.assertEqual(tr.extend_template("obj", new), len(TEMPLATE))
        np.testing.assert_allclose(tr._objects["obj"].centroid, CENTROID)
        full = np.vstack([TEMPLATE, new])
        for k in range(1, 7):
            shift = np.array([0.002 * k, 0.0, 0.0])
            pts = full + shift
            views = views_with(pts)
            # head images only the original face, wrist only the new points
            tracks = {"head": make_track("head", project_into(views["head"], pts[:5])),
                      "wrist": make_track("wrist", project_into(views["wrist"], pts[5:]))}
            index = {"head": np.arange(5), "wrist": np.arange(5, 8)}
            res = tr.lift("obj", views, tracks, point_index=index)
        self.assertEqual(res.meta["mode"], "green")
        np.testing.assert_allclose(res.world_point, CENTROID + shift, atol=3e-3)

    def test_extend_unknown_object_is_a_noop(self):
        self.assertIsNone(PoseTracker3D().extend_template("nope", TEMPLATE))


class TestRotationObservability(unittest.TestCase):
    def test_rotation_std_follows_pixel_noise_and_cluster_size(self):
        tr = PoseTracker3D()
        sigma = {i: 0.01 for i in range(5)}                 # 1 cm per point
        big = tr._rotation_std(TEMPLATE, range(5), sigma)
        small = tr._rotation_std(CENTROID + (TEMPLATE - CENTROID) * 0.25, range(5), sigma)
        self.assertAlmostEqual(small / big, 4.0, places=6)
        self.assertEqual(tr._rotation_std(TEMPLATE, range(5), {}), tr.kf_config.rot_meas_std)

    def test_unmeasurable_rotation_is_held(self):
        # Huge assumed pixel noise -> rotation std above the limit -> translation-only fit.
        tr = PoseTracker3D(px_noise=50.0)
        for k in range(6):
            pts = posed(rot_z(3.0 * k), (0.001 * k, 0.0, 0.0))
            views = views_with(pts)
            res = tr.lift("obj", views, tracks_for(views, pts), seed_world_points=TEMPLATE)
        self.assertFalse(res.meta["rotation_observable"])
        self.assertAlmostEqual(res.pose.rotation_deg, 0.0, delta=1e-6)

class TestFreeSpaceContradiction(unittest.TestCase):
    """A pose one camera's patches verify must still not be where another camera sees
    through (the D4 wrong lock: head-only 4/4 inliers, 5/5 verified, 0.40 m off)."""

    def _views(self, wrist_sees_object):
        views = views_with(TEMPLATE)
        if not wrist_sees_object:
            views["wrist"] = views_with(TEMPLATE + np.array([0.3, 0.0, 0.0]))["wrist"]
        rng = np.random.default_rng(3)
        for view in views.values():
            view.rgb = rng.integers(0, 255, size=view.rgb.shape, dtype=np.uint8)
        return views

    def test_counts(self):
        from tracking.reacquire import free_space_counts
        s, c = free_space_counts(self._views(True), TEMPLATE)
        self.assertEqual(c, 0)
        self.assertEqual(s, 2 * len(TEMPLATE))
        s, c = free_space_counts(self._views(False), TEMPLATE)
        self.assertEqual((s, c), (len(TEMPLATE), len(TEMPLATE)))

    def test_reacquire_rejects_pose_another_camera_sees_through(self):
        from tracking.reacquire import reacquire
        eye = ObjectPose(R=np.eye(3), t=np.zeros(3), source="measured", mode="green")
        for wrist_sees, accepted in ((True, True), (False, False)):
            views = self._views(wrist_sees)
            banks = {"head": capture_bank(views["head"], eye, TEMPLATE, 0, half=4, angles=(0,))}
            R, _t, info = reacquire(views, banks, TEMPLATE, [("predicted", eye, 20)])
            self.assertEqual(R is not None, accepted, info)
            if not accepted:
                self.assertIn("free space", info["why"])


class TestInjectPose(unittest.TestCase):
    def test_inject_reinitialises_filter_and_next_fit_is_tentative(self):
        tracker = PoseTracker3D()
        for k in range(4):
            pts = posed(shift=(0.002 * k, 0.0, 0.0))
            views = views_with(pts)
            tracker.lift("obj", views, tracks_for(views, pts), seed_world_points=TEMPLATE)
        shift = np.array([0.1, 0.0, 0.0])
        self.assertTrue(tracker.inject_pose("obj", np.eye(3), shift, pos_std=0.005, reason="test"))
        st = tracker._objects["obj"]
        np.testing.assert_allclose(st.kf.position, CENTROID + shift, atol=1e-9)
        self.assertEqual(st.unmeasured_frames, 0)
        pts = posed(shift=shift)
        views = views_with(pts)
        res = tracker.lift("obj", views, tracks_for(views, pts), seed_world_points=TEMPLATE)
        self.assertTrue(res.meta.get("tentative"))
        self.assertNotEqual(res.meta["mode"], "green")
        self.assertFalse(tracker.inject_pose("unknown", np.eye(3), shift))


class TestCoherenceTwist(unittest.TestCase):
    def test_noisy_rotation_on_a_translation_does_not_bend_the_path(self):
        rng = np.random.default_rng(4)
        cv, tw = RigidBodyKF(KFConfig(motion_model="cv")), RigidBodyKF(KFConfig(motion_model="twist"))
        v = np.array([0.01, 0.0, 0.0])
        for k in range(25):
            R = rot_z(float(rng.normal(0.0, 1.0)))       # incoherent 1 deg rotation noise
            for kf in (cv, tw):
                kf.predict()
                kf.update(v * k, R)
        gap = np.linalg.norm(cv.predicted_pose(10)[0] - tw.predicted_pose(10)[0])
        self.assertLess(gap, 0.01)


class TestCoastDamping(unittest.TestCase):
    def _coast(self, tau, frames=40):
        kf = RigidBodyKF(KFConfig(motion_model="cv", coast_hold=10.0, coast_tau=tau,
                                  max_coast=100))
        v = np.array([0.02, 0.0, 0.0])
        for k in range(10):
            kf.predict()
            kf.update(v * k, np.eye(3))
        start = kf.position
        for _ in range(frames):
            kf.predict()
        return float(np.linalg.norm(kf.position - start))

    def test_short_coast_keeps_full_velocity(self):
        self.assertAlmostEqual(self._coast(10.0, frames=10), self._coast(0.0, frames=10), places=9)

    def test_long_coast_drift_is_bounded(self):
        undamped, damped = self._coast(0.0), self._coast(10.0)
        self.assertGreater(undamped, 0.75)                  # 40 frames x 2 cm
        self.assertLess(damped, 0.2 + 0.2 + 0.02)           # hold (10 x 2 cm) + |v| * tau


if __name__ == "__main__":
    unittest.main()
