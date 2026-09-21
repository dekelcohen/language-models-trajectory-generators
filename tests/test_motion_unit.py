"""Unit tests for :mod:`tracking.motion` - multi-point tracks -> motion semantics.

**Pure math: no simulator, no GPU, no tracker.** Every case is a synthetic motion with a
closed-form answer, so "the hinge axis is right" is checked against the axis that was used
to *generate* the data rather than against a previous run's output.

The three tests that matter most:
  * ``test_reflection_case_returns_a_proper_rotation`` - the ``det(R) < 0`` Kabsch trap.
  * ``test_axis_point_is_recovered_exactly`` - a hinge *location*, which a centroid tracker
    fundamentally cannot produce.
  * ``TestWhyScrewBeatsCentroidVelocity`` - numerical proof that frame-to-frame centroid
    velocity gives the *wrong* answer for a rotating object, i.e. the justification for
    this whole module.

Run with::

    python -m pytest tests/test_motion_unit.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking import motion as mo  # noqa: E402

RNG = np.random.default_rng(7)

#: A well-conditioned, non-collinear, non-planar blob ~10 cm across.
CLOUD = np.array([
    [0.00, 0.00, 0.00],
    [0.10, 0.01, 0.02],
    [0.02, 0.09, -0.01],
    [0.03, 0.02, 0.08],
    [0.08, 0.07, 0.05],
    [-0.04, 0.05, 0.03],
], dtype=float) + np.array([0.35, 0.10, 0.25])


def rodrigues(axis, theta):
    """Rotation matrix for ``theta`` radians right-handed about unit ``axis``."""
    n = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def screw_transform(axis, theta, axis_point, pitch=0.0):
    """``R, t`` of a rotation about the line ``(axis_point, axis)`` plus ``pitch`` along it."""
    n = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    q = np.asarray(axis_point, dtype=float)
    R = rodrigues(n, theta)
    return R, q - R @ q + pitch * n


def angle_between(a, b):
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def distance_to_line(point, line_point, line_dir):
    d = np.asarray(line_dir, dtype=float) / np.linalg.norm(line_dir)
    v = np.asarray(point, dtype=float) - np.asarray(line_point, dtype=float)
    return float(np.linalg.norm(v - (v @ d) * d))


def buffer_from_frames(frames, masks=None, indices=None, capacity=None):
    buf = mo.MotionBuffer(capacity=capacity or max(2, len(frames)))
    for i, pts in enumerate(frames):
        buf.push(pts,
                 mask=None if masks is None else masks[i],
                 point_index=None if indices is None else indices[i],
                 frame_idx=i)
    return buf


def rotating_frames(axis, axis_point, step_rad, n_frames, cloud=CLOUD, pitch=0.0):
    """``n_frames`` frames of ``cloud`` rotating ``step_rad`` per frame about the axis."""
    out = []
    for i in range(n_frames):
        R, t = screw_transform(axis, step_rad * i, axis_point, pitch * i)
        out.append(cloud @ R.T + t)
    return out


def sliding_frames(direction, step_m, n_frames, cloud=CLOUD):
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return [cloud + d * step_m * i for i in range(n_frames)]


class TestMotionBuffer(unittest.TestCase):

    def test_window_shape_and_order(self):
        buf = buffer_from_frames(sliding_frames([1, 0, 0], 0.01, 4))
        pts, mask, frames = buf.window()
        self.assertEqual(pts.shape, (4, len(CLOUD), 3))
        self.assertEqual(mask.shape, (4, len(CLOUD)))
        np.testing.assert_allclose(frames, [0, 1, 2, 3])
        # Oldest first: frame 0 is the unmoved cloud.
        np.testing.assert_allclose(pts[0], CLOUD, atol=1e-12)

    def test_ring_overwrites_oldest(self):
        buf = mo.MotionBuffer(capacity=3)
        for i in range(5):
            buf.push(CLOUD + i * 0.01, frame_idx=i)
        self.assertEqual(len(buf), 3)
        _pts, _mask, frames = buf.window()
        np.testing.assert_allclose(frames, [2, 3, 4])

    def test_points_appearing_and_disappearing(self):
        """A point that vanishes for a frame must come back to its own slot."""
        buf = mo.MotionBuffer(capacity=4)
        buf.push(CLOUD[:4], point_index=[0, 1, 2, 3], frame_idx=0)
        buf.push(CLOUD[[0, 2, 3]], point_index=[0, 2, 3], frame_idx=1)       # slot 1 gone
        buf.push(CLOUD[[0, 1, 2, 3, 4]], point_index=[0, 1, 2, 3, 4], frame_idx=2)  # slot 4 new
        _pts, mask, _frames = buf.window()
        self.assertEqual(buf.n_points, 5)
        np.testing.assert_array_equal(mask[1], [True, False, True, True, False])
        np.testing.assert_array_equal(buf.common_valid(), [True, False, True, True, False])

    def test_centroid_fallback_index_is_ignored(self):
        """``point_index == -1`` means "no correspondence" and must not occupy a slot."""
        buf = mo.MotionBuffer(capacity=2)
        buf.push(CLOUD[:3], point_index=[0, -1, 2], frame_idx=0)
        _pts, mask, _frames = buf.window()
        np.testing.assert_array_equal(mask[0][:3], [True, False, True])

    def test_invalid_mask_and_nan_rows_are_dropped(self):
        pts = CLOUD[:3].copy()
        pts[2] = np.nan
        buf = mo.MotionBuffer(capacity=2)
        buf.push(pts, mask=[True, False, True], frame_idx=0)
        _p, mask, _f = buf.window()
        np.testing.assert_array_equal(mask[0], [True, False, False])

    def test_centroids_skip_empty_frames(self):
        buf = mo.MotionBuffer(capacity=4)
        buf.push(CLOUD, frame_idx=0)
        buf.push(np.zeros((0, 3)), frame_idx=1)
        buf.push(CLOUD + 0.02, frame_idx=2)
        cents, frames, counts = buf.centroids()
        self.assertEqual(cents.shape, (2, 3))
        np.testing.assert_allclose(frames, [0, 2])
        np.testing.assert_array_equal(counts, [len(CLOUD), len(CLOUD)])


class TestFitRigidMotion(unittest.TestCase):

    def test_exact_recovery_of_rotation_and_translation(self):
        R, t = screw_transform([0.3, -0.5, 0.8], 0.37, [0.4, -0.2, 0.9], pitch=0.03)
        fit = mo.fit_rigid_motion(CLOUD, CLOUD @ R.T + t)
        np.testing.assert_allclose(fit.R, R, atol=1e-9)
        np.testing.assert_allclose(fit.t, t, atol=1e-9)
        self.assertLess(fit.rmse, 1e-9)
        self.assertEqual(fit.n_used, len(CLOUD))

    def test_rotation_is_always_special_orthogonal(self):
        R, t = screw_transform([0, 0, 1], 0.8, [0.1, 0.1, 0.0])
        fit = mo.fit_rigid_motion(CLOUD, CLOUD @ R.T + t)
        np.testing.assert_allclose(fit.R @ fit.R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(fit.R)), 1.0, places=9)

    def test_reflection_case_returns_a_proper_rotation(self):
        """The classic ``det(R) < 0`` bug: mirrored data must not yield a reflection."""
        mirror = np.diag([1.0, 1.0, -1.0])
        P1 = CLOUD @ mirror.T

        # Prove the naive (uncorrected) solution really is a reflection here, so the test
        # exercises the correction branch rather than passing vacuously.
        A = CLOUD - CLOUD.mean(axis=0)
        B = P1 - P1.mean(axis=0)
        U, _S, Vt = np.linalg.svd(A.T @ B)
        naive = Vt.T @ U.T
        self.assertLess(float(np.linalg.det(naive)), 0.0)

        fit = mo.fit_rigid_motion(CLOUD, P1)
        self.assertAlmostEqual(float(np.linalg.det(fit.R)), 1.0, places=9)
        np.testing.assert_allclose(fit.R @ fit.R.T, np.eye(3), atol=1e-9)
        self.assertGreater(fit.rmse, 1e-3, "a mirror is not reachable by a rotation")

    def test_planar_points_still_recover_their_rotation(self):
        """Coplanar points make the SVD rank-deficient - the other face of the det bug."""
        planar = np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.0, 0.12, 0.2],
                           [0.09, 0.11, 0.2], [0.04, 0.03, 0.2]])
        R, t = screw_transform([0, 0, 1], 0.42, [0.05, 0.05, 0.2])
        fit = mo.fit_rigid_motion(planar, planar @ R.T + t)
        np.testing.assert_allclose(fit.R, R, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(fit.R)), 1.0, places=9)
        self.assertLess(fit.rmse, 1e-9)

    def test_outlier_is_rejected(self):
        R, t = screw_transform([0.1, 0.2, 0.97], 0.25, [0.3, 0.1, 0.2])
        P1 = CLOUD @ R.T + t
        P1[3] += np.array([0.25, -0.18, 0.11])     # one blown correspondence
        fit = mo.fit_rigid_motion(CLOUD, P1)
        self.assertFalse(bool(fit.inliers[3]), "the gross outlier should be dropped")
        np.testing.assert_allclose(fit.R, R, atol=1e-9)
        self.assertLess(fit.rmse, 1e-9)

    def test_exact_data_keeps_every_point(self):
        """Residuals of ~1e-16 must not look like outliers to the MAD threshold."""
        R, t = screw_transform([0.0, 1.0, 0.0], 0.15, [0.2, 0.0, 0.3])
        fit = mo.fit_rigid_motion(CLOUD, CLOUD @ R.T + t)
        self.assertEqual(fit.n_used, len(CLOUD))

    def test_too_few_points_returns_infinite_rmse_without_raising(self):
        fit = mo.fit_rigid_motion(CLOUD[:2], CLOUD[:2] + 0.01)
        self.assertFalse(np.isfinite(fit.rmse))
        self.assertEqual(fit.n_used, 2)


class TestScrewDecompose(unittest.TestCase):

    AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    POINT = np.array([0.42, -0.17, 0.93])

    def test_angle_and_axis_direction_are_exact(self):
        R, t = screw_transform(self.AXIS, 0.31, self.POINT)
        screw = mo.screw_decompose(R, t)
        self.assertAlmostEqual(screw.angle_rad, 0.31, places=9)
        np.testing.assert_allclose(screw.axis_dir, self.AXIS, atol=1e-9)

    def test_axis_point_is_recovered_exactly(self):
        """A point on the hinge line - the thing a centroid tracker cannot give you."""
        R, t = screw_transform(self.AXIS, 0.31, self.POINT)
        screw = mo.screw_decompose(R, t, ref_point=CLOUD.mean(axis=0))
        self.assertLess(distance_to_line(screw.axis_point, self.POINT, self.AXIS), 1e-9)
        self.assertLess(screw.residual, 1e-9)

    def test_axis_point_without_reference_is_closest_to_the_origin(self):
        R, t = screw_transform(self.AXIS, 0.31, self.POINT)
        screw = mo.screw_decompose(R, t)
        self.assertLess(distance_to_line(screw.axis_point, self.POINT, self.AXIS), 1e-9)
        # Minimum-norm lstsq solution => orthogonal to the axis direction.
        self.assertAlmostEqual(float(screw.axis_point @ self.AXIS), 0.0, places=9)

    def test_reference_point_selects_the_nearest_point_on_the_axis(self):
        R, t = screw_transform(self.AXIS, 0.31, self.POINT)
        ref = CLOUD.mean(axis=0)
        screw = mo.screw_decompose(R, t, ref_point=ref)
        far = mo.screw_decompose(R, t).axis_point
        self.assertLess(float(np.linalg.norm(screw.axis_point - ref)),
                        float(np.linalg.norm(far - ref)))
        # And it is the perpendicular foot of ``ref`` on the line.
        self.assertAlmostEqual(float((ref - screw.axis_point) @ self.AXIS), 0.0, places=9)

    def test_screw_pitch_is_separated_from_the_rotation(self):
        R, t = screw_transform(self.AXIS, 0.31, self.POINT, pitch=0.07)
        screw = mo.screw_decompose(R, t, ref_point=CLOUD.mean(axis=0))
        self.assertAlmostEqual(screw.axis_translation, 0.07, places=9)
        self.assertLess(distance_to_line(screw.axis_point, self.POINT, self.AXIS), 1e-9)

    def test_pure_translation_has_no_axis(self):
        t = np.array([0.03, -0.01, 0.02])
        screw = mo.screw_decompose(np.eye(3), t, ref_point=CLOUD.mean(axis=0))
        self.assertIsNone(screw.axis_dir)
        self.assertIsNone(screw.axis_point)
        self.assertEqual(screw.angle_rad, 0.0)
        self.assertAlmostEqual(screw.axis_translation, float(np.linalg.norm(t)), places=12)

    def test_near_zero_rotation_has_no_axis(self):
        R, t = screw_transform(self.AXIS, 1e-12, self.POINT)
        self.assertIsNone(mo.screw_decompose(R, t).axis_dir)

    def test_half_turn_axis(self):
        """theta = pi kills the skew part of R; the eigenvector branch must take over."""
        R, t = screw_transform(self.AXIS, np.pi, self.POINT)
        screw = mo.screw_decompose(R, t, ref_point=self.POINT)
        self.assertAlmostEqual(screw.angle_rad, np.pi, places=9)
        self.assertAlmostEqual(abs(float(screw.axis_dir @ self.AXIS)), 1.0, places=7)
        self.assertLess(distance_to_line(screw.axis_point, self.POINT, self.AXIS), 1e-7)

    def test_round_trip_through_fit_and_decompose(self):
        R, t = screw_transform(self.AXIS, 0.22, self.POINT, pitch=0.01)
        fit = mo.fit_rigid_motion(CLOUD, CLOUD @ R.T + t)
        screw = mo.screw_decompose(fit.R, fit.t, ref_point=CLOUD.mean(axis=0))
        self.assertAlmostEqual(screw.angle_rad, 0.22, places=8)
        self.assertLess(angle_between(screw.axis_dir, self.AXIS), 1e-7)
        self.assertLess(distance_to_line(screw.axis_point, self.POINT, self.AXIS), 1e-7)


class TestPlaneFit(unittest.TestCase):

    def test_normal_and_residual_of_an_exact_plane(self):
        normal = np.array([0.0, 0.0, 1.0])
        pts = np.column_stack([RNG.uniform(-0.1, 0.1, 8), RNG.uniform(-0.1, 0.1, 8),
                               np.full(8, 0.3)])
        plane = mo.fit_plane(pts, view_point=[0.0, 0.0, 2.0])
        np.testing.assert_allclose(np.abs(plane.normal), normal, atol=1e-9)
        self.assertLess(plane.residual, 1e-12)
        self.assertGreater(plane.confidence, 0.99)

    def test_normal_points_towards_the_camera(self):
        pts = np.column_stack([RNG.uniform(-0.1, 0.1, 8), RNG.uniform(-0.1, 0.1, 8),
                               np.full(8, 0.3)])
        above = mo.fit_plane(pts, view_point=[0.0, 0.0, 2.0])
        below = mo.fit_plane(pts, view_point=[0.0, 0.0, -2.0])
        self.assertGreater(float(above.normal[2]), 0.0)
        self.assertLess(float(below.normal[2]), 0.0)

    def test_view_direction_flips_the_sign_the_other_way(self):
        pts = np.column_stack([RNG.uniform(-0.1, 0.1, 8), RNG.uniform(-0.1, 0.1, 8),
                               np.full(8, 0.3)])
        # Camera looking down -Z sees a surface whose normal points up +Z.
        plane = mo.fit_plane(pts, view_dir=[0.0, 0.0, -1.0])
        self.assertGreater(float(plane.normal[2]), 0.0)

    def test_tilted_plane(self):
        truth = np.array([0.4, -0.2, 1.0])
        truth /= np.linalg.norm(truth)
        basis = np.linalg.svd(truth.reshape(1, 3))[2][1:]
        uv = RNG.uniform(-0.1, 0.1, (10, 2))
        pts = np.array([0.2, 0.1, 0.3]) + uv @ basis
        plane = mo.fit_plane(pts, view_point=[0.2, 0.1, 3.0])
        self.assertLess(angle_between(plane.normal, truth), 1e-7)

    def test_collinear_points_give_no_normal(self):
        line = np.linspace(0, 0.2, 6)[:, None] * np.array([1.0, 0.5, -0.3])
        plane = mo.fit_plane(line, view_point=[0.0, 0.0, 2.0])
        self.assertIsNone(plane.normal)
        self.assertEqual(plane.confidence, 0.0)

    def test_two_points_do_not_raise(self):
        plane = mo.fit_plane(CLOUD[:2], view_point=[0.0, 0.0, 2.0])
        self.assertIsNone(plane.normal)
        self.assertEqual(plane.n_points, 2)


class TestCentroidDirection(unittest.TestCase):

    def test_direction_and_speed_of_a_straight_slide(self):
        d = np.array([0.0, 1.0, 0.0])
        buf = buffer_from_frames(sliding_frames(d, 0.01, 6))
        line = mo.centroid_direction(buf, dt=0.1)
        np.testing.assert_allclose(line.direction, d, atol=1e-9)
        self.assertAlmostEqual(line.speed, 0.1, places=9)     # 0.01 m per 0.1 s
        self.assertAlmostEqual(line.travel, 0.05, places=9)
        self.assertLess(line.residual, 1e-12)
        self.assertGreater(line.confidence, 0.9)

    def test_direction_sign_follows_travel(self):
        buf = buffer_from_frames(sliding_frames([-1, 0, 0], 0.01, 5))
        line = mo.centroid_direction(buf)
        np.testing.assert_allclose(line.direction, [-1, 0, 0], atol=1e-9)
        self.assertGreater(line.speed, 0.0)

    def test_standing_still_yields_no_direction(self):
        buf = buffer_from_frames([CLOUD + RNG.normal(0, 2e-5, CLOUD.shape) for _ in range(6)])
        line = mo.centroid_direction(buf)
        self.assertIsNone(line.direction)
        self.assertEqual(line.confidence, 0.0)

    def test_line_fit_beats_frame_to_frame_differencing_under_noise(self):
        """The docstring's claim, measured: TLS over K frames vs a single difference."""
        d = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(3)
        tls_err, diff_err = [], []
        for _ in range(40):
            frames = [f + rng.normal(0, 1.5e-3, f.shape) for f in sliding_frames(d, 0.004, 10)]
            buf = buffer_from_frames(frames)
            tls_err.append(angle_between(mo.centroid_direction(buf).direction, d))
            cents, _f, _c = buf.centroids()
            diff_err.append(angle_between(cents[-1] - cents[-2], d))
        self.assertLess(float(np.mean(tls_err)), 0.5 * float(np.mean(diff_err)))

    def test_accepts_a_raw_centroid_array(self):
        cents = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.0, 0.0, 0.02]])
        line = mo.centroid_direction(cents)
        np.testing.assert_allclose(line.direction, [0, 0, 1], atol=1e-9)


class TestClassification(unittest.TestCase):

    def test_prismatic_from_a_pure_slide(self):
        buf = buffer_from_frames(sliding_frames([0, 1, 0], 0.01, 6))
        est = mo.estimate_motion(buf)
        self.assertEqual(est.kind, mo.PRISMATIC)
        np.testing.assert_allclose(est.direction, [0, 1, 0], atol=1e-9)
        self.assertIsNone(est.axis_dir)
        self.assertGreater(est.confidence, 0.5)

    def test_revolute_from_a_hinge(self):
        axis = np.array([0.0, 0.0, 1.0])
        q = np.array([0.30, -0.15, 0.25])
        buf = buffer_from_frames(rotating_frames(axis, q, 0.05, 6))
        est = mo.estimate_motion(buf)
        self.assertEqual(est.kind, mo.REVOLUTE)
        self.assertLess(angle_between(est.axis_dir, axis), 1e-7)
        self.assertLess(distance_to_line(est.axis_point, q, axis), 1e-7)
        self.assertAlmostEqual(est.angle_rad, 0.25, places=7)
        self.assertGreater(est.confidence, 0.5)

    def test_static_object(self):
        buf = buffer_from_frames([CLOUD.copy() for _ in range(6)])
        est = mo.estimate_motion(buf)
        self.assertEqual(est.kind, mo.STATIC)
        self.assertIsNone(est.direction)
        self.assertIsNone(est.axis_dir)
        self.assertEqual(est.speed, 0.0)

    def test_non_rigid_scatter_is_free(self):
        cloud = np.vstack([CLOUD, CLOUD + 0.05])
        rng = np.random.default_rng(11)
        frames = [cloud + rng.normal(0, 0.05, cloud.shape) for _ in range(4)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertEqual(est.kind, mo.FREE)

    def test_stateless_classifier_flips_on_borderline_input(self):
        """Baseline for the hysteresis test below: without state, the label chatters."""
        th = mo.DEFAULT_THRESHOLDS
        labels = []
        for i in range(10):
            angle = th.angle_enter_rad * (1.05 if i % 2 else 0.95)
            labels.append(mo.classify_motion(angle, 0.02, 1e-6, 0.05, motion_scale=0.02))
        self.assertGreater(len(set(labels)), 1)
        flips = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        self.assertGreaterEqual(flips, 5)

    def test_hysteresis_stops_the_label_flickering(self):
        """Once a label is established, borderline frames must not knock it off."""
        th = mo.DEFAULT_THRESHOLDS
        clf = mo.MotionClassifier()
        for _ in range(3):
            clf.update(0.2, 0.02, 1e-6, 0.05, motion_scale=0.02)
        self.assertEqual(clf.label, mo.REVOLUTE)

        labels = []
        for i in range(10):
            angle = th.angle_enter_rad * (1.05 if i % 2 else 0.95)
            labels.append(clf.update(angle, 0.02, 1e-6, 0.05, motion_scale=0.02))
        flips = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        self.assertEqual(flips, 0, f"label chattered: {labels}")
        self.assertEqual(labels[-1], mo.REVOLUTE)

    def test_borderline_input_never_wins_the_dwell(self):
        """From a cold start, input that cannot make up its mind adopts nothing."""
        th = mo.DEFAULT_THRESHOLDS
        clf = mo.MotionClassifier()
        labels = [clf.update(th.angle_enter_rad * (1.05 if i % 2 else 0.95),
                             0.02, 1e-6, 0.05, motion_scale=0.02) for i in range(10)]
        self.assertEqual(len(set(labels)), 1, f"label chattered: {labels}")

    def test_hysteresis_still_follows_a_real_change(self):
        clf = mo.MotionClassifier()
        for _ in range(4):
            clf.update(0.0, 0.03, 1e-6, 1e-6, motion_scale=0.03)
        self.assertEqual(clf.label, mo.PRISMATIC)
        for _ in range(4):
            clf.update(0.4, 0.03, 1e-6, 0.06, motion_scale=0.06)
        self.assertEqual(clf.label, mo.REVOLUTE)
        self.assertGreaterEqual(clf.transitions, 2)

    def test_transition_is_logged(self):
        class Spy:
            def __init__(self):
                self.lines = []

            def info(self, msg):
                self.lines.append(msg)

        spy = Spy()
        clf = mo.MotionClassifier(logger=spy)
        for _ in range(4):
            clf.update(0.0, 0.03, 1e-6, 1e-6, motion_scale=0.03)
        self.assertTrue(any("static -> prismatic" in line for line in spy.lines), spy.lines)


class TestConfidenceGates(unittest.TestCase):

    def test_fewer_than_three_points_gives_zero_confidence(self):
        frames = [CLOUD[:2] + i * np.array([0.01, 0.0, 0.0]) for i in range(5)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "too_few_points")
        self.assertIsNone(est.direction)
        self.assertIsNone(est.axis_dir)

    def test_collinear_points_give_zero_confidence(self):
        line = np.linspace(0.0, 0.2, 5)[:, None] * np.array([1.0, 0.5, -0.3])
        frames = [line + i * np.array([0.0, 0.01, 0.0]) for i in range(5)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "collinear_points")
        self.assertIsNone(est.direction)

    def test_collinearity_is_detected_from_the_singular_spectrum(self):
        line = np.linspace(0.0, 0.2, 5)[:, None] * np.array([1.0, 0.5, -0.3])
        self.assertTrue(mo.is_collinear(line))
        self.assertFalse(mo.is_collinear(CLOUD))
        s = mo.point_spread(line)
        self.assertGreater(s[0], 0.1)
        self.assertLess(s[1], 1e-12)

    def test_single_frame_does_not_raise(self):
        est = mo.estimate_motion(buffer_from_frames([CLOUD]))
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "insufficient_frames")

    def test_empty_buffer_does_not_raise(self):
        est = mo.estimate_motion(mo.MotionBuffer())
        self.assertEqual(est.confidence, 0.0)
        self.assertIsNone(est.direction)

    def test_near_zero_motion_never_returns_a_unit_vector(self):
        rng = np.random.default_rng(5)
        frames = [CLOUD + rng.normal(0, 1e-4, CLOUD.shape) for _ in range(6)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertIsNone(est.direction, "a direction from jitter is pure noise")
        self.assertEqual(est.kind, mo.STATIC)

    def test_disjoint_correspondence_is_rejected(self):
        """No point survives both ends of the window -> nothing can be fitted."""
        buf = mo.MotionBuffer(capacity=4)
        buf.push(CLOUD[:3], point_index=[0, 1, 2], frame_idx=0)
        buf.push(CLOUD[3:6], point_index=[3, 4, 5], frame_idx=1)
        est = mo.estimate_motion(buf)
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "no_correspondence")

    def test_low_confidence_is_logged_once_per_reason(self):
        class Spy:
            def __init__(self):
                self.lines = []

            def info(self, msg):
                self.lines.append(msg)

        spy = Spy()
        clf = mo.MotionClassifier(logger=spy)
        line = np.linspace(0.0, 0.2, 5)[:, None] * np.array([1.0, 0.5, -0.3])
        for i in range(5):
            buf = buffer_from_frames([line, line + np.array([0.0, 0.01, 0.0]) * (i + 1)])
            mo.estimate_motion(buf, classifier=clf)
        hits = [ln for ln in spy.lines if "collinear_points" in ln]
        self.assertEqual(len(hits), 1, f"rejection logging should not spam: {spy.lines}")


class TestNoiseDegradation(unittest.TestCase):
    """Noise must degrade the estimate, not turn it into garbage."""

    def test_hinge_survives_millimetre_noise(self):
        axis = np.array([0.1, -0.2, 0.97])
        axis = axis / np.linalg.norm(axis)
        q = np.array([0.30, -0.15, 0.20])
        rng = np.random.default_rng(17)
        errs, offsets = [], []
        for _ in range(20):
            frames = [f + rng.normal(0, 1e-3, f.shape)
                      for f in rotating_frames(axis, q, 0.08, 8)]
            est = mo.estimate_motion(buffer_from_frames(frames))
            self.assertEqual(est.kind, mo.REVOLUTE)
            self.assertGreater(est.confidence, 0.0)
            errs.append(angle_between(est.axis_dir, axis))
            offsets.append(distance_to_line(est.axis_point, q, axis))
        self.assertLess(float(np.median(errs)), np.radians(10.0))
        self.assertLess(float(np.median(offsets)), 0.03)

    def test_slide_direction_survives_millimetre_noise(self):
        d = np.array([0.0, 1.0, 0.0])
        rng = np.random.default_rng(19)
        errs = []
        for _ in range(20):
            frames = [f + rng.normal(0, 1e-3, f.shape) for f in sliding_frames(d, 0.01, 8)]
            est = mo.estimate_motion(buffer_from_frames(frames))
            self.assertEqual(est.kind, mo.PRISMATIC)
            errs.append(angle_between(est.direction, d))
        self.assertLess(float(np.median(errs)), np.radians(5.0))

    def test_confidence_falls_as_noise_rises(self):
        d = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(23)

        def mean_conf(sigma):
            vals = []
            for _ in range(15):
                frames = [f + rng.normal(0, sigma, f.shape) for f in sliding_frames(d, 0.004, 8)]
                vals.append(mo.estimate_motion(buffer_from_frames(frames)).confidence)
            return float(np.mean(vals))

        self.assertGreater(mean_conf(2e-4), mean_conf(4e-3))

    def test_residual_reports_the_noise_level(self):
        rng = np.random.default_rng(29)
        frames = [f + rng.normal(0, 2e-3, f.shape) for f in sliding_frames([1, 0, 0], 0.01, 6)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertGreater(est.residual, 5e-4)
        self.assertLess(est.residual, 2e-2)


class TestWhyScrewBeatsCentroidVelocity(unittest.TestCase):
    """The justification for the whole module, measured numerically.

    For a rotating object the centroid's frame-to-frame velocity is the **chord** of an
    arc, and it is the same vector for every point of the body. The true instantaneous
    velocity of a point is ``omega_hat x (p - p_axis)``, which differs in direction (by
    half the swept angle at the centroid) and, crucially, differs from point to point -
    so "the handle moves that way" cannot be answered by a centroid at all.
    """

    AXIS = np.array([0.0, 0.0, 1.0])
    Q = np.array([0.30, -0.15, 0.25])
    STEP = 0.12

    def _frames(self, n=6):
        return rotating_frames(self.AXIS, self.Q, self.STEP, n)

    def test_centroid_velocity_direction_differs_from_the_true_tangent(self):
        frames = self._frames()
        buf = buffer_from_frames(frames)
        est = mo.estimate_motion(buf)

        c_last = frames[-1].mean(axis=0)
        radius = c_last - self.Q
        radius -= float(radius @ self.AXIS) * self.AXIS
        truth_tangent = np.cross(self.AXIS, radius)
        truth_tangent /= np.linalg.norm(truth_tangent)

        # The screw answer is the true tangent, to machine precision.
        self.assertEqual(est.kind, mo.REVOLUTE)
        self.assertLess(angle_between(est.direction, truth_tangent), 1e-7)

        # The naive answer (last centroid step) is measurably different: for a chord
        # spanning ``STEP`` radians the error is half the swept angle.
        cents, _f, _c = buf.centroids()
        naive = cents[-1] - cents[-2]
        naive_err = angle_between(naive, truth_tangent)
        self.assertAlmostEqual(naive_err, self.STEP / 2.0, places=6)
        self.assertGreater(naive_err, np.radians(3.0))

    def test_centroid_velocity_is_wrong_for_points_away_from_the_centroid(self):
        """Every point of a rotating body has a different velocity; a centroid has one."""
        frames = self._frames()
        buf = buffer_from_frames(frames)
        est = mo.estimate_motion(buf)

        cents, _f, _c = buf.centroids()
        naive_step = cents[-1] - cents[-2]

        R_step, t_step = screw_transform(self.AXIS, self.STEP, self.Q)
        truth_next = frames[-1] @ R_step.T + t_step

        omega = est.angle_rad / (len(frames) - 1)
        R_pred, t_pred = screw_transform(est.axis_dir, omega, est.axis_point)
        screw_next = frames[-1] @ R_pred.T + t_pred
        naive_next = frames[-1] + naive_step

        screw_err = float(np.max(np.linalg.norm(screw_next - truth_next, axis=1)))
        naive_err = float(np.max(np.linalg.norm(naive_next - truth_next, axis=1)))
        self.assertLess(screw_err, 1e-9)
        self.assertGreater(naive_err, 100.0 * max(screw_err, 1e-9))
        self.assertGreater(naive_err, 1e-3, "the naive prediction is off by millimetres")


class TestMotionEstimateSerialisation(unittest.TestCase):

    def test_to_dict_is_json_serialisable(self):
        import json

        axis = np.array([0.0, 0.0, 1.0])
        buf = buffer_from_frames(rotating_frames(axis, [0.3, -0.15, 0.25], 0.05, 6))
        est = mo.estimate_motion(buf, view_point=[0.0, 0.0, 2.0], dt=0.1)
        payload = json.loads(json.dumps(est.to_dict()))
        self.assertEqual(payload["kind"], mo.REVOLUTE)
        self.assertEqual(len(payload["axis_dir"]), 3)
        self.assertEqual(len(payload["direction"]), 3)
        self.assertIsInstance(payload["confidence"], float)
        self.assertEqual(payload["window_frames"], 6)
        self.assertEqual(payload["n_points"], len(CLOUD))

    def test_none_fields_survive_serialisation(self):
        import json

        est = mo.estimate_motion(mo.MotionBuffer())
        payload = json.loads(json.dumps(est.to_dict()))
        self.assertIsNone(payload["direction"])
        self.assertIsNone(payload["axis_point"])
        self.assertIsNone(payload["perpendicularity"])

    def test_component_fits_serialise(self):
        import json

        buf = buffer_from_frames(sliding_frames([0, 0, 1], 0.01, 5))
        est = mo.estimate_motion(buf, view_point=[0.0, 0.0, 2.0])
        json.dumps([est.plane.to_dict(), est.line.to_dict()])
        fit = mo.fit_rigid_motion(CLOUD, CLOUD + 0.01)
        json.dumps(fit.to_dict())


class TestDrawerSemantics(unittest.TestCase):
    """The drawer case end to end: slide direction + perpendicularity to the front face."""

    @staticmethod
    def _front_face(n=9):
        u = np.linspace(-0.08, 0.08, 3)
        grid = np.array([[a, 0.0, b] for a in u for b in u])[:n]
        return grid + np.array([0.35, 0.40, 0.25])

    def test_pull_straight_out_is_perpendicular_to_the_face(self):
        face = self._front_face()
        buf = buffer_from_frames(sliding_frames([0, -1, 0], 0.01, 6, cloud=face))
        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertEqual(est.kind, mo.PRISMATIC)
        np.testing.assert_allclose(est.direction, [0, -1, 0], atol=1e-9)
        self.assertAlmostEqual(est.perpendicularity, 1.0, places=9)
        self.assertGreater(float(est.plane_normal @ np.array([0.0, -1.0, 0.0])), 0.0)

    def test_sliding_along_the_face_is_not_perpendicular(self):
        face = self._front_face()
        buf = buffer_from_frames(sliding_frames([1, 0, 0], 0.01, 6, cloud=face))
        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertLess(est.perpendicularity, 1e-9)


class TestUnpackingApi(unittest.TestCase):

    def test_rigid_fit_unpacks_as_r_t_rmse(self):
        R_true, t_true = screw_transform([0, 0, 1], 0.2, [0.1, 0.1, 0.1])
        R, t, rmse = mo.fit_rigid_motion(CLOUD, CLOUD @ R_true.T + t_true)
        np.testing.assert_allclose(R, R_true, atol=1e-9)
        np.testing.assert_allclose(t, t_true, atol=1e-9)
        self.assertLess(rmse, 1e-9)


if __name__ == "__main__":
    unittest.main()
