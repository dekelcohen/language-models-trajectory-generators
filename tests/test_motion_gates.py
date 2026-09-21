"""The two headline outputs of :mod:`tracking.motion`, and the gates that protect them.

Companion to ``tests/test_motion_unit.py`` (which proves the math). This file proves the
*contract* a planner relies on:

  * **Door handle** - the pull direction is ``omega_hat x (p_handle - p_axis)`` evaluated
    **at the handle**, not at the centroid and not the centroid velocity, and the hinge
    line comes back so the arc can be predicted rather than chased.
  * **Drawer** - ``|d_hat . n_hat|`` (plus the angle) against an SVD plane normal, sign
    conventions included.
  * **Gates** - displacement floor, point count / collinearity, per-camera disagreement,
    sign-aligned EMA smoothing, and a confidence that actually tracks angular error.

Run with::

    python -m pytest tests/test_motion_gates.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking import motion as mo  # noqa: E402
from tests.test_motion_unit import (  # noqa: E402
    CLOUD,
    angle_between,
    buffer_from_frames,
    distance_to_line,
    rotating_frames,
    screw_transform,
    sliding_frames,
)


class Spy:
    """Minimal logger stand-in."""

    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


def door_frames(n=8, step=0.06, axis=(0.0, 0.0, 1.0), hinge=(0.0, 0.0, 0.0)):
    """A door panel: a flat slab hinged at ``hinge``, handle at the far edge (slot 0)."""
    axis = np.asarray(axis, dtype=float)
    hinge = np.asarray(hinge, dtype=float)
    # x = distance from the hinge, z = height; the panel lies in the x-z plane (y = 0).
    panel = np.array([[0.60, 0.0, 0.95],      # slot 0: the handle, far from the hinge
                      [0.60, 0.0, 0.60],
                      [0.60, 0.0, 1.30],
                      [0.30, 0.0, 0.95],
                      [0.10, 0.0, 0.60],
                      [0.10, 0.0, 1.30]]) + hinge
    frames = []
    for i in range(n):
        R, t = screw_transform(axis, step * i, hinge)
        frames.append(panel @ R.T + t)
    return frames, panel, axis / np.linalg.norm(axis), hinge


class TestDoorHandlePullDirection(unittest.TestCase):
    """``omega_hat x (p_handle - p_axis)`` - at the handle, which is not the centroid."""

    def test_pull_direction_at_the_handle_matches_the_analytic_tangent(self):
        frames, _panel, axis, hinge = door_frames()
        buf = buffer_from_frames(frames)
        est = mo.estimate_motion(buf, query_index=0)

        self.assertEqual(est.kind, mo.REVOLUTE)
        handle = frames[-1][0]
        radius = handle - hinge
        radius -= float(radius @ axis) * axis
        truth = np.cross(axis, radius)
        truth /= np.linalg.norm(truth)

        self.assertLess(angle_between(est.direction, truth), 1e-9)
        np.testing.assert_allclose(est.query_point, handle, atol=1e-9)

    def test_handle_direction_differs_from_the_centroid_direction(self):
        """The whole point of taking a query point: a door's handle is not its centroid."""
        frames, _panel, _axis, _hinge = door_frames()
        buf = buffer_from_frames(frames)
        at_handle = mo.estimate_motion(buf, query_index=0)
        at_centroid = mo.estimate_motion(buf)
        gap = angle_between(at_handle.direction, at_centroid.direction)
        self.assertGreater(gap, np.radians(5.0),
                           "handle and centroid tangents should be measurably different")

    def test_pull_direction_at_re_evaluates_anywhere_on_the_body(self):
        frames, _panel, axis, hinge = door_frames()
        est = mo.estimate_motion(buffer_from_frames(frames))
        for idx in range(len(frames[-1])):
            p = frames[-1][idx]
            radius = p - hinge
            radius -= float(radius @ axis) * axis
            truth = np.cross(axis, radius)
            truth /= np.linalg.norm(truth)
            self.assertLess(angle_between(est.pull_direction_at(p), truth), 1e-7)

    def test_pull_direction_is_not_the_centroid_velocity(self):
        """The naive answer (last centroid step) is wrong at the handle by a wide margin."""
        frames, _panel, axis, hinge = door_frames()
        buf = buffer_from_frames(frames)
        est = mo.estimate_motion(buf, query_index=0)

        handle = frames[-1][0]
        radius = handle - hinge
        radius -= float(radius @ axis) * axis
        truth = np.cross(axis, radius) / np.linalg.norm(np.cross(axis, radius))

        cents, _f, _c = buf.centroids()
        naive = cents[-1] - cents[-2]
        self.assertGreater(angle_between(naive, truth), np.radians(5.0))
        self.assertLess(angle_between(est.direction, truth), 1e-9)

    def test_hinge_axis_predicts_the_whole_arc(self):
        """A planner gets the *line*, so it can place the next pose instead of chasing."""
        frames, _panel, axis, hinge = door_frames(step=0.06)
        est = mo.estimate_motion(buffer_from_frames(frames), query_index=0)

        self.assertIsNotNone(est.hinge)
        reported_dir, reported_point = est.hinge
        self.assertLess(angle_between(reported_dir, axis), 1e-7)
        self.assertLess(distance_to_line(reported_point, hinge, axis), 1e-7)

        omega = est.angle_rad / (len(frames) - 1)
        R_pred, t_pred = screw_transform(reported_dir, omega, reported_point)
        predicted = frames[-1] @ R_pred.T + t_pred
        R_true, t_true = screw_transform(axis, 0.06, hinge)
        truth = frames[-1] @ R_true.T + t_true
        self.assertLess(float(np.max(np.linalg.norm(predicted - truth, axis=1))), 1e-9)

    def test_axis_point_is_reported_next_to_the_query_point(self):
        frames, _panel, axis, hinge = door_frames()
        est = mo.estimate_motion(buffer_from_frames(frames), query_index=0)
        handle = frames[-1][0]
        # The reported axis point is the foot of the perpendicular from the handle.
        self.assertAlmostEqual(float((handle - est.axis_point) @ axis), 0.0, places=7)

    def test_query_point_on_the_axis_has_no_pull_direction(self):
        axis = np.array([0.0, 0.0, 1.0])
        hinge = np.array([0.30, -0.15, 0.25])
        est = mo.estimate_motion(buffer_from_frames(rotating_frames(axis, hinge, 0.08, 6)))
        on_axis = hinge + np.array([0.0, 0.0, 0.4])
        self.assertIsNone(est.pull_direction_at(on_axis))
        self.assertIsNone(mo.pull_direction(None, hinge, on_axis))
        self.assertIsNone(mo.pull_direction(axis, hinge, None))

    def test_explicit_query_point_is_accepted(self):
        frames, _panel, axis, hinge = door_frames()
        handle = frames[-1][0]
        by_index = mo.estimate_motion(buffer_from_frames(frames), query_index=0)
        by_point = mo.estimate_motion(buffer_from_frames(frames), query_point=handle)
        np.testing.assert_allclose(by_point.direction, by_index.direction, atol=1e-12)

    def test_invalid_query_index_falls_back_to_the_centroid_and_says_so(self):
        frames, _panel, _axis, _hinge = door_frames()
        spy = Spy()
        est = mo.estimate_motion(buffer_from_frames(frames), query_index=99, logger=spy)
        self.assertIsNotNone(est.direction)
        self.assertTrue(any("query_index" in line for line in spy.lines), spy.lines)

    def test_a_slide_has_the_same_pull_direction_everywhere(self):
        buf = buffer_from_frames(sliding_frames([0, 1, 0], 0.01, 6))
        est = mo.estimate_motion(buf)
        for p in (CLOUD[0], CLOUD[3], CLOUD[0] + 5.0):
            np.testing.assert_allclose(est.pull_direction_at(p), [0, 1, 0], atol=1e-9)

    def test_a_rejected_estimate_never_yields_a_pull_direction(self):
        est = mo.estimate_motion(mo.MotionBuffer())
        self.assertEqual(est.confidence, 0.0)
        self.assertIsNone(est.pull_direction_at([0.0, 0.0, 0.0]))
        self.assertIsNone(est.hinge)


class TestDrawerPerpendicularity(unittest.TestCase):
    """``|d_hat . n_hat|`` against the SVD plane normal, plus the angle in degrees."""

    @staticmethod
    def _front_face():
        u = np.linspace(-0.08, 0.08, 3)
        grid = np.array([[a, 0.0, b] for a in u for b in u])
        return grid + np.array([0.35, 0.40, 0.25])

    def test_straight_out_reads_one_and_zero_degrees(self):
        buf = buffer_from_frames(sliding_frames([0, -1, 0], 0.02, 6, cloud=self._front_face()))
        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertAlmostEqual(est.perpendicularity, 1.0, places=9)
        self.assertAlmostEqual(est.perpendicularity_deg, 0.0, places=6)

    def test_sliding_along_the_face_reads_zero_and_ninety_degrees(self):
        buf = buffer_from_frames(sliding_frames([1, 0, 0], 0.02, 6, cloud=self._front_face()))
        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertLess(est.perpendicularity, 1e-9)
        self.assertAlmostEqual(est.perpendicularity_deg, 90.0, places=5)

    def test_a_skewed_pull_reads_the_angle_between(self):
        d = np.array([0.0, -np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
        buf = buffer_from_frames(sliding_frames(d, 0.02, 6, cloud=self._front_face()))
        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertAlmostEqual(est.perpendicularity_deg, 30.0, places=5)
        self.assertAlmostEqual(est.perpendicularity, np.cos(np.radians(30.0)), places=7)

    def test_the_normal_sign_convention_cannot_change_the_answer(self):
        """The normal's sign is a viewing convention; the abs() is what makes it safe."""
        frames = sliding_frames([0, -1, 0], 0.02, 6, cloud=self._front_face())
        front = mo.estimate_motion(buffer_from_frames(frames), view_point=[0.35, -2.0, 0.25])
        behind = mo.estimate_motion(buffer_from_frames(frames), view_point=[0.35, 3.0, 0.25])
        self.assertLess(float(front.plane_normal @ behind.plane_normal), 0.0)
        self.assertAlmostEqual(front.perpendicularity, behind.perpendicularity, places=12)

    def test_no_plane_means_no_perpendicularity_rather_than_a_guess(self):
        """A fat, non-planar blob still has a normal; a *line* must report nothing."""
        est = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6)))
        self.assertIsNotNone(est.direction)
        thin = mo.fit_plane(np.linspace(0, 0.2, 6)[:, None] * np.array([1.0, 0.5, -0.3]))
        self.assertIsNone(thin.normal)
        self.assertEqual(mo._perpendicularity([0, 1, 0], None), (None, None))

    def test_dropped_points_are_carried_forward_into_the_plane_fit(self):
        """Documented fallback: occluded points ride the rigid transform into the latest pose."""
        face = self._front_face()
        idx_all = np.arange(len(face))
        keep = idx_all[:5]
        frames = sliding_frames([0, -1, 0], 0.02, 4, cloud=face)
        buf = mo.MotionBuffer(capacity=4)
        buf.push(frames[0], point_index=idx_all, frame_idx=0)
        for i, f in enumerate(frames[1:], start=1):
            last = i == len(frames) - 1
            sel = keep if last else idx_all
            buf.push(f[sel], point_index=sel, frame_idx=i)

        est = mo.estimate_motion(buf, view_point=[0.35, -2.0, 0.25])
        self.assertEqual(est.plane.n_points, len(face),
                         "the four occluded points should have been carried forward")
        self.assertAlmostEqual(est.perpendicularity, 1.0, places=7)


class TestDisplacementGate(unittest.TestCase):
    """Gate 1: no direction at all below ``min_direction_m``."""

    def test_sub_centimetre_slide_emits_nothing(self):
        th = mo.DEFAULT_THRESHOLDS
        step = 0.6 * th.min_direction_m / 5.0          # 60% of the floor over the window
        buf = buffer_from_frames(sliding_frames([0, 1, 0], step, 6))
        est = mo.estimate_motion(buf)
        self.assertLess(est.displacement, th.min_direction_m)
        self.assertIsNone(est.direction)
        self.assertIsNone(est.axis_dir)
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "near_zero_motion")

    def test_a_slide_past_the_floor_does_emit(self):
        th = mo.DEFAULT_THRESHOLDS
        step = 1.6 * th.min_direction_m / 5.0
        est = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], step, 6)))
        self.assertGreater(est.displacement, th.min_direction_m)
        self.assertIsNotNone(est.direction)
        self.assertGreater(est.confidence, 0.0)

    def test_a_tiny_rotation_emits_no_axis(self):
        axis = np.array([0.0, 0.0, 1.0])
        hinge = np.array([0.30, -0.15, 0.25])
        buf = buffer_from_frames(rotating_frames(axis, hinge, 0.005, 6))
        est = mo.estimate_motion(buf)
        self.assertIsNone(est.axis_dir)
        self.assertIsNone(est.direction)
        self.assertEqual(est.confidence, 0.0)
        self.assertEqual(est.reason, "near_zero_motion")

    def test_the_floor_is_a_threshold_not_a_constant(self):
        th = mo.MotionThresholds(min_direction_m=0.001)
        step = 0.6 * mo.DEFAULT_THRESHOLDS.min_direction_m / 5.0
        buf = buffer_from_frames(sliding_frames([0, 1, 0], step, 6))
        self.assertIsNotNone(mo.estimate_motion(buf, thresholds=th).direction)

    def test_jitter_alone_is_gated(self):
        rng = np.random.default_rng(5)
        frames = [CLOUD + rng.normal(0, 2e-3, CLOUD.shape) for _ in range(6)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertIsNone(est.direction)
        self.assertEqual(est.confidence, 0.0)


class TestPointCountAndCollinearityGates(unittest.TestCase):
    """Gate 2: at least three valid, non-collinear points."""

    def test_two_points_are_refused(self):
        frames = [CLOUD[:2] + i * np.array([0.02, 0.0, 0.0]) for i in range(5)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertEqual(est.reason, "too_few_points")
        self.assertEqual(est.confidence, 0.0)
        self.assertIsNone(est.direction)

    def test_three_non_collinear_points_are_enough(self):
        tri = np.array([[0.0, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.10, 0.03]]) + 0.3
        est = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6,
                                                                   cloud=tri)))
        self.assertGreater(est.confidence, 0.0)
        self.assertIsNotNone(est.direction)

    def test_collinearity_comes_from_the_singular_spectrum(self):
        """Nearly - not exactly - collinear points must be caught too."""
        line = np.linspace(0.0, 0.2, 6)[:, None] * np.array([1.0, 0.5, -0.3])
        rng = np.random.default_rng(31)
        nearly = line + rng.normal(0, 1e-4, line.shape)
        s = mo.point_spread(nearly)
        self.assertLess(s[1] / s[0], mo.DEFAULT_THRESHOLDS.collinear_ratio)
        self.assertTrue(mo.is_collinear(nearly))
        frames = [nearly + i * np.array([0.0, 0.02, 0.0]) for i in range(5)]
        est = mo.estimate_motion(buffer_from_frames(frames))
        self.assertEqual(est.reason, "collinear_points")
        self.assertIsNone(est.direction)

    def test_a_barely_two_dimensional_cloud_is_penalised_not_trusted(self):
        flat = CLOUD.copy()
        flat[:, 1] = 0.10 + 0.01 * (flat[:, 1] - flat[:, 1].mean())   # squash one axis
        est = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6,
                                                                   cloud=flat)))
        full = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6)))
        self.assertLessEqual(est.confidence, full.confidence + 1e-12)


class TestDisagreementGate(unittest.TestCase):
    """Gate 3: frames whose cameras disagree are refused at push time."""

    def test_a_disagreeing_frame_is_not_stored(self):
        buf = mo.MotionBuffer(capacity=6, max_disagreement=0.08)
        buf.push(CLOUD, frame_idx=0, disagreement=0.01)
        buf.push(CLOUD + 0.02, frame_idx=1, disagreement=0.20)
        buf.push(CLOUD + 0.04, frame_idx=2, disagreement=0.00)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.rejected_frames, 1)
        _pts, _mask, frames = buf.window()
        np.testing.assert_allclose(frames, [0, 2])

    def test_a_disagreeing_frame_cannot_drag_the_axis(self):
        """The reason the gate exists, measured."""
        frames, _panel, axis, hinge = door_frames(n=8, step=0.05)
        bad = frames[-1].copy()
        bad[0] += np.array([0.25, 0.20, -0.15])       # one camera matched something else

        with_bad = mo.MotionBuffer(capacity=10, max_disagreement=None)
        gated = mo.MotionBuffer(capacity=10, max_disagreement=0.08)
        for i, f in enumerate(frames[:-1]):
            with_bad.push(f, frame_idx=i, disagreement=0.005)
            gated.push(f, frame_idx=i, disagreement=0.005)
        with_bad.push(bad, frame_idx=len(frames) - 1, disagreement=0.22)
        gated.push(bad, frame_idx=len(frames) - 1, disagreement=0.22)

        err_bad = angle_between(mo.estimate_motion(with_bad).axis_dir, axis)
        err_gated = angle_between(mo.estimate_motion(gated).axis_dir, axis)
        self.assertLess(err_gated, err_bad)
        self.assertLess(err_gated, np.radians(1.0))

    def test_the_gate_is_off_by_default(self):
        buf = mo.MotionBuffer(capacity=4)
        buf.push(CLOUD, frame_idx=0, disagreement=9.9)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.rejected_frames, 0)

    def test_missing_or_non_finite_disagreement_is_accepted(self):
        buf = mo.MotionBuffer(capacity=4, max_disagreement=0.08)
        buf.push(CLOUD, frame_idx=0)
        buf.push(CLOUD, frame_idx=1, disagreement=float("nan"))
        self.assertEqual(len(buf), 2)

    def test_rejection_is_logged_once_per_run_with_the_value(self):
        spy = Spy()
        buf = mo.MotionBuffer(capacity=6, max_disagreement=0.08, logger=spy)
        for i in range(4):
            buf.push(CLOUD, frame_idx=i, disagreement=0.20)
        hits = [ln for ln in spy.lines if "disagreement" in ln]
        self.assertEqual(len(hits), 1, spy.lines)
        self.assertIn("0.2000", hits[0])
        self.assertEqual(buf.rejected_frames, 4)

    def test_push_state_applies_both_frame_gates(self):
        class State:
            def __init__(self, point, disagreement=0.0, lost=False):
                self.world_point = np.asarray(point, dtype=float)
                self.disagreement = disagreement
                self.lost = lost

        buf = mo.MotionBuffer(capacity=6, max_disagreement=0.08)
        buf.push_state(State([0.0, 0.0, 0.0]), frame_idx=0)
        buf.push_state(State([0.0, 0.0, 0.01], disagreement=0.5), frame_idx=1)
        buf.push_state(State([0.0, 0.0, 0.02], lost=True), frame_idx=2)
        buf.push_state(State([0.0, 0.0, 0.03]), points=CLOUD, frame_idx=3)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.rejected_frames, 2)

    def test_reset_clears_the_rejection_counter(self):
        buf = mo.MotionBuffer(capacity=4, max_disagreement=0.01)
        buf.push(CLOUD, frame_idx=0, disagreement=0.5)
        buf.reset()
        self.assertEqual(buf.rejected_frames, 0)


class TestDirectionSmoothing(unittest.TestCase):
    """Gate 4: sign-aligned EMA. Without the alignment the average cancels to nothing."""

    @staticmethod
    def _flipping_samples(n=12, sigma=0.05, seed=3):
        """An axis estimate that keeps flipping sign, as real solvers hand it back."""
        rng = np.random.default_rng(seed)
        truth = np.array([0.0, 0.0, 1.0])
        out = []
        for i in range(n):
            v = truth + rng.normal(0, sigma, 3)
            v /= np.linalg.norm(v)
            out.append(v if i % 2 == 0 else -v)
        return truth, out

    def test_naive_averaging_cancels_the_estimate(self):
        """The failure this gate exists to prevent - asserted, not assumed."""
        _truth, samples = self._flipping_samples()
        naive = np.mean(samples, axis=0)
        self.assertLess(float(np.linalg.norm(naive)), 0.15,
                        "sign-flipping unit vectors must cancel when averaged raw")

    def test_naive_ema_fails_where_the_smoother_succeeds(self):
        truth, samples = self._flipping_samples()

        # The same EMA, without the sign alignment.
        naive = None
        for v in samples:
            naive = np.asarray(v, dtype=float) if naive is None else 0.4 * v + 0.6 * naive
        naive_err = angle_between(naive, truth) if np.linalg.norm(naive) > 1e-9 else np.pi

        sm = mo.DirectionSmoother()
        for v in samples:
            sm.update(v)

        self.assertGreater(naive_err, np.radians(30.0), "naive EMA should be far off")
        self.assertLess(angle_between(sm.value, truth), np.radians(5.0))
        self.assertAlmostEqual(float(np.linalg.norm(sm.value)), 1.0, places=9)
        self.assertGreater(sm.flips, 0)

    def test_smoothing_reduces_the_noise_on_a_stable_axis(self):
        rng = np.random.default_rng(11)
        truth = np.array([0.0, 0.0, 1.0])
        sm = mo.DirectionSmoother(alpha=0.3)
        last_raw = None
        for _ in range(20):
            v = truth + rng.normal(0, 0.08, 3)
            v /= np.linalg.norm(v)
            last_raw = v
            sm.update(v)
        self.assertLess(angle_between(sm.value, truth), angle_between(last_raw, truth))

    def test_first_sample_passes_through_and_junk_is_ignored(self):
        sm = mo.DirectionSmoother()
        np.testing.assert_allclose(sm.update([0.0, 0.0, 2.0]), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(sm.update([0.0, 0.0, 0.0]), [0, 0, 1], atol=1e-12)
        self.assertIsNone(mo.DirectionSmoother().update(None))

    def test_the_flip_is_logged_once(self):
        spy = Spy()
        sm = mo.DirectionSmoother(logger=spy, name="axis")
        for v in ([0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, -1]):
            sm.update(v)
        hits = [ln for ln in spy.lines if "sign-flipped" in ln]
        self.assertEqual(len(hits), 1, spy.lines)

    def test_motion_smoother_stabilises_an_estimate_stream(self):
        axis = np.array([0.0, 0.0, 1.0])
        hinge = np.array([0.30, -0.15, 0.25])
        rng = np.random.default_rng(13)
        sm = mo.MotionSmoother(alpha=0.3)
        raw_errs, smooth_errs = [], []
        for _ in range(12):
            frames = [f + rng.normal(0, 3e-3, f.shape)
                      for f in rotating_frames(axis, hinge, 0.08, 8)]
            buf = buffer_from_frames(frames)
            raw = mo.estimate_motion(buf)
            smoothed = mo.estimate_motion(buf, smoother=sm)
            raw_errs.append(angle_between(raw.axis_dir, axis))
            smooth_errs.append(angle_between(smoothed.axis_dir, axis))
        self.assertLess(float(np.mean(smooth_errs[3:])), float(np.mean(raw_errs[3:])))

    def test_a_rejected_window_is_never_blended_in(self):
        sm = mo.MotionSmoother()
        good = mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6)),
                                  smoother=sm)
        before = np.array(sm.direction.value, copy=True)
        rejected = mo.estimate_motion(mo.MotionBuffer(), smoother=sm)
        self.assertIsNone(rejected.direction)
        np.testing.assert_allclose(sm.direction.value, before, atol=1e-12)
        self.assertIsNotNone(good.direction)

    def test_changing_kind_resets_the_history(self):
        sm = mo.MotionSmoother()
        mo.estimate_motion(buffer_from_frames(sliding_frames([0, 1, 0], 0.02, 6)), smoother=sm)
        axis = np.array([0.0, 0.0, 1.0])
        est = mo.estimate_motion(buffer_from_frames(
            rotating_frames(axis, [0.30, -0.15, 0.25], 0.08, 6)), smoother=sm)
        self.assertEqual(est.kind, mo.REVOLUTE)
        self.assertEqual(sm.direction.n_updates, 1, "history from the slide must be dropped")

    def test_smoothing_keeps_perpendicularity_consistent(self):
        face = np.array([[a, 0.0, b] for a in np.linspace(-0.08, 0.08, 3)
                         for b in np.linspace(-0.08, 0.08, 3)]) + np.array([0.35, 0.40, 0.25])
        sm = mo.MotionSmoother()
        est = None
        for _ in range(3):
            est = mo.estimate_motion(
                buffer_from_frames(sliding_frames([0, -1, 0], 0.02, 6, cloud=face)),
                view_point=[0.35, -2.0, 0.25], smoother=sm)
        self.assertAlmostEqual(est.perpendicularity,
                               float(abs(est.direction @ est.plane_normal)), places=12)


class TestConfidenceCalibration(unittest.TestCase):
    """Gate 5: confidence has to track the angular error, not just look plausible."""

    AXIS = np.array([0.0, 0.0, 1.0])
    HINGE = np.array([0.30, -0.15, 0.20])
    SIGMAS = (0.0, 2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)

    def _sweep(self, trials=24, seed=101):
        """``[(sigma, mean confidence, median axis error rad)]`` over the noise sweep."""
        rows = []
        for sigma in self.SIGMAS:
            rng = np.random.default_rng(seed)
            confs, errs = [], []
            for _ in range(trials):
                frames = [f + rng.normal(0, sigma, f.shape)
                          for f in rotating_frames(self.AXIS, self.HINGE, 0.08, 8)]
                est = mo.estimate_motion(buffer_from_frames(frames))
                confs.append(est.confidence)
                if est.axis_dir is not None:
                    errs.append(angle_between(est.axis_dir, self.AXIS))
            rows.append((sigma, float(np.mean(confs)),
                         float(np.median(errs)) if errs else np.pi))
        return rows

    def test_confidence_decreases_monotonically_with_noise(self):
        rows = self._sweep()
        confs = [c for _s, c, _e in rows]
        for a, b in zip(confs, confs[1:]):
            self.assertLessEqual(b, a + 1e-9, f"confidence is not monotone: {confs}")
        self.assertGreater(confs[0], 0.9)
        self.assertLess(confs[-1], 0.4)

    def test_four_millimetre_noise_is_clearly_degraded(self):
        """The calibration bug this replaces reported 0.911 at a measured 5.8 deg error."""
        rows = {s: (c, e) for s, c, e in self._sweep()}
        conf, err = rows[4e-3]
        self.assertGreater(err, np.radians(3.0), "4 mm noise really does cost several degrees")
        self.assertLessEqual(conf, 0.6, f"confidence {conf:.3f} is too generous at 4 mm noise")
        self.assertGreater(conf, 0.2, "...but it should not collapse either")

    def test_high_confidence_implies_low_angular_error(self):
        rng = np.random.default_rng(77)
        for _ in range(60):
            sigma = float(rng.uniform(0.0, 0.01))
            frames = [f + rng.normal(0, sigma, f.shape)
                      for f in rotating_frames(self.AXIS, self.HINGE, 0.08, 8)]
            est = mo.estimate_motion(buffer_from_frames(frames))
            if est.confidence >= 0.8:
                self.assertLess(angle_between(est.axis_dir, self.AXIS), np.radians(3.0),
                                f"confidence {est.confidence:.3f} promised more than it delivered")

    def test_confidence_is_relative_to_the_motion_not_the_residual_alone(self):
        """1 mm of residual is garbage on a 2 cm motion and fine on a 40 cm one."""
        rng = np.random.default_rng(5)
        small = [f + rng.normal(0, 1e-3, f.shape) for f in sliding_frames([0, 1, 0], 0.004, 6)]
        large = [f + rng.normal(0, 1e-3, f.shape) for f in sliding_frames([0, 1, 0], 0.08, 6)]
        c_small = mo.estimate_motion(buffer_from_frames(small)).confidence
        c_large = mo.estimate_motion(buffer_from_frames(large)).confidence
        self.assertLess(c_small, c_large)

    def test_more_points_are_worth_more_confidence(self):
        rng = np.random.default_rng(9)
        noise_a = rng.normal(0, 2e-3, (6, len(CLOUD), 3))
        few = [f[:3] + noise_a[i][:3] for i, f in enumerate(sliding_frames([0, 1, 0], 0.02, 6))]
        many = [f + noise_a[i] for i, f in enumerate(sliding_frames([0, 1, 0], 0.02, 6))]
        self.assertLess(mo.estimate_motion(buffer_from_frames(few)).confidence,
                        mo.estimate_motion(buffer_from_frames(many)).confidence)

    def test_confidence_zero_always_means_no_geometry(self):
        rng = np.random.default_rng(41)
        cases = [
            buffer_from_frames([CLOUD]),
            mo.MotionBuffer(),
            buffer_from_frames([CLOUD[:2], CLOUD[:2] + 0.02]),
            buffer_from_frames([CLOUD + rng.normal(0, 1e-4, CLOUD.shape) for _ in range(5)]),
        ]
        for buf in cases:
            est = mo.estimate_motion(buf)
            self.assertEqual(est.confidence, 0.0)
            self.assertIsNone(est.direction)
            self.assertIsNone(est.axis_dir)
            self.assertIsNone(est.axis_point)
            self.assertTrue(est.reason)

    def test_the_measured_calibration_table(self):
        """Prints the table quoted in the summary; asserts its shape rather than exact values."""
        rows = self._sweep()
        for sigma, conf, err in rows:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
            print(f"sigma={sigma * 1000:5.2f}mm  confidence={conf:.3f}  "
                  f"axis_err={np.degrees(err):5.2f}deg")


if __name__ == "__main__":
    unittest.main()
