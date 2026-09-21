"""Unit tests for :mod:`providers.tracker3d` - the multi-camera 3D lift providers.

**Pure math: no GPU, no simulator, no LAPA clone, no checkpoint.** These providers sit on
the critical path of every tracked frame, so their tests must run everywhere and fast.

The two things most worth pinning down:

1. ``triangulate`` must recover a known world point from synthetic cameras to sub-millimetre
   accuracy, and must *fail* rather than invent an answer when the geometry is degenerate.
2. **Seed-index correspondence.** Each camera may track a different subset of an object's
   seed points, so the per-camera arrays are not positionally aligned. A triangulator that
   zips them produces a confident, smooth, completely wrong point - the worst failure mode
   in the stack because nothing downstream can detect it. There is an explicit negative
   control for this below.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from providers.tracker3d import get_tracker3d  # noqa: E402
from providers.tracker3d.base import MultiCamTracker3D  # noqa: E402
from providers.tracker3d.depth_fusion import DepthFusionTracker3D  # noqa: E402
from providers.tracker3d.triangulate import (  # noqa: E402
    TriangulateTracker3D,
    reprojection_errors,
    triangulate_irls,
)
from sim_adapter import camera_math  # noqa: E402
from tracking import camera_convert as cc  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.types import CamTrack, CameraView  # noqa: E402

WIDTH, HEIGHT = 320, 256
FOV = 60.0

# Four points on a small object near the scene origin, deliberately non-collinear.
SEED_POINTS = np.array([
    [0.00, 0.40, 0.15],
    [0.06, 0.38, 0.19],
    [-0.05, 0.43, 0.12],
    [0.02, 0.45, 0.24],
], dtype=float)


def make_view(name, eye, target=(0.0, 0.40, 0.17)):
    view_matrix = camera_math.gl_view_matrix(eye, target, (0.0, 0.0, 1.0))
    projection = camera_math.gl_projection_matrix(FOV, float(WIDTH) / float(HEIGHT),
                                                  config.near_plane, config.far_plane)
    return CameraView(
        name=name,
        rgb=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
        depth=np.ones((HEIGHT, WIDTH), dtype=np.float32),
        view_matrix=geometry.mat4(view_matrix),
        projection_matrix=geometry.mat4(projection),
        near=float(config.near_plane), far=float(config.far_plane),
        position=list(eye),
    )


def two_views():
    return {
        "head": make_view("head", (0.9, 1.1, 0.8)),
        "wrist": make_view("wrist", (-0.7, 1.0, 0.6)),
    }


def project_into(view, world_points):
    """Ground-truth pixels via the repo's trusted OpenGL path."""
    return np.asarray([geometry.project_world_to_pixel(view, p)[0] for p in world_points],
                      dtype=float)


def make_track(cam, points_2d, visible=None, confidence=0.9, status="ok",
               world_point=None, health=1.0, depth_valid=True):
    n = len(points_2d)
    track = CamTrack(cam=cam, seeded=True)
    track.points_2d = np.asarray(points_2d, dtype=float)
    track.visible = np.ones(n, dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
    track.confidence = confidence
    track.status = status
    track.world_point = None if world_point is None else np.asarray(world_point, dtype=float)
    track.health = health
    track.depth_valid = depth_valid
    return track


class TestTriangulateIrls(unittest.TestCase):
    """The solver itself, independent of the provider wrapper."""

    def setUp(self):
        self.views = two_views()
        self.cams = {c: cc.camera_matrices(v) for c, v in self.views.items()}

    def _projections_and_pixels(self, world_point, cams=None):
        cams = cams or list(self.views)
        projections, pixels = [], []
        for cam in cams:
            K, w2c = self.cams[cam]
            projections.append(cc.projection_matrix_3x4(K, w2c))
            pixels.append(project_into(self.views[cam], [world_point])[0])
        return projections, np.asarray(pixels)

    def test_recovers_a_known_point_to_sub_millimetre(self):
        for target in SEED_POINTS:
            projections, pixels = self._projections_and_pixels(target)
            point, residual = triangulate_irls(projections, pixels)
            self.assertIsNotNone(point)
            self.assertLess(np.linalg.norm(point - target), 1e-4,
                            f"triangulation missed {target}")
            self.assertLess(residual, 1e-3)

    def test_two_view_residual_is_blind_to_drift(self):
        """**The two-camera cliff, as an executable assertion.**

        With exactly 2 views the DLT system is *exactly determined*: 4 equations for 4
        homogeneous unknowns. Any pair of pixels therefore triangulates to some point that
        reprojects almost perfectly, however wrong the 2D tracks are. The reprojection
        residual - the signal IRLS reweighting and LAPA's ``view_weight_head`` both depend
        on - is structurally near-zero and carries no information at 2 views.

        Consequence: at 2 cameras there is nothing for any view-weighting scheme to learn
        from, and a drifted track is silently accepted. This is the single strongest
        argument for a third camera.
        """
        projections, pixels = self._projections_and_pixels(SEED_POINTS[0])
        pixels[1] += np.array([40.0, 25.0])          # wrist drifted badly
        point, residual = triangulate_irls(projections, pixels)
        self.assertLess(residual, 1.0,
                        "2-view residual is structurally ~0; if this ever fails the "
                        "solver changed and the 3-camera argument should be revisited")
        self.assertGreater(np.linalg.norm(point - SEED_POINTS[0]), 0.01,
                           "the point really is wrong - only the residual fails to say so")

    def test_three_view_residual_does_expose_drift(self):
        """With a third view the system is over-determined and the outlier is exposed.

        Note *which* statistic exposes it. Once IRLS has successfully rejected the drifted
        view, the point is correct, so the two good views reproject near-perfectly and the
        **median** residual is ~0 - the median is a robust statistic and hides exactly what
        we are looking for. The **max** is the signal for "one camera disagrees".
        """
        self.views["side"] = make_view("side", (0.1, 1.4, -0.5))
        self.cams["side"] = cc.camera_matrices(self.views["side"])
        projections, pixels = self._projections_and_pixels(
            SEED_POINTS[0], ["head", "wrist", "side"])
        pixels[1] += np.array([40.0, 25.0])
        point, median = triangulate_irls(projections, pixels)

        self.assertLess(np.linalg.norm(point - SEED_POINTS[0]), 1e-3,
                        "the drifted view should have been rejected outright")
        self.assertLess(median, 1.0, "median hides the outlier once rejection worked")
        errors = reprojection_errors(projections, pixels, point)
        self.assertGreater(np.max(errors), 5.0, "max must still expose the bad view")

    def test_third_view_outvotes_a_single_bad_track(self):
        """The whole argument for a third camera, as an executable assertion.

        With two views there is nothing for the reweighting to choose between, so a drifted
        track corrupts the result. With three, IRLS can down-weight the outlier.
        """
        self.views["side"] = make_view("side", (0.1, 1.4, -0.5))
        self.cams["side"] = cc.camera_matrices(self.views["side"])
        target = SEED_POINTS[0]

        two_p, two_px = self._projections_and_pixels(target, ["head", "wrist"])
        two_px[1] += np.array([12.0, 9.0])
        two_point, _ = triangulate_irls(two_p, two_px)

        three_p, three_px = self._projections_and_pixels(target, ["head", "wrist", "side"])
        three_px[1] += np.array([12.0, 9.0])
        three_point, _ = triangulate_irls(three_p, three_px)

        err_two = np.linalg.norm(two_point - target)
        err_three = np.linalg.norm(three_point - target)
        self.assertLess(err_three, err_two,
                        f"3-view error {err_three:.4f} should beat 2-view {err_two:.4f}")

    def test_single_view_is_refused(self):
        projections, pixels = self._projections_and_pixels(SEED_POINTS[0], ["head"])
        point, residual = triangulate_irls(projections, pixels)
        self.assertIsNone(point)
        self.assertEqual(residual, float("inf"))

    def test_identical_cameras_are_refused_not_guessed(self):
        """Degenerate baseline: two identical cameras cannot triangulate."""
        K, w2c = self.cams["head"]
        P = cc.projection_matrix_3x4(K, w2c)
        pixel = project_into(self.views["head"], [SEED_POINTS[0]])[0]
        point, _residual = triangulate_irls([P, P], np.asarray([pixel, pixel]))
        # Either refused outright, or returned something that must not be trusted as exact.
        if point is not None:
            self.assertGreater(np.linalg.norm(point - SEED_POINTS[0]), 1e-3)


class TestTriangulateProvider(unittest.TestCase):
    def setUp(self):
        self.views = two_views()
        self.provider = TriangulateTracker3D()

    def _tracks(self, world_points, index_by_cam=None):
        tracks = {}
        for cam, view in self.views.items():
            idx = None if index_by_cam is None else index_by_cam[cam]
            pts = world_points if idx is None else world_points[idx]
            tracks[cam] = make_track(cam, project_into(view, pts))
        return tracks

    def test_recovers_the_object_centroid(self):
        tracks = self._tracks(SEED_POINTS)
        result = self.provider.lift("box", self.views, tracks)
        self.assertTrue(result.ok)
        self.assertLess(np.linalg.norm(result.world_point - SEED_POINTS.mean(axis=0)), 1e-3)
        self.assertEqual(sorted(result.used_cams), ["head", "wrist"])
        self.assertEqual(len(result.per_point), len(SEED_POINTS))

    def test_does_not_use_the_depth_buffer(self):
        """Poison every depth value; a geometry-only lift must be unaffected."""
        tracks = self._tracks(SEED_POINTS)
        clean = self.provider.lift("box", self.views, tracks).world_point
        for view in self.views.values():
            view.depth[:] = np.float32(1e6)
        poisoned = self.provider.lift("box", self.views, tracks).world_point
        np.testing.assert_allclose(clean, poisoned, atol=1e-12)
        self.assertFalse(self.provider.needs_depth)

    def test_single_camera_yields_nothing(self):
        tracks = self._tracks(SEED_POINTS)
        result = self.provider.lift("box", {"head": self.views["head"]},
                                    {"head": tracks["head"]})
        self.assertFalse(result.ok)
        self.assertIn("need 2 cams", result.meta.get("why", ""))

    def test_invisible_points_are_skipped(self):
        tracks = self._tracks(SEED_POINTS)
        tracks["wrist"].visible = np.array([True, False, False, True])
        result = self.provider.lift("box", self.views, tracks)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.per_point), 2)

    def test_no_overlapping_visible_point_yields_nothing(self):
        """Both cameras see points, but never the *same* point - nothing is triangulable."""
        tracks = self._tracks(SEED_POINTS)
        tracks["head"].visible = np.array([True, True, False, False])
        tracks["wrist"].visible = np.array([False, False, True, True])
        result = self.provider.lift("box", self.views, tracks)
        self.assertFalse(result.ok)

    def test_mismatched_seed_subsets_are_aligned_by_index(self):
        """The core correspondence guarantee.

        ``head`` tracks seed points [0,1,2,3]; ``wrist`` lost points 0 and 2 on a re-seed and
        tracks only [1,3]. Aligned by seed index this is exact; zipped positionally it would
        match wrist's point 1 against head's point 0.
        """
        wrist_idx = np.array([1, 3])
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS)),
            "wrist": make_track("wrist", project_into(self.views["wrist"],
                                                      SEED_POINTS[wrist_idx])),
        }
        point_index = {"head": np.arange(4), "wrist": wrist_idx}

        good = self.provider.lift("box", self.views, tracks, point_index=point_index)
        self.assertTrue(good.ok)
        self.assertEqual(sorted(good.per_point_index.tolist()), [1, 3])
        expected = SEED_POINTS[wrist_idx].mean(axis=0)
        self.assertLess(np.linalg.norm(good.world_point - expected), 1e-3)

    def test_positional_zipping_would_be_wrong(self):
        """Negative control proving the previous test is actually testing something."""
        wrist_idx = np.array([1, 3])
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS)),
            "wrist": make_track("wrist", project_into(self.views["wrist"],
                                                      SEED_POINTS[wrist_idx])),
        }
        aligned = self.provider.lift("box", self.views, tracks,
                                     point_index={"head": np.arange(4), "wrist": wrist_idx})
        # Without the index map the provider falls back to positional order, which pairs
        # head[0] with wrist[0] (= seed 1) and head[1] with wrist[1] (= seed 3).
        zipped = self.provider.lift("box", self.views, tracks, point_index=None)
        self.assertTrue(aligned.ok)
        if zipped.ok:
            self.assertGreater(np.linalg.norm(zipped.world_point - aligned.world_point), 1e-3,
                               "positional zipping should not agree with index alignment")

    def test_centroid_fallback_points_are_never_matched(self):
        """A ``-1`` index means 'no correspondence' and must not be triangulated."""
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS[:1])),
            "wrist": make_track("wrist", project_into(self.views["wrist"], SEED_POINTS[:1])),
        }
        result = self.provider.lift("box", self.views, tracks,
                                    point_index={"head": np.array([-1]),
                                                 "wrist": np.array([-1])})
        self.assertFalse(result.ok)

    def test_gross_2d_drift_is_not_caught_at_two_cameras(self):
        """Documents the limit of the reprojection gate rather than pretending it works.

        See ``test_two_view_residual_is_blind_to_drift``: at 2 views the residual is
        structurally ~0, so ``max_reproj_px`` cannot fire. The provider happily returns a
        wrong point. Only a third camera, or an independent check (epipolar pre-filter,
        cross-view feature similarity, rigid-body fit), closes this hole.
        """
        tracks = self._tracks(SEED_POINTS)
        tracks["wrist"].points_2d = tracks["wrist"].points_2d + np.array([60.0, 45.0])
        result = self.provider.lift("box", self.views, tracks)
        self.assertTrue(result.ok, "2-view triangulation cannot detect this - by construction")
        self.assertGreater(np.linalg.norm(result.world_point - SEED_POINTS.mean(axis=0)), 0.01)

    def test_gross_2d_drift_is_rejected_with_a_third_camera(self):
        self.views["side"] = make_view("side", (0.1, 1.4, -0.5))
        tracks = self._tracks(SEED_POINTS)
        tracks["wrist"].points_2d = tracks["wrist"].points_2d + np.array([60.0, 45.0])
        result = self.provider.lift("box", self.views, tracks)
        self.assertTrue(result.ok)
        self.assertLess(np.linalg.norm(result.world_point - SEED_POINTS.mean(axis=0)), 0.02,
                        "IRLS should down-weight the drifted wrist track")


class TestDepthFusionProvider(unittest.TestCase):
    """``depth_fusion`` must reproduce the arbitration rules it inherited."""

    def setUp(self):
        self.views = two_views()
        self.provider = DepthFusionTracker3D()

    def test_averages_two_healthy_cameras(self):
        a, b = np.array([0.0, 0.40, 0.15]), np.array([0.02, 0.42, 0.17])
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS),
                               world_point=a),
            "wrist": make_track("wrist", project_into(self.views["wrist"], SEED_POINTS),
                                world_point=b),
        }
        result = self.provider.lift("box", self.views, tracks)
        self.assertTrue(result.ok)
        for axis in range(3):
            self.assertGreaterEqual(result.world_point[axis], min(a[axis], b[axis]) - 1e-9)
            self.assertLessEqual(result.world_point[axis], max(a[axis], b[axis]) + 1e-9)

    def test_unhealthy_camera_is_excluded_when_a_healthy_one_exists(self):
        good = np.array([0.0, 0.40, 0.15])
        bad = np.array([0.0, 0.90, 0.15])          # an occluder, 50 cm away
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS),
                               world_point=good, status="ok", health=1.0),
            "wrist": make_track("wrist", project_into(self.views["wrist"], SEED_POINTS),
                                world_point=bad, status="occluded", health=0.1),
        }
        result = self.provider.lift("box", self.views, tracks)
        np.testing.assert_allclose(result.world_point, good, atol=1e-9)
        self.assertEqual(result.used_cams, ["head"])

    def test_disagreeing_camera_is_dropped_but_still_reported(self):
        good = np.array([0.0, 0.40, 0.15])
        bad = good + np.array([0.0, 0.0, config.track_disagree_m * 4])
        tracks = {
            "head": make_track("head", project_into(self.views["head"], SEED_POINTS),
                               world_point=good, status="ok", health=1.0),
            "wrist": make_track("wrist", project_into(self.views["wrist"], SEED_POINTS),
                                world_point=bad, status="ok", health=0.2),
        }
        result = self.provider.lift("box", self.views, tracks)
        np.testing.assert_allclose(result.world_point, good, atol=1e-9)
        self.assertIsNotNone(result.disagreement)
        self.assertGreater(result.disagreement, config.track_disagree_m,
                           "disagreement must stay visible in the report even when dropped")

    def test_no_world_points_yields_nothing(self):
        tracks = {c: make_track(c, project_into(v, SEED_POINTS), status="lost")
                  for c, v in self.views.items()}
        result = self.provider.lift("box", self.views, tracks)
        self.assertFalse(result.ok)


class TestFactory(unittest.TestCase):
    def test_returns_each_supported_provider(self):
        for name in ("depth_fusion", "triangulate"):
            provider = get_tracker3d(name)
            self.assertIsInstance(provider, MultiCamTracker3D)
            self.assertEqual(provider.name, name)

    def test_default_is_depth_fusion_so_behaviour_is_unchanged(self):
        self.assertEqual(get_tracker3d().name, "depth_fusion")
        self.assertEqual(config.tracker3d_provider_default, "depth_fusion")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_tracker3d("nope")


class TestProvidersNeverRaise(unittest.TestCase):
    """A tracking bug must not break a rollout - malformed input returns an empty result."""

    def setUp(self):
        self.views = two_views()

    def test_empty_and_malformed_inputs(self):
        cases = {
            "no tracks": {},
            "unseeded": {"head": CamTrack(cam="head", seeded=False)},
            "no points": {"head": CamTrack(cam="head", seeded=True)},
            "visible length mismatch": {
                "head": make_track("head", [[10.0, 10.0], [20.0, 20.0]],
                                   visible=[True]),
            },
        }
        for name in ("depth_fusion", "triangulate"):
            provider = get_tracker3d(name)
            for label, tracks in cases.items():
                with self.subTest(provider=name, case=label):
                    result = provider.lift("box", self.views, tracks)
                    self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
