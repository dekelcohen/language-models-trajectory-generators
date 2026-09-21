"""Unit tests for :mod:`providers.tracker3d.rigid_refine`.

**Pure math: no GPU, no simulator, no LAPA.** The headline claim under test is the one the
module exists for: at exactly two cameras the reprojection residual is structurally blind to
a drifted 2D track (see ``tests/test_tracker3d.py::test_two_view_residual_is_blind_to_drift``),
so ``triangulate`` *cannot* catch it - and the rigid-body constraint can, because it uses a
completely independent signal, the object's known shape.

The second most important test is the negative one: when *many* points disagree the object
has genuinely deformed or the track has collapsed, and snapping every point onto that pose
would produce a confident, smooth, entirely wrong trajectory. The provider must fall through
to the raw wrapped output instead.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from providers.tracker3d import SUPPORTED, get_tracker3d  # noqa: E402
from providers.tracker3d.rigid_refine import RigidRefineTracker3D  # noqa: E402
from providers.tracker3d.triangulate import (  # noqa: E402
    TriangulateTracker3D,
    reprojection_errors,
)
from sim_adapter import camera_math  # noqa: E402
from tracking import camera_convert as cc  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.types import CamTrack, CameraView  # noqa: E402

WIDTH, HEIGHT = 320, 256
FOV = 60.0

# A small rigid object: eight corners of a box plus two off-face points, so the cloud is
# genuinely 3D (no degenerate singular value) and has enough redundancy for one point to be
# outvoted.
BOX = np.array([
    [0.00, 0.40, 0.12], [0.08, 0.40, 0.12], [0.08, 0.48, 0.12], [0.00, 0.48, 0.12],
    [0.00, 0.40, 0.20], [0.08, 0.40, 0.20], [0.08, 0.48, 0.20], [0.00, 0.48, 0.20],
    [0.04, 0.44, 0.24], [0.04, 0.36, 0.16],
], dtype=float)


def make_view(name, eye, target=(0.04, 0.44, 0.17)):
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
    return {"head": make_view("head", (0.9, 1.1, 0.8)),
            "wrist": make_view("wrist", (-0.7, 1.0, 0.6))}


def project_into(view, world_points):
    return np.asarray([geometry.project_world_to_pixel(view, p)[0] for p in world_points],
                      dtype=float)


def make_track(cam, points_2d):
    track = CamTrack(cam=cam, seeded=True)
    track.points_2d = np.asarray(points_2d, dtype=float)
    track.visible = np.ones(len(points_2d), dtype=bool)
    track.confidence = 0.9
    track.status = "ok"
    track.health = 1.0
    track.depth_valid = True
    return track


def tracks_for(views, world_points):
    return {cam: make_track(cam, project_into(view, world_points))
            for cam, view in views.items()}


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class RigidRefineTestBase(unittest.TestCase):
    def setUp(self):
        self.views = two_views()
        self.provider = RigidRefineTracker3D(base="triangulate")
        self.raw = TriangulateTracker3D()

    def lift(self, world_points, provider=None):
        provider = provider or self.provider
        return provider.lift("box", self.views, tracks_for(self.views, world_points))

    def capture_template(self, world_points=BOX):
        """Feed one clean frame so the template exists."""
        result = self.lift(world_points)
        self.assertEqual(result.meta.get("refine"), "template_captured")
        return result


class TestTwoCameraBlindSpot(RigidRefineTestBase):
    """The whole reason this module exists, as an executable contrast."""

    def _drifted_tracks(self, point_row=3, drift=(47.0, 31.0)):
        """Clean pixels everywhere except one point in the wrist view."""
        tracks = tracks_for(self.views, BOX)
        tracks["wrist"].points_2d = tracks["wrist"].points_2d.copy()
        tracks["wrist"].points_2d[point_row] += np.asarray(drift, dtype=float)
        return tracks

    def test_triangulate_cannot_see_the_drift_but_rigid_refine_can(self):
        """Two cameras, one point drifted by 47 px in one view.

        ``triangulate`` accepts it: with V=2 the DLT system is exactly determined, so the
        corrupted pixel pair still reprojects to a fraction of a pixel. Its ``max_reproj_px``
        gate therefore never fires and the 3D point is silently wrong by centimetres.

        ``rigid_refine`` sees it immediately, because it is not looking at the residual at
        all - it is asking "does this point still sit where the object's shape says it
        should, given where the other nine points are?".
        """
        self.capture_template()
        tracks = self._drifted_tracks()

        raw = self.raw.lift("box", self.views, tracks)
        self.assertTrue(raw.ok, "2-view triangulation accepts the corrupted pair by construction")
        raw_row = int(np.nonzero(raw.per_point_index == 3)[0][0])
        raw_error = float(np.linalg.norm(raw.per_point[raw_row] - BOX[3]))
        self.assertGreater(raw_error, 0.01, "the drifted point really is wrong by >1 cm")

        # ... and the residual that IRLS / LAPA's view weighting depend on says nothing.
        projections = [cc.projection_matrix_3x4(*cc.camera_matrices(self.views[c]))
                       for c in ("head", "wrist")]
        pixels = np.asarray([tracks[c].points_2d[3] for c in ("head", "wrist")])
        residual_px = float(np.max(reprojection_errors(projections, pixels,
                                                       raw.per_point[raw_row])))
        self.assertLess(residual_px, self.raw.max_reproj_px / 5.0,
                        f"a {raw_error * 100:.1f} cm error shows up as {residual_px:.2f} px, "
                        "nowhere near the max_reproj_px gate - if this ever fails the 2-view "
                        "cliff changed and the motivation for this module should be revisited")

        refined = self.provider.lift("box", self.views, tracks)
        self.assertEqual(refined.meta["refine"], "applied")
        self.assertEqual(refined.meta["n_corrected"], 1)
        self.assertEqual(refined.meta["corrected_index"], [3])
        row = int(np.nonzero(refined.per_point_index == 3)[0][0])
        refined_error = float(np.linalg.norm(refined.per_point[row] - BOX[3]))
        self.assertLess(refined_error, 1e-3,
                        f"corrected point should land on truth, got {refined_error:.5f} m")
        self.assertLess(refined_error, raw_error / 10.0)

    def test_centroid_is_pulled_back_towards_truth(self):
        self.capture_template()
        tracks = self._drifted_tracks()
        raw = self.raw.lift("box", self.views, tracks)
        refined = self.provider.lift("box", self.views, tracks)
        truth = BOX.mean(axis=0)
        self.assertLess(np.linalg.norm(refined.world_point - truth),
                        np.linalg.norm(raw.world_point - truth))

    def test_correction_survives_object_motion(self):
        """The constraint is on *shape*, not position: the object may move meanwhile."""
        self.capture_template()
        moved = BOX @ rotation((0.2, 0.3, 1.0), 0.35).T + np.array([0.05, -0.03, 0.02])
        tracks = tracks_for(self.views, moved)
        tracks["wrist"].points_2d = tracks["wrist"].points_2d.copy()
        tracks["wrist"].points_2d[6] += np.array([40.0, -28.0])

        refined = self.provider.lift("box", self.views, tracks)
        self.assertEqual(refined.meta["refine"], "applied")
        self.assertIn(6, refined.meta["corrected_index"])
        row = int(np.nonzero(refined.per_point_index == 6)[0][0])
        self.assertLess(np.linalg.norm(refined.per_point[row] - moved[6]), 2e-3)


class TestFallThrough(RigidRefineTestBase):
    """The single most important behaviour: do not 'correct' onto a bogus pose."""

    def test_many_corrupted_points_fall_through_to_raw_output(self):
        self.capture_template()
        tracks = tracks_for(self.views, BOX)
        tracks["wrist"].points_2d = tracks["wrist"].points_2d.copy()
        for row in (0, 1, 2, 4, 5, 7, 8):
            tracks["wrist"].points_2d[row] += np.array([50.0, 35.0])

        raw = self.raw.lift("box", self.views, tracks)
        refined = self.provider.lift("box", self.views, tracks)
        self.assertEqual(refined.meta["refine"], "fell_through")
        self.assertEqual(refined.meta["why_no_refine"], "low_inlier_frac")
        self.assertNotIn("n_corrected", refined.meta)
        np.testing.assert_allclose(refined.world_point, raw.world_point, atol=1e-12)

    def test_non_rigid_deformation_falls_through(self):
        """An object that genuinely deformed is not 'repaired' onto the stale template."""
        self.capture_template()
        deformed = BOX.copy()
        deformed[:5] += np.array([0.0, 0.0, 0.06])          # half the cloud pulled apart
        refined = self.lift(deformed)
        self.assertEqual(refined.meta["refine"], "fell_through")

    def test_template_is_dropped_after_repeated_fall_through(self):
        """Persistent disagreement means the template is stale, not the measurement."""
        provider = RigidRefineTracker3D(base="triangulate", max_stale_frames=3)
        self.lift(BOX, provider)
        deformed = BOX.copy()
        deformed[:5] += np.array([0.0, 0.0, 0.06])
        for _ in range(3):
            self.assertEqual(self.lift(deformed, provider).meta["refine"], "fell_through")
        # Template gone, so the next frame re-captures against what is actually tracked now.
        self.assertEqual(self.lift(deformed, provider).meta["refine"], "template_captured")
        self.assertEqual(self.lift(deformed, provider).meta["refine"], "applied")


class TestNoFalsePositives(RigidRefineTestBase):
    """Clean data must come through essentially untouched."""

    def _assert_identity(self, moved, label):
        raw = self.raw.lift("box", self.views, tracks_for(self.views, moved))
        refined = self.lift(moved)
        self.assertEqual(refined.meta["refine"], "applied", label)
        self.assertEqual(refined.meta["n_corrected"], 0, f"{label}: spurious correction")
        np.testing.assert_allclose(refined.per_point, raw.per_point, atol=1e-9,
                                   err_msg=f"{label}: points should be untouched")
        self.assertLess(np.linalg.norm(refined.world_point - moved.mean(axis=0)), 1e-3, label)

    def test_pure_translation(self):
        self.capture_template()
        self._assert_identity(BOX + np.array([0.06, -0.04, 0.03]), "translation")

    def test_pure_rotation(self):
        self.capture_template()
        centre = BOX.mean(axis=0)
        R = rotation((0.0, 0.0, 1.0), 0.6)
        self._assert_identity((BOX - centre) @ R.T + centre, "rotation")

    def test_screw_motion(self):
        self.capture_template()
        R = rotation((0.3, -0.5, 1.0), 0.45)
        self._assert_identity(BOX @ R.T + np.array([0.04, 0.05, -0.02]), "screw")

    def test_rigid_rmse_is_reported_and_small_on_clean_data(self):
        self.capture_template()
        result = self.lift(BOX + np.array([0.01, 0.0, 0.0]))
        self.assertLess(result.meta["rigid_rmse_m"], 1e-3)
        self.assertEqual(result.meta["rigid_inlier_frac"], 1.0)
        self.assertGreaterEqual(result.meta["rigid_n_used"], len(BOX) - 3)


class TestDegenerateGeometry(RigidRefineTestBase):
    def test_two_points_never_capture_a_template(self):
        result = self.lift(BOX[:2])
        self.assertEqual(result.meta["refine"], "too_few_points")
        self.assertTrue(result.ok)

    def test_three_points_are_refused(self):
        """3 points determine a pose exactly: zero redundancy, nothing detectable."""
        result = self.lift(BOX[:3])
        self.assertEqual(result.meta["refine"], "too_few_points")

    def test_collinear_template_is_refused(self):
        line = np.array([[0.0, 0.40, 0.12 + 0.02 * i] for i in range(6)], dtype=float)
        result = self.lift(line)
        self.assertEqual(result.meta["refine"], "template_collinear")

    def test_near_zero_extent_template_is_refused(self):
        tiny = BOX.mean(axis=0) + 1e-4 * (BOX - BOX.mean(axis=0))
        result = self.lift(tiny)
        self.assertEqual(result.meta["refine"], "template_extent_too_small")

    def test_no_template_yet_is_a_clean_passthrough(self):
        raw = self.raw.lift("box", self.views, tracks_for(self.views, BOX))
        first = self.lift(BOX)
        np.testing.assert_allclose(first.world_point, raw.world_point, atol=1e-12)


class TestCorrespondenceAndTemplateLifecycle(RigidRefineTestBase):
    def _tracks_with_index(self, index_by_cam):
        tracks, point_index = {}, {}
        for cam, view in self.views.items():
            idx = np.asarray(index_by_cam[cam], dtype=int)
            tracks[cam] = make_track(cam, project_into(view, BOX[idx]))
            point_index[cam] = idx
        return tracks, point_index

    def test_template_grows_as_points_become_visible(self):
        """Occlusion means not every seed point is visible at t=0; the template adapts."""
        subset = np.arange(6)
        tracks, point_index = self._tracks_with_index({"head": subset, "wrist": subset})
        first = self.provider.lift("box", self.views, tracks, point_index=point_index)
        self.assertEqual(first.meta["n_template"], 6)

        full = np.arange(len(BOX))
        tracks, point_index = self._tracks_with_index({"head": full, "wrist": full})
        second = self.provider.lift("box", self.views, tracks, point_index=point_index)
        self.assertEqual(second.meta["refine"], "applied")
        self.assertEqual(second.meta["n_template"], len(BOX))

    def test_reseed_to_a_disjoint_point_set_invalidates_the_template(self):
        first = np.arange(5)
        tracks, point_index = self._tracks_with_index({"head": first, "wrist": first})
        self.provider.lift("box", self.views, tracks, point_index=point_index)

        second = np.arange(5, 10)              # re-seed kept nothing in common
        tracks, point_index = self._tracks_with_index({"head": second, "wrist": second})
        result = self.provider.lift("box", self.views, tracks, point_index=point_index)
        self.assertEqual(result.meta["refine"], "template_captured")
        self.assertEqual(result.meta["n_template"], 5)

    def test_centroid_fallback_points_are_excluded(self):
        """``-1`` means 'no correspondence' - such a point cannot belong to a template."""
        idx = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        tracks, _ = self._tracks_with_index({"head": np.arange(len(BOX)),
                                             "wrist": np.arange(len(BOX))})
        result = self.provider.lift("box", self.views, tracks,
                                    point_index={"head": idx, "wrist": idx})
        self.assertFalse(result.ok)
        self.assertNotIn("box", self.provider._templates)

    def test_reset_forgets_the_template(self):
        self.capture_template()
        self.provider.reset()
        self.assertEqual(self.lift(BOX).meta["refine"], "template_captured")


class TestContract(RigidRefineTestBase):
    """The ABC's 'never raise' rule, and the factory wiring."""

    def test_registered_in_the_factory(self):
        self.assertIn("rigid_refine", SUPPORTED)
        provider = get_tracker3d("rigid_refine")
        self.assertIsInstance(provider, RigidRefineTracker3D)
        self.assertEqual(provider.base_name, "triangulate")

    def test_wrapped_provider_is_configurable(self):
        provider = get_tracker3d("rigid_refine", base="depth_fusion")
        self.assertEqual(provider.base_name, "depth_fusion")
        self.assertTrue(provider.needs_depth)
        self.assertFalse(RigidRefineTracker3D(base="triangulate").needs_depth)

    def test_never_raises_on_hostile_input(self):
        cases = [
            ("no cameras", {}, {}),
            ("one camera", {"head": self.views["head"]},
             {"head": make_track("head", project_into(self.views["head"], BOX))}),
            ("unseeded", self.views, {c: CamTrack(cam=c, seeded=False) for c in self.views}),
            ("nan pixels", self.views,
             {c: make_track(c, np.full((4, 2), np.nan)) for c in self.views}),
        ]
        for label, views, tracks in cases:
            with self.subTest(label):
                result = self.provider.lift("box", views, tracks)
                self.assertIsNotNone(result)
                self.assertFalse(result.ok)

    def test_a_raising_base_provider_is_absorbed(self):
        class Exploding:
            name = "boom"

            def lift(self, *_args, **_kwargs):
                raise RuntimeError("checkpoint missing")

        provider = RigidRefineTracker3D(base=Exploding())
        result = provider.lift("box", self.views, tracks_for(self.views, BOX))
        self.assertFalse(result.ok)
        self.assertIn("boom", result.meta["base"])
        self.assertIn("RuntimeError", result.meta["why"])

    def test_meta_records_the_wrapped_provider(self):
        result = self.lift(BOX)
        self.assertEqual(result.meta["provider"], "rigid_refine")
        self.assertEqual(result.meta["base"], "triangulate")


if __name__ == "__main__":
    unittest.main()
