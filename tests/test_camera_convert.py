"""Unit tests for :mod:`tracking.camera_convert` - the OpenGL <-> OpenCV bridge.

**Pure math: no GPU, no simulator, no LAPA clone.** That is deliberate - this module is the
most likely source of a silent failure in the 3D tracking stack, so its tests must run
everywhere, in under a second, with no optional dependency able to skip them.

The central assertion is a *cross-check between two independent code paths*: the existing
OpenGL pipeline (``tracking.geometry.project_world_to_pixel``, which the whole repo already
trusts) and the new OpenCV pipeline (``K``/``w2c``). Both must land on the same pixel. A
conversion bug that survives both is essentially impossible.

Run with::

    python -m pytest tests/test_camera_convert.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from sim_adapter import camera_math  # noqa: E402
from tracking import camera_convert as cc  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.types import CameraView  # noqa: E402

WIDTH, HEIGHT = 320, 256
FOV = 60.0


def make_view(name="head", eye=(0.9, 1.1, 0.8), target=(0.0, 0.4, 0.15),
              width=WIDTH, height=HEIGHT):
    """A CameraView with real GL matrices but a dummy depth buffer.

    Uses ``sim_adapter.camera_math``, i.e. the exact same flat-16 column-major matrices
    PyBullet and Genesis produce, so the conventions under test are the production ones.
    """
    view_matrix = camera_math.gl_view_matrix(eye, target, (0.0, 0.0, 1.0))
    projection = camera_math.gl_projection_matrix(FOV, float(width) / float(height),
                                                  config.near_plane, config.far_plane)
    return CameraView(
        name=name,
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.ones((height, width), dtype=np.float32),
        view_matrix=geometry.mat4(view_matrix),
        projection_matrix=geometry.mat4(projection),
        near=float(config.near_plane), far=float(config.far_plane),
        position=list(eye),
    )


# A spread of points around the scene origin, all comfortably inside the frustum.
WORLD_POINTS = np.array([
    [0.00, 0.40, 0.15],
    [0.12, 0.35, 0.22],
    [-0.10, 0.45, 0.08],
    [0.05, 0.50, 0.30],
    [-0.06, 0.32, 0.18],
], dtype=float)


class TestFlatMatrixHandling(unittest.TestCase):
    """``mat4`` must treat a flat 16 as column-major, like PyBullet does."""

    def test_flat_16_is_column_major(self):
        m = np.arange(16, dtype=float).reshape(4, 4)
        flat = list(m.flatten(order="F"))
        np.testing.assert_allclose(cc.mat4(flat), m)

    def test_4x4_passes_through_untransposed(self):
        # Round-tripping an existing 4x4 through a C-order reshape would transpose it;
        # that silent transpose is exactly what this guards against.
        m = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_allclose(cc.mat4(m), m)

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            cc.mat4(np.zeros((3, 3)))


class TestGlToOpenCv(unittest.TestCase):
    """The axis flip itself."""

    def test_flip_is_its_own_inverse(self):
        np.testing.assert_allclose(cc.GL_TO_CV @ cc.GL_TO_CV, np.eye(4), atol=1e-12)

    def test_round_trip_gl_to_cv_to_gl(self):
        view = make_view()
        w2c = cc.gl_view_to_opencv_w2c(view.view_matrix)
        back = cc.opencv_w2c_to_gl_view(w2c)
        np.testing.assert_allclose(back, geometry.mat4(view.view_matrix), atol=1e-12)

    def test_points_in_front_have_positive_z(self):
        """The single most important property: OpenCV looks down +Z.

        In OpenGL the same points have *negative* z, so a missing flip inverts this test.
        """
        view = make_view()
        _K, w2c = cc.camera_matrices(view)
        _uv, z_cv = cc.project_opencv(_K, w2c, WORLD_POINTS)
        self.assertTrue(np.all(z_cv > 0.0), f"expected z_cam > 0, got {z_cv}")

        # ... and confirm the un-flipped GL matrix really does give the opposite sign,
        # so the test above is meaningful rather than vacuously true.
        gl = geometry.mat4(view.view_matrix)
        z_gl = (gl @ np.hstack([WORLD_POINTS, np.ones((len(WORLD_POINTS), 1))]).T).T[:, 2]
        self.assertTrue(np.all(z_gl < 0.0))
        np.testing.assert_allclose(z_cv, -z_gl, atol=1e-9)

    def test_opencv_depth_matches_geometry_z_eye(self):
        """``z_cam`` must equal the metric depth the existing GL path reports."""
        view = make_view()
        K, w2c = cc.camera_matrices(view)
        _uv, z_cv = cc.project_opencv(K, w2c, WORLD_POINTS)
        for point, z in zip(WORLD_POINTS, z_cv):
            _pixel, z_eye = geometry.project_world_to_pixel(view, point)
            self.assertAlmostEqual(z, z_eye, places=9)


class TestIntrinsics(unittest.TestCase):

    def test_principal_point_is_image_centre(self):
        view = make_view()
        K = cc.intrinsics_from_gl_projection(view.projection_matrix, WIDTH, HEIGHT)
        self.assertAlmostEqual(K[0, 2], WIDTH / 2.0, places=9)
        self.assertAlmostEqual(K[1, 2], HEIGHT / 2.0, places=9)
        self.assertEqual(K[0, 1], 0.0)

    def test_focal_length_matches_fov(self):
        """fy = (H/2) / tan(fov/2); fx follows from the aspect ratio (square pixels)."""
        view = make_view()
        K = cc.intrinsics_from_gl_projection(view.projection_matrix, WIDTH, HEIGHT)
        expected_fy = (HEIGHT / 2.0) / np.tan(np.radians(FOV) / 2.0)
        self.assertAlmostEqual(K[1, 1], expected_fy, places=6)
        # gl_projection_matrix divides m[0,0] by aspect, so fx == fy for square pixels.
        self.assertAlmostEqual(K[0, 0], expected_fy, places=6)

    def test_intrinsics_scale_with_resolution(self):
        """K is expressed in the resolution the 2D points were measured at."""
        view = make_view()
        k1 = cc.intrinsics_from_gl_projection(view.projection_matrix, WIDTH, HEIGHT)
        k2 = cc.intrinsics_from_gl_projection(view.projection_matrix, 2 * WIDTH, 2 * HEIGHT)
        np.testing.assert_allclose(k2[:2], 2.0 * k1[:2], rtol=1e-12)


class TestProjectionAgreesWithExistingPipeline(unittest.TestCase):
    """The cross-check: OpenCV K/w2c vs the repo's trusted OpenGL path."""

    def test_pixels_match_geometry_project_world_to_pixel(self):
        for eye, target in [((0.9, 1.1, 0.8), (0.0, 0.4, 0.15)),
                            ((-0.7, 0.2, 1.3), (0.0, 0.45, 0.1)),
                            ((0.0, 1.6, 0.35), (0.0, 0.4, 0.2))]:
            view = make_view(eye=eye, target=target)
            K, w2c = cc.camera_matrices(view)
            uv, _z = cc.project_opencv(K, w2c, WORLD_POINTS)
            for point, got in zip(WORLD_POINTS, uv):
                expected, _z_eye = geometry.project_world_to_pixel(view, point)
                self.assertIsNotNone(expected)
                np.testing.assert_allclose(
                    got, expected, atol=1e-6,
                    err_msg=f"eye={eye} point={point}: OpenCV {got} vs OpenGL {expected}")

    def test_non_square_aspect_does_not_swap_axes(self):
        """A square image would hide an fx/fy or width/height mix-up; this cannot."""
        view = make_view(width=640, height=256)
        K, w2c = cc.camera_matrices(view)
        uv, _z = cc.project_opencv(K, w2c, WORLD_POINTS)
        for point, got in zip(WORLD_POINTS, uv):
            expected, _ = geometry.project_world_to_pixel(view, point)
            np.testing.assert_allclose(got, expected, atol=1e-6)


class TestTriangulationRoundTrip(unittest.TestCase):
    """Two correct cameras must triangulate back to the original world point.

    This is the end-to-end proof that K, the axis flip and P = K[R|t] are mutually
    consistent - and it is the same DLT that LAPA anchors its prediction on.
    """

    @staticmethod
    def _triangulate(projections, pixels):
        rows = []
        for P, (u, v) in zip(projections, pixels):
            rows.append(u * P[2] - P[0])
            rows.append(v * P[2] - P[1])
        _u, _s, vt = np.linalg.svd(np.asarray(rows))
        X = vt[-1]
        return X[:3] / X[3]

    def test_two_view_dlt_recovers_world_points(self):
        views = [make_view("head", eye=(0.9, 1.1, 0.8)),
                 make_view("side", eye=(-0.8, 0.9, 0.6))]
        mats = [cc.camera_matrices(v) for v in views]
        projections = [cc.projection_matrix_3x4(K, w2c) for K, w2c in mats]
        pixels_per_view = [cc.project_opencv(K, w2c, WORLD_POINTS)[0] for K, w2c in mats]

        for i, truth in enumerate(WORLD_POINTS):
            got = self._triangulate(projections, [p[i] for p in pixels_per_view])
            self.assertLess(float(np.linalg.norm(got - truth)), 1e-9,
                            f"triangulated {got} != {truth}")

    def test_triangulation_fails_loudly_without_the_axis_flip(self):
        """Sanity: the round-trip above must genuinely depend on the conversion.

        Using raw OpenGL extrinsics for P produces points that do not match - proving the
        passing test above is not an accident of symmetry.
        """
        views = [make_view("head", eye=(0.9, 1.1, 0.8)),
                 make_view("side", eye=(-0.8, 0.9, 0.6))]
        mats = [cc.camera_matrices(v) for v in views]
        pixels_per_view = [cc.project_opencv(K, w2c, WORLD_POINTS)[0] for K, w2c in mats]
        bad_projections = [np.asarray(K) @ geometry.mat4(v.view_matrix)[:3, :]
                           for (K, _w2c), v in zip(mats, views)]

        errors = [float(np.linalg.norm(
            self._triangulate(bad_projections, [p[i] for p in pixels_per_view]) - truth))
            for i, truth in enumerate(WORLD_POINTS)]
        self.assertGreater(max(errors), 1e-3,
                           "un-flipped extrinsics should NOT reconstruct correctly")


class TestAabbNormalisation(unittest.TestCase):

    def test_normalize_denormalize_round_trip(self):
        center, half = cc.aabb_from_points(WORLD_POINTS)
        norm = cc.normalize_points(WORLD_POINTS, center, half)
        back = cc.denormalize_points(norm, center, half)
        np.testing.assert_allclose(back, WORLD_POINTS, atol=1e-12)

    def test_points_land_inside_the_unit_box(self):
        center, half = cc.aabb_from_points(WORLD_POINTS)
        norm = cc.normalize_points(WORLD_POINTS, center, half)
        self.assertTrue(np.all(np.abs(norm) <= 1.0 + 1e-9), f"outside [-1,1]: {norm}")

    def test_outlier_does_not_blow_up_the_box(self):
        """One stray point (e.g. a seed that deprojected onto the background) must be
        rejected, not allowed to inflate ``half`` and collapse every real point to ~0."""
        polluted = np.vstack([WORLD_POINTS, [[50.0, 50.0, 50.0]]])
        center_clean, half_clean = cc.aabb_from_points(WORLD_POINTS)
        center_dirty, half_dirty = cc.aabb_from_points(polluted)
        np.testing.assert_allclose(center_dirty, center_clean, atol=1e-9)
        np.testing.assert_allclose(half_dirty, half_clean, atol=1e-9)

    def test_genuine_spread_is_not_rejected(self):
        """The rejector must not mistake a legitimately wide object for an outlier."""
        spread = np.array([[0.0, 0.4, 0.1], [0.3, 0.4, 0.1],
                           [0.0, 0.7, 0.1], [0.0, 0.4, 0.4]])
        _center, half = cc.aabb_from_points(spread)
        self.assertGreater(float(np.min(half)), 0.1,
                           f"a 30 cm object was shrunk to half={half}")

    def test_degenerate_cloud_gets_a_finite_box(self):
        center, half = cc.aabb_from_points(np.tile([0.1, 0.2, 0.3], (4, 1)))
        self.assertTrue(np.all(half > 0.0), "a coincident cloud must still get half > 0")
        np.testing.assert_allclose(center, [0.1, 0.2, 0.3], atol=1e-9)

    def test_build_w2c_normalized_matches_manual_transform(self):
        """The normalised extrinsics must map X_norm exactly as w2c maps X_world."""
        view = make_view()
        _K, w2c = cc.camera_matrices(view)
        center, half = cc.aabb_from_points(WORLD_POINTS)
        w2c_norm = cc.build_w2c_normalized(w2c, center, half)

        norm = cc.normalize_points(WORLD_POINTS, center, half)
        via_norm = (w2c_norm @ np.hstack([norm, np.ones((len(norm), 1))]).T).T[:, :3]
        via_world = (w2c @ np.hstack([WORLD_POINTS, np.ones((len(WORLD_POINTS), 1))]).T).T[:, :3]
        np.testing.assert_allclose(via_norm, via_world, atol=1e-9)

    def test_projection_is_identical_through_normalized_extrinsics(self):
        """End-to-end: LAPA's normalised pipeline must reproduce the same pixels."""
        view = make_view()
        K, w2c = cc.camera_matrices(view)
        center, half = cc.aabb_from_points(WORLD_POINTS)
        w2c_norm = cc.build_w2c_normalized(w2c, center, half)
        norm = cc.normalize_points(WORLD_POINTS, center, half)

        uv_world, _ = cc.project_opencv(K, w2c, WORLD_POINTS)
        uv_norm, _ = cc.project_opencv(K, w2c_norm, norm)
        np.testing.assert_allclose(uv_norm, uv_world, atol=1e-6)


class TestGuardrail(unittest.TestCase):
    """``assert_opencv_frame_sane`` turns the silent-freeze failure into a loud one."""

    def test_passes_for_correct_extrinsics(self):
        view = make_view()
        K, w2c = cc.camera_matrices(view)
        uv, z = cc.assert_opencv_frame_sane(K, w2c, WORLD_POINTS,
                                            image_size=(WIDTH, HEIGHT), name="head")
        self.assertEqual(uv.shape, (len(WORLD_POINTS), 2))
        self.assertTrue(np.all(z > 0))

    def test_raises_on_unflipped_opengl_extrinsics(self):
        view = make_view()
        K, _w2c = cc.camera_matrices(view)
        with self.assertRaises(AssertionError) as ctx:
            cc.assert_opencv_frame_sane(K, geometry.mat4(view.view_matrix), WORLD_POINTS)
        self.assertIn("behind the camera", str(ctx.exception))
        self.assertIn("gl_view_to_opencv_w2c", str(ctx.exception),
                      "the error must name the fix, not just the symptom")

    def test_raises_on_out_of_frame_points(self):
        """A point that is genuinely *in front* but far off-axis must be reported as
        out of frame - not mistaken for the behind-the-camera case."""
        view = make_view()
        K, w2c = cc.camera_matrices(view)
        # Build the point in camera space (z=1 m ahead, 10 m to the side) and map it back
        # to world, so "in front but outside the image" is guaranteed by construction
        # rather than by a hand-picked literal.
        cam_point = np.array([10.0, 0.0, 1.0, 1.0])
        far_off = (np.linalg.inv(w2c) @ cam_point)[:3].reshape(1, 3)
        _uv, z = cc.project_opencv(K, w2c, far_off)
        self.assertGreater(float(z[0]), 0.0, "test point must be in front of the camera")

        with self.assertRaises(AssertionError) as ctx:
            cc.assert_opencv_frame_sane(K, w2c, far_off, image_size=(WIDTH, HEIGHT))
        self.assertIn("outside", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
