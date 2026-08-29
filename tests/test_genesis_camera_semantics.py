"""Pins Genesis camera semantics against the PyBullet contract.

Runs under **vlm_genesis** (skips itself elsewhere):

    & ...\\envs\\vlm_genesis\\python.exe -m pytest tests\\test_genesis_camera_semantics.py -q

The Genesis adapter's camera code rests on four empirical facts. Each is asserted here so
that a Genesis upgrade cannot silently invalidate it — every one of them would otherwise
fail as subtly wrong 3D coordinates rather than as a crash.

1. ``camera.projection_matrix`` is PyBullet's ``computeProjectionMatrixFOV`` **transposed**.
2. ``inv(camera.transform)`` is PyBullet's ``computeViewMatrix`` **transposed**.
   Both therefore flatten with ``order='F'`` to the exact vectors ``utils`` expects, so
   ``utils.get_intrinsics_extrinsics`` needs no Genesis-specific branch.
3. Depth is **linear metric z_eye** (metres along the optical axis) — not euclidean range
   and not an OpenGL non-linear [0,1] buffer. ``utils.get_world_point_world_frame`` assumes
   the latter, hence the ``depth_encoding="linear_metric"`` branch.
4. ``camera.extrinsics`` is a ``@cached_property`` that goes **stale** when the camera moves.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import genesis as gs
    GENESIS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the interpreter in use
    gs = None
    GENESIS_IMPORT_ERROR = exc

FOV = 60.0
RES = (256, 256)
NEAR, FAR = 0.01, 100.0
CAM_HEIGHT = 2.0

# Genesis returns float32 matrices, so cross-sim comparisons bottom out around 1e-7.
MATRIX_TOL = 1e-5
# Depth is float32 metres; 2.0 m comes back as 1.9999847.
DEPTH_TOL = 1e-3


def ref_projection_matrix(fov_deg, aspect, near, far):
    """PyBullet ``computeProjectionMatrixFOV`` as a 4x4 in column-major (OpenGL) order."""
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (2.0 * far * near) / (near - far)
    return m


def ref_view_matrix(eye, target, up):
    """PyBullet ``computeViewMatrix`` as a 4x4 in column-major (OpenGL) order."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, np.asarray(up, dtype=np.float64))
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)

    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0], m[1, 0], m[2, 0] = side
    m[0, 1], m[1, 1], m[2, 1] = true_up
    m[0, 2], m[1, 2], m[2, 2] = -forward
    m[3, 0] = -np.dot(side, eye)
    m[3, 1] = -np.dot(true_up, eye)
    m[3, 2] = np.dot(forward, eye)
    m[3, 3] = 1.0
    return m


def _as_2d_depth(raw):
    depth = np.asarray(raw, dtype=np.float64)
    return depth[..., 0] if depth.ndim == 3 else depth


@unittest.skipIf(gs is None, f"genesis is not importable here ({GENESIS_IMPORT_ERROR})")
class TestGenesisCameraSemantics(unittest.TestCase):
    """One scene for the whole class: gs.init and scene.build are expensive."""

    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.cpu, logging_level="warning")
        cls.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 240.0, gravity=(0.0, 0.0, -9.81)),
            show_viewer=False,
        )
        cls.scene.add_entity(gs.morphs.Plane())
        # debug=True is what makes scene.draw_debug_* show up in offscreen renders.
        cls.cam = cls.scene.add_camera(
            res=RES, pos=(0.0, 0.0, CAM_HEIGHT), lookat=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
            fov=FOV, near=NEAR, far=FAR, GUI=False, debug=True,
        )
        cls.scene.build()

    def setUp(self):
        self._look_straight_down()

    def _look_straight_down(self):
        self.cam.set_pose(pos=(0.0, 0.0, CAM_HEIGHT), lookat=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))

    def test_projection_matrix_is_pybullet_transposed(self):
        proj = np.asarray(self.cam.projection_matrix, dtype=np.float64)
        expected = ref_projection_matrix(FOV, RES[0] / RES[1], NEAR, FAR)

        self.assertEqual(proj.shape, (4, 4), "same dims as PyBullet's 4x4 projection")
        np.testing.assert_allclose(
            proj.T, expected, atol=MATRIX_TOL,
            err_msg="Genesis projection_matrix is no longer PyBullet's transposed",
        )
        # The actual contract: flatten(order='F') must equal PyBullet's flat 16-vector,
        # because utils.get_intrinsics_extrinsics does reshape(4, 4, order='F').
        np.testing.assert_allclose(
            proj.flatten(order="F"), expected.flatten(order="C"), atol=MATRIX_TOL
        )

    def test_view_matrix_from_inverse_transform_matches_pybullet(self):
        # Asymmetric pose on purpose: an axis-aligned one gives a near-identity rotation
        # that matches its own transpose and would prove nothing about the layout.
        eye, target, up = (0.7, -1.3, 1.9), (0.1, 0.25, 0.4), (0.0, 0.0, 1.0)
        self.cam.set_pose(pos=eye, lookat=target, up=up)

        view = np.linalg.inv(np.asarray(self.cam.transform, dtype=np.float64))
        expected = ref_view_matrix(eye, target, up)

        self.assertEqual(view.shape, (4, 4))
        np.testing.assert_allclose(
            view.T, expected, atol=MATRIX_TOL,
            err_msg="inv(camera.transform) is no longer PyBullet's viewMatrix transposed",
        )
        np.testing.assert_allclose(
            view.flatten(order="F"), expected.flatten(order="C"), atol=MATRIX_TOL
        )

    def test_depth_is_linear_metric_not_opengl_ndc(self):
        depth = _as_2d_depth(self.cam.render(rgb=False, depth=True)[1])
        centre = float(depth[depth.shape[0] // 2, depth.shape[1] // 2])

        self.assertAlmostEqual(
            centre, CAM_HEIGHT, delta=DEPTH_TOL,
            msg=f"expected metres to the ground plane ({CAM_HEIGHT}), got {centre}",
        )
        ndc = (1.0 / CAM_HEIGHT - 1.0 / NEAR) / (1.0 / FAR - 1.0 / NEAR)
        self.assertGreater(
            abs(centre - ndc), 0.5,
            "depth looks like an OpenGL non-linear buffer; the linear_metric branch is wrong",
        )

    def test_depth_is_z_eye_not_euclidean_range(self):
        """The ground plane is perpendicular to the optical axis.

        z_eye => every pixel reads the same value. Euclidean range => corners read
        ~1/cos(theta) larger. Getting this backwards skews reconstructed 3D points
        radially outward, worst at the image edges.
        """
        depth = _as_2d_depth(self.cam.render(rgb=False, depth=True)[1])
        spread = float(depth.max() - depth.min())

        half = np.deg2rad(FOV) / 2.0
        corner_factor = float(np.sqrt(1.0 + 2.0 * np.tan(half) ** 2))
        euclidean_spread = CAM_HEIGHT * (corner_factor - 1.0)

        self.assertLess(
            spread, DEPTH_TOL,
            f"depth varies by {spread} across a perpendicular plane; that is euclidean "
            f"range (would spread ~{euclidean_spread:.3f}), not z_eye",
        )

    def test_background_pixels_return_far_not_zero_or_inf(self):
        """No-hit pixels must be detectable; unprojecting them would emit bogus points."""
        self.cam.set_pose(pos=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 5.0), up=(0.0, 1.0, 0.0))
        sky = _as_2d_depth(self.cam.render(rgb=False, depth=True)[1])

        self.assertFalse(np.isnan(sky).any(), "background depth contains NaN")
        self.assertFalse(np.isinf(sky).any(), "background depth contains inf")
        self.assertAlmostEqual(
            float(sky.max()), FAR, delta=0.05,
            msg=f"background should read ~far ({FAR}), got {float(sky.max())}",
        )
        self.assertGreater(float(sky.min()), FAR * 0.99,
                           "background is not uniformly at the far plane")

    def test_debug_markers_appear_in_offscreen_render(self):
        """Why the PyBullet massless-MultiBody marker hack is unnecessary on Genesis."""
        before = np.asarray(self.cam.render(rgb=True)[0], dtype=np.int64)
        node = self.scene.draw_debug_sphere(pos=(0.0, 0.0, 0.3), radius=0.25,
                                            color=(1.0, 0.0, 0.0, 1.0))
        try:
            after = np.asarray(self.cam.render(rgb=True)[0], dtype=np.int64)
            changed = int((np.abs(after - before).sum(axis=-1) > 8).sum())
            self.assertGreater(
                changed, 50,
                "debug markers are invisible offscreen; check the camera's debug=True flag "
                "(vis/rasterizer.py: skip_markers = not camera.debug)",
            )
        finally:
            self.scene.clear_debug_object(node)

        restored = np.asarray(self.cam.render(rgb=True)[0], dtype=np.int64)
        self.assertLessEqual(int(np.abs(restored - before).max()), 8,
                             "clear_debug_object did not remove the marker")

    def test_extrinsics_cached_property_goes_stale(self):
        """Canary. ``camera.extrinsics`` caches and never refreshes after set_pose.

        The adapter must recompute ``inv(camera.transform)`` on every capture. If Genesis
        ever fixes this, this test fails and the workaround can be revisited.
        """
        self._look_straight_down()
        first_transform = np.asarray(self.cam.transform, dtype=np.float64).copy()
        first_extrinsics = np.asarray(self.cam.extrinsics, dtype=np.float64).copy()

        self.cam.set_pose(pos=(1.5, 1.5, 1.5), lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
        transform_delta = float(np.abs(np.asarray(self.cam.transform, dtype=np.float64)
                                       - first_transform).max())
        extrinsics_delta = float(np.abs(np.asarray(self.cam.extrinsics, dtype=np.float64)
                                        - first_extrinsics).max())

        self.assertGreater(transform_delta, 1e-6, "camera.transform did not track set_pose")
        self.assertEqual(
            extrinsics_delta, 0.0,
            "camera.extrinsics now refreshes after set_pose - Genesis appears to have fixed "
            "the cached_property; the adapter's manual inv(transform) can be reconsidered",
        )


if __name__ == "__main__":
    unittest.main()
