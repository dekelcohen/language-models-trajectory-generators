"""The depth-encoding bridge in ``utils.get_world_point_world_frame``.

PyBullet hands back a non-linear OpenGL z-buffer; Genesis hands back metres along the
optical axis. Both must unproject to the *same* world point through the one shared
inverse view-projection path, and the PyBullet payload must keep behaving exactly as it
did before the branch existed.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import utils  # noqa: E402
from sim_adapter import camera_math  # noqa: E402


class _StubArgs:
    """``utils`` reads its CLI args from a module global that ``main.py`` installs."""
    save_grasp_inputs = False


utils.args = _StubArgs


NEAR, FAR = config.near_plane, config.far_plane
FOV, ASPECT = config.fov, config.aspect
IMAGE_SIZE = (config.image_width, config.image_height)

EYE = [0.4, -0.9, 0.8]
TARGET = [0.0, 0.1, 0.35]
UP = [0.0, 0.0, 1.0]


def _matrices():
    view = np.array(camera_math.gl_view_matrix(EYE, TARGET, UP), dtype=float)
    proj = np.array(camera_math.gl_projection_matrix(FOV, ASPECT, NEAR, FAR), dtype=float)
    return view, proj


def _cam_info(depth_encoding):
    view, proj = _matrices()
    return {
        "head": {"viewMatrix": list(view), "projectionMatrix": list(proj),
                 "znear": float(NEAR), "zfar": float(FAR)},
        "depth_encoding": depth_encoding,
        "new_3d_proj": True,
    }


def _project(world_point):
    """World point -> (pixel_x, pixel_y, opengl_depth, metric_depth)."""
    view, proj = _matrices()
    VP = proj.reshape(4, 4, order="F") @ view.reshape(4, 4, order="F")
    clip = VP @ np.append(np.asarray(world_point, float), 1.0)
    ndc = clip / clip[3]
    px = (ndc[0] + 1.0) * IMAGE_SIZE[0] / 2.0
    py = (1.0 - ndc[1]) * IMAGE_SIZE[1] / 2.0
    opengl_depth = (ndc[2] + 1.0) / 2.0
    metric_depth = camera_math.depth_to_metric(np.array([opengl_depth]),
                                               camera_math.DEPTH_OPENGL, NEAR, FAR)[0]
    return px, py, opengl_depth, float(metric_depth)


WORLD_POINTS = [
    [0.0, 0.1, 0.35],
    [-0.2, 0.4, 0.10],
    [0.15, -0.05, 0.60],
    [-0.11, 0.04, 0.25],
]


class TestDepthEncodingBranch(unittest.TestCase):

    def test_opengl_round_trip(self):
        """The pre-existing PyBullet path still unprojects to the original point."""
        info = _cam_info(camera_math.DEPTH_OPENGL)
        for wp in WORLD_POINTS:
            with self.subTest(world_point=wp):
                px, py, d_gl, _ = _project(wp)
                got = utils.get_world_point_world_frame(
                    EYE, [0, 0, 0, 1], "head", IMAGE_SIZE, [px, py, d_gl], cam_info=info)
                np.testing.assert_allclose(np.asarray(got, float), wp, atol=1e-9)

    def test_linear_metric_matches_opengl(self):
        """Genesis' metric depth must land on the same world point as PyBullet's."""
        info_gl = _cam_info(camera_math.DEPTH_OPENGL)
        info_m = _cam_info(camera_math.DEPTH_LINEAR_METRIC)
        for wp in WORLD_POINTS:
            with self.subTest(world_point=wp):
                px, py, d_gl, d_m = _project(wp)
                a = utils.get_world_point_world_frame(
                    EYE, [0, 0, 0, 1], "head", IMAGE_SIZE, [px, py, d_gl], cam_info=info_gl)
                b = utils.get_world_point_world_frame(
                    EYE, [0, 0, 0, 1], "head", IMAGE_SIZE, [px, py, d_m], cam_info=info_m)
                np.testing.assert_allclose(np.asarray(b, float), np.asarray(a, float), atol=1e-6)
                np.testing.assert_allclose(np.asarray(b, float), wp, atol=1e-6)

    def test_metric_depth_is_the_distance_along_the_optical_axis(self):
        """Sanity-check the quantity itself, not just the round trip."""
        view = np.array(camera_math.gl_view_matrix(EYE, TARGET, UP), dtype=float).reshape(4, 4, order="F")
        for wp in WORLD_POINTS:
            with self.subTest(world_point=wp):
                _, _, _, d_m = _project(wp)
                eye_space = view @ np.append(np.asarray(wp, float), 1.0)
                # OpenGL looks down -z, so z_eye is the negated camera-space z.
                self.assertAlmostEqual(d_m, -eye_space[2], places=5)

    def test_missing_encoding_defaults_to_opengl(self):
        """Old payloads without the key must not change behaviour."""
        info = _cam_info(camera_math.DEPTH_OPENGL)
        info.pop("depth_encoding")
        px, py, d_gl, _ = _project(WORLD_POINTS[0])
        got = utils.get_world_point_world_frame(
            EYE, [0, 0, 0, 1], "head", IMAGE_SIZE, [px, py, d_gl], cam_info=info)
        np.testing.assert_allclose(np.asarray(got, float), WORLD_POINTS[0], atol=1e-9)

    def test_per_camera_encoding_overrides_top_level(self):
        info = _cam_info(camera_math.DEPTH_OPENGL)
        info["head"]["depth_encoding"] = camera_math.DEPTH_LINEAR_METRIC
        px, py, _, d_m = _project(WORLD_POINTS[1])
        got = utils.get_world_point_world_frame(
            EYE, [0, 0, 0, 1], "head", IMAGE_SIZE, [px, py, d_m], cam_info=info)
        np.testing.assert_allclose(np.asarray(got, float), WORLD_POINTS[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
