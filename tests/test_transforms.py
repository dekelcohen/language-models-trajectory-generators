"""Asserts ``sim_adapter.transforms`` against pybullet itself.

The Genesis adapter uses these pure-numpy conversions instead of Genesis' own ``geom``
helpers (which take degrees). That is only safe if they agree with PyBullet bit-for-bit
in meaning, because ``robot.py`` builds camera framing and IK targets out of them. This
runs under ``vlm_traj`` and skips where pybullet is unavailable.
"""

import math
import unittest

import numpy as np

from sim_adapter import transforms

try:
    import pybullet as p
    PYBULLET_ERROR = None
except Exception as exc:  # pragma: no cover
    p = None
    PYBULLET_ERROR = exc

# Deliberately awkward angles: signs, wraps, near-gimbal-lock and the exact values
# config.py uses for the head camera.
EULERS = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 4.0],
    [math.pi, 0.0, 0.0],
    [0.1422, 0.0, 0.1975],
    [-0.3, 1.2, 2.9],
    [1.7, -0.4, -2.2],
    [0.0, math.pi / 2 - 1e-6, 0.0],
    [-math.pi / 2, math.pi / 4, math.pi],
    [2.5, 0.7, -1.1],
]

TOL = 1e-12


@unittest.skipIf(p is None, f"pybullet unavailable ({PYBULLET_ERROR})")
class TestTransformsMatchPyBullet(unittest.TestCase):
    def test_quat_from_euler(self):
        for euler in EULERS:
            with self.subTest(euler=euler):
                np.testing.assert_allclose(
                    transforms.quat_from_euler(euler),
                    p.getQuaternionFromEuler(euler),
                    atol=TOL, rtol=0,
                )

    def test_euler_from_quat(self):
        for euler in EULERS:
            quat = p.getQuaternionFromEuler(euler)
            with self.subTest(euler=euler):
                np.testing.assert_allclose(
                    transforms.euler_from_quat(quat),
                    p.getEulerFromQuaternion(quat),
                    atol=1e-9, rtol=0,
                )

    def test_matrix_from_quat(self):
        for euler in EULERS:
            quat = p.getQuaternionFromEuler(euler)
            with self.subTest(euler=euler):
                np.testing.assert_allclose(
                    transforms.matrix_from_quat(quat),
                    p.getMatrixFromQuaternion(quat),
                    atol=TOL, rtol=0,
                )

    def test_quat_from_axis_angle(self):
        cases = [
            ([1, 0, 0], math.pi),
            ([0, 0, 1], -0.75),
            ([0.3, -0.5, 0.81], 2.4),
            ([1, 1, 1], 0.0),
        ]
        for axis, angle in cases:
            axis_unit = (np.asarray(axis, dtype=float) / np.linalg.norm(axis)).tolist()
            with self.subTest(axis=axis, angle=angle):
                # pybullet computes this one in single precision (~5e-8 error), unlike
                # getQuaternionFromEuler. It is only used for cosmetic debug-marker
                # orientation, so float32 agreement is the right bar.
                np.testing.assert_allclose(
                    transforms.quat_from_axis_angle(axis_unit, angle),
                    p.getQuaternionFromAxisAngle(axis_unit, angle),
                    atol=1e-6, rtol=0,
                )

    def test_matrix_rotates_the_camera_basis_the_same_way(self):
        """The check that actually matters: robot.py rotates these two vectors."""
        for euler in EULERS:
            quat = p.getQuaternionFromEuler(euler)
            mine = np.array(transforms.matrix_from_quat(quat)).reshape(3, 3)
            theirs = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            for vec in ([0, 0, 1], [-1, 0, 0]):
                with self.subTest(euler=euler, vec=vec):
                    np.testing.assert_allclose(mine.dot(vec), theirs.dot(vec),
                                               atol=TOL, rtol=0)


class TestQuaternionLayout(unittest.TestCase):
    """No pybullet needed: pure layout bookkeeping, but it is the #1 Genesis bug."""

    def test_roundtrip(self):
        q = [0.1, 0.2, 0.3, 0.927]
        self.assertEqual(transforms.wxyz_to_xyzw(transforms.xyzw_to_wxyz(q)), q)

    def test_ordering(self):
        self.assertEqual(transforms.xyzw_to_wxyz([1, 2, 3, 4]), [4.0, 1.0, 2.0, 3.0])
        self.assertEqual(transforms.wxyz_to_xyzw([4, 1, 2, 3]), [1.0, 2.0, 3.0, 4.0])


class TestNumericalEdges(unittest.TestCase):
    def test_zero_axis_returns_identity(self):
        self.assertEqual(transforms.quat_from_axis_angle([0, 0, 0], 1.0), [0.0, 0.0, 0.0, 1.0])

    def test_euler_from_quat_survives_gimbal_lock(self):
        # sinp rounds marginally past 1.0 here; asin() would raise.
        roll, pitch, yaw = transforms.euler_from_quat(
            transforms.quat_from_euler([0.0, math.pi / 2, 0.0]))
        self.assertAlmostEqual(pitch, math.pi / 2, places=6)

    def test_euler_quat_roundtrip(self):
        for euler in EULERS:
            if abs(abs(euler[1]) - math.pi / 2) < 1e-3:
                # Bullet's 0.99999 gimbal branch deliberately snaps pitch to +-pi/2 and
                # folds roll into yaw, so the round trip is lossy there by design.
                continue
            with self.subTest(euler=euler):
                back = transforms.euler_from_quat(transforms.quat_from_euler(euler))
                # Compare via the quaternion, and only up to sign: Euler triples are not
                # unique and q and -q are the same rotation.
                got = np.array(transforms.quat_from_euler(back))
                want = np.array(transforms.quat_from_euler(euler))
                delta = min(np.abs(got - want).max(), np.abs(got + want).max())
                self.assertLess(delta, 1e-9)

    def test_gimbal_branch_matches_pybullet(self):
        """The lossy pole branch must be lossy in exactly PyBullet's way."""
        if p is None:
            self.skipTest("pybullet unavailable")
        for pitch in (math.pi / 2 - 1e-7, -math.pi / 2 + 1e-7):
            quat = p.getQuaternionFromEuler([0.0, pitch, 0.0])
            with self.subTest(pitch=pitch):
                np.testing.assert_allclose(
                    transforms.euler_from_quat(quat),
                    p.getEulerFromQuaternion(quat),
                    atol=1e-9, rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
