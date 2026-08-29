"""Rotation conversions in **PyBullet's** convention, implemented in pure numpy.

Why not delegate to each simulator's own helpers? Because ``env.py`` and ``robot.py`` do
real geometry with these results — the wrist camera's "drone" framing rotates
``[0,0,1]`` and ``[-1,0,0]`` by ``matrix_from_quat``, and every IK target is built from
``quat_from_euler``. If two providers disagreed about Euler order or handedness by so much
as a sign, the two simulators would silently frame different pictures and nothing would
crash. Genesis' own ``geom`` helpers take **degrees** and are not guaranteed to use the
same axis order, so the Genesis adapter uses these instead.

Convention (identical to ``pybullet``):

* Quaternions are ``[x, y, z, w]``.
* Euler is ``[roll, pitch, yaw]`` in **radians**, applied as extrinsic X-Y-Z
  (equivalently intrinsic Z-Y'-X'').
* Rotation matrices are returned row-major as a flat 9-vector, like
  ``p.getMatrixFromQuaternion``.

``tests/test_transforms.py`` asserts every function here against pybullet itself, so the
Genesis path is held to the PyBullet numbers rather than to my arithmetic.
"""

import math

import numpy as np


def quat_from_euler(euler):
    """``[roll, pitch, yaw]`` radians -> ``[x, y, z, w]``."""
    roll, pitch, yaw = (float(v) for v in euler)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def euler_from_quat(quat_xyzw):
    """``[x, y, z, w]`` -> ``[roll, pitch, yaw]`` radians.

    This reproduces Bullet's ``b3ComputeEulerFromQuaternion`` exactly, including its
    ``0.99999`` gimbal-lock threshold and the branch that collapses roll into yaw. A
    textbook ``asin`` implementation differs from PyBullet by ~1e-6 near the poles, and
    ``config.head_camera_orientation_e`` sits close enough to a pole for that to matter.
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    sqx, sqy, sqz, sqw = x * x, y * y, z * z, w * w

    sarg = -2.0 * (x * z - w * y)
    if sarg <= -0.99999:
        return [0.0, -0.5 * math.pi, 2.0 * math.atan2(x, -y)]
    if sarg >= 0.99999:
        return [0.0, 0.5 * math.pi, 2.0 * math.atan2(-x, y)]
    return [
        math.atan2(2.0 * (y * z + w * x), sqw - sqx - sqy + sqz),
        math.asin(sarg),
        math.atan2(2.0 * (x * y + w * z), sqw + sqx - sqy - sqz),
    ]


def matrix_from_quat(quat_xyzw):
    """``[x, y, z, w]`` -> row-major flat 9-element rotation matrix."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n < 1e-12 else 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return [
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy),
    ]


def quat_from_axis_angle(axis, angle):
    """Unit ``axis`` + ``angle`` radians -> ``[x, y, z, w]``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis = axis / norm
    half = float(angle) * 0.5
    s = math.sin(half)
    return [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s), math.cos(half)]


# --- Quaternion layout: the single most common Genesis porting bug ------------

def xyzw_to_wxyz(quat_xyzw):
    """PyBullet/IPC layout -> Genesis layout."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return [w, x, y, z]


def wxyz_to_xyzw(quat_wxyz):
    """Genesis layout -> PyBullet/IPC layout."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return [x, y, z, w]
