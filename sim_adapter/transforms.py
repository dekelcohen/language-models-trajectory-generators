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


def matrix_from_euler(euler):
    """``[roll, pitch, yaw]`` radians -> row-major flat 9-element rotation matrix.

    Same convention as everything else here: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    return matrix_from_quat(quat_from_euler(euler))


def euler_from_matrix(matrix9):
    """Row-major flat 9-element rotation matrix -> ``[roll, pitch, yaw]`` radians.

    Why this exists next to :func:`euler_from_quat`: that function reproduces Bullet, which
    derives pitch with ``asin`` and therefore **can only ever return pitch in
    [-pi/2, pi/2]**. The end-effector's top-down home orientation is
    ``config.ee_start_orientation_e = [0, pi, -pi/2]``, whose pitch is ``pi`` - a value
    Bullet's converter can never hand back. Round-tripping a top-down pose through
    ``euler_from_quat`` silently rewrites it into an equivalent triple with a different
    pitch, which is fine for the sim but useless when we want the *authored* angles back.

    This converter keeps the branch cut in the same place (``asin`` on ``-R[0,2]``) so it
    agrees with :func:`euler_from_quat` wherever that one is valid, and is only used to
    turn an approach-frame rotation matrix into the triple that ``quat_from_euler``
    consumes. ``quat_from_euler(euler_from_matrix(m))`` always reproduces ``m``.
    """
    m = [float(v) for v in matrix9]
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  =>  R[2,0] = -sin(pitch)
    sarg = -m[6]
    if sarg <= -0.99999 or sarg >= 0.99999:
        # Gimbal lock: roll and yaw rotate about the same axis. Collapse roll into yaw,
        # matching euler_from_quat's behaviour.
        pitch = -0.5 * math.pi if sarg < 0 else 0.5 * math.pi
        return [0.0, pitch, math.atan2(-m[1], m[4])]
    return [
        math.atan2(m[7], m[8]),   # atan2(R[2,1], R[2,2])
        math.asin(sarg),
        math.atan2(m[3], m[0]),   # atan2(R[1,0], R[0,0])
    ]


def _mat_mul(a, b):
    """Multiply two row-major flat 9-element matrices."""
    return [
        sum(a[r * 3 + k] * b[k * 3 + c] for k in range(3))
        for r in range(3) for c in range(3)
    ]


def _rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]


def matrix_from_approach(tilt, azimuth, roll):
    """Approach-frame parametrisation of an end-effector orientation.

    ``R = Rz(azimuth) @ Ry(pi - tilt) @ Rz(roll)``, row-major flat 9-vector.

    The gripper's **approach axis** is the end-effector +Z (the direction the hand points,
    i.e. from the wrist toward the object), which is the third column of ``R``::

        approach = [sin(tilt)cos(azimuth), sin(tilt)sin(azimuth), -cos(tilt)]

    * ``tilt = 0``   -> approach is straight down ``[0, 0, -1]``: the classic top-down grasp.
    * ``tilt = pi/2``-> approach is horizontal at ``azimuth``: the side grasp needed for
      vertical bar handles, which have no graspable top face.

    ``roll`` spins the hand about its own approach axis and so decides where the fingers
    sit. The fingers close along the end-effector +Y (second column of ``R``); at
    ``roll = 0`` with ``tilt = pi/2`` that is ``[-sin(azimuth), cos(azimuth), 0]`` - purely
    horizontal and perpendicular to the approach - which is exactly how you pinch a
    vertical bar.

    At ``tilt = 0`` the parametrisation is gimbal-degenerate (``azimuth`` and ``roll`` both
    spin about world Z). :func:`ee_euler_from_approach` pins ``roll = 0`` there so the
    top-down case reproduces the legacy pose bit for bit.
    """
    return _mat_mul(_mat_mul(_rot_z(azimuth), _rot_y(math.pi - tilt)), _rot_z(roll))


def ee_euler_from_approach(tilt, azimuth, roll=0.0):
    """:func:`matrix_from_approach` as a ``[roll, pitch, yaw]`` triple for ``robot.move``.

    The whole point of routing through a matrix rather than composing Euler angles is that
    the intermediate orientations are nowhere near representable in a single XYZ triple
    without care; the matrix has no such ambiguity.
    """
    return euler_from_matrix(matrix_from_approach(tilt, azimuth, roll))


def approach_axis_from_euler(euler_xyz):
    """World direction the gripper points in: the end-effector frame's +Z column.

    This is the axis a grasp closes *along the way in*, so it is what any depth/standoff
    offset has to be measured against. Exactly ``[0, 0, -1]`` for the top-down poses.
    """
    m = matrix_from_euler(euler_xyz)
    return [m[2], m[5], m[8]]


def slerp(quat_a_xyzw, quat_b_xyzw, t):
    """Shortest-arc spherical interpolation between two ``[x, y, z, w]`` quaternions.

    Interpolating Euler triples component-wise is fine for the legacy top-down poses, where
    only yaw ever changes, but produces garbage the moment two poses differ in more than
    one axis (it can swing the gripper through orientations neither endpoint asked for).
    Trajectories mixing top-down and side approaches do exactly that, so orientation is
    interpolated here instead.
    """
    a = np.asarray(quat_a_xyzw, dtype=np.float64)
    b = np.asarray(quat_b_xyzw, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)

    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the short way round; q and -q are the same rotation
        b = -b
        dot = -dot

    t = float(t)
    if dot > 0.9995:  # almost parallel: lerp is numerically safer than slerp here
        out = a + (b - a) * t
        return list(out / max(float(np.linalg.norm(out)), 1e-12))

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s_a = math.sin(theta_0 - theta) / sin_theta_0
    s_b = math.sin(theta) / sin_theta_0
    return list(a * s_a + b * s_b)


def quat_angle_between(quat_a_xyzw, quat_b_xyzw):
    """Absolute angle in radians between two orientations, ignoring quaternion sign.

    ``q`` and ``-q`` denote the same rotation, hence the ``abs`` on the dot product.
    """
    a = np.asarray(quat_a_xyzw, dtype=np.float64)
    b = np.asarray(quat_b_xyzw, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


# --- Quaternion layout: the single most common Genesis porting bug ------------

def xyzw_to_wxyz(quat_xyzw):
    """PyBullet/IPC layout -> Genesis layout."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return [w, x, y, z]


def wxyz_to_xyzw(quat_wxyz):
    """Genesis layout -> PyBullet/IPC layout."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return [x, y, z, w]
