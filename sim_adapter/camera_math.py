"""Sim-neutral camera math shared by every provider.

Nothing here imports a simulator. These are the formulas the app layer needs in order to
place a camera and to turn a depth sample back into a world point, expressed once so the
PyBullet and Genesis paths cannot drift apart.
"""

import math

import numpy as np

from sim_adapter.base import DEPTH_LINEAR_METRIC, DEPTH_OPENGL


def spherical_camera_pose(target, distance, yaw_deg, pitch_deg):
    """Return (camera_position, camera_orientation_euler) for PyBullet's
    yaw/pitch spherical camera convention.

    This must match ``p.computeViewMatrixFromYawPitchRoll(target, distance,
    yaw, pitch, 0, upAxisIndex=2)`` and ``p.resetDebugVisualizerCamera``:

        forward = (-cos(pitch) * sin(yaw), cos(pitch) * cos(yaw), sin(pitch))
        camera  = target - distance * forward

    i.e. yaw=0 places the camera on -y looking towards +y, and yaw increases
    clockwise when seen from above. Note this is rotated 90 degrees from the
    naive ``(cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch))`` spherical
    formula, which silently reports a camera position that does not match the
    view matrix actually used for rendering.

    Kept sim-neutral: Genesis has no yaw/pitch camera helper, so its adapter builds the
    eye/target pair from exactly this function and gets a pixel-identical framing.
    """
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    tx, ty, tz = list(map(float, target))
    fx = -math.cos(pitch) * math.sin(yaw)
    fy = math.cos(pitch) * math.cos(yaw)
    fz = math.sin(pitch)
    position = [
        float(tx - distance * fx),
        float(ty - distance * fy),
        float(tz - distance * fz),
    ]
    # Euler orientation of the camera's optical (+z) axis, matching the
    # convention used by the non-spherical head-camera branch in robot.py.
    orientation_e = [0.0, float(pitch), float(math.atan2(fy, fx))]
    return position, orientation_e


def gl_projection_matrix(fov_deg, aspect, near, far):
    """The standard OpenGL perspective matrix, flat 16 column-major.

    Identical in form to ``p.computeProjectionMatrixFOV`` and to Genesis'
    ``camera.projection_matrix`` (verified equal to 0.0 max-abs-diff in
    ``tests/test_genesis_camera_semantics.py``), which is why
    ``utils.get_intrinsics_extrinsics`` needs no per-simulator branch.
    """
    f = 1.0 / math.tan(math.radians(float(fov_deg)) / 2.0)
    near = float(near)
    far = float(far)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / float(aspect)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return [float(v) for v in m.flatten(order="F")]


def gl_view_matrix(eye, target, up):
    """The standard OpenGL look-at matrix, flat 16 column-major.

    Matches ``p.computeViewMatrix`` and ``inv(genesis_camera.transform)``.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return [float(v) for v in m.flatten(order="F")]


def depth_to_metric(depth, encoding, near, far):
    """Convert a raw depth buffer to metres along the optical axis (``z_eye``).

    ``opengl``        - non-linear z-buffer in [0, 1] (PyBullet).
    ``linear_metric`` - already metres (Genesis); returned unchanged.

    The input dtype is preserved (PyBullet hands back float32 and the saved depth PNG
    quantises from it; silently widening to float64 would shift the odd byte).
    """
    depth = np.asarray(depth)
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float64)
    if encoding == DEPTH_LINEAR_METRIC:
        return depth
    if encoding != DEPTH_OPENGL:
        raise ValueError(f"Unknown depth encoding '{encoding}'")
    near = float(near)
    far = float(far)
    return (2.0 * near * far) / (far + near - (2.0 * depth - 1.0) * (far - near))


def metric_to_ndc_z(z_eye, near, far):
    """Metres along the optical axis -> OpenGL clip-space z in [-1, 1].

    This is the bridge that lets ``utils.get_world_point_world_frame`` keep its single
    inverse view-projection code path: a Genesis depth sample is converted here and the
    existing math runs unchanged.
    """
    z_eye = np.asarray(z_eye, dtype=np.float64)
    near = float(near)
    far = float(far)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((far + near) / (far - near)) + (2.0 * far * near) / ((near - far) * z_eye)
