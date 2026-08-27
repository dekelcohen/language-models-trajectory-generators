"""Small PyBullet lookup helpers shared by sim-env profiles."""

import math
import traceback

import pybullet as p


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


def get_joint_index_by_name(body_id, joint_name):
    try:
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
    except Exception as e:
        print(f"[Env] Error reading joints of body {body_id}:", e)
        traceback.print_exc()
    return None


def get_link_index_by_name(body_id, link_name):
    """Return the link index (same as the joint index in PyBullet)
    by matching the child link name from getJointInfo(...)[12].

    Example: for a URDF joint named 'latch_joint' with child link 'latch',
    this helper returns the index to use with p.getLinkState(...).
    """
    try:
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            # info[12] is the child link name (bytes)
            child_link_name = info[12].decode("utf-8") if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if child_link_name == link_name:
                return j
    except Exception as e:
        print(f"[Env] Error reading links of body {body_id}:", e)
        traceback.print_exc()
    return None


def list_joint_names(body_id):
    """Return {joint_name: joint_index} for every joint of ``body_id``.

    Useful when a scene URDF must be wired up by name instead of by the
    brittle hard-coded indices used by upstream asset repositories.
    """
    names = {}
    try:
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            names[name] = j
    except Exception as e:
        print(f"[Env] Error listing joints of body {body_id}:", e)
        traceback.print_exc()
    return names


def list_link_names(body_id):
    """Return {child_link_name: link_index} for every link of ``body_id``."""
    names = {}
    try:
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            name = info[12].decode("utf-8") if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            names[name] = j
    except Exception as e:
        print(f"[Env] Error listing links of body {body_id}:", e)
        traceback.print_exc()
    return names
