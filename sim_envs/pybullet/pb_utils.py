"""Small PyBullet lookup helpers shared by sim-env profiles.

Note: ``spherical_camera_pose`` moved to :mod:`sim_adapter.camera_math` (it is pure math
and both simulators need it); it is re-exported here so existing imports keep working.
The name lookups below remain for ``franka_kitchen``, which still talks to PyBullet
directly. New code should use ``SimAdapter.get_joint_index_by_name`` / ``list_joint_names``.
"""

import traceback

import pybullet as p

from sim_adapter.camera_math import spherical_camera_pose  # noqa: F401  (re-export)


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
