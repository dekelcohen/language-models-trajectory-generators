import math
import os
import shutil
from typing import Iterable

import config
from config import images_folder, trajectory_folder, overlay_folder, video_folder
from sim_adapter.transforms import ee_euler_from_approach

class Trajectory:
    def __init__(self, points, desc):
        self.points = points # a straight-line end-effector trajectory between two EE poses
        self.desc = desc # short sentence to describe the motion and its end_pose


# --- End-effector pose formats ------------------------------------------------
#
# Two pose lengths travel through the trajectory pipeline, and the length IS the
# discriminator - there is no mode flag anywhere:
#
#   len 4: [x, y, z, rotation]              top-down. The historical format. Orientation is
#                                           config.ee_start_orientation_e + [0, 0, rotation],
#                                           i.e. the gripper always points straight down and
#                                           only spins about world Z.
#   len 6: [x, y, z, roll, pitch, yaw]      arbitrary orientation, absolute Euler radians in
#                                           the repo-wide PyBullet convention.
#
# The len-4 path is deliberately left byte-identical to what it always was, so adding the
# side approach cannot perturb any existing top-down task. LLM-generated code is expected to
# build poses through grasp_pose()/side_grasp_pose() rather than writing Euler angles by
# hand: the choice is then a named binary one ("which builder?") instead of six free numbers.

SIDE_APPROACH_TILT = math.pi / 2


def grasp_pose(x, y, z, rotation=0.0):
    """Top-down end-effector pose - the default for essentially every tabletop grasp.

    ``rotation`` is the existing rotation value: the direction of the gripper's closing
    motion, in radians about the world Z axis. Returns the classic length-4 pose, so this
    is exactly equivalent to writing ``[x, y, z, rotation]`` by hand.
    """
    return [float(x), float(y), float(z), float(rotation)]


def side_grasp_pose(x, y, z, rotation, approach_yaw):
    """Horizontal-approach end-effector pose, for targets with no graspable top face.

    Use for **vertical** bar handles (fridge/cabinet/room doors), where a top-down gripper
    would have to close on the handle's tiny end cap.

    Args:
        x, y, z: end-effector target position, metres.
        approach_yaw: azimuth in radians of the horizontal direction the gripper points
            **toward the target** (``atan2(dy, dx)`` from gripper to handle). For a door,
            this is the inward normal of the door face.
        rotation: roll about that approach axis, radians. ``0`` closes the fingers
            horizontally - the correct pinch for a **vertical** bar. ``pi/2`` closes them
            vertically, for a horizontal bar approached from the side.

    Returns a length-6 ``[x, y, z, roll, pitch, yaw]`` pose.
    """
    euler = ee_euler_from_approach(SIDE_APPROACH_TILT, float(approach_yaw), float(rotation))
    return [float(x), float(y), float(z), euler[0], euler[1], euler[2]]


def top_down_euler(rotation):
    """The orientation a length-4 pose stands for, as ``robot.move`` consumes it."""
    start = config.ee_start_orientation_e
    return [float(start[0]), float(start[1]), float(start[2]) + float(rotation)]


def pose_euler(pose):
    """Orientation of a length-4 **or** length-6 pose as ``[roll, pitch, yaw]``."""
    if len(pose) == 4:
        return top_down_euler(pose[3])
    if len(pose) == 6:
        return [float(pose[3]), float(pose[4]), float(pose[5])]
    raise ValueError(
        "End-effector pose must have length 4 ([x,y,z,rotation], top-down) or 6 "
        "([x,y,z,roll,pitch,yaw]); got length %d" % len(pose)
    )


def ensure_image_dirs_exist(delete: bool = False, extra_dirs: Iterable[str] | None = None) -> None:
    """Ensure image directories exist; optionally delete existing contents first.

    Uses the same directory set as tests and runtime:
    - ./images
    - ./images/trajectory
    - ./images/overlay
    - ./images/videos

    If delete is True, removes these directories (if present) before recreating.
    Extra directories can be provided via extra_dirs.
    """
    dirs = [
        images_folder,
        trajectory_folder,
        overlay_folder,
        video_folder,
    ]
    if extra_dirs:
        for d in extra_dirs:
            if d not in dirs:
                dirs.append(d)

    if delete:
        for d in dirs:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d)
            except Exception:
                # Best effort cleanup; leave to creation phase
                pass

    for d in dirs:
        os.makedirs(d, exist_ok=True)

