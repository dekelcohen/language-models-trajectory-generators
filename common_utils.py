import math
import os
import shutil
from typing import Iterable

import numpy as np

import config
from config import images_folder, trajectory_folder, overlay_folder, video_folder
from sim_adapter.transforms import ee_euler_from_approach, matrix_from_euler

class Trajectory:
    """An end-effector trajectory, guaranteed to be transport-safe by construction.

    ``points`` is always a list of lists of plain Python floats and ``desc`` a plain ``str``,
    because a Trajectory is sent to the simulator process over ``multiprocessing.Pipe``
    (PyBullet) or JSON (Genesis). Coercion happens in ``__init__`` rather than at the send
    site so there is no way to build one that cannot travel.

    Anything the model defines inside its own code block is exactly what cannot travel: it is
    pickled *by reference* as ``<module>.<name>``, and since the exec namespace is seeded from
    ``agent_runner.globals()`` such a class claims ``__module__ == "agent_runner"`` while not
    being an attribute of it - the send dies with ``PicklingError``. Numpy is stripped for a
    related reason: the JSON transport rejects arrays outright, and pickle's array format is
    version-sensitive across the two interpreters.

    Use :meth:`normalize` for input that may not be a Trajectory yet.
    """

    _HELP = (
        "execute_trajectory() expects the Trajectory returned by generate_linear_trajectory(), "
        "or a plain list of [x, y, z, rotation] poses. {problem} Note that a class YOU define in "
        "a code block cannot be sent to the simulator (it lives in a separate process and cannot "
        "reconstruct your class) - build the motion out of plain lists of floats, or chain several "
        "generate_linear_trajectory() calls, instead of wrapping poses in a custom object."
    )

    def __init__(self, points, desc=""):
        self.points = Trajectory._coerce_points(points) # a straight-line end-effector trajectory between two EE poses
        self.desc = "" if desc is None else desc if isinstance(desc, str) else str(desc) # short sentence to describe the motion and its end_pose

    @staticmethod
    def _coerce_points(points):
        """Every pose as a list of plain Python floats, or ``TypeError`` explaining why not."""
        if isinstance(points, np.ndarray):
            points = points.tolist()
        if isinstance(points, (str, bytes)) or not isinstance(points, Iterable):
            raise TypeError(Trajectory._HELP.format(
                problem=f"Got {type(points).__name__} instead."))

        coerced = []
        for i, pose in enumerate(points):
            if isinstance(pose, np.ndarray):
                pose = pose.tolist()
            if isinstance(pose, (str, bytes)) or not isinstance(pose, Iterable):
                raise TypeError(Trajectory._HELP.format(
                    problem=f"Pose {i} is a {type(pose).__name__}, not a list of numbers."))
            try:
                coerced.append([float(v) for v in pose])
            except (TypeError, ValueError):
                raise TypeError(Trajectory._HELP.format(
                    problem=f"Pose {i} = {pose!r} contains a value that is not a number.")) from None
        return coerced

    @staticmethod
    def normalize(trajectory):
        """Whatever was handed to ``execute_trajectory``, as a real :class:`Trajectory`.

        Accepts a ``Trajectory`` (returned unchanged - the constructor already coerced it),
        any duck-typed object exposing ``.points`` (the common case: the model imitating this
        class to build an arc), or a bare sequence of poses. Raises ``TypeError`` with an
        actionable message otherwise; the model reads it and retries within the same attempt.
        """
        if isinstance(trajectory, Trajectory):
            return trajectory
        if hasattr(trajectory, "points"):
            return Trajectory(trajectory.points, getattr(trajectory, "desc", ""))
        return Trajectory(trajectory)



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


def normalize_grasp_viz_poses(poses):
    """Classify whatever was handed to ``visualize_grasp_pose`` into ``(kind, poses)``.

    ``kind`` is ``"ee"`` for end-effector poses (length 4 or 6 - the poses the trajectory
    pipeline executes) and ``"matrix"`` for the pre-computed GraspGen/GraspNet 4x4
    candidates. ``poses`` is always a plain list, one entry per pose.

    Accepted inputs: a single pose, a list/array of poses, a single ``(4,4)`` matrix, or an
    ``(N,4,4)`` stack.

    The one genuinely ambiguous shape is a bare ``(4,4)`` array - four top-down poses, or
    one homogeneous matrix? It is read as a matrix only when its last row is ``[0,0,0,1]``,
    which a stack of four end-effector poses would essentially never be.
    """
    try:
        arr = np.asarray(poses, dtype=float)
    except (ValueError, TypeError):
        # Ragged input: a list mixing length-4 and length-6 end-effector poses.
        return "ee", [[float(v) for v in pose] for pose in poses]
    if arr.ndim == 3:
        if arr.shape[1:] != (4, 4):
            raise ValueError(f"Expected an (N,4,4) grasp matrix stack; got shape {arr.shape}.")
        return "matrix", [m for m in arr]
    if arr.ndim == 1:
        return "ee", [arr.tolist()]
    if arr.ndim == 2:
        if arr.shape == (4, 4) and np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            return "matrix", [arr]
        if arr.shape[1] in (4, 6):
            return "ee", [row.tolist() for row in arr]
        raise ValueError(
            f"Expected end-effector poses of length 4 or 6, or a 4x4 matrix; got shape {arr.shape}."
        )
    raise ValueError(f"Unsupported grasp pose input of shape {arr.shape}.")


def ee_pose_to_grasp_matrix(pose):
    """An end-effector pose as the 4x4 "grasp frame" the visualiser draws.

    Two frames are in play and they do NOT agree, which is the whole reason this helper
    exists:

    * **End-effector frame** (what ``robot.move`` executes, see
      ``sim_adapter.transforms.matrix_from_approach``): ``+Z`` is the approach axis and
      ``+Y`` is the finger-closing direction.
    * **Grasp frame** (GraspNet/AnyGrasp convention, what ``env.draw_grasp_pose`` draws):
      ``+Z`` is the approach axis and ``+X`` is the finger-closing direction.

    So the columns are re-ordered ``X <- ee_Y``, ``Y <- -ee_X``, ``Z <- ee_Z``, which keeps
    the frame right-handed (``cross(ee_Y, -ee_X) == ee_Z``). Drawing the end-effector
    matrix directly would put the fingers 90 degrees off.

    Accepts either pose length (see the format table at the top of this module), so the
    caller can hand over exactly the pose it is about to execute - top-down
    ``[x, y, z, rotation]`` or the length-6 horizontal-approach pose.

    Returns a 4x4 row-major nested list.
    """
    if len(pose) not in (4, 6):
        raise ValueError(
            "visualize_grasp_pose expects a length-4 ([x,y,z,rotation]) or length-6 "
            "([x,y,z,roll,pitch,yaw]) end-effector pose; got length %d" % len(pose)
        )
    m = matrix_from_euler(pose_euler(pose))
    ee_x = [m[0], m[3], m[6]]
    ee_y = [m[1], m[4], m[7]]
    ee_z = [m[2], m[5], m[8]]
    origin = [float(pose[0]), float(pose[1]), float(pose[2])]
    return [
        [ee_y[r], -ee_x[r], ee_z[r], origin[r]] for r in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


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

