"""Per-robot constants that used to be magic numbers inside ``robot.py``.

Keeping them here means a new simulator (or a new arm) supplies data, not code. The
PyBullet values are transcribed verbatim from the pre-refactor ``robot.py`` so the golden
regression traces are unaffected.

``genesis_kp`` / ``genesis_kv`` / ``genesis_force_range`` exist because Genesis has no
implicit position-control gains: ``set_dofs_kp`` / ``set_dofs_kv`` must be called or the
arm simply will not track its target. PyBullet ignores these fields.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import config


@dataclass
class RobotProfile:
    name: str

    urdf: str
    #: Link that the IK solver drives and that the wrist camera rides on. Resolved by
    #: *name* first, because simulator link indices do not agree: Genesis has no
    #: PyBullet-style index and drops fixed-joint links unless asked to keep them.
    #: ``ee_index`` is the PyBullet fallback and the value the goldens were recorded with.
    ee_link_name: str
    ee_index: int

    #: Joint driven to open/close the gripper. Two entries for a two-finger arm whose
    #: fingers are independently actuated (franka); the same index twice for a single
    #: motor (sawyer robotiq).
    gripper_joint_indices: Sequence[int]
    #: Joint read back by ``step_env_and_record`` to detect a gripper keyframe.
    gripper_state_joint: int

    gripper_open_position: float
    gripper_closed_position: float
    gripper_depth_offset: float
    arm_movement_force: float
    gripper_movement_force: float

    #: How many leading entries of ``joint_indices`` the IK result actually drives.
    #: ``None`` means "all of them" (sawyer); franka uses ``-2`` to exclude the fingers.
    arm_joint_count: Optional[int] = None

    #: Name-based equivalent of ``gripper_joint_indices``, in the same order. Used on
    #: simulators that have no PyBullet joint indices.
    gripper_joint_names: Sequence[str] = ()

    # Genesis-only PD control. Seeded from
    # genesis-world/examples/tutorials/IK_motion_planning_grasp.py
    genesis_kp: List[float] = field(default_factory=list)
    genesis_kv: List[float] = field(default_factory=list)
    genesis_force_range: List[float] = field(default_factory=list)

    #: Links that must survive fixed-joint merging (Genesis merges by default).
    links_to_keep: Sequence[str] = ()


def franka_profile():
    return RobotProfile(
        name="franka",
        urdf="franka_robot/panda.urdf",
        # PyBullet link 11. NOT panda_hand: the URDF puts a further fixed joint
        # (panda_grasptarget_hand) between the hand and the actual grasp point, and
        # config.ee_index_franka has always pointed at the latter.
        ee_link_name="panda_grasptarget",
        ee_index=config.ee_index_franka,
        gripper_joint_indices=(9, 10),
        gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        gripper_state_joint=9,
        gripper_open_position=config.gripper_goal_position_open_franka,
        gripper_closed_position=config.gripper_goal_position_closed_franka,
        gripper_depth_offset=config.gripper_depth_offset_franka,
        arm_movement_force=config.arm_movement_force_franka,
        gripper_movement_force=config.gripper_movement_force_franka,
        arm_joint_count=7,
        genesis_kp=[4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        genesis_kv=[450, 450, 350, 350, 200, 200, 200, 10, 10],
        genesis_force_range=[87, 87, 87, 87, 12, 12, 12, 100, 100],
        links_to_keep=("panda_link8", "panda_hand", "panda_grasptarget"),
    )


def sawyer_profile():
    return RobotProfile(
        name="sawyer",
        urdf="sawyer_robot/sawyer_description/urdf/sawyer.urdf",
        ee_link_name="right_hand",
        ee_index=config.ee_index_sawyer,
        gripper_joint_indices=(config.robotiq_motor_joint, config.robotiq_motor_joint),
        gripper_state_joint=config.robotiq_motor_joint,
        gripper_open_position=config.gripper_goal_position_open_sawyer,
        gripper_closed_position=config.gripper_goal_position_closed_sawyer,
        gripper_depth_offset=config.gripper_depth_offset_sawyer,
        arm_movement_force=config.arm_movement_force_sawyer,
        gripper_movement_force=config.gripper_movement_force_sawyer,
        arm_joint_count=None,
    )


_BUILDERS = {"franka": franka_profile, "sawyer": sawyer_profile}


def get_robot_profile(name):
    """Profiles are built on demand: they read mutable ``config`` values that sim-envs
    override in ``configure_robot_pose()``, so a module-level singleton would freeze the
    wrong numbers."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown robot '{name}'. Supported: {sorted(_BUILDERS)}") from None
    return builder()
