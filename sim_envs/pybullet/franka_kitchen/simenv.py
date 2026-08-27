"""Franka Kitchen sim-env profile (PyBullet).

Scene assets were adapted from https://github.com/kwonathan/franka-kitchen-pybullet
(itself derived from D4RL / relay-policy-learning MuJoCo models) and copied into
``my_assets/franka_kitchen``. See ``my_assets/franka_kitchen/LICENSE.md``.

The upstream repository ships **no physics setup**: every movable joint declares
``effort="0"`` (so PyBullet's default motors apply zero force) with only
``damping="1.0"``, and no ``changeDynamics`` call is made, leaving handles at the
default ``lateralFriction=0.5``. All friction / resting-motor tuning here is ours.

Upstream ``updateFrankaKitchen`` is replicated in :meth:`SimEnvKitchen.step_hook`,
which must run once per simulation step, and is wired by name rather than by the
hard-coded link indices used upstream.
"""

import math
import traceback

import numpy as np
import pybullet as p

import config
from sim_envs.pybullet.base import SimEnvBase
from sim_envs.pybullet.pb_utils import list_joint_names, list_link_names, spherical_camera_pose
from sim_envs.pybullet.franka_kitchen.tasks import (
    HANDLE_LINKS,
    JOINT_DAMPING,
    MOVABLE_LINK_MASS,
    KETTLE_MASS,
    HEAD_CAMERA_YAW,
    KITCHEN_GLOBAL_SCALING,
    KITCHEN_TASKS,
    KNOB_TO_BURNER,
    LIGHT_BLOCK_JOINT,
    LIGHT_LINK,
    LIGHT_SWITCH_JOINT,
    RESTING_FRICTION_FORCE,
    SUCCESS_THRESHOLD,
    unscaled,
    POSITION_SUCCESS_THRESHOLD,
)

KITCHEN_URDF = "my_assets/franka_kitchen/kitchen_env_model.urdf"
KETTLE_URDF = "my_assets/franka_kitchen/item_assets/kettle.urdf"
TEXTURE_DIR = "my_assets/franka_kitchen/textures"

KITCHEN_START_POSITION = [0.6, 0.1, 0.0]
KITCHEN_START_ORIENTATION_E = [0.0, 0.0, -math.pi / 2]
KETTLE_START_POSITION = [0.625, -0.15, 1.25]

# The kettle is re-seated on this burner once the kitchen links are resolved, so
# that it starts in front of the arm (unoccluded in the head camera) and has a
# non-trivial distance to travel to its goal burner.
KETTLE_START_BURNER = "Burner 2_link"
KETTLE_BURNER_Z_OFFSET = 0.055

# The Panda is bolted to the kitchen counter, not the floor.
ROBOT_BASE_POSITION = [0.0, 0.0, 1.25]
ROBOT_BASE_ORIENTATION_E = [0.0, 0.0, 0.0]
ROBOT_JOINT_START_POSITIONS = [0.0, 0.0, -0.35, -2.55, 0.0, 1.8675, 0.0, 0.04, 0.04]

# Home pose for the end-effector, above the hob and clear of both the counter
# (z ~1.18) and the wall cabinets (z ~1.8). This matters beyond the initial
# posture: RESET_EEF re-homes here between subtasks, and EXECUTE_TRAJECTORY
# expresses every trajectory orientation relative to ROBOT_EE_START_ORIENTATION_E.
# The repo default of [0.0, 0.6, 0.55] is below the counter for this scene and
# would drive the arm straight through it.
ROBOT_EE_START_POSITION = [0.35, 0.0, 1.55]
ROBOT_EE_START_ORIENTATION_E = [0.0, math.pi, -math.pi / 2]

# Links that get the marble / metal textures upstream applies by index.
MARBLE_LINKS = ["countersroot_2"]
METAL_LINKS = [
    "countersroot_3", "ovenroot_1",
    "knob 1_1", "knob 2_1", "knob 3_1", "knob 4_1",
    "lightswitchbaseroot", "lightswitchroot",
    "slidelink_1",
    "hingeleftdoor_1", "hingerightdoor_1",
    "microdoorroot_1", "microroot_2",
]

LIGHT_ON_COLOR = [2.0, 2.0, 2.0, 1.0]
LIGHT_OFF_COLOR = [0.1, 0.1, 0.1, 1.0]

KITCHEN_COORDS_PROMPT_SECTION = (
    "The 3D coordinate system of the environment is as follows:\n"
    "  1. The x-axis is in the depth direction, increasing away from you.\n"
    "  2. The y-axis is in the horizontal direction, increasing to the left.\n"
    "  3. The z-axis is in the vertical direction, increasing upwards."
)


class SimEnvKitchen(SimEnvBase):
    """Franka Kitchen scene, one task per ``franka_kitchen:<task_id>``."""

    def __init__(self, task_id):
        self.task_id = task_id
        self.task = KITCHEN_TASKS[task_id]

        self.kitchen_id = None
        self.kettle_id = None
        self.joints = {}   # joint name -> index
        self.links = {}    # child link name -> index
        self._light_on = None

        self.configure_cameras()

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    def required_robot(self):
        # Only the Panda has a start pose / reach envelope for this scene.
        return "franka"

    def configure_robot_pose(self):
        config.base_start_position_franka = list(ROBOT_BASE_POSITION)
        config.base_start_orientation_e_franka = list(ROBOT_BASE_ORIENTATION_E)
        config.joint_start_positions_franka = list(ROBOT_JOINT_START_POSITIONS)
        config.ee_start_position = list(ROBOT_EE_START_POSITION)
        config.ee_start_orientation_e = list(ROBOT_EE_START_ORIENTATION_E)

    def move_to_start_pos(self):
        # ROBOT_EE_START_POSITION is IK-reachable and contact-free in this scene,
        # so homing at startup is safe and keeps the initial pose consistent with
        # where RESET_EEF returns the arm.
        return True

    def get_ee_start_pose(self):
        return list(ROBOT_EE_START_POSITION), list(ROBOT_EE_START_ORIENTATION_E)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    def configure_cameras(self):
        """Frame the head camera on this task's target, axis-aligned.

        Yaw is pinned to HEAD_CAMERA_YAW (camera on -x looking towards +x) so
        that image-right maps exactly to -y and image-depth to +x, which is what
        ``get_3d_coordinates_prompt_section`` tells the model.
        """
        target = self._camera_target()
        config.camera_distance = float(self.task.camera_distance)
        config.camera_yaw = float(HEAD_CAMERA_YAW)
        config.camera_pitch = float(self.task.camera_pitch)
        config.camera_target_position = target

        pos, ori_e = self._spherical_camera_pose(
            config.camera_distance, config.camera_yaw, config.camera_pitch, target
        )
        config.head_camera_position = pos
        config.head_camera_orientation_e = ori_e

        try:
            is_gui = p.isConnected() and p.getConnectionInfo()[1] == p.GUI
        except Exception:
            is_gui = False
        config.head_camera_use_debug_view = bool(is_gui)
        config.head_camera_use_spherical_view = not bool(is_gui)

    def _camera_target(self):
        """Head-camera look-at point.

        Before the scene is loaded the link position is unknown, so a static
        estimate from the task table is used; ``load_assets`` re-runs this with
        the real link position.
        """
        base = self._target_link_position()
        if base is None:
            base = [0.5, 0.1, 1.5]
        off = self.task.camera_target_offset
        return [float(base[0] + off[0]), float(base[1] + off[1]), float(base[2] + off[2])]

    @staticmethod
    def _spherical_camera_pose(distance, yaw_deg, pitch_deg, target):
        """Camera world position + euler orientation in PyBullet's yaw/pitch
        convention (see pb_utils.spherical_camera_pose)."""
        return spherical_camera_pose(target, distance, yaw_deg, pitch_deg)

    def get_wrist_camera_params(self):
        # The kitchen is a tight scene: pull back less than the door task or the
        # camera ends up inside a cabinet.
        return {"pullback": 0.30, "up_shift": -0.10, "lateral_shift": 0.20}

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------
    def apply(self, env):
        return

    def load_assets(self, env):
        try:
            orientation_q = p.getQuaternionFromEuler(KITCHEN_START_ORIENTATION_E)
            self.kitchen_id = p.loadURDF(
                KITCHEN_URDF,
                KITCHEN_START_POSITION,
                orientation_q,
                useFixedBase=True,
                globalScaling=KITCHEN_GLOBAL_SCALING,
            )
            self.kettle_id = p.loadURDF(
                KETTLE_URDF,
                KETTLE_START_POSITION,
                orientation_q,
                useFixedBase=False,
                globalScaling=KITCHEN_GLOBAL_SCALING,
            )
            self.joints = list_joint_names(self.kitchen_id)
            self.links = list_link_names(self.kitchen_id)
            self._reset_kettle_onto_start_burner()
            self._apply_textures()
            # Now that the scene exists, re-frame the camera on the real link pose.
            self.configure_cameras()
        except Exception as e:
            print("[Env] Failed to load Franka Kitchen assets:", e)
            traceback.print_exc()

    def _reset_kettle_onto_start_burner(self):
        """Place the kettle on the burner named by ``KETTLE_START_BURNER``.

        The upstream port hardcodes a world position; deriving it from the burner
        link instead keeps the kettle seated if the kitchen pose or
        ``globalScaling`` ever changes.
        """
        if self.kettle_id is None or self.kitchen_id is None:
            return
        idx = self.links.get(KETTLE_START_BURNER)
        if idx is None:
            return
        try:
            pos = p.getLinkState(self.kitchen_id, int(idx), computeForwardKinematics=True)[0]
            _, ori = p.getBasePositionAndOrientation(self.kettle_id)
            p.resetBasePositionAndOrientation(
                self.kettle_id,
                [float(pos[0]), float(pos[1]), float(pos[2]) + KETTLE_BURNER_Z_OFFSET],
                ori,
            )
            p.resetBaseVelocity(self.kettle_id, [0, 0, 0], [0, 0, 0])
        except Exception as e:
            print("[Env] Warning: failed to seat the kettle on its start burner:", e)

    def _apply_textures(self):
        try:
            marble = p.loadTexture(f"{TEXTURE_DIR}/marble1.png")
            metal = p.loadTexture(f"{TEXTURE_DIR}/metal1.png")
        except Exception as e:
            print("[Env] Warning: kitchen textures failed to load:", e)
            return
        for link_name in MARBLE_LINKS:
            self._set_texture(link_name, marble)
        for link_name in METAL_LINKS:
            self._set_texture(link_name, metal)
        # Upstream note: wood1.png does not load, so the kettle also gets metal.
        try:
            if self.kettle_id is not None:
                p.changeVisualShape(self.kettle_id, 0, textureUniqueId=metal)
        except Exception:
            pass

    def _set_texture(self, link_name, texture_id):
        idx = self.links.get(link_name)
        if idx is None:
            print(f"[Env] Warning: kitchen link '{link_name}' not found for texturing")
            return
        try:
            p.changeVisualShape(self.kitchen_id, idx, textureUniqueId=texture_id)
        except Exception as e:
            print(f"[Env] Warning: changeVisualShape failed for '{link_name}':", e)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def tune_physics(self):
        """Add the friction/motor setup the upstream assets lack.

        Without this, doors swing freely (effort="0" => zero-force default
        motors) and the gripper slides off every handle (lateralFriction 0.5).
        """
        if self.kitchen_id is None:
            return
        for joint_name, force in RESTING_FRICTION_FORCE.items():
            idx = self.joints.get(joint_name)
            if idx is None:
                print(f"[Env] Warning: kitchen joint '{joint_name}' not found for physics tuning")
                continue
            try:
                # Resting friction: holds the pose, still yields to the arm.
                p.setJointMotorControl2(
                    self.kitchen_id, idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=float(force),
                )
            except Exception as e:
                print(f"[Env] Warning: motor setup failed for '{joint_name}':", e)

        for joint_name, damping in JOINT_DAMPING.items():
            idx = self.joints.get(joint_name)
            if idx is None:
                continue
            try:
                p.changeDynamics(self.kitchen_id, idx, jointDamping=float(damping))
            except Exception as e:
                print(f"[Env] Warning: joint damping setup failed for '{joint_name}':", e)

        for link_name, mass in MOVABLE_LINK_MASS.items():
            idx = self.links.get(link_name)
            if idx is None:
                print(f"[Env] Warning: kitchen link '{link_name}' not found for mass tuning")
                continue
            try:
                p.changeDynamics(self.kitchen_id, idx, mass=float(mass))
            except Exception as e:
                print(f"[Env] Warning: mass tuning failed for '{link_name}':", e)

        for link_name in HANDLE_LINKS:
            idx = self.links.get(link_name)
            if idx is None:
                continue
            try:
                p.changeDynamics(self.kitchen_id, idx, lateralFriction=2.0, spinningFriction=1.0)
            except Exception as e:
                print(f"[Env] Warning: changeDynamics failed for '{link_name}':", e)

        if self.kettle_id is not None:
            try:
                for link_idx in range(-1, p.getNumJoints(self.kettle_id)):
                    p.changeDynamics(
                        self.kettle_id, link_idx,
                        mass=KETTLE_MASS,
                        lateralFriction=1.5,
                        spinningFriction=0.5,
                    )
            except Exception as e:
                print("[Env] Warning: kettle changeDynamics failed:", e)

    # ------------------------------------------------------------------
    # Per-step coupling (upstream updateFrankaKitchen)
    # ------------------------------------------------------------------
    def step_hook(self):
        if self.kitchen_id is None:
            return
        self._update_burners()
        self._update_light()

    def _update_burners(self):
        """Knob angle past the halfway point pushes its burner plate down.

        Cosmetic coupling only; PyBullet has no notion of the MuJoCo tendon
        that links them.
        """
        for knob_name, burner_name in KNOB_TO_BURNER.items():
            knob_idx = self.joints.get(knob_name)
            burner_idx = self.joints.get(burner_name)
            if knob_idx is None or burner_idx is None:
                continue
            knob_angle = p.getJointState(self.kitchen_id, knob_idx)[0]
            knob_lower = p.getJointInfo(self.kitchen_id, knob_idx)[8]
            burner_lower = p.getJointInfo(self.kitchen_id, burner_idx)[8]
            if knob_lower <= knob_angle < knob_lower / 2.0:
                target = burner_lower / 2.0
            elif knob_lower / 2.0 <= knob_angle < 0.0:
                target = 0.0
            else:
                continue
            p.setJointMotorControl2(self.kitchen_id, burner_idx, p.POSITION_CONTROL, targetPosition=target)

    def _update_light(self):
        switch_idx = self.joints.get(LIGHT_SWITCH_JOINT)
        block_idx = self.joints.get(LIGHT_BLOCK_JOINT)
        light_link_idx = self.links.get(LIGHT_LINK)
        if switch_idx is None or block_idx is None:
            return
        angle = p.getJointState(self.kitchen_id, switch_idx)[0]
        switch_lower = p.getJointInfo(self.kitchen_id, switch_idx)[8]
        block_lower = p.getJointInfo(self.kitchen_id, block_idx)[8]
        if switch_lower <= angle < switch_lower / 2.0:
            on, target = True, block_lower / 2.0
        elif switch_lower / 2.0 <= angle < 0.0:
            on, target = False, 0.0
        else:
            return
        p.setJointMotorControl2(self.kitchen_id, block_idx, p.POSITION_CONTROL, targetPosition=target)
        # changeVisualShape is expensive; only call it when the state flips.
        if on != self._light_on and light_link_idx is not None:
            p.changeVisualShape(
                self.kitchen_id, light_link_idx,
                rgbaColor=LIGHT_ON_COLOR if on else LIGHT_OFF_COLOR,
            )
            self._light_on = on

    # ------------------------------------------------------------------
    # State / success
    # ------------------------------------------------------------------
    def _target_link_position(self):
        """World position of the manipulation target.

        Uses the link's AABB centre rather than its frame origin: many kitchen
        links (e.g. ``slidelink_1``, the slide-cabinet handle) have an origin
        that sits nowhere near the mesh they carry, so the origin is useless
        both for framing the camera and as a grasp target.

        For the kettle task the target is the free-floating kettle itself.
        """
        if self.kitchen_id is None:
            return None
        if self.task.goal_body and self.kettle_id is not None:
            return self._aabb_centre(self.kettle_id, -1)
        idx = self.links.get(self.task.target_link)
        if idx is None:
            return None
        return self._aabb_centre(self.kitchen_id, int(idx))

    @staticmethod
    def _aabb_centre(body_id, link_index):
        try:
            lower, upper = p.getAABB(body_id, link_index)
        except Exception:
            return None
        return [float((lower[i] + upper[i]) / 2.0) for i in range(3)]

    def _joint_values(self):
        values = {}
        if self.kitchen_id is None:
            return values
        for name, idx in self.joints.items():
            try:
                joint_type = p.getJointInfo(self.kitchen_id, idx)[2]
                if joint_type == p.JOINT_FIXED:
                    continue
                values[name] = float(p.getJointState(self.kitchen_id, idx)[0])
            except Exception:
                continue
        return values

    def _is_prismatic(self, joint_name):
        idx = self.joints.get(joint_name)
        if idx is None:
            return False
        try:
            return p.getJointInfo(self.kitchen_id, int(idx))[2] == p.JOINT_PRISMATIC
        except Exception:
            return False

    def _joint_position_unscaled(self, joint_name):
        """Joint position in unscaled (URDF / MuJoCo) units.

        ``globalScaling`` multiplies prismatic positions and limits but leaves
        revolute ones alone, so only prismatic readings need converting back.
        """
        idx = self.joints.get(joint_name)
        if idx is None:
            return None
        value = float(p.getJointState(self.kitchen_id, int(idx))[0])
        return unscaled(value) if self._is_prismatic(joint_name) else value

    def _clamped_goal(self, joint_name, goal_value):
        """Clamp an unscaled MuJoCo goal to the joint's unscaled URDF limit.

        A few prismatic goals (the burner plates, whose MuJoCo goal is -0.01 vs a
        URDF limit of -0.009) are marginally out of range and would otherwise be
        permanently unreachable.
        """
        idx = self.joints.get(joint_name)
        if idx is None:
            return goal_value
        try:
            info = p.getJointInfo(self.kitchen_id, int(idx))
            lower, upper = float(info[8]), float(info[9])
            if self._is_prismatic(joint_name):
                lower, upper = unscaled(lower), unscaled(upper)
            if lower < upper:
                return float(min(max(goal_value, lower), upper))
        except Exception:
            pass
        return goal_value

    def get_success_criteria(self):
        criteria = {"task": self.task_id, "threshold": self._threshold()}
        if self.task.goal_joints:
            criteria["goal_joints"] = {
                name: self._clamped_goal(name, value)
                for name, value in self.task.goal_joints.items()
            }
        if self.task.goal_body:
            criteria["goal_body"] = dict(self.task.goal_body)
            criteria["goal_body_pos"] = self._kettle_goal_position()
        return criteria

    def _kettle_goal_position(self):
        if not self.task.goal_body or self.kitchen_id is None:
            return None
        idx = self.links.get(self.task.goal_body.get("burner_link"))
        if idx is None:
            return None
        try:
            pos = p.getLinkState(self.kitchen_id, int(idx), computeForwardKinematics=True)[0]
        except Exception:
            return None
        return [float(pos[0]), float(pos[1]), float(pos[2]) + float(self.task.goal_body.get("z_offset", 0.0))]

    def task_error(self):
        """Euclidean distance between achieved and goal state (MuJoCo-style)."""
        if self.kitchen_id is None:
            return None
        if self.task.goal_joints:
            achieved, goal = [], []
            for name, value in self.task.goal_joints.items():
                current = self._joint_position_unscaled(name)
                if current is None:
                    return None
                achieved.append(current)
                goal.append(self._clamped_goal(name, value))
            return float(np.linalg.norm(np.array(achieved) - np.array(goal)))
        goal_pos = self._kettle_goal_position()
        if goal_pos is None or self.kettle_id is None:
            return None
        try:
            pos, _ = p.getBasePositionAndOrientation(self.kettle_id)
        except Exception:
            return None
        return float(np.linalg.norm(np.array(pos) - np.array(goal_pos)))

    def _threshold(self):
        """Joint-vector tasks use MuJoCo's BONUS_THRESH; the kettle goal is a
        world position and needs its own (tighter) metric tolerance."""
        return SUCCESS_THRESHOLD if self.task.goal_joints else POSITION_SUCCESS_THRESHOLD

    def check_success(self):
        error = self.task_error()
        if error is None:
            return None
        return bool(error < self._threshold())

    def get_state(self):
        error = self.task_error()
        state = {
            "sim_env": "franka_kitchen",
            "task": self.task_id,
            "task_label": self.task.label,
            "kitchen_id": self.kitchen_id,
            "kettle_id": self.kettle_id,
            "joint_indices": dict(self.joints),
            "joint_values": self._joint_values(),
            "target_link": self.task.target_link,
            "target_link_pos": self._target_link_position(),
            "success_criteria": self.get_success_criteria(),
            "task_error": error,
            "success": None if error is None else bool(error < self._threshold()),
        }
        try:
            if self.kettle_id is not None:
                pos, ori = p.getBasePositionAndOrientation(self.kettle_id)
                state["kettle_pos"] = list(map(float, pos))
                state["kettle_orientation_q"] = list(map(float, ori))
        except Exception as e:
            print("[Env] Warning: SimEnvKitchen.get_state failed to read kettle:", e)
        return state

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def get_3d_coordinates_prompt_section(self):
        return KITCHEN_COORDS_PROMPT_SECTION
