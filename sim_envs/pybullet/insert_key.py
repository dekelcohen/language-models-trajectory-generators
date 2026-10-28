"""Insert-key sim-env: pick a key from its stand, insert it into a lock, turn it.

Adapted from RoboDojo's dual-arm ``insert_key`` task
(https://github.com/RoboDojo-Benchmark/RoboDojo, Apache-2.0) for a single Franka Panda.
The key mesh/texture is RoboDojo's; the lock (with a revolute plug), the key stand
and the table are procedural - see ``my_assets/insert_key/tools/build_assets.py``.

Unlike RoboDojo (which only checks insertion), success here also requires the
plug to be turned, so the key must actually transmit torque through a
friction-only grasp.

Profile uses only ``self.sim`` (SimAdapter) primitives, so it runs in PyBullet and
Genesis. Everything returned by ``get_state`` / ``get_success_criteria`` is
privileged sim state for logs/eval only - never put it in a prompt (Sim2Real).
"""

import json
import math
import os
import traceback

import numpy as np

import config
from sim_adapter.camera_math import spherical_camera_pose
from sim_envs.pybullet.base import SimEnvBase

ASSET_DIR = "my_assets/insert_key"
with open(os.path.join(os.path.dirname(__file__), "..", "..", ASSET_DIR, "geometry.json")) as _fh:
    GEO = json.load(_fh)

TABLE_TOP_Z = float(GEO["table_top_z"])

# The Panda is bolted to the table top, facing +y.
ROBOT_BASE_POSITION = [0.0, -0.05, TABLE_TOP_Z]
ROBOT_BASE_ORIENTATION_E = [0.0, 0.0, math.pi / 2]
ROBOT_EE_START_POSITION = [0.0, 0.42, TABLE_TOP_Z + 0.38]
ROBOT_EE_START_ORIENTATION_E = [0.0, math.pi, -math.pi / 2]

# Nominal object placements (world xy) and per-episode jitter.
KEY_STAND_XY = (0.16, 0.46)
LOCK_XY = (-0.14, 0.50)
XY_JITTER = 0.02
# Yaws are about world z. The home gripper closes along world x; the key's thin axis
# (bow faces, key-local y) lines up with it at yaw 90 deg. 60 deg keeps the bow visible
# to the head camera and asks for a ~30 deg wrist turn. The lock's keyway long axis
# is key-local x, so at 90 deg it accepts a key held by an unrotated gripper.
KEY_YAW_NOMINAL = math.radians(60)
KEY_YAW_RANGE = math.radians(15)
LOCK_YAW_NOMINAL = math.radians(90)
LOCK_YAW_RANGE = math.radians(25)

# Success thresholds (RoboDojo: insert >= 2.5 cm, xy < 7 mm; turning is ours).
INSERT_DEPTH_MIN = 0.025
INSERT_XY_MAX = 0.007
TURN_ANGLE_MIN = math.radians(60)

# Plug Coulomb friction so it does not drift, yet a held key turns it. Per simulator: PyBullet
# models it as a capped zero-velocity motor, Genesis as joint frictionloss, and Genesis' softer
# finger contacts let the key slip in the grasp above ~0.01 N*m.
PLUG_RESIST_TORQUE = {"pybullet": 0.05, "genesis": 0.01}
PLUG_RESIST_TORQUE_DEFAULT = 0.05
# Genesis defaults joint armature to 0.1 kg*m^2 (plug inertia is ~4e-5); PyBullet ignores it.
PLUG_ARMATURE = 1e-4
KEY_FRICTION = 1.5

HEAD_TARGET = [0.0, 0.47, TABLE_TOP_Z + 0.03]
HEAD_DISTANCE = 0.62
HEAD_YAW = 180.0       # camera on +y looking towards -y (facing the robot)
HEAD_PITCH = -45.0

COORDS_PROMPT_SECTION = (
    "The 3D coordinate system of the environment is as follows:\n"
    "  1. The x-axis is in the horizontal direction, increasing to the left.\n"
    "  2. The y-axis is in the depth direction, decreasing away from you.\n"
    "  3. The z-axis is in the vertical direction, increasing upwards."
)


def _yaw_quat(sim, yaw):
    return sim.quat_from_euler([0.0, 0.0, float(yaw)])


def _rot_z(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]]


class SimEnvInsertKey(SimEnvBase):
    """Single-arm insert-and-turn key task on a tabletop."""

    def __init__(self, sim=None, seed=None, randomize=True):
        super().__init__(sim)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.table_id = None
        self.key_id = None
        self.stand_id = None
        self.lock_id = None
        self.plug_joint = None
        self.plug_link = None
        self.key_stand_pose = None
        self.lock_pose = None
        self.configure_cameras()

    # ------------------------------------------------------------------ robot
    def required_robot(self):
        return "franka"

    def configure_robot_pose(self):
        config.base_start_position_franka = list(ROBOT_BASE_POSITION)
        config.base_start_orientation_e_franka = list(ROBOT_BASE_ORIENTATION_E)
        config.ee_start_position = list(ROBOT_EE_START_POSITION)
        config.ee_start_orientation_e = list(ROBOT_EE_START_ORIENTATION_E)

    def get_ee_start_pose(self):
        return list(ROBOT_EE_START_POSITION), list(ROBOT_EE_START_ORIENTATION_E)

    # ---------------------------------------------------------------- cameras
    def configure_cameras(self):
        config.camera_distance = HEAD_DISTANCE
        config.camera_yaw = HEAD_YAW
        config.camera_pitch = HEAD_PITCH
        config.camera_target_position = list(HEAD_TARGET)
        pos, ori_e = spherical_camera_pose(HEAD_TARGET, HEAD_DISTANCE, HEAD_YAW, HEAD_PITCH)
        config.head_camera_position = pos
        config.head_camera_orientation_e = ori_e
        try:
            is_gui = bool(self.sim.is_gui()) if self.sim is not None else False
        except Exception:
            is_gui = False
        config.head_camera_use_debug_view = is_gui
        config.head_camera_use_spherical_view = not is_gui

    def get_3d_coordinates_prompt_section(self):
        return COORDS_PROMPT_SECTION

    # ----------------------------------------------------------------- assets
    def apply(self, env):
        self.configure_cameras()

    def _sample_pose(self, xy, yaw_nominal, yaw_range):
        if not self.randomize:
            return [float(xy[0]), float(xy[1])], float(yaw_nominal)
        dx, dy = self.rng.uniform(-XY_JITTER, XY_JITTER, 2)
        yaw = float(yaw_nominal + self.rng.uniform(-yaw_range, yaw_range))
        return [float(xy[0] + dx), float(xy[1] + dy)], yaw

    def load_assets(self, env):
        sim = self.sim
        try:
            self.table_id = sim.load_urdf(f"{ASSET_DIR}/table.urdf", [0, 0, 0], [0, 0, 0, 1], fixed_base=True)

            stand_xy, stand_yaw = self._sample_pose(KEY_STAND_XY, KEY_YAW_NOMINAL, KEY_YAW_RANGE)
            self.key_stand_pose = (stand_xy, stand_yaw)
            self.stand_id = sim.load_urdf(
                f"{ASSET_DIR}/key_stand.urdf", [stand_xy[0], stand_xy[1], TABLE_TOP_Z],
                _yaw_quat(sim, stand_yaw), fixed_base=True)

            # Key rests on its shoulder on the stand's top face, blade down in the slot.
            stand_top = TABLE_TOP_Z + float(GEO["stand_size"][2])
            key_z = stand_top - float(GEO["key_shoulder"]["z"][0]) + 0.0005
            self.key_id = sim.load_urdf(
                f"{ASSET_DIR}/key.urdf", [stand_xy[0], stand_xy[1], key_z],
                _yaw_quat(sim, stand_yaw), fixed_base=False)

            lock_xy, lock_yaw = self._sample_pose(LOCK_XY, LOCK_YAW_NOMINAL, LOCK_YAW_RANGE)
            self.lock_pose = (lock_xy, lock_yaw)
            self.lock_id = sim.load_urdf(
                f"{ASSET_DIR}/lock.urdf", [lock_xy[0], lock_xy[1], TABLE_TOP_Z],
                _yaw_quat(sim, lock_yaw), fixed_base=True, links_to_keep=["plug"])
            self.plug_joint = sim.get_joint_index_by_name(self.lock_id, "plug_joint")
            self.plug_link = sim.get_joint_child_link(self.lock_id, self.plug_joint)
        except Exception as e:
            print("[Env] Failed to load insert_key assets:", e)
            traceback.print_exc()

    def tune_physics(self):
        sim = self.sim
        try:
            if self.key_id is not None:
                sim.change_dynamics(self.key_id, -1, lateralFriction=KEY_FRICTION,
                                    spinningFriction=0.01, rollingFriction=0.0)
            if self.lock_id is not None and self.plug_joint is not None:
                sim.set_joint_armature(self.lock_id, self.plug_joint, PLUG_ARMATURE)
                sim.set_joint_friction(self.lock_id, self.plug_joint,
                                       PLUG_RESIST_TORQUE.get(getattr(sim, "name", ""),
                                                              PLUG_RESIST_TORQUE_DEFAULT))
        except Exception as e:
            print("[Env] insert_key tune_physics failed:", e)
            traceback.print_exc()

    # ---------------------------------------------------------- ground truth
    def _plug_angle(self):
        return float(self.sim.get_joint_state(self.lock_id, self.plug_joint).position)

    def _plug_top_world(self):
        pos, _q = self.sim.get_link_pose(self.lock_id, self.plug_link)
        return np.array([pos[0], pos[1], pos[2] + float(GEO["lock_plug_top_local_z"])])

    def _key_tip_world(self):
        pos, q = self.sim.get_base_pose(self.key_id)
        R = np.array(self.sim.matrix_from_quat(q)).reshape(3, 3)
        tip_local = np.array([0.0, 0.0, float(GEO["key_blade"]["z"][0])])
        return np.asarray(pos) + R @ tip_local, R

    def get_state(self):
        state = {"key_pos": None, "key_up_cos": None, "plug_angle_deg": None,
                 "plug_top": None, "key_tip": None,
                 "key_stand_pose": self.key_stand_pose, "lock_pose": self.lock_pose}
        try:
            kp, _ = self.sim.get_base_pose(self.key_id)
            tip, R = self._key_tip_world()
            top = self._plug_top_world()
            state.update(
                key_pos=[float(v) for v in kp],
                key_tip=[float(v) for v in tip],
                key_up_cos=float(R[2, 2]),
                plug_top=[float(v) for v in top],
                plug_angle_deg=math.degrees(self._plug_angle()),
            )
        except Exception as e:
            print("[Env] insert_key get_state failed:", e)
        return state

    def get_success_criteria(self):
        try:
            tip, R = self._key_tip_world()
            top = self._plug_top_world()
            depth = float(top[2] - tip[2])
            xy = float(np.linalg.norm(tip[:2] - top[:2]))
            angle = self._plug_angle()
        except Exception as e:
            return {"error": str(e)}
        inserted = depth >= INSERT_DEPTH_MIN and xy <= INSERT_XY_MAX
        turned = abs(angle) >= TURN_ANGLE_MIN
        return {
            "insert_depth_m": depth, "insert_depth_min_m": INSERT_DEPTH_MIN,
            "key_tip_xy_err_m": xy, "key_tip_xy_max_m": INSERT_XY_MAX,
            "plug_angle_deg": math.degrees(angle), "turn_angle_min_deg": math.degrees(TURN_ANGLE_MIN),
            "inserted": inserted, "turned": turned, "success": bool(inserted and turned),
        }

    def check_success(self):
        crit = self.get_success_criteria()
        if "success" not in crit:
            return None
        return bool(crit["success"])
