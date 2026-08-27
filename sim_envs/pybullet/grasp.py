"""Grasp sim-env: keep existing camera/robot poses and load a YCB object."""

import traceback

import pybullet as p

import config
from sim_envs.pybullet.base import SimEnvBase


class SimEnvGrasp(SimEnvBase):
    """Grasp task: keep existing camera/robot poses and load a YCB object."""

    def __init__(self):
        # Keep defaults from config for cameras and robot
        self.object_id = None

    def apply(self, env):
        # Nothing additional to set for grasp
        return

    def load_assets(self, env):
        try:
            object_start_position = config.object_start_position
            object_start_orientation_q = p.getQuaternionFromEuler(config.object_start_orientation_e)
            self.object_id = p.loadURDF(
                "ycb_assets/003_cracker_box.urdf",
                object_start_position,
                object_start_orientation_q,
                useFixedBase=False,
                globalScaling=config.global_scaling,
            )

        except Exception as e:
            print("[Env] Warning: failed to load grasp object:", e)
            traceback.print_exc()

    def get_3d_coordinates_prompt_section(self):
        return config.three_d_coordinates_prompt_section

    def get_state(self):
        """Return pos + dims of the grasp object (self.object_id)."""
        state = {"object_id": self.object_id, "object_pos": None, "object_dims": None}
        try:
            if self.object_id is not None:
                pos, _ori = p.getBasePositionAndOrientation(self.object_id)
                aabb_min, aabb_max = p.getAABB(self.object_id, -1)
                dims = [float(aabb_max[0] - aabb_min[0]), float(aabb_max[1] - aabb_min[1]), float(aabb_max[2] - aabb_min[2])]
                state["object_pos"] = list(map(float, pos))
                state["object_dims"] = dims
        except Exception as e:
            print("[Env] Warning: SimEnvGrasp.get_state failed:", e)
            traceback.print_exc()
        return state
