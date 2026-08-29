"""Grasp sim-env: keep existing camera/robot poses and load a YCB object."""

import traceback

import config
from sim_envs.pybullet.base import SimEnvBase


class SimEnvGrasp(SimEnvBase):
    """Grasp task: keep existing camera/robot poses and load a YCB object."""

    def __init__(self, sim=None):
        # Keep defaults from config for cameras and robot
        super().__init__(sim)
        self.object_id = None

    def apply(self, env):
        # Nothing additional to set for grasp
        return

    def load_assets(self, env):
        try:
            object_start_position = config.object_start_position
            object_start_orientation_q = self.sim.quat_from_euler(config.object_start_orientation_e)
            self.object_id = self.sim.load_urdf(
                "ycb_assets/003_cracker_box.urdf",
                object_start_position,
                object_start_orientation_q,
                fixed_base=False,
                scaling=config.global_scaling,
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
                pos, _ori = self.sim.get_base_pose(self.object_id)
                aabb_min, aabb_max = self.sim.get_aabb(self.object_id, -1)
                dims = [float(aabb_max[0] - aabb_min[0]), float(aabb_max[1] - aabb_min[1]), float(aabb_max[2] - aabb_min[2])]
                state["object_pos"] = list(map(float, pos))
                state["object_dims"] = dims
        except Exception as e:
            print("[Env] Warning: SimEnvGrasp.get_state failed:", e)
            traceback.print_exc()
        return state
