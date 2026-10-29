import math
import unittest

import numpy as np


class TestInsertKey(unittest.TestCase):
    """insert_key scene physics: key rests in its stand, an aligned key seats in the
    keyway, the keyway transmits torque between key and plug, and success needs both
    insertion and a >= 60 deg turn."""

    def setUp(self):
        from sim_adapter import get_adapter
        from sim_envs.pybullet.insert_key import SimEnvInsertKey

        self.sim = get_adapter("pybullet")
        self.sim.connect(gui=False)
        self.sim.set_gravity(0, 0, -9.81)
        self.se = SimEnvInsertKey(self.sim, seed=0)
        self.se.load_assets(None)
        self.se.tune_physics()

    def tearDown(self):
        self.sim.disconnect()

    def _step(self, n):
        for _ in range(n):
            self.sim.step()

    def _drop_key_over_plug(self, dx=0.0, dy=0.0):
        top = self.se.get_state()["plug_top"]
        _xy, lock_yaw = self.se.lock_pose
        self.sim.set_base_pose(self.se.key_id, [top[0] + dx, top[1] + dy, top[2] + 0.05],
                               self.sim.quat_from_euler([0, 0, lock_yaw]))
        self._step(480)

    def test_key_rests_upright_in_stand(self):
        self._step(480)
        state = self.se.get_state()
        (sx, sy), _ = self.se.key_stand_pose
        self.assertGreater(state["key_up_cos"], 0.99)
        self.assertLess(np.hypot(state["key_pos"][0] - sx, state["key_pos"][1] - sy), 0.005)
        self.assertFalse(self.se.check_success())

    def test_inserted_key_turns_plug_and_succeeds(self):
        self._drop_key_over_plug()
        crit = self.se.get_success_criteria()
        self.assertTrue(crit["inserted"], crit)
        self.assertFalse(crit["success"], crit)

        # Drive the plug: the blade must be captured by the keyway and turn with it.
        key_yaw0 = self.sim.euler_from_quat(self.sim.get_base_pose(self.se.key_id)[1])[2]
        self.sim.set_joint_position(self.se.lock_id, self.se.plug_joint, math.radians(75), force=5.0)
        self._step(480)
        key_yaw1 = self.sim.euler_from_quat(self.sim.get_base_pose(self.se.key_id)[1])[2]
        turned = math.degrees((key_yaw1 - key_yaw0 + math.pi) % (2 * math.pi) - math.pi)
        self.assertGreater(turned, 60.0)
        self.assertTrue(self.se.check_success(), self.se.get_success_criteria())

    def test_key_resting_off_the_keyway_is_not_inserted(self):
        self._drop_key_over_plug(dx=0.015)
        crit = self.se.get_success_criteria()
        self.assertFalse(crit["inserted"], crit)
        self.assertFalse(crit["success"], crit)


if __name__ == "__main__":
    unittest.main()
