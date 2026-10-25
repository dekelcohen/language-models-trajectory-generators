import unittest

import numpy as np
import pybullet as p


class TestKettleMugWater(unittest.TestCase):
    """Particle water for franka_kitchen:kettle_mug: nothing pours while the kettle is
    upright; tilting the spout over the mug fills it and satisfies goal_pour."""

    def setUp(self):
        from sim_envs.pybullet.franka_kitchen import simenv as kitchen
        self.k = kitchen
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        self.se = kitchen.SimEnvKitchen("kettle_mug")
        self.se.load_assets(None)

    def tearDown(self):
        p.disconnect(self.cid)

    def _step(self, n):
        for _ in range(n):
            self.se.step_hook()
            p.stepSimulation()

    def _water(self):
        return self.se.get_state()["water"]

    def test_upright_kettle_keeps_water_tilted_kettle_fills_mug(self):
        se, k = self.se, self.k
        self.assertIsNotNone(se.mug_id)
        self._step(240)
        self.assertEqual(self._water(), {"total": k.WATER_PARTICLES, "in_kettle": k.WATER_PARTICLES,
                                         "in_mug": 0, "outside_mug": 0})
        self.assertFalse(se.check_success())

        # Hold the kettle tilted toward +y (the spout side) with its spout tip above the mug.
        kp, kq = p.getBasePositionAndOrientation(se.kettle_id)
        tilted = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([-0.9, 0, 0]), [0, 0, 0], kq)[1]
        p.resetBasePositionAndOrientation(se.kettle_id, kp, tilted)
        tip, _ = se._kettle_local_to_world(k.KETTLE_SPOUT_TIP_LOCAL)
        mug = np.array(k.MUG_POSITION)
        shift = np.array([mug[0], mug[1], mug[2] + k.MUG_HEIGHT + 0.08]) - tip
        p.resetBasePositionAndOrientation(se.kettle_id, (np.array(kp) + shift).tolist(), tilted)
        p.createConstraint(se.kettle_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                           (np.array(kp) + shift).tolist(), [0, 0, 0, 1], tilted)

        self._step(k.WATER_PARTICLES * k.WATER_EMIT_EVERY + 480)
        water = self._water()
        self.assertEqual(water["in_kettle"], 0, f"not all water released: {water}")
        self.assertGreaterEqual(water["in_mug"], 0.8 * k.WATER_PARTICLES, f"{water}")
        self.assertTrue(se.check_success(), f"{water}, task_error={se.task_error()}")


if __name__ == "__main__":
    unittest.main()
