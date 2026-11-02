"""Genesis kettle_mug (native SPH water): particle counting + scene spawn/state.

The counting tests are pure numpy and run anywhere. The scene test needs Genesis (the
vlm_genesis env, which has no pytest; ~90 s on CPU):
`conda run -n vlm_genesis python -m unittest tests.test_kettle_mug_genesis`.
"""

import importlib.util
import math
import unittest

import numpy as np

from sim_envs.genesis import kettle_mug as km

HAS_GENESIS = importlib.util.find_spec("genesis") is not None
IDENT = [0, 0, 0, 1]


class TestCountWater(unittest.TestCase):

    def setUp(self):
        self.mug = np.array([0.2, 0.55, 0.02])
        self.kettle = np.array([-0.1, 0.52, 0.02])

    def test_classifies_mug_kettle_spout_and_floor(self):
        pts = [self.mug + [0, 0, 0.03],                      # inside the mug
               self.mug + [0.01, -0.01, 0.08],               # near the mug rim, still inside
               self.kettle + [0, 0, 0.05],                   # kettle body
               self.kettle + np.array(km.spout_tip_local()) * 0.6,  # inside the spout duct
               self.mug + [km.MUG_R + 0.02, 0, 0.01],        # spilled next to the mug
               [0.0, 0.3, 0.02]]                             # on the board
        c = km.count_water(pts, self.mug, IDENT, self.kettle, IDENT)
        self.assertEqual(c, {"total": 6, "in_kettle": 2, "in_mug": 2, "outside_mug": 2})

    def test_counts_follow_tilted_mug(self):
        # Mug knocked over onto its side (rotated 90 deg about x): a point 3 cm "up" its axis
        # now lies along -y in world.
        q = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
        c = km.count_water([self.mug + [0, -0.03, 0.0], self.mug + [0, 0, 0.06]], self.mug, q)
        self.assertEqual((c["in_mug"], c["outside_mug"]), (1, 1))


@unittest.skipUnless(HAS_GENESIS, "genesis not importable (use the vlm_genesis python)")
class TestKettleMugScene(unittest.TestCase):

    def test_spawn_water_rests_in_kettle(self):
        import config
        from sim_adapter.genesis_adapter import GenesisAdapter

        sim = GenesisAdapter(backend="cpu", dt=config.control_dt, scene_options=km.SCENE_OPTIONS)
        sim.connect(gui=False)
        self.addCleanup(sim.disconnect)
        se = km.SimEnvKettleMug(sim)
        se.load_assets(None)
        sim.build()
        for _ in range(120):
            sim.step()
        st = se.get_state()
        w = st["water"]
        self.assertGreater(w["total"], 300)
        self.assertGreaterEqual(w["in_kettle"], 0.98 * w["total"])
        self.assertEqual(w["in_mug"], 0)
        self.assertFalse(st["success"])
        self.assertFalse(se.check_success())


if __name__ == "__main__":
    unittest.main()
