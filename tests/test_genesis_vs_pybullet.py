"""Cross-simulator parity: Genesis vs the committed PyBullet goldens.

**Run this with the Genesis interpreter**, e.g.::

    <vlm_genesis>/python.exe -m pytest tests/test_genesis_vs_pybullet.py

The goldens in ``tests/golden/cross_sim/`` are produced by
``tests/tools/dump_sim_state.py`` under ``vlm_traj``. Regenerate them with::

    <vlm_traj>/python.exe tests/tools/dump_sim_state.py --sim pybullet --task door \
        -o tests/golden/cross_sim/pybullet_door.json

Tolerances are deliberately split by kind, because they mean different things:

* camera matrices - pure math, must agree to float32 precision (~1e-5). This is the
  test that answers "are the projection and camera matrices the same dims and the same
  meaning?" with a number instead of an opinion.
* world positions of static scene objects - must agree to ~1 mm. These feed perception,
  so a real divergence here is a real bug.
* the settled arm - two different solvers, integrators and contact models running for
  180 steps. Physical agreement (~2 cm) is the honest bar; anything tighter would be
  pinning solver noise.

Index tables are deliberately *not* compared: PyBullet and Genesis genuinely disagree
on link/joint numbering, which is exactly why the app layer resolves everything by name.
"""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.tools.dump_sim_state import snapshot  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "cross_sim")

#: float32 round-trip through Genesis' renderer; measured 3.1e-08 / 1.9e-07 in practice.
MATRIX_ATOL = 1e-5
#: Static geometry the two sims place from the same URDF.
STATIC_POSITION_ATOL = 1e-3
#: Settled arm after 180 steps of two different solvers.
ARM_POSITION_ATOL = 2e-2
ARM_JOINT_ATOL = 5e-2

#: Per-task scene state keys that are physical 3-vectors, so must agree in metres.
STATE_VECTORS = {
    "door": ("door_handle_pos", "latch_pos", "hinge_pos", "pole_pos", "pole_dims"),
    "grasp": ("object_pos", "object_dims"),
}
#: Keys that are opaque handles or index tables; only their presence is contractual.
STATE_OPAQUE = ("door_id", "pole_id", "object_id",
                "door_hinge_index", "latch_index", "door_handle_latch")


def _genesis_importable():
    try:
        import genesis  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_genesis_importable(),
                     "run this suite with the vlm_genesis interpreter")
class TestGenesisVsPyBullet(unittest.TestCase):

    TASKS = ("door", "grasp")

    @classmethod
    def setUpClass(cls):
        cls.golden = {}
        cls.actual = {}
        for task in cls.TASKS:
            path = os.path.join(GOLDEN_DIR, f"pybullet_{task}.json")
            if not os.path.exists(path):
                raise unittest.SkipTest(f"missing golden '{path}'; see this file's docstring")
            with open(path) as fh:
                cls.golden[task] = json.load(fh)
            # One Genesis boot per task; gs.init is process-global so they run serially.
            cls.actual[task] = snapshot("genesis", task, cls.golden[task]["settle_steps"])

    def test_projection_matrix_is_the_same_matrix(self):
        """Requirement (b): same dims, same meaning - as a number."""
        for task in self.TASKS:
            with self.subTest(task=task):
                a = np.array(self.golden[task]["head_projection_matrix"], dtype=float)
                b = np.array(self.actual[task]["head_projection_matrix"], dtype=float)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(len(a), 16, "a flat 4x4, column-major, in both sims")
                np.testing.assert_allclose(b, a, rtol=0, atol=MATRIX_ATOL)

    def test_view_matrix_is_the_same_matrix(self):
        for task in self.TASKS:
            with self.subTest(task=task):
                a = np.array(self.golden[task]["head_view_matrix"], dtype=float)
                b = np.array(self.actual[task]["head_view_matrix"], dtype=float)
                self.assertEqual(len(a), len(b))
                np.testing.assert_allclose(b, a, rtol=0, atol=MATRIX_ATOL)

    def test_projection_matrix_is_a_gl_perspective_matrix(self):
        """Guards against 'both wrong the same way' - check the structure too."""
        for task in self.TASKS:
            for sim, data in (("pybullet", self.golden[task]), ("genesis", self.actual[task])):
                with self.subTest(task=task, sim=sim):
                    m = np.array(data["head_projection_matrix"], dtype=float).reshape(4, 4, order="F")
                    self.assertAlmostEqual(m[3, 2], -1.0, places=5, msg="must look down -z")
                    self.assertAlmostEqual(m[3, 3], 0.0, places=5)
                    self.assertAlmostEqual(m[0, 1], 0.0, places=5)
                    self.assertGreater(m[0, 0], 0.0)
                    self.assertGreater(m[1, 1], 0.0)

    def test_scene_state_keys_are_identical(self):
        """The sim-env profiles are shared, so their public state must be too."""
        for task in self.TASKS:
            with self.subTest(task=task):
                self.assertEqual(set(self.golden[task]["state"]),
                                 set(self.actual[task]["state"]))

    def test_static_scene_positions_match(self):
        for task in self.TASKS:
            for key in STATE_VECTORS[task]:
                with self.subTest(task=task, field=key):
                    a = np.array(self.golden[task]["state"][key], dtype=float)
                    b = np.array(self.actual[task]["state"][key], dtype=float)
                    np.testing.assert_allclose(b, a, rtol=0, atol=STATIC_POSITION_ATOL)

    def test_scene_handles_are_present_but_not_compared(self):
        for task in self.TASKS:
            golden_state = self.golden[task]["state"]
            for key in STATE_OPAQUE:
                if key in golden_state:
                    with self.subTest(task=task, field=key):
                        self.assertIn(key, self.actual[task]["state"])

    def test_end_effector_settles_to_the_same_place(self):
        for task in self.TASKS:
            with self.subTest(task=task):
                a = np.array(self.golden[task]["ee_pos"], dtype=float)
                b = np.array(self.actual[task]["ee_pos"], dtype=float)
                error = float(np.linalg.norm(b - a))
                self.assertLess(error, ARM_POSITION_ATOL,
                                f"end effector is {error * 100:.1f} cm apart")

    def test_joint_positions_settle_together(self):
        for task in self.TASKS:
            with self.subTest(task=task):
                a = np.array(self.golden[task]["joint_positions"], dtype=float)
                b = np.array(self.actual[task]["joint_positions"], dtype=float)
                self.assertEqual(a.shape, b.shape,
                                 "both sims must expose the same movable-joint count")
                np.testing.assert_allclose(b, a, rtol=0, atol=ARM_JOINT_ATOL)

    def test_link_names_are_a_superset(self):
        """Names are the portable handle, so every link name PyBullet has must exist."""
        for task in self.TASKS:
            with self.subTest(task=task):
                missing = (set(self.golden[task]["robot_link_names"])
                           - set(self.actual[task]["robot_link_names"]))
                self.assertEqual(missing, set(), f"Genesis is missing links {sorted(missing)}")

    def test_movable_joint_names_are_a_superset(self):
        """Only *movable* joints. Genesis drops fixed joints from its joint table entirely,
        which is precisely why robot.py resolves the arm and gripper joints by name."""
        for task in self.TASKS:
            with self.subTest(task=task):
                golden = self.golden[task]
                movable_indices = set(golden["joint_indices"])
                movable = {name for name, index in golden["robot_joint_names"].items()
                           if index in movable_indices}
                self.assertTrue(movable, "golden must record at least one movable joint")
                missing = movable - set(self.actual[task]["robot_joint_names"])
                self.assertEqual(missing, set(), f"Genesis is missing joints {sorted(missing)}")

    def test_depth_encoding_is_declared_and_differs(self):
        """The sims really do disagree here; utils bridges it. Pin both sides."""
        for task in self.TASKS:
            with self.subTest(task=task):
                self.assertEqual(self.golden[task]["depth_encoding"], "opengl")
                self.assertEqual(self.actual[task]["depth_encoding"], "linear_metric")


if __name__ == "__main__":
    unittest.main()
