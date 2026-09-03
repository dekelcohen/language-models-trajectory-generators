"""Side-approach (non-top-down) end-effector poses.

Two tiers, both free of LLM calls:

* **Tier 0** - pure maths and plumbing: the approach-frame parametrisation, the pose
  builders, trajectory interpolation, and the ``env.py`` pose-length dispatch. Runs in
  milliseconds, no simulator.
* **Tier 1** - headless PyBullet: does the arm *actually* end up pointing where a
  ``side_grasp_pose`` asked it to? Guards against silent orientation drift and records how
  reachable horizontal approaches are, which is the practical limit of the feature.

The single most important assertion here is
:meth:`TopDownEquivalence.test_matches_legacy_expression`: the length-4 path must keep
producing exactly what it produced before the side approach existed.
"""

import math
import types
import unittest

import numpy as np

import config
import common_utils
from common_utils import grasp_pose, side_grasp_pose, pose_euler
from sim_adapter import transforms


def _matrix(euler):
    return np.array(transforms.matrix_from_quat(transforms.quat_from_euler(euler))).reshape(3, 3)


def _approach_axis(euler):
    """Direction the gripper points (end-effector +Z)."""
    return _matrix(euler)[:, 2]


def _closing_axis(euler):
    """Direction the fingers travel when closing (end-effector +Y)."""
    return _matrix(euler)[:, 1]


# --------------------------------------------------------------------------------------
# Tier 0: maths
# --------------------------------------------------------------------------------------

class TopDownEquivalence(unittest.TestCase):
    """The length-4 path must be indistinguishable from the pre-feature behaviour."""

    def test_matches_legacy_expression(self):
        for rotation in np.linspace(-math.pi, math.pi, 33):
            legacy = np.array(config.ee_start_orientation_e) + np.array([0, 0, rotation])
            rebuilt = transforms.ee_euler_from_approach(0.0, -math.pi / 2 + rotation, 0.0)
            angle = transforms.quat_angle_between(
                transforms.quat_from_euler(legacy), transforms.quat_from_euler(rebuilt)
            )
            self.assertLess(angle, 1e-9, f"rotation={rotation}")

    def test_pose_euler_of_len4_is_legacy(self):
        pose = grasp_pose(0.1, 0.2, 0.3, 0.5)
        self.assertEqual(len(pose), 4)
        expected = np.array(config.ee_start_orientation_e) + np.array([0, 0, 0.5])
        np.testing.assert_allclose(pose_euler(pose), expected)

    def test_top_down_points_straight_down(self):
        np.testing.assert_allclose(_approach_axis(pose_euler(grasp_pose(0, 0, 0, 0.0))),
                                   [0, 0, -1], atol=1e-9)


class ApproachFrame(unittest.TestCase):
    """``R(tilt, azimuth, roll) = Rz(azimuth) @ Ry(pi - tilt) @ Rz(roll)``."""

    def test_side_approach_axis_is_horizontal_at_azimuth(self):
        for azimuth in (0.0, math.pi / 2, -math.pi / 2, math.pi, 1.1, -2.4):
            euler = transforms.ee_euler_from_approach(math.pi / 2, azimuth, 0.0)
            np.testing.assert_allclose(_approach_axis(euler),
                                       [math.cos(azimuth), math.sin(azimuth), 0.0], atol=1e-9)

    def test_zero_roll_closes_horizontally(self):
        """rotation=0 pinches a VERTICAL bar: fingers move in the horizontal plane."""
        for azimuth in (0.0, math.pi / 2, 1.1):
            euler = transforms.ee_euler_from_approach(math.pi / 2, azimuth, 0.0)
            np.testing.assert_allclose(_closing_axis(euler),
                                       [-math.sin(azimuth), math.cos(azimuth), 0.0], atol=1e-9)

    def test_quarter_roll_closes_vertically(self):
        """rotation=pi/2 is the horizontal-bar case: fingers move along world Z."""
        euler = transforms.ee_euler_from_approach(math.pi / 2, 0.7, math.pi / 2)
        self.assertAlmostEqual(abs(float(_closing_axis(euler)[2])), 1.0, places=9)

    def test_tilt_sweeps_between_down_and_horizontal(self):
        for tilt in np.linspace(0.0, math.pi / 2, 9):
            axis = _approach_axis(transforms.ee_euler_from_approach(tilt, 0.3, 0.0))
            self.assertAlmostEqual(float(axis[2]), -math.cos(tilt), places=9)


class EulerFromMatrix(unittest.TestCase):
    """``euler_from_quat`` cannot represent pitch=pi; ``euler_from_matrix`` round-trips."""

    def test_round_trip_over_random_rotations(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            q = rng.normal(size=4)
            q = list(q / np.linalg.norm(q))
            m = transforms.matrix_from_quat(q)
            m2 = transforms.matrix_from_quat(transforms.quat_from_euler(transforms.euler_from_matrix(m)))
            np.testing.assert_allclose(m2, m, atol=1e-9)

    def test_agrees_with_euler_from_quat_where_that_is_valid(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            q = rng.normal(size=4)
            q = list(q / np.linalg.norm(q))
            np.testing.assert_allclose(transforms.euler_from_matrix(transforms.matrix_from_quat(q)),
                                       transforms.euler_from_quat(q), atol=1e-9)

    def test_top_down_pitch_is_out_of_asin_range(self):
        """Documents *why* euler_from_matrix exists (and why move()'s old check was dead)."""
        q = transforms.quat_from_euler(config.ee_start_orientation_e)
        self.assertLessEqual(abs(transforms.euler_from_quat(q)[1]), math.pi / 2 + 1e-9)
        self.assertAlmostEqual(config.ee_start_orientation_e[1], math.pi)


class Slerp(unittest.TestCase):

    def test_endpoints_and_shortest_arc(self):
        a = transforms.quat_from_euler([0.0, 0.0, 0.0])
        b = transforms.quat_from_euler([0.0, 0.0, 2.0])
        self.assertLess(transforms.quat_angle_between(transforms.slerp(a, b, 0.0), a), 1e-9)
        self.assertLess(transforms.quat_angle_between(transforms.slerp(a, b, 1.0), b), 1e-9)
        mid = transforms.slerp(a, b, 0.5)
        self.assertAlmostEqual(transforms.quat_angle_between(a, mid),
                               transforms.quat_angle_between(mid, b), places=6)

    def test_sign_flipped_quaternion_takes_short_way(self):
        a = transforms.quat_from_euler([0.0, 0.0, 0.1])
        b = [-v for v in transforms.quat_from_euler([0.0, 0.0, 0.2])]
        self.assertLess(transforms.quat_angle_between(transforms.slerp(a, b, 1.0), b), 1e-9)


class PoseBuilders(unittest.TestCase):

    def test_side_pose_shape_and_position(self):
        pose = side_grasp_pose(0.4, 0.5, 0.6, 0.0, math.pi / 2)
        self.assertEqual(len(pose), 6)
        np.testing.assert_allclose(pose[:3], [0.4, 0.5, 0.6])
        np.testing.assert_allclose(_approach_axis(pose[3:]), [0, 1, 0], atol=1e-9)

    def test_pose_euler_rejects_other_lengths(self):
        for bad in ([1, 2, 3], [1, 2, 3, 4, 5], [1] * 7):
            with self.assertRaises(ValueError):
                pose_euler(bad)


class InterpreterNamespace(unittest.TestCase):
    """What the generated code can actually call.

    ``main_prompt.py``'s own worked examples bind ``grasp_pose`` to a *list*
    (``grasp_pose = [pos_x, pos_y, z_grasp, grip_orientation]``, and later
    ``grasp_pose[2] - downward_press_distance``). Injecting a callable of that name put a
    function and a list behind one identifier in the same namespace, so a later
    ``grasp_pose(...)`` call would hit whichever the model happened to bind last. Top-down
    poses are therefore plain lists, exactly as every prompt example teaches, and only the
    genuinely new capability gets a name.
    """

    def _exec_locals(self):
        import helpers.main_utils as main_utils
        api_stub = types.SimpleNamespace(**{
            name: (lambda *a, **k: None)
            for name in ("detect_object", "get_grasp_poses", "visualize_grasp_pose",
                         "execute_trajectory", "open_gripper", "close_gripper",
                         "task_completed", "generate_linear_trajectory")
        })
        return main_utils.get_exec_locals(api_stub, logger=None)

    def test_side_builder_is_injected(self):
        self.assertIs(self._exec_locals()["side_grasp_pose"], side_grasp_pose)

    def test_grasp_pose_name_is_left_free_for_the_prompt_examples(self):
        self.assertNotIn("grasp_pose", self._exec_locals())

    def test_prompt_examples_still_bind_grasp_pose_as_a_list(self):
        """If the prompts ever stop doing this, the collision above no longer applies."""
        from prompts.main_prompt import IN_CONTEXT_EXAMPLE, IN_CONTEXT_EXAMPLE_GRASP
        for example in (IN_CONTEXT_EXAMPLE, IN_CONTEXT_EXAMPLE_GRASP):
            self.assertIn("grasp_pose = [", example)

    def test_main_prompt_says_nothing_about_side_approaches(self):
        """Gating is skill-only: a plain tabletop task must not see the extra DOF.

        A sentence here changes the prompt for *every* task, which invalidates the LLM
        cache and perturbs behaviour on tasks this feature was never meant to touch.
        """
        from prompts.main_prompt import MAIN_PROMPT
        for leaked in ("side_grasp_pose", "side approach", "horizontal (side)"):
            self.assertNotIn(leaked, MAIN_PROMPT)


# --------------------------------------------------------------------------------------
# Tier 0: trajectory interpolation and env dispatch
# --------------------------------------------------------------------------------------

def _api_stub():
    """Enough of ``API`` for ``generate_linear_trajectory``, which is otherwise pure."""
    return types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda *a, **k: None),
        head_camera_position=None, head_camera_orientation_q=None,
        cam_info=None, head_image_size=None,
    )


def _generate(start, end, n=5):
    import api
    return api.API.generate_linear_trajectory(_api_stub(), "desc", start, end, n)


class LinearTrajectory(unittest.TestCase):

    def test_len4_path_is_byte_identical_to_legacy_interpolation(self):
        start, end, n = [0.0, 0.0, 0.5, 0.0], [0.2, 0.3, 0.1, 1.2], 20
        legacy = []
        for i in range(n):
            t = i / (n - 1)
            legacy.append([start[j] + (end[j] - start[j]) * t for j in range(4)])
        self.assertEqual(_generate(start, end, n).points, legacy)

    def test_mixed_lengths_promote_to_len6_and_hit_both_endpoints(self):
        start = grasp_pose(0.0, 0.5, 0.6, 0.0)
        end = side_grasp_pose(0.05, 0.55, 0.4, 0.0, math.pi / 2)
        points = _generate(start, end, 5).points

        self.assertTrue(all(len(p) == 6 for p in points))
        for produced, wanted in ((points[0], start), (points[-1], end)):
            angle = transforms.quat_angle_between(transforms.quat_from_euler(produced[3:]),
                                                  transforms.quat_from_euler(pose_euler(wanted)))
            self.assertLess(angle, 1e-9)
        np.testing.assert_allclose(points[-1][:3], end[:3])

    def test_orientation_advances_monotonically(self):
        start = grasp_pose(0.0, 0.5, 0.6, 0.0)
        end = side_grasp_pose(0.0, 0.5, 0.6, 0.0, math.pi / 2)
        q_start = transforms.quat_from_euler(pose_euler(start))
        angles = [transforms.quat_angle_between(q_start, transforms.quat_from_euler(p[3:]))
                  for p in _generate(start, end, 9).points]
        for earlier, later in zip(angles, angles[1:]):
            self.assertLessEqual(earlier, later + 1e-9)

    def test_rejects_bad_lengths(self):
        with self.assertRaises(ValueError):
            _generate([0, 0, 0, 0], [0, 0, 0, 0, 0], 5)
        with self.assertRaises(ValueError):
            _generate([0, 0, 0], [0, 0, 0], 5)


class EnvPoseDispatch(unittest.TestCase):
    """``env.trajectory_point_orientation`` is the one place the two formats meet."""

    def test_len4_uses_the_legacy_expression(self):
        import env
        start = config.ee_start_orientation_e
        np.testing.assert_allclose(
            env.trajectory_point_orientation(start, [0.1, 0.2, 0.3, 0.5]),
            np.array(start) + np.array([0, 0, 0.5]),
        )

    def test_len6_is_passed_through_absolutely(self):
        import env
        pose = side_grasp_pose(0.4, 0.5, 0.6, 0.0, math.pi / 2)
        np.testing.assert_allclose(
            env.trajectory_point_orientation(config.ee_start_orientation_e, pose), pose[3:]
        )

    def test_other_lengths_raise(self):
        import env
        for bad in ([1, 2, 3], [1, 2, 3, 4, 5]):
            with self.assertRaises(ValueError):
                env.trajectory_point_orientation(config.ee_start_orientation_e, bad)


class IpcRoundTrip(unittest.TestCase):
    """Length-6 poses must survive the JSON transport used by the Genesis child process."""

    def test_mixed_trajectory_survives_encode_decode(self):
        import json
        from providers.json_ipc import encode, decode

        traj = common_utils.Trajectory(
            [grasp_pose(0.1, 0.2, 0.3, 0.4), side_grasp_pose(0.4, 0.5, 0.6, 0.0, 1.0)], "mixed"
        )
        out = decode(json.loads(json.dumps(encode(traj))))
        self.assertEqual(out.points, traj.points)
        self.assertEqual(out.desc, "mixed")


# --------------------------------------------------------------------------------------
# Tier 1: does the arm really go there? (headless PyBullet)
# --------------------------------------------------------------------------------------

# Horizontal approaches are far more demanding of the arm than top-down ones, so a pose can
# be perfectly well-formed and still not be reachable. These bounds separate "the maths or
# plumbing broke" (large, systematic error) from "the Panda cannot bend that way" (a
# reachability fact, reported by test_reachability_atlas rather than asserted).
ORIENTATION_TOLERANCE = 0.05   # rad, for poses expected to be comfortably reachable
REACHABLE_THRESHOLD = 0.10     # rad, what counts as "reached" in the atlas


class SimReach(unittest.TestCase):
    """Drive real IK with side poses and measure the orientation actually achieved."""

    @classmethod
    def setUpClass(cls):
        import env as envmod
        from sim_adapter.factory import get_adapter
        from robot import Robot

        cls.logger = types.SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
        cls.sim = get_adapter("pybullet")
        cls.sim.connect(gui=False)
        cls.sim.set_asset_search_path()
        cls.sim.set_gravity(0, 0, -9.81)
        cls.sim.load_urdf("plane.urdf")

        class Args:
            mode = "default"
            robot = "franka"
            task = "grasp"

        cls.env = envmod.Environment(Args, cls.sim)
        cls.env.simenv.configure_robot_pose()
        cls.env.load()
        cls.robot = Robot(Args, cls.logger, cls.sim)
        cls.sim.build()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.sim.disconnect()
        except Exception:
            pass

    def _achieved_error(self, position, euler, settles=3):
        for _ in range(settles):
            self.robot.move(self.env, position, euler, gripper_open=True, is_trajectory=False)
        _, quat = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return transforms.quat_angle_between(quat, self.sim.quat_from_euler(euler)), quat

    def test_top_down_still_reaches(self):
        error, quat = self._achieved_error([0.0, 0.5, 0.5], list(config.ee_start_orientation_e))
        self.assertLess(error, ORIENTATION_TOLERANCE)
        axis = np.array(self.sim.matrix_from_quat(quat)).reshape(3, 3)[:, 2]
        np.testing.assert_allclose(axis, [0, 0, -1], atol=0.05)

    def test_side_approach_reaches_and_points_horizontally(self):
        # Facing away from the base (+y) and diagonally are comfortable for the Panda.
        for azimuth in (math.pi / 2, math.pi / 4):
            pose = side_grasp_pose(0.0, 0.45, 0.45, 0.0, azimuth)
            error, quat = self._achieved_error(pose[:3], pose_euler(pose))
            self.assertLess(error, ORIENTATION_TOLERANCE, f"azimuth={azimuth}")

            axis = np.array(self.sim.matrix_from_quat(quat)).reshape(3, 3)[:, 2]
            self.assertLess(abs(float(axis[2])), 0.1, "approach axis should be horizontal")
            np.testing.assert_allclose(axis[:2], [math.cos(azimuth), math.sin(azimuth)], atol=0.05)

    def test_reachability_atlas(self):
        """Report, don't fail: which horizontal approaches the Panda can actually strike.

        The feature's practical limit is reachability, not maths. This prints a small map so
        a failing side-approach task can be triaged as "unreachable pose" without a full
        agent run.
        """
        rows = []
        for azimuth_deg in (0, 45, 90, 135, 180, -90):
            pose = side_grasp_pose(0.0, 0.45, 0.45, 0.0, math.radians(azimuth_deg))
            error, _ = self._achieved_error(pose[:3], pose_euler(pose))
            rows.append((azimuth_deg, error, error < REACHABLE_THRESHOLD))
        print("\nside-approach reachability at [0.00, 0.45, 0.45]:")
        for azimuth_deg, error, ok in rows:
            print(f"  azimuth {azimuth_deg:>4}deg  err={error:.4f} rad  {'reachable' if ok else 'OUT OF REACH'}")
        self.assertTrue(any(ok for _, _, ok in rows), "no horizontal approach was reachable at all")


if __name__ == "__main__":
    unittest.main()
