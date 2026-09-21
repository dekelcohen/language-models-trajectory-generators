"""``--vis-grasp``: drawing the grasp pose the robot actually executes.

Everything here is pure maths and plumbing - no simulator, no LLM - and the point of it is
that a marker which lies is worse than no marker at all. Three things can silently go
wrong and each has a test:

1. **Frame confusion.** ``env.draw_grasp_pose`` speaks the GraspNet convention (``+X`` =
   fingers, ``+Z`` = approach); the end-effector frame uses ``+Y`` = fingers. Drawing the
   end-effector matrix as-is would rotate the fingers by 90 degrees and still look
   plausible in a screenshot.
2. **Depth.** Trajectory points name where the *fingers* go, and
   ``Robot._apply_gripper_depth_offset`` drives the end-effector link a further
   ``gripper_depth_offset`` along the approach axis. A marker that ignores that is 6 cm
   from the real hand on a franka.
3. **Prompt leakage.** The flag is off by default, and a prompt that changed anyway would
   invalidate the LLM cache and perturb every unrelated task.
"""

import math
import types
import unittest

import numpy as np

import config
import common_utils
from common_utils import ee_pose_to_grasp_matrix, grasp_pose, normalize_grasp_viz_poses, side_grasp_pose


def _frame(pose):
    """``(fingers, third, approach, origin)`` of the drawn marker, as world vectors."""
    m = np.array(ee_pose_to_grasp_matrix(pose), dtype=float)
    return m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]


class GraspMatrixFromEEPose(unittest.TestCase):

    def test_len4_is_top_down_with_fingers_along_rotation(self):
        # The user's reported pose. rotation is the closing direction about world Z, and
        # the prompt's own definition says the fingers travel along it.
        fingers, _, approach, origin = _frame([-0.285, 0.017, 0.692, 0.851])
        np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
        np.testing.assert_allclose(origin, [-0.285, 0.017, 0.692], atol=1e-12)
        self.assertAlmostEqual(abs(float(np.dot(fingers, [math.cos(0.851), math.sin(0.851), 0.0]))), 1.0, places=9)

    def test_len4_fingers_track_the_rotation_value(self):
        for rotation in np.linspace(-math.pi, math.pi, 17):
            fingers, _, approach, _ = _frame(grasp_pose(0.1, 0.2, 0.3, rotation))
            np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
            self.assertAlmostEqual(fingers[2], 0.0, places=9)
            expected = np.array([math.cos(rotation), math.sin(rotation), 0.0])
            self.assertAlmostEqual(abs(float(np.dot(fingers, expected))), 1.0, places=9)

    def test_len6_side_grasp_approaches_horizontally_with_horizontal_fingers(self):
        for azimuth in (0.0, math.pi / 2, -math.pi / 2, 1.1, -2.4):
            pose = side_grasp_pose(0.4, 0.5, 0.6, 0.0, azimuth)
            fingers, _, approach, origin = _frame(pose)
            np.testing.assert_allclose(approach, [math.cos(azimuth), math.sin(azimuth), 0.0], atol=1e-9)
            # roll=0 pinches a vertical bar: fingers stay in the horizontal plane.
            self.assertAlmostEqual(fingers[2], 0.0, places=9)
            self.assertAlmostEqual(float(np.dot(fingers, approach)), 0.0, places=9)
            np.testing.assert_allclose(origin, [0.4, 0.5, 0.6], atol=1e-12)

    def test_len6_roll_rotates_the_fingers_out_of_the_horizontal_plane(self):
        pose = side_grasp_pose(0.4, 0.5, 0.6, math.pi / 2, 0.0)
        fingers, _, approach, _ = _frame(pose)
        np.testing.assert_allclose(approach, [1, 0, 0], atol=1e-9)
        self.assertAlmostEqual(abs(fingers[2]), 1.0, places=9)

    def test_frame_is_orthonormal_and_right_handed(self):
        for pose in ([0.1, 0.2, 0.3, 0.7], side_grasp_pose(0.1, 0.2, 0.3, 0.4, 1.2)):
            fingers, third, approach, _ = _frame(pose)
            for axis in (fingers, third, approach):
                self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)
            np.testing.assert_allclose(np.cross(fingers, third), approach, atol=1e-9)

    def test_fingers_match_the_axis_the_robot_really_closes_along(self):
        """The marker's red axis must be the end-effector +Y, not some other column."""
        for pose in ([0.1, 0.2, 0.3, -1.3], side_grasp_pose(0.0, 0.5, 0.4, 0.3, -0.9)):
            euler = common_utils.pose_euler(pose)
            from sim_adapter import transforms
            r = np.array(transforms.matrix_from_euler(euler)).reshape(3, 3)
            fingers, _, approach, _ = _frame(pose)
            np.testing.assert_allclose(fingers, r[:, 1], atol=1e-9)
            np.testing.assert_allclose(approach, r[:, 2], atol=1e-9)

    def test_rejects_other_pose_lengths(self):
        with self.assertRaises(ValueError):
            ee_pose_to_grasp_matrix([0.1, 0.2, 0.3])


class MarkerDepth(unittest.TestCase):
    """The drawn TCP must land where ``Robot._apply_gripper_depth_offset`` puts the hand."""

    def test_top_down_tcp_sits_a_depth_offset_below_the_commanded_point(self):
        import robot as robot_module

        pose = [-0.285, 0.017, 0.692, 0.851]
        offset = config.gripper_depth_offset_franka
        _, _, approach, origin = _frame(pose)
        drawn_tcp = origin + approach * offset

        moved = robot_module.Robot._apply_gripper_depth_offset(
            None, pose[:3], common_utils.pose_euler(pose), offset)
        np.testing.assert_allclose(drawn_tcp, moved, atol=1e-9)
        self.assertAlmostEqual(float(drawn_tcp[2]), 0.692 - offset, places=9)

    def test_side_grasp_tcp_follows_the_horizontal_approach(self):
        import robot as robot_module

        pose = side_grasp_pose(0.0, 0.45, 0.45, 0.0, math.pi / 2)
        offset = config.gripper_depth_offset_franka
        _, _, approach, origin = _frame(pose)
        moved = robot_module.Robot._apply_gripper_depth_offset(
            None, pose[:3], common_utils.pose_euler(pose), offset)
        np.testing.assert_allclose(origin + approach * offset, moved, atol=1e-9)


class MarkerContactDepth(unittest.TestCase):
    """``tcp_depth`` is a *contact* depth, not the IK offset - they differ per robot.

    Measured in PyBullet off the loaded URDFs: franka's ``panda_grasptarget`` (the IK
    target) sits exactly at the finger pad centre, while sawyer's ``right_hand`` is a bare
    wrist frame with the robotiq 2f-85 bolted on 0.168 m further along the approach. Using
    ``gripper_depth_offset`` alone drew the sawyer marker 16.8 cm away from the pads, on the
    opposite side of the commanded point.
    """

    def test_franka_contact_depth_is_the_ik_offset(self):
        import env
        from robot_profiles import get_robot_profile

        profile = get_robot_profile("franka")
        self.assertAlmostEqual(profile.ee_to_finger_contact, 0.0, places=9)
        self.assertAlmostEqual(env.ee_grasp_marker_depth(profile),
                               config.gripper_depth_offset_franka, places=9)

    def test_sawyer_contact_depth_accounts_for_the_detached_robotiq_hand(self):
        import env
        from robot_profiles import get_robot_profile

        profile = get_robot_profile("sawyer")
        depth = env.ee_grasp_marker_depth(profile)
        self.assertAlmostEqual(depth, config.gripper_depth_offset_sawyer + 0.168, places=9)
        # The pads end up *ahead* of the commanded point even though the IK offset is
        # negative: a marker drawn at the raw gripper_depth_offset would be on the wrong
        # side entirely.
        self.assertGreater(depth, 0.0)
        self.assertGreater(abs(depth - profile.gripper_depth_offset), 0.15)

    def test_marker_dimensions_come_from_the_active_robot(self):
        from robot_profiles import get_robot_profile

        franka, sawyer = get_robot_profile("franka"), get_robot_profile("sawyer")
        self.assertNotEqual(franka.finger_half_spread, sawyer.finger_half_spread)
        for profile in (franka, sawyer):
            self.assertGreater(profile.finger_length, 0.0)
            self.assertGreater(profile.finger_half_spread, 0.0)


class PayloadNormalisation(unittest.TestCase):
    """What the model is allowed to pass to ``visualize_grasp_pose``."""

    def test_single_len4_pose(self):
        kind, poses = normalize_grasp_viz_poses([-0.285, 0.017, 0.692, 0.851])
        self.assertEqual(kind, "ee")
        self.assertEqual(len(poses), 1)
        np.testing.assert_allclose(poses[0], [-0.285, 0.017, 0.692, 0.851])

    def test_single_len6_pose(self):
        kind, poses = normalize_grasp_viz_poses(side_grasp_pose(0.1, 0.2, 0.3, 0.0, 1.0))
        self.assertEqual(kind, "ee")
        self.assertEqual(len(poses[0]), 6)

    def test_list_of_poses(self):
        kind, poses = normalize_grasp_viz_poses([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 1.0]])
        self.assertEqual((kind, len(poses)), ("ee", 2))

    def test_mixed_length_list(self):
        kind, poses = normalize_grasp_viz_poses(
            [[0.1, 0.2, 0.3, 0.0], side_grasp_pose(0.4, 0.5, 0.6, 0.0, 1.0)])
        self.assertEqual((kind, len(poses[0]), len(poses[1])), ("ee", 4, 6))

    def test_single_4x4_matrix_is_read_as_a_matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [0.1, 0.2, 0.3]
        kind, poses = normalize_grasp_viz_poses(matrix)
        self.assertEqual((kind, len(poses)), ("matrix", 1))

    def test_four_top_down_poses_are_not_mistaken_for_a_matrix(self):
        """The one ambiguous shape: a (4,4) whose last row is not [0,0,0,1]."""
        kind, poses = normalize_grasp_viz_poses([[0.1, 0.2, 0.3, 0.0]] * 4)
        self.assertEqual((kind, len(poses)), ("ee", 4))

    def test_stack_of_matrices(self):
        kind, poses = normalize_grasp_viz_poses(np.stack([np.eye(4)] * 7))
        self.assertEqual((kind, len(poses)), ("matrix", 7))

    def test_rejects_nonsense_widths(self):
        with self.assertRaises(ValueError):
            normalize_grasp_viz_poses([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


class EnvDispatch(unittest.TestCase):
    """The IPC payload the env subprocess unpacks."""

    def test_tagged_payload_round_trip(self):
        import env

        kind, poses, desc = env._parse_grasp_viz_payload(
            {"kind": "ee", "poses": [[0.1, 0.2, 0.3, 0.4]], "desc": "grasp the handle"})
        self.assertEqual((kind, desc), ("ee", "grasp the handle"))
        self.assertEqual(len(poses), 1)

    def test_bare_matrix_payload_stays_supported(self):
        """The GraspGen path used to send a raw array; it must keep rendering as before."""
        import env

        kind, poses, desc = env._parse_grasp_viz_payload(np.stack([np.eye(4)] * 3))
        self.assertEqual((kind, len(poses), desc), ("matrix", 3, ""))

    def test_unknown_kind_is_rejected(self):
        import env

        with self.assertRaises(ValueError):
            env._parse_grasp_viz_payload({"kind": "bogus", "poses": []})


class PromptGating(unittest.TestCase):

    def _prompt(self, enabled):
        import agent_runner

        return agent_runner._build_main_prompt(
            "detect", "detect-initial", [0, 0, 0], "open the door", "coords", "example",
            vis_grasp=enabled)

    def test_section_absent_by_default(self):
        prompt = self._prompt(False)
        self.assertNotIn("visualize_grasp_pose", prompt)
        self.assertNotIn("[INSERT VISUALIZE GRASP TOOL]", prompt)

    def test_disabled_prompt_is_byte_identical_to_the_pre_feature_text(self):
        """A blank line left behind would still bust the LLM cache for every task."""
        from prompts import main_prompt

        without_placeholder = main_prompt.MAIN_PROMPT.replace("[INSERT VISUALIZE GRASP TOOL]\n", "")
        self.assertNotIn("[INSERT VISUALIZE GRASP TOOL]", without_placeholder)
        rendered = self._prompt(False)
        self.assertNotIn("\n\n[INSERT SKILL TOOLS]", rendered)

    def test_section_present_when_enabled(self):
        prompt = self._prompt(True)
        self.assertIn("visualize_grasp_pose", prompt)
        self.assertIn("[INSERT VISUALIZE GRASP TOOL]", main_prompt_template())

    def test_section_does_not_leak_side_approach_vocabulary(self):
        """``tests/test_side_approach.py`` guards MAIN_PROMPT; the section must comply too."""
        from prompts.main_prompt import VISUALIZE_GRASP_TOOL

        for leaked in ("side_grasp_pose", "side approach", "horizontal (side)"):
            self.assertNotIn(leaked, VISUALIZE_GRASP_TOOL)


def main_prompt_template():
    from prompts.main_prompt import MAIN_PROMPT
    return MAIN_PROMPT


class ApiSurface(unittest.TestCase):
    """``--vis-grasp`` off must stay a pure no-op: no IPC traffic at all."""

    def _api(self, vis_grasp):
        import api as api_module

        sent = []
        stub = api_module.API.__new__(api_module.API)
        stub.args = types.SimpleNamespace(vis_grasp=vis_grasp)
        stub.logger = types.SimpleNamespace(info=lambda *a, **k: None)
        stub.main_connection = types.SimpleNamespace(
            send=lambda payload: sent.append(payload),
            recv=lambda: ["ok"],
        )
        return stub, sent

    def test_noop_without_the_flag(self):
        stub, sent = self._api(False)
        stub.visualize_grasp_pose([0.1, 0.2, 0.3, 0.4])
        stub.clear_grasp_markers()
        self.assertEqual(sent, [])

    def test_sends_tagged_ee_payload(self):
        stub, sent = self._api(True)
        stub.visualize_grasp_pose([-0.285, 0.017, 0.692, 0.851], desc="door handle")
        self.assertEqual(len(sent), 1)
        message, payload = sent[0]
        self.assertEqual(message, config.VISUALIZE_GRASP_POSE)
        self.assertEqual(payload["kind"], "ee")
        self.assertEqual(payload["desc"], "door handle")
        np.testing.assert_allclose(payload["poses"][0], [-0.285, 0.017, 0.692, 0.851])

    def test_clear_sends_the_clear_message(self):
        stub, sent = self._api(True)
        stub.clear_grasp_markers()
        self.assertEqual(sent, [[config.CLEAR_GRASP_MARKERS]])


if __name__ == "__main__":
    unittest.main()
