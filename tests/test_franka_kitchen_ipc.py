"""IPC audit for the Franka Kitchen sim-env.

Walks every message ``env.run_simulation_environment`` handles and asserts it is
meaningful for the kitchen scene, not just that it fails to crash. In particular
it pins down the two things that are easy to break when adding a new sim-env:

* the handshake must carry the kitchen's own coordinate prompt and task state;
* ``RESET_EEF`` must re-home the arm to a pose that is actually above the
  counter (the repo-wide default ``config.ee_start_position`` is below it).
"""

import multiprocessing as mp
import traceback
import unittest

import numpy as np

import config
from sim_envs.pybullet.franka_kitchen.simenv import ROBOT_EE_START_POSITION


TASK = "franka_kitchen:microwave"

# Counter surface height in this scene; anything at or below it is inside the
# cabinetry rather than above it.
COUNTER_TOP_Z = 1.18


class _Args:
    mode = "default"
    robot = "sawyer"  # deliberately wrong: the sim-env must force franka
    task = TASK
    sim = "pybullet"
    save_grasp_inputs = False


def _run_env(conn):
    import logging
    from env import run_simulation_environment

    logger = logging.getLogger("env")
    logger.setLevel(logging.INFO)
    try:
        run_simulation_environment(_Args, conn, logger)
    except Exception:
        traceback.print_exc()


class TestKitchenIPC(unittest.TestCase):
    def setUp(self):
        self.parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=_run_env, args=(child_conn,), daemon=True)
        self.proc.start()
        self.handshake = self.parent_conn.recv()

    def tearDown(self):
        try:
            self.parent_conn.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.join(timeout=10)
        except Exception:
            pass

    def _request(self, *message):
        self.parent_conn.send(list(message))
        return self.parent_conn.recv()

    def test_handshake_reports_kitchen_state_and_coords(self):
        eef_pos, coords_section, sim_state, message = self.handshake

        self.assertGreater(
            eef_pos[2], COUNTER_TOP_Z,
            f"end-effector starts at z={eef_pos[2]:.3f}, inside/below the counter",
        )
        # The head camera is pinned to an axis-aligned pose, so the prompt must
        # describe exactly that mapping.
        self.assertIn("x-axis", coords_section)
        self.assertIn("y-axis", coords_section)
        self.assertIn("z-axis", coords_section)

        self.assertEqual(sim_state.get("sim_env"), "franka_kitchen")
        self.assertEqual(sim_state.get("task"), "microwave")
        self.assertIsNotNone(sim_state.get("task_error"))
        self.assertFalse(sim_state.get("success"), "task must not start already solved")
        self.assertIsNotNone(sim_state.get("target_link_pos"))
        self.assertIn("Finished setting up environment", message)

    def test_capture_images_returns_usable_camera_info(self):
        reply = self._request(config.CAPTURE_IMAGES)
        head_pos, head_orientation_q, wrist_pos, wrist_orientation_q, message, cam_info = reply

        self.assertIn("Finished capturing", message)
        self.assertIsNotNone(cam_info["head"]["viewMatrix"])
        self.assertIsNotNone(cam_info["head"]["projectionMatrix"])
        self.assertTrue(cam_info.get("new_3d_proj"))

        # The head camera must sit on -x looking towards +x, which is what the
        # coordinates prompt promises.
        self.assertLess(head_pos[0], 0.0, f"head camera is not on -x: {head_pos}")
        self.assertGreater(wrist_pos[2], COUNTER_TOP_Z)

    def test_reset_eef_rehomes_above_the_counter(self):
        # Drive the arm somewhere else first, so re-homing has work to do.
        self._request(config.EXECUTE_TRAJECTORY, [[0.40, 0.30, 1.45, 0.0]])
        moved = self._request(config.GET_STATE)["eef_pos"]

        message = self._request(config.RESET_EEF)[0]
        self.assertIn("Finished re-homing", message)

        homed = self._request(config.GET_STATE)["eef_pos"]
        self.assertGreater(
            homed[2], COUNTER_TOP_Z,
            f"RESET_EEF put the end-effector at z={homed[2]:.3f}, through the counter",
        )
        # Compare against the sim-env constant, not config.ee_start_position:
        # the sim-env rewrites that at startup inside the *child* process, so the
        # module-level default is still the repo-wide one here.
        self.assertLess(
            float(np.linalg.norm(np.array(homed) - np.array(ROBOT_EE_START_POSITION))),
            0.1,
            f"RESET_EEF left the arm at {homed}, not at {ROBOT_EE_START_POSITION}",
        )
        self.assertGreater(
            float(np.linalg.norm(np.array(homed) - np.array(moved))),
            0.05,
            "arm never left the home pose, so re-homing was not exercised",
        )

    def test_grippers_and_task_completed(self):
        self.parent_conn.send([config.OPEN_GRIPPER])
        self.parent_conn.send([config.CLOSE_GRIPPER])
        # Neither gripper command replies; the next request round-trips only if
        # both were consumed correctly.
        state = self._request(config.GET_STATE)
        self.assertIn("eef_pos", state)
        self.assertIn("Finished executing all generated trajectories", self._request(config.TASK_COMPLETED)[0])

    def test_execute_trajectory_moves_the_arm_and_updates_state(self):
        before = self._request(config.GET_STATE)
        reply = self._request(
            config.EXECUTE_TRAJECTORY,
            [[0.40, 0.25, 1.50, 0.0], [0.42, 0.40, 1.42, 0.0]],
        )
        self.assertIn("Finished executing generated trajectory", reply[0])

        after = self._request(config.GET_STATE)
        self.assertGreater(
            float(np.linalg.norm(np.array(after["eef_pos"]) - np.array(before["eef_pos"]))),
            0.05,
            "EXECUTE_TRAJECTORY did not move the end-effector",
        )
        # Kitchen state must stay live, not be a stale snapshot from the handshake.
        self.assertEqual(after["sim_state"]["task"], "microwave")
        self.assertIsNotNone(after["sim_state"]["task_error"])

    def test_add_bounding_cubes_and_trajectory_points(self):
        cube = [[0.4, 0.4, 1.4]] * 9
        self.assertIn(
            "Finished adding bounding cubes",
            self._request(config.ADD_BOUNDING_CUBES, [cube])[0],
        )
        # ADD_TRAJECTORY_POINTS does not reply; verify the loop stays responsive.
        self.parent_conn.send([config.ADD_TRAJECTORY_POINTS, [[0.4, 0.3, 1.45, 0.0]]])
        self.assertIn("eef_pos", self._request(config.GET_ROBOT_STATE))


class _KettleArgs(_Args):
    task = "franka_kitchen:kettle"


def _run_kettle_env(conn):
    import logging
    from env import run_simulation_environment

    logger = logging.getLogger("env")
    logger.setLevel(logging.INFO)
    try:
        run_simulation_environment(_KettleArgs, conn, logger)
    except Exception:
        traceback.print_exc()


def _line(start, end, n):
    return [list(np.array(start) + (np.array(end) - np.array(start)) * t) for t in np.linspace(0.0, 1.0, n)]


class TestKettleHeldAcrossSubtaskBoundary(unittest.TestCase):
    """Regression: "pick up the kettle and pour ..." - after subtask 1 lifted the kettle
    it ended up back on the stove before subtask 2, with no explicit open_gripper.

    Pinned here, with plain contact physics (no weld): the env keeps stepping while the
    VLM reviews (minutes) and the pinched
    kettle must stay in hand; GET_ROBOT_STATE must measure that the closed fingers are
    blocked by it (gripper_holding); and the next subtask / retry prompt must say the
    gripper is HOLDING an object. Previously the prompt hard-coded "the gripper open" and
    a false "grasp failed" review made the retry call open_gripper(), dropping the kettle.
    """

    # Hover/grasp/lift the planner generated in the failing run (x, y, yaw).
    HANDLE = [0.268, 0.191]
    YAW = -0.013
    GRASP_Z = 1.416
    LIFT_Z = 1.716
    MIN_LIFT = 0.15
    # The VLM review window, during which the sim keeps stepping.
    IDLE_S = 20.0

    def setUp(self):
        self.parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=_run_kettle_env, args=(child_conn,), daemon=True)
        self.proc.start()
        self.handshake = self.parent_conn.recv()

    def tearDown(self):
        try:
            self.parent_conn.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.join(timeout=10)
        except Exception:
            pass

    def _request(self, *message):
        self.parent_conn.send(list(message))
        return self.parent_conn.recv()

    def _kettle_pos(self):
        return np.array(self._request(config.GET_STATE)["sim_state"]["kettle_pos"])

    def test_kettle_stays_in_hand_and_next_subtask_is_told_gripper_closed(self):
        import logging
        import time
        import agent_runner

        logger = logging.getLogger("test")
        rest = self._kettle_pos()
        eef = self._request(config.GET_ROBOT_STATE)
        self.assertTrue(eef["gripper_open"], "gripper must start open")

        # Subtask 1: hover, open, descend, close, lift.
        hover = self.HANDLE + [1.55, self.YAW]
        self._request(config.EXECUTE_TRAJECTORY, _line(eef["eef_pos"] + [0.0], hover, 15))
        self.parent_conn.send([config.OPEN_GRIPPER])
        grasp = self.HANDLE + [self.GRASP_Z, self.YAW]
        self._request(config.EXECUTE_TRAJECTORY, _line(hover, grasp, 12))
        self.parent_conn.send([config.CLOSE_GRIPPER])
        self._request(config.EXECUTE_TRAJECTORY, _line(grasp, self.HANDLE + [self.LIFT_Z, self.YAW], 20))
        lifted = self._kettle_pos()
        self.assertGreater(lifted[2] - rest[2], self.MIN_LIFT, f"kettle not lifted: {rest} -> {lifted}")

        # The VLM review between subtasks: the sim keeps stepping.
        time.sleep(self.IDLE_S)

        # Subtask boundary: exactly what execute_task() queries to build the next prompt.
        robot_state = agent_runner.log_world_state(self.parent_conn, logger, "subtask start")
        self.assertIs(robot_state["gripper_open"], False)
        self.assertIs(robot_state["gripper_holding"], True,
                      f"closed fingers not blocked by the kettle: width={robot_state['gripper_width']}")
        self.parent_conn.send([config.GET_ROBOT_STATE, {}])
        raw = self.parent_conn.recv()
        self.assertGreater(raw["eef_pos"][2] - raw["eef_pos_at_close"][2], self.MIN_LIFT,
                           f"lift since the grasp not measured: {raw}")
        prompt = agent_runner._build_main_prompt(
            "detect", "detect-initial", robot_state["eef_pos"], "pour the kettle", "coords", "",
            gripper_open=robot_state["gripper_open"], gripper_holding=robot_state["gripper_holding"],
            gripper_width=robot_state["gripper_width"], base_pos=robot_state["base_pos"])
        self.assertIn("and the gripper CLOSED and HOLDING an object", prompt)
        self.assertNotIn("and the gripper open.", prompt)
        # The pour skill caps the hand height relative to the measured arm base.
        self.assertIn(f"(the robot arm's base is mounted at {robot_state['base_pos']})", prompt)
        self.assertAlmostEqual(robot_state["base_pos"][2], 1.25, delta=0.05)

        # The held kettle must survive the idle review and the boundary queries.
        after = self._kettle_pos()
        self.assertLess(float(np.linalg.norm(after - lifted)), 0.02,
                        f"kettle moved while held across the subtask boundary: {lifted} -> {after}")
        self.assertGreater(after[2] - rest[2], self.MIN_LIFT, f"kettle dropped: {after}")

        # Pour evidence for the review: tilt toward +y, tilt back upright; the largest measured
        # tilt (and the side that went down) is kept after the hand is upright again.
        import api as api_module
        from common_utils import pour_pose
        hand = self.HANDLE + [self.LIFT_Z]
        tilts = np.linspace(0.0, 0.8, 15)
        tilt_in = [pour_pose(*hand, self.YAW, float(t), self.YAW + np.pi / 2) for t in tilts]
        self._request(config.EXECUTE_TRAJECTORY, tilt_in)
        self._request(config.EXECUTE_TRAJECTORY, tilt_in[::-1])
        poured = self._request(config.GET_ROBOT_STATE)
        tilt = poured["max_tilt_since_close"]
        self.assertTrue(poured["gripper_holding"], f"kettle lost while tilting: {poured}")
        self.assertAlmostEqual(tilt["deg"], np.degrees(0.8), delta=8, msg=f"{tilt}")
        self.assertGreater(tilt["dip_dir"][1], 0.9, f"lowered side should be +y: {tilt}")
        self.assertIn("tipped toward +y", api_module.API._tilt_while_held(poured))
        after = self._kettle_pos()

        # open_gripper still releases it.
        self.parent_conn.send([config.OPEN_GRIPPER])
        time.sleep(2.0)
        self.assertLess(self._kettle_pos()[2], after[2] - 0.1, "kettle still held after open_gripper")

        # Closing on nothing measures EMPTY, not holding.
        self.parent_conn.send([config.CLOSE_GRIPPER])
        empty = agent_runner.query_robot_state(self.parent_conn, logger)
        self.assertIs(empty["gripper_holding"], False, f"empty gripper reported holding: {empty}")


class TestGripperStatePrompt(unittest.TestCase):
    def test_open_gripper_prompt_is_unchanged(self):
        """Open/unknown gripper keeps the historical wording (LLM-cache keys unaffected)."""
        import agent_runner

        for state in (True, None):
            prompt = agent_runner._build_main_prompt(
                "detect", "detect-initial", [0, 0, 0], "task", "coords", "", gripper_open=state)
            self.assertIn("with the rotation value at 0, and the gripper open.\n", prompt)
            self.assertNotIn("[INSERT GRIPPER STATE]", prompt)

    def test_review_gets_measured_gripper_state(self):
        """The reviewer is told what the width sensor measured (it hallucinated an empty
        gripper lifting away while the wrist cam showed the kettle in hand)."""
        import api as api_module

        class _Conn:
            def __init__(self, reply):
                self.reply = reply

            def send(self, msg):
                assert msg[0] == config.GET_ROBOT_STATE

            def recv(self):
                return self.reply

        class _Log:
            def info(self, *_):
                pass

        def evidence(reply):
            stub = object.__new__(api_module.API)
            stub.main_connection, stub.logger = _Conn(reply), _Log()
            return stub._gripper_review_evidence()

        held = evidence({"gripper_open": False, "gripper_width": 0.0256, "gripper_holding": True})
        self.assertIn("IS HOLDING", held)
        self.assertIn("2.6 cm", held)
        lifted = evidence({"gripper_open": False, "gripper_width": 0.0256, "gripper_holding": True,
                           "eef_pos": [0.29, 0.16, 1.54], "eef_pos_at_close": [0.29, 0.16, 1.39]})
        self.assertIn("carried UP by ~15 cm", lifted)
        still = evidence({"gripper_open": False, "gripper_width": 0.0256, "gripper_holding": True,
                          "eef_pos": [0.29, 0.16, 1.40], "eef_pos_at_close": [0.29, 0.16, 1.39]})
        self.assertIn("has NOT moved since the fingers closed", still)
        empty = evidence({"gripper_open": False, "gripper_width": 0.001, "gripper_holding": False})
        self.assertIn("NOTHING is held", empty)
        self.assertIsNone(evidence({"gripper_open": False, "gripper_width": None}))

    def test_planner_result_carries_measured_gripper_state(self):
        """A reviewer false negative must not make the planner re-grasp a held object."""
        import agent_runner

        held = agent_runner.robot_state_sentence(
            {"gripper_open": False, "gripper_holding": True, "gripper_width": 0.023})
        self.assertIn("HOLDING an object", held)
        self.assertIn("Do not dispatch a new grasp", held)
        self.assertIn("CLOSED on nothing", agent_runner.robot_state_sentence(
            {"gripper_open": False, "gripper_holding": False, "gripper_width": 0.001}))
        self.assertIsNone(agent_runner.robot_state_sentence({"gripper_open": None}))
        summary = agent_runner.TaskResult(False, 2, [], robot_state_after=held).as_summary_dict()
        self.assertEqual(summary["robot_state_after"], held)
        self.assertNotIn("robot_state_after", agent_runner.TaskResult(True, 1, []).as_summary_dict())


if __name__ == "__main__":
    unittest.main()
