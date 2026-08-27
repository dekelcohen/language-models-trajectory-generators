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


if __name__ == "__main__":
    unittest.main()
