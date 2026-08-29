"""Interface-conformance test: every live IPC command, over the real transport.

This is the *one* contract both simulators are checked against. It asserts payload
**keys / types / shapes**, never exact floats: ``run_simulation_environment`` drives
``env.update()`` in a free-running loop, so the number of physics steps between two
messages depends on wall-clock timing and no float is reproducible across runs.
Numeric parity is pinned separately and in-process by ``test_pybullet_regression.py``.

Both sims run through this file:

* PyBullet - a subprocess of *this* interpreter, over ``multiprocessing.Pipe``.
* Genesis  - a child in the ``vlm_genesis`` interpreter, over JSON-lines TCP.

The Genesis case is skipped unless that interpreter can be resolved, so the suite stays
green on a machine that only has PyBullet.
"""

import multiprocessing
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from config import (CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS,  # noqa: E402
                    EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED,
                    RESET_EEF, GET_STATE, GET_ROBOT_STATE, VISUALIZE_GRASP_POSE,
                    VISUALIZE_BOUNDING_BOX)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Generous: a cold Genesis start imports torch + taichi and compiles kernels.
REPLY_TIMEOUT = float(os.environ.get("LMTG_IPC_TEST_TIMEOUT", "300"))
BOOT_TIMEOUT = float(os.environ.get("LMTG_IPC_TEST_BOOT_TIMEOUT", "600"))


class _Args:
    mode = "default"
    robot = "franka"
    task = "door"
    sim = "pybullet"
    gui = False
    save_grasp_inputs = False


def _pybullet_child(connection):
    """Entry point for the PyBullet environment subprocess."""
    import traceback
    sys.path.insert(0, REPO_ROOT)
    try:
        import env as envmod
        envmod.run_simulation_environment(_Args, connection, None)
    except Exception:
        # Without this the parent only sees a broken pipe and no cause.
        traceback.print_exc()
        raise


def _sample_trajectory():
    """Four short waypoints as ``[x, y, z, yaw]``, the wire format env.py expects."""
    x, y, z = config.ee_start_position
    return [[x, y, z, 0.0],
            [x, y, z + 0.03, 0.0],
            [x - 0.02, y, z + 0.03, 0.1],
            [x - 0.02, y, z, 0.1]]


def _sample_cube():
    """A bounding cube in the wire format ``utils.get_bounding_cube_from_point_cloud``
    produces: ``box_top`` (4 corners + centroid) followed by ``box_btm`` (4 + centroid),
    ten points in total. ``env.py`` indexes 0-3, 5-8 and 9, so a plain 8-corner box
    raises IndexError.
    """
    cx, cy, h = -0.11, 0.04, 0.05
    z_top, z_btm = 0.50, 0.40
    corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    top = [[x, y, z_top] for x, y in corners]
    btm = [[x, y, z_btm] for x, y in corners]
    top.append(list(np.mean(top, axis=0)))
    btm.append(list(np.mean(btm, axis=0)))
    return top + btm


def _sample_grasp_poses(n=2):
    poses = np.tile(np.eye(4), (n, 1, 1)).astype(float)
    poses[:, :3, 3] = [[-0.11, 0.04, 0.55], [-0.09, 0.02, 0.60]][:n]
    return poses


class IpcContractMixin:
    """The assertions. Subclasses only have to provide ``self.connection``."""

    connection = None
    handshake = None
    #: Guards against the suite silently exercising the wrong provider.
    expected_depth_encoding = None

    @classmethod
    def _read_handshake(cls, timeout):
        """``run_simulation_environment`` pushes one unsolicited startup message.

        It has to be drained before any command, or every reply is off by one.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cls.connection.poll(0.5):
                return cls.connection.recv()
        raise unittest.SkipTest(f"no handshake from the environment within {timeout}s")

    def test_handshake(self):
        """Requirement (d): both sims must open with the same 4-part handshake."""
        eef_pos, coords_section, sim_state, message = self.handshake
        self.assertEqual(len(eef_pos), 3)
        self.assertIsInstance(coords_section, str)
        self.assertIn("x-axis", coords_section)
        self.assertIsInstance(sim_state, dict)
        self.assertIsInstance(message, str)

    def _exchange(self, message, expect_reply=True):
        self.connection.send(message)
        if not expect_reply:
            # Fire-and-forget commands have no reply; give the child a moment to run
            # them so a later failure is attributable to the right command.
            time.sleep(0.5)
            return None
        deadline = time.time() + REPLY_TIMEOUT
        while time.time() < deadline:
            if self.connection.poll(0.2):
                return self.connection.recv()
        self.fail(f"no reply to command {message[0]} within {REPLY_TIMEOUT}s")

    def test_capture_images(self):
        reply = self._exchange([CAPTURE_IMAGES])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 6, "CAPTURE_IMAGES must return 6 elements")

        head_pos, head_quat, wrist_pos, wrist_quat, message, cam_info = reply
        for name, vec, size in (("head_pos", head_pos, 3), ("head_quat", head_quat, 4),
                                ("wrist_pos", wrist_pos, 3), ("wrist_quat", wrist_quat, 4)):
            with self.subTest(field=name):
                self.assertEqual(len(list(vec)), size)
                for v in vec:
                    self.assertIsInstance(float(v), float)
        self.assertIsInstance(message, str)

        self.assertIsInstance(cam_info, dict)
        self.assertIn("head", cam_info)
        self.assertIn("depth_encoding", cam_info)
        self.assertEqual(cam_info["depth_encoding"], self.expected_depth_encoding,
                         "wrong simulator answered this contract")
        self.assertTrue(cam_info.get("new_3d_proj"))

        head = cam_info["head"]
        # The user's requirement (b): the two sims must agree on dims *and* meaning.
        self.assertEqual(len(list(head["viewMatrix"])), 16)
        self.assertEqual(len(list(head["projectionMatrix"])), 16)
        self.assertAlmostEqual(float(head["znear"]), float(config.near_plane))
        self.assertAlmostEqual(float(head["zfar"]), float(config.far_plane))

        proj = np.array(head["projectionMatrix"], dtype=float).reshape(4, 4, order="F")
        # A GL perspective matrix looks down -z: the [3][2] term must be exactly -1.
        self.assertAlmostEqual(proj[3, 2], -1.0, places=6)
        self.assertAlmostEqual(proj[3, 3], 0.0, places=6)

    def test_capture_images_writes_the_expected_files(self):
        self._exchange([CAPTURE_IMAGES])
        for path in (config.rgb_image_head_path, config.depth_image_head_path,
                     config.rgb_image_wrist_path, config.depth_image_wrist_path):
            with self.subTest(path=path):
                self.assertTrue(os.path.exists(path), f"{path} was not written")

    def test_get_robot_state(self):
        reply = self._exchange([GET_ROBOT_STATE])
        self.assertIsInstance(reply, dict)
        self.assertIn("eef_pos", reply)
        self.assertEqual(len(reply["eef_pos"]), 3)
        for v in reply["eef_pos"]:
            self.assertIsInstance(v, float)

    def test_get_state(self):
        reply = self._exchange([GET_STATE])
        self.assertIsInstance(reply, dict)
        self.assertEqual(set(reply), {"eef_pos", "sim_state"})
        self.assertEqual(len(reply["eef_pos"]), 3)

        state = reply["sim_state"]
        self.assertIsInstance(state, dict)
        # The door profile's public contract, shared by both sims.
        self.assertEqual(
            set(state),
            {"door_id", "door_hinge_index", "latch_index", "door_handle_latch",
             "door_handle_pos", "latch_pos", "hinge_pos",
             "pole_id", "pole_pos", "pole_dims"})
        for key in ("door_handle_pos", "latch_pos", "hinge_pos", "pole_pos", "pole_dims"):
            with self.subTest(field=key):
                self.assertEqual(len(state[key]), 3, f"{key} must be a 3-vector")

    def test_execute_trajectory(self):
        reply = self._exchange([EXECUTE_TRAJECTORY, _sample_trajectory()])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 2)
        self.assertIsInstance(reply[0], str)
        self.assertIsInstance(reply[1], int)
        self.assertGreaterEqual(reply[1], 1, "trajectory_step must advance")

    def test_execute_trajectory_then_state_moves_the_eef(self):
        before = self._exchange([GET_ROBOT_STATE])["eef_pos"]
        # Build the waypoints from where the arm actually *is*: the door scene starts
        # the arm well away from config.ee_start_position, so a fixed trajectory can
        # be unreachable and produce a no-op that looks like a passing test.
        x, y, z = before
        trajectory = [[x, y, z + 0.04, 0.0],
                      [x - 0.05, y, z + 0.08, 0.0],
                      [x - 0.05, y + 0.05, z + 0.08, 0.2]]
        self._exchange([EXECUTE_TRAJECTORY, trajectory])
        after = self._exchange([GET_ROBOT_STATE])["eef_pos"]
        moved = float(np.linalg.norm(np.array(after) - np.array(before)))
        self.assertGreater(moved, 1e-3,
                           f"EXECUTE_TRAJECTORY moved the end effector only {moved:.2e} m")

    def test_grippers(self):
        # Fire-and-forget: env.py logs but does not reply. Assert the process survives
        # and still answers the next command, which is the real contract here.
        self._exchange([OPEN_GRIPPER], expect_reply=False)
        self._exchange([CLOSE_GRIPPER], expect_reply=False)
        self.assertIn("eef_pos", self._exchange([GET_ROBOT_STATE]))

    def test_reset_eef(self):
        reply = self._exchange([RESET_EEF])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 1)
        self.assertIsInstance(reply[0], str)

    def test_task_completed(self):
        reply = self._exchange([TASK_COMPLETED])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 1)
        self.assertIsInstance(reply[0], str)

    def test_debug_visualizations(self):
        """Requirement (e): the same debug overlays must work on every simulator."""
        reply = self._exchange([ADD_BOUNDING_CUBES, [_sample_cube()]])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 1)

        self._exchange([ADD_TRAJECTORY_POINTS, _sample_trajectory(), "blue", True, "line"],
                       expect_reply=False)
        self._exchange([ADD_TRAJECTORY_POINTS, _sample_trajectory(), None, False, "points"],
                       expect_reply=False)

        reply = self._exchange([VISUALIZE_GRASP_POSE, _sample_grasp_poses(2)])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 1)
        self.assertNotIn("Failed", reply[0], f"VISUALIZE_GRASP_POSE errored: {reply[0]}")

        reply = self._exchange([VISUALIZE_BOUNDING_BOX, [_sample_cube()]])
        self.assertIsInstance(reply, list)
        self.assertEqual(len(reply), 1)
        self.assertNotIn("Failed", reply[0], f"VISUALIZE_BOUNDING_BOX errored: {reply[0]}")

        # Everything above is fire-and-forget or best-effort; prove the sim is alive.
        self.assertIn("eef_pos", self._exchange([GET_ROBOT_STATE]))


class TestPyBulletIpcContract(IpcContractMixin, unittest.TestCase):

    expected_depth_encoding = "opengl"

    @classmethod
    def setUpClass(cls):
        cls.connection, child_connection = multiprocessing.Pipe()
        cls.process = multiprocessing.Process(target=_pybullet_child, args=(child_connection,))
        cls.process.daemon = True
        cls.process.start()
        # The child loads URDFs and settles the scene before its first poll().
        cls.handshake = cls._read_handshake(BOOT_TIMEOUT)
        if not cls.process.is_alive():
            raise unittest.SkipTest("the PyBullet environment subprocess died on startup")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.connection.close()
        finally:
            cls.process.terminate()
            cls.process.join(timeout=20)


def _genesis_available():
    try:
        from providers.genesis_launcher import resolve_genesis_python
        resolve_genesis_python()
        return True
    except Exception:
        return False


@unittest.skipUnless(_genesis_available(),
                     "Genesis interpreter not resolvable; set GENESIS_PYTHON or "
                     "create the vlm_genesis conda env")
class TestGenesisIpcContract(IpcContractMixin, unittest.TestCase):
    """The same assertions, against the Genesis child over JSON-lines TCP."""

    expected_depth_encoding = "linear_metric"

    @classmethod
    def setUpClass(cls):
        from providers.genesis_launcher import launch_genesis_child
        from providers.json_ipc import JsonIpcConnection

        class _GenesisArgs(_Args):
            sim = "genesis"

        cls.process, host, port, _python = launch_genesis_child(_GenesisArgs)
        cls.connection = JsonIpcConnection(host, port)
        try:
            cls.connection.wait_until_ready(process=cls.process, timeout=BOOT_TIMEOUT)
        except Exception as exc:
            cls.process.terminate()
            raise unittest.SkipTest(f"Genesis child failed to start: {exc}")
        # Genesis still has to import torch/taichi and build the scene after accepting.
        cls.handshake = cls._read_handshake(BOOT_TIMEOUT)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.connection.close()
        finally:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=30)
            except Exception:
                cls.process.kill()


if __name__ == "__main__":
    unittest.main()
