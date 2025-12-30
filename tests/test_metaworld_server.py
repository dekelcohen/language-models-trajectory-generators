import os
import sys
import json
import subprocess
import time
import shutil
import unittest
import importlib
# Ensure repo root on sys.path for direct execution
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
config = importlib.import_module('config')

# Single global: timeout in seconds (default 15.0). <=0 means no timeout.
TEST_TIMEOUT_SECS = 15.0
 

class TestMetaworldServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure images dirs exist and are clean (reuse util)
        _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from common_utils import ensure_image_dirs_exist
        ensure_image_dirs_exist(delete=True)
        # Use the same interpreter/environment used to run tests
        cls.python = sys.executable
        env = os.environ.copy()
        # Heuristic: if METAWORLD_REPO not set, try common sibling locations
        if 'METAWORLD_REPO' not in env:
            guesses = [
                os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'Simulators_Envs', 'Metaworld')),
                os.path.abspath(os.path.join(os.getcwd(), '..', 'Simulators_Envs', 'Metaworld')),
            ]
            for guess in guesses:
                if os.path.isdir(guess):
                    env['METAWORLD_REPO'] = guess
                    break
        # Start WS server in a background process
        cmd = [cls.python, 'providers/metaworld_server.py', '--env', 'sawyer_door_v3', '--ws-host', '127.0.0.1', '--ws-port', '8899']
        # Always pass timeout to the server: <=0 encoded as '0' (no timeout)
        to_arg = '0' if float(TEST_TIMEOUT_SECS) <= 0 else str(float(TEST_TIMEOUT_SECS))
        cmd += ['--timeout', to_arg]
        cls.proc = subprocess.Popen(
            cmd,
            stdin=None,
            stdout=None,
            stderr=None,
            env=env,
        )
        # Connect WS client with a single unified timeout value
        from providers.ws_connection import WsJSONConnection
        # Connection-level override: negative means infinite for connect/send/close
        conn_to = -1.0 if float(TEST_TIMEOUT_SECS) <= 0 else float(TEST_TIMEOUT_SECS)
        cls._timeout = conn_to
        cls.conn = WsJSONConnection('ws://127.0.0.1:8899', timeout=conn_to)
        # Read ready banner (Queue API needs None for infinite)
        ready = cls.conn.recv(timeout=None if conn_to < 0 else conn_to)
        if not (isinstance(ready, dict) and ready.get('status') == 'ready'):
            raise RuntimeError('Did not receive WS ready banner')

    @classmethod
    def tearDownClass(cls):
        try:
            cls.conn.close()
        except Exception:
            pass
        try:
            cls.proc.kill()
        except Exception:
            pass

    def _rpc(self, obj):
        # Use WS connection
        self.__class__.conn.send([obj.get('cmd'), obj.get('args')])
        to = None if self.__class__._timeout < 0 else self.__class__._timeout
        return self.__class__.conn.recv(timeout=to)

    def test_capture_images_and_camera_info(self):
        # Capture once
        _ = self._rpc({"cmd": config.CAPTURE_IMAGES, "args": None})
        # Check files exist
        self.assertTrue(os.path.exists('./images/rgb_image_head.png'))
        self.assertTrue(os.path.exists('./images/depth_image_head.png'))
        resp = self._rpc({"cmd": config.GET_CAMERA_INFO, "args": None})
        self.assertIn('head', resp)
        K = resp['head']['K']
        self.assertEqual(len(K), 3)

    def test_get_state_and_annotation(self):
        state = self._rpc({"cmd": config.GET_STATE, "args": {"objects": ["handle"]}})
        self.assertIn('eef_pos', state)
        # Annotate (throttled)
        resp = self._rpc({"cmd": config.CAPTURE_ANNOTATED_IMAGES, "args": {"objects": ["handle"]}})
        self.assertIn('logged', resp)

    def test_move_eef_abs_and_reprojection(self):
        state = self._rpc({"cmd": config.GET_STATE, "args": None})
        start = state['eef_pos']
        target = [start[0] + 0.02, start[1], start[2]]
        res = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": target, "iters": 40, "open_gripper": True}})
        self.assertLess(res['pos_err'], 0.02)
        # Step and ensure not terminated immediately
        res2 = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 5}})
        self.assertIn('terminated', res2)

    def test_open_door_script(self):
        # Try to approach and grasp; then pull handle toward the goal position
        state0 = self._rpc({"cmd": config.GET_STATE, "args": {"objects": ["handle", "goal"]}})
        print(f'*** test_open_door_script after GET_STATE state0={state0}')
        objs = state0.get('objects', {}) or {}
        handle = (objs.get('handle') or {}).get('pos')
        goal = (objs.get('goal') or {}).get('pos')
        if handle and goal:
            hx, hy, hz = handle
            gx, gy, gz = goal
            # Approach above the handle slightly to avoid collisions
            approach = [hx, hy, hz + 0.02]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": approach, "iters": 100, "open_gripper": True}})
            # Descend to contact and close
            contact = [hx, hy, hz]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": contact, "iters": 80, "open_gripper": True}})
            _ = self._rpc({"cmd": config.CLOSE_GRIPPER, "args": None})

            # Curved half‑circle style pull:
            # Phase A: pull inward toward robot base (negative X), keep Y almost constant
            # Phase B: sweep left (negative Y) while continuing slight inward pull
            import math
            dx, dy = (gx - hx), (gy - hy)
            total = max(1e-6, math.hypot(dx, dy))
            # Split the journey: ~70% inward first, then leftward
            inward_x = dx * 0.7
            inward_y = dy * 0.2
            sweep_x = dx - inward_x
            sweep_y = dy - inward_y

            def _linspace(n):
                return [i / float(max(1, n)) for i in range(1, n + 1)]

            # Phase A: 8 small inward steps
            for t in _linspace(8):
                tx = hx + inward_x * t
                ty = hy + inward_y * t
                tgt = [tx, ty, hz]
                _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": tgt, "iters": 70, "open_gripper": False}})
                _ = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 2}})

            # Phase B: 8 step sweep left, adding a gentle inward bias
            for t in _linspace(8):
                tx = hx + inward_x + sweep_x * t
                ty = hy + inward_y + sweep_y * t
                tgt = [tx, ty, hz]
                _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": tgt, "iters": 70, "open_gripper": False}})
                _ = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 2}})
        # Step a bit to allow env to evaluate the success condition
        _ = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 10}})
                        
        # Export a trajectory video for visual inspection BEFORE assertion
        # Create video directly (no WS RPC)
        from debug.dbg_utils import create_video_from_images
        create_video_from_images(
            folder_path=config.trajectory_folder,
            base_name=config.trajectory_image_base,
            start_idx=0,
            end_idx=float('inf'),
            fps=config.trajectory_video_fps,
        )
        expected_video = os.path.join(config.trajectory_folder, f"{config.trajectory_image_base}_0_inf.mp4")
        self.assertTrue(os.path.exists(expected_video))

        # Now query env success flag
        res = self._rpc({"cmd": config.QUERY_ENV_ATTR, "args": {"name": "reachCompleted"}})
        # Must exist and be True for success
        self.assertIn('value', res)
        self.assertIsNotNone(res['value'])
        self.assertIsInstance(res['value'], bool)
        self.assertTrue(res['value'])

    def test_make_trajectory_video(self):
        """Generate a short trajectory and export a video to aid debugging."""
        # Take a few small steps to ensure frames are saved
        state = self._rpc({"cmd": config.GET_STATE, "args": None})
        start = state['eef_pos']
        for i in range(5):
            tgt = [start[0] + 0.01 * (i + 1), start[1], start[2]]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": tgt, "iters": 20, "open_gripper": True}})
        # Create the video from saved frames
        from debug.dbg_utils import create_video_from_images
        create_video_from_images(
            folder_path=config.trajectory_folder,
            base_name=config.trajectory_image_base,
            start_idx=0,
            end_idx=float('inf'),
            fps=config.trajectory_video_fps,
        )
        expected_video = os.path.join(config.trajectory_folder, f"{config.trajectory_image_base}_0_inf.mp4")
        self.assertTrue(os.path.exists(expected_video))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--timeout', type=float, default=15.0, help='Timeout seconds; <=0 disables timeouts')
    args, rest = p.parse_known_args()
    # Apply CLI override
    TEST_TIMEOUT_SECS = float(args.timeout)
    # Rebuild argv for unittest, dropping our custom flag
    sys.argv = [sys.argv[0]] + rest
    unittest.main()
