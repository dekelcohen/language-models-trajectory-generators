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
        # Ensure images dirs exist and are clean
        for p in ['./images', './images/trajectory', './images/overlay']:
            os.makedirs(p, exist_ok=True)
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
        # Try to approach and grasp; then query env success flag (reachCompleted)        
        state0 = self._rpc({"cmd": config.GET_STATE, "args": {"objects": ["handle"]}})
        print('*** test_open_door_script after GET_STATE')                
        handle_pos = state0.get('objects', {}).get('handle', {}).get('pos', None)
        if handle_pos:
            # Approach above the handle slightly to avoid collisions
            approach = [handle_pos[0], handle_pos[1], handle_pos[2] + 0.01]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": approach, "iters": 80, "open_gripper": True}})
            # Descend to contact and close
            contact = [handle_pos[0], handle_pos[1], handle_pos[2]]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": contact, "iters": 60, "open_gripper": True}})
            _ = self._rpc({"cmd": config.CLOSE_GRIPPER, "args": None})
            # Pull along -X to rotate the door
            for k in range(5):
                tgt = [handle_pos[0] - 0.02 * (k + 1), handle_pos[1], handle_pos[2]]
                _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": tgt, "iters": 50, "open_gripper": False}})
                _ = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 4}})
        # Step a bit to allow env to evaluate the success condition
        _ = self._rpc({"cmd": config.STEP_N, "args": {"action": [0,0,0,0], "n": 10}})
                        
        # Export a trajectory video for visual inspection BEFORE assertion
        vid_resp = self._rpc({
            "cmd": config.MAKE_TRAJECTORY_VIDEO,
            "args": {"folder": config.trajectory_folder, "base": config.trajectory_image_base, "fps": config.trajectory_video_fps}
        })
        if not vid_resp.get('ok', False):
            print("MAKE_TRAJECTORY_VIDEO response:", vid_resp)
            self.fail(f"Video creation failed: {vid_resp}")
        if 'video' in vid_resp and isinstance(vid_resp['video'], str):
            self.assertTrue(os.path.exists(vid_resp['video']))

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
        vid_resp = self._rpc({
            "cmd": config.MAKE_TRAJECTORY_VIDEO,
            "args": {"folder": config.trajectory_folder, "base": config.trajectory_image_base, "fps": config.trajectory_video_fps}
        })
        if not vid_resp.get('ok', False):
            print("MAKE_TRAJECTORY_VIDEO response:", vid_resp)
            self.fail(f"Video creation failed: {vid_resp}")
        if 'video' in vid_resp and isinstance(vid_resp['video'], str):
            self.assertTrue(os.path.exists(vid_resp['video']))


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
