import os
import sys
import json
import subprocess
import time
import shutil
import unittest
import importlib

config = importlib.import_module('config')


class TestMetaworldServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure images dirs exist and are clean
        for p in [
            './images',
            './images/trajectory',
            './images/overlay',
        ]:
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
        cls.proc = subprocess.Popen(
            [cls.python, 'providers/metaworld_server.py', '--env', 'sawyer_door_v3'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,   # binary pipes for Windows robustness
            bufsize=0,
            env=env,
        )
        # Wait for a JSON ready banner
        start_deadline = time.time() + 15.0
        buffer = []
        while time.time() < start_deadline:
            line = cls.proc.stdout.readline()
            if not line:
                if cls.proc.poll() is not None:
                    try:
                        rem = cls.proc.stdout.read()
                        if rem:
                            buffer.append(rem.decode('utf-8', errors='ignore'))
                    except Exception:
                        pass
                    raise RuntimeError('server exited during startup. Output:\n' + ''.join(buffer))
                time.sleep(0.02)
                continue
            try:
                s = line.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if s:
                buffer.append(s + '\n')
            if not s:
                continue
            if s[0] not in '{[':
                continue
            try:
                msg = json.loads(s)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get('status') == 'ready':
                # Echo the startup banner so camera names/IDs are visible in test output
                print(s)
                break
        else:
            raise RuntimeError('server did not start with a JSON banner. Output so far:\n' + ''.join(buffer))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.proc.kill()
        except Exception:
            pass

    def _rpc(self, obj):
        # Send request (ensure process is alive)
        if self.__class__.proc.poll() is not None:
            # Drain any remaining output to aid debugging
            try:
                leftover = self.__class__.proc.stdout.read()
            except Exception:
                leftover = ''
            raise RuntimeError('server not running; output: ' + (leftover or ''))
        payload = (json.dumps(obj) + "\n").encode('utf-8')
        self.__class__.proc.stdin.write(payload)
        self.__class__.proc.stdin.flush()
        # Read until a valid JSON line
        deadline = time.time() + 10.0
        while time.time() < deadline:
            line = self.__class__.proc.stdout.readline()
            if not line:
                if self.__class__.proc.poll() is not None:
                    raise RuntimeError('server exited unexpectedly')
                time.sleep(0.01)
                continue
            try:
                s = line.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if not s:
                continue
            if s[0] not in '{[':
                continue
            try:
                return json.loads(s)
            except Exception:
                continue
        raise TimeoutError('timed out waiting for JSON response')

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

    def test_open_door_success_flag(self):
        # Try to approach and grasp; then query env success flag (reachCompleted)
        state0 = self._rpc({"cmd": config.GET_STATE, "args": {"objects": ["handle"]}})
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
            "args": {"folder": config.trajectory_folder, "base": config.trajectory_image_base, "start": 0, "end": 9999, "fps": config.trajectory_video_fps}
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
            "args": {"folder": config.trajectory_folder, "base": config.trajectory_image_base, "start": 0, "end": 9999, "fps": config.trajectory_video_fps}
        })
        if not vid_resp.get('ok', False):
            print("MAKE_TRAJECTORY_VIDEO response:", vid_resp)
            self.fail(f"Video creation failed: {vid_resp}")
        if 'video' in vid_resp and isinstance(vid_resp['video'], str):
            self.assertTrue(os.path.exists(vid_resp['video']))


if __name__ == '__main__':
    unittest.main()
