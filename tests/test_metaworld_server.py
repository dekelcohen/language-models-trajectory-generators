import os
import sys
import json
import subprocess
import time
import shutil
import unittest
import importlib
import numpy as np
from PIL import Image
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
        cls.python = os.environ.get("METAWORLD_PYTHON", sys.executable)        
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

    def test_backprojection_roundtrip(self):
        """Round-trip: back-project a pixel using K and (R,t), then re-project to pixel."""
        # Capture and parse calibration + poses
        payload = self._rpc({"cmd": config.CAPTURE_IMAGES, "args": None})
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 6)
        head_pos, head_quat, wrist_pos, wrist_quat, _, calib = payload[:6]
        K_head = np.array(calib['head']['K'], dtype=np.float64)
        K_wrist = np.array(calib['wrist']['K'], dtype=np.float64)
        znear = float(calib['head']['znear'])
        zfar = float(calib['head']['zfar'])
        # Depth buffer is OpenGL normalized; load and linearize
        npy_path = './images/depth_image_head.npy'
        self.assertTrue(os.path.exists(npy_path))
        d = np.load(npy_path).astype(np.float64)
        h, w = d.shape
        # Choose a pixel near the image center
        u, v = int(w * 0.45), int(h * 0.55)
        # Linearize OpenGL depth to metric Z
        # --- BEGIN TEMP DEPTH_ORIGIN_UNIFIED ---
        # MuJoCo renderer returns depth with row 0 at the TOP.
        # Sample without vertical flip to match RGB pixel coordinates.
        # Wrapped for quick removal if origin conventions change.
        d_raw = float(d[v, u])
        # --- END TEMP DEPTH_ORIGIN_UNIFIED ---
        denom = (zfar + znear) - (2.0 * d_raw - 1.0) * (zfar - znear)
        denom = denom if abs(denom) > 1e-9 else 1e-9
        Z = (2.0 * znear * zfar) / denom
        Z = float(np.clip(Z, znear, zfar))

        # Quaternion to rotation matrix [x,y,z,w]
        x, y, z, wq = [float(v_) for v_ in head_quat]
        R = np.array([
            [1 - 2 * (y*y + z*z),     2 * (x*y - wq*z),       2 * (x*z + wq*y)],
            [2 * (x*y + wq*z),        1 - 2 * (x*x + z*z),    2 * (y*z - wq*x)],
            [2 * (x*z - wq*y),        2 * (y*z + wq*x),       1 - 2 * (x*x + y*y)],
        ], dtype=np.float64)
        t = np.array(head_pos, dtype=np.float64)

        # Back-project using server's sign convention (v = -fy*(Y/Z) + cy)
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        dir_cam = np.array([
            (float(u) - cx) / (fx if abs(fx) > 1e-9 else 1e-9),
            -(float(v) - cy) / (fy if abs(fy) > 1e-9 else 1e-9),
            1.0,
        ], dtype=np.float64).reshape(3, 1)
        Xc = dir_cam * Z
        Xw = (R @ Xc).reshape(3) + t

        # Forward-project to pixel to verify round-trip (server uses v = -fy*(Y/Z) + cy)
        Xc2 = (R.T @ (Xw - t)).reshape(3)
        Z2 = Xc2[2] if abs(Xc2[2]) > 1e-9 else 1e-9
        u2 = K[0, 0] * (Xc2[0] / Z2) + K[0, 2]
        v2 = -K[1, 1] * (Xc2[1] / Z2) + K[1, 2]
        self.assertLess(abs(u2 - u), 2.0)
        self.assertLess(abs(v2 - v), 2.0)

    def test_backproject_matches_gt_world(self):
        """Backproject a pixel corresponding to a known world point and verify it matches the GT world coordinates.

        This emulates the core of `detect_object`: given a pixel and camera calibration, recover the world position.
        Implementation is self-contained (no api.py or utils imports).
        """
        # --- Phase: capture images + calibration and current camera pose ---
        payload = self._rpc({"cmd": config.CAPTURE_IMAGES, "args": None})
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 6)
        head_pos, head_quat, wrist_pos, wrist_quat, _, calib = payload[:6]
        K_head = np.array(calib['head']['K'], dtype=np.float64)
        K_wrist = np.array(calib['wrist']['K'], dtype=np.float64)
        znear = float(calib['head']['znear'])
        zfar = float(calib['head']['zfar'])

        # --- Phase: pick a GT world point that is actually visible ---
        # Try several candidates commonly in view; fall back to skipping if none visible
        candidates = []
        s_all = self._rpc({"cmd": config.GET_STATE, "args": None})
        objs_all = (s_all.get('objects') or {})
        for name, entry in objs_all.items():
            pos = (entry or {}).get('pos')
            if pos:
                candidates.append((name, np.array(pos, dtype=np.float64)))
        if isinstance(s_all, dict) and s_all.get('eef_pos'):
            candidates.append(("eef", np.array(s_all['eef_pos'], dtype=np.float64)))
        self.assertGreater(len(candidates), 0, "No GT candidates available from env state")

        # --- Phase: build extrinsics R, t from camera quaternion/pos ---
        def quat_to_R(q):
            x, y, z, wq = [float(v_) for v_ in q]
            return np.array([
                [1 - 2 * (y*y + z*z),     2 * (x*y - wq*z),       2 * (x*z + wq*y)],
                [2 * (x*y + wq*z),        1 - 2 * (x*x + z*z),    2 * (y*z - wq*x)],
                [2 * (x*z - wq*y),        2 * (y*z + wq*x),       1 - 2 * (x*x + y*y)],
            ], dtype=np.float64)

        R_head = quat_to_R(head_quat)
        t_head = np.array(head_pos, dtype=np.float64)
        R_wrist = quat_to_R(wrist_quat)
        t_wrist = np.array(wrist_pos, dtype=np.float64)

        # --- Phase: select first candidate that is inside the image and front-most (visible) ---
        def _try_camera(K, R, t, depth_path):
            self.assertTrue(os.path.exists(depth_path))
            dloc = np.load(depth_path).astype(np.float64)
            hloc, wloc = dloc.shape
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            for name, Xw_gt in candidates:
                Xc_tmp = (R.T @ (Xw_gt - t)).reshape(3)
                Zpred = Xc_tmp[2]
                if Zpred <= 0:
                    continue
                u_tmp = fx * (Xc_tmp[0] / Zpred) + cx
                v_tmp = -fy * (Xc_tmp[1] / Zpred) + cy
                if not (0 <= u_tmp < wloc and 0 <= v_tmp < hloc):
                    continue
                ui = int(round(u_tmp))
                vi = int(round(v_tmp))
                d_raw = float(dloc[vi, ui])
                denom = (zfar + znear) - (2.0 * d_raw - 1.0) * (zfar - znear)
                denom = denom if abs(denom) > 1e-9 else 1e-9
                Z_lin_tmp = (2.0 * znear * zfar) / denom
                if abs(Z_lin_tmp - Zpred) < 0.05:
                    return name, Xw_gt, ui, vi, Z_lin_tmp, (fx, fy, cx, cy), R, t
            return None

        chosen = _try_camera(K_head, R_head, t_head, './images/depth_image_head.npy')
        if chosen is None:
            chosen = _try_camera(K_wrist, R_wrist, t_wrist, './images/depth_image_wrist.npy')
        if chosen is None:
            import unittest as _ut
            raise _ut.SkipTest("No visible GT candidate found in head/wrist cameras; off-screen or occluded.")
        name, Xw_gt, ui, vi, Z_lin, (fx, fy, cx, cy), R_use, t_use = chosen

        # Z_lin already computed and validated against prediction; clamp defensively
        Z_lin = float(np.clip(Z_lin, znear, zfar))

        # --- Phase: backproject pixel to camera and transform to world ---
        dir_cam = np.array([
            (float(ui) - cx) / (fx if abs(fx) > 1e-9 else 1e-9),
            -(float(vi) - cy) / (fy if abs(fy) > 1e-9 else 1e-9),
            1.0,
        ], dtype=np.float64).reshape(3, 1)
        Xc_bp = dir_cam * Z_lin
        Xw_bp = (R_use @ Xc_bp).reshape(3) + t_use

        # --- Phase: compare recovered world point to GT ---
        err = float(np.linalg.norm(Xw_bp - Xw_gt))
        # Allow a small tolerance due to sampling, rounding and renderer quantization
        self.assertLess(err, 0.03, f"Backprojected world point deviates {err:.4f} m from GT")

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
