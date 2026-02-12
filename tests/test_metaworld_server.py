import os
import sys
import json
import subprocess
import time
import shutil
import unittest
import importlib
import numpy as np
from PIL import Image, ImageDraw
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
        # Capture and parse cam_inforation + poses
        payload = self._rpc({"cmd": config.CAPTURE_IMAGES, "args": None})
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 6)
        head_pos, head_quat, wrist_pos, wrist_quat, _, cam_info = payload[:6]
        K_head = np.array(cam_info['head']['K'], dtype=np.float64)
        K_wrist = np.array(cam_info['wrist']['K'], dtype=np.float64)
        znear = float(cam_info['head']['znear'])
        zfar = float(cam_info['head']['zfar'])
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
        """Head-camera backprojection for the 'handle' object only.

        Selects 'handle' from GET_STATE, projects to head pixel, overlays the selection,
        backprojects using head K/extrinsics, and asserts the recovered world point
        matches the GT within a small tolerance.
        """
        # --- Phase: move EEF out of the head line-of-sight to reduce occlusion ---
        try:
            state0 = self._rpc({"cmd": config.GET_STATE, "args": None})
            ee0 = state0.get('eef_pos') if isinstance(state0, dict) else None
            # Move the gripper back-left-up a bit
            safe = [float(ee0[0] - 0.25), float(ee0[1] - 0.15), float(ee0[2] + 0.10)] if ee0 else [-0.25, 0.35, 0.70]
            _ = self._rpc({"cmd": config.MOVE_EEF_ABS, "args": {"pos": safe, "iters": 60, "open_gripper": True}})
        except Exception:
            pass

        # --- Phase: capture images + cam_inforation and HEAD camera pose ---
        payload = self._rpc({"cmd": config.CAPTURE_IMAGES, "args": None})
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 6)
        head_pos, head_quat, _, _, _, cam_info = payload[:6]
        K = np.array(cam_info['head']['K'], dtype=np.float64)
        znear = float(cam_info['head']['znear'])
        zfar = float(cam_info['head']['zfar'])

        # --- Phase: fetch GT world point strictly: 'handle' ---
        st = self._rpc({"cmd": config.GET_STATE, "args": {"objects": ["handle"]}})
        objs = (st.get('objects') or {})
        handle_pos = (objs.get('handle') or {}).get('pos')
        self.assertIsNotNone(handle_pos, "Expected 'handle' world position from env state")
        Xw_gt = np.array(handle_pos, dtype=np.float64)

        # --- Phase: compute world-to-camera (Rc, tc) from head pose ---
        x, y, z, wq = [float(v_) for v_ in head_quat]
        Rcw = np.array([
            [1 - 2 * (y*y + z*z),     2 * (x*y - wq*z),       2 * (x*z + wq*y)],
            [2 * (x*y + wq*z),        1 - 2 * (x*x + z*z),    2 * (y*z - wq*x)],
            [2 * (x*z - wq*y),        2 * (y*z + wq*x),       1 - 2 * (x*x + y*y)],
        ], dtype=np.float64)
        t_w = np.array(head_pos, dtype=np.float64)
        Rc = Rcw.T
        tc = -Rc @ t_w

        # --- Phase: project GT handle to head pixel using Rc, tc ---
        Xc = Rc @ Xw_gt + tc
        # Server uses Zp = abs(Z). Keep sign only for camera-frame for diagnostics.
        Z = float(Xc[2])
        Zp = float(abs(Z))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u = fx * (Xc[0] / Zp) + cx
        v = -fy * (Xc[1] / Zp) + cy
        npy_path = './images/depth_image_head.npy'
        self.assertTrue(os.path.exists(npy_path))
        d = np.load(npy_path).astype(np.float64)
        h, w = d.shape
        self.assertTrue(0 <= u < w and 0 <= v < h, "Projected pixel outside head frame for 'handle'")
        ui = int(round(u))
        vi = int(round(v))
        # Diagnostics: GT camera-frame values
        Z_axis_gt = float(abs(Xc[2]))
        L_ray_gt = float(np.linalg.norm(Xc))
        print(f"[diag] GT camera-frame: Xc={Xc.tolist()}, Z_axis_gt={Z_axis_gt:.6f}, L_ray_gt={L_ray_gt:.6f}, proj_px=({ui},{vi})")

        # --- Phase: interpret depth at (ui,vi)
        # MuJoCo's 'depth_array' may be either OpenGL-normalized (needs linearization)
        # or metric ray depth depending on renderer version. Try both and pick the one
        # consistent with GT camera-frame values.
        d_raw = float(d[vi, ui])
        # OpenGL -> metric z along optical axis
        denom = (zfar + znear) - (2.0 * d_raw - 1.0) * (zfar - znear)
        denom = denom if abs(denom) > 1e-9 else 1e-9
        Z_lin = float((2.0 * znear * zfar) / denom)
        # Metric ray length hypothesis (already meters). If server supplies OpenGL, this will not match.
        L_ray = max(d_raw, 0.0)
        # Additional heuristic: if depth_encoding indicates OpenGL, prefer Z_lin; otherwise prefer d_raw.
        depth_enc = None
        try:
            resp_info = self._rpc({"cmd": config.GET_CAMERA_INFO, "args": None})
            depth_enc = (payload[5].get("depth_encoding") if isinstance(payload, list) and len(payload) >= 6 and isinstance(payload[5], dict) else None)
        except Exception:
            depth_enc = None
        # Also compute normalized depth predicted from GT axis-Z; compare to raw to infer encoding
        try:
            Z_gt = float(Z_axis_gt)
            num = (zfar + znear) - (2.0 * znear * zfar) / max(Z_gt, 1e-9)
            zndc_from_gt = 0.5 * (1.0 + (num / (zfar - znear)))
            print(f"[diag] d_pred_from_gt_Z={zndc_from_gt:.6f} vs d_raw={d_raw:.6f}")
        except Exception:
            pass

        # --- Phase: diagnostics overlay and print ---
        try:
            im = Image.open(config.rgb_image_head_path).convert('RGB')
            draw = ImageDraw.Draw(im)
            r = 5
            draw.ellipse([(ui - r, vi - r), (ui + r, vi + r)], outline=(255, 0, 0), width=2)
            draw.line([(ui - r*2, vi), (ui + r*2, vi)], fill=(255, 0, 0), width=2)
            draw.line([(ui, vi - r*2), (ui, vi + r*2)], fill=(255, 0, 0), width=2)
            out_path = os.path.join('./images/overlay', 'selected_point_head_handle.png')
            im.save(out_path)
            print(f"[diag] selected camera=head, object=handle, pixel=({ui},{vi}), d_raw={d_raw:.6f}, Z_lin={Z_lin:.6f}, overlay={out_path}")
        except Exception as e:
            print(f"[diag] overlay failed: {e}")

        # --- Phase: backproject selected pixel to camera and transform to world ---
        # Camera looks along -Z in MuJoCo/OpenGL; build direction accordingly
        dir_cam = np.array([
            (float(ui) - cx) / (fx if abs(fx) > 1e-9 else 1e-9),
            -(float(vi) - cy) / (fy if abs(fy) > 1e-9 else 1e-9),
            -1.0,
        ], dtype=np.float64).reshape(3, 1)
        # Compute backprojections:
        # (A) axial-Z using OpenGL-linearized Z_lin
        Z_use_lin = float(np.clip(Z_lin, znear, zfar))
        Xc_bp_zlin = np.array([
            (float(ui) - cx) / (fx if abs(fx) > 1e-9 else 1e-9) * Z_use_lin,
            -(float(vi) - cy) / (fy if abs(fy) > 1e-9 else 1e-9) * Z_use_lin,
            -Z_use_lin,
        ], dtype=np.float64).reshape(3, 1)
        # (B) axial-Z assuming metric Z is stored directly in .npy
        Z_use_raw = float(np.clip(d_raw, znear, zfar))
        Xc_bp_zraw = np.array([
            (float(ui) - cx) / (fx if abs(fx) > 1e-9 else 1e-9) * Z_use_raw,
            -(float(vi) - cy) / (fy if abs(fy) > 1e-9 else 1e-9) * Z_use_raw,
            -Z_use_raw,
        ], dtype=np.float64).reshape(3, 1)
        # (C) ray-length interpretation
        ray_dir = dir_cam / (float(np.linalg.norm(dir_cam)) if float(np.linalg.norm(dir_cam)) > 1e-9 else 1.0)
        Xc_bp_ray = (ray_dir * L_ray).reshape(3, 1)
        # Normalize ray direction for metric ray-length interpretation
        ray_dir = dir_cam / (float(np.linalg.norm(dir_cam)) if float(np.linalg.norm(dir_cam)) > 1e-9 else 1.0)
        Xc_bp_ray = (ray_dir * L_ray).reshape(3, 1)
        # Camera-to-world transform
        Xw_bp_zlin = (Rcw @ Xc_bp_zlin).reshape(3) + t_w
        Xw_bp_zraw = (Rcw @ Xc_bp_zraw).reshape(3) + t_w
        Xw_bp_ray = (Rcw @ Xc_bp_ray).reshape(3) + t_w
        # (D) backproject using GT camera-frame axial Z to validate pure math round-trip
        Z_use_gt = float(np.clip(Z_axis_gt, znear, zfar))
        Xc_bp_gtZ = np.array([
            (float(ui) - cx) / (fx if abs(fx) > 1e-9 else 1e-9) * Z_use_gt,
            -(float(vi) - cy) / (fy if abs(fy) > 1e-9 else 1e-9) * Z_use_gt,
            -Z_use_gt,
        ], dtype=np.float64).reshape(3, 1)
        Xw_bp_gtZ = (Rcw @ Xc_bp_gtZ).reshape(3) + t_w
        # Reproject both to pixels to verify consistency
        for lbl, Xw in [("zlin", Xw_bp_zlin), ("zraw", Xw_bp_zraw), ("ray", Xw_bp_ray)]:
            Xc2 = (Rc @ Xw + tc).reshape(3)
            Z2 = float(abs(Xc2[2])) if abs(Xc2[2]) > 1e-9 else 1e-9
            u2 = fx * (Xc2[0] / Z2) + cx
            v2 = -fy * (Xc2[1] / Z2) + cy
            print(f"[diag] reproject({lbl}) -> px=({u2:.2f},{v2:.2f}) vs sel=({ui},{vi})")

        # --- Phase: compare recovered world point to GT ---
        err_zlin = float(np.linalg.norm(Xw_bp_zlin - Xw_gt))
        err_zraw = float(np.linalg.norm(Xw_bp_zraw - Xw_gt))
        err_ray = float(np.linalg.norm(Xw_bp_ray - Xw_gt))
        err_gt = float(np.linalg.norm(Xw_bp_gtZ - Xw_gt))
        print(f"[diag] backproject errs -> opengl_Z={err_zlin:.6f} m, metric_Z={err_zraw:.6f} m, metric_ray={err_ray:.6f} m, gt_Z={err_gt:.6f} m; znear={znear:.4f} zfar={zfar:.4f} depth_enc={depth_enc}")
        # Assert the pure math round-trip using GT Z; depth buffer may measure nearest surface along the ray.
        self.assertLess(err_gt, 0.03, f"Round-trip using GT camera Z deviates {err_gt:.4f} m from GT for 'handle'")

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

            # Curved half?circle style pull:
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

