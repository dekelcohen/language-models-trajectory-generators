import os
import sys
# Ensure repository root (parent of this file's directory) is on sys.path
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
import time
import numpy as np
from PIL import Image
import argparse
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="sawyer_door_v3", help="Metaworld task/env name")
    parser.add_argument("--viewer", action="store_true", help="Open Mujoco viewer for local testing")
    parser.add_argument(
        "--cameras",
        type=str,
        default="both",
        help="Select cameras to capture: 'head', 'wrist', or 'both' (default)."
    )
    args = parser.parse_args()
    # Load metaworld from an external env path if provided
    # Expect env var METAWORLD_REPO or fallback to current sys.path
    meta_root = os.environ.get("METAWORLD_REPO")
    if meta_root and meta_root not in sys.path:
        sys.path.append(meta_root)

    # Select environment by name (extendable later)
    if args.env.lower() == "sawyer_door_v3":
        from metaworld.envs.sawyer_door_v3 import SawyerDoorEnvV3 as EnvCls
    else:
        # Fallback to door task for now
        from metaworld.envs.sawyer_door_v3 import SawyerDoorEnvV3 as EnvCls

    # Basic image paths mirrored from config defaults to reduce coupling
    rgb_head_path = config.rgb_image_head_path
    depth_head_path = config.depth_image_head_path
    rgb_wrist_path = config.rgb_image_wrist_path
    depth_wrist_path = config.depth_image_wrist_path
    rgb_traj_path_tpl = config.rgb_image_trajectory_path
    depth_traj_path_tpl = config.depth_image_trajectory_path
    overlay_traj_path_tpl = config.overlay_image_path

    # Ensure output directories exist
    os.makedirs(os.path.dirname(rgb_head_path), exist_ok=True)
    os.makedirs(os.path.dirname(rgb_wrist_path), exist_ok=True)
    os.makedirs(os.path.dirname(rgb_traj_path_tpl.format(step=0)), exist_ok=True)
    os.makedirs(os.path.dirname(overlay_traj_path_tpl.format(step=0)), exist_ok=True)

    env = EnvCls(render_mode=None, width=256, height=256)
    # Metaworld v3 environments expect a task to be set before use.
    # Since we are running a single env class directly (not via MetaWorld wrappers),
    # allow reset() to sample a fresh random vector and mark the task as set.
    # This mirrors how MetaWorld generates tasks internally.
    try:
        # Unfreeze randomization so _get_state_rand_vec() samples on reset
        setattr(env, "_freeze_rand_vec", False)
        # Satisfy decorators that require set_task() to have been called
        setattr(env, "_set_task_called", True)
    except Exception:
        pass
    obs, info = env.reset()

    # Choose cameras by name; robust fallback if names are unavailable
    try:
        import mujoco
        ncam = int(getattr(env.model, "ncam", 0))
        cam_names = []
        for i in range(ncam):
            name = None
            try:
                # MuJoCo >= 2.3 API
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            except Exception:
                try:
                    # Legacy mujoco_py-style convenience
                    name = env.model.id2name(i, mujoco.mjtObj.mjOBJ_CAMERA)
                except Exception:
                    name = None
            if not name:
                name = f"camera_{i}"
            cam_names.append(name)
    except Exception:
        try:
            # Fallback without mujoco: synthesize names by count if available
            ncam = int(getattr(getattr(env, "model", None), "ncam", 0))
            cam_names = [f"camera_{i}" for i in range(ncam)]
        except Exception:
            cam_names = []
    head_id = 0
    wrist_id = 0
    # Prefer intuitive viewpoints: head → topview or first corner; wrist → gripperPOV/behindGripper
    lower_names = [n.lower() if isinstance(n, str) else "" for n in cam_names]
    # Head selection
    try:
        head_candidates = [i for i, n in enumerate(lower_names) if "topview" in n]
        if head_candidates:
            head_id = head_candidates[0]
        else:
            corner_candidates = [i for i, n in enumerate(lower_names) if "corner" in n]
            if corner_candidates:
                head_id = corner_candidates[0]
    except Exception:
        pass
    # Wrist selection
    try:
        wrist_candidates = [i for i, n in enumerate(lower_names) if "gripperpov" in n]
        if wrist_candidates:
            wrist_id = wrist_candidates[0]
        else:
            wrist_candidates = [i for i, n in enumerate(lower_names) if "behindgripper" in n]
            if wrist_candidates:
                wrist_id = wrist_candidates[0]
            else:
                generic_gripper = [i for i, n in enumerate(lower_names) if "gripper" in n]
                if generic_gripper:
                    wrist_id = generic_gripper[0]
    except Exception:
        pass

    gripper_open = True
    traj_step = 1
    frame_counter = 0
    perception_logged = 0

    def _resolve_selected_cameras():
        sel = (getattr(args, "cameras", "both") or "both").strip().lower()
        requested = []
        if sel in ("both", "all", "head+wrist"):
            requested = [("head", head_id), ("wrist", wrist_id)]
        elif sel == "head":
            requested = [("head", head_id)]
        elif sel == "wrist":
            requested = [("wrist", wrist_id)]
        else:
            parts = [p.strip() for p in sel.split(",") if p.strip()]
            for p in parts:
                if p in ("head", "top", "corner", "topview"):
                    requested.append(("head", head_id))
                elif p in ("wrist", "gripper"):
                    requested.append(("wrist", wrist_id))
        # De-duplicate by camera ID while preserving order
        seen = set()
        selected = []
        for label, cid in requested:
            if cid not in seen:
                selected.append((label, cid))
                seen.add(cid)
        return selected

    # Print a single JSON line as a ready banner with camera info
    cameras_info = {
        "names": cam_names,
        "selected": {"head": int(head_id), "wrist": int(wrist_id)},
        "selected_names": {
            "head": cam_names[int(head_id)] if int(head_id) < len(cam_names) else None,
            "wrist": cam_names[int(wrist_id)] if int(wrist_id) < len(cam_names) else None,
        },
    }
    print(json.dumps({"status": "ready", "env": args.env, "cameras": cameras_info}))
    sys.stdout.flush()

    # Optional local viewer for sanity checking
    if args.viewer:
        try:
            import mujoco
            from mujoco import viewer
            with viewer.launch_passive(env.model, env.data) as v:
                v.cam.azimuth = 120
                v.cam.elevation = -20
                v.cam.distance = 2.0
                # Minimal non-JSON notice; tests skip non-JSON lines
                print("[viewer] Close the window (or press ESC) to exit.")
                while v.is_running():
                    action = np.zeros(4, dtype=np.float32)
                    try:
                        _, _, terminated, truncated, _ = env.step(action)
                    except Exception as e:
                        # If gymnasium complains about stepping after truncate, force a reset
                        if "truncate==True" in str(e):
                            env.reset()
                            continue
                        raise
                    if terminated or truncated:
                        # Keep randomization behavior consistent across resets
                        try:
                            setattr(env, "_freeze_rand_vec", False)
                            setattr(env, "_set_task_called", True)
                        except Exception:
                            pass
                        env.reset()
                    v.sync()
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Viewer launch failed: {e}")

    def set_active_camera(cam_id):
        # Switch the env and renderer to a specific camera if supported.
        try:
            env.camera_id = int(cam_id)
        except Exception:
            pass
        try:
            rend = getattr(env, "mujoco_renderer", None)
            if rend is not None and hasattr(rend, "camera_id"):
                rend.camera_id = int(cam_id)
        except Exception:
            pass

    def render_and_save(cam_id, rgb_path, depth_path):
        # Ensure the renderer uses the same camera whose intrinsics we use
        set_active_camera(cam_id)
        rgb = env.mujoco_renderer.render(render_mode="rgb_array")
        depth_raw = env.mujoco_renderer.render(render_mode="depth_array")
        Image.fromarray(rgb).save(rgb_path)
        # Try to linearize OpenGL depth into meters for better contrast
        try:
            K, zn, zf, _, _ = get_intrinsics(cam_id)
            z_ndc = depth_raw * 2.0 - 1.0
            z_eye = (2.0 * zn * zf) / (zf + zn - z_ndc * (zf - zn))
            # Map a close range to 0..255 for visualization
            vis_far = min(zf, zn + 2.0)
            d_norm = np.clip((z_eye - zn) / (vis_far - zn + 1e-6), 0.0, 1.0)
            d_vis = (1.0 - d_norm)  # near is bright
            Image.fromarray((d_vis * 255).astype(np.uint8)).save(depth_path)
        except Exception:
            d = np.clip(depth_raw, 0.0, 1.0)
            Image.fromarray((d * 255).astype(np.uint8)).save(depth_path)

    def _rotmat_to_quat_xyzw(R):
        """Convert 3x3 rotation matrix to quaternion (x, y, z, w) with NumPy only.
        Returns a length-4 np.ndarray.
        """
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        # Normalize to be safe
        n = np.linalg.norm(q)
        if n > 0:
            q /= n
        return q

    def get_cam_pose(cam_id):
        pos = env.data.cam_xpos[cam_id].copy()
        R = env.data.cam_xmat[cam_id].reshape(3, 3)
        quat = _rotmat_to_quat_xyzw(R)
        return pos.tolist(), quat.tolist()

    def get_cam_RT(cam_id):
        # world-to-camera extrinsics: R^T and t
        R = env.data.cam_xmat[cam_id].reshape(3, 3)
        t = env.data.cam_xpos[cam_id].copy()
        Rc = R.T
        tc = -Rc @ t
        return Rc, tc

    def get_intrinsics(cam_id, width=None, height=None):
        import mujoco
        fovy = float(env.model.cam_fovy[cam_id])
        # Prefer actual render size when provided; fall back to env defaults
        if width is None or height is None:
            width = int(getattr(env, "width", 0) or 0)
            height = int(getattr(env, "height", 0) or 0)
            # Some renderers don’t expose width/height; derive from a single frame
            if width <= 0 or height <= 0:
                set_active_camera(cam_id)
                arr = env.mujoco_renderer.render(render_mode="rgb_array")
                height, width = int(arr.shape[0]), int(arr.shape[1])
        znear = float(env.model.vis.map.znear)
        zfar = float(env.model.vis.map.zfar)
        fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy * (width / float(height))
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
        return K, znear, zfar, width, height

    def project_world_to_pixel(cam_id, pts_world, width=None, height=None):
        Rc, tc = get_cam_RT(cam_id)
        K, *_ = get_intrinsics(cam_id, width=width, height=height)
        pts = []
        for P in pts_world:
            # Camera coordinates (MuJoCo cameras look along -Z)
            Xc = Rc @ P + tc
            Z = -Xc[2] if Xc[2] < 0 else Xc[2]
            if abs(Z) < 1e-9:
                Z = 1e-9
            # Image axes: +u right, +v down. Camera Y is up → flip sign for v.
            u = K[0, 0] * (Xc[0] / Z) + K[0, 2]
            v = -K[1, 1] * (Xc[1] / Z) + K[1, 2]
            pts.append([float(u), float(v), float(Z)])
        return pts

    def draw_overlay(cam_id, points_world, path):
        from PIL import ImageDraw
        set_active_camera(cam_id)
        rgb = env.mujoco_renderer.render(render_mode="rgb_array")
        h, w = rgb.shape[0], rgb.shape[1]
        pts_px = project_world_to_pixel(cam_id, points_world, width=w, height=h)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        for (u, v, z) in pts_px:
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(255, 0, 0), width=2)
        im.save(path)

    def get_objects_pose(names):
        out = {}
        for name in names:
            pos = None
            quat = None
            try:
                pos = env.data.geom(name).xpos.copy()
                R = env.data.geom(name).xmat.reshape(3, 3)
                quat = _rotmat_to_quat_xyzw(R)
            except Exception:
                try:
                    pos = env.data.site(name).xpos.copy()
                    R = env.data.site(name).xmat.reshape(3, 3)
                    quat = _rotmat_to_quat_xyzw(R)
                except Exception:
                    try:
                        pos = env.data.body(name).xpos.copy()
                        R = env.data.body(name).xmat.reshape(3, 3)
                        quat = _rotmat_to_quat_xyzw(R)
                    except Exception:
                        pass
            if pos is not None and quat is not None:
                out[name] = {"pos": pos.tolist(), "quat": quat.tolist()}
        return out

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        cmd = req.get("cmd")
        args = req.get("args")
        try:
            if cmd == config.CAPTURE_IMAGES:  # CAPTURE_IMAGES
                for label, cid in _resolve_selected_cameras():
                    if label == "head":
                        render_and_save(cid, rgb_head_path, depth_head_path)
                    elif label == "wrist":
                        render_and_save(cid, rgb_wrist_path, depth_wrist_path)
                head_pos, head_quat = get_cam_pose(head_id)
                wrist_pos, wrist_quat = get_cam_pose(wrist_id)
                # Build calibration dict with exact intrinsics and near/far
                try:
                    import mujoco
                    fovy_h = float(env.model.cam_fovy[head_id])
                    fovy_w = float(env.model.cam_fovy[wrist_id])
                    width = int(env.width)
                    height = int(env.height)
                    znear = float(env.model.vis.map.znear)
                    zfar = float(env.model.vis.map.zfar)
                    fy_h = height / (2.0 * np.tan(np.deg2rad(fovy_h) / 2.0))
                    fx_h = fy_h * (width / float(height))
                    K_head = [[fx_h, 0.0, width / 2.0], [0.0, fy_h, height / 2.0], [0.0, 0.0, 1.0]]
                    fy_w = height / (2.0 * np.tan(np.deg2rad(fovy_w) / 2.0))
                    fx_w = fy_w * (width / float(height))
                    K_wrist = [[fx_w, 0.0, width / 2.0], [0.0, fy_w, height / 2.0], [0.0, 0.0, 1.0]]
                    calib = {
                        "head": {"K": K_head, "znear": znear, "zfar": zfar, "width": width, "height": height, "fovy": fovy_h},
                        "wrist": {"K": K_wrist, "znear": znear, "zfar": zfar, "width": width, "height": height, "fovy": fovy_w},
                        "depth_encoding": "opengl",
                    }
                except Exception:
                    calib = None
                print(json.dumps([head_pos, head_quat, wrist_pos, wrist_quat, "\u001b[92mFinished capturing head camera image!\u001b[0m", calib]))
                sys.stdout.flush()
            elif cmd == config.ADD_BOUNDING_CUBES:  # ADD_BOUNDING_CUBES (no-op)
                print(json.dumps(["\u001b[92mFinished adding bounding cubes to the environment!\u001b[0m"]))
                sys.stdout.flush()

            elif cmd == config.ADD_TRAJECTORY_POINTS:  # ADD_TRAJECTORY_POINTS (no-op)
                # Accept and ignore
                pass

            elif cmd == config.EXECUTE_TRAJECTORY:  # EXECUTE_TRAJECTORY
                traj = args or []
                for pt in traj:
                    target = np.array(pt[:3], dtype=np.float64)
                    for _ in range(30):
                        ee = env.get_endeff_pos().copy()
                        delta = target - ee
                        action = np.zeros(4, dtype=np.float32)
                        if np.linalg.norm(delta) < 1e-3:
                            break
                        action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                        action[3] = 1.0 if gripper_open else -1.0
                        env.step(action)
                    # Save a trajectory frame similar to pybullet behavior
                    # Throttled logging to avoid sim slowdown
                    if traj_step % int(getattr(config, "trajectory_log_every", 5)) == 0:
                        rgb_p = rgb_traj_path_tpl.format(step=traj_step)
                        d_p = depth_traj_path_tpl.format(step=traj_step)
                        render_and_save(head_id, rgb_p, d_p)
                    traj_step += 1

            elif cmd == config.OPEN_GRIPPER:  # OPEN_GRIPPER
                gripper_open = True

            elif cmd == config.CLOSE_GRIPPER:  # CLOSE_GRIPPER
                gripper_open = False

            elif cmd == config.TASK_COMPLETED:  # TASK_COMPLETED
                print(json.dumps(["\u001b[92mFinished executing all generated trajectories!\u001b[0m"]))
                sys.stdout.flush()

            elif cmd == config.RESET_ENVIRONMENT:  # RESET_ENVIRONMENT
                env.reset()
                gripper_open = True
                traj_step = 1
                print(json.dumps(["\u001b[92mFinished resetting environment!\u001b[0m"]))
                sys.stdout.flush()

            elif cmd == config.GET_STATE:  # GET_STATE
                eef = env.get_endeff_pos().copy()
                names = []
                if args and isinstance(args, dict):
                    names = args.get("objects", []) or []
                objs = get_objects_pose(names)
                # Door joint angle if exists
                try:
                    qpos = float(env.data.joint("doorjoint").qpos.item())
                except Exception:
                    qpos = None
                print(json.dumps({
                    "eef_pos": eef.tolist(),
                    "objects": objs,
                    "doorjoint_angle": qpos,
                }))
                sys.stdout.flush()

            elif cmd == config.GET_CAMERA_INFO:  # GET_CAMERA_INFO
                try:
                    import mujoco
                    K_h, zn_h, zf_h, w, h = get_intrinsics(head_id)
                    K_w, zn_w, zf_w, _, _ = get_intrinsics(wrist_id)
                    head_pos, head_quat = get_cam_pose(head_id)
                    wrist_pos, wrist_quat = get_cam_pose(wrist_id)
                    print(json.dumps({
                        "head": {"K": K_h.tolist(), "znear": zn_h, "zfar": zf_h, "width": w, "height": h, "pos": head_pos, "quat": head_quat},
                        "wrist": {"K": K_w.tolist(), "znear": zn_w, "zfar": zf_w, "width": w, "height": h, "pos": wrist_pos, "quat": wrist_quat}
                    }))
                except Exception:
                    print(json.dumps(None))
                sys.stdout.flush()

            elif cmd == config.CAPTURE_ANNOTATED_IMAGES:  # CAPTURE_ANNOTATED_IMAGES
                # Log perception data sparsely
                max_first = int(getattr(config, "perception_log_first_n", 1))
                interval = int(getattr(config, "perception_log_interval_frames", 0))
                should_log = perception_logged < max_first or (interval > 0 and frame_counter % interval == 0)
                if should_log:
                    # Use TCP center for better gripper visualization
                    try:
                        eef = env.tcp_center.copy()
                    except Exception:
                        eef = env.get_endeff_pos().copy()
                    names = []
                    if args and isinstance(args, dict):
                        names = args.get("objects", []) or []
                    objs = get_objects_pose(names)
                    pts_world = [eef]
                    for obj in objs.values():
                        pts_world.append(np.array(obj["pos"], dtype=np.float64))
                    overlay_p = overlay_traj_path_tpl.format(step=traj_step)
                    # Use the first selected camera for overlays (defaults to head)
                    selected = _resolve_selected_cameras()
                    cam_for_overlay = selected[0][1] if selected else head_id
                    draw_overlay(cam_for_overlay, pts_world, overlay_p)
                    perception_logged += 1
                frame_counter += 1
                print(json.dumps({"logged": should_log, "frame": frame_counter}))
                sys.stdout.flush()

            elif cmd == config.MOVE_EEF_ABS:  # MOVE_EEF_ABS
                target = np.array(args.get("pos", [0, 0, 0]), dtype=np.float64)
                iters = int(args.get("iters", 30))
                nonlocal_gripper_open = bool(args.get("open_gripper", True))
                max_steps = max(200, iters * 5)
                tol = 5e-3
                gain = 2.5
                for _ in range(max_steps):
                    ee = env.get_endeff_pos().copy()
                    delta = target - ee
                    if np.linalg.norm(delta) < tol:
                        break
                    action = np.zeros(4, dtype=np.float32)
                    action[:3] = np.clip(gain * (delta / getattr(env, 'action_scale', 1.0)), -1.0, 1.0)
                    action[3] = 1.0 if nonlocal_gripper_open else -1.0
                    try:
                        _, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            # keep going from reset state but same target
                            env.reset()
                    except Exception:
                        env.reset()
                ee_final = env.get_endeff_pos().copy()
                print(json.dumps({"eef_pos": ee_final.tolist(), "pos_err": float(np.linalg.norm(target - ee_final))}))
                sys.stdout.flush()

            elif cmd == config.STEP_N:  # STEP_N
                action = np.array(args.get("action", [0, 0, 0, 0]), dtype=np.float32)
                n = int(args.get("n", 1))
                terminated = False
                truncated = False
                for _ in range(n):
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                print(json.dumps({"terminated": terminated, "truncated": truncated}))
                sys.stdout.flush()

            elif cmd == config.SET_SEED:  # SET_SEED
                seed = int(args.get("seed", 42))
                env.seed(seed)
                print(json.dumps({"seed": seed}))
                sys.stdout.flush()

            elif cmd == config.SET_TASK_FROM_RAND_VEC:  # SET_TASK_FROM_RAND_VEC
                rv = np.array(args.get("rand_vec", [0.1, 0.95, 0.15]), dtype=np.float64)
                try:
                    setattr(env, "_freeze_rand_vec", True)
                    setattr(env, "_last_rand_vec", rv)
                    setattr(env, "_set_task_called", True)
                except Exception:
                    pass
                env.reset()
                print(json.dumps({"rand_vec": rv.tolist()}))
                sys.stdout.flush()
            else:
                # Unknown
                print(json.dumps({"error": "unknown_cmd", "cmd": cmd}))
                sys.stdout.flush()
        except Exception as e:
            # Do not crash the server; report error
            try:
                print(json.dumps({"error": str(e)}))
                sys.stdout.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
