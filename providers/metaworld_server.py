import os
import sys
# Ensure repository root (parent of this file's directory) is on sys.path
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
import time
import asyncio
import numpy as np
from PIL import Image
import argparse
import config
from providers.env_sim_util import _get_objects_pose
from debug.dbg_utils import install_debug_reminder

# GUI Viewer 
def launch_viewer(env):
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
        
# --- Top-level helpers (extracted from main) ---
def _rotmat_to_quat_xyzw(R):
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
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q

def _set_active_camera(env, cam_id):
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

def _get_cam_pose(env, cam_id):
    try:
        pos = env.data.cam_xpos[cam_id].copy()
        R = env.data.cam_xmat[cam_id].reshape(3, 3)
        quat = _rotmat_to_quat_xyzw(R)
        return pos.tolist(), quat.tolist()
    except Exception:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

def _get_cam_RT(env, cam_id):
    R = env.data.cam_xmat[cam_id].reshape(3, 3)
    t = env.data.cam_xpos[cam_id].copy()
    Rc = R.T
    tc = -Rc @ t
    return Rc, tc

def _get_intrinsics(env, cam_id, width=None, height=None):
    import mujoco
    fovy = float(env.model.cam_fovy[cam_id])
    if width is None or height is None:
        width = int(getattr(env, "width", 0) or 0)
        height = int(getattr(env, "height", 0) or 0)
        if width <= 0 or height <= 0:
            _set_active_camera(env, cam_id)
            arr = env.mujoco_renderer.render(render_mode="rgb_array")
            height, width = int(arr.shape[0]), int(arr.shape[1])
    znear = float(env.model.vis.map.znear)
    zfar = float(env.model.vis.map.zfar)
    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy * (width / float(height))
    K = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    return K, znear, zfar, width, height

def _project_world_to_pixel(env, cam_id, pts_world, width=None, height=None):
    Rc, tc = _get_cam_RT(env, cam_id)
    K, *_ = _get_intrinsics(env, cam_id, width=width, height=height)
    pts = []
    for P in pts_world:
        Xc = Rc @ P + tc
        Z = -Xc[2] if Xc[2] < 0 else Xc[2]
        if abs(Z) < 1e-9:
            Z = 1e-9
        u = K[0, 0] * (Xc[0] / Z) + K[0, 2]
        v = -K[1, 1] * (Xc[1] / Z) + K[1, 2]
        pts.append([float(u), float(v), float(Z)])
    return pts

def _draw_overlay(env, cam_id, points_world, path):
    from PIL import ImageDraw
    _set_active_camera(env, cam_id)
    rgb = env.mujoco_renderer.render(render_mode="rgb_array")
    h, w = rgb.shape[0], rgb.shape[1]
    pts_px = _project_world_to_pixel(env, cam_id, points_world, width=w, height=h)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for (u, v, z) in pts_px:
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(255, 0, 0), width=2)
    im.save(path)

# _get_objects_pose now imported from providers.env_sim_util for reuse

def _render_and_save(env, cam_id, rgb_path, depth_path):
    _set_active_camera(env, cam_id)
    rgb = env.mujoco_renderer.render(render_mode="rgb_array")
    depth_raw = env.mujoco_renderer.render(render_mode="depth_array")
    Image.fromarray(rgb).save(rgb_path)
    # Save a visualization PNG clamped to [0,1] for quick viewing
    d_vis = np.clip(depth_raw, 0.0, 1.0)
    Image.fromarray((d_vis * 255).astype(np.uint8)).save(depth_path)
    # Save the exact metric depth as float32 .npy next to the PNG
    try:
        np.save(os.path.splitext(depth_path)[0] + ".npy", depth_raw.astype(np.float32))
    except Exception:
        pass

def _render_and_save_rgb(env, cam_id, rgb_path):
    _set_active_camera(env, cam_id)
    rgb = env.mujoco_renderer.render(render_mode="rgb_array")
    Image.fromarray(rgb).save(rgb_path)

gripper_open = True
traj_step = 1
frame_counter = 0
perception_logged = 0

# Trajectory pre-processing helpers 
# Motivation: Avoid calling env.step more than allowed max_path_length --> Exception You must reset the env manually once truncate==True 
def _filter_nearby_points(points, eps=2e-3):
    """Remove successive points closer than eps (meters)."""
    filtered = []
    prev = None
    for pt in list(points or []):
        pos = np.array(pt[:3], dtype=np.float64)
        if prev is None or float(np.linalg.norm(pos - prev)) >= float(eps):
            filtered.append(pt)
            prev = pos
    return filtered

def _estimate_path_length(env, points, start_eef=None):
    """Sum Euclidean distances from start EEF through points."""
    try:
        cur = np.array((start_eef if start_eef is not None else env.get_endeff_pos()), dtype=np.float64)
    except Exception:
        cur = np.zeros(3, dtype=np.float64)
    path = 0.0
    for pt in list(points or []):
        nxt = np.array(pt[:3], dtype=np.float64)
        path += float(np.linalg.norm(nxt - cur))
        cur = nxt
    return path

def _compute_iters_budget(path_len, n_points, user_iters_per_point):
    """Effective per-point iteration budget based on path length and count."""
    small_path = float(path_len) < 0.06  # 6 cm
    total_budget = 120 if small_path else 450
    n_pts = max(1, int(n_points))
    user = max(1, int(user_iters_per_point))
    return max(5, min(user, int(total_budget // n_pts)))


# Shared trajectory executor for both EXECUTE_TRAJECTORY and MOVE_EEF_ABS
async def _exec_traj(env, points, iters_per_point=15, open_override=None, cam_for_logging = None):
    global gripper_open, traj_step
    last_target = None
    if open_override is not None:
        gripper_open = bool(open_override)
    # Slightly larger per-step move to reduce total steps
    max_step = 0.03  # 3 cm per sim step
    # Pre-process points, estimate path length, and compute effective per-point iterations
    pts_in = list(points or [])
    filtered = _filter_nearby_points(pts_in, eps=2e-3)
    try:
        eef0 = env.get_endeff_pos().copy()
    except Exception:
        eef0 = None
    path_len = _estimate_path_length(env, filtered, start_eef=eef0)
    iters_eff = _compute_iters_budget(path_len, len(filtered), iters_per_point)
    print(f"[traj] points_in={len(pts_in)} points_used={len(filtered)} path_len={path_len:.3f}m iters_per_point={iters_eff} max_step={max_step}", flush=True)
    for pt in filtered:
        target = np.array(pt[:3], dtype=np.float64)
        last_target = target
        last_dist = None
        worse = 0
        for _ in range(max(int(iters_eff), 1)):
            ee = env.get_endeff_pos().copy()
            delta = target - ee
            dist = float(np.linalg.norm(delta))
            # Looser convergence threshold to avoid excess micro-steps
            if dist < 2e-3:
                break
            # Adaptive step: move toward target but clamp step length
            step_len = min(dist, max_step)
            if dist > 0:
                move_vec = (delta / dist) * step_len
            else:
                move_vec = np.zeros(3, dtype=np.float64)
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(move_vec / float(getattr(env, 'action_scale', 1.0)), -1.0, 1.0)
            action[3] = 1.0 if gripper_open else -1.0
            # Step the environment; if truncated==True occurs, stop early without reset
            try:
                env.step(action)
            except Exception as e:
                msg = str(e)
                if "truncate==True" in msg or "truncated" in msg:
                    # Log concise note and return current state without resetting
                    print("[traj] Error: !!!!!!! truncated early; returning without reset", flush=True)
                    ee_now = env.get_endeff_pos().copy()
                    pos_err = float(np.linalg.norm((last_target if last_target is not None else ee_now) - ee_now))
                    return ee_now.tolist(), pos_err
                # Propagate other exceptions
                raise
            # Log trajectory RGB at configured interval per sim step
            try:
                if traj_step % int(config.trajectory_log_every) == 0:                    
                    rgb_p = config.rgb_image_trajectory_path.format(step=traj_step)
                    cam_id = int(cam_for_logging) if cam_for_logging is not None else 0
                    _render_and_save_rgb(env, cam_id, rgb_p)
            except Exception as ex:
                print(f'++++ Failed to write env-cam traj rgb images', flush=True)
                
            traj_step += 1
            # If getting worse repeatedly, gently shrink step to stabilize
            if last_dist is not None and dist > last_dist + 1e-4:
                worse += 1
                if worse >= 3:
                    max_step *= 0.7
                    worse = 0
            else:
                worse = 0
            last_dist = dist
    ee_final = env.get_endeff_pos().copy()
    pos_err = float(np.linalg.norm((last_target if last_target is not None else ee_final) - ee_final))
    return ee_final.tolist(), pos_err
        
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
    parser.add_argument("--ws-host", type=str, default=os.environ.get("METAWORLD_WS_HOST", "127.0.0.1"), help="WebSocket host to bind")
    parser.add_argument("--ws-port", type=int, default=int(os.environ.get("METAWORLD_WS_PORT", "8765")), help="WebSocket port to bind")
    args = parser.parse_args()
    install_debug_reminder()
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
    # In WS mode the ready banner is sent on client connect

    # Optional local viewer for sanity checking
    if args.viewer:
        launch_viewer(env)

    # WebSocket endpoint replacing stdin/stdout loop
    async def _ws_handler(websocket):
        # Ensure access to outer-scope state
        global gripper_open, traj_step, frame_counter
        # Local helpers
        def _set_active_camera(cam_id):
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
        
        # Send ready banner
        await websocket.send(json.dumps({"status": "ready", "env": args.env, "cameras": cameras_info}))        
        while True:
            try:
                line = await websocket.recv()
            except Exception:
                break
            if not line or not str(line).strip():
                continue
            try:
                req = json.loads(line)
            except Exception:
                continue
            cmd = req.get("cmd")
            req_args = req.get("args")
            try:
                # The body of the handler is identical to the previous loop,
                # but responses are sent via websocket.send(...)
                if cmd == config.CAPTURE_IMAGES:
                    for label, cid in _resolve_selected_cameras():
                        if label == "head":
                            _render_and_save(env, cid, rgb_head_path, depth_head_path)
                        elif label == "wrist":
                            _render_and_save(env, cid, rgb_wrist_path, depth_wrist_path)
                    head_pos, head_quat = _get_cam_pose(env, head_id)
                    wrist_pos, wrist_quat = _get_cam_pose(env, wrist_id)
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
                        # TEMP: depth sidecar is currently raw OpenGL normalized from MuJoCo; mark encoding accordingly
                        cam_info = {
                            "head": {"K": K_head, "projectionMatrix": None, "viewMatrix": None, "znear": znear, "zfar": zfar, "width": width, "height": height, "fovy": fovy_h},
                            "wrist": {"K": K_wrist, "projectionMatrix": None, "viewMatrix": None, "znear": znear, "zfar": zfar, "width": width, "height": height, "fovy": fovy_w},
                            # MuJoCo renderer depth is OpenGL-normalized buffer; client will linearize to metric Z
                            "depth_encoding": "opengl",
                        }
                    except Exception:
                        cam_info = None
                    await websocket.send(json.dumps([head_pos, head_quat, wrist_pos, wrist_quat, "\u001b[92mFinished capturing head camera image!\u001b[0m", cam_info]))
                elif cmd == config.ADD_BOUNDING_CUBES:
                    await websocket.send(json.dumps(["\u001b[92mFinished adding bounding cubes to the environment!\u001b[0m"]))
                elif cmd == config.ADD_TRAJECTORY_POINTS:
                    pass
                elif cmd == config.EXECUTE_TRAJECTORY:
                    if isinstance(req_args, dict):
                        pts = req_args.get("points", [])
                        iters = int(req_args.get("iters_per_point", 15))
                        reply = bool(req_args.get("reply", False))
                        open_override = req_args.get("open_gripper")
                    else:
                        pts = req_args or []
                        iters = 15
                        reply = False
                        open_override = None
                    # Use first selected camera (head by default) for logging frames
                    selected = _resolve_selected_cameras()
                    cam_for_logging = selected[0][1] if selected else head_id
                    ee_final, pos_err = await _exec_traj(env, pts, iters_per_point=iters, open_override=open_override, cam_for_logging=cam_for_logging)
                    if reply:
                        await websocket.send(json.dumps({"eef_pos": ee_final, "pos_err": float(pos_err)}))
                elif cmd == config.OPEN_GRIPPER:
                    gripper_open = True
                    await websocket.send(json.dumps({"gripper_open": True}))
                elif cmd == config.CLOSE_GRIPPER:
                    gripper_open = False
                    await websocket.send(json.dumps({"gripper_open": False}))
                elif cmd == config.TASK_COMPLETED:
                    await websocket.send(json.dumps(["\u001b[92mFinished executing all generated trajectories!\u001b[0m"]))
                elif cmd == config.RESET_ENVIRONMENT:
                    env.reset()
                    gripper_open = True
                    traj_step = 1
                    await websocket.send(json.dumps(["\u001b[92mFinished resetting environment!\u001b[0m"]))
                elif cmd == config.GET_STATE:                    
                    eef = env.get_endeff_pos().copy()
                    names = []
                    if req_args and isinstance(req_args, dict):
                        names = req_args.get("objects", []) or []
                    objs = _get_objects_pose(env, names)
                    try:
                        qpos = float(env.data.joint("doorjoint").qpos.item())
                    except Exception:
                        qpos = None
                    await websocket.send(json.dumps({"eef_pos": eef.tolist(), "objects": objs, "doorjoint_angle": qpos}))
                elif cmd == config.GET_CAMERA_INFO:
                    try:
                        K_h, zn_h, zf_h, w, h = _get_intrinsics(env, head_id)
                        K_w, zn_w, zf_w, _, _ = _get_intrinsics(env, wrist_id)
                        await websocket.send(json.dumps({
                            "head": {"K": K_h, "znear": zn_h, "zfar": zf_h, "width": w, "height": h},
                            "wrist": {"K": K_w, "znear": zn_w, "zfar": zf_w, "width": w, "height": h},
                        }))
                    except Exception:
                        await websocket.send(json.dumps(None))
                elif cmd == config.CAPTURE_ANNOTATED_IMAGES:
                    # Log perception data sparsely: first N frames, then at interval
                    max_first = int(getattr(config, 'perception_log_first_n', 1))
                    interval = int(getattr(config, 'perception_log_interval_frames', 0))
                    should_log = (perception_logged < max_first) or (interval > 0 and frame_counter % interval == 0)
                    if should_log:
                        try:
                            try:
                                eef = env.tcp_center.copy()
                            except Exception:
                                eef = env.get_endeff_pos().copy()
                            names = []
                            if req_args and isinstance(req_args, dict):
                                names = req_args.get('objects', []) or []
                            objs = _get_objects_pose(env, names)
                            pts_world = [np.array(eef, dtype=np.float64)]
                            for obj in objs.values():
                                pts_world.append(np.array(obj['pos'], dtype=np.float64))
                            overlay_p = overlay_traj_path_tpl.format(step=traj_step)
                            selected = _resolve_selected_cameras()
                            cam_for_overlay = selected[0][1] if selected else head_id
                            _draw_overlay(env, cam_for_overlay, pts_world, overlay_p)
                        finally:
                            perception_logged += 1
                    frame_counter += 1
                    await websocket.send(json.dumps({"logged": should_log, "frame": frame_counter}))
                elif cmd == config.MOVE_EEF_ABS:
                    print(f'MOVE_EEF_ABS args={req_args}', flush=True)
                    
                    target = np.array(req_args.get("pos", [0, 0, 0]), dtype=np.float64)
                    iters = int(req_args.get("iters", 15))
                    open_override = req_args.get("open_gripper")                   
                    selected = _resolve_selected_cameras()
                    cam_for_logging = selected[0][1] if selected else head_id
                    ee_final, pos_err = await _exec_traj(env, [target.tolist()], iters_per_point=iters, open_override=open_override, cam_for_logging=cam_for_logging)
                    await websocket.send(json.dumps({"eef_pos": ee_final, "pos_err": float(pos_err)}))
                elif cmd == config.STEP_N:
                    action = np.array(req_args.get("action", [0, 0, 0, 0]), dtype=np.float32)
                    n = int(req_args.get("n", 1))
                    terminated = False
                    truncated = False
                    for _ in range(n):
                        _, _, terminated, truncated, _ = env.step(action)
                        if terminated or truncated:
                            break
                    await websocket.send(json.dumps({"terminated": terminated, "truncated": truncated}))
                elif cmd == config.SET_SEED:
                    seed = int(req_args.get("seed", 42))
                    env.seed(seed)
                    await websocket.send(json.dumps({"seed": seed}))
                elif cmd == config.SET_TASK_FROM_RAND_VEC:
                    rv = np.array(req_args.get("rand_vec", [0.1, 0.95, 0.15]), dtype=np.float64)
                    try:
                        setattr(env, "_freeze_rand_vec", True)
                        setattr(env, "_last_rand_vec", rv)
                        setattr(env, "_set_task_called", True)
                    except Exception:
                        pass
                    env.reset()
                    await websocket.send(json.dumps({"rand_vec": rv.tolist()}))
                elif cmd == config.QUERY_ENV_ATTR:
                    name = str(req_args.get("name", "")) if req_args else ""
                    result = None
                    try:
                        if hasattr(env, name):
                            attr = getattr(env, name)
                            result = attr() if callable(attr) else attr
                    except Exception:
                        result = None
                    await websocket.send(json.dumps({"name": name, "value": result}))                
                else:
                    await websocket.send(json.dumps({"error": "unknown_cmd", "cmd": cmd}))
            except Exception as e:
                try:
                    print(f'Exception {str(e)}', flush=True)
                    await websocket.send(json.dumps({"error": str(e)}))
                except Exception:
                    pass

    async def _serve():
        try:
            import websockets
        except Exception:
            print("[error] websockets is required. pip install websockets")
            raise
        async with websockets.serve(_ws_handler, args.ws_host, args.ws_port, max_size=None):
            await asyncio.Future()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass

    
if __name__ == "__main__":
    main()



