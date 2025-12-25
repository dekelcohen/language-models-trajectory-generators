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

    # Choose cameras by name fallback
    try:
        import mujoco
        cam_names = [env.model.id2name(i, mujoco.mjtObj.mjOBJ_CAMERA) for i in range(env.model.ncam)]
    except Exception:
        cam_names = []
    head_id = 0
    wrist_id = 0
    for i, name in enumerate(cam_names):
        if name and "corner" in name.lower():
            head_id = i
        if name and "gripper" in name.lower():
            wrist_id = i

    gripper_open = True
    traj_step = 1
    frame_counter = 0
    perception_logged = 0

    # Print a single JSON line as a ready banner for tests
    print(json.dumps({"status": "ready", "env": args.env}))
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

    def render_and_save(cam_id, rgb_path, depth_path):
        rgb = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=cam_id)
        depth = env.mujoco_renderer.render(render_mode="depth_array", camera_id=cam_id)
        Image.fromarray(rgb).save(rgb_path)
        # Preserve raw OpenGL-style depth in 0..1 range; client will linearize
        d = np.clip(depth, 0.0, 1.0)
        Image.fromarray((d * 255).astype(np.uint8)).convert("L").save(depth_path)

    def get_cam_pose(cam_id):
        import numpy as np
        from scipy.spatial.transform import Rotation
        pos = env.data.cam_xpos[cam_id].copy()
        R = env.data.cam_xmat[cam_id].reshape(3, 3)
        quat = Rotation.from_matrix(R).as_quat()  # x,y,z,w
        return pos.tolist(), quat.tolist()

    def get_cam_RT(cam_id):
        # world-to-camera extrinsics: R^T and t
        R = env.data.cam_xmat[cam_id].reshape(3, 3)
        t = env.data.cam_xpos[cam_id].copy()
        Rc = R.T
        tc = -Rc @ t
        return Rc, tc

    def get_intrinsics(cam_id):
        import mujoco
        fovy = float(env.model.cam_fovy[cam_id])
        width = int(env.width)
        height = int(env.height)
        znear = float(env.model.vis.map.znear)
        zfar = float(env.model.vis.map.zfar)
        fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        fx = fy * (width / float(height))
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
        return K, znear, zfar, width, height

    def project_world_to_pixel(cam_id, pts_world):
        Rc, tc = get_cam_RT(cam_id)
        K, *_ = get_intrinsics(cam_id)
        pts = []
        for P in pts_world:
            Xc = Rc @ P + tc
            u = (K[0, 0] * Xc[0] + K[0, 2] * Xc[2]) / Xc[2]
            v = (K[1, 1] * Xc[1] + K[1, 2] * Xc[2]) / Xc[2]
            pts.append([float(u), float(v), float(Xc[2])])
        return pts

    def draw_overlay(cam_id, points_world, path):
        from PIL import ImageDraw
        rgb = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=cam_id)
        pts_px = project_world_to_pixel(cam_id, points_world)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        for (u, v, z) in pts_px:
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(255, 0, 0), width=2)
        im.save(path)

    def get_objects_pose(names):
        from scipy.spatial.transform import Rotation
        out = {}
        for name in names:
            pos = quat = None
            try:
                pos = env.data.geom(name).xpos.copy()
                R = env.data.geom(name).xmat.reshape(3, 3)
                quat = Rotation.from_matrix(R).as_quat()
            except Exception:
                try:
                    pos = env.data.site(name).xpos.copy()
                    R = env.data.site(name).xmat.reshape(3, 3)
                    quat = Rotation.from_matrix(R).as_quat()
                except Exception:
                    try:
                        pos = env.data.body(name).xpos.copy()
                        R = env.data.body(name).xmat.reshape(3, 3)
                        quat = Rotation.from_matrix(R).as_quat()
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

        if cmd == config.CAPTURE_IMAGES:  # CAPTURE_IMAGES
            render_and_save(head_id, rgb_head_path, depth_head_path)
            render_and_save(wrist_id, rgb_wrist_path, depth_wrist_path)
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
                # EEF and requested objects in world
                eef = env.get_endeff_pos().copy()
                names = []
                if args and isinstance(args, dict):
                    names = args.get("objects", []) or []
                objs = get_objects_pose(names)
                pts_world = [eef]
                for obj in objs.values():
                    pts_world.append(np.array(obj["pos"], dtype=np.float64))
                overlay_p = overlay_traj_path_tpl.format(step=traj_step)
                draw_overlay(head_id, pts_world, overlay_p)
                perception_logged += 1
            frame_counter += 1
            print(json.dumps({"logged": should_log, "frame": frame_counter}))
            sys.stdout.flush()

        elif cmd == config.MOVE_EEF_ABS:  # MOVE_EEF_ABS
            target = np.array(args.get("pos", [0, 0, 0]), dtype=np.float64)
            iters = int(args.get("iters", 30))
            nonlocal_gripper_open = bool(args.get("open_gripper", True))
            for _ in range(iters):
                ee = env.get_endeff_pos().copy()
                delta = target - ee
                action = np.zeros(4, dtype=np.float32)
                if np.linalg.norm(delta) < 1e-4:
                    break
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = 1.0 if nonlocal_gripper_open else -1.0
                env.step(action)
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
            pass


if __name__ == "__main__":
    main()
