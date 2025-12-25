import os
import sys
import json
import time
import numpy as np
from PIL import Image
import argparse


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
    rgb_head_path = "./images/rgb_image_head.png"
    depth_head_path = "./images/depth_image_head.png"
    rgb_wrist_path = "./images/rgb_image_wrist.png"
    depth_wrist_path = "./images/depth_image_wrist.png"
    rgb_traj_path_tpl = "./images/trajectory/rgb_image_{step}.png"
    depth_traj_path_tpl = "./images/trajectory/depth_image_{step}.png"

    # Ensure output directories exist
    os.makedirs(os.path.dirname(rgb_head_path), exist_ok=True)
    os.makedirs(os.path.dirname(rgb_wrist_path), exist_ok=True)
    os.makedirs(os.path.dirname(rgb_traj_path_tpl.format(step=0)), exist_ok=True)

    env = EnvCls(render_mode=None, width=256, height=256)
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

    print(json.dumps(["\u001b[92mFinished setting up environment!\u001b[0m"]))
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
                for _ in range(300):
                    env.step(np.zeros(4, dtype=np.float32))
                    v.sync()
                # Close viewer and exit if only testing
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

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        cmd = req.get("cmd")
        args = req.get("args")

        if cmd == 1:  # CAPTURE_IMAGES
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

        elif cmd == 2:  # ADD_BOUNDING_CUBES (no-op)
            print(json.dumps(["\u001b[92mFinished adding bounding cubes to the environment!\u001b[0m"]))
            sys.stdout.flush()

        elif cmd == 3:  # ADD_TRAJECTORY_POINTS (no-op)
            # Accept and ignore
            pass

        elif cmd == 4:  # EXECUTE_TRAJECTORY
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
                rgb_p = rgb_traj_path_tpl.format(step=traj_step)
                d_p = depth_traj_path_tpl.format(step=traj_step)
                render_and_save(head_id, rgb_p, d_p)
                traj_step += 1

        elif cmd == 5:  # OPEN_GRIPPER
            gripper_open = True

        elif cmd == 6:  # CLOSE_GRIPPER
            gripper_open = False

        elif cmd == 7:  # TASK_COMPLETED
            print(json.dumps(["\u001b[92mFinished executing all generated trajectories!\u001b[0m"]))
            sys.stdout.flush()

        elif cmd == 8:  # RESET_ENVIRONMENT
            env.reset()
            gripper_open = True
            traj_step = 1
            print(json.dumps(["\u001b[92mFinished resetting environment!\u001b[0m"]))
            sys.stdout.flush()

        else:
            # Unknown
            pass


if __name__ == "__main__":
    main()
