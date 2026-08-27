"""Tune / audit the Franka Kitchen head-camera framing.

For a candidate (distance, pitch, target offset) this measures how much of the
task's target link the head camera can actually see, by rendering the scene
twice: once normally, and once with the robot teleported out of the world. The
ratio of target-link pixels between the two renders is the exact fraction of the
target occluded by the arm -- which is what "the arm is in the way" means for
the VLM.

Yaw is never swept: it stays pinned to HEAD_CAMERA_YAW so the 3D-coordinates
prompt stays truthful. Lateral framing is changed via the *target offset*, which
at a pinned yaw translates the camera sideways without rotating it.

Usage:
    python -m tests.tools.tune_kitchen_head_camera --task slide_cabinet
    python -m tests.tools.tune_kitchen_head_camera --audit        # all 7 tasks
"""

import argparse
import os

import numpy as np
import pybullet as p
import pybullet_data

import config
import env as envmod
from debug.dbg_utils import init_loguru_logger
from robot import Robot
from sim_envs.pybullet.franka_kitchen.tasks import HEAD_CAMERA_YAW, KITCHEN_TASKS

RENDER_SIZE = 512
ROBOT_PARKING_POSITION = [0.0, 0.0, -50.0]


class KitchenCameraProbe:
    """Loads one kitchen task and scores candidate head-camera framings."""

    def __init__(self, task_id, size=RENDER_SIZE):
        self.size = size
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        class _Args:
            mode = "default"
            robot = "franka"
            task = f"franka_kitchen:{task_id}"

        self.logger = init_loguru_logger("kitchen_camera_tuning.log")
        self.env = envmod.Environment(_Args)
        self.env.simenv.configure_robot_pose()
        self.env.load()
        self.robot = Robot(_Args, self.logger)
        self.robot.move(self.env, config.ee_start_position,
                        config.ee_start_orientation_e,
                        gripper_open=True, is_trajectory=False)

        self.simenv = self.env.simenv
        self.task = self.simenv.task
        self.kitchen_id = self.simenv.kitchen_id
        self.target = np.array(self.simenv._target_link_position(), dtype=float)
        # Mirror _target_link_position(): a goal_body task (the kettle) is scored
        # against the free body, everything else against a kitchen link.
        self.tracks_free_body = bool(self.task.goal_body) and self.simenv.kettle_id is not None
        if self.tracks_free_body:
            self.target_body = self.simenv.kettle_id
            self.target_link = None
        else:
            self.target_body = self.kitchen_id
            self.target_link = self._link_index(self.task.target_link)
        self.robot_home = p.getBasePositionAndOrientation(self.robot.id)
        self.projection = p.computeProjectionMatrixFOV(
            config.fov, 1.0, config.near_plane, config.far_plane)

    def _link_index(self, link_name):
        if link_name is None:
            return None
        for i in range(p.getNumJoints(self.kitchen_id)):
            if p.getJointInfo(self.kitchen_id, i)[12].decode() == link_name:
                return i
        return None

    def _view(self, distance, pitch, offset):
        look_at = [float(self.target[i] + offset[i]) for i in range(3)]
        return p.computeViewMatrixFromYawPitchRoll(
            look_at, distance, HEAD_CAMERA_YAW, pitch, 0, 2)

    def _render(self, view):
        """Return (target-link mask, rgb frame, robot mask).

        ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX is mandatory: without it the
        mask carries only body ids and every link-index test silently reads
        garbage (which looks exactly like "the target is never visible").
        """
        img = p.getCameraImage(self.size, self.size, view, self.projection,
                               renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        seg = np.array(img[4]).reshape(self.size, self.size)
        body = seg & ((1 << 24) - 1)
        link = (seg >> 24) - 1
        if self.target_link is None:
            mask = body == self.target_body
        else:
            mask = (body == self.target_body) & (link == self.target_link)
        rgb = np.array(img[2], dtype=np.uint8).reshape(self.size, self.size, 4)[:, :, :3]
        return mask, rgb, (body == self.robot.id)

    def score(self, distance, pitch, offset):
        view = self._view(distance, pitch, offset)
        mask, rgb, arm = self._render(view)

        # Same view with the arm parked out of the world -> how much of the
        # target *could* be seen. The difference is the arm's occlusion.
        p.resetBasePositionAndOrientation(self.robot.id, ROBOT_PARKING_POSITION,
                                          self.robot_home[1])
        clear_mask, _, _ = self._render(view)
        p.resetBasePositionAndOrientation(self.robot.id, *self.robot_home)

        potential = int(clear_mask.sum())
        visible = int(mask.sum())
        if potential == 0:
            return None  # target not in frame at all
        ys, xs = np.nonzero(clear_mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        half = self.size / 2.0
        return {
            "distance": distance,
            "pitch": pitch,
            "offset": tuple(round(float(o), 3) for o in offset),
            "visible_px": visible,
            "potential_px": potential,
            "occluded": round(1.0 - visible / potential, 3),
            "target_px": (int(cx), int(cy)),
            # 0 = dead centre, 1 = at the frame edge
            "off_centre": round(max(abs(cx - half), abs(cy - half)) / half, 3),
            "arm_px_frac": round(float(arm.mean()), 3),
            "rgb": rgb,
        }

    def save(self, result, path):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(result["rgb"]).save(path)

    def close(self):
        p.disconnect(self.client)


def _fmt(r):
    return (f"dist={r['distance']:<4} pitch={r['pitch']:<5} offset={r['offset']} "
            f"occluded={r['occluded']:<6} visible={r['visible_px']:<5}/{r['potential_px']:<5} "
            f"off_centre={r['off_centre']:<6} arm={r['arm_px_frac']}")


def audit(task_ids, save_dir=None):
    """Report the occlusion of each task's *current* configured framing."""
    report = []
    for task_id in task_ids:
        probe = KitchenCameraProbe(task_id)
        task = probe.task
        r = probe.score(task.camera_distance, task.camera_pitch,
                        task.camera_target_offset)
        if r is None:
            print(f"{task_id:16s} TARGET NOT IN FRAME")
            report.append((task_id, 1.0, 1.0))
        else:
            print(f"{task_id:16s} {_fmt(r)}")
            report.append((task_id, r["occluded"], r["off_centre"]))
            if save_dir:
                probe.save(r, os.path.join(save_dir, f"{task_id}.png"))
        probe.close()
    return report


def sweep(task_id, save_dir=None, top=8, size=256):
    probe = KitchenCameraProbe(task_id, size=size)
    task = probe.task
    current = probe.score(task.camera_distance, task.camera_pitch,
                          task.camera_target_offset)
    print("CURRENT:", _fmt(current) if current else "TARGET NOT IN FRAME")

    results = []
    for distance in [1.4, 1.6, 1.8, 2.0, 2.2]:
        for pitch in [-20, -15, -10, -5, 0]:
            for dy in [-0.5, -0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5]:
                for dz in [-0.25, -0.15, -0.05, 0.0, 0.1]:
                    r = probe.score(distance, pitch, (0.0, dy, dz))
                    if r is None or r["off_centre"] > 0.55:
                        continue
                    results.append(r)

    # Prefer an unoccluded, well-centred target that is large enough to segment.
    results.sort(key=lambda r: (r["occluded"], r["off_centre"], -r["visible_px"]))
    print(f"\ncandidates in frame: {len(results)}   top {top}:")
    for i, r in enumerate(results[:top]):
        print(f"  [{i}] {_fmt(r)}")
        if save_dir:
            probe.save(r, os.path.join(save_dir, f"{task_id}_cand{i}.png"))
    probe.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="slide_cabinet")
    parser.add_argument("--audit", action="store_true",
                        help="report current framing for every kitchen task")
    parser.add_argument("--save-dir", default="./images/kitchen_camera")
    parser.add_argument("--size", type=int, default=256,
                        help="render size used while sweeping (audit always uses 512)")
    args = parser.parse_args()

    if args.audit:
        audit(list(KITCHEN_TASKS), save_dir=args.save_dir)
    else:
        sweep(args.task, save_dir=args.save_dir, size=args.size)


if __name__ == "__main__":
    main()
