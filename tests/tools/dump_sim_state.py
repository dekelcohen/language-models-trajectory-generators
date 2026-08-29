"""Dump a small, comparable state snapshot from whichever sim this interpreter has.

Run under ``vlm_traj`` for PyBullet and ``vlm_genesis`` for Genesis, then diff the two
JSON files. Kept as a script (not a pytest) because the two sims live in different
conda envs and cannot be imported by the same test process.

    <vlm_traj>/python.exe   tests/tools/dump_sim_state.py --sim pybullet --task door -o pb.json
    <vlm_genesis>/python.exe tests/tools/dump_sim_state.py --sim genesis  --task door -o gs.json
    <vlm_traj>/python.exe   tests/tools/dump_sim_state.py --compare pb.json gs.json
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.ravel()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def snapshot(sim_name, task, settle_steps):
    import importlib
    import config

    # Sim-env profiles reconfigure camera and robot poses by writing straight into the
    # config module, and those writes outlive the scene. Taking two snapshots in one
    # process would otherwise stage the second task with the first task's camera - a
    # divergence that looks exactly like a simulator bug. Reload for a clean slate.
    importlib.reload(config)

    # config.RANDOM_TARGET_GRASP_OBJ_POSE randomises the grasp object at *import* time,
    # so two interpreters would otherwise stage different scenes and never agree.
    config.object_start_position = [-0.2, 0.4, 0.1]
    config.object_start_orientation_e = [0.0, 0.0, 0.0]

    import env as envmod
    from robot import Robot
    from sim_adapter.factory import get_adapter
    import logging

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("dump_sim_state")

    class _Args:
        mode = "default"
        robot = "franka"
        sim = sim_name
    _Args.task = task

    sim = get_adapter(sim_name)
    if hasattr(sim, "reserve_camera"):
        sim.reserve_camera(config.image_width, config.image_height,
                           fov=config.fov, near=config.near_plane, far=config.far_plane)
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    env = envmod.Environment(_Args, sim)
    env.simenv.configure_robot_pose()
    env.load()
    robot = Robot(_Args, logger, sim)
    sim.build()

    for _ in range(settle_steps):
        sim.step()

    ee_pos, ee_quat = sim.get_link_pose(robot.id, robot.ee_index)
    out = {
        "sim": sim_name,
        "task": task,
        "settle_steps": settle_steps,
        "ee_index": int(robot.ee_index),
        "ee_pos": _jsonable(ee_pos),
        "ee_quat_xyzw": _jsonable(ee_quat),
        "ee_euler": _jsonable(sim.euler_from_quat(ee_quat)),
        "joint_indices": _jsonable(robot.joint_indices),
        "joint_positions": _jsonable(
            [sim.get_joint_state(robot.id, j).position for j in robot.joint_indices]),
        "gripper_joint_indices": _jsonable(list(robot.gripper_joint_indices)),
        "robot_link_names": _jsonable(sim.list_link_names(robot.id)),
        "robot_joint_names": _jsonable(sim.list_joint_names(robot.id)),
        "state": _jsonable(env.simenv.get_state()),
    }

    # Camera matrices: the specific thing the user asked to pin ("same dims/meaning?").
    from sim_adapter import camera_math
    view = sim.compute_view_matrix(config.head_camera_position, config.camera_target_position, [0, 0, 1])
    proj = sim.compute_projection_matrix(config.fov, config.aspect, config.near_plane, config.far_plane)
    out["head_view_matrix"] = _jsonable(view)
    out["head_projection_matrix"] = _jsonable(proj)
    out["view_matrix_len"] = len(list(view))
    out["projection_matrix_len"] = len(list(proj))
    out["depth_encoding"] = getattr(sim, "depth_encoding", "opengl")
    _ = camera_math  # imported to assert the shared math module loads in both envs

    sim.disconnect()
    return out


def compare(path_a, path_b, atol):
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)

    worst = []
    for key in sorted(set(a) & set(b)):
        va, vb = a[key], b[key]
        if isinstance(va, list) and va and isinstance(va[0], (int, float)):
            if len(va) != len(vb):
                print(f"{key:28s} LENGTH {len(va)} vs {len(vb)}")
                continue
            diff = float(np.max(np.abs(np.array(va, float) - np.array(vb, float))))
            worst.append((diff, key))
            flag = "ok " if diff <= atol else "DIFF"
            print(f"{flag} {key:28s} max|d| = {diff:.6g}")
        elif va != vb:
            print(f"DIFF {key:28s} {va!r} != {vb!r}")
        else:
            print(f"ok  {key:28s} identical")

    if worst:
        print("\nlargest numeric divergences:")
        for diff, key in sorted(worst, reverse=True)[:8]:
            print(f"  {diff:12.6g}  {key}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default="pybullet")
    parser.add_argument("--task", default="door")
    parser.add_argument("--settle-steps", type=int, default=180)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.atol)

    data = snapshot(args.sim, args.task, args.settle_steps)
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
