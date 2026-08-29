"""Before/after regression guard for the PyBullet path.

The upcoming sim-adapter refactor rewrites every ``p.*`` call in ``env.py`` and
``robot.py``. This test pins the *numeric* behaviour of that code so the rewrite has
to be behaviour-preserving:

1. Boot ``grasp`` and ``door`` headlessly through the production classes
   (``env.Environment`` + ``robot.Robot``), stepping a fixed number of times so the
   physics state is deterministic.
2. Drive a scripted, fixed sequence of robot operations with tracing enabled.
3. Compare the resulting trace against ``tests/golden/pybullet/<task>.jsonl`` with a
   tight tolerance (rtol=0, atol=1e-9).

Deliberately in-process: ``run_simulation_environment`` calls ``env.update()`` in a
free-running loop, so how far the physics advances between two IPC messages depends on
wall-clock timing. Structural coverage of the IPC layer lives in
``test_pybullet_ipc_contract.py`` instead.

Regenerate the goldens with::

    $env:LMTG_UPDATE_GOLDEN = "1"; python -m pytest tests/test_pybullet_regression.py
"""

import json
import math
import os
import sys
import unittest

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
from debug import trace_utils

# rtol=0 / atol=1e-9: anything looser would let a real logic change slip through, and
# PyBullet DIRECT is bit-reproducible for a fixed step count.
ABS_TOLERANCE = 1e-9

TASKS = ["grasp", "door"]

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "pybullet")

# Steps used to settle the scene before anything is measured. Must stay fixed: it is
# part of the golden's definition.
SETTLE_STEPS = 180

# config.RANDOM_TARGET_GRASP_OBJ_POSE spawns the grasp object at a random pose, which
# makes the scene irreproducible. These are config's own documented fixed values.
PINNED_OBJECT_START_POSITION = [-0.2, 0.4, 0.1]
PINNED_OBJECT_START_ORIENTATION_E = [0.0, 0.0, 0.0]

# A short scripted trajectory (x, y, z, yaw-offset), mirroring the shape of what
# EXECUTE_TRAJECTORY receives from the agent.
SCRIPTED_TRAJECTORY = [
    [0.0, 0.55, 0.50, 0.0],
    [0.02, 0.58, 0.45, 0.1],
    [-0.02, 0.60, 0.52, -0.1],
]


class _Args:
    """Minimal stand-in for the argparse namespace the env/robot expect."""

    mode = "default"
    robot = "franka"
    task = None
    save_grasp_inputs = False


def _boot(task_name):
    """Bring up ``task_name`` in DIRECT mode and settle the physics deterministically."""
    import env as env_module
    from robot import Robot
    from debug.dbg_utils import init_loguru_logger
    from sim_adapter import get_adapter

    if p.isConnected():
        p.disconnect()
    sim = get_adapter("pybullet")
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    args = _Args()
    args.task = task_name
    utils.args = args

    # config.py randomizes the grasp object's spawn pose at *import* time when
    # RANDOM_TARGET_GRASP_OBJ_POSE is on. Pin it (to config's own documented
    # non-random values) so the golden is reproducible; production is untouched.
    config.object_start_position = list(PINNED_OBJECT_START_POSITION)
    config.object_start_orientation_e = list(PINNED_OBJECT_START_ORIENTATION_E)

    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger("pybullet_regression.log"), sim)
    sim.build()
    for _ in range(SETTLE_STEPS):
        environment.update()
    return environment, robot


def _eef_pose(robot):
    state = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)
    return {
        "pos": list(map(float, state[0])),
        "quat_xyzw": list(map(float, state[1])),
        "euler": list(map(float, p.getEulerFromQuaternion(state[1]))),
    }


def _joint_states(robot):
    return {
        "positions": [float(p.getJointState(robot.id, j)[0]) for j in robot.joint_indices],
        "velocities": [float(p.getJointState(robot.id, j)[1]) for j in robot.joint_indices],
        "applied_torque": [float(p.getJointState(robot.id, j)[3]) for j in robot.joint_indices],
    }


def _camera_record(robot, environment, camera):
    """Capture the pinhole parameters the agent actually receives for ``camera``."""
    position, orientation_q, view_matrix, projection_matrix = robot.get_camera_image(
        camera, environment, False, None, None
    )
    view = np.asarray(view_matrix, dtype=float)
    projection = np.asarray(projection_matrix, dtype=float)
    return {
        "camera_position": list(map(float, np.asarray(position, dtype=float).ravel())),
        "camera_orientation_q": list(map(float, np.asarray(orientation_q, dtype=float).ravel())),
        # Shape is recorded as well as the values: a 16-vector and a 4x4 carry the same
        # numbers but are not interchangeable across the IPC boundary.
        "view_matrix": view.ravel().tolist(),
        "view_shape": list(view.shape),
        "projection_matrix": projection.ravel().tolist(),
        "projection_shape": list(projection.shape),
        "znear": float(config.near_plane),
        "zfar": float(config.far_plane),
    }


def _pinhole_round_trip(robot, environment):
    """Project a ground-truth point to a pixel and back, through the production utils.

    This is the check that the projection/view matrices keep their dimensions *and*
    their meaning after the refactor, not just their numeric values.
    """
    state = environment.simenv.get_state()
    target = None
    for key in ("target_link_pos", "target_position", "handle_position", "door_handle_pos", "object_pos"):
        if state.get(key) is not None:
            target = np.array(state[key], dtype=float)
            break
    if target is None:
        return None

    position, orientation_q, view_matrix, projection_matrix = robot.get_camera_image(
        "head", environment, False, None, None
    )
    image_size = (config.image_width, config.image_height)
    cam_info = {
        "head": {"viewMatrix": view_matrix, "projectionMatrix": projection_matrix},
        "new_3d_proj": True,
    }
    pixel = utils.project_3d_world_pos_to_2d_pixel(
        position, orientation_q, "head", image_size, target, cam_info
    )
    if not pixel:
        return {"pixel": None}

    _, _, _, depth_buffer, _ = p.getCameraImage(
        config.image_width,
        config.image_height,
        viewMatrix=np.asarray(view_matrix).flatten(order="F"),
        projectionMatrix=np.asarray(projection_matrix).flatten(order="F"),
        renderer=p.ER_TINY_RENDERER,
    )
    depth = np.array(depth_buffer).reshape(config.image_height, config.image_width)
    px, py = int(pixel[0]), int(pixel[1])
    px = min(max(px, 0), config.image_width - 1)
    py = min(max(py, 0), config.image_height - 1)
    depth_value = float(depth[py, px])

    reconstructed = np.asarray(
        utils.get_world_point_world_frame(
            position, orientation_q, "head", image_size, [px, py, depth_value], cam_info=cam_info
        ),
        dtype=float,
    ).squeeze()

    return {
        "target": target.tolist(),
        "pixel": [px, py],
        "depth": depth_value,
        "reconstructed": reconstructed.tolist(),
    }


def _run_scenario(task_name, trace_path):
    """Execute the scripted scenario for ``task_name``, writing a trace to disk."""
    from common_utils import ensure_image_dirs_exist

    ensure_image_dirs_exist(delete=False)

    trace_utils.start(trace_path)
    try:
        trace_utils.set_context(sim="pybullet", task=task_name, robot=_Args.robot)
        environment, robot = _boot(task_name)

        trace_utils.trace_value("phase", "after_settle")
        trace_utils.trace_value("eef_pose", _eef_pose(robot))
        trace_utils.trace_value("joint_states", _joint_states(robot))
        trace_utils.trace_value("sim_state", environment.simenv.get_state())
        trace_utils.trace_value("coords_prompt", environment.simenv.get_3d_coordinates_prompt_section())
        trace_utils.trace_value("wrist_camera_params", environment.simenv.get_wrist_camera_params())
        trace_utils.trace_value("camera.head", _camera_record(robot, environment, "head"))
        trace_utils.trace_value("camera.wrist", _camera_record(robot, environment, "wrist"))
        trace_utils.trace_value("pinhole_round_trip", _pinhole_round_trip(robot, environment))

        # Gripper close / open, exactly as the CLOSE_GRIPPER / OPEN_GRIPPER handlers do.
        pose = _eef_pose(robot)
        robot.move(environment, pose["pos"], pose["euler"], gripper_open=False, is_trajectory=False)
        robot.gripper_open = False
        trace_utils.trace_value("phase", "after_close_gripper")
        trace_utils.trace_value("eef_pose", _eef_pose(robot))
        trace_utils.trace_value("joint_states", _joint_states(robot))

        pose = _eef_pose(robot)
        robot.move(environment, pose["pos"], pose["euler"], gripper_open=True, is_trajectory=False)
        robot.gripper_open = True
        trace_utils.trace_value("phase", "after_open_gripper")
        trace_utils.trace_value("eef_pose", _eef_pose(robot))

        # Scripted trajectory, mirroring the EXECUTE_TRAJECTORY handler.
        for i, point in enumerate(SCRIPTED_TRAJECTORY):
            robot.move(
                environment,
                point[:3],
                np.array(robot.ee_start_orientation_e) + np.array([0, 0, point[3]]),
                gripper_open=robot.gripper_open,
                is_trajectory=True,
                desc="regression" if i == 0 else None,
            )
        for _ in range(100):
            robot.step_env_and_record(environment, force_record=False)
        robot.step_env_and_record(environment, force_record=True)

        trace_utils.trace_value("phase", "after_trajectory")
        trace_utils.trace_value("eef_pose", _eef_pose(robot))
        trace_utils.trace_value("joint_states", _joint_states(robot))
        trace_utils.trace_value("trajectory_step", robot.trajectory_step)
        trace_utils.trace_value("sim_state", environment.simenv.get_state())
        trace_utils.trace_value("camera.head", _camera_record(robot, environment, "head"))
        trace_utils.trace_value("camera.wrist", _camera_record(robot, environment, "wrist"))
    finally:
        trace_utils.stop()
        if p.isConnected():
            p.disconnect()

    with open(trace_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_records(expected, actual, tolerance=ABS_TOLERANCE, path="root"):
    """Return a list of human-readable differences between two decoded traces."""
    diffs = []

    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                diffs.append(f"{path}.{key}: unexpected key (value={actual[key]!r})")
            elif key not in actual:
                diffs.append(f"{path}.{key}: missing key (expected={expected[key]!r})")
            else:
                diffs.extend(compare_records(expected[key], actual[key], tolerance, f"{path}.{key}"))
        return diffs

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(actual)} != expected {len(expected)}")
            return diffs
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs.extend(compare_records(e, a, tolerance, f"{path}[{i}]"))
        return diffs

    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            diffs.append(f"{path}: {actual!r} != expected {expected!r}")
        return diffs

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isnan(float(expected)) and math.isnan(float(actual)):
            return diffs
        delta = abs(float(actual) - float(expected))
        if delta > tolerance:
            diffs.append(f"{path}: {actual!r} != expected {expected!r} (delta={delta:.3e})")
        return diffs

    if expected != actual:
        diffs.append(f"{path}: {actual!r} != expected {expected!r}")
    return diffs


class TestPyBulletRegression(unittest.TestCase):
    maxDiff = None

    def tearDown(self):
        if p.isConnected():
            p.disconnect()

    def test_scenarios_match_golden(self):
        update = os.environ.get("LMTG_UPDATE_GOLDEN", "0") == "1"
        os.makedirs(GOLDEN_DIR, exist_ok=True)

        for task_name in TASKS:
            with self.subTest(task=task_name):
                golden_path = os.path.join(GOLDEN_DIR, f"{task_name}.jsonl")
                trace_path = os.path.join(config.images_folder, f"regression_{task_name}.jsonl")
                actual = _run_scenario(task_name, trace_path)
                self.assertTrue(actual, f"{task_name}: scenario produced no trace records")

                if update or not os.path.exists(golden_path):
                    with open(golden_path, "w", encoding="utf-8") as handle:
                        for record in actual:
                            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    print(f"[Regression] wrote golden {golden_path} ({len(actual)} records)")
                    continue

                with open(golden_path, encoding="utf-8") as handle:
                    expected = [json.loads(line) for line in handle if line.strip()]

                diffs = compare_records(expected, actual, path=task_name)
                self.assertFalse(
                    diffs,
                    f"{task_name}: {len(diffs)} difference(s) vs golden (atol={ABS_TOLERANCE}):\n"
                    + "\n".join(diffs[:40]),
                )


if __name__ == "__main__":
    unittest.main()
