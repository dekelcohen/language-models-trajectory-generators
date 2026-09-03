import os
import numpy as np
from debug.dbg_utils import init_loguru_logger
from PIL import Image
import traceback
import time
import config
import math
from robot import Robot
from common_utils import Trajectory
from debug import trace_utils
from providers.env_sim_util import _rotmat_to_quat_xyzw
from sim_adapter import get_adapter
from sim_envs.registry import get_simenv
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_EEF, GET_ROBOT_STATE, GET_STATE, VISUALIZE_GRASP_POSE, VISUALIZE_BOUNDING_BOX
# --- Debug helpers ------------------------------------------------------
# These build app-level overlays (bounding boxes, grasp poses, trajectory previews) out of
# two adapter primitives - ``draw_marker_sphere`` and ``draw_marker_cylinder`` - so the
# composition is written once and every simulator renders the same picture. Each takes the
# adapter as its first argument rather than reaching for a module-level client.
def add_debug_sphere(sim, pos_xyz, radius=0.015, color=(1, 0, 0, 1)):
    """
    Create a visual-only sphere at the specified 3D world coordinate.
    No collision shape and 0 mass so it doesn't affect physics.
    Returns a marker id or None on failure.
    """
    return sim.draw_marker_sphere(pos_xyz, radius, color)


def remove_debug_sphere(sim, marker_id):
    """Remove a previously created debug sphere by its marker id."""
    sim.remove_marker(marker_id)


def add_debug_cylinder_between(sim, p1, p2, radius=0.004, color=(1, 0, 0, 1)):
    """Draw a thin visual cylinder between two 3D points; returns a marker id or None.
    Rendered as a camera-visible marker (not a viewer overlay) so it shows up in captures.
    """
    return sim.draw_marker_cylinder(p1, p2, radius, color)

def draw_bounding_box(sim, cube_coords, line_radius=0.002, sphere_radius=0.004, color=(0, 1, 0, 1)):
    """Draw a 3D bounding box using visual cylinders/spheres so it renders in camera captures.

    Args:
        sim: the active SimAdapter.
        cube_coords: List of 9 coordinates.
                     Indices 0-3 are the bottom face corners.
                     Indices 5-8 are the top face corners.
                     Index 4 is typically the center point.
        line_radius: Radius of the cylinders drawing the edges.
        sphere_radius: Radius of the spheres at the corners and center.
        color: RGBA color of the bounding box (default green).

    Returns:
        list of marker IDs (for optional removal later via sim.remove_marker).
    """
    body_ids = []
    
    # Box edges defined by pairs of indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (5, 6), (6, 7), (7, 8), (8, 5),  # Top face
        (0, 5), (1, 6), (2, 7), (3, 8)   # Vertical edges connecting the faces
    ]
    
    # Draw cylindrical edges
    for i, j in edges:
        if i < len(cube_coords) and j < len(cube_coords):
            bid = add_debug_cylinder_between(sim, cube_coords[i], cube_coords[j], radius=line_radius, color=color)
            if bid is not None:
                body_ids.append(bid)

    # Draw spheres at the corners and the center point (index 4)
    for i in range(min(9, len(cube_coords))):
        # Optional: Make the center sphere slightly larger or a different color if needed
        # current_color = (1, 0, 0, 1) if i == 4 else color 
        bid = add_debug_sphere(sim, cube_coords[i], radius=sphere_radius, color=color)
        if bid is not None:
            body_ids.append(bid)

    return body_ids
    
def draw_grasp_pose(sim, pose_4x4, axis_length=0.06, finger_length=0.04, finger_spread=0.04, cylinder_radius=0.003, tcp_depth=0.08):
    """Draw a gripper pose in 3D using visual cylinders/spheres.

    Works headless as well as in a GUI (unlike viewer-overlay debug lines).

    Grasp pose convention (e.g. GraspNet / AnyGrasp):
      - Origin: gripper base frame (between finger pivot points).
      - X-axis: finger closing/opening direction.
      - Z-axis: approach direction (from base toward fingertips/object).
      - TCP (contact point) is at origin + Z * tcp_depth.

    Draws:
      - RGB coordinate axes (X=red, Y=green, Z=blue) at the TCP.
      - A simplified gripper: two fingers spread along X, extending
        backward (-Z) from the TCP toward the wrist.

    Args:
        sim: the active SimAdapter.
        pose_4x4: 4x4 homogeneous transformation matrix (numpy array).
        axis_length: length of each drawn axis (metres).
        finger_length: length of each finger line.
        finger_spread: half-distance between the two fingers (gripper opening).
        cylinder_radius: radius of the drawn cylinders.
        tcp_depth: distance from pose origin to fingertip contact point along Z.

    Returns:
        list of marker IDs (for optional removal later via sim.remove_marker).
    """
    pose = np.array(pose_4x4, dtype=float)
    origin = pose[:3, 3]
    x_axis = pose[:3, 0]
    y_axis = pose[:3, 1]
    z_axis = pose[:3, 2]  # approach direction

    # TCP (contact point) is ahead of the frame origin along approach
    tcp = origin + z_axis * tcp_depth

    body_ids = []

    # Draw coordinate frame axes at TCP
    body_ids.append(add_debug_cylinder_between(sim, tcp, tcp + x_axis * axis_length, radius=cylinder_radius, color=(1, 0, 0, 1)))
    body_ids.append(add_debug_cylinder_between(sim, tcp, tcp + y_axis * axis_length, radius=cylinder_radius, color=(0, 1, 0, 1)))
    body_ids.append(add_debug_cylinder_between(sim, tcp, tcp + z_axis * axis_length, radius=cylinder_radius, color=(0, 0, 1, 1)))

    # Draw simplified gripper fingers
    # X-axis = finger closing/opening direction; Z-axis = approach direction
    # Fingertips at TCP; fingers extend backward (-Z, toward wrist)
    finger_left_tip = tcp + x_axis * finger_spread
    finger_right_tip = tcp - x_axis * finger_spread
    finger_left_base = finger_left_tip - z_axis * finger_length
    finger_right_base = finger_right_tip - z_axis * finger_length

    gripper_color = (1, 0.6, 0, 1)  # orange
    # Finger lines
    body_ids.append(add_debug_cylinder_between(sim, finger_left_base, finger_left_tip, radius=cylinder_radius, color=gripper_color))
    body_ids.append(add_debug_cylinder_between(sim, finger_right_base, finger_right_tip, radius=cylinder_radius, color=gripper_color))
    # Crossbar connecting finger bases
    body_ids.append(add_debug_cylinder_between(sim, finger_left_base, finger_right_base, radius=cylinder_radius, color=gripper_color))
    # Palm line (connecting crossbar center further back toward wrist)
    crossbar_center = (finger_left_base + finger_right_base) / 2
    palm_end = crossbar_center - z_axis * 0.01
    body_ids.append(add_debug_cylinder_between(sim, crossbar_center, palm_end, radius=cylinder_radius * 0.7, color=gripper_color))

    # TCP sphere (grasp contact point)
    body_ids.append(add_debug_sphere(sim, tcp.tolist(), radius=cylinder_radius * 2, color=gripper_color))

    return [bid for bid in body_ids if bid is not None]




# --- Trajectory visualization helpers ----------------------------------
COLOR_TABLE = [
    (1,0,0,0.8),(0,0.6,0,0.8),(0,0.5,1,0.8),(1,0.5,0,0.8),
    (0.7,0,1,0.8),(1,0,0.7,0.8),(0,0.8,0.8,0.8),(0.8,0.8,0,0.8),
    (0.5,0.5,0.5,0.8),(0.2,0.2,1,0.8)
]

def _resolve_color(spec, color_cycle_idx):
    if isinstance(spec, str):
        name = spec.strip().lower()
        named = {
            'red': (1,0,0,0.8), 'green': (0,0.6,0,0.8), 'blue': (0,0.5,1,0.8),
            'orange': (1,0.5,0,0.8), 'purple': (0.7,0,1,0.8), 'magenta': (1,0,0.7,0.8),
            'cyan': (0,0.8,0.8,0.8), 'yellow': (0.8,0.8,0,0.8), 'gray': (0.5,0.5,0.5,0.8)
        }
        if name == 'random':
            c = COLOR_TABLE[color_cycle_idx % len(COLOR_TABLE)]
            return c, (color_cycle_idx + 1) % len(COLOR_TABLE)
        if name in named:
            return named[name], color_cycle_idx
    return (0,1,1,0.8), color_cycle_idx  # default cyan



def handle_add_trajectory_points(sim, trajectory, color_spec, permanent, marker_spec, logger,
                                 marker_steps, permanent_ids, color_cycle_idx):
    """Add trajectory preview visualization with spheres (+ optional polyline).
    - marker_steps: list of lists of ids (rolling history for non-permanent)
    - permanent_ids: list of ids that persist
    - Returns updated (marker_steps, permanent_ids, color_cycle_idx)
    """
    rgba, color_cycle_idx = _resolve_color(color_spec, color_cycle_idx)
    trajectory_points = [point[:3] for point in trajectory]
    
    # Add spheres and optional segments
    step_ids = []
    try:
        # Optional connecting polyline
        if isinstance(marker_spec, str) and marker_spec.lower() == 'line' and len(trajectory_points) > 1:
            rgb = [rgba[0], rgba[1], rgba[2], rgba[3] if len(rgba) > 3 else 0.8]
            for i in range(len(trajectory_points) - 1):
                _lid = add_debug_cylinder_between(sim, trajectory_points[i], trajectory_points[i+1], radius=0.004, color=rgb)
                if _lid is not None:
                    if permanent:
                        permanent_ids.append(_lid)
                    else:
                        step_ids.append(_lid)
        # Spheres at points
        for _pt in trajectory_points:
            _mid = add_debug_sphere(sim, _pt, radius=0.015, color=rgba)
            if _mid is not None:
                if permanent:
                    permanent_ids.append(_mid)
                else:
                    step_ids.append(_mid)
    except Exception as e:
        print("[Env] Warning: handle_add_trajectory_points draw failed:", e)
        try:
            traceback.print_exc()
        except Exception:
            pass

    # Rolling window pruning for non-permanent batches
    if not permanent and len(step_ids) > 0:
        marker_steps.append(step_ids)
        try:
            while len(marker_steps) > getattr(config, "visualize_traj_history_steps", 6):
                old_ids = marker_steps.pop(0)
                for _id in old_ids:
                    try:
                        remove_debug_sphere(sim, _id)
                    except Exception:
                        pass
        except Exception:
            pass

    # Debug points (RGB only)
    try:
        sim.draw_debug_points(trajectory_points, [[rgba[0], rgba[1], rgba[2]]] * len(trajectory_points), point_size=5, life_time=0)
    except Exception:
        pass

    if logger is not None:
        try:
            logger.info(OK + "Finished adding trajectory points to the environment!" + ENDC)
        except Exception:
            pass
    return marker_steps, permanent_ids, color_cycle_idx

# --- Simulation Environment Profiles -----------------------------------
# Sim-env profiles (grasp / door / franka_kitchen / ...) live in the
# sim_envs package. Resolve a --task name via sim_envs.registry.get_simenv.

def trajectory_point_orientation(ee_start_orientation_e, point):
    """Orientation a trajectory point asks for, as ``robot.move`` consumes it.

    Pose length is the format discriminator (see ``common_utils``):
      * len 4 -> ``[x, y, z, rotation]``, top-down. The historical expression, kept
        literally so no existing top-down task can drift.
      * len 6 -> ``[x, y, z, roll, pitch, yaw]``, absolute orientation, produced by
        ``side_grasp_pose()`` for targets with no graspable top face.
    """
    if len(point) == 6:
        return np.array(point[3:6], dtype=float)
    if len(point) == 4:
        return np.array(ee_start_orientation_e) + np.array([0, 0, point[3]])
    raise ValueError(
        f"Trajectory point must have length 4 or 6, got {len(point)}: {point}"
    )


class Environment:

    def __init__(self, args, sim):
        self.mode = args.mode
        # Make the sim honor --task similarly to Metaworld
        self.task = args.task
        #: The active SimAdapter. Everything below - and every sim-env profile - issues
        #: primitives through it, so this class has no simulator import at all.
        self.sim = sim
        self.simenv = get_simenv(self.task, sim)
        self._apply_required_robot(args)

    def _apply_required_robot(self, args):
        """Force the robot model when the scene only works with one.

        Runs in the environment subprocess, so mutating args is local to it and
        never rewrites the user's CLI choice for the rest of the pipeline.
        """
        try:
            required = self.simenv.required_robot()
        except Exception as e:
            print("[Env] Warning: required_robot() failed:", e)
            return
        if not required:
            return
        current = args.robot
        if current != required:
            print(
                f"[Env] Task '{self.task}' requires robot '{required}'; "
                f"overriding --robot {current}."
            )
            try:
                args.robot = required
            except Exception as e:
                print("[Env] Warning: could not override robot selection:", e)

    def load(self):
        # Apply per-task overrides and load assets via the selected SimEnv
        try:
            self.simenv.apply(self)
        except Exception as e:
            print("[Env] Warning: failed to apply SimEnv overrides:", e)
            traceback.print_exc()

        # Debug visualizer camera
        self.sim.reset_viewer_camera(
            config.camera_distance,
            config.camera_yaw,
            config.camera_pitch,
            config.camera_target_position,
        )

        # Load task-specific assets
        try:            
            self.simenv.load_assets(self)
        except Exception as e:
            print("[Env] Warning: failed to load SimEnv assets:", e)
            traceback.print_exc()

        # Scene-specific friction / joint motor setup
        try:
            self.simenv.tune_physics()
        except Exception as e:
            print("[Env] Warning: failed to tune SimEnv physics:", e)
            traceback.print_exc()

        if self.mode == "default":
            self.sim.set_viewer_options(show_gui=False, shadows=False)



    def update(self):

        # Scenes with coupled mechanisms the physics engine cannot simulate on its own
        # (e.g. Franka Kitchen knob -> burner plate) drive them here, every step.
        try:
            self.simenv.step_hook()
        except Exception as e:
            print("[Env] Warning: SimEnv step_hook failed:", e)
            traceback.print_exc()

        self.sim.step()
        time.sleep(config.control_dt)
    
def step_env_and_record_loop(env, robot):
    for _ in range(100):
        robot.step_env_and_record(env, force_record=False)
    # Force one final frame after the 100-step settling period
    robot.step_env_and_record(env, force_record=True)    
                
class _TracingConnection:
    """Delegates to the real connection while recording every message.

    Wrapping the connection instead of editing each handler keeps the IPC contract
    captured in one place, which is what the before/after refactor diff compares.
    Inert unless ``LMTG_TRACE`` is set.
    """

    def __init__(self, connection):
        self._connection = connection

    def send(self, payload):
        trace_utils.trace_value("ipc.send", payload)
        return self._connection.send(payload)

    def recv(self, *a, **kw):
        payload = self._connection.recv(*a, **kw)
        trace_utils.trace_value("ipc.recv", payload)
        return payload

    def poll(self, *a, **kw):
        return self._connection.poll(*a, **kw)

    def __getattr__(self, item):
        return getattr(self._connection, item)


def run_simulation_environment(args, env_connection, logger, sim=None):
    """The single, sim-agnostic app layer.

    Every IPC command is handled here for **all** simulators; only the
    :class:`~sim_adapter.base.SimAdapter` and the sim-env profile differ. An app-level
    change to ``EXECUTE_TRAJECTORY`` or ``CAPTURE_IMAGES`` is therefore made in exactly
    one place.

    ``sim`` lets a bootstrap (``sim_envs/genesis/genesis_env.py``) hand in an adapter it
    has already configured - Genesis has to reserve its render resolutions before the
    scene is built, which is impossible if the adapter is created here.
    """

    # Environment set-up
    # Initialize env process logger (console + ANSI-stripped file)
    sim_name = getattr(args, "sim", "pybullet") or "pybullet"
    logger = init_loguru_logger(f"env_{sim_name}.log")
                    
    logger.info(PROGRESS + "Setting up environment..." + ENDC)

    env_connection = _TracingConnection(env_connection)
    trace_utils.set_context(sim=sim_name, task=getattr(args, "task", None), robot=getattr(args, "robot", None))

    sim = get_adapter(sim_name) if sim is None else sim
    sim.connect(gui=False)  # headless offscreen rendering
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    plane = sim.load_urdf("plane.urdf")

    env = Environment(args, sim)
    # Let the task configure the robot/base pose before any URDFs are loaded
    # so the default (grasp) posture is preserved for the door task.    
    env.simenv.configure_robot_pose()
    env.load()

    robot = Robot(args, logger, sim)
    # Some simulators (Genesis) forbid adding entities after this point; PyBullet no-ops.
    sim.build()
    # Hold ids for visual debug spheres of trajectory points between requests
    trajectory_debug_marker_steps = []
    permanent_marker_ids = []
    color_cycle_idx = 0
    if env.simenv.move_to_start_pos():
        robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)

    # Diagnostics: compare GUI debug camera vs config spherical params and head image stats
    dbg_info = {
        "available": False,
        "yaw": None,
        "pitch": None,
        "dist": None,
        "target": None,
        "tuple_len": None,
    }
    try:
        dbg = sim.get_viewer_camera()
        dbg_info["tuple_len"] = dbg.get("tuple_len") if dbg else None
        if dbg and dbg.get("available"):
            dbg_info["available"] = True
            dbg_info["yaw"] = float(dbg["yaw"])
            dbg_info["pitch"] = float(dbg["pitch"])
            dbg_info["dist"] = float(dbg["distance"])
            dbg_info["target"] = list(map(float, dbg["target"]))
    except Exception:
        pass

    # Capture a quick head image to compute simple stats (mean/var) so we can detect blank frames
    head_stats = {
        "mean": None,
        "var": None,
        "size": None,
    }
    try:
        # Save head image to the standard path
        _ = robot.get_camera_image("head", env, save_camera_image=True,
                                   rgb_image_path=config.rgb_image_head_path,
                                   depth_image_path=config.depth_image_head_path)
        img = Image.open(config.rgb_image_head_path)
        arr = np.array(img, dtype=np.uint8)
        head_stats["size"] = list(arr.shape)
        head_stats["mean"] = float(arr.mean())
        head_stats["var"] = float(arr.var())
    except Exception:
        pass

    # Compute the current end-effector position to report back to the caller
    try:
        _eef_pos, _ = sim.get_link_pose(robot.id, robot.ee_index)
        eef_pos = list(map(float, _eef_pos))  # world position (x,y,z)
    except Exception:
        # Fallback to configured start position if link state is unavailable
        eef_pos = list(map(float, config.ee_start_position))

    # Add per-task 3D coordinate prompt section to the first status message
    try:
        _coords_section = env.simenv.get_3d_coordinates_prompt_section()
    except Exception:
        _coords_section = config.three_d_coordinates_prompt_section
    # Query environment-specific state via polymorphic get_state    
    sim_state = env.simenv.get_state()
    
    
    
    env_connection_message = (
        OK
        + (
            f"Finished setting up environment! sim={sim_name} conn={'headless'} "
            f"head_debug_view={getattr(config, 'head_camera_use_debug_view', False)} "
            f"dbg_cam_available={dbg_info['available']} dbg_tuple_len={dbg_info['tuple_len']} "
            f"dbg(yaw={dbg_info['yaw']}, pitch={dbg_info['pitch']}, dist={dbg_info['dist']}, target={dbg_info['target']}) "
            f"cfg(yaw={config.camera_yaw}, pitch={config.camera_pitch}, dist={config.camera_distance}, target={config.camera_target_position}) "
            f"head_img(size={head_stats['size']}, mean={head_stats['mean']}, var={head_stats['var']}) "
        )
        + ENDC
    )
    # Send EE position, coords prompt section, env-specific state, and the status message
    env_connection.send([eef_pos, _coords_section, sim_state, env_connection_message])

    while True:

        if env_connection.poll():

            env_connection_received = env_connection.recv()

            if env_connection_received[0] == CAPTURE_IMAGES:

                robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_trajectory_path.format(step=0), depth_image_path=None)
                robot.get_camera_image("wrist", env, save_camera_image=True, rgb_image_path=config.wrist_rgb_image_trajectory_path.format(step=0), depth_image_path=None)
                head_camera_position, head_camera_orientation_q, view_head, proj_head = robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_head_path, depth_image_path=config.depth_image_head_path)
                wrist_camera_position, wrist_camera_orientation_q, view_wrist, proj_wrist = robot.get_camera_image("wrist", env, save_camera_image=True, rgb_image_path=config.rgb_image_wrist_path, depth_image_path=config.depth_image_wrist_path)

                env_connection_message = OK + "Finished capturing head camera image!" + ENDC
                # Build cam_info with head intrinsics and view matrix                
                cam_info = {
                    "head": {"viewMatrix": view_head if "view_head" in locals() else None, "projectionMatrix": proj_head if "proj_head" in locals() else None, "znear": float(config.near_plane), "zfar": float(config.far_plane)},
                    "depth_encoding": sim.depth_encoding, "new_3d_proj" : True
                }
                env_connection.send([head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message, cam_info])

            elif env_connection_received[0] == ADD_BOUNDING_CUBES:

                bounding_cubes_world_coordinates = env_connection_received[1]

                for bounding_cube_world_coordinates in bounding_cubes_world_coordinates:
                    sim.draw_debug_line(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[1], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[2], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[3], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[0], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[5], bounding_cube_world_coordinates[6], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[6], bounding_cube_world_coordinates[7], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[7], bounding_cube_world_coordinates[8], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[8], bounding_cube_world_coordinates[5], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[5], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[6], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[7], [0, 1, 0], life_time=0)
                    sim.draw_debug_line(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[8], [0, 1, 0], life_time=0)
                    sim.draw_debug_points(bounding_cube_world_coordinates, [[0, 1, 0]] * len(bounding_cube_world_coordinates), point_size=5, life_time=0)

                env_connection_message = OK + "Finished adding bounding cubes to the environment!" + ENDC
                env_connection.send([env_connection_message])

            elif env_connection_received[0] == ADD_TRAJECTORY_POINTS:

                trajectory = env_connection_received[1]
                color_spec = env_connection_received[2] if len(env_connection_received) > 2 else None
                permanent = bool(env_connection_received[3]) if len(env_connection_received) > 3 else False
                marker_spec = env_connection_received[4] if len(env_connection_received) > 4 else "points"

                trajectory_debug_marker_steps, permanent_marker_ids, color_cycle_idx = handle_add_trajectory_points(
                    sim, trajectory, color_spec, permanent, marker_spec, logger,
                    trajectory_debug_marker_steps, permanent_marker_ids, color_cycle_idx
                )

            elif env_connection_received[0] == EXECUTE_TRAJECTORY:

                trajectory_obj = env_connection_received[1]
                if isinstance(trajectory_obj, Trajectory):
                    trajectory = trajectory_obj.points
                    desc = trajectory_obj.desc
                else:
                    trajectory = trajectory_obj
                    desc = None

                for i, point in enumerate(trajectory):
                    try:
                        ee_target_orientation_e = trajectory_point_orientation(robot.ee_start_orientation_e, point)
                    except ValueError as e:
                        raise ValueError(f"Trajectory point {i}: {e}") from e
                    robot.move(env, point[:3], ee_target_orientation_e, gripper_open=robot.gripper_open, is_trajectory=True, desc=desc if i == 0 else None)

                step_env_and_record_loop(env, robot)                
                
                env_connection.send([OK + "Finished executing generated trajectory!" + ENDC, robot.trajectory_step])

            elif env_connection_received[0] == OPEN_GRIPPER:

                ee_current_position, ee_current_orientation_q = sim.get_link_pose(robot.id, robot.ee_index)
                ee_current_orientation_e = sim.euler_from_quat(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=True, is_trajectory=False)

                robot.gripper_open = True

                logger.info(OK + "Finished opening gripper!" + ENDC)

            elif env_connection_received[0] == CLOSE_GRIPPER:

                ee_current_position, ee_current_orientation_q = sim.get_link_pose(robot.id, robot.ee_index)
                ee_current_orientation_e = sim.euler_from_quat(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=False, is_trajectory=False)

                robot.gripper_open = False

                logger.info(OK + "Finished closing gripper!" + ENDC)

            elif env_connection_received[0] == TASK_COMPLETED:

                env_connection_message = OK + "Finished executing all generated trajectories!" + ENDC
                env_connection.send([env_connection_message])

            elif env_connection_received[0] == RESET_EEF:
                # Re-home the ARM ONLY to its start pose. Does NOT reset object/world
                # state, and deliberately does NOT reset robot.trajectory_step so that
                # trajectory image frames from previous subtasks are preserved.
                robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)
                robot.gripper_open = True

                step_env_and_record_loop(env, robot)

                env_connection_message = OK + "Finished re-homing end-effector!" + ENDC
                env_connection.send([env_connection_message])
            elif env_connection_received[0] == GET_ROBOT_STATE:
                try:
                    _eef_pos, _ = sim.get_link_pose(robot.id, robot.ee_index)
                    eef_pos = list(map(float, _eef_pos))
                except Exception:
                    eef_pos = list(map(float, config.ee_start_position))
                try:
                    env_connection.send({"eef_pos": eef_pos})
                except Exception:
                    env_connection.send({})

            elif env_connection_received[0] == GET_STATE:
                try:
                    _eef_pos, _ = sim.get_link_pose(robot.id, robot.ee_index)
                    eef_pos = list(map(float, _eef_pos))
                except Exception:
                    eef_pos = list(map(float, config.ee_start_position))
                try:
                    sim_state = env.simenv.get_state()
                except Exception:
                    sim_state = {}
                try:
                    env_connection.send({"eef_pos": eef_pos, "sim_state": sim_state})
                except Exception:
                    env_connection.send({})

            elif env_connection_received[0] == VISUALIZE_GRASP_POSE:
                grasp_poses = env_connection_received[1]
                try:
                    if isinstance(grasp_poses, np.ndarray) and grasp_poses.ndim == 3:
                        # Multiple poses (N, 4, 4)
                        for pose in grasp_poses:
                            draw_grasp_pose(sim, pose)
                    else:
                        # Single pose (4, 4)
                        draw_grasp_pose(sim, grasp_poses)
                    env_connection.send([OK + f"Visualized {len(grasp_poses) if isinstance(grasp_poses, np.ndarray) and grasp_poses.ndim == 3 else 1} grasp pose(s)." + ENDC])
                except Exception as e:
                    env_connection.send([FAIL + f"Failed to visualize grasp pose: {e}" + ENDC])

            elif env_connection_received[0] == VISUALIZE_BOUNDING_BOX:
                box_cubes = env_connection_received[1]
                try:
                    count = 0
                    for cube_coords in box_cubes:
                        draw_bounding_box(sim, cube_coords)
                        count += 1
                    env_connection.send([OK + f"Visualized {count} bounding box(es) (--vis-box)." + ENDC])
                except Exception as e:
                    env_connection.send([FAIL + f"Failed to visualize bounding box: {e}" + ENDC])

        env.update()

def get_grasp_pose_candidates(object_name):
    """Load pre-computed grasp pose candidates from an .npz file.

    Args:
        object_name: Name used to locate the file at ./outputs/graspgen/grasp_poses_{object_name}.npz

    Returns:
        poses: np.ndarray of shape (N, 4, 4) – 4x4 homogeneous transformation matrices.
        scores: np.ndarray of shape (N,) – grasp quality scores (higher is better).
    """
    npz_path = './outputs/graspgen/grasp_poses_open_door.npz'
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Grasp poses file not found: {npz_path}")

    data = np.load(npz_path)
    poses = data["poses"]   # (N, 4, 4)
    scores = data["scores"] # (N,)
    return poses, scores


# --- Minimal GUI demo to interactively test scene kinematics ---
def run_sim_demo(task_p='door', disable_forces: bool = False,
                 gui: bool = True,
                 sim_name: str = "pybullet",
                 ee_offset_from_base=(0.0, 0.08, 0.45),
                 ee_orientation_e_override=None,
                 strengthen_door=True):
    """
    Launch a minimal interactive session that loads the environment with the scene.
    - ``gui=True`` opens the simulator's interactive viewer; ``gui=False`` runs headless.
    - Disables door joint motor forces for easy mouse-pick/drag of the hinge/latch (GUI).
    - In GUI, idles so the scene can be inspected; headless, exits after setup.
    - ``sim_name`` selects the provider, so the same demo drives PyBullet or Genesis.
    """
    logger = init_loguru_logger(f"env_{sim_name}.log")
    try:
        tag = "[Env GUI]" if gui else "[Env Direct]"
        sim = get_adapter(sim_name)
        sim.connect(gui=gui)
        sim.set_asset_search_path()
        sim.set_gravity(0, 0, -9.81)
        _ = sim.load_urdf("plane.urdf")

        class _Args:
            mode = "default"
            robot = "franka"
            task = task_p

        env = Environment(_Args, sim)
        # Same ordering as the production path (run_simulation_environment): the sim-env
        # may override the robot base pose, which must be set before the robot is loaded.
        env.simenv.configure_robot_pose()
        env.load()

        # The blocks below are door-task debug scratch space; skip them for other sim-envs.
        is_door_task = getattr(env.simenv, "door_id", None) is not None

        DRAW_GRASP_POST_MAT = is_door_task
        if DRAW_GRASP_POST_MAT:
            grasp_pose =   np.array([
                [ 0.854062,   -0.5120964,   0.09129835, -0.34211737], #-0.34211737
                [-0.49542382, -0.7473097,   0.44281337,  -0.12207034], #-0.12207034
                [-0.15853497, -0.42342147, -0.89195347,  0.9260877], # 0.9260877
                [ 0.0,         0.0,         0.0,         1.0]
            ])
            
            # pose[9] - mid lever form above (not sure exactly perpedicular but close)
            
            poses, scores = get_grasp_pose_candidates("door handle")
            
        
            draw_grasp_pose(sim, poses[47])
        GRASP_POSE = False 
        if GRASP_POSE:
            trajectory = [[-0.279, -0.126, 0.600],[-0.279, -0.126, 0.700],[-0.300, 0.100, 0.700],[-0.300, 0.100, 0.600]]
            trajectory_debug_marker_steps, permanent_marker_ids, color_cycle_idx = handle_add_trajectory_points(
                    sim, trajectory = trajectory, color_spec='blue', permanent=True, marker_spec='line', logger=logger,
                    marker_steps = [], permanent_ids = [], color_cycle_idx = 0)
                
        OVERLAY_COORD_TEST = is_door_task
        if OVERLAY_COORD_TEST:
            # Temp coordinates check 
            # door_handle_pos = [-0.07745519744833454, -0.00880230021590278, 0.672376] # from pybullet 
            door_handle_pos = [-0.319, -0.127, 0.638] # From perception 
            # hinge_pos = [-0.21654298331956742, -0.14765775679373436, 0.697376] # from pybullet
            hinge_pos = [-0.09 + door_handle_pos[0], -0.2 + door_handle_pos[1], door_handle_pos[2] + 0.05] # moved manually in sim
            print('hinge_pos', hinge_pos)
            object_pos = hinge_pos # door_handle_pos
            # object_pos = [0,0,0] 
            object_start_orientation_q = sim.quat_from_euler(config.object_start_orientation_e)
                        
            add_debug_sphere(sim, object_pos)
            
            
            if False:
                sim.load_urdf(
                    "ycb_assets/002_master_chef_can.urdf",
                    object_pos,
                    object_start_orientation_q,
                    fixed_base=False,
                    scaling=config.global_scaling)
    
        # Print camera poses for debugging
        try:
            
            print(tag + " Debug camera params:")
            print(f"  distance={config.camera_distance} yaw={config.camera_yaw} pitch={config.camera_pitch}")
            print(f"  target={config.camera_target_position}")
            if gui:
                dbg = sim.get_viewer_camera()
                print(f"  viewer camera: {dbg}")
        except Exception as e:
            print(tag + " Warning: reading the viewer camera failed:", e)

        print(tag + " Head camera params:")
        print(f"  position={config.head_camera_position}")
        print(f"  orientation_e={config.head_camera_orientation_e}")
        
        # Hold the door closed at startup with stronger motor forces (optional)
        if strengthen_door:
            try:
                if getattr(env.simenv, "door_id", None) is not None:
                    if getattr(env.simenv, "door_hinge_index", None) is not None:
                        sim.reset_joint_state(env.simenv.door_id, env.simenv.door_hinge_index, 0.0)
                        sim.set_joint_position(env.simenv.door_id, env.simenv.door_hinge_index,
                                               target=0.0, force=200)
                    if getattr(env.simenv, "latch_index", None) is not None:
                        sim.reset_joint_state(env.simenv.door_id, env.simenv.latch_index, 0.0)
                        sim.set_joint_position(env.simenv.door_id, env.simenv.latch_index,
                                               target=0.0, force=200)
            except Exception as e:
                # Default forces from load() remain if this fails
                print(tag + " Warning: could not strengthen door motors:", e)

        robot = Robot(_Args, logger, sim)
        sim.build()
        if env.simenv.move_to_start_pos():
            # Prefer the sim-env's declared home pose (set by configure_robot_pose); fall back
            # to a safe EE pose near and above the robot base to avoid falling into the door.
            simenv_home = env.simenv.get_ee_start_pose()
            if simenv_home is not None:
                safe_ee_pos, home_ee_ori = list(simenv_home[0]), list(simenv_home[1])
            else:
                base_x, base_y, base_z = config.base_start_position_franka
                safe_ee_pos = [
                    base_x + ee_offset_from_base[0],
                    base_y + ee_offset_from_base[1],
                    base_z + ee_offset_from_base[2],
                ]
                home_ee_ori = config.ee_start_orientation_e
            safe_ee_ori = ee_orientation_e_override if ee_orientation_e_override is not None else home_ee_ori
            print(f'safe_ee_pos={safe_ee_pos}')
            # Place the gripper above base immediately; do not let gravity + IK swing it into the door
            try:
                robot.move(env, safe_ee_pos, safe_ee_ori, gripper_open=True, is_trajectory=False)
            except Exception as e:
                print(tag + " Warning: initial EE placement failed:", e)

        print(f'config.base_start_position_franka={config.base_start_position_franka}')
        print(f'config.base_start_orientation_e_franka={config.base_start_orientation_e_franka}')
        print(f'config.joint_start_positions_franka={config.joint_start_positions_franka}')
        

        # Capture and save the head camera image using the debug view
        head_stats = {"size": None, "mean": None, "var": None}
        try:
            head_pos, head_q, view_head, proj_head = robot.get_camera_image(
                "head",
                env,
                save_camera_image=True,
                rgb_image_path=config.rgb_image_head_path,
                depth_image_path=config.depth_image_head_path,
            )
            print(tag + " Saved head camera image:", config.rgb_image_head_path)
            print(tag + " Head camera actual pose:")
            print("  position=", head_pos)
            print("  orientation_q=", head_q)
            try:
                img = Image.open(config.rgb_image_head_path)
                arr = np.array(img, dtype=np.uint8)
                head_stats["size"] = list(arr.shape)
                head_stats["mean"] = float(arr.mean())
                head_stats["var"] = float(arr.var())
            except Exception:
                pass
        except Exception as e:
            print(tag + " Warning: failed to capture head camera image:", e)

        if disable_forces:
            # Disable motors so user drag isn't resisted
            try:
                if getattr(env.simenv, "door_id", None) is not None:
                    if getattr(env.simenv, "door_hinge_index", None) is not None:
                        sim.set_joint_position(env.simenv.door_id, env.simenv.door_hinge_index, target=None, force=0)
                    if getattr(env.simenv, "latch_index", None) is not None:
                        sim.set_joint_position(env.simenv.door_id, env.simenv.latch_index, target=None, force=0)
            except Exception as e:
                print(tag + " Failed to disable door motors:", e)
                traceback.print_exc()

        # Diagnostic line mirroring the headless setup
        dbg_available = False
        dbg_tuple_len = None
        dbg_yaw = dbg_pitch = dbg_dist = None
        dbg_target = None
        try:
            if gui:
                dbg = sim.get_viewer_camera()
                dbg_tuple_len = dbg.get("tuple_len") if dbg else None
                if dbg and dbg.get("available"):
                    dbg_available = True
                    dbg_yaw = float(dbg["yaw"])
                    dbg_pitch = float(dbg["pitch"])
                    dbg_dist = float(dbg["distance"])
                    dbg_target = list(map(float, dbg["target"]))
        except Exception:
            pass

        print(
            f"{tag} Setup: sim={sim_name} conn={'gui' if gui else 'headless'} "
            f"head_debug_view={getattr(config, 'head_camera_use_debug_view', False)} "
            f"dbg_cam_available={dbg_available} dbg_tuple_len={dbg_tuple_len} "
            f"dbg(yaw={dbg_yaw}, pitch={dbg_pitch}, dist={dbg_dist}, target={dbg_target}) "
            f"cfg(yaw={config.camera_yaw}, pitch={config.camera_pitch}, dist={config.camera_distance}, target={config.camera_target_position}) "
            f"head_img(size={head_stats['size']}, mean={head_stats['mean']}, var={head_stats['var']})"
        )

        # Real-time simulation for natural interaction (GUI only)
        if gui:
            sim.set_real_time(True)
            print(tag + " Running. Click-and-drag the door; press ESC to quit.")
            print(tag + " Click in the viewport to print the board's current pose "
                        "(paste it into SimEnvDoor._load_board).")
            board_id = getattr(env.simenv, "board_id", None)
            # Scenes with machine-checkable success (e.g. franka_kitchen:*) print their
            # live task state so a manual drag can be checked against the goal.
            has_criteria = bool(env.simenv.get_success_criteria())
            if has_criteria:
                print(tag + " Task criteria:", env.simenv.get_success_criteria())
                print(tag + " Drag the target joint/object; task state prints once a second.")
            last_state_print = 0.0
            while sim.viewer_is_open():
                # Real-time stepping is handled by the simulator, but coupled
                # mechanisms still need their per-step hook driven manually.
                try:
                    env.simenv.step_hook()
                except Exception:
                    pass
                if has_criteria and (time.time() - last_state_print) >= 1.0:
                    last_state_print = time.time()
                    try:
                        st = env.simenv.get_state()
                        tgt = st.get('target_link_pos')
                        tgt_s = None if tgt is None else [round(v, 3) for v in tgt]
                        print(f"{tag} task={st.get('task')} error={st.get('task_error')} "
                              f"success={st.get('success')} target_link={st.get('target_link')} "
                              f"target_link_pos={tgt_s}")
                    except Exception as e:
                        print(tag + " get_state() failed:", e)
                # On any mouse button press, print the board pose so it can be
                # copied back into _load_board() as the initial position/orientation.
                if board_id is not None:
                    for _ev in sim.get_click_events():
                        b_pos, b_orn_q = sim.get_base_pose(board_id)
                        b_orn_e = sim.euler_from_quat(b_orn_q)
                        print(f"{tag} board_position = [{b_pos[0]:.4f}, {b_pos[1]:.4f}, {b_pos[2]:.4f}]")
                        print(f"{tag} board_orientation_e = [{b_orn_e[0]:.4f}, {b_orn_e[1]:.4f}, {b_orn_e[2]:.4f}]  "
                              f"# quaternion = [{b_orn_q[0]:.4f}, {b_orn_q[1]:.4f}, {b_orn_q[2]:.4f}, {b_orn_q[3]:.4f}]")
                time.sleep(0.01)
    except Exception as e:
        print(tag + " Exception:", e)
        traceback.print_exc()
    finally:
        pass


if __name__ == "__main__":
    # Entry point for quick, no-code sim-env inspection in GUI mode.
    #   python env.py                                   # door (default) on pybullet
    #   python env.py --task franka_kitchen:microwave
    #   python env.py --task grasp --direct
    import argparse as _argparse
    from sim_adapter import SUPPORTED_SIMS as _SUPPORTED_SIMS
    from sim_envs.registry import list_task_ids as _list_task_ids

    _parser = _argparse.ArgumentParser(
        description="Manual/debug visualisation of a sim-env. "
                    "Drag joints with the mouse; task state is printed live.")
    _parser.add_argument("--task", default="door",
                         help=f"sim-env task id. Available: {', '.join(_list_task_ids())}")
    _parser.add_argument("--sim", default="pybullet", choices=list(_SUPPORTED_SIMS),
                         help="simulator provider to run the scene in")
    _parser.add_argument("--direct", action="store_true",
                         help="headless instead of the GUI (load smoke test)")
    _parser.add_argument("--disable-forces", action="store_true",
                         help="zero the joint motor forces so joints can be freely dragged")
    _parser.add_argument("--no-strengthen-door", dest="strengthen_door",
                         action="store_false", default=True,
                         help="door task only: do not hold the door shut at startup")
    _cli = _parser.parse_args()

    run_sim_demo(task_p=_cli.task,
                 disable_forces=_cli.disable_forces,
                 gui=not _cli.direct,
                 sim_name=_cli.sim,
                 strengthen_door=_cli.strengthen_door)



