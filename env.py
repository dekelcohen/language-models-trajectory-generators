import pybullet as p
import numpy as np
import pybullet_data
from PIL import Image
import traceback
import time
import config
from robot import Robot
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT
from config import SET_DOOR_STATE, CAPTURE_TRAJECTORY_FRAME

class Environment:

    def __init__(self, args):

        self.mode = args.mode

    def load(self):

        p.resetDebugVisualizerCamera(config.camera_distance, config.camera_yaw, config.camera_pitch, config.camera_target_position)

        object_start_position = config.object_start_position
        object_start_orientation_q = p.getQuaternionFromEuler(config.object_start_orientation_e)
        object_model = p.loadURDF("ycb_assets/003_cracker_box.urdf", object_start_position, object_start_orientation_q, useFixedBase=False, globalScaling=config.global_scaling)

        # Load Adroit door URDF (fixed frame with hinge and latch)
        try:
            door_start_position = [-0.11, 0.04, 0.45]
            door_start_orientation_q = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
            self.door_id = p.loadURDF("my_assets/adroit_door/adroit_door.urdf", door_start_position, door_start_orientation_q, useFixedBase=True)

            # Cache joint indices by name for control convenience
            self.door_hinge_index = self._get_joint_index_by_name(self.door_id, "door_hinge")
            self.latch_index = self._get_joint_index_by_name(self.door_id, "latch_joint")

            # Initialize motors to hold at zero (closed door and latch)
            if self.door_hinge_index is not None:
                p.setJointMotorControl2(self.door_id, self.door_hinge_index, p.POSITION_CONTROL, targetPosition=0.0, force=50)
            if self.latch_index is not None:
                p.setJointMotorControl2(self.door_id, self.latch_index, p.POSITION_CONTROL, targetPosition=0.0, force=30)
        except Exception as e:
            print("[Env] Failed to load or initialize adroit_door URDF:", e)
            traceback.print_exc()

        if self.mode == "default":

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)



    def update(self):

        p.stepSimulation()
        time.sleep(config.control_dt)

    def _get_joint_index_by_name(self, body_id, joint_name):
        try:
            for j in range(p.getNumJoints(body_id)):
                info = p.getJointInfo(body_id, j)
                if info[1].decode("utf-8") == joint_name:
                    return j
        except Exception as e:
            print(f"[Env] Error reading joints of body {body_id}:", e)
            traceback.print_exc()
        return None

    def set_door_state(self, door_angle=None, latch_angle=None):
        # Position-control door hinge and latch if provided
        try:
            if door_angle is not None and self.door_hinge_index is not None:
                p.setJointMotorControl2(self.door_id, self.door_hinge_index, p.POSITION_CONTROL, targetPosition=float(door_angle), force=50)
            if latch_angle is not None and self.latch_index is not None:
                p.setJointMotorControl2(self.door_id, self.latch_index, p.POSITION_CONTROL, targetPosition=float(latch_angle), force=30)
        except Exception as e:
            print("[Env] Failed to set door state:", e)
            traceback.print_exc()

    def set_door_state_array(self, door_angle, latch_angle):
        # Control both joints in a single array call
        try:
            indices = []
            targets = []
            forces = []
            if self.door_hinge_index is not None:
                indices.append(self.door_hinge_index)
                targets.append(float(door_angle))
                forces.append(50)
            if self.latch_index is not None:
                indices.append(self.latch_index)
                targets.append(float(latch_angle))
                forces.append(30)
            if indices:
                p.setJointMotorControlArray(self.door_id, indices, p.POSITION_CONTROL, targetPositions=targets, forces=forces)
        except Exception as e:
            print("[Env] Failed to set door state array:", e)
            traceback.print_exc()



def run_simulation_environment(args, env_connection, logger):

    # Environment set-up
    logger.info(PROGRESS + "Setting up environment..." + ENDC)

    physics_client = p.connect(p.DIRECT) # Dekel: Changed for headless offscreen (no GUI) - was p.GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    env = Environment(args)
    env.load()

    robot = Robot(args)
    robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)

    env_connection_message = OK + "Finished setting up environment!" + ENDC
    env_connection.send([env_connection_message])

    while True:

        if env_connection.poll():

            env_connection_received = env_connection.recv()

            if env_connection_received[0] == CAPTURE_IMAGES:

                _, _ = robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_trajectory_path.format(step=0), depth_image_path=config.depth_image_trajectory_path.format(step=0))
                head_camera_position, head_camera_orientation_q = robot.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_head_path, depth_image_path=config.depth_image_head_path)
                wrist_camera_position, wrist_camera_orientation_q = robot.get_camera_image("wrist", env, save_camera_image=True, rgb_image_path=config.rgb_image_wrist_path, depth_image_path=config.depth_image_wrist_path)

                env_connection_message = OK + "Finished capturing head camera image!" + ENDC
                env_connection.send([head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message])

            elif env_connection_received[0] == ADD_BOUNDING_CUBES:

                bounding_cubes_world_coordinates = env_connection_received[1]

                for bounding_cube_world_coordinates in bounding_cubes_world_coordinates:
                    p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[1], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[2], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[3], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[0], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[5], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[6], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[7], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[8], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
                    p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
                    p.addUserDebugPoints(bounding_cube_world_coordinates, [[0, 1, 0]] * len(bounding_cube_world_coordinates), pointSize=5, lifeTime=0)

                env_connection_message = OK + "Finished adding bounding cubes to the environment!" + ENDC
                env_connection.send([env_connection_message])

            elif env_connection_received[0] == ADD_TRAJECTORY_POINTS:

                trajectory = env_connection_received[1]

                trajectory_points = [point[:3] for point in trajectory]
                p.addUserDebugPoints(trajectory_points, [[0, 1, 1]] * len(trajectory_points), pointSize=5, lifeTime=0)

                logger.info(OK + "Finished adding trajectory points to the environment!" + ENDC)

            elif env_connection_received[0] == EXECUTE_TRAJECTORY:

                trajectory = env_connection_received[1]

                for point in trajectory:
                    robot.move(env, point[:3], np.array(robot.ee_start_orientation_e) + np.array([0, 0, point[3]]), gripper_open=robot.gripper_open, is_trajectory=True)

                for _ in range(100):
                    env.update()

                logger.info(OK + "Finished executing generated trajectory!" + ENDC)

            elif env_connection_received[0] == SET_DOOR_STATE:

                # Payload: {"door_angle": float | None, "latch_angle": float | None}
                payload = env_connection_received[1] if len(env_connection_received) > 1 else {}
                door_angle = payload.get("door_angle") if isinstance(payload, dict) else None
                latch_angle = payload.get("latch_angle") if isinstance(payload, dict) else None
                try:
                    self.set_door_state(door_angle=door_angle, latch_angle=latch_angle)
                    # Run a few sim steps for visual settling
                    for _ in range(5):
                        self.update()
                    env_connection_message = OK + "Updated door state." + ENDC
                    env_connection.send([env_connection_message])
                except Exception as e:
                    env_connection_message = FAIL + f"Failed to update door state: {e}" + ENDC
                    env_connection.send([env_connection_message])

            elif env_connection_received[0] == CAPTURE_TRAJECTORY_FRAME:

                # Render and save a trajectory frame via robot API
                try:
                    step_idx = env_connection_received[1] if len(env_connection_received) > 1 else None
                    if step_idx is None:
                        step_idx = 0

                    robot.get_camera_image(
                        "head",
                        env,
                        save_camera_image=True,
                        rgb_image_path=config.rgb_image_trajectory_path.format(step=step_idx),
                        depth_image_path=config.depth_image_trajectory_path.format(step=step_idx),
                    )

                    env_connection_message = OK + f"Captured trajectory frame {step_idx}." + ENDC
                    env_connection.send([env_connection_message])
                except Exception as e:
                    env_connection_message = FAIL + f"Failed to capture trajectory frame: {e}" + ENDC
                    env_connection.send([env_connection_message])

            elif env_connection_received[0] == OPEN_GRIPPER:

                ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
                ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
                ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=True, is_trajectory=False)

                robot.gripper_open = True

                logger.info(OK + "Finished opening gripper!" + ENDC)

            elif env_connection_received[0] == CLOSE_GRIPPER:

                ee_current_position = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[0]
                ee_current_orientation_q = p.getLinkState(robot.id, robot.ee_index, computeForwardKinematics=True)[1]
                ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)

                robot.move(env, ee_current_position, ee_current_orientation_e, gripper_open=False, is_trajectory=False)

                robot.gripper_open = False

                logger.info(OK + "Finished closing gripper!" + ENDC)

            elif env_connection_received[0] == TASK_COMPLETED:

                env_connection_message = OK + "Finished executing all generated trajectories!" + ENDC
                env_connection.send([env_connection_message])

            elif env_connection_received[0] == RESET_ENVIRONMENT:

                robot.move(env, robot.ee_start_position, robot.ee_start_orientation_e, gripper_open=True, is_trajectory=False)
                robot.gripper_open = True
                robot.trajectory_step = 1

                for _ in range(100):
                    env.update()

                env_connection_message = OK + "Finished resetting environment!" + ENDC
                env_connection.send([env_connection_message])

        env.update()
# --- Minimal GUI demo to interactively test door kinematics ---
def run_gui_demo(disable_forces : bool = False):
    """
    Launch a minimal PyBullet GUI session that loads the environment with the door.
    - Uses p.GUI so the debug visualizer opens.
    - Disables door joint motor forces for easy mouse-pick/drag of the hinge/latch.
    - Enables real-time simulation and idles, letting you click and drag the door.
    """
    import logging
    logger = logging.getLogger("env_gui")
    logger.setLevel(logging.INFO)
    try:
        physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        _ = p.loadURDF("plane.urdf")

        class _Args:
            mode = "default"
            robot = "franka"

        env = Environment(_Args)
        env.load()

        if disable_forces:
            # Disable motors so user drag isn't resisted
            try:
                if getattr(env, "door_id", None) is not None:
                    if getattr(env, "door_hinge_index", None) is not None:
                        p.setJointMotorControl2(env.door_id, env.door_hinge_index, p.POSITION_CONTROL, force=0)
                    if getattr(env, "latch_index", None) is not None:
                        p.setJointMotorControl2(env.door_id, env.latch_index, p.POSITION_CONTROL, force=0)
            except Exception as e:
                print("[Env GUI] Failed to disable door motors:", e)
                traceback.print_exc()

        # Real-time simulation for natural interaction
        p.setRealTimeSimulation(1)
        print("[Env GUI] Running. Click-and-drag the door; press ESC to quit.")
        while p.isConnected():
            time.sleep(0.01)
    except Exception as e:
        print("[Env GUI] Exception:", e)
        traceback.print_exc()
    finally:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    # Entry point for quick, no-code door kinematics testing in GUI mode.
    run_gui_demo(disable_forces=False)
