"""Door sim-env: face robot toward the Adroit door, set cameras, load the door."""

import traceback

import numpy as np
import pybullet as p

import config
from sim_envs.pybullet.base import SimEnvBase
from sim_envs.pybullet.pb_utils import get_joint_index_by_name, get_link_index_by_name, spherical_camera_pose


class SimEnvDoor(SimEnvBase):
    """Door task: face robot toward door, set cameras, and load the door.

    Head camera pose is set to match the GUI debug visualizer camera.
    URDF loading and controller force setup occur here.
    """

    def __init__(self):
        # Debug visualizer camera (GUI) used in run_gui_demo
        config.camera_distance = 1.0
        config.camera_yaw = 190.0
        config.camera_pitch = -40.0
        config.camera_target_position = [0.0, 0.64, 0.70]

        # Door members initialized for later direct access in get_state
        self.door_id = None
        self.door_hinge_index = None
        self.latch_index = None
        self.door_handle_latch = None
        self.board_id = None
        self.pole_id = None

        # Compute head camera pose identical to GUI spherical camera
        # and use it statically in DIRECT (no dynamic debug mirroring).
        pos, ori_e = self._head_from_debug(
            config.camera_distance,
            config.camera_yaw,
            config.camera_pitch,
            config.camera_target_position,
        )
        config.head_camera_position = pos
        config.head_camera_orientation_e = ori_e
        # Decide camera behavior by connection type
        # In GUI: mirror the debug visualizer; in DIRECT: use spherical view
        try:
            _is_gui = p.isConnected() and p.getConnectionInfo()[1] == p.GUI
        except Exception:
            _is_gui = False
        config.head_camera_use_debug_view = bool(_is_gui)
        config.head_camera_use_spherical_view = not bool(_is_gui)

        # Do NOT change robot base orientation/pose here; we want grasp defaults
        # so the robot stands upright and stable (not looking at the door).

    def _head_from_debug(self, distance, yaw_deg, pitch_deg, target):
        # Keep this helper small and correct; used only to print params.
        return spherical_camera_pose(target, distance, yaw_deg, pitch_deg)

    def apply(self, env):
        # Camera and door assets handled here; robot pose remains grasp default
        return

    def load_assets(self, env):
        # Load Adroit door and set strong hold forces
        try:
            door_start_position = [-0.11, 0.04, 0.25]
            door_start_orientation_q = p.getQuaternionFromEuler([0.0, 0.0, 4.0])
            self.door_id = p.loadURDF(
                "my_assets/adroit_door/adroit_door.urdf",
                door_start_position,
                door_start_orientation_q,
                useFixedBase=True,
            )
            # Cosmetics:
            # 2. Load the image texture into PyBullet
            wood_texture_id = p.loadTexture("my_assets/adroit_door/wood.png")

            # 3. Apply the texture to the Frame (Base Link: index -1)
            p.changeVisualShape(self.door_id, 0, textureUniqueId=wood_texture_id)

            # 4. Apply the texture to the Door Panel (Link: index 0)
            p.changeVisualShape(self.door_id, 1, textureUniqueId=wood_texture_id)

            # Resolve indices based on the newly loaded door
            self.door_hinge_index = get_joint_index_by_name(self.door_id, "door_hinge")
            self.latch_index = get_joint_index_by_name(self.door_id, "latch_joint")
            # Door handle is the child link named 'latch' in URDF; look it up by link name
            self.door_handle_latch = get_link_index_by_name(self.door_id, "latch")

            if self.door_hinge_index is not None:
                # p.setJointMotorControl2(self.door_id, self.door_hinge_index, p.POSITION_CONTROL, targetPosition=0.0, force=200)
                # Instead of rigidly holding it at 0.0 with 200 force, give it a resting friction
                p.setJointMotorControl2(
                    self.door_id,
                    self.door_hinge_index,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=2.0  # Just enough force to keep it from swinging on its own, but weak enough for the robot to pull
                )
                # Increase friction on the handle (latch link)
                p.changeDynamics(self.door_id, self.door_handle_latch, lateralFriction=2.0, spinningFriction=1.0)

            if self.latch_index is not None:
                p.setJointMotorControl2(self.door_id, self.latch_index, p.POSITION_CONTROL, targetPosition=0.0, force=200)

            HIDE_DOOR_WITH_OBJECT = True
            if HIDE_DOOR_WITH_OBJECT:
                self._load_pole()
                #self._load_board()
        except Exception as e:
            print("[Env] Failed to load or initialize adroit_door URDF:", e)
            traceback.print_exc()

    def _load_pole(self):
        """Add a vertical pole standing between the robot arm and the door.

        The pole has mass and a collision shape, so the robot arm can push it
        over and make it fall to the ground.
        """
        # Robot base ~[-0.3, 0.5, 0.0], door ~[-0.11, 0.04, 0.25]; place pole between
        # them WITHOUT overlapping the door. Overlapping the door at spawn causes a
        # large contact/penetration force that ejects the pole and topples it at
        # sim start. This clear position spawns upright and stable, yet the robot
        # arm can still push it over later.
        pole_height = 1.0
        pole_radius = 0.15
        pole_position = [-0.15, 0.30, pole_height / 2.0]
        pole_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=pole_radius, height=pole_height
        )
        pole_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=pole_radius,
            length=pole_height,
            rgbaColor=[0.5, 0.5, 0.55, 1.0],
        )
        self.pole_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=pole_collision,
            baseVisualShapeIndex=pole_visual,
            basePosition=pole_position,
        )
        p.changeDynamics(self.pole_id, -1, lateralFriction=1.0, spinningFriction=0.5)
        return self.pole_id

    def _load_board(self):
        """Add a vertical board leaning against the door.

        The board has mass and a collision shape, so the robot arm can push it
        over. It starts tilted toward the door so it comes to rest leaning on
        the door panel (it may settle/fall onto the door when the sim starts).
        """
        # Robot base ~[-0.3, 0.5, 0.0], door ~[-0.11, 0.04, 0.25]; lean board on the door.
        board_height = 0.6
        board_width = 0.6   # 10x the previous pole width (0.06)
        board_depth = 0.06
        half_extents = [board_width / 2.0, board_depth / 2.0, board_height / 2.0]

        # Tilt the board toward the door (pitch about y-axis) so it leans instead
        # of standing upright (a thin tall board is unstable and would topple).
        board_orientation_q = p.getQuaternionFromEuler([0.1422, 0.0000, 0.1975])
        # Place the base near the door so the tilted top rests against the panel.
        board_position = [-0.2429, 0.2063, 0.4992]

        board_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        board_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.5, 0.5, 0.55, 1.0],
        )
        self.board_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=board_collision,
            baseVisualShapeIndex=board_visual,
            basePosition=board_position,
            baseOrientation=board_orientation_q,
        )
        p.changeDynamics(self.board_id, -1, lateralFriction=1.0, spinningFriction=0.5)
        return self.board_id

    def get_state(self):
        """Return door-related indices and world positions for diagnostics.
        """
        state = {
            "door_id": self.door_id,
            "door_hinge_index": self.door_hinge_index,
            "latch_index": self.latch_index,
            "door_handle_latch": self.door_handle_latch,
            "door_handle_pos": None,
            "latch_pos": None,
            "hinge_pos": None,
            "pole_id": self.pole_id,
            "pole_pos": None,
            "pole_dims": None,
        }
        try:
            if self.door_id is not None:
                if self.door_handle_latch is not None and self.door_handle_latch >= 0:
                    _dhl = p.getLinkState(self.door_id, int(self.door_handle_latch), computeForwardKinematics=True)
                    state["door_handle_pos"] = list(map(float, _dhl[0]))
                if self.latch_index is not None and self.latch_index >= 0:
                    _lat = p.getLinkState(self.door_id, int(self.latch_index), computeForwardKinematics=True)
                    state["latch_pos"] = list(map(float, _lat[0]))
                if self.door_hinge_index is not None and self.door_hinge_index >= 0:
                    _hinge = p.getLinkState(self.door_id, int(self.door_hinge_index), computeForwardKinematics=True)
                    state["hinge_pos"] = list(map(float, _hinge[0]))

        except Exception as e:
            print("[Env] Warning: SimEnvDoor.get_state failed to read positions:", e)
            traceback.print_exc()
        try:
            if self.pole_id is not None:
                pos, _ori = p.getBasePositionAndOrientation(self.pole_id)
                aabb_min, aabb_max = p.getAABB(self.pole_id, -1)
                state["pole_pos"] = list(map(float, pos))
                state["pole_dims"] = [
                    float(aabb_max[0] - aabb_min[0]),
                    float(aabb_max[1] - aabb_min[1]),
                    float(aabb_max[2] - aabb_min[2]),
                ]
        except Exception as e:
            print("[Env] Warning: SimEnvDoor.get_state failed to read pole:", e)
            traceback.print_exc()
        return state

    def configure_robot_pose(self):
        """
        Override robot pose for door task to be identical to grasp defaults.
        This keeps the robot upright and stable (not facing the door).
        """
        config.base_start_position_franka = [-0.3, 0.5, 0.0]
        # Euler angles [roll, pitch, yaw]; yaw = -pi/2 faces the door
        config.base_start_orientation_e_franka = [0.0, 0.0, -np.pi / 3]
        # Problem: robot eef collapses into the door and hides the handled
        # Solution: Make the robot 'lean back' by rotating the joint above the base (idx 1)
        config.joint_start_positions_franka[1] = -1.5

    def move_to_start_pos(self):
        """
        Return False, since if moved - it collides with the door and collapses
        """
        return False

    def get_3d_coordinates_prompt_section(self):
        return (
            "The 3D coordinate system of the environment is as follows:\n"
            "  1. The x-axis is in the horizontal direction, increasing to the left.\n"
            "  2. The y-axis is in the depth direction, decreasing away from you.\n"
            "  3. The z-axis is in the vertical direction, increasing upwards."
        )
