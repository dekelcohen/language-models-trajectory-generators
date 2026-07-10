import pybullet as p
import numpy as np
import os
import math
import config
from config import OK, PROGRESS, FAIL, ENDC
from helpers.image_utils import draw_text_overlay_image
from PIL import Image
from config import fov, aspect, near_plane, far_plane

class Robot:

    def __init__(self, args, logger):
        self.logger = logger
        self.desc = ""
        if args.robot == "sawyer":
            self.base_start_position = config.base_start_position_sawyer
            self.base_start_orientation_q = p.getQuaternionFromEuler(config.base_start_orientation_e_sawyer)
            self.joint_start_positions = config.joint_start_positions_sawyer
            self.id = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
            self.robot = "sawyer"
            self.ee_index = config.ee_index_sawyer
            self.gripper_id = p.loadURDF("robotiq_2f_85/robotiq_2f_85.urdf", config.ee_start_position, p.getQuaternionFromEuler(config.ee_start_orientation_e))
            self.gripper_motor = config.robotiq_motor_joint
            p.createConstraint(self.id, self.ee_index, self.gripper_id, 0, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, -0.07], childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        elif args.robot == "franka":
            self.base_start_position = config.base_start_position_franka
            self.base_start_orientation_q = p.getQuaternionFromEuler(config.base_start_orientation_e_franka)
            self.joint_start_positions = config.joint_start_positions_franka
            self.id = p.loadURDF("franka_robot/panda.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
            self.gripper_id = self.id
            self.robot = "franka"
            self.ee_index = config.ee_index_franka
        self.ee_start_position = config.ee_start_position
        self.ee_start_orientation_e = config.ee_start_orientation_e
        self.ee_current_position = config.ee_start_position
        self.ee_current_orientation_e = config.ee_start_orientation_e

        self.gripper_open = True
        self.trajectory_step = 1

        i = 0
        self.joint_indices = []
        for j in range(p.getNumJoints(self.id)):
            joint_type = p.getJointInfo(self.id, j)[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                p.resetJointState(self.id, j, self.joint_start_positions[i])
                i += 1
                self.joint_indices.append(j)


    def get_joint_effort(self, joint_index, body_id=None):
        """
        Reads the 1D applied motor torque for a specific joint.
        Real: This simulates reading the 'Present Current' or 'Present Load'
        from a real Feetech STS3215 servo on the SO-101.
        """
        # Default to the main robot gripper id (could be the same as body id - franka) - if no gripper_id is provided
        if body_id is None:
            body_id = self.id
            
        # getJointState returns: (position, velocity, reactionForces, appliedJointMotorTorque)
        # Index 3 is the torque the motor is applying to hold its target
        state = p.getJointState(body_id, joint_index)
        motor_torque = state[3]
        
        return abs(motor_torque)
    
    def get_gripper_effort(self, gripper1_index, gripper2_index):
        """
        Returns the total squeezing effort (torque) of the gripper.
        """
        effort = self.get_joint_effort(gripper1_index, body_id=self.gripper_id)
        
        # If 2 gripper motors (franka) --> Sum the effort of both independent finger motors    
        if gripper1_index != gripper2_index:
            effort += self.get_joint_effort(gripper2_index, body_id=self.gripper_id)
            
        return effort
    
    def step_env_and_record(self, env, force_record=False):
        """
        steps (adv) the env and captures image - only if robot moved + 1 frame per settele (inertia). anyway no more that 5 fps.
        Pros:
        No wasted frames: If the robot is waiting or taking 100 idle steps to settle, it will not generate 5 identical frames. It will capture the initial move, the motion, and precisely 1 frame when motion stops.
        VLM Friendly: Because redundant frames are filtered out by the distance thresholds (0.01 m, 0.05 rad), the VLM gets a highly compressed, meaningful storyboard of the action (well under 50 frames per trajectory).
        Debug Friendly: Because it drops frames only when the robot isn't moving, the resulting video remains visually perfectly contiguous.
        Captures the Gripper: Gripper action (OPEN_GRIPPER) now natively generates frames because grip_changed > 0.005 triggers a keyframe recording.
        """
        env.update()
        
        # Initialize tracking variables on first run
        if not hasattr(self, 'sim_step_counter'):
            self.sim_step_counter = 0
            self.last_record_step = -9999
            self.last_record_eef_pos = None
            self.last_record_eef_ori = None
            self.last_record_gripper_pos = None

        self.sim_step_counter += 1

        # Rate limiter: max 5 FPS (48 steps at 240Hz). Ignore if we are forcing a keyframe.
        steps_per_frame = int(1.0 / (config.control_dt * 5.0))
        if not force_record and (self.sim_step_counter - self.last_record_step) < steps_per_frame:
            return

        # Get current state
        eef_pos, eef_ori_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[:2]
        eef_ori_e = p.getEulerFromQuaternion(eef_ori_q)
        
        # Get gripper state
        if self.robot == "sawyer":
            gripper_pos = p.getJointState(self.gripper_id, self.gripper_motor)[0]
        else: # franka
            gripper_pos = p.getJointState(self.id, 9)[0]

        # Check if we moved significantly
        moved_significantly = False
        if self.last_record_eef_pos is None:
            moved_significantly = True
        else:
            dist_moved = np.linalg.norm(np.array(eef_pos) - np.array(self.last_record_eef_pos))
            ori_changed = np.linalg.norm(np.array(eef_ori_e) - np.array(self.last_record_eef_ori))
            grip_changed = abs(gripper_pos - self.last_record_gripper_pos)

            # Thresholds: 1cm movement, ~3 degrees rotation, or small gripper movement
            if dist_moved > 0.01 or ori_changed > 0.05 or grip_changed > 0.005:
                moved_significantly = True

        # Save frame if motion detected or if it's a forced keyframe (settled)
        if force_record or moved_significantly:
            self.get_camera_image("head", env, save_camera_image=True, rgb_image_path=config.rgb_image_trajectory_path.format(step=self.trajectory_step), depth_image_path=None)
            self.get_camera_image("wrist", env, save_camera_image=True, rgb_image_path=config.wrist_rgb_image_trajectory_path.format(step=self.trajectory_step), depth_image_path=None)
            
            # Update memory
            self.last_record_eef_pos = eef_pos
            self.last_record_eef_ori = eef_ori_e
            self.last_record_gripper_pos = gripper_pos
            self.last_record_step = self.sim_step_counter
            
            self.trajectory_step += 1    
		
    def move(self, env, ee_target_position, ee_target_orientation_e, gripper_open, is_trajectory, desc=None):

        if desc is not None:
            self.desc = desc

        if self.robot == "sawyer":
            gripper1_index = self.gripper_motor
            gripper2_index = self.gripper_motor
            gripper_target_position = config.gripper_goal_position_open_sawyer if gripper_open else config.gripper_goal_position_closed_sawyer
            if is_trajectory:
                ee_target_position = list(ee_target_position)
                ee_target_position[2] -= config.gripper_depth_offset_sawyer
        elif self.robot == "franka":
            gripper1_index = 9
            gripper2_index = 10
            gripper_target_position = config.gripper_goal_position_open_franka if gripper_open else config.gripper_goal_position_closed_franka
            if is_trajectory:
                ee_target_position = list(ee_target_position)
                ee_target_position[2] -= config.gripper_depth_offset_franka

        min_joint_positions = [p.getJointInfo(self.id, i)[8] for i in range(p.getNumJoints(self.id)) if p.getJointInfo(self.id, i)[2] == p.JOINT_PRISMATIC or p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]
        max_joint_positions = [p.getJointInfo(self.id, i)[9] for i in range(p.getNumJoints(self.id)) if p.getJointInfo(self.id, i)[2] == p.JOINT_PRISMATIC or p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]
        joint_ranges = [abs(max_joint_position - min_joint_position) for min_joint_position, max_joint_position in zip(min_joint_positions, max_joint_positions)]
        rest_poses = list((np.array(max_joint_positions) + np.array(min_joint_positions)) / 2)

        ee_target_orientation_q = p.getQuaternionFromEuler(ee_target_orientation_e)

        ee_current_position = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
        ee_current_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
        ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)
        if self.robot == "sawyer":
            gripper1_current_position = p.getJointState(self.gripper_id, gripper1_index)[0]
            gripper2_current_position = p.getJointState(self.gripper_id, gripper2_index)[0]
        elif self.robot == "franka":
            gripper1_current_position = p.getJointState(self.id, gripper1_index)[0]
            gripper2_current_position = p.getJointState(self.id, gripper2_index)[0]

        time_step = 0

        while (not (ee_current_position[0] <= ee_target_position[0] + config.margin_error and ee_current_position[0] >= ee_target_position[0] - config.margin_error and
                    ee_current_position[1] <= ee_target_position[1] + config.margin_error and ee_current_position[1] >= ee_target_position[1] - config.margin_error and
                    ee_current_position[2] <= ee_target_position[2] + config.margin_error and ee_current_position[2] >= ee_target_position[2] - config.margin_error and
                    ee_current_orientation_e[0] <= ee_target_orientation_e[0] + config.margin_error and ee_current_orientation_e[0] >= ee_target_orientation_e[0] - config.margin_error and
                    ee_current_orientation_e[1] <= ee_target_orientation_e[1] + config.margin_error and ee_current_orientation_e[1] >= ee_target_orientation_e[1] - config.margin_error and
                    ee_current_orientation_e[2] <= ee_target_orientation_e[2] + config.margin_error and ee_current_orientation_e[2] >= ee_target_orientation_e[2] - config.margin_error and
                    gripper1_current_position <= gripper_target_position + config.gripper_margin_error and gripper1_current_position >= gripper_target_position - config.gripper_margin_error and
                    gripper2_current_position <= gripper_target_position + config.gripper_margin_error and gripper2_current_position >= gripper_target_position - config.gripper_margin_error)):

            target_joint_positions = p.calculateInverseKinematics(self.id, self.ee_index, ee_target_position, targetOrientation=ee_target_orientation_q, lowerLimits=min_joint_positions, upperLimits=max_joint_positions, jointRanges=joint_ranges, restPoses=rest_poses, maxNumIterations=500)

            if self.robot == "sawyer":
                p.setJointMotorControlArray(self.id, self.joint_indices, p.POSITION_CONTROL, targetPositions=target_joint_positions, forces=[config.arm_movement_force_sawyer] * 8)
                current_joints = [p.getJointState(self.gripper_id, i)[0] for i in range(p.getNumJoints(self.gripper_id))]
                joint_idx = [6, 3, 8, 5, 10]
                target_joints = [current_joints[1], -current_joints[1], -current_joints[1], current_joints[1], current_joints[1]]
                p.setJointMotorControlArray(self.gripper_id, joint_idx, p.POSITION_CONTROL, target_joints, positionGains=np.ones(5))
                p.setJointMotorControl2(self.gripper_id, self.gripper_motor, p.POSITION_CONTROL, targetPosition=gripper_target_position, force=config.gripper_movement_force_sawyer)
            elif self.robot == "franka":
                p.setJointMotorControlArray(self.id, self.joint_indices[:-2], p.POSITION_CONTROL, targetPositions=target_joint_positions[:-2], forces=[config.arm_movement_force_franka] * 7)
                p.setJointMotorControl2(self.id, gripper1_index, p.POSITION_CONTROL, targetPosition=gripper_target_position, force=config.gripper_movement_force_franka)
                p.setJointMotorControl2(self.id, gripper2_index, p.POSITION_CONTROL, targetPosition=gripper_target_position, force=config.gripper_movement_force_franka)

            # steps env and capture cameras images 
            self.step_env_and_record(env, force_record=False)
            
            # monitor torque of wrist joint motor (can replace with gripper1_index)
            if False: # not gripper_open and is_trajectory:
                actual_wrist_motor_index = self.joint_indices[-1]
                wrist_torque = self.get_joint_effort(actual_wrist_motor_index, body_id=self.id)
                gripper_torque = self.get_gripper_effort(gripper1_index, gripper2_index)
                self.logger.info(OK + f'wrist_torque: {wrist_torque} gripper_torque: {gripper_torque}' + ENDC)
            
            if is_trajectory:
                self.trajectory_step += 1

            ee_current_position = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0]
            ee_current_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
            ee_current_orientation_e = p.getEulerFromQuaternion(ee_current_orientation_q)
            if self.robot == "sawyer":
                gripper1_new_position = p.getJointState(self.gripper_id, gripper1_index)[0]
                gripper2_new_position = p.getJointState(self.gripper_id, gripper2_index)[0]
            elif self.robot == "franka":
                gripper1_new_position = p.getJointState(self.id, gripper1_index)[0]
                gripper2_new_position = p.getJointState(self.id, gripper2_index)[0]

            self.ee_current_position = ee_current_position
            self.ee_current_orientation_e = ee_current_orientation_e
            self.gripper_open = gripper_open

            if ((ee_current_position[0] <= ee_target_position[0] + config.margin_error and ee_current_position[0] >= ee_target_position[0] - config.margin_error and
                ee_current_position[1] <= ee_target_position[1] + config.margin_error and ee_current_position[1] >= ee_target_position[1] - config.margin_error and
                ee_current_position[2] <= ee_target_position[2] + config.margin_error and ee_current_position[2] >= ee_target_position[2] - config.margin_error and
                ee_current_orientation_e[0] <= ee_target_orientation_e[0] + config.margin_error and ee_current_orientation_e[0] >= ee_target_orientation_e[0] - config.margin_error and
                ee_current_orientation_e[1] <= ee_target_orientation_e[1] + config.margin_error and ee_current_orientation_e[1] >= ee_target_orientation_e[1] - config.margin_error and
                ee_current_orientation_e[2] <= ee_target_orientation_e[2] + config.margin_error and ee_current_orientation_e[2] >= ee_target_orientation_e[2] - config.margin_error) and
                (not gripper_open) and
                math.isclose(gripper1_new_position, gripper1_current_position, rel_tol=config.rel_tol, abs_tol=config.abs_tol) and
                math.isclose(gripper2_new_position, gripper2_current_position, rel_tol=config.rel_tol, abs_tol=config.abs_tol)):
                break

            gripper1_current_position = gripper1_new_position
            gripper2_current_position = gripper2_new_position

            time_step += 1

            if is_trajectory:
                if time_step > 0:
                    break
            else:
                if time_step > 99:
                    break
        # Guarantee a frame exactly at the end of the movement
        self.step_env_and_record(env, force_record=True)


    def get_camera_image(self, camera, env, save_camera_image, rgb_image_path, depth_image_path):
        """
        
        """
        if camera == "wrist":
            camera_position = list(p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[0])
            if self.robot == "sawyer":
                camera_position[2] -= config.wrist_camera_offset_sawyer
            camera_orientation_q = p.getLinkState(self.id, self.ee_index, computeForwardKinematics=True)[1]
        elif camera == "head":
            camera_position = config.head_camera_position
            camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
        # print(PROGRESS + f"get_camera_image projection_matrix.type: {type(projection_matrix)} projection_matrix {projection_matrix}"+ ENDC)
        # Special handling: head camera per-task behavior
        if camera == "head" and config.head_camera_use_debug_view:
            # GUI: copy view/projection from debug visualizer
            view_matrix, projection_matrix, camera_position = self._debug_view_matrices_and_pos()
        elif camera == "head" and config.head_camera_use_spherical_view:
            # DIRECT: build view from spherical params (identical to GUI angle)
            view_matrix, camera_position = self._view_and_pos_from_spherical(
                target=config.camera_target_position,
                distance=config.camera_distance,
                yaw_deg=config.camera_yaw,
                pitch_deg=config.camera_pitch,
            )
        else:
            rotation_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
            
            if camera == "wrist":
                init_camera_vector = np.array([0, 0, 1])
                camera_vector = rotation_matrix.dot(init_camera_vector)
                
                wrist_pos = np.array(camera_position)
                
                # --- "Drone" Over-the-Shoulder Tracking ---
                # 1. Pull straight back along the line of sight
                pullback_pos = wrist_pos - (config.wrist_camera_pullback * camera_vector)
                
                # 2. Vertical offset (negative lowers the pose to see the hinged handle)
                global_up_shift = np.array([0.0, 0.0, config.wrist_camera_up_shift])
                
                # 3. Shift sideways (right->left view) to peek around the elbow/forearm
                # We calculate global 'right' by taking the cross product of where we are looking and the ceiling
                right_vector = np.cross(camera_vector, np.array([0, 0, 1]))
                if np.linalg.norm(right_vector) > 1e-3:
                    right_vector = right_vector / np.linalg.norm(right_vector)
                else:
                    right_vector = np.array([1, 0, 0]) # Fallback if looking straight down
                
                lateral_shift = config.wrist_camera_lateral_shift * right_vector
                
                # Apply all shifts to get the final camera position
                camera_position = pullback_pos + global_up_shift + lateral_shift
                
                # Force the camera to look at the gripper (or slightly ahead of it)
                target_position = wrist_pos + (0.05 * camera_vector)
                
                # Force the camera's image to stay perfectly level with the room
                up_vector =[0, 0, 1]
                
            elif camera == "head":
                init_camera_vector =[0, 0, 1]
                init_up_vector = [-1, 0, 0]
                
                camera_vector = rotation_matrix.dot(init_camera_vector)
                up_vector = rotation_matrix.dot(init_up_vector)
                target_position = camera_position + camera_vector
                
            # Compute the view matrix directly
            view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position, 
                                              cameraTargetPosition=target_position, 
                                              cameraUpVector=up_vector)


        image = p.getCameraImage(
            config.image_width,
            config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        img_w, img_h = image[0], image[1]
        rgb_buffer = image[2]
        depth_buffer = image[3]

        # Ensure numpy arrays with correct shape
          # Q: How come is was working in main branch (azure linux) ? reviewed metaworld branch new changes and none explains it. A: Not sure 
          #- PyBulletâ€™s getCameraImage returns a tuple where the color buffer is a flat sequence (or a buffer with alpha) rather than a ready-to-use NumPy array.
          #- robot.get_camera_image passed that tuple element directly into PIL: Image.fromarray(rgb_buffer). Since it wasnâ€™t a NumPy array (nor an array-like with a valid array_interface), PIL raised AttributeError: 'tuple' object has no attribute 'array_interface'.
                  
        try:
            rgb_array = np.array(rgb_buffer, dtype=np.uint8).reshape(img_h, img_w, 4)
        except Exception:
            # Fallback if already shaped but not ndarray
            rgb_array = np.asarray(rgb_buffer, dtype=np.uint8)
            if rgb_array.ndim == 1:
                rgb_array = rgb_array.reshape(img_h, img_w, 4)

        # Explanation: PyBullet may return a flattened tuple/list (W*H*4 RGBA) or an already-shaped array.
        # The conversion below reshapes to (H, W, 4) and drops alpha so PIL consistently gets (H, W, 3) uint8.
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(depth_buffer, dtype=np.float32).reshape(img_h, img_w)

        LEGACY_NORMALIZE_DEPTH = False # TODO:False when moving to test like projection matrix     
        
        
        if save_camera_image:
            rgb_image = Image.fromarray(rgb_array, mode="RGB")
            # Render desc text at the bottom
            if self.desc:
                draw_text_overlay_image(self.desc, rgb_image)                
            rgb_image.save(rgb_image_path)
            if depth_image_path:           
                n = config.near_plane
                f = config.far_plane
                # Convert OpenGL depth to linear depth in [0,1]
                z = depth_array
                linear_depth = (2.0 * n * f) / (f + n - (2.0 * z - 1.0) * (f - n))
                linear_depth = np.clip(linear_depth, 0.0, 1.0)
                depth_u8 = (linear_depth * 255.0).astype(np.uint8)
                depth_image = Image.fromarray(depth_u8, mode="L")
                    
                depth_image.save(depth_image_path)
                np.save(os.path.splitext(depth_image_path)[0] + ".npy", depth_array.astype(np.float32))

        return camera_position, camera_orientation_q, view_matrix, projection_matrix

    def _debug_view_matrices_and_pos(self):
        """Return (view, projection, camera_pos) mirroring the GUI debug view.
        Falls back to spherical params from config if the debug camera tuple
        is unavailable (e.g., in DIRECT mode).
        """
        try:
            dbg = p.getDebugVisualizerCamera()
            # Validate tuple structure
            if not isinstance(dbg, (list, tuple)) or len(dbg) != 12:
                raise RuntimeError("debug camera unavailable")

            view_matrix = dbg[2]
            projection_matrix = dbg[3]
            yaw_deg = float(dbg[8])
            pitch_deg = float(dbg[9])
            dist = float(dbg[10])
            target = dbg[11] if isinstance(dbg[11], (list, tuple)) else [0.0, 0.0, 0.0]
            tx, ty, tz = list(map(float, target))

            # Treat zero/degenerate values as invalid in DIRECT mode
            if dist <= 1e-6 or (abs(yaw_deg) <= 1e-6 and abs(pitch_deg) <= 1e-6 and abs(tx) <= 1e-6 and abs(ty) <= 1e-6 and abs(tz) <= 1e-6):
                raise RuntimeError("debug camera invalid (degenerate)")

            yaw = np.deg2rad(yaw_deg)
            pitch = np.deg2rad(pitch_deg)
            fx = np.cos(pitch) * np.cos(yaw)
            fy = np.cos(pitch) * np.sin(yaw)
            fz = np.sin(pitch)
            camera_pos = [tx - dist * fx, ty - dist * fy, tz - dist * fz]
            return view_matrix, projection_matrix, camera_pos
        except Exception:
            # Fallback: compute view + position from spherical params
            view_matrix, camera_pos = self._view_and_pos_from_spherical(
                target=config.camera_target_position,
                distance=config.camera_distance,
                yaw_deg=config.camera_yaw,
                pitch_deg=config.camera_pitch,
            )
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
            return view_matrix, projection_matrix, camera_pos

    def _view_and_pos_from_spherical(self, target, distance, yaw_deg, pitch_deg):
        """Compute a view matrix and world camera position from spherical
        parameters (target, distance, yaw, pitch) with Z-up. Keeps logic
        small and reusable for head-camera debug mirroring fallback.
        """
        # Use positional args; some PyBullet builds require 'upAxisIndex' p ositional
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            target,
            distance,
            yaw_deg,
            pitch_deg,
            0.0,
            2,
        )
        yaw = np.deg2rad(float(yaw_deg))
        pitch = np.deg2rad(float(pitch_deg))
        tx, ty, tz = list(map(float, target))
        fx = np.cos(pitch) * np.cos(yaw)
        fy = np.cos(pitch) * np.sin(yaw)
        fz = np.sin(pitch)
        camera_position = [
            tx - distance * fx,
            ty - distance * fy,
            tz - distance * fz,
        ]
        return view_matrix, camera_position



