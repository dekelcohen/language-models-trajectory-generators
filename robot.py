import numpy as np
import os
import math
import config
from config import OK, PROGRESS, FAIL, ENDC
from debug import trace_utils
from helpers.image_utils import draw_text_overlay_image
from robot_profiles import get_robot_profile
from sim_adapter import camera_math
from sim_adapter.camera_math import spherical_camera_pose
from PIL import Image
from config import fov, aspect, near_plane, far_plane

class Robot:

    def __init__(self, args, logger, sim):
        self.logger = logger
        self.sim = sim
        self.desc = ""
        self.profile = get_robot_profile(args.robot)
        if args.robot == "sawyer":
            self.base_start_position = config.base_start_position_sawyer
            self.base_start_orientation_q = self.sim.quat_from_euler(config.base_start_orientation_e_sawyer)
            self.joint_start_positions = config.joint_start_positions_sawyer
            self.id = self.sim.load_urdf(self.profile.urdf, self.base_start_position, self.base_start_orientation_q, fixed_base=True)
            self.robot = "sawyer"
            self.ee_index = self.profile.ee_index
            self.gripper_id = self.sim.load_urdf("robotiq_2f_85/robotiq_2f_85.urdf", config.ee_start_position, self.sim.quat_from_euler(config.ee_start_orientation_e))
            self.gripper_motor = config.robotiq_motor_joint
            self.sim.create_fixed_constraint(self.id, self.ee_index, self.gripper_id, 0, parent_frame_position=[0, 0, 0], child_frame_position=[0, 0, -0.07], child_frame_orientation_q=self.sim.quat_from_euler([0, 0, 0]))
        elif args.robot == "franka":
            self.base_start_position = config.base_start_position_franka
            self.base_start_orientation_q = self.sim.quat_from_euler(config.base_start_orientation_e_franka)
            self.joint_start_positions = config.joint_start_positions_franka
            self.id = self.sim.load_urdf(self.profile.urdf, self.base_start_position, self.base_start_orientation_q, fixed_base=True)
            self.gripper_id = self.id
            self.robot = "franka"
            self.ee_index = self.profile.ee_index
        self.ee_start_position = config.ee_start_position
        self.ee_start_orientation_e = config.ee_start_orientation_e
        self.ee_current_position = config.ee_start_position
        self.ee_current_orientation_e = config.ee_start_orientation_e

        self.gripper_open = True
        self.trajectory_step = 1

        i = 0
        self.joint_indices = []
        for joint in self.sim.get_movable_joints(self.id):
            self.sim.reset_joint_state(self.id, joint.index, self.joint_start_positions[i])
            i += 1
            self.joint_indices.append(joint.index)


    def get_joint_effort(self, joint_index, body_id=None):
        """
        Reads the 1D applied motor torque for a specific joint.
        Real: This simulates reading the 'Present Current' or 'Present Load'
        from a real Feetech STS3215 servo on the SO-101.
        """
        # Default to the main robot gripper id (could be the same as body id - franka) - if no gripper_id is provided
        if body_id is None:
            body_id = self.id

        # The torque the motor is applying to hold its target.
        motor_torque = self.sim.get_joint_state(body_id, joint_index).applied_torque

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
    
    @trace_utils.traced("Robot.step_env_and_record", args=False, result=False)
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
        eef_pos, eef_ori_q = self.sim.get_link_pose(self.id, self.ee_index)
        eef_ori_e = self.sim.euler_from_quat(eef_ori_q)

        # Get gripper state
        if self.robot == "sawyer":
            gripper_pos = self.sim.get_joint_state(self.gripper_id, self.gripper_motor).position
        else: # franka
            gripper_pos = self.sim.get_joint_state(self.id, self.profile.gripper_state_joint).position

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
		
    @trace_utils.traced("Robot.move")
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
            gripper1_index, gripper2_index = self.profile.gripper_joint_indices
            gripper_target_position = config.gripper_goal_position_open_franka if gripper_open else config.gripper_goal_position_closed_franka
            if is_trajectory:
                ee_target_position = list(ee_target_position)
                ee_target_position[2] -= config.gripper_depth_offset_franka

        movable_joints = self.sim.get_movable_joints(self.id)
        min_joint_positions = [j.lower_limit for j in movable_joints]
        max_joint_positions = [j.upper_limit for j in movable_joints]
        joint_ranges = [abs(max_joint_position - min_joint_position) for min_joint_position, max_joint_position in zip(min_joint_positions, max_joint_positions)]
        rest_poses = list((np.array(max_joint_positions) + np.array(min_joint_positions)) / 2)

        ee_target_orientation_q = self.sim.quat_from_euler(ee_target_orientation_e)

        ee_current_position, ee_current_orientation_q = self.sim.get_link_pose(self.id, self.ee_index)
        ee_current_orientation_e = self.sim.euler_from_quat(ee_current_orientation_q)
        if self.robot == "sawyer":
            gripper1_current_position = self.sim.get_joint_state(self.gripper_id, gripper1_index).position
            gripper2_current_position = self.sim.get_joint_state(self.gripper_id, gripper2_index).position
        elif self.robot == "franka":
            gripper1_current_position = self.sim.get_joint_state(self.id, gripper1_index).position
            gripper2_current_position = self.sim.get_joint_state(self.id, gripper2_index).position

        time_step = 0

        while (not (ee_current_position[0] <= ee_target_position[0] + config.margin_error and ee_current_position[0] >= ee_target_position[0] - config.margin_error and
                    ee_current_position[1] <= ee_target_position[1] + config.margin_error and ee_current_position[1] >= ee_target_position[1] - config.margin_error and
                    ee_current_position[2] <= ee_target_position[2] + config.margin_error and ee_current_position[2] >= ee_target_position[2] - config.margin_error and
                    ee_current_orientation_e[0] <= ee_target_orientation_e[0] + config.margin_error and ee_current_orientation_e[0] >= ee_target_orientation_e[0] - config.margin_error and
                    ee_current_orientation_e[1] <= ee_target_orientation_e[1] + config.margin_error and ee_current_orientation_e[1] >= ee_target_orientation_e[1] - config.margin_error and
                    ee_current_orientation_e[2] <= ee_target_orientation_e[2] + config.margin_error and ee_current_orientation_e[2] >= ee_target_orientation_e[2] - config.margin_error and
                    gripper1_current_position <= gripper_target_position + config.gripper_margin_error and gripper1_current_position >= gripper_target_position - config.gripper_margin_error and
                    gripper2_current_position <= gripper_target_position + config.gripper_margin_error and gripper2_current_position >= gripper_target_position - config.gripper_margin_error)):

            target_joint_positions = self.sim.inverse_kinematics(self.id, self.ee_index, ee_target_position, orientation_q=ee_target_orientation_q, lower_limits=min_joint_positions, upper_limits=max_joint_positions, joint_ranges=joint_ranges, rest_poses=rest_poses, max_iterations=500)

            if self.robot == "sawyer":
                self.sim.set_joint_positions(self.id, self.joint_indices, target_joint_positions, forces=[config.arm_movement_force_sawyer] * len(self.joint_indices))
                current_joints = [self.sim.get_joint_state(self.gripper_id, i).position for i in range(self.sim.num_joints(self.gripper_id))]
                joint_idx = [6, 3, 8, 5, 10]
                target_joints = [current_joints[1], -current_joints[1], -current_joints[1], current_joints[1], current_joints[1]]
                self.sim.set_joint_positions(self.gripper_id, joint_idx, target_joints, position_gains=np.ones(5))
                self.sim.set_joint_position(self.gripper_id, self.gripper_motor, gripper_target_position, force=config.gripper_movement_force_sawyer)
            elif self.robot == "franka":
                arm_n = self.profile.arm_joint_count
                self.sim.set_joint_positions(self.id, self.joint_indices[:arm_n], target_joint_positions[:arm_n], forces=[config.arm_movement_force_franka] * arm_n)
                self.sim.set_joint_position(self.id, gripper1_index, gripper_target_position, force=config.gripper_movement_force_franka)
                self.sim.set_joint_position(self.id, gripper2_index, gripper_target_position, force=config.gripper_movement_force_franka)

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

            ee_current_position, ee_current_orientation_q = self.sim.get_link_pose(self.id, self.ee_index)
            ee_current_orientation_e = self.sim.euler_from_quat(ee_current_orientation_q)
            if self.robot == "sawyer":
                gripper1_new_position = self.sim.get_joint_state(self.gripper_id, gripper1_index).position
                gripper2_new_position = self.sim.get_joint_state(self.gripper_id, gripper2_index).position
            elif self.robot == "franka":
                gripper1_new_position = self.sim.get_joint_state(self.id, gripper1_index).position
                gripper2_new_position = self.sim.get_joint_state(self.id, gripper2_index).position

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


    @trace_utils.traced("Robot.get_camera_image")
    def get_camera_image(self, camera, env, save_camera_image, rgb_image_path, depth_image_path):
        """
        
        """
        if camera == "wrist":
            _wrist_pos, camera_orientation_q = self.sim.get_link_pose(self.id, self.ee_index)
            camera_position = list(_wrist_pos)
            if self.robot == "sawyer":
                camera_position[2] -= config.wrist_camera_offset_sawyer
        elif camera == "head":
            camera_position = config.head_camera_position
            camera_orientation_q = self.sim.quat_from_euler(config.head_camera_orientation_e)

        projection_matrix = self.sim.compute_projection_matrix(fov, aspect, near_plane, far_plane)
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
            rotation_matrix = np.array(self.sim.matrix_from_quat(camera_orientation_q)).reshape(3, 3)
            
            if camera == "wrist":
                init_camera_vector = np.array([0, 0, 1])
                camera_vector = rotation_matrix.dot(init_camera_vector)
                
                wrist_pos = np.array(camera_position)

                # Framing offsets come from the active sim-env so each scene can
                # tune them; config values are the fallback default.
                wrist_params = {
                    "pullback": config.wrist_camera_pullback,
                    "up_shift": config.wrist_camera_up_shift,
                    "lateral_shift": config.wrist_camera_lateral_shift,
                }
                try:
                    wrist_params.update(env.simenv.get_wrist_camera_params() or {})
                except Exception:
                    pass

                # --- "Drone" Over-the-Shoulder Tracking ---
                # 1. Pull straight back along the line of sight
                pullback_pos = wrist_pos - (wrist_params["pullback"] * camera_vector)
                
                # 2. Vertical offset (negative lowers the pose to see the hinged handle)
                global_up_shift = np.array([0.0, 0.0, wrist_params["up_shift"]])
                
                # 3. Shift sideways (right->left view) to peek around the elbow/forearm
                # We calculate global 'right' by taking the cross product of where we are looking and the ceiling
                right_vector = np.cross(camera_vector, np.array([0, 0, 1]))
                if np.linalg.norm(right_vector) > 1e-3:
                    right_vector = right_vector / np.linalg.norm(right_vector)
                else:
                    right_vector = np.array([1, 0, 0]) # Fallback if looking straight down
                
                lateral_shift = wrist_params["lateral_shift"] * right_vector
                
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
            view_matrix = self.sim.compute_view_matrix(eye=camera_position,
                                                       target=target_position,
                                                       up=up_vector)


        frame = self.sim.render_camera(
            config.image_width,
            config.image_height,
            view_matrix,
            projection_matrix,
        )

        img_w, img_h = frame.width, frame.height
        rgb_array = frame.rgb
        depth_array = frame.depth

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
                # Convert the simulator's raw depth to metric depth, then to [0,1] for the
                # 8-bit preview. Genesis already reports metres, PyBullet a GL z-buffer;
                # camera_math hides that difference.
                linear_depth = camera_math.depth_to_metric(depth_array, self.sim.depth_encoding, n, f)
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
            dbg = self.sim.get_viewer_camera()
            # Validate structure
            if not dbg or not dbg.get("available"):
                raise RuntimeError("debug camera unavailable")

            view_matrix = dbg["view_matrix"]
            projection_matrix = dbg["projection_matrix"]
            yaw_deg = float(dbg["yaw"])
            pitch_deg = float(dbg["pitch"])
            dist = float(dbg["distance"])
            target = dbg["target"] if isinstance(dbg["target"], (list, tuple)) else [0.0, 0.0, 0.0]
            tx, ty, tz = list(map(float, target))

            # Treat zero/degenerate values as invalid in headless mode
            if dist <= 1e-6 or (abs(yaw_deg) <= 1e-6 and abs(pitch_deg) <= 1e-6 and abs(tx) <= 1e-6 and abs(ty) <= 1e-6 and abs(tz) <= 1e-6):
                raise RuntimeError("debug camera invalid (degenerate)")

            camera_pos, _ = spherical_camera_pose([tx, ty, tz], dist, yaw_deg, pitch_deg)
            return view_matrix, projection_matrix, camera_pos
        except Exception:
            # Fallback: compute view + position from spherical params
            view_matrix, camera_pos = self._view_and_pos_from_spherical(
                target=config.camera_target_position,
                distance=config.camera_distance,
                yaw_deg=config.camera_yaw,
                pitch_deg=config.camera_pitch,
            )
            projection_matrix = self.sim.compute_projection_matrix(fov, aspect, near_plane, far_plane)
            return view_matrix, projection_matrix, camera_pos

    def _view_and_pos_from_spherical(self, target, distance, yaw_deg, pitch_deg):
        """Compute a view matrix and world camera position from spherical
        parameters (target, distance, yaw, pitch) with Z-up. Keeps logic
        small and reusable for head-camera debug mirroring fallback.
        """
        view_matrix = self.sim.compute_view_matrix_from_yaw_pitch_roll(
            target,
            distance,
            yaw_deg,
            pitch_deg,
            0.0,
            2,
        )
        camera_position, _ = spherical_camera_pose(target, distance, yaw_deg, pitch_deg)
        return view_matrix, camera_position



