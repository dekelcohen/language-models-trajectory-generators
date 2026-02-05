import numpy as np
import sys
import torch
import math
import os
import config
import json
import models
from segmentation_adapter import get_segmentation_output
import utils
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT

class API:

    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model, device):

        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        utils.logger = self.logger # injects logger into utils global scope 
        self.client = client
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device
        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.attempted_task = False
        self.completed_task = False
        self.failed_task = False
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.command = None
        # Tracking provider selection ("xmem" or "none")
        self.track_provider = getattr(args, "track_provider", "xmem")



    def detect_object(self, segmentation_text):
        """
        segment in 2D --> transform to 3D world coordinates in Sim
        Workflow:
        ) Send to env CAPTURE_IMAGES message
        ) env.py: robot.get_camera_image("head") 
        ) def robot.py: get_camera_image
            camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)
            if camera == "head" and config.head_camera_use_spherical_view: # Door task head camera
              view_matrix, camera_position = self._view_and_pos_from_spherical(target pos, distance, yaw_deg, pitch_deg)
              def _view_and_pos_from_spherical:
                view_matrix = p.computeViewMatrixFromYawPitchRoll( ... )
                  camera_position = <complex calc with target, distance, yaw_deg, pitch_deg>
                  return view_matrix, camera_position
            return camera_position, camera_orientation_q 
            
          ) env::CAPTURE_IMAGES respond with head_camera_position=camera_position, head_camera_orientation_q=camera_orientation_q
          ) utils.get_bounding_cube_from_point_cloud(head_camera_position, head_camera_orientation_q, K_override=None -if pybullet)
            ) contour_pixel_points = <countour of segmented object in 2d image>
            ) get_world_point_world_frame(camera_position, camera_orientation_q, 'head', pixel_point for contour_pixel_points)
                K, Rt = get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q, K_override=K_use)
                elif camera == "head":
                  pixel_point = [-pixel_point[1], -pixel_point[0], pixel_point[2]]
                world_point_camera_frame = (np.linalg.inv(K) @ pixel_point) * point[2]
                world_point_world_frame = Rt @ np.vstack((world_point_camera_frame, np.array([1.0])))
            
        )   
        """
        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        recv_payload = self.main_connection.recv()
        # Robustly parse payload from either PyBullet Pipe or Metaworld WS
        head_camera_position = head_camera_orientation_q = None
        wrist_camera_position = wrist_camera_orientation_q = None
        env_connection_message = None
        self.calibration = None
        if isinstance(recv_payload, list):
            if len(recv_payload) >= 6:
                head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message = recv_payload[:5]
                self.calibration = recv_payload[5]
            elif len(recv_payload) == 5:
                head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message = recv_payload
            elif len(recv_payload) == 1:
                # Only a status line; proceed with saved images but no poses/calibration
                env_connection_message = recv_payload[0]
            else:
                raise ValueError(f"Unexpected CAPTURE_IMAGES payload length: {len(recv_payload)}")
        else:
            # Single non-list message; treat as status string
            env_connection_message = recv_payload
        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        depth_image_head = Image.open(config.depth_image_head_path).convert("L")
        # Handle depth according to depth_format and backend calibration
        depth_format = getattr(self.args, "depth_format", "norm_1m")
        depth_array = np.array(depth_image_head) / 255.
        if getattr(self, "calibration", None) and isinstance(self.calibration, dict):
            head_cal = self.calibration.get("head")
            if head_cal and self.calibration.get("depth_encoding") == "opengl":
                znear = float(head_cal.get("znear", 0.01))
                zfar = float(head_cal.get("zfar", 100.0))
                d = depth_array.astype(np.float64)
                Z = (znear * zfar) / (zfar - d * (zfar - znear))
                if depth_format == "raw":
                    depth_array = np.clip(Z, znear, zfar)
                elif depth_format == "norm_zfar":
                    depth_array = np.clip(Z / zfar, 0.0, 1.0)
                else:  # norm_1m
                    depth_array = np.clip(Z, 0.0, 1.0)

        if self.segmentation_count == 0:
            xmem_image = Image.fromarray(np.zeros_like(depth_array)).convert("L")
            xmem_image.save(config.xmem_input_path)

        segmentation_texts = [segmentation_text]

        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        # Provider-agnostic segmentation; defaults to LangSAM when not specified. supports RoboFlow SAM3 api
        model_predictions, boxes, segmentation_texts = get_segmentation_output(
            rgb_image_head,
            self.langsam_model,
            segmentation_texts,
            self.segmentation_count,
            provider=getattr(self.args, "seg_provider", "langsam"),
        )
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        # Save a segmentation overlay image for observability across all providers
        try:
            from models import visualize_segmentation_overlay
            prov = getattr(self.args, "seg_provider", "langsam")
            out_path = config.seg_overlay_image_path.format(provider=str(prov), object=self.segmentation_count)
            status = visualize_segmentation_overlay(rgb_image_head, model_predictions, boxes, segmentation_texts, out_path)
            fname = os.path.join(os.path.dirname(out_path), os.path.basename(out_path))
            if status.get("had_masks") or status.get("had_boxes"):
                self.logger.info(OK + f"Saved segmentation overlay to {fname}" + ENDC)
            else:
                self.logger.info(PROGRESS + f"Saved empty segmentation overlay to {fname} (no masks/bboxes)" + ENDC)
        except Exception as e:
            self.logger.info(PROGRESS + f"Warning: failed to save segmentation overlay: {e}" + ENDC)

        masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)

        # If calibration available from server, pass head K to utils for accurate projection
        K_head = None
        if getattr(self, "calibration", None) and isinstance(self.calibration, dict):
            head_cal = self.calibration.get("head")
            if head_cal and head_cal.get("K"):
                K_head = head_cal.get("K")

        self.logger.info(PROGRESS + f"************************ Before bounding_cubes_world_coordinates len(masks)={len(masks)}" + ENDC)
        bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(            
            rgb_image_head,
            masks,
            depth_array,
            self.head_camera_position,
            self.head_camera_orientation_q,
            self.segmentation_count,
            K_override=K_head,
        )

        utils.save_xmem_image(masks)

        self.segmentation_texts.extend(segmentation_texts)

        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

            bounding_cube_world_coordinates[4][2] -= config.bounding_cube_depth_offset

            object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
            object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
            object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

            print("Position of " + segmentation_texts[i] + ":", list(np.around(bounding_cube_world_coordinates[4], 3)))

            print("Dimensions:")
            print("Width:", object_width)
            print("Length:", object_length)
            print("Height:", object_height)

            if object_width < object_length:
                print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
                print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
            else:
                print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
                print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")

        self.segmentation_count += 1



    def execute_trajectory(self, trajectory):

        self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
        self.main_connection.send([ADD_TRAJECTORY_POINTS, trajectory])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])

        self.trajectory_length += len(trajectory)



    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])



    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])



    def task_completed(self):

        if self.attempted_task:
            self.completed_task = True
        else:
            # Create a trajectory video at the beginning for easier debugging.
            # Redirect only create_video stdout to stderr so main's exec() stdout capture
            # does not treat prints as LLM feedback triggers. Logger remains unaffected.
            try:
                from debug.dbg_utils import create_video_from_images
                from contextlib import redirect_stdout
                import sys
                with redirect_stdout(sys.stderr):
                    create_video_from_images(
                        folder_path=config.trajectory_folder,
                        base_name=config.trajectory_image_base,
                        start_idx=0,
                        end_idx=float('inf'),
                        fps=config.trajectory_video_fps,
                    )
                self.logger.info(OK + "Saved trajectory video from captured frames." + ENDC)
            except Exception as e:
                self.logger.info(PROGRESS + f"Warning: could not create trajectory video: {e}" + ENDC)

            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([TASK_COMPLETED])
            [env_connection_message] = self.main_connection.recv()
            self.logger.info(env_connection_message)

            # If tracking is disabled, skip XMem-based verification entirely
            if self.track_provider == "none":
                self.logger.info(PROGRESS + "Tracking disabled (--track-provider=none); skipping XMem verification." + ENDC)
                self.completed_task = True
                return

            self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
            masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)
            self.logger.info(OK + "Finished generating XMem output!" + ENDC)

            num_objects = len(np.unique(masks[0])) - 1

            new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command)
            new_prompt += "\n"

            self.logger.info(PROGRESS + "Calculating object bounding cubes..." + ENDC)

            for object in range(1, num_objects + 1):

                object_positions = []
                object_orientations = []

                idx_offset = 0

                for i, mask in enumerate(masks):

                    rgb_image = Image.open(config.rgb_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("RGB")
                    depth_image = Image.open(config.depth_image_trajectory_path.format(step=i * config.xmem_output_every)).convert("L")
                    depth_array = np.array(depth_image) / 255.

                    object_mask = mask.copy()
                    object_mask[object_mask != object] = False
                    object_mask[object_mask == object] = True
                    object_mask = torch.Tensor(object_mask)

                    # Reuse head calibration if available
                    K_head = None
                    if getattr(self, "calibration", None) and isinstance(self.calibration, dict):
                        head_cal = self.calibration.get("head")
                        if head_cal and head_cal.get("K"):
                            K_head = head_cal.get("K")

                    bounding_cubes, orientations = utils.get_bounding_cube_from_point_cloud(
                        rgb_image,
                        [object_mask],
                        depth_array,
                        self.head_camera_position,
                        self.head_camera_orientation_q,
                        object - 1,
                        K_override=K_head,
                    )

                    if len(bounding_cubes) == 0:

                        self.logger.info("No bounding cube found: removed.")
                        idx_offset += 1

                    else:

                        [bounding_cube] = bounding_cubes
                        [orientation] = orientations
                        position = bounding_cube[4]
                        orientation = orientation[0]
                        orientation = np.mod(orientation + math.pi, 2 * math.pi) - math.pi

                        object_positions.append(position)

                        if i == 0:

                            object_orientations.append(orientation)

                        else:

                            previous_orientation = object_orientations[i - 1 - idx_offset]
                            possible_orientations = np.array([np.mod(orientation + i * math.pi / 2 + math.pi, 2 * math.pi) - math.pi for i in range(4)])
                            circular_difference = np.minimum(np.abs(possible_orientations - previous_orientation), 2 * math.pi - np.abs(possible_orientations - previous_orientation))
                            min_index = np.argmin(circular_difference)
                            orientation = possible_orientations[min_index]
                            object_orientations.append(orientation)

                new_prompt += self.segmentation_texts[object - 1] + " trajectory positions and orientations:\n"
                new_prompt += "Positions:\n"
                new_prompt += str(np.around([position for p, position in enumerate(object_positions) if p % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "Orientations:\n"
                new_prompt += str(np.around([orientation for o, orientation in enumerate(object_orientations) if o % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "\n"

            self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)

            self.attempted_task = True

            messages = []

            self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(self.client, self.args.language_model, new_prompt, messages, "system", file=sys.stderr)
            self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

            code_block = messages[-1]["content"].split("```python")

            task_completed = self.task_completed
            task_failed = self.task_failed

            for block in code_block:
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    exec(code)



    def task_failed(self):

        self.failed_task = True

        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
