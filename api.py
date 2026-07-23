import numpy as np
import sys
import torch
import math
import os
import re
import config
import json
import models
import segmentation_adapter
from segmentation_adapter import get_segmentation_output
import utils
from common_utils import Trajectory
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC, WARNING
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_EEF, VISUALIZE_GRASP_POSE, VISUALIZE_BOUNDING_BOX
from helpers.image_utils import list_file_paths
from task_state import TaskState

def create_trajectory_videos(logger):
    """Create trajectory debug videos from captured frames (head and wrist).
    Uses `config.trajectory_folder`, `trajectory_image_base`, `trajectory_wrist_image_base`, and `trajectory_video_fps`. Logs warnings instead of raising on errors.
    """
    try:
        from debug.dbg_utils import create_video_from_images
        from contextlib import redirect_stdout
        import sys as _sys
        with redirect_stdout(_sys.stderr):
            create_video_from_images(
                folder_path=config.trajectory_folder,
                base_name=config.trajectory_image_base,
                start_idx=0,
                end_idx=float('inf'),
                fps=config.trajectory_video_fps,
            )
        logger.info(OK + "Saved trajectory video from captured frames." + ENDC)
        try:
            with redirect_stdout(_sys.stderr):
                create_video_from_images(
                    folder_path=config.trajectory_folder,
                    base_name=config.trajectory_wrist_image_base,
                    start_idx=0,
                    end_idx=float('inf'),
                    fps=config.trajectory_video_fps,
                )
            logger.info(OK + "Saved wrist trajectory video from captured frames." + ENDC)
        except Exception as e_w:
            logger.info(PROGRESS + f"Warning: could not create wrist trajectory video: {e_w}" + ENDC)
    except Exception as e:
        logger.info(PROGRESS + f"Warning: could not create trajectory video: {e}" + ENDC)

class API:

    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model, device):

        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        utils.logger = self.logger # injects logger into utils global scope 
        utils.args = self.args     # injects args into utils global scope
        segmentation_adapter.logger = self.logger # injects logger into utils global scope         
        # Parse optional override bbox string "x1,y1,x2,y2"
        segmentation_adapter.set_override_bbox_from_string(args.ovr_bbox)
        
        self.client = client
        self.llm_cache = None
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device
        # Current trajectory step (the number on saved images) - LONG-LIVED, continuous across subtasks
        self.trajectory_step = 1
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.head_image_size = None
        self.wrist_image_size = None
        # Per-(sub)task mutable state. Replaced with a fresh TaskState per subtask.
        self.task = TaskState(start_trajectory_step=self.trajectory_step)

    def _visualize_matched_bounding_boxes(self, bounding_cubes_world_coordinates, segmentation_texts):
        """Visualize matched 3D bounding boxes when --vis-box is enabled."""
        vis_box_regex = getattr(self.args, "vis_box", None)
        if not vis_box_regex or len(bounding_cubes_world_coordinates) == 0:
            return

        vis_box_pat = re.compile(vis_box_regex)
        matched_cubes = [
            bounding_cubes_world_coordinates[i]
            for i, txt in enumerate(segmentation_texts)
            if vis_box_pat.search(txt or "")
        ]
        if not matched_cubes:
            return

        self.logger.info(
            PROGRESS + f"--vis-box: Visualizing {len(matched_cubes)} bounding box(es) matching '{vis_box_regex}'..." + ENDC
        )
        self.main_connection.send([VISUALIZE_BOUNDING_BOX, matched_cubes])
        [vis_msg] = self.main_connection.recv()
        self.logger.info(vis_msg)


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
              ) metaworld_server also returns: calib {K_head, K_wrist - intrinsic of head cam }
          ) utils.get_bounding_cube_from_point_cloud(head_camera_position, head_camera_orientation_q, K_override=None -if pybullet)
            ) contour_pixel_points = <countour of segmented object in 2d image>
            ) get_world_point_world_frame(camera_position, camera_orientation_q, 'head', pixel_point for contour_pixel_points, K_override)
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
        self.cam_info = None
        if isinstance(recv_payload, list):
            if len(recv_payload) >= 6:
                head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message = recv_payload[:5]
                self.cam_info = recv_payload[5]
            elif len(recv_payload) == 5:
                head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message = recv_payload
            elif len(recv_payload) == 1:
                # Only a status line; proceed with saved images but no poses/cam_info
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
        self.head_image_size = rgb_image_head.size
        # Prefer raw depth from .npy if available; fall back to 8-bit image
        depth_npy_path = os.path.splitext(config.depth_image_head_path)[0] + ".npy"
        depth_format = getattr(self.args, "depth_format", "norm_1m")
        if os.path.exists(depth_npy_path):
            depth_array = np.load(depth_npy_path).astype(np.float32)
            if os.environ.get("DEBUG_PINHOLE", "0") == "1":
                ############ Test depth ##########
                px = 156
                py = 72
                pz = depth_array[py, px]
                point=[px, py, pz]
                self.logger.info(PROGRESS + f"**************** api.detect_object: After load .npy depth test  {point}" + ENDC)
        else:
            depth_image_head = Image.open(config.depth_image_head_path).convert("L")
            depth_array = (np.array(depth_image_head).astype(np.float32)) / 255.0
        
        if self.task.segmentation_count == 0:
            xmem_image = Image.fromarray(np.zeros_like(depth_array)).convert("L")
            xmem_image.save(config.xmem_input_path)

        segmentation_texts = [segmentation_text]

        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        # Provider-agnostic segmentation; defaults to LangSAM when not specified. supports RoboFlow SAM3 api
        model_predictions, boxes, segmentation_texts = get_segmentation_output(
            rgb_image_head,
            self.langsam_model,
            segmentation_texts,
            self.task.segmentation_count,
            provider=getattr(self.args, "seg_provider", "langsam"),
        )
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        # Save a segmentation overlay image for observability across all providers
        try:
            from models import visualize_segmentation_overlay
            prov = getattr(self.args, "seg_provider", "langsam")
            out_path = config.seg_overlay_image_path.format(provider=str(prov), object=self.task.segmentation_count)
            status = visualize_segmentation_overlay(rgb_image_head, model_predictions, boxes, segmentation_texts, out_path)
            fname = os.path.join(os.path.dirname(out_path), os.path.basename(out_path))
            if status.get("had_masks") or status.get("had_boxes"):
                self.logger.info(OK + f"Saved segmentation overlay to {fname}" + ENDC)
            else:
                self.logger.info(PROGRESS + f"Saved empty segmentation overlay to {fname} (no masks/bboxes)" + ENDC)
        except Exception as e:
            self.logger.info(PROGRESS + f"Warning: failed to save segmentation overlay: {e}" + ENDC)

        masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)

        if self.args.save_grasp_inputs:
            self._save_seg_masks(masks, segmentation_texts)

        self.logger.info(PROGRESS + f"*** Before bounding_cubes_world_coordinates len(masks)={len(masks)}" + ENDC)
        bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(            
            rgb_image_head,
            masks,
            depth_array,
            self.head_camera_position,
            self.head_camera_orientation_q,
            self.task.segmentation_count,
            cam_info=self.cam_info,
        )

        utils.save_xmem_image(masks)

        self.task.segmentation_texts.extend(segmentation_texts)

        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self._visualize_matched_bounding_boxes(bounding_cubes_world_coordinates, segmentation_texts)

        for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

            bounding_cube_world_coordinates[4][2] -= config.bounding_cube_depth_offset

            object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
            object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
            object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

            obj_position = list(np.around(bounding_cube_world_coordinates[4], 3))
            print("Position of " + segmentation_texts[i] + ":", obj_position)            
            proj_2d_pixel = utils.project_3d_world_pos_to_2d_pixel(self.head_camera_position, self.head_camera_orientation_q, camera="head", image_size=self.head_image_size, world_pos=obj_position, cam_info=self.cam_info )
            self.logger.info(PROGRESS + f"Projected 2D pixel of {segmentation_texts[i]} Position: {proj_2d_pixel}" + ENDC) 
            self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
                

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

        self.task.segmentation_count += 1


    def get_grasp_poses(self, object_name):
        """Return pre-computed grasp pose candidates for the given object.

        Loads poses from outputs/graspgen/grasp_poses_{object_name}.npz and
        prints a summary so the LLM can reason about them.

        Returns:
            poses: np.ndarray (N, 4, 4) – grasp candidate 4x4 matrices.
            scores: np.ndarray (N,) – quality scores (higher is better).
        """
        from providers.grasp_provider import get_grasp_pose_candidates
        poses, scores = get_grasp_pose_candidates(object_name)

        # Sort by descending score
        sorted_idx = np.argsort(-scores)
        poses = poses[sorted_idx]
        scores = scores[sorted_idx]

        print(f"Grasp poses for '{object_name}': {len(poses)} candidates loaded.")
        PRINT_TOP_GRASP_POSES = False
        if PRINT_TOP_GRASP_POSES:
            print(f"Top-5 scores: {scores[:5]}")
            print(f"Top-5 poses (4x4 matrices):")
            for i in range(min(5, len(poses))):
                print(f"  Pose {i}: score={scores[i]:.4f}, position=({poses[i, 0, 3]:.4f}, {poses[i, 1, 3]:.4f}, {poses[i, 2, 3]:.4f}), R_zz={poses[i, 2, 2]:.4f}")

        self.logger.info(PROGRESS + f"Loaded {len(poses)} grasp pose candidates for '{object_name}'." + ENDC)
        return poses, scores

    def visualize_grasp_pose(self, poses):
        """Send grasp pose(s) to the simulation environment for 3D visualization.

        Only active when --vis-grasp flag is set; otherwise a no-op.

        Args:
            poses: A single 4x4 matrix or an (N, 4, 4) array of grasp poses.
        """
        if not getattr(self.args, "vis_grasp", False):
            self.logger.info(PROGRESS + "Skipping grasp visualization (--vis-grasp not set)." + ENDC)
            return
        poses = np.array(poses, dtype=float)
        self.main_connection.send([VISUALIZE_GRASP_POSE, poses])
        resp = self.main_connection.recv()
        if isinstance(resp, list):
            self.logger.info(resp[0])
        else:
            self.logger.info(str(resp))


    def _save_seg_masks(self, masks, segmentation_texts):
        """Save each binary segmentation mask to images_folder as <text>_seg_mask.npy."""
        for i, mask in enumerate(masks):
            label = segmentation_texts[i] if i < len(segmentation_texts) else f"mask_{i}"
            try:
                path = os.path.join(config.images_folder, f"{label}_seg_mask.npy")
                np.save(path, mask)
                self.logger.info(PROGRESS + f"Saved segmentation mask [{i}] ({label}) to {path}" + ENDC)
            except Exception as e:
                self.logger.info(PROGRESS + f"Warning: failed to save segmentation mask [{i}] ({label}): {e}" + ENDC)

    def execute_trajectory(self, trajectory):

        # Downsample preview to max 3 points: start, middle, end
        _preview = trajectory
        try:
            if isinstance(trajectory, (list, tuple)) and len(trajectory) > 5:
                _n = len(trajectory)
                idxs = [0, int(round(( _n - 1) * 0.25)), int(round(( _n - 1) * 0.5)), int(round(( _n - 1) * 0.75)), _n - 1]
                # Ensure strictly increasing unique indices
                _uniq = []
                for _i in idxs:
                    if _i not in _uniq:
                        _uniq.append(_i)
                _preview = [trajectory[i] for i in _uniq]                
        except Exception:
            _preview = trajectory
        if self.args.vis_traj:
            self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
            self.main_connection.send([ADD_TRAJECTORY_POINTS, _preview, "random", False])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])
        try:
            resp = self.main_connection.recv()
            if isinstance(resp, list) and len(resp) >= 2:
                _msg, step = resp[0], resp[1]
                try:
                    self.trajectory_step = int(step)
                except Exception:
                    pass
                try:
                    self.logger.info(_msg)
                except Exception:
                    pass
        except Exception:
            pass
        self.task.trajectory_length += len(trajectory.points)




    def generate_linear_trajectory(self, desc, start_pose, end_pose, num_points=20):
        """Return a linear [x, y, z, theta] trajectory and log the call.
        Logs: desc, start/end poses, and 2D head-camera projection of end pose.
        """
        if not isinstance(start_pose, (list, tuple)) or not isinstance(end_pose, (list, tuple)):
            raise ValueError("start_pose and end_pose must be list/tuple of length 4")
        if len(start_pose) != 4 or len(end_pose) != 4:
            raise ValueError("start_pose and end_pose must be length 4 [x,y,z,theta]")
        if int(num_points) < 2:
            raise ValueError("num_points must be >= 2")
        try:
            sx, sy, sz, s_theta = [float(v) for v in start_pose]
            ex, ey, ez, e_theta = [float(v) for v in end_pose]
        except Exception as e:
            raise ValueError("Invalid pose values: %s" % e)

        # Project end pose (x,y,z) to 2D pixels if camera info is available
        end_px = None
        try:
            if self.head_camera_position is not None and self.head_camera_orientation_q is not None and hasattr(self, 'cam_info') and self.cam_info is not None and self.head_image_size is not None:
                end_px = utils.project_3d_world_pos_to_2d_pixel(
                    self.head_camera_position,
                    self.head_camera_orientation_q,
                    camera='head',
                    image_size=self.head_image_size,
                    world_pos=[ex, ey, ez],
                    cam_info=self.cam_info,
                )
        except Exception:
            end_px = None

        self.logger.info(PROGRESS + "generate_linear_trajectory desc='" + str(desc) + "' start=" + str([sx, sy, sz, s_theta]) + " end=" + str([ex, ey, ez, e_theta]) + " end_px=" + str(end_px) + ENDC)

        traj = []
        n = int(num_points)
        for i in range(n):
            t = i / (n - 1)
            traj.append([
                sx + (ex - sx) * t,
                sy + (ey - sy) * t,
                sz + (ez - sz) * t,
                s_theta + (e_theta - s_theta) * t,
            ])
        return Trajectory(traj, desc)
    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])



    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])


    def run_vlm_review(self):
        """Subsample trajectory frames and ask VLM to judge success.
        Expects strict JSON: { "success": true/false, "reasoning": "..." }.
        Does not mutate the environment.
        """
        from prompts.review_prompt import REVIEW_PROMPT        
        start_idx = self.task.start_attempt_trajectory_step
        # Determine review model: "vlm" uses main model, "vlm:<model>" uses specified model
        review_provider = self.args.review_provider
        if review_provider.startswith("vlm:"):
            review_model = review_provider[len("vlm:"):]
        else:
            review_model = self.args.language_model
        # Subsample every 5th frame from head RGB, starting at the attempt's first step
        frame_paths = list_file_paths(
            root=config.trajectory_folder,
            base_name=config.trajectory_image_base,
            start_idx=start_idx,
        )
        # Subsample every 7th frame from wrist RGB camera, aligned to the same start_idx
        frame_paths += list_file_paths(
            root=config.trajectory_folder,
            base_name=config.trajectory_wrist_image_base,
            start_idx=start_idx,
            skip=7,
        )
        
        # GPT-5 in azure limits to 50 images in a request - test if it confuses the model (the cutoff of wrist images ....)
        if len(frame_paths) > config.max_allowed_vlm_images:
            self.logger.info(WARNING + f"Cutoff wrist frames from the end. frame_paths in preview > config.max_allowed_vlm_images ({config.max_allowed_vlm_images})" + ENDC)
            frame_paths = frame_paths[:config.max_allowed_vlm_images - 1]
                
        # Build prompt with placeholders
        prompt = REVIEW_PROMPT.replace("[INSERT TASK]", str(self.task.command)) \
                              .replace("[INSERT 3D COORDINATES PROMPT SECTION]", self.coords_section) \
                              .replace("[INSERT FRAME PATHS]", "\n".join(frame_paths))
                              
        # Use conversation history; do not summarize. Strip previously accumulated
        # images so only THISreview's fresh trajectory frames are sent — keeps the request under
        # provider image caps (e.g. Bedrock max 20) and focuses the reviewer.
        messages = self.task.conversation_messages
        models.strip_images_from_messages(messages)
        self.logger.info(PROGRESS + f"==================== VLM review using {len(frame_paths)} frames (stride=5), start_idx={start_idx}, model={review_model}." + ENDC)
        messages = models.call_llm_cached(self.main_connection, self.client, review_model, prompt, messages, "user", file=sys.stderr, image_paths=frame_paths, options={"log_msgs": True, "max_tokens": self.args.max_tokens, "reasoning_effort": self.args.reasoning_effort, "cache": self.llm_cache})
        # Drop the dozens of fresh review frames now that the review is done, so they don't persist in the shared conversation 
        models.strip_images_from_messages(messages)
        # Update shared conversation
        self.task.conversation_messages = messages
        # Parse assistant JSON
        raw = messages[-1]["content"] if messages and isinstance(messages[-1], dict) else ""
        try:
            s = raw.strip()
            # Robust extraction: find JSON object boundaries if extra characters slipped in
            if not s.startswith("{"):
                _l = s.find('{'); _r = s.rfind('}')
                if _l >= 0 and _r >= 0 and _r > _l:
                    s = s[_l:_r+1]
            resp = json.loads(s)
        except Exception as e:
            self.logger.info(FAIL + f"Review JSON parse error: {e}. Raw=\n{raw}" + ENDC)
            return
        success = bool(resp.get("success") is True)
        reason = resp.get("reasoning", "")
        improvement_steps = resp.get("improvement_steps", "")
        self.task.review_reason = reason
        self.task.review_improvement_steps = improvement_steps
        self.logger.info((OK if success else FAIL) + f"Review: success={success}. See details in above json" + ENDC)
        if success:
            self.task.completed_task = True
            self.task.review_succeeded = True
        else:
            self.task.failed_task = True

    def task_completed(self):        
        create_trajectory_videos(self.logger)
        
        self.task.attempt_number += 1
        max_attempts = self.task.max_attempts or self.args.attempts
        is_replay = bool(self.args.replay_log)
        # On the final attempt, skip review and accept the result
        # (in replay mode, ignore max_attempts so all blocks execute)
        if not is_replay and self.task.attempt_number >= max_attempts:
            self.task.completed_task = True
            self.task.accepted_without_review = True
            self.logger.info(PROGRESS + f"task_completed final attempt {self.task.attempt_number}/{max_attempts} -- accepting result" + ENDC)
            return
        self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
        self.main_connection.send([TASK_COMPLETED])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        # Skip review entirely when replaying without --replay-vlm-review
        if is_replay and not self.args.replay_vlm_review:
            self.logger.info(PROGRESS + f"Replay mode: skipping VLM review (attempt {self.task.attempt_number})" + ENDC)
            self.task.failed_task = True
            return

        # Dispatch review provider
        if self.args.review_provider == "xmem":
            # Legacy XMem path preserved
            self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
            masks = models.get_xmem_output(self.xmem_model, self.device, self.task.trajectory_length)
            self.logger.info(OK + "Finished generating XMem output!" + ENDC)

            num_objects = len(np.unique(masks[0])) - 1
            new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.task.command)
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
                    bounding_cubes, orientations = utils.get_bounding_cube_from_point_cloud(
                        rgb_image,
                        [object_mask],
                        depth_array,
                        self.head_camera_position,
                        self.head_camera_orientation_q,
                        object - 1,
                        cam_info=self.cam_info,
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
                new_prompt += self.task.segmentation_texts[object - 1] + " trajectory positions and orientations:\n"
                new_prompt += "Positions:\n"
                new_prompt += str(np.around([position for p, position in enumerate(object_positions) if p % config.xmem_lm_input_every == 0], 3)) + "\n"
                new_prompt += "Orientations:\n"
                new_prompt += str(np.around([orientation for o, orientation in enumerate(object_orientations) if o % config.xmem_lm_input_every == 0], 3)) + "\n"
            self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)
            messages = []
            self.logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.call_llm_cached(self.main_connection, self.client, self.args.language_model, new_prompt, messages, "system", file=sys.stderr, options={"max_tokens": self.args.max_tokens, "reasoning_effort": self.args.reasoning_effort, "cache": self.llm_cache})
            self.logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
            code_block = messages[-1]["content"].split("```python")
            task_completed = self.task_completed
            task_failed = self.task_failed
            for block in code_block:
                if len(block.split("```")) > 1:
                    code = block.split("```")[0]
                    exec(code)
            return
        # VLM review path
        self.logger.info(PROGRESS + f"VLM review after attempt {self.task.attempt_number}/{max_attempts}..." + ENDC)
        try:
            self.run_vlm_review()
        except Exception as e:
            self.logger.info(FAIL + f"VLM review failed: {e}" + ENDC)
            # Fall back to marking failure; main loop will replan
            self.task.failed_task = True
        return
    
    def task_failed(self):
        """Mark failure without resetting the environment.
        Keeps the current sim state to mimic real-world retries.
        """
        self.task.failed_task = True
        # Do not reset env or counters; retries will continue from current state

    def reset_eef(self):
        """--reset-eef Re-home the arm to its start pose (RESET_EEF).        """
        self.main_connection.send([RESET_EEF])
        try:
            [env_connection_message] = self.main_connection.recv()
        except Exception:
            env_connection_message = self.main_connection.recv()
        self.logger.info(env_connection_message)









