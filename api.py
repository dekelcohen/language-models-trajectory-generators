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
import common_utils
from common_utils import Trajectory, pose_euler
from sim_adapter import transforms
from PIL import Image
from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC, WARNING
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_EEF, VISUALIZE_GRASP_POSE, VISUALIZE_BOUNDING_BOX, CLEAR_GRASP_MARKERS, TOUCH_INSERT
from helpers.image_utils import list_file_paths
from task_state import TaskState

def create_trajectory_videos(logger, start_idx=0):
    """Build this attempt's review clips (head + wrist) and grow the per-camera full videos.

    Only the frames from `start_idx` (the attempt's first trajectory step) are encoded;
    each clip is then appended to `<config.video_folder>/<base>_full.mp4` with ffmpeg
    stream copy, so the full session video stays inspectable mid-run without re-encoding
    it from frame 0 every time. See helpers/video_utils.py.

    Returns: list of attempt clip paths (head first, then wrist); [] when no frames.
    """
    from helpers.video_utils import build_review_clips
    try:
        clips = build_review_clips(logger, start_idx)
    except Exception as e:
        logger.info(PROGRESS + f"Warning: could not create trajectory videos: {e}" + ENDC)
        return []
    if clips:
        logger.info(OK + f"Saved attempt trajectory clip(s) + updated full video(s): {clips}" + ENDC)
    else:
        logger.info(PROGRESS + f"No trajectory frames from step {start_idx} - no video created." + ENDC)
    return clips

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
        # Rollout tracking (tracking/session.py). Lives in the simulator process; the API
        # only starts/stops it and pulls reports.
        self.tracking_active = False
        self.tracking_summary = None
        # Per-trajectory proprioception (env._trajectory_proprio): commanded vs reached hand
        # motion + finger state; feeds the reviewer's pull evidence (helpers.pull_proprio).
        self.motion_log = []
        # Execution budgets (helpers.run_budget.RunBudget); set by agent_runner.init_agent.
        self.budget = None
        # Per-(sub)task mutable state. Replaced with a fresh TaskState per subtask.
        self.task = TaskState(start_trajectory_step=self.trajectory_step)

    def _visualize_matched_bounding_boxes(self, bounding_cubes_world_coordinates, segmentation_texts):
        """Visualize matched 3D bounding boxes when --vis-box is enabled."""
        vis_box_regex = self.args.vis_box
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


    def _capture_head_image_and_depth(self, capture=True):
        """Send CAPTURE_IMAGES, parse camera poses/cam_info, load head RGB + depth.

        Shared by detect_object, convert_2d_point_to_3d_world and scene perception. Sets
        self.head/wrist camera pose fields, self.cam_info and self.head_image_size.
        With capture=False the env is NOT asked for a new frame: the files on disk (and
        the camera poses/cam_info from the previous capture) are reused - use it only
        right after a capture, to avoid re-rendering the same state twice.
        Returns (rgb_image_head: PIL.Image, depth_array: np.ndarray[H,W] float32).
        """
        if not capture:
            return self._load_head_image_and_depth()
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

        return self._load_head_image_and_depth()

    def _load_head_image_and_depth(self):
        """Load the head RGB + depth already written to disk by the last capture.

        No env round-trip. Sets self.head_image_size. Depth comes from the raw .npy when
        present, else from the 8-bit depth PNG.
        """
        rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        self.head_image_size = rgb_image_head.size
        # Prefer raw depth from .npy if available; fall back to 8-bit image
        depth_npy_path = os.path.splitext(config.depth_image_head_path)[0] + ".npy"
        if os.path.exists(depth_npy_path):
            depth_array = np.load(depth_npy_path).astype(np.float32)
            if os.environ.get("DEBUG_PINHOLE", "0") == "1":
                ############ Test depth ##########
                px = 156
                py = 72
                pz = depth_array[py, px]
                point = [px, py, pz]
                self.logger.info(PROGRESS + f"**************** api._capture_head_image_and_depth: After load .npy depth test  {point}" + ENDC)
        else:
            depth_image_head = Image.open(config.depth_image_head_path).convert("L")
            depth_array = (np.array(depth_image_head).astype(np.float32)) / 255.0

        return rgb_image_head, depth_array

    def _overlay_affordance_points(self, rgb_image_head, points_xy, object_name):
        """Draw all 2D affordance points on the head RGB, colored best->worst
        (green -> yellow by descending quality/order). Saves to images_folder."""
        try:
            import cv2
            img = cv2.cvtColor(np.array(rgb_image_head), cv2.COLOR_RGB2BGR)
            n = max(1, len(points_xy))
            for i, (x, y) in enumerate(points_xy):
                # green (0,255,0) at best -> yellow (0,255,255) at worst (BGR)
                t = i / max(1, n - 1)
                color = (0, 255, int(255 * t))
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 0), 1)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(object_name))[:60] or "object"
            # One file per perception call (same step as scene_analysis_head_<step>.png), so every
            # affordance pointing of a rollout is kept, not only the last one.
            step = getattr(self, "trajectory_step", 0)
            out_path = os.path.join(config.images_folder, f"affordance_points_{safe}_{step}.png")
            cv2.imwrite(out_path, img)
            self.logger.info(OK + f"Saved affordance-points overlay to {out_path}" + ENDC)
            return out_path
        except Exception as e:
            self.logger.info(PROGRESS + f"Warning: failed to save affordance-points overlay: {e}" + ENDC)
            return None

    def convert_2d_point_to_3d_world(self, points_xy, object_name, print_out=False, capture=True,
                                     mask=None):
        """Convert ranked 2D affordance pixel points to 3D world coordinates.

        The depth under each point is sampled over a small patch rather than read from the
        single pixel (see ``utils.sample_surface_depth``): the features a VLM points at are
        only a few pixels thick, so a 1-2 px pointing error read pixel-wise returns the
        backdrop's depth and puts the 3D point 10 cm to 1 m behind the thing to grasp.

        Args:
            points_xy: list of [x, y] pixel points (already denormalized), ordered
                best-first (descending predicted grasp quality).
            object_name: identifying name of the target object (for logging/overlay).
            capture: re-capture the head camera first. Pass False when the caller has
                just captured (e.g. scene perception): the 2D points refer to THAT image,
                so re-rendering would risk mismatched pixels and waste a render.
            mask: optional boolean segmentation mask of the target object. When given, the
                patch is confined to it and the sampled depth is cross-checked against the
                object's own depth range; a point that fails is dropped rather than
                returned as a confident coordinate on the wrong surface.
        Returns:
            list of 3D world points (np.ndarray) aligned with the input order; entries
            for invalid/out-of-range points are None.
        """
        rgb_image_head, depth_array = self._capture_head_image_and_depth(capture=capture)
        h, w = depth_array.shape[:2]

        world_points = []
        for (x, y) in points_xy:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                self.logger.info(PROGRESS + f"Warning: affordance point ({xi},{yi}) out of image bounds {w}x{h}; skipping." + ENDC)
                world_points.append(None)
                continue
            z, info = utils.sample_surface_depth(depth_array, xi, yi, mask=mask)
            if z is None or not np.isfinite(z):
                self.logger.info(PROGRESS + f"Warning: non-finite depth at affordance point ({xi},{yi}); skipping." + ENDC)
                world_points.append(None)
                continue
            ok, check = utils.depth_matches_object(z, depth_array, mask)
            if not ok:
                self.logger.info(PROGRESS + f"Warning: affordance point ({xi},{yi}) depth {z:.4f} is outside "
                                 f"{object_name}'s depth range [{check['low']:.4f}, {check['high']:.4f}] "
                                 f"(+/-{check['tolerance']:.4f}); skipping." + ENDC)
                world_points.append(None)
                continue
            center = info.get("center")
            if center is not None and abs(center - z) > 1e-6:
                self.logger.info(PROGRESS + f"Affordance point ({xi},{yi}): patch depth {z:.4f} replaces "
                                 f"center-pixel {center:.4f} ({info['n_used']}/{info['n_total']} px sampled)." + ENDC)
            world_point = utils.get_world_point_world_frame(
                self.head_camera_position, self.head_camera_orientation_q, "head",
                self.head_image_size, [xi, yi, z], cam_info=self.cam_info,
            )
            world_points.append(world_point)
            if print_out:
                print(f"Affordance-pointing of {object_name}: {list(np.around(np.array(world_point).flatten(), 3))}")

        self._overlay_affordance_points(rgb_image_head, points_xy, object_name)
        return world_points

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
        rgb_image_head, depth_array = self._capture_head_image_and_depth()

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
            provider=self.args.seg_provider,
        )
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        # Save a segmentation overlay image for observability across all providers
        try:
            from models import visualize_segmentation_overlay
            prov = self.args.seg_provider
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

        self._snapshot_detections(rgb_image_head, depth_array, masks, segmentation_texts)

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

    def _load_head_robot_mask(self):
        """Robot self-mask of the last head capture (bool HxW) or None (sim without segmentation)."""
        path = os.path.splitext(config.depth_image_head_path)[0] + "_robot_mask.npy"
        try:
            return np.load(path) if os.path.exists(path) else None
        except Exception:
            return None

    def _snapshot_detections(self, rgb_image_head, depth_array, masks, segmentation_texts):
        """First detection of each label in this subtask -> displacement snapshot (pixels + depth).

        Labels first seen before the robot moved at all in this run are also kept as run-level
        baselines (``self.run_displacement_snaps``), so a later subtask's reviewer can tell that
        e.g. the door was already pushed open by an earlier subtask.
        """
        if not getattr(config, "review_displacement_evidence", True):
            return
        from helpers import object_displacement as od
        if not hasattr(self, "run_displacement_snaps"):
            self.run_displacement_snaps = {}
            self.run_start_trajectory_step = self.trajectory_step
        run_ok = self.trajectory_step <= self.run_start_trajectory_step
        start = getattr(self.task, "start_trajectory_step", None)
        if start is not None and self.trajectory_step > start:
            # after the robot moved, a "first" detection shows the object where it was moved TO;
            # comparing against that would report a relocated object as "in place"
            return
        try:
            depth_m = None
            for mask, text in zip(masks, segmentation_texts):
                key = str(text).strip().lower()
                if not key or key in self.task.displacement_snaps:
                    continue
                if depth_m is None:
                    depth_m = od.metric_depth(depth_array, self.cam_info)
                snap = od.snapshot(text, np.array(rgb_image_head), depth_m, np.asarray(mask),
                                   camera_position=self.head_camera_position, step=self.trajectory_step,
                                   robot_mask=self._load_head_robot_mask())
                if snap is not None:
                    self.task.displacement_snaps[key] = snap
                    if run_ok and key not in self.run_displacement_snaps:
                        self.run_displacement_snaps[key] = snap
                    self.logger.info(PROGRESS + f"Displacement: snapshot of '{text}' bbox={snap['bbox']} "
                                     f"({int(snap['mask'].sum())} px) at step {self.trajectory_step}" + ENDC)
        except Exception as e:                   # evidence is optional; never break detection
            self.logger.info(WARNING + f"Displacement snapshot failed: {type(e).__name__}: {e}" + ENDC)

    def _displacement_review_evidence(self):
        """Fresh head RGB-D vs each detected object's first snapshot -> ``(facts_list, text)``.

        Also compares the run-level baselines (taken before the robot first moved) that are
        older than this subtask, reported as a separate "since the start of the command" block.
        """
        snaps = getattr(self.task, "displacement_snaps", None) or {}
        start = getattr(self.task, "start_trajectory_step", None)
        run_snaps = {k: s for k, s in (getattr(self, "run_displacement_snaps", None) or {}).items()
                     if start is not None and s.get("step") is not None and s["step"] < start}
        if (not snaps and not run_snaps) or not getattr(config, "review_displacement_evidence", True):
            return [], None
        from helpers import object_displacement as od
        try:
            rgb, depth_raw = self._capture_head_image_and_depth()
            depth_m = od.metric_depth(depth_raw, self.cam_info)
            robot_mask = self._load_head_robot_mask()
            facts = [od.compare(s, np.array(rgb), depth_m, camera_position=self.head_camera_position,
                                robot_mask=robot_mask)
                     for s in snaps.values()]
            run_facts = [od.compare(s, np.array(rgb), depth_m, camera_position=self.head_camera_position,
                                    robot_mask=robot_mask)
                         for s in run_snaps.values()]
        except Exception as e:
            self.logger.info(WARNING + f"Review: displacement evidence failed: {type(e).__name__}: {e}" + ENDC)
            return [], None
        facts = [f for f in facts if f]
        run_facts = [f for f in run_facts if f]
        od.mark_shared_regions(list(snaps.values()), facts)
        od.mark_shared_regions(list(run_snaps.values()), run_facts)
        already = [lbl for lbl, v in (self.task.displacement_prev or {}).items() if v == "left_place"]
        text = od.format_facts(facts, already=already)
        run_text = od.format_run_facts(run_facts)
        if run_text:
            text = (text + "\n" + run_text) if text else run_text
        for tag, fl in (("", facts), ("since command start ", run_facts)):
            for f in fl:
                self.logger.info(PROGRESS + f"Review: displacement {tag}'{f['label']}' -> {f['verdict']} "
                                 f"farther={f['farther']} nearer={f['nearer']} changed={f['changed']} ncc={f['ncc']}"
                                 + ENDC)
        self.task.displacement_prev = {f["label"]: f["verdict"] for f in facts}
        return facts, text

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

    def visualize_grasp_pose(self, poses, desc=""):
        """Draw grasp pose(s) in the simulation so they show up in the images and video.

        Only active when --vis-grasp is set; otherwise a no-op.

        Args:
            poses: an end-effector pose the robot will execute - ``[x, y, z, rotation]``
                (top-down) or ``[x, y, z, roll, pitch, yaw]`` (horizontal approach) - a
                list of such poses, or a pre-computed ``(4,4)`` / ``(N,4,4)`` grasp matrix
                stack (GraspGen candidates).
            desc: optional short label logged next to the pose.
        """
        if not self.args.vis_grasp:
            self.logger.info(PROGRESS + "Skipping grasp visualization (--vis-grasp not set)." + ENDC)
            return
        try:
            kind, norm_poses = common_utils.normalize_grasp_viz_poses(poses)
        except Exception as e:
            self.logger.info(FAIL + f"visualize_grasp_pose: {e}" + ENDC)
            raise
        self.logger.info(PROGRESS + f"Visualizing {len(norm_poses)} {kind} grasp pose(s){(' - ' + desc) if desc else ''}: "
                                    f"{[np.around(np.asarray(p, dtype=float), 3).tolist() for p in norm_poses[:3]]}" + ENDC)
        self.main_connection.send([VISUALIZE_GRASP_POSE, {"kind": kind, "poses": norm_poses, "desc": desc}])
        resp = self.main_connection.recv()
        if isinstance(resp, list):
            self.logger.info(resp[0])
        else:
            self.logger.info(str(resp))

    def clear_grasp_markers(self):
        """Remove every grasp marker drawn so far (called between sub-task attempts)."""
        if not self.args.vis_grasp:
            return
        try:
            self.main_connection.send([CLEAR_GRASP_MARKERS])
            resp = self.main_connection.recv()
            if isinstance(resp, list):
                self.logger.info(resp[0])
        except Exception as e:
            self.logger.info(FAIL + f"Failed to clear grasp markers: {e}" + ENDC)


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

    def locate_with_wrist_camera(self, description, max_points=4):
        """Point at ``description`` in the wrist-camera image and return 3D world candidates.

        For small features the head camera cannot resolve (a USB port on a side face): the
        perception VLM (--planner-perception-vlm) points at every visible match in the wrist
        RGB, each point is lifted to 3D with the wrist depth + calibrated wrist camera pose.
        Returns a best-first list of ``{"label", "position": [x, y, z], "pixel": [u, v]}``;
        entries without valid depth are dropped. Uses only camera images (Sim2Real safe).
        """
        from llm_providers.model_registry import get_model_info
        self._capture_head_image_and_depth()
        wrist_info = (self.cam_info or {}).get("wrist") if isinstance(self.cam_info, dict) else None
        if self.wrist_camera_position is None or not wrist_info or wrist_info.get("viewMatrix") is None:
            print("locate_with_wrist_camera: no calibrated wrist camera in this environment.")
            return []
        rgb = Image.open(config.rgb_image_wrist_path).convert("RGB")
        width, height = rgb.size
        self.wrist_image_size = rgb.size
        depth_npy = os.path.splitext(config.depth_image_wrist_path)[0] + ".npy"
        if not os.path.exists(depth_npy):
            print("locate_with_wrist_camera: wrist depth unavailable.")
            return []
        depth = np.load(depth_npy).astype(np.float32)

        self.task.wrist_locate_count = getattr(self.task, "wrist_locate_count", 0) + 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(description))[:40] or "target"
        stem = os.path.join(config.images_folder, f"wrist_locate_{safe}_{self.trajectory_step}_{self.task.wrist_locate_count}")
        image_path = stem + ".png"
        rgb.save(image_path)

        model = self.args.planner_perception_vlm
        coords_key = get_model_info(model).pointing_coords_format
        coords_format = config.affordance_coords_format_by_key[coords_key]
        prompt = (
            "This image is from a robot's wrist camera (the white robot gripper is in view). "
            f"Point at EVERY visible instance of: {description}. Best match first, at most {int(max_points)}. "
            "Point at the centre of each instance; skip ones hidden by the gripper. "
            f"Coordinates format: {coords_format}. Answer ONLY a JSON list like "
            '[{"point": [..], "label": "short label"}]; [] if none is visible.'
        )
        try:
            messages = models.call_llm_cached(
                self.main_connection, self.client, model, prompt, [], role="user", image_paths=[image_path],
                options={"max_tokens": self.args.max_tokens, "reasoning_effort": self.args.reasoning_effort,
                         "cache": self.llm_cache, "conversation_key": "wrist_locate"},
            )
            text = (messages[-1]["content"] if messages and isinstance(messages[-1], dict) else "") or ""
        except Exception as e:
            print(f"locate_with_wrist_camera: perception VLM failed: {e}")
            return []
        m = re.search(r"\[.*\]", text, re.DOTALL)
        try:
            items = json.loads(m.group(0)) if m else []
        except Exception:
            items = []
        if isinstance(items, dict):
            items = [items]

        results = []
        for item in items[: int(max_points)]:
            p = item.get("point") if isinstance(item, dict) else None
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            if coords_key == "yx_norm_1000":
                from features_markers.bbox_providers.gemini_bbox_provider import denormalize_yx_point_to_xy_pixels
                u, v = denormalize_yx_point_to_xy_pixels(p, width, height)
            else:
                u, v = float(p[0]), float(p[1])
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < width and 0 <= vi < height):
                continue
            z, _info = utils.sample_surface_depth(depth, ui, vi)
            if z is None or not np.isfinite(z):
                continue
            world = utils.get_world_point_world_frame(
                self.wrist_camera_position, self.wrist_camera_orientation_q, "wrist",
                (width, height), [ui, vi, z], cam_info=self.cam_info)
            results.append({"label": str(item.get("label") or description),
                            "position": [round(float(c), 4) for c in np.asarray(world).flatten()[:3]],
                            "pixel": [ui, vi]})

        try:
            import cv2
            img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            for i, r in enumerate(results):
                cv2.circle(img, tuple(r["pixel"]), 4, (0, 0, 255), 1)
                cv2.putText(img, str(i + 1), (r["pixel"][0] + 5, r["pixel"][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imwrite(stem + "_points.png", img)
        except Exception:
            pass
        self.logger.info(PROGRESS + f"[wrist-locate] '{description}': {results} (image {image_path})" + ENDC)
        print(f"locate_with_wrist_camera('{description}'): {len(results)} candidate(s), best first:")
        for i, r in enumerate(results):
            print(f"  {i + 1}. {r['label']}: {r['position']}")
        return results

    def insert_with_touch(self, axis, tool_length=0.025, max_s=20.0, grip_force=20.0, tool_offset=None,
                          search_radius=None, search_line=None, search_half=0.02, search_cross=None,
                          insert_force=None, opening_width=None):
        """Push the held part along world ``axis`` into a hole/port, guided by force feedback.

        Start with the part's tip a few mm in front of the opening, aligned by vision. The robot
        then probes, spirals on the face to find the opening (vision is ~1-2 mm off) and pushes it
        in. ``tool_length``: fingertip centre -> part tip along ``axis`` (m). ``tool_offset``: optional
        measured world vector eef_pos -> part tip (overrides ``tool_length``). ``search_radius`` (m,
        default 0.003): spiral search radius; larger is slow (~100 s at 0.008). ``search_line``: a world
        direction across the axis (e.g. [0, 1, 0]) to search by feel along a LINE instead of the spiral,
        +-``search_half`` m (default 0.02) around the start, in a few passes ~0.4 mm apart across it: use
        it when the opening's height is known but its position along the face is uncertain (> 3 mm).
        ``search_cross`` (m, default 0.0008): half-width of the band ACROSS the line covered by those
        passes; widen it (e.g. 0.002) when the height is uncertain by more than ~1 mm (slower: time
        grows with the number of passes). ``max_s`` is extended to fit the sweep. Before reporting
        'seated', the hand is pushed 2 mm sideways: a seated part can't move, while a part that slid in the
        grasp can, which gives 'slipped'. Returns a dict with
        ``outcome`` ('seated' | 'jammed' | 'search_exhausted' | 'no_contact' | 'slipped' | 'over_force' |
        'timeout' | 'unsupported' | 'disabled'), ``advance_mm``, ``tip_pos`` and ``events`` (proprioception only).

        Part-specific tuning (defaults fit a USB-A-like plug; ADAPT them across retries from the outcome):
        ``insert_force`` (N, default 18): the push that seats the part - above the opening's retention, below
        what makes the part slip in the grasp ('jammed' with the tip already in -> raise it ~30%;
        'slipped' -> lower it or re-grasp with a higher ``grip_force``). ``opening_width`` (m, default
        0.012): the opening's width along the search line, as measured by vision; the line search's
        edge finder brackets an opening by two partial catches about this far apart.
        """
        if not getattr(self.args, "touch", True):
            print("insert_with_touch: touch is disabled (--no-touch).")
            return {"outcome": "disabled"}
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        off = (np.asarray(tool_offset, dtype=float) if tool_offset is not None else a * float(tool_length))
        config = {}
        if search_radius:
            config["spiral_r_max"] = float(search_radius)
        if insert_force:
            f = float(insert_force)
            config.update(f_insert=f, f_max=max(35.0, 1.8 * f))
        if opening_width:
            w = float(opening_width)
            config.update(edge_min_sep=0.75 * w, edge_max_sep=1.35 * w)
        if search_line is not None:
            from touch.insertion import InsertConfig
            c = InsertConfig()
            config.update(sweep_dir=[float(x) for x in search_line], sweep_half=float(search_half))
            rows = c.sweep_rows
            if search_cross:
                rows = 1 + 2 * int(np.ceil(float(search_cross) / c.sweep_pitch - 1e-9))
                config["sweep_rows"] = rows
            travel = float(search_half) * (1 + 2 * rows)
            # + edge-finder cycles (column raster across the line each)
            local = c.max_edge_cycles * (2 * round(c.col_half / c.col_pitch) + 1) * (2 * c.cross_half + c.col_pitch)
            max_s = max(float(max_s), (travel + local) / c.spiral_speed + 10.0)
        spec = {"axis": a.tolist(), "tool_offset": off.tolist(),
                "max_s": float(max_s), "grip_force": float(grip_force)}
        if config:
            spec["config"] = config
        self.logger.info(PROGRESS + f"[touch] insert_with_touch {spec}" + ENDC)
        self.main_connection.send([TOUCH_INSERT, spec])
        msg, res = self.main_connection.recv()
        self.logger.info(msg)
        for ev in res.get("events", []):
            self.logger.info(PROGRESS + f"[touch]   {ev}" + ENDC)
        summary = {k: v for k, v in res.items() if k != "events"}
        print(f"insert_with_touch: {summary}")
        self._record_touch("insert", res)
        self._budget_tick("insert_with_touch")
        return res

    def move_until_contact(self, direction, max_travel=0.03, force=4.0, speed=0.01, grip_force=20.0, max_s=10.0):
        """Guarded move: move the hand along world ``direction`` until a contact force is felt.

        Force-feedback primitive (impedance control + the measured external force). The hand advances
        at ``speed`` m/s for at most ``max_travel`` m; contact = resistance above ``force`` N. It then
        rests lightly on the surface and stops. ``grip_force`` > 0 keeps squeezing a held part; 0 leaves
        the fingers as they are (open hand). Returns ``outcome`` ('contact' | 'no_contact' | 'over_force'
        | 'unsupported' | 'disabled'), ``eef_pos`` (measured EE position at rest, same frame as
        ``get_robot_state()["eef_pos"]``), ``finger_bottom_z`` (height of the closed finger bottoms - a
        closed-hand downward touch measures an object's top height), ``travel_mm`` and ``contact_N``.
        Uses: measure a part's top height before grasping (close the gripper, touch down on it), in-hand
        offset calibration (touch a held part against a known face), touching a door / drawer front
        before pushing, checking a part is there.
        """
        if not getattr(self.args, "touch", True):
            print("move_until_contact: touch is disabled (--no-touch).")
            return {"outcome": "disabled"}
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        spec = {"mode": "contact", "axis": d.tolist(), "tool_offset": [0.0, 0.0, 0.0], "max_s": float(max_s),
                "grip_force": float(grip_force),
                "guarded": {"max_travel": float(max_travel), "f_contact": float(force), "speed": float(speed)}}
        self.logger.info(PROGRESS + f"[touch] move_until_contact {spec}" + ENDC)
        self.main_connection.send([TOUCH_INSERT, spec])
        msg, res = self.main_connection.recv()
        self.logger.info(msg)
        for ev in res.get("events", []):
            self.logger.info(PROGRESS + f"[touch]   {ev}" + ENDC)
        out = {"outcome": res.get("outcome"), "eef_pos": res.get("eef_pos"),
               "travel_mm": res.get("advance_mm"), "contact_N": res.get("contact_N")}
        if out["eef_pos"] is not None:
            # Robot geometry: with CLOSED fingers moving down, the finger bottoms touched at this height.
            out["finger_bottom_z"] = round(float(out["eef_pos"][2]) - config.FINGER_BOTTOM_BELOW_EEF, 4)
            width = res.get("gripper_width_mm")
            if (res.get("outcome") == "contact" and d[2] < -0.9 and width is not None and width < 3.0):
                self._last_top_touch = {"z": out["finger_bottom_z"], "xy": list(out["eef_pos"][:2])}
        if res.get("outcome") == "unsupported":
            out["reason"] = res.get("reason", "no torque control on this simulator")
        print(f"move_until_contact: {out}")
        self._record_touch("contact", {**res, **out})
        self._budget_tick("move_until_contact", out.get("eef_pos"))
        return out

    def get_robot_state(self):
        """Measured proprioception: {'eef_pos': [x, y, z], 'gripper_open', 'gripper_width' (m),
        'gripper_holding' (closed but stalled on an object), 'rotation' (measured top-down hand yaw, same
        convention as a pose's 4th value; None if the hand is tilted)}. Values are None when not reported."""
        state = {"eef_pos": None, "gripper_open": None, "gripper_width": None, "gripper_holding": None,
                 "rotation": None}
        try:
            self.main_connection.send([config.GET_ROBOT_STATE, {}])
            resp = self.main_connection.recv() or {}
        except Exception as e:
            self.logger.info(WARNING + f"get_robot_state failed: {e}" + ENDC)
            return state
        for k in state:
            if resp.get(k) is not None:
                state[k] = resp[k]
        if state["eef_pos"] is not None:
            state["eef_pos"] = [float(v) for v in state["eef_pos"]]
        print(f"get_robot_state: {state}")
        return state

    def gripper_holding(self):
        """True if the closed fingers are stalled on an object (measured finger width), else False;
        None if the gripper does not report it."""
        return self.get_robot_state()["gripper_holding"]

    def _record_touch(self, kind, res):
        recs = getattr(self.task, "touch_results", None)
        if recs is not None and isinstance(res, dict):
            recs.append({"kind": kind, "step": self.trajectory_step, "result": res})

    def _touch_review_evidence(self, start_idx):
        from helpers.touch_evidence import format_touch_evidence
        recs = [r for r in (getattr(self.task, "touch_results", None) or [])
                if int(r.get("step", 0)) >= int(start_idx)]
        text = format_touch_evidence(recs)
        if text:
            self.logger.info(PROGRESS + "Review: force feedback -> "
                             + "; ".join(f"{r['kind']}:{r['result'].get('outcome')}" for r in recs) + ENDC)
        return text

    def pull_with_touch(self, direction, max_s=20.0, grip_force=30.0, pull_config=None):
        """Pull the grasped handle with force feedback; the mechanism steers the hand (hinge arc or slide).

        ``direction``: initial world pull direction (horizontal part used). Returns a dict with
        ``outcome`` ('opened' | 'end_stop' | 'blocked' | 'slipped' | 'timeout' | 'unsupported' |
        'disabled'), ``travel_cm``, ``swept_deg``, ``direction`` (current estimate), ``hinge``
        (fitted centre/radius or None), ``slip_pos`` and ``events`` (proprioception only).
        """
        if not getattr(self.args, "touch", True):
            print("pull_with_touch: touch is disabled (--no-touch).")
            return {"outcome": "disabled"}
        d = [float(v) for v in direction][:3]
        d += [0.0] * (3 - len(d))  # a horizontal [dx, dy] is fine
        spec = {"direction": d, "max_s": float(max_s),
                "grip_force": float(grip_force), "config": dict(pull_config or {})}
        self.logger.info(PROGRESS + f"[touch] pull_with_touch {spec}" + ENDC)
        self.main_connection.send([config.TOUCH_PULL, spec])
        msg, res = self.main_connection.recv()
        self.logger.info(msg)
        for ev in res.get("events", []):
            self.logger.info(PROGRESS + f"[touch]   {ev}" + ENDC)
        print(f"pull_with_touch: {({k: v for k, v in res.items() if k != 'events'})}")
        self._record_touch("pull", res)
        self._budget_tick("pull_with_touch")
        return res

    def execute_trajectory(self, trajectory, max_speed=None, max_rot_speed=None):
        """Run a Trajectory. ``max_speed`` (m/s) / ``max_rot_speed`` (rad/s): optional hand speed
        limits - poses are interpolated so the arm moves smoothly instead of lunging between sparse
        waypoints (e.g. 0.1 m/s and 0.5 rad/s when carrying or tilting a container of liquid).
        Returns the measured outcome ``{"reached", "stalled", "shortfall_m", "eef_pos", "holding"}``
        (None if not reported); a stall (hand stopped > 2 cm short of the end) is also printed."""
        # Rebuild into a plain Trajectory before anything is sent: a custom class from the
        # model's own code block cannot be reconstructed in the simulator process.
        trajectory = common_utils.Trajectory.normalize(trajectory)
        if getattr(self, "_gripper_closed", False):
            # Carrying / pulling: sparse waypoints otherwise command several m/s (beyond a real
            # arm's limits) and fling a friction-held part out of the fingers.
            if max_speed is None:
                max_speed = config.CLOSED_GRIPPER_MAX_SPEED
            if max_rot_speed is None:
                max_rot_speed = getattr(config, "CLOSED_GRIPPER_MAX_ROT_SPEED", None)
        if max_speed or max_rot_speed:
            n0 = len(trajectory.points)
            trajectory = common_utils.Trajectory(
                common_utils.limit_speed(trajectory.points, max_speed, max_rot_speed), trajectory.desc)
            self.logger.info(PROGRESS + f"Speed limit max_speed={max_speed} max_rot_speed={max_rot_speed}: "
                             f"{n0} -> {len(trajectory.points)} poses" + ENDC)

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
        tracking_abort = None
        grasp_slip = None
        motion = None
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
                if len(resp) >= 3 and isinstance(resp[2], dict):
                    if resp[2].get("tracking_abort"):
                        tracking_abort = resp[2]
                    if isinstance(resp[2].get("grasp_slip"), dict):
                        grasp_slip = resp[2]["grasp_slip"]
                    if isinstance(resp[2].get("proprio"), dict):
                        rec = dict(resp[2]["proprio"])
                        try:    # commanded first waypoint, in the LLM's trajectory coordinates
                            rec["traj_start"] = [float(v) for v in list(trajectory.points[0])[:3]]
                        except Exception:
                            pass
                        self.motion_log.append(rec)
                        motion = self._motion_feedback(rec)
        except Exception:
            pass
        self.task.trajectory_length += len(trajectory.points)
        if tracking_abort is not None:
            self._handle_tracking_abort(tracking_abort)
        if grasp_slip is not None:
            if grasp_slip.get("on_trip") == "return":
                text = self.grasp_slip_feedback(grasp_slip)
                print(f"[guard] {text}")
                self.logger.info(FAIL + text + ENDC)
                motion = dict(motion or {})
                motion["guard"] = grasp_slip
            else:
                self._handle_grasp_slip(grasp_slip)
        self._check_repeated_no_progress(trajectory, motion)
        try:
            target = list(trajectory.points[-1])
        except Exception:
            target = None
        self._budget_tick("execute_trajectory", (motion or {}).get("eef_pos"), target)
        return motion

    def _budget_tick(self, name, eef_pos=None, target=None):
        """Execution budgets (helpers.run_budget): per-block action cap, churn, time budgets."""
        budget = getattr(self, "budget", None)
        if budget is not None:
            budget.on_robot_action(name, eef_pos, target)

    #: Consecutive calls with the SAME commanded end pose and no measured hand motion before the
    #: code is stopped (a ``while`` loop whose exit condition the arm cannot meet).
    REPEAT_NO_PROGRESS_CALLS = 15

    def _check_repeated_no_progress(self, trajectory, motion):
        try:
            target = np.asarray(list(trajectory.points[-1]), dtype=float)
            end = None if not motion else np.asarray(motion["eef_pos"], dtype=float)
        except Exception:
            return
        prev = getattr(self, "_repeat_state", None)
        same = (prev is not None and end is not None and prev["end"] is not None
                and target.shape == prev["target"].shape
                and float(np.max(np.abs(target - prev["target"]))) < 1e-3
                and float(np.linalg.norm(end - prev["end"])) < 0.002)
        n = prev["n"] + 1 if same else 0
        self._repeat_state = {"target": target, "end": end, "n": n}
        if n + 1 >= self.REPEAT_NO_PROGRESS_CALLS:
            self._repeat_state = None
            raise RuntimeError(
                f"execute_trajectory was called {n + 1} times in a row with the same target "
                f"{[round(float(v), 4) for v in target[:3]]} and the hand did not move (it rests at pose "
                f"{motion.get('pose')}, {motion.get('shortfall_m', 0) * 100:.1f} cm from that target). "
                "A loop is waiting for a condition the arm cannot reach - stop retrying the same move. "
                "Accept the measured pose (it may already be close enough), or change the plan.")

    #: The hand ending farther than this from the trajectory's last target = it was blocked...
    STALL_SHORTFALL_M = 0.02
    #: ...if it also (nearly) stopped during the final settle; more motion = still catching up.
    STALL_SETTLE_MOTION_M = 0.005

    def _motion_feedback(self, rec):
        """Encoder-measured outcome of one trajectory (what libfranka reports): did the hand reach
        the last target, and if not, where it stopped. Printed so the model's code output shows
        a stall, and returned so code can chain the next move from the measured pose."""
        end, cmd = rec.get("ee_end"), rec.get("cmd_end")
        if end is None or cmd is None:
            return None
        shortfall = float(np.linalg.norm(np.subtract(cmd[:3], end[:3])))
        pre = rec.get("ee_pre_settle")
        settling = pre is not None and float(np.linalg.norm(np.subtract(end[:3], pre[:3]))) > self.STALL_SETTLE_MOTION_M
        # measured hand position in the frame trajectory poses are written in (eef_pos is the
        # fingertip centre, ~0.06 m from the pose point along the approach axis)
        cmd_pose = rec.get("cmd_pose_end")
        pose = end[:3] if cmd_pose is None else np.add(end[:3], np.subtract(cmd_pose[:3], cmd[:3]))
        out = {"reached": shortfall <= self.STALL_SHORTFALL_M, "shortfall_m": round(shortfall, 4),
               "eef_pos": [round(float(v), 4) for v in end[:3]],
               "pose": [round(float(v), 4) for v in pose],
               "holding": rec.get("holding_end"),
               "stalled": shortfall > self.STALL_SHORTFALL_M and not settling}
        if out["stalled"]:
            msg = (f"[motion] STALLED '{rec.get('desc') or ''}': hand stopped {shortfall * 100:.1f} cm short "
                   f"of the trajectory end, at pose {out['pose']} (trajectory frame; holding={out['holding']}) - "
                   "blocked or resisted. Start the next move from this measured pose, not from the planned end.")
            print(msg)
            self.logger.info(PROGRESS + msg + ENDC)
        return out

    @staticmethod
    def grasp_slip_feedback(v):
        """Retry feedback for a guard verdict (touch/slip_guard.py), proprioception/vision only."""
        if v.get("mode") == "push":
            msg = (f"PUSH {'JAM' if v.get('check') == 'jam' else 'CONTACT LOST'} ({v.get('check')} check): "
                   f"{v.get('reason')}. The hand had moved {100 * (v.get('travel_m') or 0):.1f} cm from "
                   f"{v.get('grasp_pos')} along {v.get('direction')}; stopped at {v.get('slip_pos')}; last in "
                   f"contact at {v.get('last_held_pos')}; peak felt load {v.get('peak_load_N')} N.")
            msg += (" The part does not move this way: push along another direction / closer to its centre."
                    if v.get("check") == "jam" else
                    " Re-locate the part, put the hand back against it (lower, closer to its centre) and push again.")
            return msg + API._reacquired_text(v)
        msg = (f"GRASP {'JAM' if v.get('check') == 'jam' else 'SLIP'} ({v.get('check')} check): {v.get('reason')}. The hand had moved "
               f"{100 * (v.get('travel_m') or 0):.1f} cm from the grasp {v.get('grasp_pos')} along "
               f"{v.get('direction')}; stopped at {v.get('slip_pos')}; last held at {v.get('last_held_pos')}. "
               f"Finger width {v.get('width_at_grasp_mm')} -> {v.get('width_mm')} mm; peak felt load {v.get('peak_load_N')} N")
        if v.get("peak_load_pos"):
            msg += f" at {v['peak_load_pos']}"
        if v.get("track_dist_m") is not None:
            msg += f"; tracked '{v.get('track_name')}' {100 * v['track_dist_m']:.1f} cm further from the fingers than at the grasp"
        msg += (". The part is still held but the arm cannot move it this way." if v.get("check") == "jam"
                else ". The part is NOT held any more.")
        if (v.get("peak_load_N") or 0) >= 8.0:
            hint = ("It resisted this motion" + ("" if v.get("check") == "jam" else " before slipping") +
                    ": the commanded path was not the part's free path "
                    "(e.g. the wrong hinge side, or a straight pull where the part swings on an arc). Re-grasp, "
                    "then move a few cm the other way / along the arc and check the hand still holds it; or "
                    "use pull_with_touch, which lets the mechanism steer the hand.")
        else:
            hint = ("It slid out with little resistance: the grasp was shallow or on the wrong part (e.g. the "
                    "lever's hub instead of the bar). Re-detect it and grasp deeper, at the middle of the bar.")
        return msg + " " + hint + API._reacquired_text(v)

    @staticmethod
    def _reacquired_text(v):
        r = v.get("reacquired")
        if not r:
            return ""
        if r.get("lost") or r.get("part_pos") is None:
            return f" Camera tracking lost '{r.get('part')}' after the trip: re-detect it."
        return (f" After a {v.get('settle_s')} s settle the camera tracks '{r['part']}' at {r['part_pos']} "
                f"({'trusted' if r.get('trusted') else 'UNTRUSTED - prefer last_held_pos or re-detect'}).")

    def guard_motion(self, mode="hold", part=None, on_trip="stop", max_force=None, settle_s=None):
        """Configure the contact guard that watches every following trajectory (touch/slip_guard.py).

        ``mode``: "hold" (default; arms after a close that holds something: width, force, tracking
        checks), "push" (arms now, without a grasp: contact lost / jam while pushing), "off".
        ``part``: tracked object name (track_objects) to watch / re-acquire. ``on_trip``: "stop"
        (default; the code stops and the attempt fails with feedback) or "return" (the trajectory
        stops, the code continues, and ``execute_trajectory`` returns ``motion["guard"]`` = verdict
        with check, reason, last_held_pos, slip_pos, travel_m, and ``reacquired`` = where tracking
        sees ``part`` after a ~0.5 s settle). ``max_force`` (N): jam threshold (default 250).
        Resets to hold/stop at the next subtask. Returns the active spec."""
        spec = {"mode": mode, "part": part, "on_trip": on_trip, "max_force": max_force,
                "settle_s": float(settle_s if settle_s is not None else
                                  (getattr(config, "slip_guard_settle_s", 0.5) if on_trip == "return" else 0.0))}
        self.main_connection.send([config.SET_MOTION_GUARD, spec])
        resp = self.main_connection.recv()
        try:
            self.logger.info(resp[0])
        except Exception:
            pass
        if not isinstance(resp, list) or len(resp) < 2 or resp[1] is None:
            raise ValueError(f"guard_motion failed: {resp[0] if isinstance(resp, list) and resp else resp}")
        print(f"guard_motion: {resp[1]}")
        return resp[1]

    def estimate_face_normal(self, point, radius=0.06):
        """Outward normal of the vertical device face around ``point`` (world [x, y, z], e.g. a port
        located by the wrist camera) from head depth (touch/face_normal.py). The insertion axis into
        that face is ``axis`` = -normal - a device may stand at any heading, so never assume a world
        axis. Returns {"normal", "axis", "along", "point" (on the face), "n_inliers", "rms_m"} or None
        (face not visible from the head camera: estimate it by two touches, face_normal_from_touches)."""
        from touch.door_plane import depth_to_world
        from touch.face_normal import fit_vertical_face
        _, depth = self._capture_head_image_and_depth()
        pts = depth_to_world(depth, self.head_camera_position, self.head_camera_orientation_q,
                             self.cam_info, robot_mask=self._load_head_robot_mask())
        res = fit_vertical_face(pts, point, cam_xy=np.asarray(self.head_camera_position, float)[:2], r=radius)
        self.logger.info(PROGRESS + f"estimate_face_normal(point={np.round(np.asarray(point, float), 4).tolist()}): "
                         f"{res}" + ENDC)
        print(f"estimate_face_normal: {res}")
        return res

    def estimate_door_plane(self, handle):
        """Door face, pull direction and hinge side from head depth (touch/door_plane.py).

        ``handle`` = world [x, y, z] of the door handle (any point on it; the rose where it meets the
        door is located from depth). Returns the plane dict (``pull_dir``, ``hinge``, ``free_edge``, ``rose``, ``side_ratio``,
        ``confident``, ``lever_grasp`` = {``xyz``, ``rotation``} mid-lever top-down grasp ...) or None (no vertical face visible - clear occluders first). The plane is
        remembered: a following ``guarded_pull`` re-checks the door angle from depth."""
        from touch.door_plane import depth_to_world, estimate_door_plane
        _, depth = self._capture_head_image_and_depth()
        pts = depth_to_world(depth, self.head_camera_position, self.head_camera_orientation_q,
                             self.cam_info, robot_mask=self._load_head_robot_mask())
        h = np.asarray(handle, float).ravel()[:3]
        ref = getattr(self, "_door_plane", None)
        if ref:
            from touch.door_plane import measure_door_angle
            deg, n = measure_door_angle(pts, ref[0], ref[1])
            if deg is not None and deg > 10.0:
                # door:hidden r6: a re-estimate on the OPEN door replaced the closed-face reference and every
                # later door angle read None / 0 deg while the door was at 72 deg. Keep the closed face.
                plane = dict(ref[0], door_open_deg=deg)
                self.logger.info(PROGRESS + f"estimate_door_plane: the door is already open {deg:.0f} deg; "
                                 f"returning the closed-door plane measured before ({n} panel points)" + ENDC)
                print(f"estimate_door_plane: door already open {deg:.0f} deg; closed-door plane: {plane}")
                return plane
        plane = estimate_door_plane(pts, h, np.asarray(self.head_camera_position, float)[:2])
        if ref and plane:
            # door:hidden r8: re-estimates with the arm in view stopped the panel at the arm's shadow
            # (width 0.46 -> 0.26 / 0.36 m) and moved the hinge 10-20 cm in; the push then missed.
            # Occlusion only shrinks the panel, so keep the wider face when the normal agrees.
            same = abs(float(np.dot(plane["pull_dir"][:2], ref[0]["pull_dir"][:2]))) > 0.99
            if same and plane["width_m"] < ref[0]["width_m"] - 0.03:
                self.logger.info(PROGRESS + f"estimate_door_plane: new panel {plane['width_m']} m narrower than "
                                 f"the earlier {ref[0]['width_m']} m (occluded); keeping the earlier plane" + ENDC)
                plane = dict(ref[0])
            elif same and ref[0].get("hinge_source") == "swung_panel":
                # the hinge was MEASURED (the panel moved); a fresh look only has the panel-end hypothesis
                plane = dict(plane, hinge=ref[0]["hinge"], free_edge=ref[0]["free_edge"], hinge_source="swung_panel")
        self.logger.info(PROGRESS + f"estimate_door_plane(handle={np.round(h, 3).tolist()}): {plane} "
                         f"({len(pts)} depth points)" + ENDC)
        print(f"estimate_door_plane: {plane}")
        self._door_plane = (plane, float(h[2])) if plane else ref
        return plane

    def measure_door_angle(self):
        """Door opening in deg about the hinge of the last ``estimate_door_plane``, from a new head
        depth capture (None when unknown)."""
        if not getattr(self, "_door_plane", None):
            return None
        from touch.door_plane import depth_to_world, measure_door_angle
        plane, z = self._door_plane
        _, depth = self._capture_head_image_and_depth()
        pts = depth_to_world(depth, self.head_camera_position, self.head_camera_orientation_q,
                             self.cam_info, robot_mask=self._load_head_robot_mask())
        deg, n = measure_door_angle(pts, plane, z)
        self.logger.info(PROGRESS + f"measure_door_angle: {deg} deg ({n} points on the panel)" + ENDC)
        return deg

    def grasp_door_lever(self, door_deg=0.0):
        """Top-down grasp at the depth-measured lever middle of the last ``estimate_door_plane`` (fingers across
        the bar, +-2 cm retries, lever pressed 3 cm). ``door_deg``: the door's current opening (the lever has
        swung with it). Returns {"holding", "xyz", "rotation", "width", "tries"}."""
        from helpers.door_open import grasp_door_lever
        ref = getattr(self, "_door_plane", None)
        if not ref:
            raise RuntimeError("grasp_door_lever: call estimate_door_plane(handle_xyz) first")
        res = grasp_door_lever(self, ref[0], door_deg=door_deg)
        print(f"grasp_door_lever: {res}")
        return res

    def find_hinge_from_depth(self, hand_xy=None):
        """Hinge of the last ``estimate_door_plane`` measured from the SWUNG panel in a new head depth capture
        (touch.door_plane.hinge_from_swung_panel; the door must be a few deg open). Updates the stored plane
        (hinge, free_edge, hinge_source="swung_panel") so measure_door_angle / grasps use it. None if not seen."""
        if not getattr(self, "_door_plane", None):
            return None
        from touch.door_plane import depth_to_world, hinge_from_swung_panel
        plane, z = self._door_plane
        _, depth = self._capture_head_image_and_depth()
        pts = depth_to_world(depth, self.head_camera_position, self.head_camera_orientation_q,
                             self.cam_info, robot_mask=self._load_head_robot_mask())
        found = hinge_from_swung_panel(pts, plane, z, hand_xy=hand_xy)
        self.logger.info(PROGRESS + f"find_hinge_from_depth: {found} (hypothesis was {plane.get('hinge')})" + ENDC)
        if found:
            self._door_plane = (dict(plane, hinge=found["hinge"], free_edge=found["free_edge"],
                                     hinge_source="swung_panel"), z)
        return found

    def probe_door_hinge(self, pull_m=0.10):
        """With the door lever held (after grasp_door_lever): pull ``pull_m`` straight out along the door
        normal, then measure the hinge from the swung panel in depth (no hinge-side guess needed).
        Returns {"hinge", "free_edge", "open_deg", "flipped", "pulled_m", "source", "holding"}; pass
        guarded_pull(direction=p["pull_dir"], hinge=r["hinge"], start_deg=r["open_deg"]) next."""
        from helpers.door_open import probe_hinge
        ref = getattr(self, "_door_plane", None)
        if not ref:
            raise RuntimeError("probe_door_hinge: call estimate_door_plane(handle_xyz) first")
        res = probe_hinge(self, ref[0], pull_m=pull_m)
        print(f"probe_door_hinge: {res}")
        self._record_touch("probe_door_hinge", {k: res.get(k) for k in ("source", "open_deg", "pulled_m", "flipped")})
        return res

    def open_door(self, handle, goal_deg=65.0, max_pulls=3):
        """One call that opens a hinged door wide from depth + proprioception (helpers/door_open.py):
        estimate_door_plane -> grasp_door_lever -> guarded_pull -> measure_door_angle, re-grasping and pulling
        again up to ``max_pulls`` times until the door is >= ``goal_deg``; releases and lifts at the end.
        Returns {"outcome": "opened" | "partly_open" | "grasp_failed" | "no_door_face", "door_deg", "pulls"}."""
        from helpers.door_open import open_door
        res = open_door(self, handle, goal_deg=goal_deg, max_pulls=max_pulls)
        out = {k: v for k, v in res.items() if k != "plane"}
        print(f"open_door: {out}")
        self._record_touch("open_door", {"outcome": res["outcome"], "door_deg": res["door_deg"],
                                         "pulls": len(res["pulls"])})
        return out

    def guarded_pull(self, direction, hinge=None, travel=0.55, max_deg=80.0, part=None, **kw):
        """Pull the held handle in small guarded steps with automatic correction (helpers/guarded_pull.py):
        wrong hinge side -> mirror + retry, slip -> re-grasp at the re-acquired / last-held point,
        late jam -> end stop. Returns {"outcome", "travel_m", "swept_deg", "hinge", "flips", "regrasps", "events"}.
        After ``estimate_door_plane`` the result also has ``vision_deg`` (door angle from depth); an
        "opened" claim the depth contradicts becomes ``"swung_shut"``."""
        from helpers.guarded_pull import guarded_pull
        res = guarded_pull(self, direction, hinge=hinge, travel=travel, max_deg=max_deg, part=part, **kw)
        if getattr(self, "_door_plane", None):
            try:
                deg = self.measure_door_angle()
                res["vision_deg"] = deg
                if deg is not None and res.get("outcome") == "opened" and deg < 0.5 * float(res.get("swept_deg") or 0):
                    # harness door_vis3/5: the panel swung back shut behind the hand; the hand path can't see it
                    msg = (f"depth shows the door at {deg:.0f} deg, not the {res['swept_deg']:.0f} deg the hand "
                           f"swept: it swung back shut")
                    res["events"] = list(res.get("events") or []) + [msg]
                    res["outcome"] = "swung_shut"
                    self.logger.info(FAIL + "guarded_pull: " + msg + ENDC)
                elif deg is not None and res.get("outcome") == "opened" and deg < 55.0:
                    # door:hidden r6: the hand swept 61 deg, the panel was at 38 (the push lags the hand)
                    res["events"] = list(res.get("events") or []) + [
                        f"depth shows the door at {deg:.0f} deg (< 60): only partly open"]
                    res["outcome"] = "partly_open"
            except Exception as e:
                self.logger.info(PROGRESS + f"guarded_pull: door-angle check failed: {e}" + ENDC)
        print(f"guarded_pull: {({k: v for k, v in res.items() if k != 'events'})}")
        self._record_touch("guarded_pull", res)
        self._budget_tick("guarded_pull")
        return res

    def _handle_grasp_slip(self, verdict):
        """The grasp-slip guard stopped the trajectory: fail the attempt with detailed feedback,
        the same path as a tracking abort (remaining code skipped, retry prompt + reviewer told)."""
        text = self.grasp_slip_feedback(verdict)
        self.logger.info(FAIL + text + ENDC)
        if verdict.get("check") == "jam" and (verdict.get("travel_m") or 0) >= 0.15:
            # Moved far, then stuck: likely the part's end stop (door fully open). Stop the code
            # but let the reviewer judge the result instead of failing the attempt.
            raise RuntimeError("Trajectory stopped by the grasp-slip guard: the held part stopped moving after "
                               f"{100 * verdict['travel_m']:.0f} cm - probably its end stop (e.g. fully open). "
                               "Check whether the goal is already met before retrying. " + text)
        try:
            self.task.failed_task = True
            self.task.review_reason = text
            steps = self.task.review_improvement_steps or []
            self.task.review_improvement_steps = list(steps) + [text]
        except Exception:
            pass
        raise RuntimeError("Trajectory stopped by the grasp-slip guard. " + text)

    def _handle_tracking_abort(self, payload):
        """A tracking monitor stopped the rollout: fail the attempt and tell the model why.

        The reason is raised as an exception so it travels the existing error-correction
        path in ``agent_runner.execute_python_blocks`` (remaining code blocks are skipped
        and the message is fed back), and it is also recorded on the task so the retry
        prompt and the reviewer both see it.
        """
        reason = payload.get("reason") or "tracking monitor aborted the trajectory"
        summary = payload.get("summary") or {}
        self.tracking_summary = summary
        self.logger.info(FAIL + f"Tracking monitor aborted the trajectory: {reason}" + ENDC)
        try:
            self.task.failed_task = True
            self.task.review_reason = f"Tracking monitor: {reason}"
            steps = self.task.review_improvement_steps or []
            self.task.review_improvement_steps = list(steps) + [
                "The tracked object left the gripper during the trajectory. Re-detect the "
                "object, re-plan the grasp (deeper/narrower approach) and retry."
            ]
        except Exception:
            pass
        raise RuntimeError(
            f"Trajectory aborted by the tracking monitor: {reason}. "
            "The object is no longer where the plan assumed - re-detect it before continuing."
        )

    def track_objects(self, targets, monitor=None, track_gripper=True):
        """Start tracking objects in 3D during the following trajectory executions.

        ``targets``: list of ``{"name": str, "world_points": [[x, y, z], ...]}`` - the 3D
        points (from ``detect_object`` / affordance points) that identify the object.
        Three to eight well-spread points on the object give the best robustness: they are
        tracked independently in each camera and fused with outlier rejection.

        ``monitor``: an optional ``monitor(state) -> dict`` invariant evaluated on every
        tracked frame. Returning ``{"status": "abort", "reason": ...}`` stops the
        trajectory immediately and fails the sub-task; ``{"status": "record", ...}`` flags
        the frames for the reviewer. When omitted, ``tracking.monitors.default_monitor``
        is used (object must stay attached to the gripper once grasped).

        Returns ``None``; prints which cameras were seeded, as the other tools do.
        """
        if not self.args.tracking:
            self.logger.info(PROGRESS + "track_objects: tracking is disabled (--tracking is off); ignoring." + ENDC)
            return

        spec_targets = []
        for target in (targets or []):
            if isinstance(target, str):
                raise ValueError("track_objects targets must be dicts with 'name' and 'world_points'")
            points = np.asarray(target["world_points"], dtype=float).reshape(-1, 3)
            spec_targets.append({"name": str(target["name"]), "world_points": points.tolist()})
        if not spec_targets:
            raise ValueError("track_objects needs at least one target")

        payload = {
            "targets": spec_targets,
            "monitor": self._monitor_spec(monitor),
            "track_gripper": bool(track_gripper),
            "provider": self.args.tracker_provider,
            "tracker3d": getattr(self.args, "tracker3d", None),
            "cameras": getattr(self.args, "track_cameras", None),
            "interval": self.args.track_interval,
            "camera_fps": getattr(self.args, "track_camera_fps", None),
            "latency": getattr(self.args, "track_latency", None),
            "roma": bool(getattr(self.args, "track_roma", False)),
            "output_dir": self.args.track_log_dir,
            "save_depth": bool(self.args.track_save_depth),
        }
        self.logger.info(PROGRESS + f"Starting tracking for {[t['name'] for t in spec_targets]}..." + ENDC)
        self.main_connection.send([config.START_TRACKING, payload])
        try:
            resp = self.main_connection.recv()
            message = resp[0] if isinstance(resp, list) and resp else str(resp)
            self.logger.info(message)
            self.tracking_active = bool(isinstance(resp, list) and len(resp) > 1
                                        and isinstance(resp[1], dict) and resp[1].get("ok"))
        except Exception as e:
            self.logger.info(FAIL + f"track_objects: no response from the environment: {e}" + ENDC)
            self.tracking_active = False
            return
        if self.tracking_active:
            # seed points per target: carried by the tracked pose to rebuild the object's
            # points for the reviewer's tracking evidence (see _tracking_review_evidence)
            self.tracking_targets = {t["name"]: t["world_points"] for t in spec_targets}
        for target in spec_targets:
            print(f"Tracking '{target['name']}' with {len(target['world_points'])} points.")

    @staticmethod
    def _monitor_spec(monitor):
        """Turn a monitor callable into something that survives the IPC hop.

        The simulator runs in a separate process, so a closure cannot be sent; its source
        is shipped instead and recompiled there (``tracking.monitor.compile_monitor``).
        Built-in monitors are referenced by name so no source is needed.
        """
        if monitor is None:
            return None
        if isinstance(monitor, (str, dict)):
            return monitor
        builtin_module = "tracking.monitors"
        spec = getattr(monitor, "spec", None)
        if spec is not None:
            return spec
        if getattr(monitor, "__module__", None) == builtin_module:
            return {"builtin": monitor.__name__}
        import inspect
        try:
            source = inspect.getsource(monitor)
        except (OSError, TypeError) as e:
            raise ValueError(
                "track_objects: could not read the monitor's source "
                f"({e}). Define the monitor as a module-level function, pass its source "
                "as a string, or use a built-in from tracking.monitors."
            )
        return {"source": inspect.cleandoc(source) if source.startswith(" ") else source,
                "name": getattr(monitor, "__name__", None)}

    def get_tracking_report(self, include_frames=False):
        """Fetch the tracking summary for the rollout so far (also used by the reviewer)."""
        if not getattr(self, "tracking_active", False):
            return None
        self.main_connection.send([config.GET_TRACKING_REPORT, bool(include_frames)])
        try:
            resp = self.main_connection.recv()
        except Exception as e:
            self.logger.info(PROGRESS + f"Could not read the tracking report: {e}" + ENDC)
            return None
        if not (isinstance(resp, list) and len(resp) > 1) or resp[1] is None:
            return None
        summary = resp[1]
        self.tracking_summary = summary
        return summary

    def stop_tracking(self):
        """End the tracking session and return its summary (safe to call when inactive)."""
        if not getattr(self, "tracking_active", False):
            return None
        self.main_connection.send([config.STOP_TRACKING])
        summary = None
        try:
            resp = self.main_connection.recv()
            if isinstance(resp, list) and len(resp) > 1:
                summary = resp[1]
        except Exception as e:
            self.logger.info(PROGRESS + f"Could not stop tracking cleanly: {e}" + ENDC)
        self.tracking_active = False
        if summary:
            self.tracking_summary = summary
        return summary




    def generate_linear_trajectory(self, desc, start_pose, end_pose, num_points=20):
        """Return a linear end-effector trajectory and log the call.

        Accepts either pose length (see ``common_utils`` for the format contract):
        length-4 ``[x, y, z, rotation]`` (top-down) or length-6
        ``[x, y, z, roll, pitch, yaw]``. Mixing the two is allowed - a length-4 endpoint is
        promoted to the length-6 orientation it already stands for - so a top-down hover can
        be chained into a side approach.

        Position is interpolated linearly. Orientation is interpolated component-wise on the
        rotation value in the pure top-down case (unchanged legacy behaviour, bit for bit),
        and by quaternion slerp as soon as any endpoint is length-6: lerping Euler triples
        that differ on more than one axis swings the gripper through orientations neither
        endpoint asked for.

        Logs: desc, start/end poses, and 2D head-camera projection of end pose.
        """
        if not isinstance(start_pose, (list, tuple)) or not isinstance(end_pose, (list, tuple)):
            raise ValueError("start_pose and end_pose must be list/tuple of length 4 or 6")
        if len(start_pose) not in (4, 6) or len(end_pose) not in (4, 6):
            raise ValueError(
                "start_pose and end_pose must be length 4 [x,y,z,rotation] or length 6 "
                "[x,y,z,roll,pitch,yaw]; got %d and %d" % (len(start_pose), len(end_pose))
            )
        if int(num_points) < 2:
            raise ValueError("num_points must be >= 2")
        try:
            start_pose = [float(v) for v in start_pose]
            end_pose = [float(v) for v in end_pose]
        except Exception as e:
            raise ValueError("Invalid pose values: %s" % e)

        sx, sy, sz = start_pose[:3]
        ex, ey, ez = end_pose[:3]

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

        self.logger.info(PROGRESS + "generate_linear_trajectory desc='" + str(desc) + "' start=" + str(start_pose) + " end=" + str(end_pose) + " end_px=" + str(end_px) + ENDC)

        n = int(num_points)
        traj = []

        if len(start_pose) == 4 and len(end_pose) == 4:
            s_theta, e_theta = start_pose[3], end_pose[3]
            for i in range(n):
                t = i / (n - 1)
                traj.append([
                    sx + (ex - sx) * t,
                    sy + (ey - sy) * t,
                    sz + (ez - sz) * t,
                    s_theta + (e_theta - s_theta) * t,
                ])
            return Trajectory(traj, desc)

        s_quat = transforms.quat_from_euler(pose_euler(start_pose))
        e_quat = transforms.quat_from_euler(pose_euler(end_pose))
        for i in range(n):
            t = i / (n - 1)
            euler = transforms.euler_from_matrix(
                transforms.matrix_from_quat(transforms.slerp(s_quat, e_quat, t))
            )
            traj.append([
                sx + (ex - sx) * t,
                sy + (ey - sy) * t,
                sz + (ez - sz) * t,
                euler[0], euler[1], euler[2],
            ])
        return Trajectory(traj, desc)
    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])
        self._gripper_closed = False



    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])
        self._gripper_closed = True
        self._record_grasp()
        self._budget_tick("close_gripper")

    def _record_grasp(self):
        """Remember where the hand closed (+ a closed-hand top touch of the part just before) so a
        LATER subtask can recover the in-hand height of the part (get_grasp_info)."""
        try:
            self.main_connection.send([config.GET_ROBOT_STATE, {}])
            eef = (self.main_connection.recv() or {}).get("eef_pos")
        except Exception:
            return
        if eef is None:
            return
        eef = [round(float(v), 4) for v in eef]
        top = getattr(self, "_last_top_touch", None)
        near = top is not None and np.linalg.norm(np.subtract(top["xy"], eef[:2])) < 0.03
        self.grasp_info = {"grasp_eef_pos": eef, "touched_top_z": top["z"] if near else None}
        if near:
            self.grasp_info["grasp_depth_below_top"] = round(top["z"] - eef[2], 4)
        self.logger.info(PROGRESS + f"[grasp] recorded {self.grasp_info}" + ENDC)

    def get_grasp_info(self):
        """Measured facts about the current grasp, kept across subtasks: ``grasp_eef_pos`` (fingertip
        centre when the gripper closed), ``touched_top_z`` (part top measured by a closed-hand
        ``move_until_contact([0, 0, -1])`` within 3 cm just before, else None) and, with a touch,
        ``grasp_depth_below_top`` (= touched_top_z - grasp z; check it against part height / 2 right
        after closing). None if nothing recorded.
        In-hand height of the part centre = touched_top_z - part height / 2 - grasp_eef_pos[2]."""
        info = getattr(self, "grasp_info", None)
        print(f"get_grasp_info: {info}")
        return info


    def _tracking_review_evidence(self, start_idx):
        """Measured pull facts for this attempt -> ``(facts, prompt_text)`` or ``(None, None)``.

        Only for articulated pull/open tasks with an active tracking session (the agent
        called ``track_objects``). Nothing about the hinge is assumed: the opening direction
        comes from the handle's own tracked motion, or from the door front's measured normal
        when the door did not move (see ``tracking.review_evidence.summarize_pull``).
        """
        from tracking.review_evidence import ARTICULATED_TASK_RE, evidence_from_tracking_frames
        if not getattr(config, "review_tracking_evidence", True):
            return None, None
        targets = getattr(self, "tracking_targets", None) or {}
        if not getattr(self, "tracking_active", False) or not targets:
            return None, None
        name = next((n for n in targets if ARTICULATED_TASK_RE.search(n)), None)
        if name is None and ARTICULATED_TASK_RE.search(str(self.task.command or "")):
            name = next(iter(targets))
        if name is None:
            self.logger.info(PROGRESS + f"Review: tracking evidence skipped (not an articulated "
                             f"pull task; targets={list(targets)})" + ENDC)
            return None, None
        try:
            report = self.get_tracking_report(include_frames=True) or {}
            out = evidence_from_tracking_frames(
                report.get("frames") or [], targets[name], name, start_step=start_idx,
                logger=lambda m: self.logger.info(PROGRESS + m + ENDC))
        except Exception as e:                  # evidence is optional; never break the review
            self.logger.info(WARNING + f"Review: tracking evidence failed: {type(e).__name__}: {e}" + ENDC)
            return None, None
        if out is None:
            return None, None
        facts, text = out
        self.logger.info(PROGRESS + f"Review: tracking evidence for '{name}': flags={facts['flags']} "
                         f"angle={facts['angle_deg']} rec={facts.get('recommended_pull_dir')}" + ENDC)
        return facts, text

    @staticmethod
    def _carried_since_close(state):
        """How far the hand - and so the held object - moved since the gripper closed, or None.

        Joint encoders measure this exactly; the VLM, judging a 256 px clip, has called a
        kettle lifted 15 cm "still on the stovetop, the arm never moved".
        """
        at_close, now = state.get("eef_pos_at_close"), state.get("eef_pos")
        if not at_close or not now:
            return None
        dz = (float(now[2]) - float(at_close[2])) * 100
        dxy = math.hypot(float(now[0]) - float(at_close[0]), float(now[1]) - float(at_close[1])) * 100
        if abs(dz) < 2 and dxy < 2:
            return ("the hand has NOT moved since the fingers closed on it (measured by the arm's joint "
                    "encoders): the held object is still where it was grasped.")
        direction = "UP" if dz > 0 else "DOWN"
        return (f"the fingers closed on it with the hand at z={float(at_close[2]):.3f} m; the hand is now at "
                f"z={float(now[2]):.3f} m (measured by the arm's joint encoders), so the held object has been "
                f"carried {direction} by ~{abs(dz):.0f} cm and {dxy:.0f} cm horizontally since it was grasped. "
                "Judge from the clips whether that motion achieved the task, not whether the arm moved.")

    @staticmethod
    def _tilt_while_held(state):
        """Sentence on the largest measured hand tilt while holding (a pour), or None if it stayed ~upright."""
        tilt = state.get("max_tilt_since_close")
        if not tilt or tilt.get("deg") is None or float(tilt["deg"]) < 15:
            return None
        dip = tilt.get("dip_dir")
        side = ""
        if dip:
            dx, dy = float(dip[0]), float(dip[1])
            axis = (("+x" if dx > 0 else "-x") if abs(dx) >= abs(dy) else ("+y" if dy > 0 else "-y"))
            side = (f", lowering its world {axis} side (direction [{dx:+.2f}, {dy:+.2f}]): the held object was "
                    f"tipped toward {axis}")
        return (f"while holding it, the hand was tilted up to ~{float(tilt['deg']):.0f} deg from upright "
                f"(measured by the arm's joint encoders){side}. Use the clips to judge whether the tipped "
                "object's opening/spout was over the target.")

    def _gripper_review_evidence(self):
        """End-of-attempt gripper proprioception for the reviewer, or None.

        A real gripper reports its finger opening; "closed but stalled short of fully
        closed" means an object is between the fingers. Small low-res clips mislead the
        VLM both ways (a lifted kettle judged "still on the stove", or an empty gripper
        judged holding), so this measured fact is stated explicitly.
        """
        try:
            self.main_connection.send([config.GET_ROBOT_STATE, {}])
            state = self.main_connection.recv() or {}
        except Exception as e:
            self.logger.info(WARNING + f"Review: gripper state query failed: {e}" + ENDC)
            return None
        width = state.get("gripper_width")
        if width is None or state.get("gripper_open") is None:
            return None
        width_cm = float(width) * 100
        if state.get("gripper_open"):
            verdict = f"commanded OPEN, finger opening {width_cm:.1f} cm: the gripper is not holding anything."
        elif state.get("gripper_holding"):
            verdict = (f"commanded CLOSED, but the fingers stalled at {width_cm:.1f} cm (fully closed is ~0.1 cm): "
                       "they are blocked by an object, so the gripper IS HOLDING something between its fingers "
                       "right now. Use the clips to judge WHICH object it is and where it is - do not conclude "
                       "that the gripper is empty or that the grasp missed.")
            carried = self._carried_since_close(state)
            if carried:
                verdict += "\n  - " + carried
            tilted = self._tilt_while_held(state)
            if tilted:
                verdict += "\n  - " + tilted
        else:
            verdict = (f"commanded CLOSED and the fingers are fully closed ({width_cm:.1f} cm): NOTHING is held "
                       "between them (the grasp missed, or the object slipped out).")
        self.logger.info(PROGRESS + f"Review: gripper evidence -> {verdict}" + ENDC)
        return ("GRIPPER PROPRIOCEPTION (measured by the gripper's width sensor at the END of this attempt - a "
                "direct physical measurement, more reliable than small details in the low-resolution clips):\n"
                "  - " + verdict)

    def _pull_proprio_review_evidence(self, start_idx, camera_moved=()):
        """Measured grasp-and-pull facts of this attempt -> ``(facts, prompt_text)`` or ``(None, None)``.

        Built from the arm's own per-trajectory log (commanded vs reached hand motion,
        finger width) - what a real arm reports; no scene/joint state of the sim is used.
        """
        from helpers.pull_proprio import summarize_pull_proprio, format_pull_proprio, is_articulated_task
        if not getattr(config, "review_pull_proprio", True):
            return None, None
        if not is_articulated_task(self.task.command):
            # handle/door wording; pick/insert tasks get the gripper-width evidence instead
            self.logger.info(PROGRESS + "Review: pull proprioception skipped (not an open/close/pull task)" + ENDC)
            return None, None
        records = [r for r in (self.motion_log or []) if int(r.get("step_start", 0)) >= int(start_idx)]
        try:
            facts = summarize_pull_proprio(records, logger=lambda m: self.logger.info(PROGRESS + m + ENDC))
        except Exception as e:                  # evidence is optional; never break the review
            self.logger.info(WARNING + f"Review: pull proprioception failed: {type(e).__name__}: {e}" + ENDC)
            return None, None
        if facts is None:
            self.logger.info(PROGRESS + f"Review: no pull proprioception ({len(records)} trajectories, "
                             "no closed-gripper motion or no width sensor)" + ENDC)
            return None, None
        text = format_pull_proprio(facts, camera_moved=camera_moved)
        self.logger.info(PROGRESS + f"Review: pull proprioception -> outcome={facts['outcome']} "
                         f"max_carried={facts['max_carried_m']} pulls={len(facts['pulls'])}" + ENDC)
        return facts, text

    def _door_angle_review_evidence(self):
        """Depth-measured door angle at review time (after ``estimate_door_plane``) -> prompt text or None.

        door:hidden r6: the door was at 72 deg, but the reviewer failed the attempt because "the final
        door angle could not be measured" from the frames."""
        if not getattr(self, "_door_plane", None):
            return None
        try:
            deg = self.measure_door_angle()
        except Exception as e:                  # evidence is optional; never break the review
            self.logger.info(WARNING + f"Review: door angle evidence failed: {type(e).__name__}: {e}" + ENDC)
            return None
        self.logger.info(PROGRESS + f"Review: door angle evidence -> {deg}" + ENDC)
        if deg is None:
            return ("DOOR ANGLE (head depth camera): not measurable now (too few panel points visible); "
                    "judge from the frames.")
        return (f"DOOR ANGLE (head depth camera, measured now about the hinge seen before the door "
                f"opened; about +-5 deg): the door is open {deg:.0f} deg (60 deg or more = wide open).")

    def _apply_pull_proprio_verdict(self, facts, success, reason, improvement_steps, camera_moved=()):
        """Measured-veto + retry direction bookkeeping -> (success, reason, improvement_steps).

        ``camera_moved``: labels the head camera measured as having left their place; then the
        part DID move (pushed, not carried), so neither the veto nor "wrong direction" applies.
        """
        from helpers.pull_proprio import measured_no_motion_veto, retry_direction_hint
        if not facts:
            return success, reason, improvement_steps
        if camera_moved and facts["outcome"] == "part_did_not_move":
            self.logger.info(PROGRESS + f"Review: pull proprioception says not carried, but camera saw "
                             f"{camera_moved} leave their place -> no veto / direction change" + ENDC)
            self.task.retry_direction_hint = None
            return success, reason, improvement_steps
        veto = measured_no_motion_veto(facts, self.task.command)
        if success and veto:
            self.logger.info(FAIL + "Review: success verdict OVERRULED by measured pull proprioception: "
                             + veto + ENDC)
            success = False
            reason = veto + (" (The VLM reviewer had judged success: " + str(reason) + ")"
                             if reason else " (final attempt, accepted without VLM review)")
        hint = None
        if not success:
            hint = retry_direction_hint(facts, self.task.failed_pull_dirs)
        if facts["outcome"] == "part_did_not_move":
            self.task.failed_pull_dirs = list(self.task.failed_pull_dirs) + list(facts["pull_dirs"])
        self.task.retry_direction_hint = hint
        if hint:
            self.logger.info(PROGRESS + f"Review: retry direction recommendation -> {hint}" + ENDC)
            if isinstance(improvement_steps, list):
                improvement_steps = [hint] + improvement_steps
            else:
                improvement_steps = (hint + "\n" + str(improvement_steps or "")).strip()
        return success, reason, improvement_steps

    def run_vlm_review(self):
        """Ask a VLM to judge whether the attempt achieved the task goal.

        Visual evidence is EITHER the per-attempt trajectory videos (head + wrist mp4,
        preferred - see models.model_supports_video) OR, for review models without video
        input (Azure OpenAI, Claude on Bedrock), the legacy subsampled key frames.
        Expects strict JSON: { "success": true/false, "reasoning": "...", "improvement_steps": [...] }.
        Does not mutate the environment.
        """
        from prompts.review_prompt import (
            REVIEW_PROMPT, REVIEW_SCENE_ANALYSIS_SECTION,
            REVIEW_MEDIA_FRAMES_SECTION, REVIEW_MEDIA_VIDEO_SECTION,
            REMAINING_MEDIA_SENTENCE_FRAMES, REMAINING_MEDIA_SENTENCE_VIDEO,
            REVIEW_OVERALL_COMMAND_SECTION,
        )
        start_idx = self.task.start_attempt_trajectory_step
        # Determine review model: "vlm" uses main model, "vlm:<model>" uses specified model
        review_provider = self.args.review_provider
        if review_provider.startswith("vlm:"):
            review_model = review_provider[len("vlm:"):]
        else:
            review_model = self.args.language_model

        # Attempt clips built by task_completed() (head, then wrist).
        video_paths = [p for p in (self.task.review_clips or []) if p and os.path.exists(p)]
        use_video = bool(video_paths) and models.model_supports_video(review_model)

        # Key frames (image-only review models): change-aware selection that always keeps the
        # first and FINAL frame of each camera (helpers/review_frames.py).
        from helpers.review_frames import select_key_frames, frames_listing
        head_frames = select_key_frames(config.trajectory_folder, config.trajectory_image_base, start_idx,
                                        getattr(config, "review_key_frames_head", 24))
        wrist_frames = select_key_frames(config.trajectory_folder, config.trajectory_wrist_image_base, start_idx,
                                         getattr(config, "review_key_frames_wrist", 14))
        frame_paths = [p for _, p in head_frames] + [p for _, p in wrist_frames]

        # No trajectory frames/clips since the attempt started => the robot never moved.
        # Reviewing on conversation text alone lets the VLM hallucinate a success
        # (it "sees" frames described in the plan that do not exist) - fail hard instead.
        if not frame_paths and not video_paths:
            reason = (f"No trajectory frames were captured for this attempt (start_idx={start_idx}, "
                      f"trajectory_step={self.trajectory_step}): the robot did not move, so none of the "
                      "planned trajectories were actually executed.")
            self.logger.info(FAIL + f"Review: success=False (no frames) | task=\"{self.task.title()}\". {reason}" + ENDC)
            self.task.review_reason = reason
            self.task.review_improvement_steps = [
                "Re-emit and execute every planned trajectory: no execute_trajectory call reached the environment.",
                "Do not call task_completed() before the trajectories have actually run.",
            ]
            self.task.failed_task = True
            return
        
        # Start-of-attempt scene image (perception VLM input). Sent FIRST, before the
        # trajectory videos / key frames, and explicitly described in the prompt so the
        # reviewer never confuses it with the execution evidence.
        scene_image_path = self.task.scene_analysis_image_path
        scene_analysis_text = (self.task.scene_analysis or "").strip()
        has_scene_image = bool(scene_image_path and os.path.exists(scene_image_path))

        if use_video:
            frame_paths = []  # videos replace the key frames entirely
            cam_labels = ["head cam", "wrist cam"]
            video_list = "\n".join(
                f"  {i+1}. {cam_labels[i] if i < len(cam_labels) else 'cam'}: {p}"
                for i, p in enumerate(video_paths)
            )
            media_section = (
                REVIEW_MEDIA_VIDEO_SECTION
                .replace("[INSERT VIDEO LIST]", video_list)
                .replace("[INSERT VIDEO FPS]", str(config.trajectory_video_fps))
            )
            remaining_media_sentence = REMAINING_MEDIA_SENTENCE_VIDEO
        else:
            # GPT-5 in azure limits to 50 images in a request - test if it confuses the model (the cutoff of wrist images ....)
            max_frames = config.max_allowed_vlm_images - 1 - (1 if has_scene_image else 0)
            if len(frame_paths) > max_frames:
                # keep the final frames: drop evenly from the middle of the wrist list first
                self.logger.info(WARNING + f"Key frames {len(frame_paths)} > cap {max_frames}: thinning wrist frames" + ENDC)
                excess = len(frame_paths) - max_frames
                if excess < len(wrist_frames) - 1:
                    drop = set(np.linspace(1, len(wrist_frames) - 2, excess).round().astype(int).tolist())
                    wrist_frames = [f for j, f in enumerate(wrist_frames) if j not in drop]
                else:
                    wrist_frames = wrist_frames[-1:]
                    head_frames = head_frames[:max(1, max_frames - 2)] + head_frames[-1:]
                frame_paths = [p for _, p in head_frames] + [p for _, p in wrist_frames]
            media_section = REVIEW_MEDIA_FRAMES_SECTION.replace("[INSERT FRAME PATHS]",
                                                                frames_listing(head_frames, wrist_frames))
            remaining_media_sentence = REMAINING_MEDIA_SENTENCE_FRAMES
            if video_paths:
                self.logger.info(WARNING + f"Review model {review_model} has no video input - falling back to {len(frame_paths)} key frames." + ENDC)
            video_paths = []

        if has_scene_image or scene_analysis_text:
            scene_section = (
                REVIEW_SCENE_ANALYSIS_SECTION
                .replace("[INSERT SCENE ANALYSIS IMAGE PATH]", str(scene_image_path))
                .replace("[INSERT REMAINING MEDIA SENTENCE]", remaining_media_sentence)
                .replace("[INSERT SCENE ANALYSIS]", scene_analysis_text or "(unavailable)")
            )
        else:
            scene_section = ""

        tracking_facts, tracking_text = self._tracking_review_evidence(start_idx)
        if tracking_text:
            media_section = media_section + "\n\n" + tracking_text + "\n"
        gripper_text = self._gripper_review_evidence()
        if gripper_text:
            media_section = media_section + "\n\n" + gripper_text + "\n"
        disp_facts, disp_text = self._displacement_review_evidence()
        camera_moved = [f["label"] for f in disp_facts
                        if f.get("verdict") == "left_place" and not f.get("same_region_as")]
        pull_facts, pull_text = self._pull_proprio_review_evidence(start_idx, camera_moved)
        if pull_text:
            media_section = media_section + "\n\n" + pull_text + "\n"
        touch_text = self._touch_review_evidence(start_idx)
        if touch_text:
            media_section = media_section + "\n\n" + touch_text + "\n"
        door_text = self._door_angle_review_evidence()
        if door_text:
            media_section = media_section + "\n\n" + door_text + "\n"
        if disp_text:
            media_section = media_section + "\n\n" + disp_text + "\n"

        overall = str(getattr(self.task, "overall_command", None) or "").strip()
        overall_section = (REVIEW_OVERALL_COMMAND_SECTION.replace("[INSERT OVERALL COMMAND]", overall)
                           if overall and overall != str(self.task.command).strip() else "")
        # Build prompt with placeholders
        prompt = REVIEW_PROMPT.replace("[INSERT TASK]", str(self.task.command)) \
                              .replace("[INSERT OVERALL COMMAND SECTION]", overall_section) \
                              .replace("[INSERT 3D COORDINATES PROMPT SECTION]", self.coords_section) \
                              .replace("[INSERT SCENE ANALYSIS SECTION]", scene_section) \
                              .replace("[INSERT MEDIA SECTION]", media_section)

        # Scene image first, then the chronological trajectory frames (frames mode);
        # in video mode the clips are attached after the scene image.
        review_image_paths = ([scene_image_path] if has_scene_image else []) + frame_paths

        # Use conversation history; do not summarize. Strip previously accumulated
        # images so only THISreview's fresh trajectory frames are sent — keeps the request under
        # provider image caps (e.g. Bedrock max 20) and focuses the reviewer.
        messages = self.task.conversation_messages
        models.strip_images_from_messages(messages)
        media_desc = f"{len(video_paths)} video(s)" if use_video else (
            f"{len(frame_paths)} key frames (head steps {[i for i, _ in head_frames]}, "
            f"wrist steps {[i for i, _ in wrist_frames]})")
        self.logger.info(PROGRESS + f"==================== VLM review using {media_desc}, start_idx={start_idx}, scene_image={'yes' if has_scene_image else 'no'}, model={review_model}." + ENDC)
        messages = models.call_llm_cached(self.main_connection, self.client, review_model, prompt, messages, "user", file=sys.stderr, image_paths=review_image_paths, video_paths=video_paths, options={"log_msgs": True, "max_tokens": self.args.max_tokens, "reasoning_effort": self.args.reasoning_effort, "cache": self.llm_cache, "conversation_key": f"{self.task.conversation_key}-review"})
        # Drop the fresh review media now that the review is done, so it doesn't persist in the shared conversation
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
        # the measured pull bookkeeping knows every direction already tried in this task; the
        # tracking recommendation only sees this attempt, so it yields when the former applies
        proprio_directs = bool(pull_facts and pull_facts["outcome"] == "part_did_not_move")
        if not success and tracking_facts and not proprio_directs:
            from tracking.review_evidence import recommendation_text
            rec = recommendation_text(tracking_facts)
            if rec:
                rec = "Tracking-based fix: " + rec
                if isinstance(improvement_steps, list):
                    improvement_steps = improvement_steps + [rec]
                else:
                    improvement_steps = (str(improvement_steps) + "\n" + rec).strip()
                self.logger.info(PROGRESS + f"Review: added to improvement steps -> {rec}" + ENDC)
        success, reason, improvement_steps = self._apply_pull_proprio_verdict(
            pull_facts, success, reason, improvement_steps, camera_moved)
        if not success:
            from helpers.object_displacement import retry_hint
            dhint = retry_hint(disp_facts, self.task.command)
            if dhint:
                self.logger.info(PROGRESS + f"Review: displacement retry hint -> {dhint}" + ENDC)
                if isinstance(improvement_steps, list):
                    improvement_steps = [dhint] + improvement_steps
                else:
                    improvement_steps = (dhint + "\n" + str(improvement_steps or "")).strip()
        self.task.review_reason = reason
        self.task.review_improvement_steps = improvement_steps
        self.logger.info((OK if success else FAIL) + f"Review: success={success} | task=\"{self.task.title()}\". See details in above json" + ENDC)
        if success:
            self.task.completed_task = True
            self.task.review_succeeded = True
        else:
            self.task.failed_task = True

    def task_completed(self):        
        self.task.review_clips = create_trajectory_videos(self.logger, self.task.start_attempt_trajectory_step)
        
        self.task.attempt_number += 1
        max_attempts = self.task.max_attempts or self.args.attempts
        is_replay = bool(self.args.replay_log)

        # Guard: task_completed() claimed while the robot never moved during this attempt
        # (no new trajectory steps). This happens when generated code blocks were not
        # executed - accepting it would report a hallucinated success.
        if not is_replay and self.trajectory_step <= self.task.start_attempt_trajectory_step:
            reason = (f"task_completed() was called but no trajectory was executed in this attempt "
                      f"(trajectory_step={self.trajectory_step}, attempt start={self.task.start_attempt_trajectory_step}).")
            self.logger.info(FAIL + reason + ENDC)
            self.task.review_reason = reason
            self.task.review_improvement_steps = [
                "Actually execute the planned trajectories (execute_trajectory / gripper actions) before calling task_completed().",
            ]
            if self.task.attempt_number >= max_attempts:
                self.task.completed_task = True
                self.task.review_succeeded = False
            else:
                self.task.failed_task = True
            return

        # On the final attempt, skip review and accept the result
        # (in replay mode, ignore max_attempts so all blocks execute)
        if (not is_replay and self.task.attempt_number >= max_attempts
                and getattr(self.task, "overall_command", None)
                and str(self.args.review_provider).startswith("vlm")):
            # Under the planner the final attempt is still reviewed: otherwise the planner gets
            # success=False with the PREVIOUS attempt's reason, even when this attempt did it.
            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([TASK_COMPLETED])
            [env_connection_message] = self.main_connection.recv()
            self.logger.info(env_connection_message)
            self.logger.info(PROGRESS + f"VLM review of final attempt {self.task.attempt_number}/{max_attempts} | "
                             f"task=\"{self.task.title()}\"..." + ENDC)
            try:
                self.run_vlm_review()
            except Exception as e:
                self.logger.info(FAIL + f"VLM review failed: {e}" + ENDC)
            self.task.failed_task = False
            self.task.completed_task = True
            return
        if not is_replay and self.task.attempt_number >= max_attempts:
            self.task.completed_task = True
            self.task.accepted_without_review = True
            self.logger.info(PROGRESS + f"task_completed final attempt {self.task.attempt_number}/{max_attempts} -- accepting result" + ENDC)
            # No review, but the measured pull facts still tell the planner when the held
            # part never moved (so it does not take the unreviewed attempt as an opening).
            facts, _ = self._pull_proprio_review_evidence(self.task.start_attempt_trajectory_step)
            _ok, reason, steps = self._apply_pull_proprio_verdict(facts, True, "", [])
            if not _ok:
                self.task.review_reason = reason
                self.task.review_improvement_steps = steps
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
        self.logger.info(PROGRESS + f"VLM review after attempt {self.task.attempt_number}/{max_attempts} | task=\"{self.task.title()}\"..." + ENDC)
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









