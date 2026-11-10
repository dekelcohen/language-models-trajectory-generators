# Rollout `slide_cabinet_slide_right_fix_SUCCESS` - SUCCESS

- Source: `images_slide_s2`
- Command: `slide upper-right-cabinet to the right`
- Planner end: Planner: overall command completed.
- Wall time: 4 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': None, 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'pybullet', 'transport': 'auto', 'task': 'franka_kitchen:slide_cabinet', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['slide upper-right-cabinet to the right'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_slide_s2', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> Slide the upper-right cabinet door (the blue upper cabinet on the right side of the back wall) to the RIGHT. Its vertical handle is near the left edge of that cabinet's front panel and is clearly visible. Grasp the handle, or push against it from the left side, and move the door horizontally to the right (toward the sink side), keeping the motion parallel to the cabinet front. The perception system thinks it is a sliding (prismatic) door, but this is low confidence. If it clearly will not slide, report that instead of forcing it. Avoid colliding with the dark rectangular object just below the 

- **Attempt 1: SUCCESS** (frames 1-122, 10/11 00:10)
  - Reviewer: Compared with the start scene, head cam step 122 shows the right door and handle shifted right, exposing a gap. The pull measurements indicate a stable grasp and approximately 26 cm of rightward travel. The gripper ended open; the arm appears returned near its starting pose, though its exact final pose is not reported.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'upper-right cabinet handle' -> left_place farther=0.97 nearer=0.0 changed=0.43 ncc=0.6
  - Evidence: Review: displacement 'upper-right cabinet door' -> partly_moved farther=0.3 nearer=0.33 changed=0.14 ncc=0.84
  - Evidence: Review: displacement 'upper-right cabinet' -> partly_moved farther=0.3 nearer=0.33 changed=0.15 ncc=0.84
  - Evidence: Review: displacement 'upper-left cabinet' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'dark rectangular object' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: [pull_proprio] outcome=part_moved_with_hand
  - Evidence: Review: pull proprioception -> outcome=part_moved_with_hand max_carried=0.076 pulls=4
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[0.353, 0.001, 1.492] gripper_open=True gripper_width=0.08 holding=False rotation=-0.0 objects={'target_link_pos': [0.528, -0.135, 1.95], 'kettle_pos': [0.292, 0.194, 1.214]}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 14 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 4 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
