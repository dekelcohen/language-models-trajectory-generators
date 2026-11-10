# Rollout `slide_cabinet_open_right_fix_SUCCESS` - SUCCESS

- Source: `images_slide_s1`
- Command: `open the right cabinet`
- Planner end: Planner: overall command completed.
- Wall time: 3 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': None, 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'pybullet', 'transport': 'auto', 'task': 'franka_kitchen:slide_cabinet', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['open the right cabinet'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_slide_s1', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> Open the right upper wall cabinet (the right one of the two blue-gray cabinets at the back, above the dark stove/backsplash with red knobs). It has a vertical handle near its left edge. Grasp that vertical handle and pull the cabinet door open. The door most likely swings outward on a hinge on the edge farthest from the handle, but it may also slide, so follow how the door actually moves. Avoid colliding with the dark stove/backsplash object below the cabinet, the left cabinet, and the sink/faucet on the right. Success = the right cabinet door is clearly open: moved well (several cm) from its 

- **Attempt 1: SUCCESS** (frames 1-85, 10/11 00:04)
  - Reviewer: The right cabinet is clearly open in head cam step 85: its handle has shifted right and the interior is exposed compared with the starting scene. The change appears between head cam steps 75 and 76, during retreat. The planned pull and arc are not confirmed successful: proprioception says the hand did not move while holding something, and a stable grasp of the handle is unverified.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'right cabinet handle' -> left_place farther=0.99 nearer=0.0 changed=0.45 ncc=0.58
  - Evidence: Review: displacement 'cabinet handle' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'right cabinet door' -> partly_moved farther=0.4 nearer=0.38 changed=0.18 ncc=0.8
  - Evidence: Review: displacement 'cabinet' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'stove' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.85
  - Evidence: Review: displacement 'backsplash' -> in_place farther=0.11 nearer=0.12 changed=0.06 ncc=0.83
  - Evidence: Review: displacement 'sink' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.99
  - Evidence: Review: displacement 'faucet' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: [pull_proprio] outcome=no_pull
  - Evidence: Review: pull proprioception -> outcome=no_pull max_carried=0.0 pulls=0
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[0.302, 0.001, 1.591] gripper_open=True gripper_width=0.08 holding=False rotation=-0.0 objects={'target_link_pos': [0.528, -0.224, 1.95], 'kettle_pos': [0.292, 0.194, 1.214]}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 21 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 4 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
