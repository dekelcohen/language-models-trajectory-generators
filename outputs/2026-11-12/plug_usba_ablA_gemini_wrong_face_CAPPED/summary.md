# Rollout `plug_usba_ablA_gemini_wrong_face_CAPPED` - INCOMPLETE

- Source: `images_plug_ablA2`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 30 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'perception_reasoning_effort': 'low', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'wrist_pointing_model': 'or-google/gemini-3.8-flash', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': 'outputs/_ablation/skills_A_no_situational', 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_ablA2', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

skill ablation, 30-min cap

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small black USB plug lying on the floor to the right of the laptop (lower-left of the scene). Grasp it by its body, taking care not to collide with the laptop beside it, then lift it clearly off the floor and hold it. Success = the USB plug is held in the gripper and lifted at least several cm above the floor, with the gripper still closed on it.

- **Attempt 1: SUCCESS** (frames 1-68, 12/11 21:52)
  - Reviewer: The black plug body is between the fingers during the lift (wrist cam step 49) and remains held above its original position in the final frames (head and wrist cam step 68). The closed gripper rose about 14 cm while holding an object; the laptop remained in place. The measured grasp depth was 3.3 mm beyond the housing-center target, slightly outside the skill's 3 mm tolerance.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.31 ncc=0.65
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.119, 0.457, 0.193] gripper_open=False gripper_width=0.0157 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.12, 0.457, 0.196], 'plug_tip_pos': [-0.096, 0.457, 0.196]}`

### Subtask 2 (max_attempts=3)

> The robot is holding the small black USB plug. Find the laptop's USB port, which is not clearly visible and is likely on the laptop's side edge facing the plug's side (right side of the laptop in the lower-left of the scene). Carry the plug there, align it with the port opening, and insert it fully into the port, avoiding collisions with the laptop body and its hinged display. Success = the USB plug is seated in the laptop's USB port, with its metal connector no longer visible or only slightly protruding, and the gripper has released it or is still holding it at the port.

- **Attempt 1: FAIL** (frames 69-1874, 12/11 22:12)
  - Reviewer: The plug was not seated: both insertion attempts returned search_exhausted, and the gripper subsequently reported that it was no longer holding anything. The final wrist frame (step 1874) shows an empty gripper; it does not establish that the plug is in the port. The regrasp initially held an object, but the grasp did not remain stable. A preceding alignment error also caused a 295 N jam.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.95
  - Evidence: Review: displacement 'USB plug' -> in_place farther=0.0 nearer=0.0 changed=0.2 ncc=0.83
  - Evidence: Review: displacement 'laptop side edge' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.98
  - Evidence: Review: displacement since command start 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.31 ncc=0.65
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.96
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact; contact:contact; contact:contact; contact:contact; contact:contact; insert:search_exhausted; insert:search_exhausted
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[0.421, 0.47, 0.082] gripper_open=False gripper_width=0.0013 holding=False rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [0.788, -3.2, 0.005], 'plug_tip_pos': [0.768, -3.214, 0.005]}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_FAIL.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 3 pointing/segmentation image(s))
- `s2_a1/` (FAIL; head/wrist start+end frames, 79 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 8 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
