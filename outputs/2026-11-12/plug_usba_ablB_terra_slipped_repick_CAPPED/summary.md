# Rollout `plug_usba_ablB_terra_slipped_repick_CAPPED` - INCOMPLETE

- Source: `images_plug_ablB1`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 30 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'perception_reasoning_effort': 'low', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'wrist_pointing_model': 'or-google/gemini-3.8-flash', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': 'outputs/_ablation/skills_B_lean', 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_ablB1', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

skill ablation, 30-min cap

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small black USB plug lying on the floor to the right of the laptop (lower-left of the scene). Grasp it by its body, taking care not to collide with the laptop beside it, then lift it clearly off the floor and hold it. Success = the USB plug is held in the gripper and lifted at least several cm above the floor, with the gripper still closed on it.

- **Attempt 1: SUCCESS** (frames 1-73, 12/11 21:22)
  - Reviewer: Wrist cam steps 51 and 73 show the black plug between the fingers and lifted from its starting position; head cam step 73 shows it elevated. The closed gripper sensor confirms a held object and a 13 cm lift. The laptop remains in place. The grasp was deeper than planned, but the pick-and-hold goal was met.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.12, 0.46, 0.185] gripper_open=False gripper_width=0.0156 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.12, 0.46, 0.188], 'plug_tip_pos': [-0.096, 0.46, 0.188]}`

### Subtask 2 (max_attempts=3)

> The robot is holding the small black USB plug. Find the laptop's USB port, which is not clearly visible and is likely on the laptop's side edge facing the plug's side (right side of the laptop in the lower-left of the scene). Carry the plug there, align it with the port opening, and insert it fully into the port, avoiding collisions with the laptop body and its hinged display. Success = the USB plug is seated in the laptop's USB port, with its metal connector no longer visible or only slightly protruding, and the gripper has released it or is still holding it at the port.


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 154 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
