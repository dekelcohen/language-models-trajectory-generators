# Rollout `plug_usba_sonnet_s3_capped_during_insert` - INCOMPLETE

- Source: `images_plug_s3_sonnet`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 15 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_s3_sonnet', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

Sonnet 5.5 medium; good pick; gate bug (24mm lower bound > 23mm expected) cost a regrasp; 20 min cap hit inside insert_with_touch

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small dark upright USB plug lying on the floor just to the right of the laptop (the laptop is at the lower-left of the head-camera view). Grasp it firmly so it stays in the gripper, then lift it clear of the floor, keeping the USB connector end free and unobstructed. Success = the USB plug is held in the gripper and lifted at least 10 cm above the floor.

- **Attempt 1: SUCCESS** (frames 1-103, 12/11 14:11)
  - Reviewer: The dark plug is held between the fingers in wrist cam steps 93 and 103, while the support remains below. The gripper sensor confirms a grasp, and the measured lift of about 30 cm exceeds the 10 cm goal.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.113, 0.447, 0.353] gripper_open=False gripper_width=0.0158 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.121, 0.447, 0.349], 'plug_tip_pos': [-0.097, 0.447, 0.349]}`

### Subtask 2 (max_attempts=3)

> While holding the USB plug in the gripper, locate the USB port on the laptop base (likely on a side edge of the base, possibly partly hidden by the raised screen; the laptop is at the lower-left of the head-camera view). Align the plug's connector with the port opening and insert it fully into the port, keeping the laptop screen out of the way. Success = the USB connector is inserted into the laptop's USB port, with the metal connector no longer visible and the plug seated flush against the port.


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 52 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
