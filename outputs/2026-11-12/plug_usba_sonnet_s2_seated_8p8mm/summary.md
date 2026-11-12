# Rollout `plug_usba_sonnet_s2_seated_8p8mm` - SUCCESS

- Source: `images_plug_s2_sonnet`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: Planner: overall command completed.
- Wall time: 17 min  |  Metric: depth 8.8mm (eval seat 10mm), lateral 0.4mm, tilt 5.7deg

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_s2_sonnet', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

Sonnet 5.5 medium; seated via lone-catch centre guess; planner+reviewer success; eval partial

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small dark upright USB plug lying on the floor just to the right of the laptop (the laptop is at the lower-left of the head-camera view). Grasp it firmly so it stays in the gripper, then lift it clear of the floor, keeping the USB connector end free and unobstructed. Success = the USB plug is held in the gripper and lifted at least 10 cm above the floor.

- **Attempt 1: SUCCESS** (frames 1-110, 12/11 13:53)
  - Reviewer: The black plug visible atop the upright support before pickup is between the fingers during the lift (wrist cam step 97); the support remains on the floor in the final frames. The gripper remains blocked at 1.6 cm and rose about 15 cm, indicating the plug was lifted and held. The connector tip is not clearly resolved in the final view.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.114, 0.449, 0.203] gripper_open=False gripper_width=0.0158 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.12, 0.449, 0.204], 'plug_tip_pos': [-0.096, 0.449, 0.204]}`

### Subtask 2 (max_attempts=3)

> While holding the USB plug in the gripper, find the USB port on the laptop (a small rectangular socket along a side edge of the laptop base; the laptop is at the lower-left of the head-camera view, partly cropped, with its screen raised). Align the plug's connector end with the port opening and insert it straight into the port, pushing until it is seated. Do not hit the laptop screen. Success = the USB plug connector is inserted into the laptop's USB port, with only the plug body sticking out of the port.

- **Attempt 1: SUCCESS** (frames 111-491, 12/11 14:06)
  - Reviewer: The insertion sensor reported seated after 8.4 mm of advance, with only 0.23 mm movement in the sideways check. Wrist cam step 491 shows the released gripper at the laptop edge; the connector itself is partly obscured. The laptop remained in place, and the final gripper measurement confirms release.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug held in the gripper' -> left_place farther=0.96 nearer=0.0 changed=0.49 ncc=0.5
  - Evidence: Review: displacement since command start 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement since command start 'USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact; contact:contact; contact:contact; contact:contact; contact:contact; insert:seated
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[0.045, 0.488, 0.06] gripper_open=True gripper_width=0.08 holding=False rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [0.076, 0.495, 0.068], 'plug_tip_pos': [0.101, 0.496, 0.069]}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 3 pointing/segmentation image(s))
- `s2_a1/` (SUCCESS; head/wrist start+end frames, 40 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
