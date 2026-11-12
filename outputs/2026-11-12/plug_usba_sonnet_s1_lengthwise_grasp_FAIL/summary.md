# Rollout `plug_usba_sonnet_s1_lengthwise_grasp_FAIL` - INCOMPLETE

- Source: `images_plug_s1_sonnet`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 20 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_s1_sonnet', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

Sonnet 5.5 medium; picked at rot 0 (closed along the plug length, 34 mm); regrasp lost the plug; 20 min cap

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small dark upright USB plug lying on the floor just to the right of the laptop (the laptop is at the lower-left of the head-camera view). Grasp it firmly so it stays in the gripper, then lift it clear of the floor, keeping the USB connector end free and unobstructed. Success = the USB plug is held in the gripper and lifted at least 10 cm above the floor.

- **Attempt 1: SUCCESS** (frames 1-91, 12/11 13:34)
  - Reviewer: The dark top of the target is no longer on the upright support after the lift (wrist cam steps 74 and 91), while a dark object is held between the fingers (head cam step 91). The gripper sensor confirms a held object and approximately 15 cm of upward travel. The connector is partly obscured, so its exposure is not visually verifiable.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 3.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'USB connector end of the plug' -> partly_moved farther=0.15 nearer=0.0 changed=0.25 ncc=0.71
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'laptop USB port' -> partly_moved farther=0.16 nearer=0.0 changed=0.26 ncc=0.68
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.113, 0.446, 0.203] gripper_open=False gripper_width=0.0358 holding=True rotation=0.0 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.119, 0.451, 0.204], 'plug_tip_pos': [-0.095, 0.45, 0.204]}`

### Subtask 2 (max_attempts=2)

> While keeping the USB plug firmly held in the gripper, move the hand so the laptop's USB port can be seen. The port is a small rectangular slot on a side edge of the laptop base, and it may be hidden by the raised screen or by the arm. Position the hand and plug clear of the laptop, near the base's side edges, with the arm not blocking the camera's view of those edges. Do not hit the laptop. Success = the plug is still held in the gripper and a USB port opening on the laptop base is visible in the head camera, not hidden by the arm.

- **Attempt 1: FAIL** (frames 92-111, 12/11 13:36)
  - Reviewer: Head cam step 111 shows the hand clear of the laptop, but no USB port opening is identifiable on the visible base edge. The width sensor confirms something is held, though the images do not clearly identify it as the plug. The laptop remains in place. Trajectory 1 moved the hand away, but its exact endpoint is unverified and the required port visibility was not achieved.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 3.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'laptop base' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug' -> left_place farther=0.54 nearer=0.0 changed=0.57 ncc=0.46
  - Evidence: Review: displacement since command start 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement since command start 'USB connector end of the plug' -> partly_moved farther=0.15 nearer=0.0 changed=0.25 ncc=0.71
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement since command start 'laptop USB port' -> partly_moved farther=0.16 nearer=0.0 changed=0.26 ncc=0.68
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: displacement retry hint -> MEASURED (head camera RGB-D): 'USB plug' is no longer where it was first detected. Do NOT act at its old position. Detect it again first; if the subtask goal (e.g. clearing it out of the way) is already met, do not move it again - only move the arm clear of the cam
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.198, 0.297, 0.187] gripper_open=False gripper_width=0.0358 holding=True rotation=0.0 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.204, 0.302, 0.188], 'plug_tip_pos': [-0.179, 0.302, 0.188]}`
- **Attempt 2: FAIL** (frames 112-170, 12/11 13:39)
  - Reviewer: The gripper remains closed on an object, consistent with the previously held plug, and the laptop is unchanged. The hand moves clear of the laptop, but no USB port is identifiable in the final head-camera view (head cam step 170); the screen obscures much of the base. Wrist cam step 170 shows part of an edge, but that does not establish head-camera visibility.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 3.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'laptop base' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug' -> left_place farther=0.54 nearer=0.0 changed=0.68 ncc=0.54
  - Evidence: Review: displacement since command start 'small dark upright USB plug on the floor right of the laptop' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement since command start 'USB connector end of the plug' -> partly_moved farther=0.15 nearer=0.0 changed=0.25 ncc=0.71
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement since command start 'laptop USB port' -> partly_moved farther=0.16 nearer=0.0 changed=0.26 ncc=0.68
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: displacement retry hint -> MEASURED (head camera RGB-D): 'USB plug' is no longer where it was first detected. Do NOT act at its old position. Detect it again first; if the subtask goal (e.g. clearing it out of the way) is already met, do not move it again - only move the arm clear of the cam
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[0.05, 0.28, 0.24] gripper_open=False gripper_width=0.0358 holding=True rotation=0.001 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [0.044, 0.285, 0.242], 'plug_tip_pos': [0.069, 0.285, 0.242]}`

### Subtask 3 (max_attempts=3)

> The USB plug is already held in the gripper; keep holding it. Insert it into the laptop's USB port. The laptop is at the lower-left of the head-camera view, partly cropped, with its screen raised. The port is a small rectangular slot on a side edge of the laptop base, and the raised screen hides it from the head camera. Use the wrist camera and your own perception to find the port opening on the base edge. Move the plug to a hover point in front of the port, align the connector with the opening, then push it straight in along the port axis until it is seated. Avoid hitting the laptop or the sc


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_FAIL.png`
- `03_s2_a2_FAIL.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 2 pointing/segmentation image(s))
- `s2_a1/` (FAIL; head/wrist start+end frames, 4 pointing/segmentation image(s))
- `s2_a2/` (FAIL; head/wrist start+end frames, 62 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 10 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
