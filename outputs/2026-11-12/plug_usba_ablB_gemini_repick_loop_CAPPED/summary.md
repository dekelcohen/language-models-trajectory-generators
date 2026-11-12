# Rollout `plug_usba_ablB_gemini_repick_loop_CAPPED` - INCOMPLETE

- Source: `images_plug_ablB2`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 30 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'perception_reasoning_effort': 'low', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'wrist_pointing_model': 'or-google/gemini-3.8-flash', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': 'outputs/_ablation/skills_B_lean', 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_ablB2', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

skill ablation, 30-min cap

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small black USB plug lying on the floor to the right of the laptop (lower-left of the scene). Grasp it by its body, taking care not to collide with the laptop beside it, then lift it clearly off the floor and hold it. Success = the USB plug is held in the gripper and lifted at least several cm above the floor, with the gripper still closed on it.

- **Attempt 1: SUCCESS** (frames 1-76, 12/11 22:23)
  - Reviewer: The final wrist cam step 76 shows the black plug held at the gripper, while the support remains below. The closed gripper holds an object and rose about 18 cm; the laptop remained in place. The grasp was deeper than planned, which may affect subsequent insertion.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.12, 0.461, 0.235] gripper_open=False gripper_width=0.0156 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.12, 0.461, 0.238], 'plug_tip_pos': [-0.096, 0.461, 0.238]}`

### Subtask 2 (max_attempts=3)

> The robot is holding the small black USB plug. Carry it to the laptop at the lower-left of the scene and insert it into the laptop's USB port. The port is not clearly visible in the head camera; look for a small rectangular USB socket on the laptop's side edge facing the arm (right or rear edge) and move the hand to view it if needed. Avoid colliding with the laptop body or its hinged display. Align the plug with the port opening and push it straight in along the port axis. Success = the USB plug is seated in the laptop's USB port, with its metal tip inserted into the socket and no visible gap

- **Attempt 1: FAIL** (frames 476-815, 12/11 22:38)
  - Reviewer: The plug is not seated in the laptop. In the final views (head cam step 815 and wrist cam step 815), the dark object remains away from the laptop and the gripper is open and empty. Several grasps registered as holding, but subsequent lifts lost the grip; no carry or insertion was completed. The repeated resistance does not establish whether the plug is tethered or the gripper caught its support.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.0 nearer=0.0 changed=0.31 ncc=0.84
  - Evidence: Review: displacement 'laptop base keyboard' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB socket on laptop side edge' -> partly_moved farther=0.0 nearer=0.0 changed=0.29 ncc=0.84
  - Evidence: Review: displacement since command start 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.118, 0.434, 0.186] gripper_open=True gripper_width=0.08 holding=False rotation=0.0 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [1.968, -6.189, 0.033], 'plug_tip_pos': [1.948, -6.2, 0.023]}`
- **Attempt 2: FAIL** (frames 816-867, 12/11 22:39)
  - Reviewer: Success requires the USB plug to be seated in the laptop port with no visible gap. At wrist cam step 869, the plug remains beneath the gripper, away from the laptop; the final gripper measurement confirms it is open and empty. The arm approached the object (head cam steps 830–855), but a stable grasp and lift are not established (wrist cam steps 855–869). No carry or insertion is visible. The cause of the repeated lift failures remains uncertain.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.0 nearer=0.0 changed=0.4 ncc=0.74
  - Evidence: Review: displacement 'laptop base keyboard' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB socket on laptop side edge' -> partly_moved farther=0.0 nearer=0.0 changed=0.35 ncc=0.68
  - Evidence: Review: displacement since command start 'small black USB plug' -> partly_moved farther=0.13 nearer=0.07 changed=0.33 ncc=0.52
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.136, 0.442, 0.052] gripper_open=True gripper_width=0.08 holding=False rotation=1.572 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [3.28, -6.505, 0.076], 'plug_tip_pos': [3.268, -6.526, 0.074]}`
- **Attempt 3: FAIL** (frames 866-1275, 12/11 22:46)
  - Reviewer: The plug is not seated in the laptop. In wrist cam step 1275, the dark object remains separate from the laptop; the final gripper is open and empty. Several grasps registered as holding, but subsequent lifts slipped. The +y slide stalled and jammed, and the final wiggle did not prevent another slip. The cable or support hypothesis remains unverified.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.0 nearer=0.0 changed=0.34 ncc=0.82
  - Evidence: Review: displacement 'laptop base keyboard' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'USB socket on laptop side edge' -> partly_moved farther=0.0 nearer=0.0 changed=0.37 ncc=0.81
  - Evidence: Review: displacement since command start 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement since command start 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.117, 0.438, 0.188] gripper_open=True gripper_width=0.08 holding=False rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-1.581, -9.671, 0.035], 'plug_tip_pos': [-1.571, -9.688, 0.05]}`

### Subtask 3 (max_attempts=3)

> The small black USB plug stands upright on the table near the laptop (right of it, near the center of the view). The gripper is currently empty and open. Pick it up: inspect it from the wrist camera first to find the plug housing, then grasp the housing at its midpoint, closing at the correct depth rather than too low. The plug may be attached to a cable or fixed to a support, so lift slowly and stop if resistance is felt; if it resists, adjust the grasp or direction instead of repeating the same motion. Avoid hitting the laptop beside it. Lift the plug clearly off the table and hold it. Succe


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_FAIL.png`
- `03_s2_a2_FAIL.png`
- `04_s2_a3_FAIL.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 2 pointing/segmentation image(s))
- `s2_a1/` (FAIL; head/wrist start+end frames, 47 pointing/segmentation image(s))
- `s2_a2/` (FAIL; head/wrist start+end frames, 0 pointing/segmentation image(s))
- `s2_a3/` (FAIL; head/wrist start+end frames, 19 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 12 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
