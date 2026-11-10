# Rollout `plug_hdmi_usba_distractor_mislocated_port_FAIL` - INCOMPLETE

- Source: `images_plug_v1_hdmi`
- Command: `Pick up the HDMI plug and insert it into the laptop's HDMI port`
- Planner end: (not reached)
- Wall time: 39 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': None, 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the HDMI plug and insert it into the laptop's HDMI port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_v1_hdmi', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the HDMI plug: the small dark object standing on the floor just to the right of the laptop, near the center-right of the head-camera view. The laptop lies at the lower left with its screen upright and its dark base on the floor. Grasp the plug by its body or connector housing, not by the cable, and keep the metal connector tip free and unobstructed so it can later be inserted into a port. Do not touch or move the laptop. After closing the gripper, lift the plug straight up so it is clearly off the floor, about 20-30 cm above it, and well clear of the laptop. Keep holding it, because it

- **Attempt 1: SUCCESS** (frames 1-76, 10/11 06:49)
  - Reviewer: The dark plug is held between the fingers in wrist cam steps 67–76, while the support remains below. The gripper lifted it vertically by approximately 19 cm. The laptop remains unchanged. The precise grasp on the housing and tip clearance are not visually certain.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 2.0 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'HDMI plug metal connector tip' -> partly_moved farther=0.37 nearer=0.0 changed=0.57 ncc=0.77
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'laptop HDMI port' -> partly_moved farther=0.2 nearer=0.0 changed=0.31 ncc=0.59
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.113, 0.449, 0.245] gripper_open=False gripper_width=0.0196 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.062], 'plug_pos': [-0.119, 0.449, 0.246], 'plug_tip_pos': [-0.096, 0.449, 0.246]}`

### Subtask 2 (max_attempts=3)

> The robot is ALREADY HOLDING the HDMI plug in its closed gripper (measured: fingers stalled at about 2 cm). Do NOT release it or re-grasp anything. Insert this held plug into the laptop's HDMI port. The laptop is at the lower left of the head-camera view: its gray screen panel is upright and its dark base lies flat on the floor. The lid hinges along the bottom edge of the screen. The HDMI port is not visible from the head camera. It is most likely a small rectangular slot on one of the side edges of the dark laptop base, probably the edge nearest the robot or the right edge. First, while still


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 93 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
