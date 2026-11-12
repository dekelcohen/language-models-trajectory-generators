# Rollout `plug_usba_ablA_terra_regrasp_drop_CAPPED` - INCOMPLETE

- Source: `images_plug_ablA1`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 29 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-sonnet-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'perception_reasoning_effort': 'low', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'wrist_pointing_model': None, 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': 'outputs/_ablation/skills_A_no_situational', 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_ablA1', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

skill ablation, 30-min cap

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small black USB plug lying on the floor to the right of the laptop (lower-left of the scene). Grasp it by its body, taking care not to collide with the laptop beside it, then lift it clearly off the floor and hold it. Success = the USB plug is held in the gripper and lifted at least several cm above the floor, with the gripper still closed on it.

- **Attempt 1: SUCCESS** (frames 1-90, 12/11 20:53)
  - Reviewer: The black plug is visible between the fingers during the lift (wrist cam steps 76 and 79). At the final state, the closed gripper is holding an object about 20 cm above its grasp height; the laptop remains in place. The grasp appears somewhat shallow, which may affect subsequent insertion.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 3.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'small black USB plug' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.65
  - Evidence: Review: displacement 'USB plug body' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.12, 0.461, 0.253] gripper_open=False gripper_width=0.0364 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.125, 0.466, 0.27], 'plug_tip_pos': [-0.122, 0.444, 0.278]}`

### Subtask 2 (max_attempts=3)

> While holding the USB plug, locate the laptop's USB port (likely on a side edge of the laptop base; the screen is hinged to the base, so do not push or move the screen). Align the plug with the port and insert it fully, keeping the plug oriented to match the port opening. Do not collide with the laptop screen. Success = the USB plug is seated in the laptop's USB port, with its metal connector no longer visible or only a small portion showing, and the plug is no longer held in the gripper or is held only at its rear end.


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 53 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
