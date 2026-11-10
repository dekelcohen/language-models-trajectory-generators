# Rollout `plug_usba_regression_floor_assumption_FAIL` - INCOMPLETE

- Source: `images_plug_p1_usba_FAIL_keep`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 14 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': None, 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_p1_usba', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small dark USB plug lying on the floor to the right of the laptop, in front of the laptop's right side. If the plug's body and its metal connector end can be told apart, grasp the plug body (the thicker housing), not the metal connector tip, so the connector stays exposed for insertion later. The laptop is to the left of the plug, with its screen upright and its base on the floor. Do not touch, push or collide with the laptop. After closing the gripper, lift the plug straight up about 20 cm above the floor and hold it there firmly in the gripper. A smaller lift is acceptable here b


## Files

- `00_start.png`
- `99_finish_INCOMPLETE.png`
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 2 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
