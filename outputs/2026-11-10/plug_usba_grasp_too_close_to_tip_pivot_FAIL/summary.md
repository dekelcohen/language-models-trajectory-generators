# Rollout `plug_usba_grasp_too_close_to_tip_pivot_FAIL` - INCOMPLETE

- Source: `images_plug_p11_usba`
- Command: `Pick up the USB plug and insert it into the laptop's USB port`
- Planner end: (not reached)
- Wall time: 17 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': None, 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'usb', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 3, 'max_planner_iter': 5, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ["Pick up the USB plug and insert it into the laptop's USB port"], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_plug_p11_usba', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> Pick up the small dark USB plug lying on the floor to the right of the laptop, in front of the laptop's right side. If the plug's body and its metal connector end can be told apart, grasp the plug body (the thicker housing), not the metal connector tip, so the connector stays exposed for insertion later. The laptop is to the left of the plug, with its screen upright and its base on the floor. Do not touch, push or collide with the laptop. After closing the gripper, lift the plug straight up about 20 cm above the floor and hold it there firmly in the gripper. A smaller lift is acceptable here b

- **Attempt 1: SUCCESS** (frames 1-92, 10/11 05:43)
  - Reviewer: The final wrist frames show the dark plug between the fingers, with the support remaining below; the gripper sensor confirms an object is held. The measured vertical lift is 16 cm, and the laptop remains unchanged. The precise grasp depth and connector exposure are not visually clear.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 1.6 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'USB plug' -> partly_moved farther=0.15 nearer=0.0 changed=0.27 ncc=0.67
  - Evidence: Review: displacement 'USB plug body' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement 'USB plug metal connector' -> partly_moved farther=0.17 nearer=0.0 changed=0.3 ncc=0.64
  - Evidence: Review: displacement 'laptop' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: displacement 'laptop base' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Evidence: Review: force feedback -> contact:contact
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.113, 0.446, 0.215] gripper_open=False gripper_width=0.0158 holding=True rotation=1.571 objects={'port_pos': [0.092, 0.496, 0.069], 'plug_pos': [-0.119, 0.446, 0.216], 'plug_tip_pos': [-0.095, 0.446, 0.216]}`

### Subtask 2 (max_attempts=3)

> The robot gripper is ALREADY HOLDING a small dark USB plug (grasped by its body, lifted about 16 cm above the floor); do NOT release it or re-grasp anything. Insert this USB plug into the laptop's USB port. The laptop is on the floor to the left of the arm: its base lies flat and its screen stands upright, hinged along the base's back edge, so the head camera sees the back of the screen. The USB port is not visible from the head camera. It is most likely a small rectangular slot on one of the side edges of the laptop base (the right side edge, nearest the arm, is the first candidate; otherwise


## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 36 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
