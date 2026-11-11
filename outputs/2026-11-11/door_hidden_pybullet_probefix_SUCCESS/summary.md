# Rollout `door_hidden_pybullet_probefix_SUCCESS` - SUCCESS

- Source: `images_door_pb_d2`
- Command: `open the door`
- Planner end: Planner: overall command completed.
- Wall time: 7 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'pybullet', 'transport': 'auto', 'task': 'door:hidden', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['open the door'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_door_pb_d2', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> A tall dark-gray upright rounded object (cylinder/slab-like) stands directly in front of a tall orange-brown door, covering most of the door face and likely hiding the door handle. Move this dark upright object out of the way so it no longer blocks the door. You may pick and place it aside on the floor, push it sideways, or topple it. It must end up clear of the door front and clear of the door's swing path; tipping it over so it lies away from the door is acceptable. Do not push it into the door or knock the door. A small gray protrusion is visible at the object's left edge; it may be the doo

- **Attempt 1: SUCCESS** (frames 1-134, 11/11 03:06)
  - Reviewer: The obstacle moves left during the push and topples beside the door (head cam steps 96–103). In the final head cam step 134, the door face is clear and the hand has returned toward its starting area. The handle remains in its original place. No grasp was attempted.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'dark upright object' -> left_place farther=0.8 nearer=0.0 changed=0.78 ncc=0.51
  - Evidence: Review: displacement 'door' -> left_place farther=0.81 nearer=0.0 changed=0.78 ncc=0.5
  - Evidence: Review: displacement 'door handle' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.381, 0.65, 0.831] gripper_open=True gripper_width=0.08 holding=False rotation=0.0 objects={'door_handle_pos': [-0.077, -0.009, 0.672], 'latch_pos': [-0.077, -0.009, 0.672], 'hinge_pos': [-0.217, -0.148, 0.697], 'pole_pos': [0.388, -0.121, 0.15], 'door_angle_deg': 0.0}`

### Subtask 2 (max_attempts=3)

> Open the tall wooden (orange-brown) door WIDE. A gray lever handle is visible on the door's face. The door most likely swings on a vertical hinge on the edge farthest from the handle. Whether it opens by pulling toward the robot or by pushing away is unknown, so check which way it moves and continue in that direction. A dark cylindrical object still stands near the door's lower edge. Avoid hitting it and keep the door's swing clear of it where possible. The robot arm currently sits close to the door, so start from a clear approach to the handle. Grasp or hook the handle (press the lever down f

- **Attempt 1: SUCCESS** (frames 135-472, 11/11 03:09)
  - Reviewer: The final door is swung well clear of the frame (head cam step 472); the depth-measured angle is 74°, above the 60° minimum. The first pull opened the door but it fell back (head cam steps 215–240); a later push reopened it (steps 389–417). One pull lost the handle and a press stalled, so not every motion succeeded as planned. The cylinder remained in place, and the gripper ended open.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'door' -> left_place farther=0.59 nearer=0.0 changed=0.58 ncc=0.24
  - Evidence: Review: displacement 'door handle' -> left_place farther=1.0 nearer=0.0 changed=1.0 ncc=0.1
  - Evidence: Review: displacement 'door hinge' -> left_place farther=1.0 nearer=0.0 changed=1.0 ncc=0.1
  - Evidence: Review: displacement 'dark cylinder' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.98
  - Evidence: Review: displacement since command start 'dark upright object' -> left_place farther=0.83 nearer=0.0 changed=0.85 ncc=0.17
  - Evidence: Review: displacement since command start 'door' -> left_place farther=0.84 nearer=0.0 changed=0.85 ncc=0.18
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.72 nearer=0.0 changed=0.72 ncc=0.44
  - Evidence: [pull_proprio] outcome=part_did_not_move
  - Evidence: Review: pull proprioception -> outcome=part_did_not_move max_carried=0.0 pulls=2
  - Evidence: Review: force feedback -> pull:unsupported; guarded_pull:door_fell_back; guarded_pull:opened; open_door:opened
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.269, 0.139, 0.789] gripper_open=True gripper_width=0.08 holding=False rotation=-0.64 objects={'door_handle_pos': [-0.745, -0.033, 0.672], 'latch_pos': [-0.745, -0.033, 0.672], 'hinge_pos': [-0.609, -0.175, 0.697], 'pole_pos': [0.384, -0.122, 0.15], 'door_angle_deg': 88.9}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 2 pointing/segmentation image(s))
- `s2_a1/` (SUCCESS; head/wrist start+end frames, 11 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
