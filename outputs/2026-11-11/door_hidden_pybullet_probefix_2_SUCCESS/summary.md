# Rollout `door_hidden_pybullet_probefix_2_SUCCESS` - SUCCESS

- Source: `images_door_pb_d3`
- Command: `open the door`
- Planner end: Planner: overall command completed.
- Wall time: 18 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'pybullet', 'transport': 'auto', 'task': 'door:hidden', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['open the door'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_door_pb_d3', 'delete_images': True}
```
</details>

## Timeline

### Subtask 1 (max_attempts=3)

> A tall dark-gray upright rounded object (cylinder/slab-like) stands directly in front of a tall orange-brown door, covering most of the door face and likely hiding the door handle. Move this dark upright object out of the way so it no longer blocks the door. You may pick and place it aside on the floor, push it sideways, or topple it. It must end up clear of the door front and clear of the door's swing path; tipping it over so it lies away from the door is acceptable. Do not push it into the door or knock the door. A small gray protrusion is visible at the object's left edge; it may be the doo

- **Attempt 1: SUCCESS** (frames 1-212, 11/11 03:17)
  - Reviewer: The object initially covered the door. It tips during the second approach (head cam steps 122–124) and lies clear of the door in the final head cam step 212. The handle remains unchanged. The hand returned near its starting position, though about 7 cm lower. The first push tilted the object but did not topple it; the later planned push occurred after it had already fallen.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'dark upright object' -> left_place farther=0.8 nearer=0.0 changed=0.77 ncc=0.52
  - Evidence: Review: displacement 'door' -> left_place farther=0.8 nearer=0.0 changed=0.78 ncc=0.51
  - Evidence: Review: displacement 'door handle' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=1.0
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.381, 0.65, 0.832] gripper_open=True gripper_width=0.08 holding=False rotation=0.0 objects={'door_handle_pos': [-0.077, -0.009, 0.672], 'latch_pos': [-0.077, -0.009, 0.672], 'hinge_pos': [-0.217, -0.148, 0.697], 'pole_pos': [0.575, 0.266, 0.15], 'door_angle_deg': 0.0}`

### Subtask 2 (max_attempts=3)

> A tall wooden (orange-brown) door stands upright near the center of the scene. Its gray door handle is now clearly visible on the front face of the door. A separate vertical wooden post (possibly the door frame) overlaps the door's left edge. A long dark rectangular object, a previously moved obstacle, lies on the floor at the lower left; avoid it. The robot arm is currently on the right, close to the door, so be careful not to collide with the door panel or the post while approaching the handle. Grasp the gray door handle and open the door wide. The door most likely swings on a vertical hinge

- **Attempt 1: SUCCESS** (frames 580-1334, 11/11 03:30)
  - Reviewer: The final head cam step 1334 shows the door swung away from its starting position; the measured swing is approximately 62° (±5°), meeting the roughly 60° goal. The dark obstacle remained in place. Not all trajectories succeeded: the push stalled at head cam step 751, and the later grasp attempts closed on air. The open final state does not establish that the handle was held or that the planned pull caused it.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'door handle' -> left_place farther=1.0 nearer=0.0 changed=0.98 ncc=0.38
  - Evidence: Review: displacement 'door' -> left_place farther=0.67 nearer=0.01 changed=0.65 ncc=0.08
  - Evidence: Review: displacement 'wooden post' -> left_place farther=0.66 nearer=0.01 changed=0.64 ncc=0.06
  - Evidence: Review: displacement 'dark rectangular object' -> in_place farther=0.02 nearer=0.0 changed=0.03 ncc=0.98
  - Evidence: Review: displacement since command start 'dark upright object' -> left_place farther=0.85 nearer=0.0 changed=0.85 ncc=0.03
  - Evidence: Review: displacement since command start 'door' -> left_place farther=0.85 nearer=0.0 changed=0.85 ncc=0.03
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.71 nearer=0.0 changed=0.71 ncc=0.44
  - Evidence: [pull_proprio] outcome=closed_on_nothing
  - Evidence: Review: pull proprioception -> outcome=closed_on_nothing max_carried=0.0 pulls=0
  - Evidence: Review: force feedback -> open_door:partly_open; open_door:no_door_face; open_door:no_door_face; open_door:no_door_face; open_door:no_door_face; open_door:no_door_face; open_door:no_door_face; open_door:no_door_face
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.614, 0.08, 0.67] gripper_open=True gripper_width=0.08 holding=False rotation=0.348 objects={'door_handle_pos': [-0.711, -0.023, 0.689], 'latch_pos': [-0.711, -0.023, 0.689], 'hinge_pos': [-0.596, -0.161, 0.697], 'pole_pos': [0.575, 0.277, 0.15], 'door_angle_deg': 85.1}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 10 pointing/segmentation image(s))
- `s2_a1/` (SUCCESS; head/wrist start+end frames, 11 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 6 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
