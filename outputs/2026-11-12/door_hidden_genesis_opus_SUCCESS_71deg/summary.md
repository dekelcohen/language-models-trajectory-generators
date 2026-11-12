# Rollout `door_hidden_genesis_opus_SUCCESS_71deg` - SUCCESS

- Source: `images_door_hidden_gs_touch`
- Command: `open the door`
- Planner end: Planner: overall command completed.
- Wall time: 3048 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'perception_reasoning_effort': 'low', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'wrist_pointing_model': None, 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'door:hidden', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['open the door'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_door_hidden_gs_touch', 'delete_images': True}
```
</details>

## Analysis (root cause / fix)

after door fixes (force-limited press, torsional friction, no stale angle, re-centre)

## Timeline

### Subtask 1 (max_attempts=3)

> A tall gray cylinder stands in the center of the scene, directly in front of a narrow white upright panel (likely a door) and hiding most of it. Move the gray cylinder out of the way so it ends clear of the white panel/door front and its opening/swing path. Any method is fine: pick it up and place it aside on the floor away from the panel, push it sideways, or topple it so it falls away from the panel. Avoid pushing the cylinder into the white panel. Then move the hand back to the pose it started this subtask from. Success = the gray cylinder no longer stands in front of the white panel/door a

- **Attempt 1: SUCCESS** (frames 1-159, 10/11 20:40)
  - Reviewer: The cylinder fell aside during the initial approach, not the planned push. It lies clear of the panel by head cam step 27; the panel remains unobstructed in head cam step 159. The measured displacement confirms the cylinder left its original place. The final hand pose was commanded but its measured accuracy is not provided.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'gray cylinder' -> left_place farther=0.94 nearer=0.03 changed=0.69 ncc=0.51
  - Evidence: Review: displacement 'white panel' -> partly_moved farther=0.09 nearer=0.0 changed=0.18 ncc=0.69
  - Evidence: Review: displacement 'door' -> partly_moved farther=0.12 nearer=0.11 changed=0.2 ncc=0.62
  - Evidence: Review: displacement 'door handle' -> partly_moved farther=0.06 nearer=0.48 changed=0.83 ncc=0.59
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.425, 0.651, 0.909] gripper_open=True gripper_width=0.08 holding=False rotation=None objects={'door_handle_pos': [-0.098, -0.013, 0.672], 'latch_pos': [-0.098, -0.013, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [2.264, 14.937, 0.15], 'door_angle_deg': 0.0}`

### Subtask 2 (max_attempts=3)

> Open the door. The door is a gray upright panel near the center of the scene with a white vertical edge, standing on the floor. Its short gray handle sticks out from the upper-middle area of the panel and is clearly visible. The robot arm is currently close to the handle, so first move the gripper clear and align it with the handle before grasping, and avoid hitting the door panel. A gray cylinder lies toppled on the floor nearby, so avoid it. Grasp the handle and pull or slide the door open. If the door is hinged, the hinge is on the edge farthest from the handle, and you should pull along th

- **Attempt 1: FAIL** (frames 1752-3804, 10/11 21:18)
  - Reviewer: Success requires a door opening of at least 60°. In head cam step 3804 the door appears nearly closed, and the last measured angle was 6.8°. Grasps made contact but were not stable: touch pulls slipped, and guarded pulls lost the handle or the door fell back. The push behind the panel was blocked. The final wrist frame and fully closed gripper do not establish a held handle.
  - Evidence: Review: gripper evidence -> commanded CLOSED and the fingers are fully closed (-0.1 cm): NOTHING is held between them (the grasp missed, or the object slipped out).
  - Evidence: Review: displacement 'door' -> hidden farther=0.06 nearer=0.56 changed=0.55 ncc=0.35
  - Evidence: Review: displacement 'door handle' -> hidden farther=0.01 nearer=0.79 changed=0.77 ncc=0.22
  - Evidence: Review: displacement 'white vertical edge of the door' -> in_place farther=0.0 nearer=0.0 changed=0.0 ncc=0.93
  - Evidence: Review: displacement 'gray cylinder' -> hidden farther=0.02 nearer=0.67 changed=0.66 ncc=0.03
  - Evidence: Review: displacement since command start 'gray cylinder' -> left_place farther=0.95 nearer=0.01 changed=0.87 ncc=0.25
  - Evidence: Review: displacement since command start 'white panel' -> partly_moved farther=0.09 nearer=0.0 changed=0.18 ncc=0.7
  - Evidence: Review: displacement since command start 'door' -> left_place farther=0.11 nearer=0.28 changed=0.65 ncc=0.07
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.52 nearer=0.0 changed=0.48 ncc=0.34
  - Evidence: [pull_proprio] outcome=part_did_not_move
  - Evidence: Review: pull proprioception -> outcome=part_did_not_move max_carried=0.0 pulls=2
  - Evidence: Review: force feedback -> pull:slipped; pull:slipped; pull:slipped; pull:slipped; pull:slipped; pull:slipped; pull:slipped; pull:slipped; pull:blocked; guarded_pull:push_blocked; guarded_pull:door_fell_back; guarded_pull:door_fell_back
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.302, 0.395, 0.758] gripper_open=False gripper_width=0.001 holding=False rotation=1.832 objects={'door_handle_pos': [-0.115, 0.0, 0.672], 'latch_pos': [-0.115, 0.0, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [40.216, 342.517, 0.15], 'door_angle_deg': 2.6}`
- **Attempt 2: FAIL** (frames 5260-7110, 10/11 21:57)
  - Reviewer: The final head cam step 7110 shows the panel substantially swung compared with the start, but the measured angle is 54° ± 5°, so the required 60° opening is not established. The final 'push back face to 63 deg' label is a command, not a measured angle. Some grasps held briefly, but slips and blocked pushes prevented reliable opening; the gripper ended empty.
  - Evidence: Review: gripper evidence -> commanded CLOSED and the fingers are fully closed (0.1 cm): NOTHING is held between them (the grasp missed, or the object slipped out).
  - Evidence: Review: displacement 'door' -> left_place farther=0.68 nearer=0.0 changed=0.51 ncc=0.4
  - Evidence: Review: displacement 'door handle' -> left_place farther=1.0 nearer=0.0 changed=0.71 ncc=0.03
  - Evidence: Review: displacement 'white vertical edge of the door' -> in_place farther=0.01 nearer=0.0 changed=0.0 ncc=0.96
  - Evidence: Review: displacement 'gray cylinder' -> left_place farther=1.0 nearer=0.0 changed=0.58 ncc=0.23
  - Evidence: Review: displacement since command start 'gray cylinder' -> left_place farther=0.97 nearer=0.0 changed=0.9 ncc=0.32
  - Evidence: Review: displacement since command start 'white panel' -> partly_moved farther=0.09 nearer=0.0 changed=0.18 ncc=0.7
  - Evidence: Review: displacement since command start 'door' -> partly_moved farther=0.47 nearer=0.24 changed=0.93 ncc=0.5
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.52 nearer=0.48 changed=0.98 ncc=0.7
  - Evidence: [pull_proprio] outcome=part_did_not_move
  - Evidence: Review: pull proprioception -> outcome=part_did_not_move max_carried=0.027 pulls=1
  - Evidence: Review: force feedback -> guarded_pull:push_blocked; pull:slipped; pull:slipped; guarded_pull:push_blocked
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.658, 0.217, 0.74] gripper_open=False gripper_width=0.001 holding=False rotation=0.009 objects={'door_handle_pos': [-0.657, 0.019, 0.672], 'latch_pos': [-0.657, 0.019, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [55.083, 469.121, 0.15], 'door_angle_deg': 75.0}`
- **Attempt 3: FAIL** (frames 7111-7491, 10/11 22:05)
  - Reviewer: The goal is a door opened at least roughly 60°. Compared with the initial scene, the door is substantially open in head cam step 7491, but the arm-clear depth reading is 59° ± 5°, so the required opening is not confirmed. The gripper held nothing, and the panel shows no clear further opening during the attempted push at head cam steps 7381–7436. The proposed later push has no execution evidence.
  - Evidence: Review: gripper evidence -> commanded CLOSED and the fingers are fully closed (0.1 cm): NOTHING is held between them (the grasp missed, or the object slipped out).
  - Evidence: Review: displacement 'door' -> left_place farther=0.61 nearer=0.1 changed=0.51 ncc=0.43
  - Evidence: Review: displacement 'door handle' -> left_place farther=0.7 nearer=0.3 changed=0.73 ncc=0.15
  - Evidence: Review: displacement 'white vertical edge of the door' -> in_place farther=0.01 nearer=0.0 changed=0.0 ncc=0.96
  - Evidence: Review: displacement 'gray cylinder' -> left_place farther=1.0 nearer=0.0 changed=0.58 ncc=0.23
  - Evidence: Review: displacement since command start 'gray cylinder' -> left_place farther=0.89 nearer=0.05 changed=0.9 ncc=0.28
  - Evidence: Review: displacement since command start 'white panel' -> partly_moved farther=0.09 nearer=0.0 changed=0.18 ncc=0.7
  - Evidence: Review: displacement since command start 'door' -> left_place farther=0.4 nearer=0.48 changed=0.91 ncc=0.28
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.52 nearer=0.48 changed=0.88 ncc=0.81
  - Evidence: [pull_proprio] outcome=closed_on_nothing
  - Evidence: Review: pull proprioception -> outcome=closed_on_nothing max_carried=0.0 pulls=0
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.301, 0.4, 0.76] gripper_open=False gripper_width=0.001 holding=False rotation=0.463 objects={'door_handle_pos': [-0.657, 0.019, 0.672], 'latch_pos': [-0.657, 0.019, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [59.563, 507.274, -4123.348], 'door_angle_deg': 75.0}`

### Subtask 3 (max_attempts=3)

> A tall gray cylinder stands in front of a white door-like panel and blocks most of the door front. The door handle is hidden, probably behind the cylinder. Move the gray cylinder out of the way. You may pick it up and place it aside on the floor, or push it or topple it. It must end clear of the door front and of the door's swing / opening path. Do not knock the door panel itself. The robot arm is in the lower-right of the view, and more robot structure enters from the right, so avoid collisions with it. Then move the hand back to the pose it started this subtask from, which is a known reachab

- **Attempt 1: SUCCESS** (frames 1-149, 12/11 22:59)
  - Reviewer: The cylinder toppled during the rise, not the planned push: head cam steps 18–34 show it falling, and head cam step 149 shows it lying clear of the door front. The push and subsequent lift stalled; the final hand pose was not numerically verified. No grasp was attempted.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'gray cylinder' -> left_place farther=0.92 nearer=0.06 changed=0.69 ncc=0.47
  - Evidence: Review: displacement 'white door panel' -> partly_moved farther=0.06 nearer=0.0 changed=0.13 ncc=0.69
  - Evidence: Review: displacement 'door handle' -> left_place farther=0.33 nearer=0.42 changed=0.6 ncc=0.32
  - Evidence: Review: pull proprioception skipped (not an open/close/pull task)
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.398, 0.639, 0.844] gripper_open=True gripper_width=0.08 holding=False rotation=0.004 objects={'door_handle_pos': [-0.098, -0.013, 0.672], 'latch_pos': [-0.098, -0.013, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [0.405, 1.178, 0.15], 'door_angle_deg': 0.0}`

### Subtask 4 (max_attempts=3)

> Open the vertical gray door that stands near the center of the scene. It has a protruding lever handle that is clearly visible. The door most likely swings on a vertical hinge on the edge farthest from the handle, but the opening direction (push or pull) is uncertain. If the door does not move one way, try the other. The robot arm is on the right side and partially overlaps the door. If it is too close to the handle, first move the hand back to a hover pose a little in front of the handle, clear of the door. Then grasp the lever handle, press it down if needed to unlatch, and swing the door op

- **Attempt 1: FAIL** (frames 149-2553, 12/11 23:16)
  - Reviewer: Success requires the door to remain open at least 60°. At head cam step 2553, the panel appears nearly closed compared with the start, consistent with the final depth measurement of 2°. The final wrist view is occluded, so the object held between the fingers cannot be identified confidently. Some grasps and pulls moved the door temporarily, but the reported 'opened' outcome conflicts with the final angle and does not establish success.
  - Evidence: Review: gripper evidence -> commanded CLOSED, but the fingers stalled at 3.5 cm (fully closed is ~0.1 cm): they are blocked by an object, so the gripper IS HOLDING something between its fingers right now. Use the clips to judge WHICH object it is and where it is - do not conclude that the gripper is
  - Evidence: Review: displacement 'door' -> partly_moved farther=0.0 nearer=0.12 changed=0.13 ncc=0.62
  - Evidence: Review: displacement 'door handle' -> partly_moved farther=0.03 nearer=0.13 changed=0.35 ncc=0.54
  - Evidence: Review: displacement 'gray cylinder' -> partly_moved farther=0.01 nearer=0.0 changed=0.25 ncc=0.77
  - Evidence: Review: displacement since command start 'gray cylinder' -> left_place farther=0.96 nearer=0.01 changed=0.67 ncc=0.51
  - Evidence: Review: displacement since command start 'white door panel' -> partly_moved farther=0.06 nearer=0.0 changed=0.14 ncc=0.69
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.58 nearer=0.0 changed=0.53 ncc=0.38
  - Evidence: [pull_proprio] outcome=part_did_not_move
  - Evidence: Review: pull proprioception -> outcome=part_did_not_move max_carried=0.005 pulls=4
  - Evidence: Review: force feedback -> contact:contact; pull:end_stop; contact:contact; pull:end_stop; contact:contact; pull:end_stop; open_door:partly_open; contact:contact; pull:end_stop; open_door:partly_open; contact:contact; pull:slipped; contact:contact; pull:slipped; contact:contact; pull:slipped; open_do
  - Evidence: Touch pull finished: slipped
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.303, -0.012, 0.638] gripper_open=False gripper_width=0.0351 holding=True rotation=-0.857 objects={'door_handle_pos': [-0.102, -0.01, 0.672], 'latch_pos': [-0.102, -0.01, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [-1.742, 10.302, 0.15], 'door_angle_deg': 0.7}`
- **Attempt 2: SUCCESS** (frames 2554-2861, 12/11 23:21)
  - Reviewer: The goal was a door open at least 60 degrees. The panel visibly swings away from its starting position (head cam steps 2554 and 2815) and remains open in the final frame (head cam step 2861); the final depth measurement is 71 degrees. The initial horizontal tests stalled, and a later touch pull slipped, but the subsequent opening succeeded. The gripper is open and not holding anything.
  - Evidence: Review: gripper evidence -> commanded OPEN, finger opening 8.0 cm: the gripper is not holding anything.
  - Evidence: Review: displacement 'door' -> partly_moved farther=0.39 nearer=0.3 changed=0.48 ncc=0.24
  - Evidence: Review: displacement 'door handle' -> hidden farther=0.06 nearer=0.76 changed=0.78 ncc=0.11
  - Evidence: Review: displacement 'gray cylinder' -> hidden farther=0.08 nearer=0.65 changed=0.79 ncc=0.21
  - Evidence: Review: displacement since command start 'gray cylinder' -> left_place farther=0.96 nearer=0.01 changed=0.87 ncc=0.3
  - Evidence: Review: displacement since command start 'white door panel' -> partly_moved farther=0.06 nearer=0.0 changed=0.14 ncc=0.7
  - Evidence: Review: displacement since command start 'door handle' -> left_place farther=0.58 nearer=0.42 changed=0.56 ncc=0.57
  - Evidence: [pull_proprio] outcome=part_moved_with_hand
  - Evidence: Review: pull proprioception -> outcome=part_moved_with_hand max_carried=0.068 pulls=4
  - Evidence: Review: force feedback -> guarded_pull:opened; contact:contact; pull:slipped; open_door:opened
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.479, 0.021, 0.702] gripper_open=True gripper_width=0.08 holding=False rotation=-0.275 objects={'door_handle_pos': [-0.264, 0.078, 0.672], 'latch_pos': [-0.264, 0.078, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [-2.297, 12.66, 0.15], 'door_angle_deg': 23.7}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_FAIL.png`
- `03_s2_a2_FAIL.png`
- `04_s2_a3_FAIL.png`
- `05_s3_a1_SUCCESS.png`
- `06_s4_a1_FAIL.png`
- `07_s4_a2_SUCCESS.png`
- `99_finish_SUCCESS.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 17 pointing/segmentation image(s))
- `s2_a1/` (FAIL; head/wrist start+end frames, 17 pointing/segmentation image(s))
- `s2_a2/` (FAIL; head/wrist start+end frames, 17 pointing/segmentation image(s))
- `s2_a3/` (FAIL; head/wrist start+end frames, 17 pointing/segmentation image(s))
- `s3_a1/` (SUCCESS; head/wrist start+end frames, 2 pointing/segmentation image(s))
- `s4_a1/` (FAIL; head/wrist start+end frames, 14 pointing/segmentation image(s))
- `s4_a2/` (SUCCESS; head/wrist start+end frames, 1 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 8 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)
