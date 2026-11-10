# Rollout `door_hidden_genesis_touchpull_fail` - INCOMPLETE

- Source: `images_door_hidden_gs_touch`
- Command: `open the door`
- Planner end: (not reached)
- Wall time: 60 min

<details><summary>Args</summary>

```
{'language_model': 'copilot-claude-opus-5.5', 'lm_images': True, 'max_tokens': 60000, 'reasoning_effort': 'medium', 'llm_cache_enabled': True, 'planner_perception_vlm': 'copilot-gpt-6.1-terra', 'review_provider': 'vlm:copilot-gpt-6.1-terra', 'robot': 'franka', 'touch': True, 'sim': 'genesis', 'transport': 'auto', 'task': 'door:hidden', 'list_tasks': False, 'mode': 'default', 'timeout': 15.0, 'seg_provider': 'moondream', 'depth_format': 'norm_1m', 'affordance_points': True, 'ovr_bbox': None, 'ovr_obj': None, 'attempts': 2, 'max_planner_iter': 4, 'no_plan': False, 'reset_eef': False, 'prepend_prompt': None, 'command': ['open the door'], 'tracking': False, 'track_roma': False, 'tracker_provider': 'template', 'tracking_server': 'no-shutdown', 'tracker3d': 'depth_fusion', 'track_cameras': ['head', 'wrist'], 'track_interval': 1, 'track_camera_fps': 30.0, 'track_latency': 'measured', 'track_save_depth': False, 'track_log_dir': './outputs/tracking', 'skills': True, 'skills_dir': None, 'list_skills': False, 'viz_point': None, 'vis_traj': False, 'vis_grasp': False, 'vis_box': None, 'save_grasp_inputs': False, 'replay_log': None, 'replay_vlm_review': False, 'learn_from_trajs': None, 'images_root': 'images_door_hidden_gs_touch', 'delete_images': True}
```
</details>

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
  - Sim world-state (EVALUATION ONLY, not seen by the agent): `eef_pos=[-0.302, 0.395, 0.758] gripper_open=True gripper_width=0.08 holding=False rotation=1.826 objects={'door_handle_pos': [-0.098, -0.013, 0.672], 'latch_pos': [-0.098, -0.013, 0.672], 'hinge_pos': [-0.399, -0.361, 0.697], 'pole_pos': [36.441, 310.368, 0.15], 'door_angle_deg': 0.0}`

## Files

- `00_start.png`
- `01_s1_a1_SUCCESS.png`
- `02_s2_a1_FAIL.png`
- `99_finish_INCOMPLETE.png`
- `s1_a1/` (SUCCESS; head/wrist start+end frames, 14 pointing/segmentation image(s))
- `s2_a1/` (FAIL; head/wrist start+end frames, 10 pointing/segmentation image(s))
- `videos/` (per-attempt head/wrist clips + full run, H.264; `metadata.jsonl` has 8 rows)
- `vlm_traj.log` (full run log)
- `perception/` (per planner turn: `scene_analysis_head_<step>.png` + `affordance_points_*_<step>.png`)

## Root causes (analysis; eval truth used for diagnosis only)

- Stopped at the 1-hour cap during planner iteration 2 (subtask 2, attempt 2). Door never opened: eval `door_angle_deg` peak 10.8, final 0.0; reviewer saw about 7 deg.
- **The Genesis top-down lever grasp is unreliable.** Most closes ended at 0.5-1 mm (nothing held) or 50-78 mm (open over the lever). Roughly 1 in 3+ grasps held.
- **Hand sweep without door motion.** The touch pull logged 40 deg and `guarded_pull` logged 48-49 deg (55 cm of travel), while vision read 0.6-0.8 deg. The hand slid along / off the lever; the door did not follow. The swept angle computed from the hand path is not a door angle when the grasp slides.
- The A/B 61 deg result (harness) used a SIDE grasp. A side grasp (`approach_yaw`) is the likely fix on Genesis.
- No privileged info was used by the agent; the world-state lines are logs only.
