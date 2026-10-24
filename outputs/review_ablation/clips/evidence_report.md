| scenario | src | angle meas/true | seen | align cos (ref) | flags | rec source | rec err deg |
|---|---|---|---|---|---|---|---|
| wide_open | real | 38.4 / 50.0 | 4/70 | 0.88 (handle_motion) | - | gripper_motion_while_holding | 21.8 |
| wide_open | oracle | 49.0 / 50.0 | 70/70 | 1.0 (handle_motion) | - | arc_tangent | 5.0 |
| slight_open | real | 9.7 / 8.0 | 10/70 | 0.99 (handle_motion) | - | arc_tangent | 2.2 |
| slight_open | oracle | 10.9 / 8.0 | 70/70 | 0.99 (handle_motion) | - | arc_tangent | 6.4 |
| slip_mid_arc | real | 19.3 / 12.0 | 12/70 | 0.94 (handle_motion) | handle_left_gripper, handle_did_not_follow_gripper | arc_tangent | 10.8 |
| slip_mid_arc | oracle | 10.5 / 12.0 | 70/70 | 0.94 (handle_motion) | handle_left_gripper, handle_did_not_follow_gripper | handle_motion | 3.1 |
| latched_slip | real | 0.0 / 0.0 | 15/70 | 1.0 (door_face_normal) | handle_left_gripper, handle_did_not_follow_gripper | face_normal | 21.7 |
| latched_slip | oracle | 0.0 / 0.0 | 70/70 | 1.0 (door_face_normal) | handle_left_gripper, handle_did_not_follow_gripper | face_normal | 21.7 |
| wrong_dir_lost | real | None / 0.0 | 1/70 | -0.01 (door_face_normal) | handle_did_not_follow_gripper, pull_direction_wrong | face_normal_after_wrong_dir | 19.8 |
| wrong_dir_lost | oracle | 0.0 / 0.0 | 70/70 | -0.01 (door_face_normal) | handle_left_gripper, handle_did_not_follow_gripper, pull_direction_wrong | face_normal_after_wrong_dir | 19.8 |
