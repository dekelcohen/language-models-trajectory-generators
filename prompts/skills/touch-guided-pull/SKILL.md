---
name: touch-guided-pull
scope: subtask
description: >
  Open a door / cabinet / drawer AFTER its handle is grasped, by force feedback: the hand pulls
  with a limited force and lets the hinge or slide steer it (no arc math), detects slip and
  re-grasps. Gives open_with_touch() and pull_with_touch().
license: MIT
metadata:
  category: manipulation
  tags: [door, cabinet, drawer, handle, lever, hinge, pull, touch, force]
  requires: [touch]
---

# Touch-guided pull

Hinged LEVER (room / passage) door: call `open_door(handle_xyz)` (door skill); it grasps the lever
and runs this pull itself. Use the steps below for other handles (cabinets, drawers, bars, knobs).

1. Grasp the handle exactly as the door skill says (top-down for a lever, incl. the unlatch
   press). Check `gripper_holding()` (measured: closed fingers stalled on the handle).
2. Do NOT script the arc and do not hunt for the hinge. Keep the gripper closed and call:
   ```python
   res = open_with_touch(direction=[dx, dy, 0], approach_yaw=None, rotation=rot, handle_name="door handle")
   ```
   `[dx, dy, 0]` = initial pull direction: away from the door face (toward the robot) for hinged
   doors and drawers, along the track for sliders. A rough direction is fine: the hinge is fitted
   from the hand's motion and the wrist yaw is compliant. `rot` = the grasp `rotation` you used
   (side grasp only: `approach_yaw` = its yaw). On a slip it re-grasps (max 2x) by itself.
3. `res["outcome"]`: `opened` = done (open the gripper, retreat along -approach). `end_stop` = the
   mechanism's limit: opened if `swept_deg` >= ~30 or `travel_cm` reached the expected travel.
   `blocked` (never moved): wrong direction or still latched - use the door skill's triage.
   `slipped` / `timeout`: re-grasp from vision. `disabled` / `unsupported`: use the open-loop procedure.
   `res["hinge"]` = fitted hinge (`centre_xy`, `radius`), e.g. for closing it later.
