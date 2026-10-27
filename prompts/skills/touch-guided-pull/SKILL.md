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

1. Grasp the handle as the door skill says. A lever parallel to the door face (~0.1 m out) needs
   the side approach with `rotation = math.pi / 2` (jaws close vertically). Check `gripper_holding`.
2. Do NOT script the arc. Keep the gripper closed and call:
   ```python
   res = open_with_touch(direction=d, approach_yaw=yaw, rotation=math.pi / 2, handle_name="door handle")
   ```
   `d` = initial pull direction (world, horizontal): away from the door face for hinged doors and
   drawers, along the track for sliders. A rough `d` is fine: it is re-estimated from the hand's
   motion, a hinge is fitted on the way, and the wrist yaw is compliant. `approach_yaw`/`rotation`
   = the grasp you used (`approach_yaw=None` for top-down). On a slip it re-grasps (max 2x) at the
   last held position, refined by tracking `handle_name` when tracking is on.
3. `res["outcome"]`: `opened` = done (open the gripper, retreat along -approach). `end_stop` = the
   mechanism's limit: opened if `swept_deg` >= ~45 or `travel_cm` reached the expected travel.
   `blocked` (never moved): wrong direction or still latched - use the door skill's triage.
   `slipped` / `timeout`: re-grasp from vision. `disabled` / `unsupported`: use the open-loop procedure.
   `res["hinge"]` = fitted hinge (`centre_xy`, `radius`), e.g. for closing it later.
