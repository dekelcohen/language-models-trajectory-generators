---
name: pour-from-held-container
scope: subtask
description: >
  Pour from a container the gripper is ALREADY HOLDING (kettle, pitcher, jug, cup, bottle)
  into a container (mug, bowl) or onto a surface. Gives pour_pose() and where to put the hand.
  Not for grasping/lifting (do that first).
license: MIT
metadata:
  category: manipulation
  tags: [pour, tilt, kettle, mug, held-object]
---

# Pouring from a held container

Pour = **rise -> carry upright -> tilt -> hold -> tilt back**, gripper closed throughout.

## Rules
- Gripper state must say CLOSED and HOLDING; otherwise report failure (the planner re-grasps).
  **Never call `open_gripper()`** while pouring.
- Keep the grasp `rotation`; only `pour_pose` changes orientation:
  `pour_pose(x, y, z, rotation, tilt, pour_yaw) -> [x, y, z, roll, pitch, yaw]`
  - `tilt` 0.8 rad. More may drop the container.
  - `pour_yaw` (side going down = spout side) must be `rotation + pi/2` or `rotation - pi/2`
    (any other axis rolls it out of the fingers; `pour_pose` raises `ValueError`). Pick the one
    pointing from the hand toward the target.
  - If the spout faces away from the target, do not spin it in the air: set it down, open,
    re-grasp with `rotation + pi`.
- **Hand position** (the container hangs below and swings back when tilted; the spout ends
  ~3-5 cm behind the hand and ~0.15 m below it):
  - into a mug/bowl: hand **~3 cm past the target's centre** along `pour_yaw`;
  - onto a surface: hand **5-8 cm past the target's near edge** (the part nearest the robot base).
  - height: **target top + 0.30**, but **never above robot base z + 0.50** (the arm is at full
    stretch there and the container slips out).
- Rise first, then move horizontally, so the container clears the target's rim and fixtures.

## Code
```python
# hand: current EE [x, y, z]; rot: grasp rotation; base_z: robot base z; (tx, ty, top_z): target centre / top
pour_yaw = rot + math.pi / 2                       # or rot - math.pi / 2: the side facing the target
d = [math.cos(pour_yaw), math.sin(pour_yaw)]
px, py = tx + 0.03 * d[0], ty + 0.03 * d[1]        # mug/bowl; for a surface use its near edge + 0.06
pz = min(top_z + 0.30, base_z + 0.50)
start = [hand[0], hand[1], hand[2], rot]
up = [hand[0], hand[1], max(hand[2], pz), rot]
carry = [px, py, pz, rot]
poured = pour_pose(px, py, pz, rot, 0.8, pour_yaw)
for t in (generate_linear_trajectory("rise upright", start, up),
          generate_linear_trajectory("carry upright over the target", up, carry),
          generate_linear_trajectory("tilt to pour", carry, poured, num_points=30),
          generate_linear_trajectory("hold the pour", poured, poured, num_points=30),
          generate_linear_trajectory("tilt back upright", poured, carry, num_points=30)):
    execute_trajectory(t)
```
End upright and still holding unless the task says to put it down (separate step).

## If it fails
| Symptom | Fix |
|---|---|
| Container fell (closed on nothing) | `pour_yaw` = rot +/- pi/2; hand <= base z + 0.50; more clearance |
| Container hit the target / knocked it over | rise more before moving; +3 cm height |
| Poured in front of / past the target | move the hand along / against `pour_yaw` by the miss distance |
| EE stops short or arm touches a cabinet | lower 2-3 cm, move 2-3 cm toward the robot |
