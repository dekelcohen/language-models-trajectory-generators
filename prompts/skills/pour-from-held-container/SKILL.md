---
name: pour-from-held-container
scope: subtask
description: >
  Pour from (tilt / tip / empty) a container that the gripper is ALREADY HOLDING - kettle,
  pitcher, jug, cup, bottle, watering can - onto or into a target (a surface, bowl, sink, plant).
  Gives the pour_pose() helper, the only tilt axis that keeps the object in the fingers, and
  where to put the hand so the spout - not the hand - ends up over the target.
  Not for grasping or lifting the container (do that first).
license: MIT
metadata:
  category: manipulation
  tags: [pour, tilt, kettle, pitcher, cup, bottle, held-object]
---

# Pouring from a held container

A pour is: **carry (upright) → tilt slowly → hold → tilt back upright**, all with the gripper
closed. Pours fail for three reasons: the object rotates out of the fingers (wrong tilt axis or
too much tilt), the container or arm hits something (the target's edge, shelves/cabinets above
it, small fixtures), or the spout ends up beside the target because the HAND - not the spout -
was placed over it.

## 0. Pre-conditions

- The gripper state line must say **CLOSED and HOLDING**. If it says OPEN or closed on
  nothing, the container is not in the hand: report failure (the planner will re-grasp) - do
  not pour with an empty hand.
- **Never call `open_gripper()` in a pour subtask**, and do not go back down to re-grasp.
- Keep the `rotation` the container was grasped with (from the earlier grasp code / its
  description; for a top-down grasp of a kettle handle it is usually close to the rotation
  the current pose line gives). Only `pour_pose` changes the orientation.

## 1. The helper

```python
pose6 = pour_pose(x, y, z, rotation, tilt, pour_yaw)   # -> [x, y, z, roll, pitch, yaw]
```

- `x, y, z`: hand position, same meaning as in a length-4 pose. Keep it fixed while tilting.
- `rotation`: the grasp rotation (unchanged).
- `tilt`: radians, 0 = upright. **Use 0.7-0.85 (40-50 deg).** Beyond ~1.0 a container held by
  a pinched handle tends to rotate out of the fingers.
- `pour_yaw`: world azimuth `atan2(dy, dx)` of the side that goes DOWN (where the spout /
  opening points). It must be **perpendicular to the fingers' closing direction**: only
  `rotation + pi/2` or `rotation - pi/2` (within 30 deg) is accepted. Tilting about any other
  axis rolls the object out of the grasp, and `pour_pose` raises `ValueError` saying so.
  - Pick the one of the two that points from the hand toward the target. For a kettle grasped
    across its top handle bar, the spout lies along the bar, i.e. on one of these two sides.
  - Decide which side the spout is on from `detect_object` / the scene analysis (where the
    spout is relative to the handle bar), not from a previous review's wording alone: reviews
    of small, tilted containers often misjudge which way the spout points.
  - If the spout really points AWAY from the target, do NOT spin the held container around in
    the air: a 180 deg wrist turn while holding dropped the kettle (the wrist runs out of
    range and the container swings out of the fingers). Put it back down upright, open, and
    re-grasp the handle with `rotation + pi` (same fingers' axis, spout now on the other
    side), then lift and pour.

The result is a normal length-6 pose: pass it to `generate_linear_trajectory` like any other
pose (a length-4 start and a length-6 end are interpolated smoothly).

## 2. Where to put the hand (measured: top-handle kettle in a Franka gripper)

A container hanging from a pinched handle pivots about the fingers. Tilted ~45 deg, the body
swings BACKWARD (away from `pour_yaw`), and the spout opening ends up only ~2-5 cm behind the
hand horizontally and ~0.15 m below it. So:

- **Horizontal:** command the hand **5-8 cm PAST the target's near edge** in the pour
  direction. Not over the target's centre (usually out of reach and the container's body would
  hit the target), and not before the edge (the spout would pour in front of the target).
- **Height:** carried upright, the container's bottom hangs ~0.25 m below the hand. Command the
  hand at **target top + 0.28-0.30 m** for both the carry and the tilt, so the container clears
  the target's top edge and the tilted spout stays above it.
- **Reach cap (overrides the height above):** never command the pour hand more than **0.50 m
  above the robot arm's base** (the base position is given next to the current end-effector
  position). Near that height the arm is close to full stretch and cannot hold the tilted
  wrist: the container rolls out of the fingers with nothing touching it. Measured with the
  hand ~0.6 m from the base: base + 0.49 holds (<1 cm slip), base + 0.55 slips 6 cm, base
  + 0.59 drops it. Detected box heights of large objects are often several cm too high, which
  is how `top + 0.29` lands above the cap - trust the cap.
- **Overhead clearance:** shelves / wall cabinets above the target limit how high and how far
  the arm can go, and the wrist rises when tilting. If the end-effector stalls short of the
  commanded point, or the clips show the arm touching a cabinet, lower 2-3 cm and/or move the
  hand 2-3 cm back toward the robot, then retry.
- **Reach:** points far from the robot base (more than ~0.75 m horizontally at this height) are
  not reachable, and a hanging container plus a tilted wrist need extra room. Pour at the part
  of the target's near edge **closest to the robot base**, moved ~5 cm inward along the edge so
  the spout is over the surface and not over its corner (e.g. the near corner of a microwave /
  shelf top). Measured: pouring at the near corner holds the container with ~1-2 cm slip, while
  aiming 15 cm further along the same edge dragged the arm out of reach and the kettle fell.
- Keep the container clear of small fixtures next to the path (switches, knobs, handles): rise
  first, then move horizontally, rather than sweeping past them low.

## 3. Workflow (one code block after the detections)

```python
# hand: current EE [x, y, z]; base_z: the arm base's z (given with the EE position); rot: the grasp rotation
top_z = target_top_z                          # position z + height / 2 of the target
pour_yaw = rot + math.pi / 2                  # or rot - math.pi / 2: the side facing the target
d = [math.cos(pour_yaw), math.sin(pour_yaw)]
# (near_edge_x, near_edge_y): the point of the target's near edge closest to the robot base,
# ~5 cm inward along the edge; then 5-8 cm past the edge along d
px, py = near_edge_x + 0.06 * d[0], near_edge_y + 0.06 * d[1]
pz = min(top_z + 0.29, base_z + 0.50)        # base_z: the robot arm's base height (reach cap)

start = [hand[0], hand[1], hand[2], rot]
up_high = [hand[0], hand[1], max(hand[2], pz), rot]        # rise first
carry = [px, py, pz, rot]                                  # upright, above the pour point
poured = pour_pose(px, py, pz, rot, 0.8, pour_yaw)         # ~45 deg tilt

t1 = generate_linear_trajectory("rise with the container upright", start, up_high)
t2 = generate_linear_trajectory("carry upright to the pour point past the target's near edge", up_high, carry)
t3 = generate_linear_trajectory("tilt slowly to pour toward the target", carry, poured, num_points=30)
t4 = generate_linear_trajectory("hold the pour", poured, poured, num_points=15)
t5 = generate_linear_trajectory("tilt back upright", poured, carry, num_points=30)
for t in (t1, t2, t3, t4, t5):
    execute_trajectory(t)
```

End upright and still holding, unless the task says to put the container down afterwards (then
do that as a separate, explicit step).

## 4. Failure triage

| Symptom (clips / gripper state) | Cause | Fix |
|---|---|---|
| Gripper state becomes closed on nothing; container fell | wrong tilt axis, too much tilt, hand above the reach cap, or it hit something while tilting | `pour_yaw` = rot +/- pi/2; tilt 0.7; hand <= base z + 0.50; more clearance above the target |
| Container knocks the target's edge during the carry | hand too low | +3 cm height; rise before moving horizontally |
| Spout pours in front of the target | hand placed before the edge | move the hand further along `pour_yaw` (5-8 cm past the near edge) |
| EE stops short / arm touches the cabinet above | overhead clearance or reach | lower 2-3 cm, come 2-3 cm closer to the robot |
| `ValueError` from `pour_pose` | pour direction not perpendicular to the fingers | use rot +/- pi/2; if the spout is on the wrong side, set down and re-grasp with rot + pi (section 1) |
