---
name: open-close-door-cabinet-drawer
scope: subtask
description: >
  Open/Close a hinged or sliding door (inc house-door, cabinet, drawer ...) 

license: MIT
metadata:
  category: manipulation
  tags: [door, cabinet, drawer, handle, lever, hinge, articulated]
---

# Opening doors, cabinets and drawers

Articulated openings fail for two different reasons: a **bad grasp on the handle** (the arm
never really holds it) or a **bad opening direction** (it holds the handle and pulls the
wrong way). Everything below is organised so you can tell those apart.

## 1. Locate and classify the opening (do this first)

From the SCENE ANALYSIS, the affordance points (prefer the top-ranked point) and
`detect_object` on the handle/door part:

- **Handle type**
  - *hinged lever* - a bar that rotates downward to unlatch (room doors, some fridges).
  - *fixed bar / D-handle* - rigid, nothing to press; grasp and pull.
  - *knob* - round, small; grasp across it.
  - *recessed lip / edge* - no handle; hook the gripper on the lip.
- **Mechanism**
  - *hinged* - the panel swings about a vertical (or horizontal, e.g. a microwave) hinge.
  - *sliding* - the panel translates sideways in a track (to the left or right)
  - *drawer* - the panel translates straight toward you.
- **Geometry** - from the printed `detect_object` output for the handle, keep its position,
  `angle_long` (orientation along its longer side) and `angle_short` (along its shorter side).
  Note which edge of the door the handle sits near: the hinge is on the **opposite** edge.

Then follow exactly one of sections 2, 3 or 4.

## 2. Hinged door with a hinged lever handle (the long case)

Start from this procedure, verbatim:

> 1. Calc the pos to push down and pull level - in the middle of the door handle lever.
> 2. Safely approach from high above.
> 3. Rotate the end-effector so the fingers themselves are parallel to the lever's long axis
>    (set rotation = handle angle_long).
> 4. Calculate a small horizontal offset along the lever's short-axis vector toward the door,
>    so that when the gripper moves down, 1 jaw slides into the gap between the lever and the
>    door itself.
> 5. Descend, close to grasp the lever thickness, apply a small downward pressure to unlatch,
>    and pull a large dist using perpendicular vector math to fully open the door.

- **Pull vs push** - pull is the common case. Push only when the door face (not its edge) is
  toward the robot and nothing affords pulling. Decide from: which side the hinge/frame is on,
  which side of the door the handle is on, and which face of the door you can see.

## 3. Sliding cabinet or sliding door (short - do NOT use the lever procedure)

No lever, no unlatching, no arc.

- Grasp / Touch the handle, bar or recessed lip; 
- Pull **laterally along the slide axis**, in a straight line, parallel to the door face.
- **The direction is left or right depending on where the handle sits on the door.** The panel
  travels toward its free side: a handle near one edge means the panel slides away from that
  edge's stop and toward the open track. Confirm with the visible track/frame and the gap
  between panels, e.g. "open the right cabinet by sliding it from left to right".
- Travel the full visible track length; stop before the end stop.

## 4. Drawer (short)

- Grasp the handle/lip, then pull **straight along the drawer face normal**, i.e. directly away
  from the cabinet face, with no rotation and no arc.
- Keep the pull axis perpendicular to the drawer front; a skewed pull jams the slides.

## 5. Failure triage: is the direction wrong, or the pose?

Before changing anything on a retry, decide which of the two failures happened. Change the
**opening DIRECTION only when all three hold**:

1. The grasp was stable - the gripper closed on the handle and still held it at the end of the
   motion (it did not slip off, and it did not close on air).
2. The door/drawer did **not** move at all.
3. The executed pull/push trajectory was geometrically sensible for the mechanism (correct
   plane, sane length, no collision that stopped it early).

Otherwise treat it as a **miscalculation and fix the current direction** instead - re-check, in
this order: handle position, the `angle_long` used for the rotation, the short-axis offset into
the lever/door gap, the descent depth, the unlatch pressure, phase separation of the approach,
hinge position, arc radius, and travel distance.

Write the subtask's success criteria so the reviewer can tell these apart, e.g.
"Success = the door is visibly open (hinge angle clearly increased). If it is not, report
whether the gripper was still holding the handle at the end."

