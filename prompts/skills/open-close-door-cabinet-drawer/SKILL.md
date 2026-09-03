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
  - *fixed bar / D-handle* - rigid, nothing to press; grasp and pull. Note whether it is
    mounted **horizontally** (grasp top-down as usual) or **vertically** (`Height` much
    larger than its footprint) - a vertical bar needs the side approach in section 2b.
  - *knob* - round, small; grasp across it.
  - *recessed lip / edge* - no handle; hook the gripper on the lip.
- **Mechanism**
  - *hinged* - the panel swings about a vertical (or horizontal, e.g. a microwave) hinge.
  - *sliding* - the panel translates sideways in a track (to the left or right)
  - *drawer* - the panel translates straight toward you.
- **Geometry** - from the printed `detect_object` output for the handle, keep its position,
  `angle_long` (orientation along its longer side) and `angle_short` (along its shorter side).
  Note which edge of the door the handle sits near: the hinge is on the **opposite** edge.

Then follow exactly one of sections 2, 3 or 4, and check section 2b for the approach
direction.

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

## 2b. Vertical handles: side approach (orthogonal to sections 2-4)

Sections 2-4 decide *what* motion opens the thing. This section decides *from which
direction the gripper arrives*, and it applies on top of whichever of those you picked.

**The default is top-down and you must keep it.** Build every pose as
`[x, y, z, rotation]` exactly as you always have. A top-down gripper works for levers,
D-handles mounted horizontally, knobs, drawer lips and every tabletop grasp.

**Switch to a side approach only if at least one trigger below is true**, and when you do,
`print` one line naming the trigger before you generate any trajectory:

1. **Vertical bar.** The handle part's printed `Height` is more than ~1.5x its larger
   footprint dimension (`Width` / `Length`). A top-down gripper would have to close on the
   bar's small end cap, which is not a grasp.
2. **No top access.** The part's top face is blocked or unreachable from above (shelf,
   overhang, the panel above it).
3. **Proven infeasible.** A previous top-down attempt on this exact part failed for a
   geometric reason (gripper closed on air, or could not descend).

If none of these hold, stay top-down. Do not switch because a side approach "seems more
natural" - the extra freedom costs you reachability and is the usual source of failed IK.

### The helper

```python
side_grasp_pose(x, y, z, rotation, approach_yaw) -> pose
```

- `approach_yaw` - azimuth in radians of the horizontal direction the gripper **points
  toward the target**: `math.atan2(handle_y - ee_y, handle_x - ee_x)`. For a door, this is
  the inward normal of the door face; prefer the door normal when you have detected the
  door panel, since it keeps the wrist square to the panel.
- `rotation` - roll about that approach axis. **`0` closes the fingers horizontally, which
  is what a vertical bar needs.** Use `math.pi / 2` only for a *horizontal* bar that you
  are forced to reach from the side.

Never write orientation angles by hand; the returned pose is longer than the top-down one
and `generate_linear_trajectory` / `execute_trajectory` accept either length. You may chain
a top-down hover straight into a side pose - the orientation is interpolated for you - but
do that **away from the door**, never while the fingers are near the panel.

### Procedure for a vertical bar handle

> 1. Stand off: hover at the handle's height but backed off ~0.20 m along `-approach_yaw`
>    (away from the door), already in the side orientation. This is where the wrist rotates,
>    in free space.
> 2. Advance horizontally along `+approach_yaw` until the open fingers straddle the bar.
>    Straight line, no vertical component.
> 3. `close_gripper()`.
> 4. Perform the section 2 / 3 / 4 motion (arc about the hinge, slide along the track, or
>    straight pull), keeping the same side orientation throughout.
> 5. `open_gripper()`, then retreat backwards along `-approach_yaw` before doing anything else.

Check the bar's thickness (its smaller footprint dimension) is < 0.08 before committing.

### Worked example - vertical bar handle on a hinged cabinet door

Given `detect_object("cabinet door handle")` printed Position `[0.35, 0.62, 0.95]`,
Width `0.03`, Length `0.04`, Height `0.24`, and `detect_object("cabinet door")` printed
Position `[0.10, 0.70, 0.95]`, Width `0.60`, Length `0.03`:

```python
import math

handle_pos = [0.35, 0.62, 0.95]
handle_w, handle_l, handle_h = 0.03, 0.04, 0.24       # detect_object("cabinet door handle")
door_center = [0.10, 0.70, 0.95]
door_w, door_l = 0.60, 0.03                            # detect_object("cabinet door")

# Trigger 1: height 0.240 > 1.5 * max(0.030, 0.040) -> vertical bar, top-down not feasible.
print(f"Side approach: trigger 1 - vertical bar (height {handle_h:.3f} vs footprint "
      f"{handle_w:.3f} x {handle_l:.3f}).")
assert min(handle_w, handle_l) < 0.08, "Bar too thick for the gripper"

# The gripper must travel along the panel's FACE NORMAL, never along the panel.
# The normal is the panel's thin horizontal axis (here Length 0.03 << Width 0.60 -> the y
# axis), and it points from the handle toward the panel centre (handle y=0.62, panel
# y=0.70 -> the gripper travels in +y).
if door_l < door_w:
    normal = [0.0, math.copysign(1.0, door_center[1] - handle_pos[1])]
else:
    normal = [math.copysign(1.0, door_center[0] - handle_pos[0]), 0.0]
approach_yaw = math.atan2(normal[1], normal[0])        # = +pi/2 for this panel

# rotation=0 -> fingers close horizontally, across the vertical bar.
grasp = side_grasp_pose(handle_pos[0], handle_pos[1], handle_pos[2], 0.0, approach_yaw)

standoff = 0.20
standoff_pose = side_grasp_pose(handle_pos[0] - standoff * normal[0],
                                handle_pos[1] - standoff * normal[1],
                                handle_pos[2], 0.0, approach_yaw)

open_gripper()

# T1: get to the stand-off pose, rotating into the side orientation in free space.
current_pose = [0.00, 0.60, 0.55, 0.0]                 # the EE position from ENVIRONMENT SET-UP, top-down
t1 = generate_linear_trajectory("T1: rotate to side approach, backed off from the door",
                                current_pose, standoff_pose, 20)
execute_trajectory(t1)

# T2: advance along the face normal onto the bar. No vertical component.
t2 = generate_linear_trajectory("T2: advance onto the vertical handle", standoff_pose, grasp, 10)
execute_trajectory(t2)

close_gripper()
```

The opening motion (T3 onward) is then whichever of sections 2 / 3 / 4 applies, with every
waypoint built by `side_grasp_pose(..., 0.0, approach_yaw)` so the wrist keeps its
orientation while the hand follows the hinge arc or the slide axis.

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
this order: handle position, the `angle_long` used for the rotation, whether the handle is a
vertical bar that needs the section 2b side approach, the short-axis offset into the
lever/door gap, the descent depth, the unlatch pressure, phase separation of the approach,
hinge position, arc radius, and travel distance.

Write the subtask's success criteria so the reviewer can tell these apart, e.g.
"Success = the door is visibly open (hinge angle clearly increased). If it is not, report
whether the gripper was still holding the handle at the end."

