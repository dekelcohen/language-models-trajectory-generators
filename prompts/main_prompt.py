# INPUT: [INSERT DETECT_OBJECT_TOOL] , [INSERT EE POSITION], [INSERT TASK], [INSERT IN CONTEXT EXAMPLE], [INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING], [INSERT COLLISION AVOIDANCE], [INSERT SCENE ANALYSIS], [INSERT INITIAL PLANNING 1], [INSERT INITIAL PLANNING 2]


DETECT_OBJECT_TOOL = """1. detect_object(object_or_object_part: str) -> None: This function will not return anything, but only print the position, orientation, and dimensions of any object or object part in the environment. This information will be printed for as many instances of the queried object or object part in the environment. If there are multiple objects or object parts to detect, call one function for each object or object part, all before executing any trajectories. The unit is in metres.
1b. 
"""

GET_GRASP_POSES = """get_grasp_poses(object_name: str) -> (poses, scores): Returns pre-computed grasp pose candidates as (N,4,4) homogeneous matrices and (N,) quality scores (higher=better), sorted by descending score. Also prints the top-5 candidates summary.
Each pose is a 4x4 matrix: [[R_xx,R_yx,R_zx,T_x],[R_xy,R_yy,R_zy,T_y],[R_xz,R_yz,R_zz,T_z],[0,0,0,1]].
Key indices: pose[2,3]=Z-height of gripper; pose[2,2]=alignment of gripper approach axis with world Z (negative means pointing down, -1.0=perfectly downward).
To rank for top-down grasps above an object: select poses where pose[2,2] < -0.8 (close to  -1.0 - perfectly straight down) (gripper aimed downwards) and pose[2,3] > object_z (above target).
1c. visualize_grasp_pose(poses) -> None: Draws the given grasp pose(s) in the 3D simulation (RGB axes + orange gripper fingers). Pass a single (4,4) matrix or an (N,4,4) array.
"""

DETECT_OBJECT_TOOL_INITIAL_PLANNING = """Then, detect the necessary objects in the environment. Stop generation after this step to wait until you obtain the printed outputs from the detect_object function calls."""

NO_DETECT_OBJECT_TOOL = """1. You cannot call the detect_object(...) tool in this session. Instead, infer and use object positions, orientations, and dimensions from the conversation history and any previously printed outputs. Do not attempt to invoke detect_object."""
 
NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING  = """Infer and use necessary object positions, orientations, and dimensions from the conversation history and any previously printed outputs. Do not attempt to invoke detect_object."""

# --- Shared planning sections (reused by both the subtask MAIN_PROMPT and the PLANNER_PROMPT) ---
# Common code-block + logging conventions. Reused via the [INSERT CODE BLOCK CONVENTIONS]
# placeholder in both the subtask MAIN_PROMPT and the PLANNER_PROMPT.
CODE_BLOCK_CONVENTIONS = """Mark any code clearly with the ```python and ``` tags. No need to import any of the AVAILABLE FUNCTIONS listed above - they (and `logger`, PROGRESS, ENDC) are already injected into the Python interpreter.
If you want to print a value to reuse later, use print(...) (floats to three decimal places) rather than writing the variable name alone.
Use logger.info(PROGRESS + f"..." + ENDC) for concise status logs instead of print for routine status. NOTE: `logger` is an already-injected function/object (loguru), NOT a module - do NOT run `import logger` (it will raise ModuleNotFoundError) and do not re-initialize it."""

COLLISION_AVOIDANCE = """COLLISION AVOIDANCE:
If the task requires interaction with multiple objects:
1. Make sure to consider the object widths, lengths, and heights so that an object does not collide with another object or with the floor, unless necessary.
2. It may help to generate additional trajectories and add specific waypoints (calculated from the given object information) to clear objects and the floor and avoid collisions, if necessary."""

# INITIAL_PLANNING_1 intentionally excludes the [INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING] line,
# which is tool/attempt-specific and only belongs in the subtask MAIN_PROMPT.
INITIAL_PLANNING_1 = """INITIAL PLANNING 1:
If the task requires interaction with an object part (as opposed to the object as a whole), describe which part of the object would be most suitable for the gripper to interact with."""

INITIAL_PLANNING_2 = """INITIAL PLANNING 2:
Then, output Python code to decide which object to interact with, if there are multiple instances of the same object.
Then, describe how best to approach the object (for example, approaching the midpoint of the object, or one of its edges, etc.), depending on the nature of the task, or the object dimensions, etc.

For thin handles or narrow gaps, prefer approaching above the object center first, then performing small lateral alignment motions at hover height before descending vertically. Avoid inferring insertion-side signs from ambiguous orientation vectors when a stable workspace-relative direction or fixed offset convention is sufficient.

Then, output a detailed step-by-step plan for the trajectory, including when to lower the gripper to make contact with the object, if necessary, rotation and position of the gripper, closing the gripper.
Tasks:
  pickup: after closing the gripper --> must lift the object up some distance to be considered a successful grasp 
    ) default - 50cm above the object's top surface. Smaller lifts are failures. 
    ) Special task requirements or collisions may dictate a different lift distance
  push/move: may approach the object from the side (depending on push direction) and make contact
  topple: approach the object from the side near its max height to have it fall 
  
"""

MAIN_PROMPT = """
You are a sentient AI that can control a robot arm by generating Python code which outputs a list of trajectory points for the robot arm end-effector to follow to complete a given user command.
Each element in the trajectory list is an end-effector pose, and should be of length 4, comprising a 3D position and a rotation value.

AVAILABLE FUNCTIONS:
You must remember that this conversation is a monologue, and that you are in control. I am not able to assist you with any questions, and you must output the final code yourself by making use of the available information, common sense, and general knowledge.
You are, however, able to call any of the following Python functions, if required, as often as you want:
[INSERT DETECT_OBJECT_TOOL]
2. open_gripper() -> None: This function will open the gripper on the robot arm, and will also not return anything.
3. close_gripper() -> None: This function will close the gripper on the robot arm, and will also not return anything.
4. task_completed() -> None: Call this function only when the task has been completed. This function will also not return anything.
5. generate_linear_trajectory(desc: str, start_pose: list, end_pose: list, num_points: int = 20) -> Trajectory
   class Trajectory:
     self.points # a straight-line end-effector trajectory between two 4D poses [x,y,z,theta]
     self.desc # short sentence to describe the motion and its end_pose
   This helper is provided by the environment and already logs motion details. do not call logger for trajectory/motion. 
6. execute_trajectory(trajectory: Trajectory) -> None: This function will execute the trajectory on the robot arm end-effector, and will also not return anything.
ENVIRONMENT SET-UP:
[INSERT 3D COORDINATES PROMPT SECTION]

The robot arm end-effector is currently positioned at [INSERT EE POSITION], with the rotation value at 0, and the gripper open.
The robot arm is in a top-down set-up, with the end-effector facing down onto a floor. The end-effector is therefore able to rotate about the z-axis, from -pi to pi radians.
The end-effector gripper has two fingers, and they are currently parallel to the x-axis.
The gripper can only grasp objects along sides which are shorter than 0.08.
Negative rotation values represent clockwise rotation, and positive rotation values represent anticlockwise rotation. The rotation values should be in radians.

GRIPPER ORIENTATION DEFINITION:
- The rotation value specifies the direction of the gripper's CLOSING MOTION (i.e., the direction along which the fingers move when closing).
- The gripper fingers themselves are perpendicular to this direction.
- Therefore, to grasp an object along a given side, the rotation must be aligned with that side.

Example:
- To grasp the shorter side of an object, set rotation = angle_short.
- To grasp the longer side, set rotation = angle_long.

[INSERT COLLISION AVOIDANCE]

VELOCITY CONTROL:
1. The default speed of the robot arm end-effector is 20 points per trajectory. If the total distance covered is small, keep the number of points low, as the task in sim-env has max-number-of-steps (~500-700) and each point is translated to several steps.
2. If you need to make the end-effector follow a particular trajectory more quickly, then generate fewer points for the trajectory, and vice versa.

CODE GENERATION:
When generating the code for the trajectory, do the following:
1. Describe briefly the shape of the motion trajectory required to complete the task.
2. The trajectory could be broken down into multiple steps. In that case, each trajectory step (at default speed) should contain at least 5 points. Define general functions which can be reused for the different trajectory steps whenever possible, but make sure to define new functions whenever a new motion is required. Output a step-by-step reasoning before generating the code.
3. If the trajectory is broken down into multiple steps, make sure to chain them such that the start point of trajectory_2 is the same as the end point of trajectory_1 and so on, to ensure a smooth overall trajectory. Call the execute_trajectory function after each trajectory step.
3b. For contact-rich manipulation tasks (for example handles, drawers, doors, switches, or articulated objects), prefer decomposing the motion into separate phases: high hover approach, lateral alignment at safe height, vertical descent, contact/grasp, articulation motion, and retreat. Avoid combining lateral insertion and vertical contact motions into a single diagonal approach whenever possible.
4. When defining the functions, specify the required parameters, and document them clearly in the code. Make sure to include the orientation parameter in both definition and calls (use). make sure all dimensions of caller arguments match the function definition and body
5. Do not print any of the trajectory variables, since the output will be too long.
6. [INSERT CODE BLOCK CONVENTIONS]
7. Make sure all used variables in a python block are defined in this block. you can merge several blocks if appropriate. Use the provided generate_linear_trajectory helper; do not redefine it.

SCENE ANALYSIS (produced by a separate perception VLM from the current head-camera image; use it as your observation of the scene):
[INSERT SCENE ANALYSIS]
If available, Perfer the first (top) target-object position from Affordace-pointing over the Position of <object name>.

[INSERT INITIAL PLANNING 1]
[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]
[INSERT INITIAL PLANNING 2]

The user command is "[INSERT TASK]".

[INSERT IN CONTEXT EXAMPLE]

Instructions:
Finally, perform each of these steps one by one. Name each trajectory variable with the trajectory number.


"""


IN_CONTEXT_EXAMPLE_OPEN_DOOR = """
Here is an in context example of opening a door
--------------------------------------------------------------------------
The user command is: Open the door using the lever handle
Desc: It is a horizontal lever handle attached to a door
 
Preception module outputs:

Predicting 1 masks
Predicted 1 masks

Object detected: door handle lever

Position of door handle lever: [handle_x, handle_y, handle_z]

Dimensions:
- Width: handle_width
- Length: handle_length
- Height: handle_height

Orientation:
- Orientation along shorter side (thickness): angle_short
- Orientation along longer side (lever axis): angle_long

Assistant (Reasoning and Plan):

Approach decision and plan:

Only one door handle lever is detected, so we will interact with that instance.

Best approach:
1. Move to a safe hover directly above the lever midpoint.
2. Perform any small lateral alignment motions while remaining at the safe hover height.
3. Descend vertically onto the lever.
4. Apply a small downward press to unlatch.
5. Pull outward in a stable workspace direction to open the door.

For thin handles and narrow insertion gaps, separating hover alignment from vertical descent is more stable than approaching diagonally.

Gripper orientation:
The gripper fingers should become parallel to the lever’s long axis. Since the gripper closing direction defines the rotation axis and the fingers are perpendicular to the closing direction, set:

rotation = angle_short

This aligns the closing motion with the lever thickness axis while making the fingers parallel to the lever long axis.

Insertion strategy:
Use a small horizontal offset along the lever short-axis direction so one jaw can slide into the gap between the lever and the door plate during vertical descent.

Choose the short-axis offset sign so the gripper shifts toward the door plate / handle support side. This lets one jaw enter the gap between the lever and the plate and prevents the handle from sliding out during pulling.

Collision avoidance:
Perform all XY translation at high Z. Descend vertically near the handle. Keep articulation motions smooth and primarily along a single workspace direction.

Step-by-Step Trajectory Execution

Step 1: Move to a safe hover above the lever midpoint

```python
# Perception-derived variables
handle_position = [handle_x, handle_y, handle_z]
handle_height = handle_height
handle_thickness = handle_length

# Gripper orientation
grip_orientation = angle_short

# Current robot state
current_pose = [ee_x, ee_y, ee_z, ee_theta]

# Derived heights
top_z = handle_z + handle_height / 2
hover_clearance = hover_offset
hover_z = top_z + hover_clearance

# Hover pose directly above the handle midpoint
hover_center_pose = [handle_x, handle_y, hover_z, grip_orientation]

trajectory_1 = generate_linear_trajectory(
    "move to safe hover above handle midpoint",
    current_pose,
    hover_center_pose
)
execute_trajectory(trajectory_1)

Step 2: Translate laterally at hover height for jaw insertion alignment

# Small insertion offset along the short-axis direction
offset_distance = handle_thickness / 2 - insertion_margin

short_axis_vector = [
    cos(angle_short),
    sin(angle_short)
]

# Choose the sign so the offset moves toward the door plate side
door_direction = [
    door_x - handle_x,
    door_y - handle_y
]

door_norm = sqrt(door_direction[0]**2 + door_direction[1]**2)

if door_norm > 1e-6:
    door_direction = [
        door_direction[0] / door_norm,
        door_direction[1] / door_norm
    ]

dot_product = (
    short_axis_vector[0] * door_direction[0] +
    short_axis_vector[1] * door_direction[1]
)

offset_sign = 1.0 if dot_product >= 0 else -1.0

offset_x = short_axis_vector[0] * offset_distance * offset_sign
offset_y = short_axis_vector[1] * offset_distance * offset_sign

hover_offset_pose = [
    handle_x + offset_x,
    handle_y + offset_y,
    hover_z,
    grip_orientation
]

trajectory_2 = generate_linear_trajectory(
    "translate at hover height for handle insertion alignment",
    hover_center_pose,
    hover_offset_pose
)
execute_trajectory(trajectory_2)

Step 3: Open the gripper

open_gripper()

Step 4: Descend to just above the handle

approach_z = top_z + surface_clearance

approach_pose = [
    hover_offset_pose[0],
    hover_offset_pose[1],
    approach_z,
    grip_orientation
]

trajectory_3 = generate_linear_trajectory(
    "descend to approach height above handle",
    hover_offset_pose,
    approach_pose
)
execute_trajectory(trajectory_3)

Step 5: Descend to grasp/contact height

grasp_z = handle_z

grasp_pose = [
    hover_offset_pose[0],
    hover_offset_pose[1],
    grasp_z,
    grip_orientation
]

trajectory_4 = generate_linear_trajectory(
    "descend to handle grasp height",
    approach_pose,
    grasp_pose
)
execute_trajectory(trajectory_4)

Step 6: Close the gripper

close_gripper()

Step 7: Apply a small downward press to unlatch

pressed_pose = [
    grasp_pose[0],
    grasp_pose[1],
    grasp_pose[2] - downward_press_distance,
    grip_orientation
]

trajectory_5 = generate_linear_trajectory(
    "apply downward pressure to unlatch handle",
    grasp_pose,
    pressed_pose
)
execute_trajectory(trajectory_5)

Step 8: Pull outward to open the door

pull_pose = [
    pressed_pose[0],
    pressed_pose[1] + pull_distance,
    pressed_pose[2],
    grip_orientation
]

trajectory_6 = generate_linear_trajectory(
    "pull outward to open door",
    pressed_pose,
    pull_pose
)
execute_trajectory(trajectory_6)

Step 9: Retreat upward and release

open_gripper()

retreat_pose = [
    pull_pose[0],
    pull_pose[1],
    hover_z,
    grip_orientation
]

trajectory_7 = generate_linear_trajectory(
    "retreat upward after opening door",
    pull_pose,
    retreat_pose
)
execute_trajectory(trajectory_7)

-------------------- End of Sample -----------------
Instructions:
Finally, perform each of these steps one by one. Name each trajectory variable with the trajectory number.

"""

IN_CONTEXT_EXAMPLE_GRASP = """
Here is an in context example of multiple LLM responses for each step
--------------------------------------------------------------------------
The user command is: Pick up the blue can
Desc: It is a blue near-cylindrical object
 
Preception module outputs:

Predicting 1 masks
Predicted 1 masks

Object detected: blue can

Position of blue can: [pos_x, pos_y, pos_z]

Dimensions:
- Width: dim_width
- Length: dim_length
- Height: dim_height

Orientation:
- Orientation along shorter side (length): angle_short
- Orientation along longer side (width): angle_long
Assistant (Reasoning and Plan):

Approach decision and plan:

Only one blue can is detected, so we will interact with that instance.

Best approach: move the end-effector directly above the can’s center, maintain a safe hover height above the top surface, then descend vertically to the mid-height of the can body to grasp the side walls.

The gripper orientation is aligned with the can’s shorter side orientation (angle_short). For a near-cylindrical object this alignment is not strictly required, but it provides consistency.

Collision avoidance: perform all XY motion above the object; descend strictly along the Z-axis.

Grasping strategy: close the gripper at mid-height, then lift vertically to confirm a stable grasp.

Step-by-Step Trajectory Execution
Step 1: Move to a safe hover pose above the can
```python

# Perception-derived variables
can_position = [pos_x, pos_y, pos_z]
can_height = dim_height
grip_orientation = angle_short

# Derived heights
top_z = pos_z + can_height / 2
hover_clearance = clearance_hover
pre_grasp_z = top_z + hover_clearance
lift_z = top_z + 50 cm up (or according to task reqs and collisions)

# Current robot state
current_pose = [ee_x, ee_y, ee_z, ee_theta]

# Target hover pose
hover_pose = [pos_x, pos_y, pre_grasp_z, grip_orientation]

trajectory_1 = generate_linear_trajectory("hover to approach target", current_pose, hover_pose)
execute_trajectory(trajectory_1)
Step 2: Open the gripper
```python
open_gripper()
```
Step 3: Descend to just above the top surface
```python
z_above_top = top_z + clearance_top

approach_pose = [pos_x, pos_y, z_above_top, grip_orientation]

trajectory_2 = generate_linear_trajectory("descend to approach height", hover_pose, approach_pose)
execute_trajectory(trajectory_2)
```
Step 4: Descend to pre-close height near grasp level
```python
z_grasp = pos_z
z_preclose = z_grasp + clearance_preclose

pre_close_pose = [pos_x, pos_y, z_preclose, grip_orientation]

trajectory_3 = generate_linear_trajectory("descend to pre-close height", approach_pose, pre_close_pose)
execute_trajectory(trajectory_3)
```
Step 5: Descend to grasp height
```python
grasp_pose = [pos_x, pos_y, z_grasp, grip_orientation]

trajectory_4 = generate_linear_trajectory("descend to grasp height", pre_close_pose, grasp_pose)
execute_trajectory(trajectory_4)
```
Step 6: Close the gripper
```python
close_gripper()
```
Step 7: Lift the object to a safe height
```python

lift_pose = [pos_x, pos_y, lift_z, grip_orientation]

trajectory_5 = generate_linear_trajectory("lift after grasp", grasp_pose, lift_pose)
execute_trajectory(trajectory_5)
```
-------------------- End of Sample -----------------
Instructions:
Finally, perform each of these steps one by one. Name each trajectory variable with the trajectory number.

"""

OPEN_HINGED_DOOR_EXTRACT_PARAMS = """

import math

def get_llm_door_prompt_data(handle_pos , pivot_pos):
    ###
    handle_pos - the door handle pos that is pulled (or pushed)
    pivot_pos - the pivot which around the rotate the door to open/close - usually door_hinge_pos 
    ###
    
    
    # 2. Calculate Radius in the XY plane (assuming door swings flat on the ground)
    dx = handle_pos[0] - pivot_pos[0]
    dy = handle_pos[1] - pivot_pos[1]
    radius = math.hypot(dx, dy)
    
    # 3. Calculate Current Angle (in radians)
    current_angle = math.atan2(dy, dx)
    
    # 4. Define target (e.g., open the door by 90 degrees / 1.57 radians)
    target_angle = current_angle + 1.5708 # Change sign (-) if door opens the other way
    
    llm_prompt_data = {
        "pivot_center_xy": [round(pivot_pos[0], 4), round(pivot_pos[1], 4)],
        "handle_start_xyz": [round(handle_pos[0], 4), round(handle_pos[1], 4), round(handle_pos[2], 4)],
        "radius_m": round(radius, 4),
        "current_angle_rad": round(current_angle, 4),
        "target_angle_rad": round(target_angle, 4)
    }
    
    return llm_prompt_data
"""

IN_CONTEXT_EXAMPLE_OPEN_DOOR = """

The user command is: "Open the door by operating the handle lever."

Perception module outputs:
Position of door: [door_x, door_y, door_z]
Dimensions: Width: door_w, Length: door_l, Height: door_h
Orientation along longer side (length): door_angle_long

Position of door handle lever: [lev_x, lev_y, lev_z]
Dimensions: Width: lev_w (long axis), Length: lev_l (short/thickness), Height: lev_h
Orientation along shorter side (length): lev_angle_short
Orientation along longer side (width): lev_angle_long

Position of door hinge: [hinge_x, hinge_y, hinge_z]   # if available; else derive from door edge


**Assistant (Reasoning and Plan):**

Interaction part: the **middle of the lever**, giving max clearance from the rosette and room for one jaw to slide into the lever–door gap.

**Critical orientation reasoning:** The rotation value specifies the gripper's *closing-motion direction*; fingers are perpendicular to it. To keep fingers **parallel to the lever's long axis**, set `rotation = lev_angle_short` (align closing motion with the lever thickness). Verify thickness `lev_l < 0.08 m` (grasp limit) before committing.

**Phase decomposition (never combine lateral + vertical into a diagonal):**
1. High hover above lever center
2. Lateral alignment (small offset toward door along short-axis) at hover height
3. Vertical descent to pre-contact
4. Vertical descent to grasp height + close
5. Downward push to unlatch
6. Pull to open along **hinge-perpendicular** vector
7. Retreat + release

```python
from math import cos, sin, sqrt, atan2, pi

# ---- Perception-derived variables (NO magic numbers; scale from dims) ----
door_pos   = [door_x, door_y, door_z]
hinge_pos  = [hinge_x, hinge_y, hinge_z]      # if unavailable, use farthest door corner from lever
lever_pos  = [lev_x, lev_y, lev_z]
lever_long = lev_w        # long axis length of the lever
lever_thk  = lev_l        # short axis = thickness to grasp
lever_h    = lev_h
lever_angle_short = lev_angle_short   # closing-motion dir => fingers parallel to long axis

current_pose = [ee_x, ee_y, ee_z, ee_theta]

# ---- Orientation ----
assert lever_thk < 0.08, "Lever thickness exceeds gripper limit"
grip_theta = lever_angle_short
print(f"grip_theta (rad): {grip_theta:.3f}")

# ---- Heights derived from lever geometry (scaled, not hard-coded) ----
lever_top_z = lever_pos[2] + lever_h / 2.0
z_high      = lever_top_z + max(0.15, 3.0 * lever_h)   # generous hover scaled to object
z_approach  = lever_top_z + 1.0 * lever_h              # just above top
z_preclose  = lever_pos[2] + 0.5 * lever_thk           # near center
z_grasp     = lever_pos[2]                             # lever mid-height
z_unlatch   = lever_pos[2] - max(0.015, 0.8 * lever_h) # push scaled to lever height
print(f"z_high {z_high:.3f} z_grasp {z_grasp:.3f} z_unlatch {z_unlatch:.3f}")

# ---- Lateral offset along lever SHORT-axis toward the door face ----
sx, sy = cos(lever_angle_short), sin(lever_angle_short)   # short-axis unit dir in XY
to_door = [door_pos[0]-lever_pos[0], door_pos[1]-lever_pos[1]]
sign = 1.0 if (sx*to_door[0] + sy*to_door[1]) >= 0 else -1.0
offset_mag = 0.5 * lever_thk        # ~half the thickness so one jaw enters the gap
off = [sign*offset_mag*sx, sign*offset_mag*sy]
print(f"offset dx {off[0]:.3f} dy {off[1]:.3f}")

# ---- Pull vector: perpendicular to the door plane about the hinge (physically correct) ----
# Door swings about hinge; radial direction lever->hinge, tangential (perpendicular) = swing dir.
radial = [lever_pos[0]-hinge_pos[0], lever_pos[1]-hinge_pos[1]]
rlen = max(1e-6, sqrt(radial[0]**2 + radial[1]**2))
radial = [radial[0]/rlen, radial[1]/rlen]
# Two perpendicular candidates; choose the one pointing toward the robot (opening toward user)
perp_a = [-radial[1], radial[0]]
perp_b = [ radial[1],-radial[0]]
to_robot = [current_pose[0]-lever_pos[0], current_pose[1]-lever_pos[1]]
pull_dir = perp_a if (perp_a[0]*to_robot[0]+perp_a[1]*to_robot[1]) >= \
                     (perp_b[0]*to_robot[0]+perp_b[1]*to_robot[1]) else perp_b
pull_distance = max(0.20, 3.0 * lever_long)   # large swing scaled to lever size
print(f"pull_dir {pull_dir[0]:.3f},{pull_dir[1]:.3f} dist {pull_distance:.3f}")

# ---- Reusable linear-motion helper ----
def move_linear(desc, start_pose, end_pose, num_points=8):
    "Straight-line EE motion between two [x,y,z,theta] poses."
    traj = generate_linear_trajectory(desc, start_pose, end_pose, num_points)
    execute_trajectory(traj)
    return end_pose


**Step 1 — High hover above lever center**
hover = [lever_pos[0], lever_pos[1], z_high, grip_theta]
p1 = move_linear("T1: high hover above lever center", current_pose, hover, 12)

**Step 2 — Lateral alignment toward door (at hover height)**
hover_off = [lever_pos[0]+off[0], lever_pos[1]+off[1], z_high, grip_theta]
p2 = move_linear("T2: lateral align toward door", p1, hover_off, 8)
open_gripper()

**Step 3 — Descend to approach height**
approach = [hover_off[0], hover_off[1], z_approach, grip_theta]
p3 = move_linear("T3: descend to approach height", p2, approach, 8)

**Step 4 — Descend to pre-close, then grasp height + close**
preclose = [hover_off[0], hover_off[1], z_preclose, grip_theta]
p4 = move_linear("T4: descend to pre-close", p3, preclose, 6)
grasp = [hover_off[0], hover_off[1], z_grasp, grip_theta]
p5 = move_linear("T5: descend to grasp height", p4, grasp, 6)
close_gripper()

**Step 5 — Downward push to unlatch**
unlatch = [grasp[0], grasp[1], z_unlatch, grip_theta]
p6 = move_linear("T6: downward push to unlatch", p5, unlatch, 6)

**Step 6 — Pull to open along hinge-perpendicular vector**
pull = [unlatch[0]+pull_dir[0]*pull_distance,
        unlatch[1]+pull_dir[1]*pull_distance,
        unlatch[2], grip_theta]
p7 = move_linear("T7: pull to open door", p6, pull, 25)   # many points => slow, controlled

**Step 7 — Retreat upward and release**
retreat = [pull[0], pull[1], z_high, grip_theta]
p8 = move_linear("T8: retreat upward", p7, retreat, 10)
open_gripper()
task_completed()
```

"""

IN_CONTEXT_EXAMPLE_OPEN_DOOR_LEARNED_29_06 = """
- Orientation mapping: Rotation encodes the closing-motion direction. To make the fingers parallel to the lever, set rotation = handle_angle_long + π/2 (not = handle_angle_long). This change was critical for a stable lever pinch across its thickness.
- Offset strategy near the door: A larger short-axis offset toward the door at hover plus a small micro-nudge at approach height reliably seats one jaw behind the lever/rosette before closing. Too-small offsets left both jaws outside.
- Grasp height: Closing slightly below the lever’s mid-height improves capture compared to closing exactly at mid-height.
- Contact-phase separation: Separate lateral alignment at safe height, vertical descent, micro-nudge, close, then press; avoid coupling lateral insertion and vertical contact in a single diagonal move.
- Post-grasp actions: Apply a decisive but bounded downward press to unlatch, then a long pull along the negative door-normal. Optionally, add a brief retreat along the negative short-axis right after closing to confirm capture before the long pull.

Generalized in-context example (parameterized; no hard-coded magic numbers)

```python
# Generalized door-opening via lever: parameterized template
# Inputs:
#   handle_center: [x, y, z] of lever midpoint
#   handle_dims:   dict with keys {"length", "thickness", "height"} in meters
#                  - length: lever long-axis extent
#                  - thickness: lever thickness to pinch across
#                  - height: vertical size of the lever assembly/rosette region
#   handle_angle_long: lever long-axis angle in radians (XY plane)
#   door_center:   [x, y, z] on the door plane near the handle
#   door_dims:     dict with keys {"width"}; width used to scale pull distance
#   gripper:       dict with keys {"max_aperture"}; e.g., {"max_aperture": 0.08}
#
# Conventions:
#   - Rotation value is the closing-motion direction; fingers are perpendicular to it.
#   - To keep fingers parallel to the lever, set rotation = handle_angle_long + π/2.
#   - Coordinate system: x increases left, y decreases away from you, z increases upward.

import math

def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))

def normalize_angle(theta):
    # Map to [-pi, pi]
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

def unit(vx, vy):
    n = math.hypot(vx, vy)
    if n == 0.0:
        return [0.0, 0.0]
    return [vx / n, vy / n]

def points_for(pose_start, pose_end, min_points=5, base_scale=40.0):
    # Heuristic: more points for longer moves; keep at least min_points
    dx = pose_end[0] - pose_start[0]
    dy = pose_end[1] - pose_start[1]
    dz = pose_end[2] - pose_start[2]
    dtheta = abs(pose_end[3] - pose_start[3])
    dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.1 * dtheta
    return max(min_points, int(clamp(dist * base_scale, min_points, 30)))

def pose(x, y, z, theta):
    return [x, y, z, theta]

def open_door_via_lever(handle_center, handle_dims, handle_angle_long, door_center, door_dims, gripper):
    # Derived unit vectors
    long_u = [math.cos(handle_angle_long), math.sin(handle_angle_long)]
    # Closing-motion direction must be perpendicular to the lever axis so fingers are parallel to lever:
    theta_work = normalize_angle(handle_angle_long + math.pi/2.0)
    short_u = [math.cos(theta_work), math.sin(theta_work)]  # closing-motion direction

    # Door-normal direction in XY plane
    to_door_xy = [door_center[0] - handle_center[0], door_center[1] - handle_center[1]]
    to_door_u = unit(to_door_xy[0], to_door_xy[1])

    # Ensure the short-axis toward the door (inner jaw toward door/rosette)
    dot_sd = short_u[0]*to_door_u[0] + short_u[1]*to_door_u[1]
    toward_door_short_u = short_u if dot_sd >= 0.0 else [-short_u[0], -short_u[1]]

    # Gripper/geometry checks
    lever_thickness = handle_dims["thickness"]
    if lever_thickness >= gripper["max_aperture"]:
        raise RuntimeError("Lever too thick for the gripper aperture; cannot pinch across thickness.")

    lever_len = handle_dims["length"]
    lever_h = handle_dims["height"]
    door_width = door_dims.get("width", None)

    # Hover and approach heights (avoid magic numbers by scaling with lever height and safety margins)
    hover_z = handle_center[2] + clamp(0.3 + 1.5 * lever_h, 0.25, 0.5)   # safe clearance above handle
    approach_z = handle_center[2] + clamp(0.1 + 0.5 * lever_h, 0.10, 0.20)
    preclose_z = handle_center[2] + clamp(0.05 * lever_h, 0.01, 0.04)
    grasp_z = handle_center[2] - clamp(0.15 * lever_h, 0.005, 0.02)       # slightly below mid-height

    # Offsets along short-axis toward door: gross offset at hover + micro-nudge at approach
    # Choose based on lever geometry and a small clearance band
    offset_hover = clamp(0.2 * lever_len, 0.02, 0.06)
    micro_nudge = clamp(0.5 * lever_thickness, 0.004, 0.010)

    # Downward press to unlatch; scaled to lever height and bounded
    press_depth = clamp(0.4 * lever_h, 0.02, 0.05)
    after_press_relief = clamp(0.2 * lever_h, 0.006, 0.02)

    # Pull distance: prefer a fraction of door width; otherwise use a conservative default
    pull_dist = clamp((0.8 * door_width) if door_width is not None else 0.4, 0.25, 0.6)
    pull_u = [-to_door_u[0], -to_door_u[1]]  # away from door

    # Optional capture-check retreat distance along negative short-axis after closing
    capture_check = clamp(0.5 * micro_nudge, 0.002, 0.006)

    # Report computed params (3 decimals) for debugging
    print(f"theta_work = {theta_work:.3f} rad")
    print(f"hover_z = {hover_z:.3f} m, approach_z = {approach_z:.3f} m, preclose_z = {preclose_z:.3f} m, grasp_z = {grasp_z:.3f} m")
    print(f"offset_hover = {offset_hover:.3f} m, micro_nudge = {micro_nudge:.3f} m")
    print(f"press_depth = {press_depth:.3f} m, after_press_relief = {after_press_relief:.3f} m")
    print(f"pull_dist = {pull_dist:.3f} m")
    print(f"to_door_u = [{to_door_u[0]:.3f}, {to_door_u[1]:.3f}], short_u(toward door) = [{toward_door_short_u[0]:.3f}, {toward_door_short_u[1]:.3f}]")

    # Start from current end-effector pose (provided by the environment)
    current_pose = CURRENT_EE_POSE  # assumed injected by environment as [x, y, z, theta]

    # Ensure gripper open
    open_gripper()

    # trajectory1: Lift to hover at current XY
    target1 = pose(current_pose[0], current_pose[1], hover_z, current_pose[3])
    trajectory1 = generate_linear_trajectory("Lift to high hover at current XY", current_pose, target1, num_points=points_for(current_pose, target1))
    execute_trajectory(trajectory1)
    current_pose = target1

    # trajectory2: In-place yaw to working orientation (closing-motion ⟂ lever axis)
    target2 = pose(current_pose[0], current_pose[1], current_pose[2], theta_work)
    trajectory2 = generate_linear_trajectory("Yaw in place to working orientation (fingers parallel to lever)", current_pose, target2, num_points=points_for(current_pose, target2))
    execute_trajectory(trajectory2)
    current_pose = target2

    # trajectory3: Move to hover above handle center
    target3 = pose(handle_center[0], handle_center[1], hover_z, theta_work)
    trajectory3 = generate_linear_trajectory("Move to hover above handle center", current_pose, target3, num_points=points_for(current_pose, target3))
    execute_trajectory(trajectory3)
    current_pose = target3

    # trajectory4: Apply short-axis offset toward door at hover
    target4 = pose(current_pose[0] + offset_hover * toward_door_short_u[0],
                   current_pose[1] + offset_hover * toward_door_short_u[1],
                   hover_z, theta_work)
    trajectory4 = generate_linear_trajectory("Short-axis offset toward door at hover", current_pose, target4, num_points=points_for(current_pose, target4))
    execute_trajectory(trajectory4)
    current_pose = target4

    # trajectory5: Descend to approach height
    target5 = pose(current_pose[0], current_pose[1], approach_z, theta_work)
    trajectory5 = generate_linear_trajectory("Descend to approach height above lever", current_pose, target5, num_points=points_for(current_pose, target5))
    execute_trajectory(trajectory5)
    current_pose = target5

    # trajectory6: Micro-nudge toward door at approach height
    target6 = pose(current_pose[0] + micro_nudge * toward_door_short_u[0],
                   current_pose[1] + micro_nudge * toward_door_short_u[1],
                   approach_z, theta_work)
    trajectory6 = generate_linear_trajectory("Micro-nudge toward door at approach height", current_pose, target6, num_points=points_for(current_pose, target6, min_points=5))
    execute_trajectory(trajectory6)
    current_pose = target6

    # trajectory7: Descend to pre-close height
    target7 = pose(current_pose[0], current_pose[1], preclose_z, theta_work)
    trajectory7 = generate_linear_trajectory("Descend to pre-close height", current_pose, target7, num_points=points_for(current_pose, target7))
    execute_trajectory(trajectory7)
    current_pose = target7

    # trajectory8: Final descend to grasp height (slightly below mid-height) and close
    target8 = pose(current_pose[0], current_pose[1], grasp_z, theta_work)
    trajectory8 = generate_linear_trajectory("Descend to grasp height", current_pose, target8, num_points=points_for(current_pose, target8, min_points=6))
    execute_trajectory(trajectory8)
    current_pose = target8
    close_gripper()

    # Optional capture-check: slight retreat along negative short-axis to confirm lever is captured
    target8b = pose(current_pose[0] - capture_check * toward_door_short_u[0],
                    current_pose[1] - capture_check * toward_door_short_u[1],
                    current_pose[2], theta_work)
    trajectory8b = generate_linear_trajectory("Capture-check retreat along negative short-axis", current_pose, target8b, num_points=points_for(current_pose, target8b, min_points=5))
    execute_trajectory(trajectory8b)
    current_pose = target8b

    # trajectory9: Downward press to unlatch
    target9 = pose(current_pose[0], current_pose[1], current_pose[2] - press_depth, theta_work)
    trajectory9 = generate_linear_trajectory("Downward press to unlatch", current_pose, target9, num_points=points_for(current_pose, target9))
    execute_trajectory(trajectory9)
    current_pose = target9

    # trajectory10: Slight raise after pressing
    target10 = pose(current_pose[0], current_pose[1], current_pose[2] + after_press_relief, theta_work)
    trajectory10 = generate_linear_trajectory("Slight raise after unlatching press", current_pose, target10, num_points=points_for(current_pose, target10))
    execute_trajectory(trajectory10)
    current_pose = target10

    # trajectory11: Pull outward along negative door-normal to open
    target11 = pose(current_pose[0] + pull_dist * pull_u[0],
                    current_pose[1] + pull_dist * pull_u[1],
                    current_pose[2], theta_work)
    trajectory11 = generate_linear_trajectory("Pull outward along negative door normal to open", current_pose, target11, num_points=points_for(current_pose, target11))
    execute_trajectory(trajectory11)
    current_pose = target11

    # Release and retreat upward
    open_gripper()
    target12 = pose(current_pose[0], current_pose[1], hover_z, theta_work)
    trajectory12 = generate_linear_trajectory("Retreat upward to hover", current_pose, target12, num_points=points_for(current_pose, target12))
    execute_trajectory(trajectory12)
    current_pose = target12

    task_completed()

# Example of how to call:
# open_door_via_lever(handle_center, {"length": L, "thickness": T, "height": H},
#                     handle_angle_long, door_center, {"width": W}, {"max_aperture": 0.08})
```

"""

IN_CONTEXT_EXAMPLE = IN_CONTEXT_EXAMPLE_GRASP # IN_CONTEXT_EXAMPLE_OPEN_DOOR # IN_CONTEXT_EXAMPLE_GRASP # + OPEN_HINGED_DOOR_EXTRACT_PARAMS


