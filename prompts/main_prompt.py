# INPUT: [INSERT DETECT_OBJECT_TOOL] , [INSERT EE POSITION], [INSERT TASK], [INSERT IN CONTEXT EXAMPLE], [INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]


DETECT_OBJECT_TOOL = """1. detect_object(object_or_object_part: str) -> None: This function will not return anything, but only print the position, orientation, and dimensions of any object or object part in the environment. This information will be printed for as many instances of the queried object or object part in the environment. If there are multiple objects or object parts to detect, call one function for each object or object part, all before executing any trajectories. The unit is in metres.
1b. get_grasp_poses(object_name: str) -> (poses, scores): Returns pre-computed grasp pose candidates as (N,4,4) homogeneous matrices and (N,) quality scores (higher=better), sorted by descending score. Also prints the top-5 candidates summary.
Each pose is a 4x4 matrix: [[R_xx,R_yx,R_zx,T_x],[R_xy,R_yy,R_zy,T_y],[R_xz,R_yz,R_zz,T_z],[0,0,0,1]].
Key indices: pose[2,3]=Z-height of gripper; pose[2,2]=alignment of gripper approach axis with world Z (negative means pointing down, -1.0=perfectly downward).
To rank for top-down grasps above an object: select poses where pose[2,2] < -0.8 (close to  -1.0 - perfectly straight down) (gripper aimed downwards) and pose[2,3] > object_z (above target).
1c. visualize_grasp_pose(poses) -> None: Draws the given grasp pose(s) in the 3D simulation (RGB axes + orange gripper fingers). Pass a single (4,4) matrix or an (N,4,4) array.
"""
DETECT_OBJECT_TOOL_INITIAL_PLANNING = """Then, detect the necessary objects in the environment. Stop generation after this step to wait until you obtain the printed outputs from the detect_object function calls."""

NO_DETECT_OBJECT_TOOL = """1. You cannot call the detect_object(...) tool in this session. Instead, infer and use object positions, orientations, and dimensions from the conversation history and any previously printed outputs. Do not attempt to invoke detect_object."""
 
NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING  = """Infer and use necessary object positions, orientations, and dimensions from the conversation history and any previously printed outputs. Do not attempt to invoke detect_object."""

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

COLLISION AVOIDANCE:
If the task requires interaction with multiple objects:
1. Make sure to consider the object widths, lengths, and heights so that an object does not collide with another object or with the floor, unless necessary.
2. It may help to generate additional trajectories and add specific waypoints (calculated from the given object information) to clear objects and the floor and avoid collisions, if necessary.

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
5. If you want to print the calculated value of a variable to use later, make sure to use the print function to three decimal places, instead of simply writing the variable name. Do not print any of the trajectory variables, since the output will be too long.
6. Mark any code clearly with the ```python and ``` tags.\n7. Use the provided generate_linear_trajectory helper; do not redefine it. Use logger.info(PROGRESS + f"..." + ENDC) for concise status logs instead of print for routine status.
7. No need to import any of the above AVAILABLE FUNCTIONS. these are already injected into the python interpreter context

INITIAL PLANNING 1:
If the task requires interaction with an object part (as opposed to the object as a whole), describe which part of the object would be most suitable for the gripper to interact with.
[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]
INITIAL PLANNING 2:
Then, output Python code to decide which object to interact with, if there are multiple instances of the same object.
Then, describe how best to approach the object (for example, approaching the midpoint of the object, or one of its edges, etc.), depending on the nature of the task, or the object dimensions, etc.

For thin handles or narrow gaps, prefer approaching above the object center first, then performing small lateral alignment motions at hover height before descending vertically. Avoid inferring insertion-side signs from ambiguous orientation vectors when a stable workspace-relative direction or fixed offset convention is sufficient.

Then, output a detailed step-by-step plan for the trajectory, including when to lower the gripper to make contact with the object, if necessary, rotation and position of the gripper, closing the gripper.
Tasks:
  pickup: after closing the gripper --> must lift the object up some distance to be considered a successful grasp 
    ) default - 50cm above the object's top surface. Smaller lifts are failures. 
    ) Special task requirements or collisions may dictate a different lift distance
  

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
IN_CONTEXT_EXAMPLE = IN_CONTEXT_EXAMPLE_GRASP  # IN_CONTEXT_EXAMPLE_OPEN_DOOR # + OPEN_HINGED_DOOR_EXTRACT_PARAMS