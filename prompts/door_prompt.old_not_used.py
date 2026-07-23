DOOR_HORIZ_LEVEL_NO_ROTATION_PROMPT="""
In-context example of door opening using a horizontal lever connected to the door 
Included: planning, coords calc and setup stage - not whole execution
-------------------------------------------------------------------------------------------------------
assistant:
Assistant (Initial Planning 1: Object parts and perception)

- Target object part: the underside mid-section of the door handle lever, near its free end. This region provides a consistent gap between the lever and the door/backplate where one jaw can be inserted. It also offers good leverage for pulling without slipping.

- Perception needed:
  - Door handle lever: position, orientation (lever axis direction), dimensions (width, length, thickness).
  - Door: plane position and normal (to know which side is “towards the door” and to compute a safe insertion path), and thickness to avoid collisions.

Proceeding to detect the required objects.

```python
detect_object("door handle lever")
```
user:
Print statement output:
Position of door handle lever: [lever_pos_x, lever_pos_y, lever_pos_z]
Dimensions:
Width: lever_dim_width
Length: lever_dim_length
Height: lever_dim_height
Orientation along shorter side (length): lever_orientation_length
Orientation along longer side (width): lever_orientation_width

assistant:

```python
detect_object("door")
```

user:
Print statement output:
Position of door: [door_pos_x, door_pos_y, door_pos_z]
Dimensions:
Width: door_dim_width
Length: door_dim_length
Height: door_dim_height
Orientation along shorter side (length): door_orientation_length
Orientation along longer side (width): door_orientation_width



assistant:
Assistant (Initial Planning 2: Decision, approach strategy, and step-by-step plan)

Approach decision:
- Only one door handle lever was detected, so we will interact with that instance at its reported center.
- We will align the gripper so the jaws are perpendicular to the lever’s length. This places one jaw favorably to slide into the gap between the lever underside and the door surface during a vertical descent.

Best approach:
- Approach the midpoint of the lever from a safe hover above it.
- Maintain all horizontal motion at a high Z to avoid collisions, then descend vertically in two stages: to just above the lever and further into the gap so one jaw is between the lever and the door.
- Close the gripper to secure contact at the lever’s midsection, then pull along the positive y-direction to open the door.

Collision avoidance:
- Perform XY moves only at a high Z hover.
- Descend vertically to minimize lateral collisions.
- Use conservative clearances above and below the lever to avoid contact with the door surface.

Planned motion shape:
- Step 1: A straight-line move in 3D (mostly XY) from the current pose to a safe hover directly above the lever, while rotating the end-effector yaw to the desired angle.
- Step 2: Open gripper (safety).
- Step 3: Vertical descent to just above the lever.
- Step 4: Further vertical descent so one jaw dips between the lever and the door surface.
- Step 5: Close gripper to secure the lever.
- Step 6: Straight pull along +y to open the door.
- Step 7: Small vertical lift to relieve contact after opening.

Key computed targets:
- Hover height above lever: +0.22 m
- Just-above-lever height: +0.03 m
- Insertion height (one jaw between lever and door): -insertion_depth m below lever z
- Gripper yaw: align with the lever’s width orientation (lever_orientation_width rad), which is perpendicular to the lever’s length (lever_orientation_length rad)

Proceeding to set up utility functions, select the lever instance, and compute target poses.

```python
# Utility: straight-line trajectory generator with orientation interpolation
def generate_linear_trajectory(start_pose, end_pose, num_points=7):
   ...
   return traj

# Current robot state (given)
current_pose = [robot_start_x, robot_start_y, robot_start_z, robot_start_yaw]

# Perception results (single detected instances)
lever_pos = [lever_pos_x, lever_pos_y, lever_pos_z]
lever_len_angle = lever_orientation_length      # orientation along lever length (shorter side per perception label)
lever_width_angle = lever_orientation_width     # orientation perpendicular to length (use for jaw alignment)

door_pos = [door_pos_x, door_pos_y, door_pos_z]  # not strictly needed for the basic pull, kept for reference

# Select lever instance (single instance -> select it)
selected_lever = lever_pos

# Approach and insertion parameters (conservative clearances)
hover_clearance = hover_safety_offset     # meters above lever for safe XY motion
above_clearance = approach_offset     # just above the lever
insert_depth = insertion_depth        # how far below lever z to insert a jaw into the gap
pull_distance = door_pull_dist       # pull distance along +y to open the door

# Compute target orientations and heights
grip_theta = lever_width_angle  # align jaws perpendicular to lever length
hover_z = selected_lever[2] + hover_clearance
z_above = selected_lever[2] + above_clearance
z_insert = selected_lever[2] - insert_depth

# Define target poses
hover_pose = [selected_lever[0], selected_lever[1], hover_z, grip_theta]
above_pose = [selected_lever[0], selected_lever[1], z_above, grip_theta]
insert_pose = [selected_lever[0], selected_lever[1], z_insert, grip_theta]
pull_pose =  [selected_lever[0], selected_lever[1] + pull_distance, z_insert, grip_theta]
lift_after_pull_pose = [pull_pose[0], pull_pose[1], z_insert + 0.05, grip_theta]

.... Continue with Execution (execute_trajectory) using the above poses starts here ...
-------------------- End of door Sample -----------------
"""