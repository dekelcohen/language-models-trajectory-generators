# Gave GPT-5.5 (free webui) the conversation until after 'Approach decision and plan' (after detect_object(s) and before ```python code)
# Prompt: Given the above task desc and data - call get_grasp_poses and select the best pose to grasp the target object - generate code to sort desc poses considering task requirements and grasp score
# See https://chatgpt.com/c/6a0dc097-88c0-83eb-8b01-34ab5d383d93
import math
import numpy as np

logger.info(PROGRESS + "Requesting grasp candidates for door handle lever..." + ENDC)

# Get candidate grasps
poses, scores = get_grasp_poses("door handle lever")

# ------------------------------------------------------------------
# Known perception outputs
# ------------------------------------------------------------------

handle_center = np.array([-0.319, -0.127, 0.638])

handle_width = 0.185      # long dimension
handle_thickness = 0.032  # short dimension (graspable)

angle_short = -0.03
angle_long = 1.541

# Desired EE rotation:
# Closing direction aligned with lever short axis,
# making fingers parallel to lever long axis.
desired_theta = angle_short

# ------------------------------------------------------------------
# Compute short-axis vector toward the door plate
# ------------------------------------------------------------------

short_axis_vec = np.array([
    math.cos(angle_short),
    math.sin(angle_short)
])

door_center = np.array([-0.196, -0.130])

# Choose direction toward the door
to_door = door_center - handle_center[:2]

if np.dot(short_axis_vec, to_door) < 0:
    short_axis_vec *= -1.0

# Small insertion offset:
# lets one jaw slide into the gap between lever and door
offset_mag = 0.012

offset_xy = short_axis_vec * offset_mag

print(f"Offset X: {offset_xy[0]:.3f}")
print(f"Offset Y: {offset_xy[1]:.3f}")

# ------------------------------------------------------------------
# Rank grasp poses using task-aware metrics
# ------------------------------------------------------------------

ranked_candidates = []

for i, (pose, score) in enumerate(zip(poses, scores)):

    # Position
    tx = pose[0, 3]
    ty = pose[1, 3]
    tz = pose[2, 3]

    pos = np.array([tx, ty, tz])

    # --------------------------------------------------------------
    # Top-down preference
    # pose[2,2] close to -1 means gripper points downward
    # --------------------------------------------------------------

    downward_alignment = -pose[2, 2]

    if downward_alignment < 0.80:
        continue

    # --------------------------------------------------------------
    # Distance to desired contact point
    # --------------------------------------------------------------

    desired_contact = np.array([
        handle_center[0] + offset_xy[0],
        handle_center[1] + offset_xy[1],
        handle_center[2]
    ])

    dist = np.linalg.norm(pos - desired_contact)

    # --------------------------------------------------------------
    # Orientation score
    # Infer EE yaw from x-axis projection
    # --------------------------------------------------------------

    yaw = math.atan2(pose[1, 0], pose[0, 0])

    yaw_error = abs(math.atan2(
        math.sin(yaw - desired_theta),
        math.cos(yaw - desired_theta)
    ))

    # --------------------------------------------------------------
    # Height preference:
    # slightly above lever center preferred
    # --------------------------------------------------------------

    z_target = handle_center[2] + 0.01
    z_error = abs(tz - z_target)

    # --------------------------------------------------------------
    # Composite task-aware score
    # --------------------------------------------------------------

    composite_score = (
        2.5 * score +
        2.0 * downward_alignment -
        4.0 * dist -
        1.5 * yaw_error -
        1.0 * z_error
    )

    ranked_candidates.append({
        "index": i,
        "pose": pose,
        "score": score,
        "composite_score": composite_score,
        "distance": dist,
        "yaw_error": yaw_error,
        "downward_alignment": downward_alignment
    })

# ------------------------------------------------------------------
# Sort descending by task-aware score
# ------------------------------------------------------------------

ranked_candidates.sort(
    key=lambda x: x["composite_score"],
    reverse=True
)

# ------------------------------------------------------------------
# Select best candidate
# ------------------------------------------------------------------

best = ranked_candidates[0]

best_pose_matrix = best["pose"]

best_x = best_pose_matrix[0, 3]
best_y = best_pose_matrix[1, 3]
best_z = best_pose_matrix[2, 3]

best_yaw = math.atan2(
    best_pose_matrix[1, 0],
    best_pose_matrix[0, 0]
)

print(f"Selected candidate index: {best['index']}")
print(f"Raw grasp score: {best['score']:.3f}")
print(f"Composite score: {best['composite_score']:.3f}")
print(f"Selected X: {best_x:.3f}")
print(f"Selected Y: {best_y:.3f}")
print(f"Selected Z: {best_z:.3f}")
print(f"Selected yaw: {best_yaw:.3f}")

# ------------------------------------------------------------------
# Final grasp pose used for trajectories
# ------------------------------------------------------------------

grasp_pose = [
    best_x,
    best_y,
    best_z,
    desired_theta
]

# High hover pose
hover_pose = [
    best_x,
    best_y,
    best_z + 0.22,
    desired_theta
]

logger.info(PROGRESS + "Best lever grasp pose selected successfully." + ENDC)