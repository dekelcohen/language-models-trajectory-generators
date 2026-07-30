import math
import random

# Simulation
control_dt = 1. / 240.
margin_error = 0.001
gripper_margin_error = 0.0001
joint_margin_error = 0.01
rel_tol = 1e-4
abs_tol = 0.0

# Robots
gripper_goal_position_open_sawyer = 0.2
gripper_goal_position_closed_sawyer = 1.0
arm_movement_force_sawyer = 5 * 240
gripper_movement_force_sawyer = 1000
ee_index_sawyer = 16

gripper_goal_position_open_franka = 0.04
gripper_goal_position_closed_franka = 0.0005
arm_movement_force_franka = 5 * 240
gripper_movement_force_franka = 1000
ee_index_franka = 11

robotiq_motor_joint = 1

# Environment
base_start_position_sawyer = [0.0, 0.0, 0.0]
base_start_orientation_e_sawyer = [0.0, 0.0, math.pi / 2]
joint_start_positions_sawyer = [-0.0304, -2.0563, -1.1631, -0.3829, 1.3152, 0.1496, 1.4462, -0.2288]
base_start_position_franka = [0.0, 0.0, 0.0]
base_start_orientation_e_franka = [0.0, 0.0, math.pi / 2]
joint_start_positions_franka = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0.04, 0.04]

ee_start_position = [0.0, 0.6, 0.55]
ee_start_orientation_e = [0.0, math.pi, -math.pi / 2]

RANDOM_TARGET_GRASP_OBJ_POSE = True # Randomness for testing - but not for debugging
if RANDOM_TARGET_GRASP_OBJ_POSE:
    object_start_position = [random.uniform(-0.2, 0.2), random.uniform(0.4, 0.8), 0.1]
    object_start_orientation_e = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
else:
    # x=-0.5 - closet to viewer, 5 - very far into the image. y=1 closest (second axis), y=-3 far into the image
    object_start_position = [-0.2, 0.4 , 0.1] # Good (Close to viewer - bottom of frame, large box): [0.0, 0.5, 0.1]
    object_start_orientation_e = [0.0, 0.0, 0.0]


global_scaling = 0.08

# Camera
fov, aspect, near_plane, far_plane = 60, 1.0, 0.01, 100
image_width = 256
image_height = 256

head_camera_position = [0.0, 1.2, 0.6]
head_camera_orientation_e = [0.0, 3 / 4.5 * math.pi, -math.pi / 2]

# Head camera control flags
# - head_camera_use_debug_view: mirror the GUI debug visualizer view (GUI only)
# - head_camera_use_spherical_view: in DIRECT/headless, build view from spherical params
head_camera_use_debug_view = False
head_camera_use_spherical_view = False

camera_distance = 0.8
camera_yaw = 225.0
camera_pitch = -30.0
camera_target_position = [0.0, 0.6, 0.3]

wrist_camera_offset_sawyer = 0.125

# Wrist camera "drone" over-the-shoulder framing (used when approaching the handle)
# - pullback: distance pulled straight back along the gripper line of sight
# - up_shift: vertical offset applied to the camera (negative = lower the pose)
# - lateral_shift: sideways offset along global 'right' so the view looks right->left
wrist_camera_pullback = 0.4
wrist_camera_up_shift = -0.2
wrist_camera_lateral_shift = 0.3

# Object grasping
point_cloud_top_surface_filter = 0.06
bounding_cube_depth_offset = 0.06
gripper_depth_offset_franka = 0.06
gripper_depth_offset_sawyer = -0.12

# Segmentation
segmentation_threshold = 0.2

# XMem configuration
xmem_config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}

xmem_visualise_every = 1
xmem_output_every = 1
xmem_lm_input_every = 20

# Multiprocessing
CAPTURE_IMAGES = 1
ADD_BOUNDING_CUBES = 2
ADD_TRAJECTORY_POINTS = 3
EXECUTE_TRAJECTORY = 4
OPEN_GRIPPER = 5
CLOSE_GRIPPER = 6
TASK_COMPLETED = 7
RESET_EEF = 8
# Extended commands for observability/testing
GET_STATE = 9
GET_CAMERA_INFO = 10
CAPTURE_ANNOTATED_IMAGES = 11
MOVE_EEF_ABS = 12
STEP_N = 13
SET_SEED = 14
SET_TASK_FROM_RAND_VEC = 15
QUERY_ENV_ATTR = 16
MAKE_TRAJECTORY_VIDEO = 17
SET_DOOR_STATE = 18
CAPTURE_TRAJECTORY_FRAME = 19
GET_ROBOT_STATE = 20
VISUALIZE_GRASP_POSE = 21
VISUALIZE_BOUNDING_BOX = 22

# LLM response cache
llm_cache_dir = "./cache"               # root cache folder (auto-created)
llm_cache_float_tolerance = 1e-2        # abs diff allowed per float when smart-matching env state

# Paths
images_folder = "./images" 
rgb_image_wrist_path = "./images/rgb_image_wrist.png"
depth_image_wrist_path = "./images/depth_image_wrist.png"
rgb_image_head_path = "./images/rgb_image_head.png"
depth_image_head_path = "./images/depth_image_head.png"
bounding_cube_mask_image_path = "./images/bounding_cube_mask_{object}_{mask}.png"

# Overlays and runs
overlay_folder = images_folder + "/overlay"
overlay_image_path = "./images/overlay/overlay_{step}.png"
runs_dir = "./runs"

# Logging throttles
# Only write trajectory frames every N steps (>=1)
trajectory_log_every = 5
trajectory_folder = "./images/trajectory"
trajectory_video_fps = 15
trajectory_image_base = "rgb_image"
trajectory_wrist_image_base = "wrist_image"
rgb_image_trajectory_path = trajectory_folder + "/rgb_image_{step}.png"
depth_image_trajectory_path = trajectory_folder + "/depth_image_{step}.png"
wrist_rgb_image_trajectory_path = trajectory_folder + "/wrist_image_{step}.png"
wrist_depth_image_trajectory_path = trajectory_folder + "/wrist_depth_image_{step}.png"
# For perception captures of static elements, log only for first N events
perception_log_first_n = 1
# Optionally, re-log every M frames (0 disables)
perception_log_interval_frames = 0

langsam_image_path = "./images/langsam_image_{object}.png"
xmem_input_path = "./images/xmem_input.png"
xmem_output_path = "./images/xmem_output_{step}.png"

# Segmentation overlay output for any provider
# Visualization
# Keep this many recent preview-steps of trajectory markers (spheres) visible (rolling window).
visualize_traj_history_steps = 6
seg_overlay_image_path = "./images/seg_overlay_{provider}_{object}.png"

# Affordance-pointing coordinate format text injected into SCENE_PERCEPTION_PROMPT
# (replaces COORDINATES_FORMAT_PLACEHOLDER), selected by the perception VLM family.
affordance_coords_format_gemini = "The points are in [y, x] format normalized to 0-1000"
affordance_coords_format_pixels = "The points are in [x, y] pixel coordinates"

# Output - ANSI escape color codes:
OK = "\033[92m"       # Bright Green
PROGRESS = "\033[97m" # Bright White
WARNING = "\033[93m"  # Bright Yellow
FAIL = "\033[91m"     # Bright Red
ENDC = "\033[0m"      # Reset to default

# 3D coordinates prompt section (default: grasp/metaworld)
three_d_coordinates_prompt_section = (
    "The 3D coordinate system of the environment is as follows:\n"
    "  1. The x-axis is in the horizontal direction, increasing to the right.\n"
    "  2. The y-axis is in the depth direction, increasing away from you.\n"
    "  3. The z-axis is in the vertical direction, increasing upwards."
)

# Prompts images and additional eef_pos 
ENABLE_EEF_POS_IMAGE = False # Currently didn't help match 

# GPT-5 in azure limits to 50 images in a request 
max_allowed_vlm_images = 50