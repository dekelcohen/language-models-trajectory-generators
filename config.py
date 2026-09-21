import math
import os
import random

# Simulation
control_dt = 1. / 240.
margin_error = 0.001
gripper_margin_error = 0.0001
joint_margin_error = 0.01
rel_tol = 1e-4
abs_tol = 0.0

# Robot.move's orientation convergence test.
#
# The legacy test compares roll/pitch/yaw component-wise against `margin_error`. That test
# is dead code: `transforms.euler_from_quat` derives pitch with `asin`, so it can only ever
# return pitch in [-pi/2, pi/2], while the top-down target `ee_start_orientation_e` has
# pitch = pi. The comparison is therefore never satisfiable and the move loop only ever
# exits via its iteration caps (1 step for trajectory points, 100 for standalone moves).
#
# `_orientation_reached` below can instead measure the true angle between the current and
# target quaternions, which does converge. It is OFF by default on purpose: enabling it
# lets standalone moves exit early, changing sim step counts, and therefore the recorded
# frame sequences and every golden fixture derived from them. Flip it only together with a
# regression re-baseline.
use_quat_orientation_convergence = False
quat_orientation_margin_error = 0.01  # radians, only used when the flag above is True

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
# Rollout tracking (see tracking/ and providers/trackers/)
START_TRACKING = 23
STOP_TRACKING = 24
GET_TRACKING_REPORT = 25
CLEAR_GRASP_MARKERS = 26

# --- Rollout tracking -------------------------------------------------------
# Tracking is opt-in (--tracking). When disabled nothing in the capture path changes,
# so the PyBullet regression goldens stay bit-identical.
tracking_enabled_default = False
tracker_provider_default = "template"   # template | csrt | cotracker | remote
# CoTracker3 provider (--tracker-provider cotracker). The online model is fetched once via
# torch.hub into TORCH_HOME and reused offline afterwards; None honours the environment
# and falls back to <repo>/cache/torch.
tracker_cotracker_variant = "cotracker3_online"
tracker_cotracker_device = None          # None = cuda:0 when available, else cpu
tracker_cotracker_torch_home = None      # None = $TORCH_HOME or <repo>/cache/torch
# Visibility probability above which a CoTracker point counts as visible (ignored when the
# predictor already returns a boolean mask, which CoTracker3 does).
tracker_cotracker_vis_threshold = 0.5
# The online model only emits a prediction every `step` frames; per-point scores lose this
# much confidence per frame the reported result is old (see TrackResult.meta["stale_frames"]).
tracker_cotracker_stale_decay = 0.05
# How the per-camera 2D tracks become one world point.
#   depth_fusion - deproject through each camera's depth buffer, then weighted-average
#   triangulate  - multi-view DLT from 2D + calibration only, no depth buffer
#   lapa         - LAPA's learned per-view weighting on top of that triangulation
tracker3d_provider_default = "depth_fusion"
tracking_cameras = ("head", "wrist")
# Run the tracker every Nth recorded keyframe (1 = every keyframe).
track_interval = 1
# Half-size, in pixels, of the square patch a point tracker follows around each point.
track_patch_half = 12
# Sub-window (in patch half-sizes) the template tracker searches around the last position.
track_search_scale = 3.0
# A projected world point counts as visible in a camera only when the rendered depth at
# that pixel agrees with the expected depth to within this many metres. This is what
# stops a re-seed from latching onto the robot arm occluding the object.
track_occlusion_tol = 0.03
# Cross-camera re-seeding hands over a point on the surface facing the *donor* camera,
# which the receiving camera sees through the object's own body. Depth closer than this
# (roughly one object thickness) is treated as self-occlusion and the tracker is seeded on
# the rendered surface instead; anything closer than this is a real occluder (the arm).
track_reseed_self_occlusion_m = 0.10
# Depth samples outside this metric range are treated as invalid (sky / near-plane).
track_depth_min = 0.02
track_depth_max = 5.0
# Per-point tracker confidence below which a point is dropped.
track_point_conf_min = 0.35
# Camera-level confidence below which the camera is a re-seed candidate ...
track_reseed_conf = 0.45
# Minimum *health* (confidence x continuity x 1/z x point support) for a camera to be
# trusted as a re-seed donor. Deliberately far below track_reseed_conf: health folds in a
# 1/z precision term, so the head camera - a metre away and tracking perfectly - scores
# around 0.2 and would otherwise never be allowed to bootstrap the wrist camera.
track_health_min = 0.12
# ... after this many consecutive unhealthy frames ...
track_reseed_patience = 2
# ... and never more often than this many tracked frames.
track_reseed_cooldown = 5
# Per-camera world points further apart than this (metres) are a "disagreement".
track_disagree_m = 0.08
# Fused world point may not jump more than this (metres) between consecutive tracked
# frames without losing temporal-continuity health.
track_max_jump_m = 0.15
# Object is reported lost after this many consecutive frames with no healthy camera.
track_lost_patience = 3
# Built-in attached_to_gripper monitor defaults.
track_attach_max_dist = 0.12
track_attach_grace_frames = 3
# Outputs
tracking_output_dir = "./outputs/tracking"
tracking_log_name = "track.jsonl"
tracking_summary_name = "summary.json"

# LLM response cache
llm_cache_dir = "./cache"               # root cache folder (auto-created)
llm_cache_float_tolerance = 1e-2        # abs diff allowed per float when smart-matching env state

# Paths
# IMAGES_ROOT env var lets concurrent runs (e.g. 2 copilot sessions on the same checkout)
# write to isolated folders instead of overwriting each other's ./images. Example:
#   $env:IMAGES_ROOT="./images_run1"; python main.py
images_folder = os.environ.get("IMAGES_ROOT", "./images")
rgb_image_wrist_path = images_folder + "/rgb_image_wrist.png"
depth_image_wrist_path = images_folder + "/depth_image_wrist.png"
rgb_image_head_path = images_folder + "/rgb_image_head.png"
depth_image_head_path = images_folder + "/depth_image_head.png"
bounding_cube_mask_image_path = images_folder + "/bounding_cube_mask_{object}_{mask}.png"

# Overlays and runs
overlay_folder = images_folder + "/overlay"
overlay_image_path = images_folder + "/overlay/overlay_{step}.png"
runs_dir = "./runs"

# Logging throttles
# Only write trajectory frames every N steps (>=1)
trajectory_log_every = 5
trajectory_folder = images_folder + "/trajectory"
video_folder = images_folder + "/videos"
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

langsam_image_path = images_folder + "/langsam_image_{object}.png"
xmem_input_path = images_folder + "/xmem_input.png"
xmem_output_path = images_folder + "/xmem_output_{step}.png"

# Segmentation overlay output for any provider
# Visualization
# Keep this many recent preview-steps of trajectory markers (spheres) visible (rolling window).
visualize_traj_history_steps = 6
seg_overlay_image_path = images_folder + "/seg_overlay_{provider}_{object}.png"

# Snapshot of the exact head-camera image the perception VLM analyzed for a given
# scene analysis. Kept separate from the trajectory frames so the reviewer VLM can be
# shown the start-of-attempt scene without it being mistaken for a trajectory frame.
scene_analysis_image_path = images_folder + "/scene_analysis_head_{step}.png"

# Affordance-pointing coordinate format text injected into SCENE_PERCEPTION_PROMPT
# (replaces COORDINATES_FORMAT_PLACEHOLDER). Selected by the perception VLM's
# ModelInfo.pointing_coords_format (see providers.llms.model_registry).
affordance_coords_format_by_key = {
    "yx_norm_1000": "The points are in [y, x] format normalized to 0-1000",
    "xy_pixels": "The points are in [x, y] pixel coordinates",
}

# Depth sampling for affordance points (see utils.sample_surface_depth).
# A VLM points at a thin feature - the adroit door lever bar is ~7 px thick at 5.5 mm/px -
# so a 1-2 px pointing error read from a SINGLE pixel lands on the door face (+0.1 m) or on
# the background (+1 m). Sampling a small patch and keeping a near percentile recovers the
# feature instead of the surface behind it.
# Radius 2 -> a 5x5 window: wide enough to contain feature pixels after a 2 px miss, narrow
# enough (27 mm) not to reach past a graspable feature.
affordance_depth_patch_radius = 2
# Which depth inside the patch counts as "the near surface". Not min(): a single bad pixel
# (depth noise, an anti-aliased silhouette edge) would win. 10th percentile tolerates ~2 such
# pixels in a 5x5 window while still selecting the foreground when only ~1/8 of the window
# covers it. Depth is monotonic in distance for BOTH encodings (PyBullet's nonlinear OpenGL
# buffer and Genesis's linear metres), so a low percentile means "near" in either.
affordance_depth_percentile = 10.0

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