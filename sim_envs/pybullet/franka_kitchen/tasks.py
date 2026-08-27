"""Franka Kitchen task table (PyBullet port).

Task ids are namespaced by the registry as ``franka_kitchen:<task_id>``.

Goal values come from the MuJoCo reference implementation
(``gymnasium_robotics/envs/franka_kitchen/kitchen_env.py`` ``OBS_ELEMENT_GOALS``)
with ``BONUS_THRESH = 0.3``.

Important scaling note
---------------------
The scene is loaded with ``globalScaling = KITCHEN_GLOBAL_SCALING``. PyBullet
scales **prismatic** joint limits (and therefore prismatic joint *positions*) by
that factor, but leaves **revolute** limits untouched.

Goals below are stored in **unscaled MuJoCo units**. ``SimEnvKitchen.task_error``
converts measured prismatic joint positions back to unscaled units before
comparing, so ``SUCCESS_THRESHOLD`` keeps exactly the meaning it has upstream.
Pre-scaling the goals instead would silently shrink the task: e.g. the slide
cabinet goal 0.37 would become 0.2775, which is *below* the 0.3 threshold and so
would be reported as solved at t=0.
"""

KITCHEN_GLOBAL_SCALING = 0.75

# MuJoCo BONUS_THRESH: euclidean distance between achieved and goal joint vector.
SUCCESS_THRESHOLD = 0.3

# The kettle goal is a world *position* (metres), not a joint vector, so it needs
# its own tolerance. MuJoCo's 0.3 would be nearly satisfied by the start pose
# (the two burners are only ~0.33 m apart), so we use a tighter radius.
POSITION_SUCCESS_THRESHOLD = 0.12

# Head camera is pinned to an axis-aligned yaw so the 3D-coordinates prompt is
# truthful. In PyBullet's yaw convention (see pb_utils.spherical_camera_pose)
# yaw=270 puts the camera on -x looking towards +x, giving:
#   image-right == -y, image-up == +z, image-depth == +x.
HEAD_CAMERA_YAW = 270.0


def scaled(value):
    """Scale a value from unscaled URDF/MuJoCo units into loaded-scene units."""
    return value * KITCHEN_GLOBAL_SCALING


def unscaled(value):
    """Inverse of :func:`scaled`: loaded-scene units -> URDF/MuJoCo units."""
    return value / KITCHEN_GLOBAL_SCALING


# Knob -> burner-plate coupling, replicated from the upstream
# ``updateFrankaKitchen``. Resolved by joint name, not by hard-coded index.
KNOB_TO_BURNER = {
    "knob_Joint_1": "bottom_right_burner",
    "knob_Joint_2": "bottom_left_burner",
    "knob_Joint_3": "top_right_burner",
    "knob_Joint_4": "top_left_burner",
}

# Light switch -> light block coupling + the link whose colour is toggled.
LIGHT_SWITCH_JOINT = "light_switch"
LIGHT_BLOCK_JOINT = "light_joint"
LIGHT_LINK = "lightblock_hinge"

# Links the gripper is expected to grasp/push. Friction is raised on these so a
# closed gripper does not slip (URDF ships the PyBullet default of 0.5).
HANDLE_LINKS = [
    "slidelink_1", "slidelink_2",
    "hingeleftdoor_1", "hingeleftdoor_2",
    "hingerightdoor_1", "hingerightdoor_2",
    "microdoorroot_1", "microdoorroot_2", "microdoorroot_3",
    "knob 1_1", "knob 1_2", "knob 2_1", "knob 2_2",
    "knob 3_1", "knob 3_2", "knob 4_1", "knob 4_2",
    "lightswitchroot",
]

# Articulated joints that must hold their pose against gravity/contact but still
# yield to the arm. Value = motor force for a zero-velocity VELOCITY_CONTROL.
# The URDF declares effort="0", so PyBullet's default motors apply no force at
# all and every door would swing freely.
RESTING_FRICTION_FORCE = {
    "microwave": 2.0,
    "slide_cabinet": 2.0,
    "left_hinge_cabinet": 2.0,
    "right_hinge_cabinet": 2.0,
    "light_switch": 0.5,
    "knob_Joint_1": 0.5,
    "knob_Joint_2": 0.5,
    "knob_Joint_3": 0.5,
    "knob_Joint_4": 0.5,
}

# The URDF gives every movable joint ``damping="1.0"``, which is far too viscous
# for hand-sized hardware: even a 4x-effort motor only drags a knob through ~18%
# of its travel per second, so an arm-driven trajectory barely moves it. These
# values are applied with ``changeDynamics(jointDamping=...)``, leaving just
# enough viscosity to keep the doors from oscillating.
JOINT_DAMPING = {
    "microwave": 0.1,
    "slide_cabinet": 0.1,
    "left_hinge_cabinet": 0.1,
    "right_hinge_cabinet": 0.1,
    "light_switch": 0.02,
    "knob_Joint_1": 0.02,
    "knob_Joint_2": 0.02,
    "knob_Joint_3": 0.02,
    "knob_Joint_4": 0.02,
}

# The URDF's inertial blocks are derived from mesh volume at an unrealistic
# density: each cabinet door weighs ~20-45 kg and the kettle 10.2 kg, well past
# the Panda's 3 kg payload, so nothing can be opened or lifted. Only *movable*
# links are re-massed -- the fixed counters/walls stay as they are, since their
# mass never enters the dynamics.
MOVABLE_LINK_MASS = {
    "slidelink_1": 1.5,
    "slidelink_2": 1.5,
    "hingeleftdoor_1": 1.5,
    "hingeleftdoor_2": 1.5,
    "hingerightdoor_1": 1.5,
    "hingerightdoor_2": 1.5,
    "microdoorroot_1": 1.0,
    "microdoorroot_2": 1.0,
    "microdoorroot_3": 1.0,
    "knob 1_1": 0.05, "knob 1_2": 0.05,
    "knob 2_1": 0.05, "knob 2_2": 0.05,
    "knob 3_1": 0.05, "knob 3_2": 0.05,
    "knob 4_1": 0.05, "knob 4_2": 0.05,
}

# Free-floating kettle: heavy enough to stay put, light enough to lift.
KETTLE_MASS = 1.0


class KitchenTask:
    """Static description of one Franka Kitchen task.

    Attributes:
        task_id: id used in ``--task franka_kitchen:<task_id>``.
        label: natural-language name handed to segmentation / the prompt.
        target_link: kitchen link whose geometry the arm must reach. Its *AABB
            centre* -- not its frame origin -- frames the head camera and is
            reported as the primary manipulation target, because several kitchen
            links (notably the cabinet handles) carry meshes that sit far from
            their own frame origin.
        goal_joints: {joint_name: target_value} in unscaled MuJoCo joint space.
        goal_body: for free bodies (the kettle) the goal is a world position
            relative to a named link instead of a joint vector.
        camera_distance / camera_pitch: head camera spherical params
            (yaw is pinned to HEAD_CAMERA_YAW).
        camera_target_offset: (dx, dy, dz) added to the target position to get
            the look-at point. Because yaw is pinned, dy/dz translate the camera
            sideways/vertically *without rotating it*, so they never affect the
            3D-coordinates prompt; only camera_pitch trades off how separable
            +x and +z look in the image. A non-zero offset also moves the target
            off frame centre, so it should stay 0 unless something must be kept
            in shot alongside the target. Values here are not hand-guessed:
            ``tests/tools/tune_kitchen_head_camera.py`` measures the exact
            fraction of the target link the arm hides (render with vs without
            the robot) and every task is regression-tested by
            ``tests/test_franka_kitchen_head_camera.py``.
    """

    def __init__(self, task_id, label, target_link, goal_joints=None,
                 goal_body=None, camera_distance=1.2, camera_pitch=-25.0,
                 camera_target_offset=(0.0, 0.0, 0.0)):
        self.task_id = task_id
        self.label = label
        self.target_link = target_link
        self.goal_joints = goal_joints or {}
        self.goal_body = goal_body
        self.camera_distance = camera_distance
        self.camera_pitch = camera_pitch
        self.camera_target_offset = list(camera_target_offset)


KITCHEN_TASK_LIST = [
    KitchenTask(
        task_id="microwave",
        label="microwave door handle",
        target_link="microdoorroot_1",
        goal_joints={"microwave": -0.75},
        camera_distance=1.6,
        camera_pitch=-10.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="slide_cabinet",
        label="slide cabinet handle",
        target_link="slidelink_1",
        goal_joints={"slide_cabinet": 0.37},
        camera_distance=1.6,
        camera_pitch=-15.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="hinge_cabinet",
        label="hinge cabinet door handle",
        target_link="hingerightdoor_1",
        goal_joints={"left_hinge_cabinet": 0.0, "right_hinge_cabinet": 1.45},
        camera_distance=1.6,
        camera_pitch=-15.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="light_switch",
        label="light switch",
        target_link="lightswitchroot",
        goal_joints={"light_switch": -0.69, "light_joint": -0.05},
        camera_distance=1.6,
        camera_pitch=-10.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="top_burner",
        # Burners 3/4 are the back row (further from the robot); their knobs are
        # the upper knob row, matching MuJoCo's "top burner".
        label="top burner knob",
        target_link="knob 3_1",
        goal_joints={"knob_Joint_3": -0.92, "top_right_burner": -0.01},
        # The upper knob row sits directly behind the arm's forearm at a shallow
        # pitch (43% of the knob was hidden at -8). Tilting the camera down to
        # -20 raises it enough to clear the arm entirely.
        camera_distance=1.6,
        camera_pitch=-20.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="bottom_burner",
        label="bottom burner knob",
        target_link="knob 1_1",
        goal_joints={"knob_Joint_1": -0.88, "bottom_right_burner": -0.01},
        camera_distance=1.6,
        camera_pitch=-5.0,
        camera_target_offset=(0.0, 0.0, 0.0),
    ),
    KitchenTask(
        task_id="kettle",
        label="kettle",
        target_link="Burner 2_link",
        # The kettle is a free body: success is a world-position goal. It starts
        # on the front-left burner (in front of the arm, so the head camera sees
        # it unoccluded) and must be moved to the back-left burner. MuJoCo's goal
        # is a 7-DoF pose in a differently-oriented frame, so it is re-expressed
        # here relative to a named burner link instead of copied verbatim.
        goal_body={"burner_link": "Burner 4_link", "z_offset": 0.055},
        camera_distance=1.9,
        camera_pitch=-22.0,
        camera_target_offset=(0.17, 0.0, 0.15),
    ),
]

KITCHEN_TASKS = {t.task_id: t for t in KITCHEN_TASK_LIST}
