"""Base sim-env profile.

Applies hard-coded runtime overrides and loads any required assets.
No global config edits happen outside of a sim-env profile.
"""

import config


class SimEnvBase:
    """Base environment profile.

    Lifecycle (driven by ``env.Environment`` / ``env.run_simulation_environment``):

    1. ``configure_robot_pose()`` - before any URDF is loaded, so the robot base
       and joint start pose can be overridden.
    2. ``required_robot()`` - consulted by ``Environment.__init__`` to force a
       robot model when the scene only makes sense with one.
    3. ``apply(env)`` - runtime config overrides (cameras, etc).
    4. ``load_assets(env)`` - scene URDFs.
    5. ``tune_physics()`` - friction / joint motor setup for the loaded scene.
    6. ``step_hook()`` - called once per simulation step, for scenes that need
       per-step kinematic coupling (e.g. Franka Kitchen knobs -> burners).
    7. ``get_state()`` / ``check_success()`` - ground-truth reporting.
    """

    def __init__(self):
        # Base class does not change defaults
        pass

    def apply(self, env):
        # Default: leave config values as-is
        return

    def load_assets(self, env):
        # Default: no additional assets
        return

    def tune_physics(self):
        """Per-scene friction / joint motor setup, run right after load_assets.

        Default is no-op: the grasp and door profiles do their tuning inline in
        ``load_assets``.
        """
        return

    def step_hook(self):
        """Called once per ``p.stepSimulation()``.

        Scenes whose URDF encodes coupled mechanisms that PyBullet cannot
        simulate on its own (knob angle -> burner plate, light switch -> lamp)
        implement the coupling here. Must be cheap and must never raise.
        """
        return

    def get_state(self):
        """Return a dict with environment-specific state for diagnostics.
        Base env has no special state.
        """
        return {}

    def get_success_criteria(self):
        """Return a dict describing ground-truth task success, or {} if the
        sim-env has no machine-checkable criteria."""
        return {}

    def check_success(self):
        """Return True/False when success is machine-checkable, else None."""
        return None

    def required_robot(self):
        """Return a robot model name ('franka' / 'sawyer') that this scene
        requires, or None to honour the user's --robot choice."""
        return None

    def configure_robot_pose(self):
        """Per-task hook to set robot/base/joint starting pose.
        Default is no-op (grasp task defaults).
        """
        return

    def configure_cameras(self):
        """Per-task hook to set head-camera framing in config.

        Called from ``apply`` implementations that opt in. Default no-op keeps
        the config defaults.
        """
        return

    def get_wrist_camera_params(self):
        """Return the wrist-camera 'drone' framing offsets for this scene.

        ``robot.get_camera_image('wrist', ...)`` reads these, falling back to
        the ``config`` defaults so existing scenes are unaffected.
        """
        return {
            "pullback": config.wrist_camera_pullback,
            "up_shift": config.wrist_camera_up_shift,
            "lateral_shift": config.wrist_camera_lateral_shift,
        }

    def move_to_start_pos(self):
        """
        Return True to move the robot to start ee position + orientation
        """
        return True

    def get_ee_start_pose(self):
        """
        Return (position, orientation_e) for the arm's home EE pose, or None to let the
        caller derive one. Sim-envs that override config.ee_start_position in
        configure_robot_pose() should return it here so debug/demo paths agree with
        the production RESET_EEF home.
        """
        return None

    def get_3d_coordinates_prompt_section(self):
        """Return the 3D coordinate system prompt section (default)."""
        return config.three_d_coordinates_prompt_section
