"""End-to-end rollout tracking on PyBullet - **no LLM, no agent, no IPC**.

The object is moved directly with ``sim.set_base_pose`` so every frame has an exact
ground-truth world position, and the real :class:`tracking.session.TrackingSession` runs
against the real ``head``/``wrist`` renders produced by ``Robot.capture_camera_view``.

Covered:

* fused world coordinates follow a scripted object motion within a few centimetres;
* the wrist camera - which cannot see the object at first - is seeded from the head
  camera's world estimate once the object enters its view (cross-camera bootstrap);
* an occluded camera is re-seeded from the healthy one instead of latching onto the
  occluder (cross-camera repair);
* the built-in ``attached_to_gripper`` invariant aborts the session when the object is
  detached from the gripper mid-rollout;
* the whole scenario replayed against every constructible ``providers/tracker3d`` lift
  provider, each with its own measured accuracy band (see ``TRACKER3D_BANDS``).

Run with::

    python -m pytest tests/test_tracking_pybullet.py -q
"""

import os
import sys
import unittest

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils
from tracking_scenario import OBJECT_START, SETTLE_STEPS, TrackingScenarioMixin


class _Args:
    mode = "default"
    robot = "franka"
    task = "grasp"
    save_grasp_inputs = False
    tracking = True
    tracker_provider = "template"
    track_interval = 1
    track_save_depth = False
    track_log_dir = None


def _boot():
    """Bring up the ``grasp`` scene headlessly through the production classes."""
    import env as env_module
    from robot import Robot
    from debug.dbg_utils import init_loguru_logger
    from sim_adapter import get_adapter

    if p.isConnected():
        p.disconnect()
    sim = get_adapter("pybullet")
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    args = _Args()
    utils.args = args
    # config randomises the grasp object's spawn pose at import time; pin it so the
    # scenario (and therefore every geometric assertion) is reproducible.
    config.object_start_position = list(OBJECT_START)
    config.object_start_orientation_e = [0.0, 0.0, 0.0]

    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger("tracking_pybullet.log"), sim)
    sim.build()
    for _ in range(SETTLE_STEPS):
        environment.update()
    return sim, environment, robot


class TestTrackingPyBullet(TrackingScenarioMixin, unittest.TestCase):
    """One booted scene for the whole class; each test re-poses the object itself."""

    @classmethod
    def setUpClass(cls):
        cls.sim, cls.env, cls.robot = _boot()
        cls.object_id = cls.env.simenv.object_id

    @classmethod
    def tearDownClass(cls):
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    unittest.main()
