"""The same rollout-tracking scenario as ``test_tracking_pybullet.py``, on Genesis.

**Run this with the Genesis interpreter** (it skips itself anywhere else)::

    <vlm_genesis>/python.exe -m pytest tests/test_tracking_genesis.py -q

Identical assertions, different simulator: every test body lives in
``tracking_scenario.TrackingScenarioMixin``, so the two suites cannot drift apart. What
this run actually proves is that the tracking geometry is simulator-agnostic - Genesis
renders **linear metric** depth where PyBullet renders a non-linear GL z-buffer, and the
whole pipeline (occlusion test, deprojection, cross-camera re-seeding, fusion) has to come
out at the same world coordinates regardless. The 3D-lift provider sweep runs here too, so
each ``providers/tracker3d`` provider is exercised on both simulators.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

try:                          # utils pulls in shapely, which vlm_genesis does not have;
    import utils              # nothing on the tracking path needs it (see dump_sim_state)
except Exception:
    utils = None
from tracking_scenario import OBJECT_START, SETTLE_STEPS, TrackingScenarioMixin, object_body

try:
    import genesis as gs
    GENESIS_IMPORT_ERROR = None
except Exception as exc:      # pragma: no cover - depends on the interpreter in use
    gs = None
    GENESIS_IMPORT_ERROR = exc


class _Args:
    mode = "default"
    robot = "franka"
    task = "grasp"
    sim = "genesis"
    save_grasp_inputs = False
    tracking = True
    tracker_provider = "template"
    track_interval = 1
    track_save_depth = False
    track_log_dir = None


def _boot():
    """Bring up the ``grasp`` scene on Genesis through the production classes."""
    import env as env_module
    from robot import Robot
    from debug.dbg_utils import init_loguru_logger
    from sim_adapter import get_adapter

    args = _Args()
    if utils is not None:
        utils.args = args
    config.object_start_position = list(OBJECT_START)
    config.object_start_orientation_e = [0.0, 0.0, 0.0]

    sim = get_adapter("genesis")
    # Genesis freezes the scene topology at build(), so the tracking cameras have to be
    # reserved before anything else is loaded - unlike PyBullet, they cannot be created
    # on demand during the rollout.
    sim.reserve_camera(config.image_width, config.image_height,
                       fov=config.fov, near=config.near_plane, far=config.far_plane)
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger("tracking_genesis.log"), sim)
    sim.build()
    for _ in range(SETTLE_STEPS):
        environment.update()
    return sim, environment, robot


@unittest.skipIf(gs is None, f"genesis is not importable here ({GENESIS_IMPORT_ERROR})")
class TestTrackingGenesis(TrackingScenarioMixin, unittest.TestCase):
    """One booted scene for the whole class: ``gs.init``/``build`` are expensive."""

    @classmethod
    def setUpClass(cls):
        cls.sim, cls.env, cls.robot = _boot()
        cls.object_id = object_body(cls.env)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.sim.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
