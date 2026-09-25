"""PyBullet coverage for the opt-in robot-base shoulder camera.

The shoulder camera is intentionally not part of ``config.tracking_cameras`` yet; these
tests exercise it directly through ``Robot.capture_camera_view("shoulder", env)`` so IPC
goldens and two-camera rollout defaults stay unchanged.
"""

import math
import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import utils  # noqa: E402
from tracking.geometry import in_bounds, is_visible, project_world_to_pixel  # noqa: E402

try:
    import pybullet as p
except ImportError:  # pragma: no cover - env without pybullet
    p = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOOR_URDF = os.path.join(REPO_ROOT, "my_assets", "adroit_door", "adroit_door.urdf")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "tracking_video")
SETTLE_STEPS = 120
MARGIN_PX = 8

HEAD_CAMERA_DEFAULTS = {
    "position": [0.0, 1.2, 0.6],
    "orientation_e": [0.0, 3 / 4.5 * math.pi, -math.pi / 2],
    "use_debug_view": False,
    "use_spherical_view": False,
    "distance": 0.8,
    "yaw": 225.0,
    "pitch": -30.0,
    "target": [0.0, 0.6, 0.3],
}
FRANKA_DEFAULTS = {
    "base_start_position": [0.0, 0.0, 0.0],
    "base_start_orientation_e": [0.0, 0.0, math.pi / 2],
    "joint_start_positions": [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0.04, 0.04],
}

# The Adroit latch link origin is inside the handle/door geometry, so strict rendered-depth
# visibility correctly rejects it.  This point is a visible surface point on the same latch
# link, expressed in the COM frame returned by ``get_link_pose``.
DOOR_LATCH_SURFACE_LOCAL = np.array([0.0, -0.163, -0.04])


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


def _reset_mutable_config():
    config.base_start_position_franka = list(FRANKA_DEFAULTS["base_start_position"])
    config.base_start_orientation_e_franka = list(FRANKA_DEFAULTS["base_start_orientation_e"])
    config.joint_start_positions_franka = list(FRANKA_DEFAULTS["joint_start_positions"])
    config.head_camera_position = list(HEAD_CAMERA_DEFAULTS["position"])
    config.head_camera_orientation_e = list(HEAD_CAMERA_DEFAULTS["orientation_e"])
    config.head_camera_use_debug_view = HEAD_CAMERA_DEFAULTS["use_debug_view"]
    config.head_camera_use_spherical_view = HEAD_CAMERA_DEFAULTS["use_spherical_view"]
    config.camera_distance = HEAD_CAMERA_DEFAULTS["distance"]
    config.camera_yaw = HEAD_CAMERA_DEFAULTS["yaw"]
    config.camera_pitch = HEAD_CAMERA_DEFAULTS["pitch"]
    config.camera_target_position = list(HEAD_CAMERA_DEFAULTS["target"])
    config.shoulder_camera_base_offset = [0.35, -0.35, 0.60]
    config.shoulder_camera_target_offset = [0.40, 0.20, 0.05]


def _boot(task):
    """Bring up one PyBullet scene through the production env/robot classes."""
    if p is None:
        raise unittest.SkipTest("pybullet is not installed")
    if task == "door" and not os.path.exists(DOOR_URDF):
        raise unittest.SkipTest("my_assets/adroit_door/adroit_door.urdf is missing")

    import env as env_module
    from debug.dbg_utils import init_loguru_logger
    from robot import Robot
    from sim_adapter import get_adapter

    _reset_mutable_config()
    if p.isConnected():
        p.disconnect()
    sim = get_adapter("pybullet")
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    args = _Args()
    args.task = task
    utils.args = args
    if task == "grasp":
        config.object_start_position = [-0.2, 0.4, 0.1]
        config.object_start_orientation_e = [0.0, 0.0, 0.0]

    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger(f"shoulder_camera_{task}.log"), sim)
    sim.build()
    for _ in range(SETTLE_STEPS):
        environment.update()
    return sim, environment, robot


def _door_target(sim, environment):
    state = environment.simenv.get_state()
    if state.get("door_id") is None or state.get("door_handle_latch") is None:
        raise unittest.SkipTest("the door scene did not load a latch link")
    pos, quat = sim.get_link_pose(state["door_id"], state["door_handle_latch"])
    rot = np.asarray(sim.matrix_from_quat(quat), dtype=float).reshape(3, 3)
    return np.asarray(pos, dtype=float) + rot.dot(DOOR_LATCH_SURFACE_LOCAL)


def _grasp_target(sim, environment):
    object_id = getattr(environment.simenv, "object_id", None)
    if object_id is None:
        raise unittest.SkipTest("the grasp scene did not load the cube object")
    pos, _quat = sim.get_base_pose(object_id)
    return np.asarray(pos, dtype=float)


def _ray_angle_deg(head_view, shoulder_view, target):
    target = np.asarray(target, dtype=float)
    head_ray = target - np.asarray(head_view.position, dtype=float)
    shoulder_ray = target - np.asarray(shoulder_view.position, dtype=float)
    head_ray /= np.linalg.norm(head_ray)
    shoulder_ray /= np.linalg.norm(shoulder_ray)
    return math.degrees(math.acos(float(np.clip(np.dot(head_ray, shoulder_ray), -1.0, 1.0))))


def _mark_target(rgb, pixel, label, visible):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    color = (0, 255, 0) if visible else (255, 64, 64)
    if pixel is not None:
        x, y = float(pixel[0]), float(pixel[1])
        draw.line([(x - 6, y), (x + 6, y)], fill=color, width=2)
        draw.line([(x, y - 6), (x, y + 6)], fill=color, width=2)
        draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], outline=color, width=2)
    draw.rectangle([(0, 0), (image.width, 16)], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 255))
    return image


def _save_contact_sheet(scene, views, target):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    panels = []
    for name in ("head", "wrist", "shoulder"):
        view = views[name]
        pixel, _z = project_world_to_pixel(view, target)
        visible, _visible_pixel, _detail = is_visible(view, target)
        panels.append(_mark_target(view.rgb, pixel, f"{scene}:{name}", visible))
    sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    sheet.save(os.path.join(OUTPUT_DIR, f"shoulder_{scene}.png"))


class TestShoulderCameraPyBullet(unittest.TestCase):
    def tearDown(self):
        if p is not None and p.isConnected():
            p.disconnect()
        _reset_mutable_config()

    def _assert_scene(self, scene):
        sim, environment, robot = _boot(scene)
        target = _grasp_target(sim, environment) if scene == "grasp" else _door_target(sim, environment)
        views = {name: robot.capture_camera_view(name, environment)
                 for name in ("head", "wrist", "shoulder")}

        shoulder = views["shoulder"]
        self.assertEqual(shoulder.rgb.shape[:2], (config.image_height, config.image_width))
        self.assertEqual(shoulder.depth.shape, (config.image_height, config.image_width))

        pixel, z_eye = project_world_to_pixel(shoulder, target)
        self.assertIsNotNone(pixel, f"{scene} target is behind the shoulder camera")
        self.assertGreater(z_eye, 0.0)
        self.assertTrue(in_bounds(shoulder, pixel, margin=MARGIN_PX),
                        f"{scene} target projects to {np.round(pixel, 1).tolist()}, too close "
                        f"to the edge of {shoulder.width}x{shoulder.height}")
        visible, visible_pixel, detail = is_visible(shoulder, target)
        self.assertTrue(visible, f"{scene} target is not visible from shoulder at "
                        f"{None if visible_pixel is None else np.round(visible_pixel, 1).tolist()}: "
                        f"{detail}")

        angle = _ray_angle_deg(views["head"], shoulder, target)
        self.assertGreaterEqual(angle, 45.0,
                                f"{scene} head/shoulder ray angle is only {angle:.1f} degrees")
        print(f"[shoulder-camera] {scene}: target={np.round(target, 4).tolist()} "
              f"pixel={np.round(pixel, 1).tolist()} angle={angle:.1f} deg "
              f"eye={np.round(shoulder.position, 4).tolist()}")
        _save_contact_sheet(scene, views, target)

    def test_grasp_shoulder_camera_sees_cube(self):
        self._assert_scene("grasp")

    def test_door_shoulder_camera_sees_latch(self):
        self._assert_scene("door")


if __name__ == "__main__":
    unittest.main()
