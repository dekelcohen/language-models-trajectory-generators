"""Round-trip pinhole test: 3D world point -> 2D pixel -> 3D world point.

The test is parametrized over every PyBullet sim-env task (the door plus all
seven Franka Kitchen tasks). For each one it:

1. Boots the sim-env headlessly (DIRECT) through the same code path production
   uses (``env.Environment`` + ``Robot.get_camera_image``).
2. Takes a *ground-truth* world position from ``simenv.get_state()`` -- the
   origin of the task's target link -- so no coordinates are hardcoded.
3. Projects it to a pixel with ``utils.project_3d_world_pos_to_2d_pixel``.
4. Samples the rendered depth buffer at that pixel.
5. Unprojects with ``utils.get_world_point_world_frame`` and asserts the result
   is within tolerance of the original point.

Steps 3 and 5 are the *production* functions, so this exercises the real
camera-matrix plumbing rather than re-deriving it locally.
"""

import unittest

import numpy as np
import pybullet as p
import pybullet_data

import config
import utils


# The reconstructed point lies on the object *surface* while the ground-truth
# point is the link origin, which is usually inside the mesh. The tolerance has
# to cover that offset plus depth-buffer quantisation.
POSITION_TOLERANCE_M = 0.12

# How much nearer to the camera the depth sample has to be before we call the
# target occluded rather than mis-reconstructed.
OCCLUSION_MARGIN_M = 0.05

# The door task's head camera looks at the door handle straight through the
# free-standing pole, so the handle is not visible and the depth sample lands on
# the pole instead. This is a pre-existing framing issue in the door scene, not a
# projection bug; it is asserted here so a regression elsewhere still fails.
KNOWN_OCCLUDED_TASKS = ["door"]

SIM_ENV_TASKS = [
    "door",
    "franka_kitchen:microwave",
    "franka_kitchen:slide_cabinet",
    "franka_kitchen:hinge_cabinet",
    "franka_kitchen:light_switch",
    "franka_kitchen:top_burner",
    "franka_kitchen:bottom_burner",
    "franka_kitchen:kettle",
]


class _Args:
    """Minimal stand-in for the argparse namespace the env/robot expect."""

    mode = "default"
    robot = "franka"
    task = None
    save_grasp_inputs = False


def _boot(task_name):
    """Bring up ``task_name`` in DIRECT mode and settle the physics."""
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
    args.task = task_name
    # utils reads args globally for the grasp-input dump; keep it disabled.
    utils.args = args

    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger("pinhole_test.log"), sim)
    sim.build()
    for _ in range(180):
        environment.update()
    return environment, robot


def _ground_truth_point(environment):
    """World position of the task's target link, straight from the sim-env."""
    state = environment.simenv.get_state()
    for key in ("target_link_pos", "target_position", "handle_position", "door_handle_pos"):
        value = state.get(key)
        if value is not None:
            return np.array(value, dtype=float)
    raise AssertionError(f"sim-env get_state() exposes no target position: {sorted(state)}")


class TestCameraUnprojection(unittest.TestCase):
    def tearDown(self):
        if p.isConnected():
            p.disconnect()

    def test_2d_pixel_coords_to_3d_world_coords(self):
        failures = []
        occluded = []
        for task_name in SIM_ENV_TASKS:
            with self.subTest(task=task_name):
                result = self._round_trip(task_name)
                print(
                    f"[Pinhole] {task_name:30s} known={np.round(result['known'], 4)} "
                    f"reconstructed={np.round(result['reconstructed'], 4)} "
                    f"error={result['error']:.4f} m "
                    f"reprojection={result['reprojection_error']} px"
                )
                # The pinhole maths itself must always round-trip exactly: the
                # reconstructed point has to project back to the pixel it came
                # from. This holds whatever surface the depth sample landed on.
                self.assertLessEqual(
                    result["reprojection_error"],
                    1,
                    f"{task_name}: unprojection is inconsistent with projection "
                    f"({result['reprojection_error']} px apart)",
                )
                if result["error"] <= POSITION_TOLERANCE_M:
                    continue
                if result["occluded"]:
                    # Something nearer to the camera owns this pixel, so the depth
                    # sample cannot be on the target. That is a scene-framing
                    # problem, not a projection bug.
                    occluded.append(task_name)
                    print(
                        f"[Pinhole] {task_name}: TARGET OCCLUDED - the depth sample is "
                        f"{result['known_ray_depth'] - result['reconstructed_ray_depth']:.3f} m "
                        "in front of the target, so the head camera cannot see it."
                    )
                    continue
                failures.append(f"{task_name}: {result['error']:.4f} m")
        self.assertFalse(
            failures,
            f"round-trip error above {POSITION_TOLERANCE_M} m for: {'; '.join(failures)}",
        )
        self.assertEqual(
            occluded,
            KNOWN_OCCLUDED_TASKS,
            "set of tasks whose target is hidden from the head camera changed; "
            f"expected {KNOWN_OCCLUDED_TASKS}, got {occluded}",
        )

    def _round_trip(self, task_name):
        environment, robot = _boot(task_name)
        try:
            known_world_pos = _ground_truth_point(environment)

            # Render through the production path so the returned camera pose and
            # matrices are exactly the ones the agent would receive.
            camera_position, camera_orientation_q, view_matrix, projection_matrix = (
                robot.get_camera_image("head", environment, False, None, None)
            )
            image_size = (config.image_width, config.image_height)
            cam_info = {
                "head": {"viewMatrix": view_matrix, "projectionMatrix": projection_matrix},
                "new_3d_proj": True,
            }

            pixel = utils.project_3d_world_pos_to_2d_pixel(
                camera_position, camera_orientation_q, "head", image_size, known_world_pos, cam_info
            )
            self.assertTrue(pixel, f"{task_name}: projection returned no pixel")
            pixel_x, pixel_y = int(pixel[0]), int(pixel[1])
            self.assertTrue(
                0 <= pixel_x < config.image_width and 0 <= pixel_y < config.image_height,
                f"{task_name}: target projects outside the head image at {pixel}; "
                "the camera framing for this task needs adjusting",
            )

            _, _, _, depth_buffer, _ = p.getCameraImage(
                config.image_width,
                config.image_height,
                viewMatrix=np.asarray(view_matrix).flatten(order="F"),
                projectionMatrix=np.asarray(projection_matrix).flatten(order="F"),
                renderer=p.ER_TINY_RENDERER,
            )
            depth = np.array(depth_buffer).reshape(config.image_height, config.image_width)
            depth_value = float(depth[pixel_y, pixel_x])
            self.assertLess(
                depth_value,
                0.999,
                f"{task_name}: pixel {pixel} sampled the background - nothing is rendered "
                "where the target should be",
            )

            reconstructed = np.asarray(
                utils.get_world_point_world_frame(
                    camera_position,
                    camera_orientation_q,
                    "head",
                    image_size,
                    [pixel_x, pixel_y, depth_value],
                    cam_info=cam_info,
                ),
                dtype=float,
            ).squeeze()

            reprojected = utils.project_3d_world_pos_to_2d_pixel(
                camera_position, camera_orientation_q, "head", image_size, reconstructed, cam_info
            )
            reprojection_error = int(
                max(abs(reprojected[0] - pixel_x), abs(reprojected[1] - pixel_y))
            ) if reprojected else 10 ** 6

            # Distance along the viewing direction, used to tell "the maths is
            # wrong" apart from "a nearer object hides the target".
            camera_xyz = np.asarray(camera_position, dtype=float)
            known_ray_depth = float(np.linalg.norm(known_world_pos - camera_xyz))
            reconstructed_ray_depth = float(np.linalg.norm(reconstructed - camera_xyz))

            return {
                "known": known_world_pos,
                "reconstructed": reconstructed,
                "error": float(np.linalg.norm(reconstructed - known_world_pos)),
                "reprojection_error": reprojection_error,
                "known_ray_depth": known_ray_depth,
                "reconstructed_ray_depth": reconstructed_ray_depth,
                "occluded": reconstructed_ray_depth < known_ray_depth - OCCLUSION_MARGIN_M,
            }
        finally:
            if p.isConnected():
                p.disconnect()


if __name__ == "__main__":
    unittest.main()
