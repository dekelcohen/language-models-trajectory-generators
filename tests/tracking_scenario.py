"""Shared, simulator-agnostic scenario helpers for the tracking integration tests.

Both ``test_tracking_pybullet.py`` and ``test_tracking_genesis.py`` drive the *real*
:class:`tracking.session.TrackingSession` against a *real* rendered scene - no LLM, no
agent loop. Objects are moved directly with ``sim.set_base_pose``, so every frame has an
exact ground-truth world position to compare the fused estimate against.

Nothing here imports a simulator: the caller passes an already-connected
:class:`sim_adapter.base.SimAdapter`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from sim_adapter import camera_math  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.types import CameraView  # noqa: E402

RES = 256
FOV = 60.0


def render_view(sim, name, eye, target, up=(0.0, 0.0, 1.0), width=RES, height=RES):
    """Render one camera through the adapter and wrap it as a :class:`CameraView`.

    Mirrors what ``Robot.capture_camera_view`` does in production, including the
    metric-depth conversion that hides the PyBullet/Genesis encoding difference.
    """
    view_matrix = sim.compute_view_matrix(eye=list(eye), target=list(target), up=list(up))
    projection = sim.compute_projection_matrix(FOV, float(width) / float(height),
                                               config.near_plane, config.far_plane)
    frame = sim.render_camera(width, height, view_matrix, projection)
    depth = camera_math.depth_to_metric(frame.depth, sim.depth_encoding,
                                        config.near_plane, config.far_plane)
    return CameraView(name=name, rgb=np.asarray(frame.rgb),
                      depth=np.asarray(depth, dtype=np.float32),
                      view_matrix=geometry.mat4(view_matrix),
                      projection_matrix=geometry.mat4(projection),
                      near=float(config.near_plane), far=float(config.far_plane),
                      position=list(eye))


def surface_seed_points(view, world_center, spread_px=5, count=5):
    """Seed points that lie on the object's *visible surface*, as detect_object would.

    Deprojecting the object's centre would place a point inside the mesh, which the
    occlusion test then (correctly) rejects. Instead the centre is projected to a pixel,
    a small cross of neighbouring pixels is sampled from the depth buffer, and those are
    deprojected back - exactly the "2D points + depth -> 3D" path the tracker itself uses.

    Returns ``(world_points (N, 3), pixels (N, 2))``; raises when the object is not
    visible, which in a test means the scene is set up wrong.
    """
    pixel, _z_eye = geometry.project_world_to_pixel(view, world_center)
    if pixel is None or not geometry.in_bounds(view, pixel):
        raise AssertionError(f"object at {world_center} is not in the '{view.name}' view")

    offsets = [(0, 0), (-spread_px, 0), (spread_px, 0), (0, -spread_px), (0, spread_px)]
    world_points, pixels = [], []
    for dx, dy in offsets[:count]:
        candidate = np.array([pixel[0] + dx, pixel[1] + dy], dtype=float)
        depth = geometry.sample_depth(view, candidate)
        if depth is None:
            continue
        world_points.append(geometry.deproject_pixel_to_world(view, candidate, depth))
        pixels.append(candidate)
    if not world_points:
        raise AssertionError(f"no valid depth around {pixel} in the '{view.name}' view")
    return np.asarray(world_points), np.asarray(pixels)


def paint_occluder(view, world_pos, radius=40, standoff=0.25):
    """Simulate the robot arm passing between the camera and the object.

    Overwrites a disc of the depth buffer with a much closer surface, so
    ``geometry.is_visible`` reports the object as occluded in that camera. Doing it on
    the depth buffer keeps the test deterministic - no dependence on where the arm
    happens to swing - while exercising exactly the code path a real occlusion hits.
    """
    pixel, z_eye = geometry.project_world_to_pixel(view, world_pos)
    if pixel is None:
        return
    cx, cy = int(round(pixel[0])), int(round(pixel[1]))
    ys, xs = np.ogrid[:view.height, :view.width]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    view.depth[mask] = max(float(z_eye) - standoff, config.track_depth_min * 2.0)
    # The occluder must look different too, or the tracker would happily keep matching
    # the object's texture through it.
    view.rgb[mask] = 30


def tracking_error(report, name, ground_truth, seed_offset):
    """Distance between the fused estimate and the expected point.

    ``seed_offset`` is the constant vector from the object's origin to the centroid of the
    seed points (which sit on the surface facing the camera). Comparing against
    ``ground_truth + seed_offset`` measures tracking error rather than that fixed bias.
    """
    obj = report.objects[name]
    if obj.world_point is None:
        return None
    expected = np.asarray(ground_truth, dtype=float) + np.asarray(seed_offset, dtype=float)
    return float(np.linalg.norm(np.asarray(obj.world_point, dtype=float) - expected))
