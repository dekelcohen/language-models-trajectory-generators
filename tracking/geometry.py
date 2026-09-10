"""Pixel <-> world geometry and multi-camera fusion for rollout tracking.

Deliberately independent of ``utils.py``: that module carries process-global state
(``utils.args``/``utils.logger``) that is only initialised in the agent process, while
tracking runs inside the simulator process. The math here is the same single
inverse-view-projection path used by ``utils.get_world_point_world_frame``, but it
consumes **metric** depth only, so PyBullet's GL z-buffer and Genesis' linear metres
are already reconciled by ``sim_adapter.camera_math.depth_to_metric`` upstream.
"""

import numpy as np

import config
from sim_adapter import camera_math
from tracking.types import CameraView


def _mat4(value):
    """Accept a 4x4 array or PyBullet's flat 16-element column-major sequence."""
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4, order="F")
    raise ValueError(f"Expected a 4x4 matrix or 16 elements, got shape {arr.shape}")


def view_projection(view: CameraView):
    return _mat4(view.projection_matrix) @ _mat4(view.view_matrix)


def project_world_to_pixel(view: CameraView, world_pos):
    """World point -> (x, y) float pixel coordinates plus the expected metric depth.

    Returns ``(pixel_xy, z_eye)``. ``z_eye`` is the distance along the optical axis,
    i.e. directly comparable to a metric depth sample. ``None`` is returned for points
    behind the camera.
    """
    vm = _mat4(view.view_matrix)
    cam_pt = vm @ np.append(np.asarray(world_pos, dtype=float), 1.0)
    # OpenGL view space looks down -z, so a point in front of the camera has cam_pt[2] < 0.
    z_eye = float(-cam_pt[2])
    if z_eye <= 0.0:
        return None, z_eye

    clip = _mat4(view.projection_matrix) @ cam_pt
    if abs(clip[3]) < 1e-12:
        return None, z_eye
    ndc = clip / clip[3]
    px = (ndc[0] + 1.0) * view.width / 2.0
    py = (1.0 - ndc[1]) * view.height / 2.0
    return np.array([px, py], dtype=float), z_eye


def deproject_pixel_to_world(view: CameraView, pixel_xy, depth_metric):
    """(x, y) pixel + metric depth -> world point (3,)."""
    ndc_x = (2.0 * float(pixel_xy[0]) / view.width) - 1.0
    ndc_y = 1.0 - (2.0 * float(pixel_xy[1]) / view.height)
    ndc_z = float(camera_math.metric_to_ndc_z(float(depth_metric), view.near, view.far))
    world_h = np.linalg.inv(view_projection(view)) @ np.array([ndc_x, ndc_y, ndc_z, 1.0])
    return world_h[:3] / world_h[3]


def in_bounds(view: CameraView, pixel_xy, margin=0):
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    return (margin <= x < view.width - margin) and (margin <= y < view.height - margin)


def sample_depth(view: CameraView, pixel_xy, window=1):
    """Median metric depth in a small window around a pixel; ``None`` when invalid.

    A median (rather than the single pixel) keeps the value stable when a track sits on
    an object edge, where one pixel may belong to the background.
    """
    if not in_bounds(view, pixel_xy):
        return None
    x, y = int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))
    x0, x1 = max(0, x - window), min(view.width, x + window + 1)
    y0, y1 = max(0, y - window), min(view.height, y + window + 1)
    patch = np.asarray(view.depth[y0:y1, x0:x1], dtype=float).reshape(-1)
    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > config.track_depth_min) & (patch < config.track_depth_max)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def is_visible(view: CameraView, world_pos, tol=None):
    """Is ``world_pos`` actually imaged by this camera (in frame and not occluded)?

    Returns ``(visible, pixel_xy, detail)``. The occlusion test compares the rendered
    depth at the projected pixel with the point's own distance from the camera: if the
    surface in front is closer by more than ``tol`` metres, something (typically the
    robot arm) is between the camera and the object.
    """
    tol = config.track_occlusion_tol if tol is None else tol
    pixel, z_eye = project_world_to_pixel(view, world_pos)
    if pixel is None:
        return False, None, "behind_camera"
    if not in_bounds(view, pixel):
        return False, pixel, "out_of_frame"
    z_rendered = sample_depth(view, pixel)
    if z_rendered is None:
        return False, pixel, "invalid_depth"
    if z_rendered < z_eye - tol:
        return False, pixel, f"occluded(dz={z_eye - z_rendered:.3f})"
    if z_rendered > z_eye + tol:
        # Rendered surface is *behind* where we expected the point: the estimate is
        # stale or floating in free space.
        return False, pixel, f"depth_mismatch(dz={z_rendered - z_eye:.3f})"
    return True, pixel, "ok"


def points_to_world(view: CameraView, points_2d, valid_mask=None):
    """Deproject a set of tracked 2D points; returns ``(world_points, valid)``.

    ``world_points`` is (N, 3) with NaN rows where the depth sample was unusable.
    """
    points_2d = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    n = points_2d.shape[0]
    world = np.full((n, 3), np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if valid_mask is not None and not bool(valid_mask[i]):
            continue
        depth = sample_depth(view, points_2d[i])
        if depth is None:
            continue
        world[i] = deproject_pixel_to_world(view, points_2d[i], depth)
        valid[i] = True
    return world, valid


def robust_centroid(world_points, valid, max_spread=None):
    """Median-based centroid of a point set with distance-based outlier rejection.

    A tracker that slips onto the background produces one wildly wrong point; the median
    is unaffected by it and the spread filter then removes it before averaging.
    """
    world_points = np.asarray(world_points, dtype=float).reshape(-1, 3)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    pts = world_points[valid & np.isfinite(world_points).all(axis=1)]
    if pts.shape[0] == 0:
        return None, 0
    if pts.shape[0] <= 2:
        return pts.mean(axis=0), pts.shape[0]
    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=1)
    limit = config.track_disagree_m if max_spread is None else max_spread
    keep = dist <= max(limit, float(np.median(dist)) * 3.0)
    if not np.any(keep):
        return med, 0
    return pts[keep].mean(axis=0), int(np.count_nonzero(keep))


def camera_weight(confidence, z_eye, n_points, depth_valid=True):
    """Fusion weight for one camera's world estimate.

    ``confidence`` x ``depth validity`` x ``1/z`` x point support. The 1/z term encodes
    that deprojection error grows with distance, so the wrist camera - which ends up
    centimetres from the object - dominates the head camera near the end of a reach.
    """
    if not depth_valid or confidence <= 0.0 or n_points <= 0:
        return 0.0
    z = max(float(z_eye), 1e-3)
    support = min(float(n_points), 8.0) / 8.0
    return float(confidence) * (1.0 / z) * (0.5 + 0.5 * support)


def fuse_world_points(estimates):
    """Weighted fusion of per-camera world points.

    ``estimates``: ``{cam_name: (world_point (3,), weight)}`` - zero/negative weights and
    ``None`` points are ignored.

    Returns ``(fused_point | None, disagreement_m | None, used_cams)`` where
    ``disagreement_m`` is the largest pairwise distance between contributing cameras.
    """
    used = [(cam, np.asarray(p, dtype=float), float(w))
            for cam, (p, w) in estimates.items()
            if p is not None and np.all(np.isfinite(np.asarray(p, dtype=float))) and float(w) > 0.0]
    if not used:
        return None, None, []
    if len(used) == 1:
        return used[0][1], None, [used[0][0]]

    pts = np.stack([p for _, p, _ in used])
    weights = np.array([w for _, _, w in used], dtype=float)
    disagreement = float(max(np.linalg.norm(pts[i] - pts[j])
                             for i in range(len(pts)) for j in range(i + 1, len(pts))))
    fused = (pts * weights[:, None]).sum(axis=0) / weights.sum()
    return fused, disagreement, [cam for cam, _, _ in used]
