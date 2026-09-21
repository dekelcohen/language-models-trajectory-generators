"""OpenGL <-> OpenCV camera-convention conversion.

Everything in this repo (PyBullet, Genesis, ``tracking/geometry.py``) speaks **OpenGL**:
a flat 16-element *column-major* view matrix whose camera looks down **-Z** with **+Y up**,
plus a GL perspective projection matrix. Multi-view 3D libraries - LAPA, OpenCV's
``triangulatePoints``, anything built on COLMAP conventions - speak **OpenCV**: a 4x4
world-to-camera matrix whose camera looks down **+Z** with **+Y down**, plus a pixel-unit
intrinsics matrix ``K``.

This module is the single, tested bridge between the two.

Why it gets its own module (and its own test):
  * The failure mode is **silent**. Feed a raw GL view matrix to a triangulator and every
    point lands *behind* the camera, so every observation is marked invalid, so the tracker
    falls back to its previous estimate and simply looks **frozen** - no exception, no
    warning, just a tracker that never moves. ``assert_opencv_frame_sane`` exists to turn
    that into a loud failure.
  * ``utils.get_intrinsics_extrinsics`` must **not** be reused here: it builds ``K`` with
    the principal point pinned at ``(0, 0)`` (callers subtract the image centre themselves)
    and it returns a *camera-to-world* ``Rt``. Both are wrong for this purpose.

Conventions used throughout:
  * world coordinates are metres, Z-up (the repo convention, see ``sim_adapter/base.py``)
  * 2D points are ``(x=column, y=row)`` in pixels, image origin top-left
  * ``K`` is in pixels **at the resolution the points were measured at**
"""

import numpy as np

#: Maps an OpenGL camera frame to an OpenCV one: flip Y (up -> down) and Z (backward ->
#: forward). It is its own inverse, so the same matrix converts in both directions.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def mat4(value):
    """Accept a 4x4 array or a flat 16-element **column-major** GL matrix.

    Mirrors :func:`tracking.geometry.mat4`. The column-major detail matters: PyBullet and
    Genesis both hand back ``m[col * 4 + row]`` ordering, and a naive C-order reshape
    silently transposes the matrix - which still "works" numerically and produces poses
    that are wrong in a way that is very hard to see.
    """
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4, order="F")
    raise ValueError(f"Expected a 4x4 matrix or 16 elements, got shape {arr.shape}")


def gl_view_to_opencv_w2c(view_matrix):
    """GL view matrix -> OpenCV **world-to-camera** 4x4.

    The GL view matrix already maps world -> camera; only the axis convention differs, so
    the conversion is a left-multiplication by :data:`GL_TO_CV`::

        X_cam_cv = GL_TO_CV @ (view_matrix @ X_world)

    After this, a point in front of the camera has ``X_cam_cv[2] > 0``.
    """
    return GL_TO_CV @ mat4(view_matrix)


def opencv_w2c_to_gl_view(w2c):
    """Inverse of :func:`gl_view_to_opencv_w2c` (``GL_TO_CV`` is an involution)."""
    return GL_TO_CV @ mat4(w2c)


def intrinsics_from_gl_projection(projection_matrix, width, height):
    """GL projection matrix + render size -> pixel-unit ``K`` (3x3).

    The GL perspective matrix stores ``m[0,0] = f / aspect`` and ``m[1,1] = f`` where
    ``f = 1 / tan(fov / 2)``, in *NDC* units. Pixels follow from the viewport transform::

        fx = m[0,0] * width  / 2
        fy = m[1,1] * height / 2

    The principal point comes from the (usually zero) NDC offsets ``m[0,2]`` / ``m[1,2]``.
    The Y sign flips because NDC y points **up** while pixel rows point **down** - the same
    flip ``tracking.geometry.project_world_to_pixel`` applies as ``(1 - ndc_y)``.
    """
    proj = mat4(projection_matrix)
    width = float(width)
    height = float(height)

    fx = float(proj[0, 0]) * width / 2.0
    fy = float(proj[1, 1]) * height / 2.0
    # proj[*, 2] is the NDC-space principal-point offset; zero for a centred frustum.
    cx = (1.0 - float(proj[0, 2])) * width / 2.0
    cy = (1.0 + float(proj[1, 2])) * height / 2.0

    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=float)


def camera_matrices(view, width=None, height=None):
    """Convenience: a :class:`tracking.types.CameraView` -> ``(K, w2c_cv)``.

    ``width``/``height`` default to the view's own render size, which is what ``K`` must be
    expressed in.
    """
    width = view.width if width is None else width
    height = view.height if height is None else height
    return (intrinsics_from_gl_projection(view.projection_matrix, width, height),
            gl_view_to_opencv_w2c(view.view_matrix))


def projection_matrix_3x4(K, w2c):
    """``P = K [R | t]`` (3x4), the matrix a DLT triangulator consumes."""
    return np.asarray(K, dtype=float) @ mat4(w2c)[:3, :]


def project_opencv(K, w2c, world_points):
    """Project world points with OpenCV conventions.

    Returns ``(pixels (N, 2), z_cam (N,))``. ``z_cam`` is metres along the optical axis and
    is **positive in front of the camera** - directly comparable to a metric depth sample,
    and the quantity :func:`assert_opencv_frame_sane` checks.
    """
    pts = np.asarray(world_points, dtype=float).reshape(-1, 3)
    cam = (mat4(w2c) @ np.hstack([pts, np.ones((pts.shape[0], 1))]).T).T[:, :3]
    z = cam[:, 2]
    # Guard the divide so a point exactly on the image plane yields inf, not a warning.
    safe_z = np.where(np.abs(z) < 1e-12, np.nan, z)
    uv = (np.asarray(K, dtype=float) @ (cam / safe_z[:, None]).T).T[:, :2]
    return uv, z


# -- AABB normalisation (what LAPA operates in) ----------------------------


def aabb_from_points(world_points, pad=0.05, min_half=1e-3, mad_scale=6.0):
    """Robust axis-aligned box around a point cloud -> ``(center (3,), half (3,))``.

    Outlier rejection matters here: a single seed point that deprojected onto the
    background sits metres away, and letting it into the box inflates ``half`` so much that
    every *real* point collapses toward 0 in normalised space. LAPA's frozen ``BatchNorm1d``
    statistics assume inputs that fill ``[-1, 1]``, so an inflated box quietly degrades the
    model.

    LAPA's own dataset builder uses 1st/99th percentiles, which works because it sees
    thousands of points per scene. We typically have 3-8 seed points, where a percentile is
    indistinguishable from min/max - so we reject on **distance from the median, scaled by
    the MAD**, which is well behaved at small N, then take the extent of the survivors.
    """
    pts = np.asarray(world_points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise ValueError("aabb_from_points needs at least one finite point")

    if pts.shape[0] >= 3:
        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        mad = float(np.median(np.abs(dist - np.median(dist))))
        # 1.4826 * MAD estimates sigma for a normal distribution; the floor keeps a
        # perfectly tight cluster from rejecting its own jitter.
        limit = max(mad_scale * 1.4826 * mad, 1e-6)
        keep = dist <= np.median(dist) + limit
        if np.count_nonzero(keep) >= 2:
            pts = pts[keep]

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    half = half + pad * np.maximum(half, min_half)
    return center, np.maximum(half, min_half)


def normalize_points(world_points, center, half):
    """World metres -> LAPA's normalised ``[-1, 1]^3`` box."""
    pts = np.asarray(world_points, dtype=float)
    return (pts - np.asarray(center, dtype=float)) / np.asarray(half, dtype=float)


def denormalize_points(norm_points, center, half):
    """Inverse of :func:`normalize_points`."""
    pts = np.asarray(norm_points, dtype=float)
    return pts * np.asarray(half, dtype=float) + np.asarray(center, dtype=float)


def build_w2c_normalized(w2c, center, half):
    """Warp a world-frame ``w2c`` so it acts on normalised coordinates.

    Numpy twin of LAPA's ``lapa.models.lapa.build_w2c_normalized`` (kept here so the
    geometry can be unit-tested with no torch and no LAPA clone on the path)::

        X_world = X_norm * half + center
        X_cam   = R X_world + t = (R diag(half)) X_norm + (R center + t)
    """
    m = mat4(w2c)
    R, t = m[:3, :3], m[:3, 3]
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)

    out = np.eye(4, dtype=float)
    out[:3, :3] = R * half[None, :]      # scale columns
    out[:3, 3] = R @ center + t
    return out


# -- guardrails ------------------------------------------------------------


def assert_opencv_frame_sane(K, w2c, world_points, image_size=None, name="camera"):
    """Raise if ``world_points`` do not land in front of an OpenCV camera.

    This is the check that catches a missing :data:`GL_TO_CV` flip. Without it the symptom
    is a tracker that silently never moves, which costs hours to diagnose; with it the
    failure is immediate and names the cause.

    ``image_size`` (``(width, height)``) additionally requires the points to be in frame.
    """
    uv, z = project_opencv(K, w2c, world_points)
    behind = int(np.count_nonzero(~(z > 0.0)))
    if behind:
        raise AssertionError(
            f"[{name}] {behind}/{len(z)} point(s) are behind the camera (z_cam <= 0). "
            "The extrinsics are almost certainly still in OpenGL convention - run them "
            "through tracking.camera_convert.gl_view_to_opencv_w2c first."
        )
    if image_size is not None:
        width, height = image_size
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        if not np.all(inside):
            raise AssertionError(
                f"[{name}] {int(np.count_nonzero(~inside))}/{len(uv)} point(s) project "
                f"outside the {width}x{height} image: {uv[~inside][:4].tolist()}"
            )
    return uv, z
