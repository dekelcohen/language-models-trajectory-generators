"""Pure-geometry multi-view triangulation - no depth buffer in the loop.

This is the control experiment for the whole LAPA evaluation. LAPA is, at its core,
IRLS-weighted DLT triangulation plus a learned per-view weight, a visibility logit and a
residual clamped to 2% of the workspace half-extent. This module is that same triangulation
with *uniform* weights. So:

* if ``triangulate`` already beats ``depth_fusion``, the win came from replacing depth
  sampling with geometry - no checkpoint, no DINOv2, no GPU required;
* if ``lapa`` then beats ``triangulate``, the win is genuinely LAPA's learned view
  weighting under occlusion, which is the claim worth paying a GPU for;
* if ``lapa`` *loses* to ``triangulate``, the adapter, the conventions or the
  normalisation are wrong - not the model.

That three-way reading is why this ships alongside the real providers even though it is a
debugging aid rather than an evaluation configuration.

**Correspondence is by seed index.** Two cameras' point arrays cannot be zipped
positionally: a re-seed drops seed points its camera cannot image, so ``head[2]`` and
``wrist[2]`` are routinely different physical points. Triangulating those against each
other yields a confident, smooth, entirely wrong 3D point - the worst kind of bug. Every
observation here is keyed on ``point_index``.
"""

import numpy as np

from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D
from tracking import camera_convert, geometry


class TriangulateTracker3D(MultiCamTracker3D):
    """Classic IRLS-weighted DLT over the per-camera 2D tracks."""

    name = "triangulate"
    needs_depth = False
    min_cameras = 2

    def __init__(self, iters=10, sigma_px=2.0, max_reproj_px=25.0, **kwargs):
        super().__init__(iters=iters, sigma_px=sigma_px, max_reproj_px=max_reproj_px, **kwargs)
        self.iters = int(iters)
        self.sigma_px = float(sigma_px)
        self.max_reproj_px = float(max_reproj_px)

    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None):
        usable = self._usable(cam_tracks)
        if len(usable) < self.min_cameras:
            return Lift3DResult(meta={"provider": self.name,
                                      "why": f"need {self.min_cameras} cams, have {len(usable)}"})

        # Group every visible observation by the seed index it belongs to.
        obs = {}                       # seed_index -> [(cam, (x, y)), ...]
        cams_seen = set()
        for cam, (points, visible) in usable.items():
            view = views.get(cam)
            if view is None:
                continue
            indices = self._indices_for(cam, points, point_index)
            for j, idx in enumerate(indices):
                if idx < 0 or not visible[j]:
                    continue
                obs.setdefault(int(idx), []).append((cam, points[j]))
            cams_seen.add(cam)

        cameras = {cam: camera_convert.camera_matrices(views[cam]) for cam in cams_seen}

        world_points, kept_index, used_cams, residuals = [], [], set(), []
        worst = []
        for idx in sorted(obs):
            entries = [(c, uv) for c, uv in obs[idx] if c in cameras]
            if len(entries) < self.min_cameras:
                continue
            projections = [camera_convert.projection_matrix_3x4(*cameras[c]) for c, _uv in entries]
            pixels = np.asarray([uv for _c, uv in entries], dtype=float)
            point, residual = triangulate_irls(projections, pixels,
                                               iters=self.iters, sigma_px=self.sigma_px)
            if point is None or residual > self.max_reproj_px:
                continue
            # A point behind any contributing camera is a geometric impossibility and the
            # classic symptom of un-flipped OpenGL extrinsics.
            if any(np.dot(camera_convert.mat4(cameras[c][1])[2, :3], point)
                   + camera_convert.mat4(cameras[c][1])[2, 3] <= 0 for c, _uv in entries):
                continue
            world_points.append(point)
            kept_index.append(idx)
            residuals.append(residual)
            # The worst view's error is what says "one camera disagrees about this point",
            # which the median deliberately hides once rejection has worked.
            worst.append(float(np.max(reprojection_errors(projections, pixels, point))))
            used_cams.update(c for c, _uv in entries)

        if not world_points:
            return Lift3DResult(meta={"provider": self.name,
                                      "why": "no seed index visible in >=2 cameras",
                                      "n_candidates": len(obs)})

        points = np.asarray(world_points, dtype=float)
        centroid, n_used = geometry.robust_centroid(points, np.ones(len(points), dtype=bool))
        if centroid is None:
            return Lift3DResult(meta={"provider": self.name, "why": "robust centroid rejected all"})

        return Lift3DResult(
            world_point=centroid,
            disagreement=_spread(points),
            used_cams=sorted(used_cams),
            per_point=points,
            per_point_index=np.asarray(kept_index, dtype=int),
            weights={c: 1.0 for c in sorted(used_cams)},
            meta={"provider": self.name, "n_points": int(n_used),
                  "median_reproj_px": round(float(np.median(residuals)), 3),
                  "worst_reproj_px": round(float(np.max(worst)), 3) if worst else None},
        )


def triangulate_irls(projections, pixels, iters=10, sigma_px=2.0):
    """Iteratively reweighted DLT triangulation of one point.

    ``projections``: list of ``(3, 4)`` OpenCV projection matrices ``K @ [R|t]``.
    ``pixels``: ``(V, 2)`` the observed ``(x, y)`` in each of those cameras.

    Returns ``(point_xyz, median_reprojection_error_px)``, or ``(None, inf)``.

    The reweighting is what buys robustness to *one* bad 2D track among several good ones:
    solve, measure how far each camera's ray missed, shrink the weight of the cameras that
    missed badly, solve again.

    Two deliberate departures from LAPA's ``lapa/geom/dlt.py``, both measured:

    * **No per-row normalisation of ``A``.** LAPA scales its rows by ``sqrt(w)`` and then
      divides every row by its own norm, which cancels the weight exactly. The consequence
      there is that the learned view weights and the IRLS rounds have *no effect at all* on
      the triangulated point (verified numerically: crushing a view's weight to 1e-5 versus
      boosting it to 1e3 moves the result by ~1e-8 m). Omitting that normalisation is what
      makes the weights actually do something.
    * **Cauchy weights** ``1 / (1 + (e/sigma)^2)`` over 10 rounds, rather than LAPA's
      Gaussian ``0.05 + 0.95*exp(-(e/sigma)^2)`` over 2. On a 75-pixel single-view drift
      with 3 cameras, measured final error: Cauchy 0.00001 m, Huber 0.0015 m, Gaussian
      0.174 m. The Gaussian collapses the *good* views' weights too once the first solve is
      dragged off, and then cannot recover; Cauchy decays gently enough to keep them alive.

    **Two cameras is a special case worth knowing.** With V=2 the system is exactly
    determined (4 equations, 4 homogeneous unknowns), so *any* pair of pixels reprojects
    almost perfectly and the residual carries no information. Reweighting cannot help and
    a drifted track is silently accepted. Robustness here starts at 3 views.
    """
    projections = [np.asarray(P, dtype=float).reshape(3, 4) for P in projections]
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(projections) != len(pixels) or len(projections) < 2:
        return None, float("inf")

    weights = np.ones(len(projections), dtype=float)
    point = None
    for _ in range(max(1, int(iters))):
        rows = []
        for P, (x, y), w in zip(projections, pixels, weights):
            rows.append(w * (x * P[2] - P[0]))
            rows.append(w * (y * P[2] - P[1]))
        A = np.asarray(rows, dtype=float)
        try:
            _u, _s, vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return None, float("inf")
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-12:            # point at infinity: parallel rays
            return None, float("inf")
        point = homogeneous[:3] / homogeneous[3]
        if not np.all(np.isfinite(point)):
            return None, float("inf")

        errors = _reprojection_errors(projections, pixels, point)
        weights = 1.0 / (1.0 + (errors / max(sigma_px, 1e-6)) ** 2)

    return point, float(np.median(_reprojection_errors(projections, pixels, point)))


def reprojection_errors(projections, pixels, point):
    """Per-view reprojection error in pixels for a solved point.

    Public because the *spread* of these errors is the diagnostic, not their median. After
    successful outlier rejection the median is ~0 by construction (it is a robust
    statistic), while the rejected view still carries a large error - so ``max`` is what
    reveals "one camera disagrees", and the median is what says "the point is trustworthy".
    """
    return _reprojection_errors(
        [np.asarray(P, dtype=float).reshape(3, 4) for P in projections],
        np.asarray(pixels, dtype=float).reshape(-1, 2), point)


def _reprojection_errors(projections, pixels, point):
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    errors = []
    for P, uv in zip(projections, pixels):
        projected = P @ homogeneous
        if abs(projected[2]) < 1e-12:
            errors.append(float("inf"))
            continue
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - uv)))
    return np.asarray(errors, dtype=float)


def _spread(points):
    """Max distance of any triangulated point from their centroid - a scatter proxy."""
    if len(points) < 2:
        return None
    centre = points.mean(axis=0)
    return float(np.max(np.linalg.norm(points - centre, axis=1)))
