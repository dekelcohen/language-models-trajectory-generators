"""Rigid-body refinement wrapped around any other 3D lift provider.

Why this exists
---------------
Every triangulating provider in this package - ``triangulate``, ``depth_fusion`` and LAPA
itself - solves **each point completely independently**. There is no inter-point coupling
anywhere in them, so nineteen good points cannot correct one drifting point.

That would be tolerable if a bad point were detectable from its reprojection residual, but
at exactly two cameras it is not: the DLT system is exactly determined (4 equations, 4
homogeneous unknowns), so *any* pair of pixels triangulates to a point that reprojects
almost perfectly. Measured in ``tests/test_tracker3d.py``: drifting one view by 47 px moves
the 3D point by more than a centimetre while the reprojection residual stays around
0.2 px. Residual-based outlier rejection is structurally blind there - and head+wrist is
exactly two cameras.

A rigid-body constraint is the only mechanism that can see that error, because it uses a
completely independent signal: the object's known shape. This provider stores the object's
point cloud once (the *template*), fits ``R, t`` from template to the current measurement
with the robust Kabsch in :mod:`tracking.motion`, and then reprojects that pose back onto
the individual points. A point that disagrees with the shape everybody else agrees on is
replaced by its predicted position.

Measured on the synthetic 10-point box in ``tests/test_rigid_refine.py``, two cameras, one
point's wrist pixel drifted by 47 px: the raw ``triangulate`` point is **0.250 m** off truth
while its reprojection residual is 2.5 px (the ``max_reproj_px`` gate sits at 25 px and
never fires); after rigid refinement the same point lands **4.5 mm** from truth averaged
over 50 trials with 0.5 px pixel noise on every track, and exactly on it with noiseless
pixels.

The failure mode that matters
-----------------------------
If *many* points disagree, the object has genuinely moved non-rigidly, the track has
collapsed, or the template is stale. "Correcting" every point onto that pose would then
manufacture a smooth, confident, entirely wrong trajectory - strictly worse than the raw
input, because the error becomes undetectable. So refinement only applies when at least
``min_inlier_frac`` of the corresponded points agree with the fitted pose; otherwise the
wrapped provider's raw output is passed through untouched and the event is logged.

Limitations
-----------
* **Genuinely non-rigid targets** (cloth, a human hand, a deformable bag) violate the
  premise. The inlier-fraction gate degrades this to a passthrough rather than corrupting
  the track, but the provider adds nothing there.
* **Articulated objects** are rigid only per link. A door handle on an opening door is
  rigid as a set of points on the handle; a template spanning *handle + door panel* is not.
  Seed points come from one object mask, so this is usually satisfied in practice.
* **Scale is assumed fixed.** No similarity transform is fitted, by design: a scale degree
  of freedom would happily absorb a depth-scale error instead of exposing it.
"""

import numpy as np

from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D
from tracking import geometry
from tracking.motion import fit_rigid_motion, point_spread


class _Template:
    """One object's captured shape, in a centroid-centred local frame."""

    __slots__ = ("index", "local", "frame", "stale")

    def __init__(self, index, local, frame):
        self.index = index          # (N,) seed indices, ascending
        self.local = local          # (N, 3) template points, centroid at origin
        self.frame = frame          # frame counter at capture
        self.stale = 0              # consecutive frames the fit was unusable


class RigidRefineTracker3D(MultiCamTracker3D):
    """Decorator: wraps another provider and enforces a rigid-body shape constraint."""

    name = "rigid_refine"

    def __init__(self, base="triangulate", base_kwargs=None, tolerance_m=0.01,
                 min_inlier_frac=0.6, min_points=4, blend=0.0, min_extent_m=0.005,
                 collinear_ratio=0.05, max_stale_frames=5, **kwargs):
        super().__init__(base=base, tolerance_m=tolerance_m, min_inlier_frac=min_inlier_frac,
                         min_points=min_points, blend=blend, min_extent_m=min_extent_m,
                         collinear_ratio=collinear_ratio, max_stale_frames=max_stale_frames,
                         **kwargs)
        extra = dict(base_kwargs or {})
        extra.setdefault("logger", kwargs.get("logger"))
        if isinstance(base, str):
            from providers.tracker3d.factory import get_tracker3d
            self.base = get_tracker3d(base, **extra)
        else:
            self.base = base
        self.base_name = getattr(self.base, "name", "custom")

        self.tolerance_m = float(tolerance_m)
        self.min_inlier_frac = float(min_inlier_frac)
        # 3 points determine a pose exactly, so a 3-point fit has zero redundancy and can
        # never expose a corrupted point. 4 is the smallest set where the constraint bites.
        self.min_points = max(4, int(min_points))
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.min_extent_m = float(min_extent_m)
        self.collinear_ratio = float(collinear_ratio)
        self.max_stale_frames = int(max_stale_frames)

        self._templates = {}
        self._frames = {}

    @property
    def needs_depth(self):
        return getattr(self.base, "needs_depth", True)

    @property
    def min_cameras(self):
        return getattr(self.base, "min_cameras", 1)

    def reset(self):
        self._templates.clear()
        self._frames.clear()
        try:
            self.base.reset()
        except Exception:                                    # never raise: ABC contract
            pass

    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None):
        try:
            result = self.base.lift(obj_name, views, cam_tracks, point_index=point_index,
                                    seed_world_points=seed_world_points)
        except Exception as exc:
            return Lift3DResult(meta={"provider": self.name, "base": self.base_name,
                                      "why": f"base raised {type(exc).__name__}: {exc}"})
        if result is None:
            return Lift3DResult(meta={"provider": self.name, "base": self.base_name,
                                      "why": "base returned nothing"})
        self._frames[obj_name] = self._frames.get(obj_name, -1) + 1
        try:
            return self._refine(obj_name, result)
        except Exception as exc:                             # never raise: ABC contract
            self._log(f"[rigid] refinement error for '{obj_name}': "
                      f"{type(exc).__name__}: {exc} - passing raw output through")
            result.meta.setdefault("provider", self.name)
            result.meta["refine"] = "error"
            return result

    # -- refinement --------------------------------------------------------
    def _refine(self, obj_name, result):
        points, index = self._corresponded(result)
        meta = result.meta
        meta["base"] = meta.get("provider", self.base_name)
        meta["provider"] = self.name

        if points is None or points.shape[0] < self.min_points:
            meta["refine"] = "too_few_points"
            meta["n_corresponded"] = 0 if points is None else int(points.shape[0])
            return result

        template = self._templates.get(obj_name)
        if template is None:
            self._capture(obj_name, points, index, meta)
            return result

        shared = np.intersect1d(template.index, index)
        if shared.shape[0] < self.min_points:
            # A re-seed replaced the tracked point set: the old template no longer describes
            # what is being measured, so it must be discarded rather than fitted against.
            self._log(f"[rigid] '{obj_name}' template invalidated: only {shared.shape[0]} of "
                      f"{template.index.shape[0]} template points still tracked (re-seed?)")
            self._templates.pop(obj_name, None)
            self._capture(obj_name, points, index, meta)
            return result

        t_rows = np.searchsorted(template.index, shared)
        m_rows = np.searchsorted(index, shared)
        source = template.local[t_rows]
        measured = points[m_rows]

        fit = fit_rigid_motion(source, measured, min_points=3, logger=None)
        rmse = float(fit.rmse)
        meta["rigid_rmse_m"] = None if not np.isfinite(rmse) else round(rmse, 6)
        meta["rigid_n_used"] = int(fit.n_used)
        meta["n_corresponded"] = int(points.shape[0])

        if not np.isfinite(rmse):
            return self._fall_through(obj_name, template, result, meta, "fit_failed")

        predicted = source @ fit.R.T + fit.t
        deviation = np.linalg.norm(measured - predicted, axis=1)
        agree = deviation <= self.tolerance_m
        inlier_frac = float(np.mean(agree))
        meta["rigid_inlier_frac"] = round(inlier_frac, 4)
        meta["rigid_max_dev_m"] = round(float(np.max(deviation)), 6)

        if inlier_frac < self.min_inlier_frac:
            # The object moved non-rigidly, the whole track collapsed, or the template is
            # wrong. Snapping every point onto this pose would be worse than doing nothing.
            return self._fall_through(obj_name, template, result, meta, "low_inlier_frac")

        template.stale = 0
        corrected = points.copy()
        bad_rows = m_rows[~agree]
        if bad_rows.size:
            replacement = predicted[~agree]
            if self.blend > 0.0:
                replacement = (self.blend * corrected[bad_rows] + (1.0 - self.blend) * replacement)
            corrected[bad_rows] = replacement

        self._grow(obj_name, template, points, index, shared, fit, rmse)

        centroid, n_used = geometry.robust_centroid(corrected, np.ones(len(corrected), dtype=bool))
        if centroid is None:
            return self._fall_through(obj_name, template, result, meta, "centroid_rejected_all")

        meta["refine"] = "applied"
        meta["n_corrected"] = int(bad_rows.size)
        meta["corrected_index"] = [int(i) for i in index[m_rows[~agree]]]
        meta["n_template"] = int(template.index.shape[0])
        meta["shift_m"] = round(float(np.linalg.norm(centroid - result.world_point)), 6)

        if bad_rows.size:
            self._log(f"[rigid] '{obj_name}' corrected {bad_rows.size}/{shared.shape[0]} point(s) "
                      f"(seed {meta['corrected_index']}), rmse={rmse:.5f}m, "
                      f"max_dev={meta['rigid_max_dev_m']:.4f}m, "
                      f"inlier_frac={inlier_frac:.2f}, centroid moved "
                      f"{meta['shift_m']:.5f}m")

        result.world_point = centroid
        result.per_point = corrected
        result.per_point_index = index
        result.disagreement = _spread(corrected)
        result.meta["n_points"] = int(n_used)
        return result

    def _corresponded(self, result):
        """Per-point 3D with a real seed index, sorted ascending; ``-1`` rows dropped."""
        if not result.ok or result.per_point is None:
            return None, None
        points = np.asarray(result.per_point, dtype=float).reshape(-1, 3)
        if result.per_point_index is None:
            index = np.arange(points.shape[0], dtype=int)
        else:
            index = np.asarray(result.per_point_index, dtype=int).reshape(-1)
        if index.shape[0] != points.shape[0]:
            return None, None
        keep = (index >= 0) & np.isfinite(points).all(axis=1)
        points, index = points[keep], index[keep]
        unique, first = np.unique(index, return_index=True)
        return points[first], unique

    def _capture(self, obj_name, points, index, meta):
        """Store the template once the geometry is good enough to constrain a pose."""
        spread = point_spread(points)
        extent = float(spread[0])
        if extent < self.min_extent_m:
            meta["refine"] = "template_extent_too_small"
            meta["template_extent_m"] = round(extent, 6)
            return
        if spread[1] <= self.collinear_ratio * max(extent, 1e-12):
            # Collinear points leave rotation about the line unobservable, so the fitted
            # pose is arbitrary and "corrections" derived from it would be invented.
            meta["refine"] = "template_collinear"
            return
        self._templates[obj_name] = _Template(index.copy(), points - points.mean(axis=0),
                                              self._frames.get(obj_name, 0))
        meta["refine"] = "template_captured"
        meta["n_template"] = int(index.shape[0])
        self._log(f"[rigid] '{obj_name}' template captured: {index.shape[0]} points "
                  f"(seed {[int(i) for i in index]}), extent={extent:.4f}m")

    def _grow(self, obj_name, template, points, index, shared, fit, rmse):
        """Adopt points that became visible after capture, using the current good pose.

        Deliberately chosen over waiting for one perfect frame: occlusion means some seed
        points may never all be visible simultaneously, and a template that only ever covers
        the t=0 subset would leave later points permanently unconstrained.
        """
        if rmse > self.tolerance_m:
            return
        new = np.setdiff1d(index, template.index)
        if new.size == 0:
            return
        rows = np.searchsorted(index, new)
        local = (points[rows] - fit.t) @ fit.R
        merged_index = np.concatenate([template.index, new])
        merged_local = np.concatenate([template.local, local], axis=0)
        order = np.argsort(merged_index)
        template.index = merged_index[order]
        template.local = merged_local[order]
        self._log(f"[rigid] '{obj_name}' template grew by {new.size} point(s) "
                  f"(seed {[int(i) for i in new]}) to {template.index.shape[0]}, "
                  f"rmse={rmse:.5f}m")

    def _fall_through(self, obj_name, template, result, meta, why):
        """Pass the wrapped provider's raw output through unmodified, and age the template."""
        meta["refine"] = "fell_through"
        meta["why_no_refine"] = why
        template.stale += 1
        self._log(f"[rigid] '{obj_name}' fell through to raw {self.base_name} output: {why} "
                  f"(inlier_frac={meta.get('rigid_inlier_frac')}, "
                  f"rmse={meta.get('rigid_rmse_m')}, stale={template.stale})")
        if template.stale >= self.max_stale_frames:
            self._log(f"[rigid] '{obj_name}' template dropped after {template.stale} "
                      f"consecutive unusable frames - will re-capture")
            self._templates.pop(obj_name, None)
        return result


def _spread(points):
    if len(points) < 2:
        return None
    return float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
