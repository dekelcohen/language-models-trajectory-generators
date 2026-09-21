"""Depth-buffer 3D lift - the historical behaviour, as a provider.

Each camera has already deprojected its own visible 2D points through its own depth buffer
(``geometry.points_to_world`` -> ``robust_centroid``) during the per-camera update, so this
provider's job is purely **trust arbitration**: decide which cameras deserve a vote and
average their world points.

Two arbitration rules, both earned the hard way and both preserved verbatim here:

1. **A healthy camera is never averaged with an unhealthy one.** If any camera's track is
   ``ok``, only ``ok`` cameras vote. An occluder's surface deprojects tens of centimetres
   off; averaging it in corrupts an otherwise perfect frame.
2. **A camera that disagrees with the clearly healthiest one is dropped**, even if it looks
   ``ok``. A template that has latched onto an occluder keeps a high match score for many
   frames, so waiting for the re-seed cooldown to repair it would poison every estimate in
   the meantime. ``disagreement`` is still reported over *all* surviving cameras so the
   failure remains visible in the report.

Known weakness, and the reason the other providers exist: a tracked point near an object
edge samples *background* depth, which throws its deprojection metres away. The 3x3 median
in ``sample_depth`` and the outlier rejection in ``robust_centroid`` soften it, but the
failure is systematic exactly when it matters - while the gripper is closing on the object.
"""

import numpy as np

import config
from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D
from tracking import geometry


class DepthFusionTracker3D(MultiCamTracker3D):
    """Weighted average of the per-camera depth deprojections."""

    name = "depth_fusion"
    needs_depth = True
    min_cameras = 1

    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None):
        estimates = {}
        healths = {}
        per_point = []
        per_index = []

        healthy = {c for c, t in cam_tracks.items()
                   if t.status == "ok" and t.world_point is not None}

        for cam, track in cam_tracks.items():
            healths[cam] = float(track.health)
            if track.world_point is None:
                continue
            if healthy and cam not in healthy:
                continue
            view = views.get(cam)
            z_eye = 1.0
            if view is not None:
                _pixel, z_eye = geometry.project_world_to_pixel(view, track.world_point)
            weight = geometry.camera_weight(track.confidence, z_eye,
                                            track.n_visible, track.depth_valid)
            estimates[cam] = (track.world_point, weight * max(track.health, 1e-3))
            per_point.append(np.asarray(track.world_point, dtype=float))
            per_index.append(-1)

        _fused_all, disagreement, _used_all = geometry.fuse_world_points(estimates)
        trusted = _drop_disagreeing(estimates, healths)
        fused, _gap, used = geometry.fuse_world_points(trusted)

        return Lift3DResult(
            world_point=fused,
            disagreement=disagreement,
            used_cams=list(used),
            per_point=np.asarray(per_point, dtype=float) if per_point else None,
            per_point_index=np.asarray(per_index, dtype=int) if per_index else None,
            weights={c: float(w) for c, (_p, w) in trusted.items()},
            meta={"provider": self.name, "n_cams": len(estimates)},
        )


def _drop_disagreeing(estimates, healths):
    """Remove cameras that disagree with the clearly healthiest one."""
    if len(estimates) < 2:
        return estimates
    best = max(estimates, key=lambda c: healths.get(c, 0.0))
    best_point = np.asarray(estimates[best][0], dtype=float)
    best_health = healths.get(best, 0.0)
    trusted = {}
    for cam, entry in estimates.items():
        gap = float(np.linalg.norm(np.asarray(entry[0], dtype=float) - best_point))
        clear_winner = best_health >= 1.25 * max(healths.get(cam, 0.0), 1e-6)
        if cam != best and gap > config.track_disagree_m and clear_winner:
            continue
        trusted[cam] = entry
    return trusted or estimates
