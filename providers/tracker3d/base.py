"""The multi-camera 3D lift interface.

One object, all cameras, one frame -> one world point.

Contract notes that matter to implementations:

* **Point correspondence is by seed index, not by position in the array.** Each camera's
  tracker may hold a *subset* of the object's seed points, because re-seeding drops seed
  points that its camera cannot actually image. ``point_index[cam][j]`` gives the seed
  index of that camera's j-th tracked point, or ``-1`` for a point with no correspondence
  (the centroid fallback). A triangulator must align on those indices; zipping the arrays
  positionally silently triangulates *different physical points* against each other, which
  produces a plausible-looking but wrong 3D position.
* Implementations must never raise for ordinary bad input (no visible points, one camera,
  degenerate geometry). Return an empty :class:`Lift3DResult` and let the session mark the
  object lost - a tracking bug must not break a rollout.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Lift3DResult:
    """One object's fused 3D estimate for one frame."""

    world_point: Optional[np.ndarray] = None            # (3,) metres, world frame
    disagreement: Optional[float] = None                # metres between camera estimates
    used_cams: List[str] = field(default_factory=list)
    per_point: Optional[np.ndarray] = None              # (K, 3) individual 3D points
    per_point_index: Optional[np.ndarray] = None        # (K,) seed index of each row
    weights: Dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.world_point is not None and bool(np.all(np.isfinite(self.world_point)))

    def to_dict(self):
        return {
            "world_point": None if self.world_point is None
            else [round(float(v), 6) for v in np.asarray(self.world_point).reshape(-1)],
            "disagreement": None if self.disagreement is None else round(float(self.disagreement), 5),
            "used_cams": list(self.used_cams),
            "n_points": 0 if self.per_point is None else int(len(self.per_point)),
            "weights": {c: round(float(w), 4) for c, w in self.weights.items()},
            "meta": dict(self.meta),
        }


class MultiCamTracker3D(ABC):
    """Fuses several cameras' 2D tracks of one object into a world point."""

    name = "abstract"

    #: Whether the provider reads the depth buffer. ``False`` for pure-geometry providers,
    #: and the reason they are expected to survive noisy foreground/background depth.
    needs_depth = True

    #: Minimum number of cameras that must contribute before a point can be produced.
    min_cameras = 1

    def __init__(self, **kwargs):
        self.options = dict(kwargs)
        self.logger = kwargs.get("logger")

    @abstractmethod
    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None):
        """Return a :class:`Lift3DResult` for ``obj_name``.

        ``views``: ``{cam: CameraView}``.
        ``cam_tracks``: ``{cam: CamTrack}`` - already updated for this frame.
        ``point_index``: ``{cam: np.ndarray(int)}`` mapping each tracked point to its seed
        index (``-1`` = no correspondence). May be ``None`` for providers that do not need
        cross-camera correspondence.
        ``seed_world_points``: ``(N, 3)`` the object's original seed geometry, when known.
        """

    def reset(self):
        """Forget any per-rollout state. Stateless providers need not override."""

    # -- helpers shared by implementations ---------------------------------
    @staticmethod
    def _usable(cam_tracks):
        """Cameras with at least one visible tracked point this frame."""
        out = {}
        for cam, track in (cam_tracks or {}).items():
            if not track.seeded or track.points_2d is None:
                continue
            points = np.asarray(track.points_2d, dtype=float).reshape(-1, 2)
            visible = (np.ones(points.shape[0], dtype=bool) if track.visible is None
                       else np.asarray(track.visible, dtype=bool).reshape(-1))
            if visible.shape[0] != points.shape[0] or not np.any(visible):
                continue
            out[cam] = (points, visible)
        return out

    @staticmethod
    def _indices_for(cam, points_2d, point_index):
        """Seed indices for one camera's points, defaulting to positional order.

        The default is only correct when no seed point was dropped; callers that need
        genuine correspondence must supply ``point_index``.
        """
        if point_index is not None and point_index.get(cam) is not None:
            idx = np.asarray(point_index[cam], dtype=int).reshape(-1)
            if idx.shape[0] == points_2d.shape[0]:
                return idx
        return np.arange(points_2d.shape[0], dtype=int)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
