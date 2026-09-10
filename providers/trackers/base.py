"""Point-tracker provider interface.

A tracker follows a **set of 2D points** (not a single bbox) through a stream of RGB
frames. That shape is what the tracking session needs - several points per object give
outlier rejection in 3D - and it is also what modern point trackers (CoTracker, TAPIR)
expose natively, so a future remote provider drops in without changing callers.

Implementations must:
  * be tolerant of points leaving the frame (report them as not visible, do not raise)
  * report a per-point confidence in [0, 1]
  * support re-``init`` at any time (the session re-seeds a camera from the other camera)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackResult:
    """One tracker update."""

    points: np.ndarray                       # (N, 2) float, (x, y) pixels
    visible: np.ndarray                      # (N,) bool
    scores: np.ndarray                       # (N,) float in [0, 1]
    meta: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Camera-level confidence: mean score over visible points (0 when all lost)."""
        vis = np.asarray(self.visible, dtype=bool)
        if not np.any(vis):
            return 0.0
        return float(np.mean(np.asarray(self.scores, dtype=float)[vis]))

    @property
    def n_visible(self) -> int:
        return int(np.count_nonzero(np.asarray(self.visible, dtype=bool)))


class PointTracker(ABC):
    """Tracks a set of 2D points across frames for one camera."""

    name = "abstract"

    def __init__(self, **kwargs):
        self.options = dict(kwargs)
        self.initialised = False
        self.n_points = 0

    @abstractmethod
    def init(self, frame: np.ndarray, points, obj_id: Optional[str] = None) -> None:
        """(Re-)seed the tracker with ``points`` (N, 2) taken from ``frame``."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackResult:
        """Advance to ``frame`` and return the new point estimates."""

    def reset(self) -> None:
        """Forget all state; ``init`` must be called before the next ``update``."""
        self.initialised = False
        self.n_points = 0

    # -- helpers shared by implementations ---------------------------------
    @staticmethod
    def _as_points(points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if pts.shape[0] == 0:
            raise ValueError("PointTracker.init requires at least one point")
        return pts

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            # Luma weights; avoids a hard cv2 dependency in the base class.
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        return arr.astype(np.float32)

    def _empty_result(self, points=None) -> TrackResult:
        n = self.n_points if points is None else len(points)
        pts = np.full((n, 2), np.nan) if points is None else np.asarray(points, dtype=float)
        return TrackResult(points=pts, visible=np.zeros(n, dtype=bool), scores=np.zeros(n))
