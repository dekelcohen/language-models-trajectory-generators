"""Hybrid point tracker: a windowed *anchor* (CoTracker3) bridged by a per-frame *bridge* (KLT).

Why
---
Online CoTracker3 emits a prediction only once per window step (8 frames). In between, the
session has no measurement and the pose filter coasts, so the error grows with the object's
motion (D4 door: up to 0.055 m just before a flush, docs/plans/3d_tracking.md §13.14). A
cheap per-frame tracker fills those gaps:

* **fresh anchor frame** (``stale_frames == 0``): output the anchor's points and re-start
  the bridge on *this* frame at them, so the bridge never accumulates more than ``step - 1``
  frames of drift and the anchor's occlusion memory stays authoritative.
* **stale anchor frame**: output the bridge's points, advanced to the current frame
  (``stale_frames = 0``: a real measurement). A point is visible only if the anchor last saw
  it *and* the bridge's forward-backward check passes on this frame.

The anchor is never re-initialised by the bridge, so CoTracker keeps its memory through
occlusions (``keeps_memory_through_occlusion``). Window flushes are forwarded as
``meta["flushes"]`` / ``_flushes`` so the session's latency model still charges them.

Select with ``--tracker-provider hybrid`` (in-process CoTracker anchor) or
``hybrid_remote`` (anchor served by ``gpu_server``; the KLT bridge runs locally on the CPU).
"""

import logging

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

logger = logging.getLogger(__name__)


class HybridTracker(PointTracker):
    name = "hybrid"
    keeps_memory_through_occlusion = True

    def __init__(self, anchor=None, bridge=None, anchor_kwargs=None, bridge_kwargs=None,
                 motion_model=None, ransac_px=None, **kwargs):
        from providers.trackers.factory import get_tracker

        # Unrecognised kwargs (device, step, url, ...) belong to the anchor.
        anchor_kwargs = dict(anchor_kwargs or {}, **kwargs)
        super().__init__()
        if anchor is None or isinstance(anchor, str):
            anchor = get_tracker(anchor or getattr(config, "tracker_hybrid_anchor", "cotracker"),
                                 **anchor_kwargs)
        if bridge is None or isinstance(bridge, str):
            bridge = get_tracker(bridge or getattr(config, "tracker_hybrid_bridge", "klt"),
                                 **dict(bridge_kwargs or {}))
        self.anchor = anchor
        self.bridge = bridge
        self.motion_model = str(motion_model or getattr(config, "tracker_hybrid_motion", "similarity"))
        self.ransac_px = float(ransac_px if ransac_px is not None
                               else getattr(config, "tracker_hybrid_ransac_px", 2.0))
        self._anchor_pts = None
        self._frame_hw = None
        self.keeps_memory_through_occlusion = bool(
            getattr(anchor, "keeps_memory_through_occlusion", False))
        self._bridge_ok = False
        self._anchor_vis = None
        self._anchor_scores = None
        self._bridged = 0

    # The session's timing reads ``tracker._flushes`` before an update.
    @property
    def _flushes(self):
        return int(getattr(self.anchor, "_flushes", 0) or 0)

    def init(self, frame, points, obj_id=None):
        self.anchor.init(frame, points, obj_id=obj_id)
        self.n_points = self.anchor.n_points
        self.initialised = True
        self._restart_bridge(frame, self._as_points(points), np.ones(self.n_points, dtype=bool),
                             np.ones(self.n_points, dtype=float))

    def reset(self):
        super().reset()
        self.anchor.reset()
        self.bridge.reset()
        self._bridge_ok = False
        self._anchor_vis = None
        self._anchor_scores = None

    def close(self):
        for part in (self.bridge, self.anchor):
            close = getattr(part, "close", None)
            if close is not None:
                close()

    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        a = self.anchor.update(frame)
        meta_a = dict(getattr(a, "meta", None) or {})
        stale = int(meta_a.get("stale_frames", 0) or 0)
        common = {"provider": self.name, "anchor": getattr(self.anchor, "name", "?"),
                  "bridge": getattr(self.bridge, "name", "?"), "anchor_stale": stale}
        for key in ("flushes", "window", "step", "device", "remote_rtt_ms"):
            if key in meta_a:
                common[key] = meta_a[key]

        if stale == 0:
            self._restart_bridge(frame, a.points, a.visible, a.scores)
            self._bridged = 0
            meta = dict(common, stale_frames=0, source="anchor")
            return (TrackResult(points=np.array(a.points, dtype=float, copy=True),
                                              visible=np.array(a.visible, dtype=bool, copy=True),
                                              scores=np.array(a.scores, dtype=float, copy=True),
                                              meta=meta))
        if not self._bridge_ok:
            meta = dict(meta_a, **common, source="anchor_stale")
            return (TrackResult(points=a.points, visible=a.visible,
                                              scores=a.scores, meta=meta))

        b = self.bridge.update(frame)
        self._bridged += 1
        points = np.asarray(b.points, dtype=float)
        visible = np.asarray(b.visible, dtype=bool) & self._anchor_vis
        scores = np.minimum(np.asarray(b.scores, dtype=float), self._anchor_scores)
        extra = {}
        if self.motion_model == "similarity":
            # Per-point KLT slides along edges on weak texture (aperture problem) and the
            # forward-backward check does not catch it; the 12 bridged points then stop
            # being rigid and the pose fit rejects them (D4: fit RMSE 0.02 -> 0.03 m in 3
            # frames). One robust 2D similarity anchor->now moves all anchor points instead.
            moved = self._similarity_bridge(points, visible)
            if moved is not None:
                points, visible, extra = moved
        scores[~visible] = 0.0
        meta = dict(common, stale_frames=0, source="bridge", bridged_frames=self._bridged,
                    bridge_visible=int(np.count_nonzero(b.visible)), **extra)
        return (TrackResult(points=points, visible=visible, scores=scores, meta=meta))

    def _similarity_bridge(self, points, visible):
        """Anchor points moved by the RANSAC similarity fitted to the bridge's tracks."""
        import cv2 as cv

        idx = np.flatnonzero(visible)
        if idx.size < 3:
            return None
        src = self._anchor_pts[idx].astype(np.float32)
        dst = points[idx].astype(np.float32)
        M, inl = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC,
                                           ransacReprojThreshold=self.ransac_px)
        if M is None or inl is None or int(inl.sum()) < 3:
            return None
        moved = self._anchor_pts @ M[:, :2].T + M[:, 2]
        # Every point the anchor last saw follows the object's 2D motion, unless it ran
        # off the image; KLT outliers are thus repaired rather than dropped.
        h, w = self._frame_hw
        inside = (moved[:, 0] >= 0) & (moved[:, 0] <= w - 1) & (moved[:, 1] >= 0) & \
            (moved[:, 1] <= h - 1)
        vis = self._anchor_vis & inside
        return moved, vis, {"bridge_inliers": int(inl.sum()), "bridge_scale": round(
            float(np.hypot(M[0, 0], M[1, 0])), 4)}

    def _restart_bridge(self, frame, points, visible, scores):
        """Re-start the bridge on ``frame`` at ``points`` (one row per seed, kept)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        vis = np.asarray(visible, dtype=bool).reshape(-1) & np.all(np.isfinite(pts), axis=1)
        self._anchor_vis = vis
        self._anchor_scores = np.asarray(scores, dtype=float).reshape(-1)
        self._frame_hw = np.asarray(frame).shape[:2]
        self._bridge_ok = False
        if not np.any(vis):
            return
        # Hidden / NaN rows keep their slot at a finite in-image placeholder; they are masked
        # out by ``_anchor_vis`` until the next fresh anchor frame.
        fill = np.where(np.isfinite(pts), pts, np.nanmean(pts[vis], axis=0))
        self._anchor_pts = fill
        try:
            self.bridge.reset()
            self.bridge.init(frame, fill)
        except Exception as exc:  # noqa: BLE001 - the anchor alone is still a valid tracker
            logger.debug("[hybrid] bridge init failed: %s", exc)
            return
        self._bridge_ok = self.bridge.n_points == len(pts)

