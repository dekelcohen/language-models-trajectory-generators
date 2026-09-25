"""Pyramidal Lucas-Kanade point tracker (base opencv, CPU).

Sparse optical flow is the classical answer to "follow these N pixels": each point is moved
by the brightness-constancy solution over a ``win x win`` window, coarse-to-fine through a
``levels``-deep image pyramid so motions larger than the window are still caught. Unlike the
template tracker it has no fixed-size appearance model to go stale, so gradual scale change
and rotation (a wrist camera closing in, a door handle swinging) are followed naturally.

Its classical weakness is drift and silent failure (aperture problem on flat surfaces,
latching onto an occluder). The forward-backward check (Kalal et al. 2010) handles that
per point: track t -> t+1, then t+1 -> t, and trust the point only if it comes back within
``fb_max_px``. A point that fails is reported invisible and retried from its last position
on the next frame; one that leaves the image stays a dead slot (the one-row-per-seed
contract, see :mod:`providers.trackers.base`).
"""

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

try:
    import cv2 as cv
except ImportError:  # pragma: no cover - cv2 is a hard requirement of the repo
    cv = None


class KLTTracker(PointTracker):
    name = "klt"

    def __init__(self, win=None, levels=None, fb_max_px=None, **kwargs):
        super().__init__(**kwargs)
        self.win = int(win if win is not None else config.track_klt_win)
        self.levels = int(levels if levels is not None else config.track_klt_levels)
        self.fb_max_px = float(fb_max_px if fb_max_px is not None else config.track_klt_fb_max_px)
        self._criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01) if cv else None
        self._prev = None
        self._points = None
        self._alive = None

    def init(self, frame, points, obj_id=None):
        if cv is None:
            raise RuntimeError("opencv (cv2) is required by KLTTracker")
        gray = self._gray_u8(frame)
        pts = self._as_points(points)
        h, w = gray.shape[:2]
        self._alive = self._inside(pts, w, h)
        if not np.any(self._alive):
            raise ValueError("KLTTracker.init: no point inside the image")
        self._points = np.array(pts, dtype=float, copy=True)
        self._prev = gray
        self.n_points = len(pts)
        self.initialised = True

    def reset(self):
        super().reset()
        self._prev = None
        self._points = None
        self._alive = None

    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        gray = self._gray_u8(frame)
        h, w = gray.shape[:2]
        n = self.n_points
        visible = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=float)
        idx = np.flatnonzero(self._alive)
        fb = np.full(n, np.inf)
        if idx.size:
            p0 = self._points[idx].astype(np.float32).reshape(-1, 1, 2)
            params = dict(winSize=(self.win, self.win), maxLevel=self.levels,
                          criteria=self._criteria)
            p1, st1, _e1 = cv.calcOpticalFlowPyrLK(self._prev, gray, p0, None, **params)
            p0r, st0, _e0 = cv.calcOpticalFlowPyrLK(gray, self._prev, p1, None, **params)
            p1 = p1.reshape(-1, 2).astype(float)
            ok = (st1.reshape(-1) == 1) & (st0.reshape(-1) == 1)
            err = np.linalg.norm(p0r.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
            fb[idx] = np.where(ok, err, np.inf)
            inside = self._inside(p1, w, h)
            good = ok & (err <= self.fb_max_px) & inside
            gi = idx[good]
            self._points[gi] = p1[good]
            visible[gi] = True
            scores[gi] = np.clip(1.0 - err[good] / (2.0 * self.fb_max_px), 0.0, 1.0)
            # Left the image for good: a dead slot. A failed FB check is retried next frame.
            self._alive[idx[ok & ~inside]] = False
        self._prev = gray
        finite = fb[np.isfinite(fb)]
        return TrackResult(points=np.array(self._points, copy=True), visible=visible,
                           scores=scores,
                           meta={"provider": self.name,
                                 "alive": int(np.count_nonzero(self._alive)),
                                 "fb_median_px": float(np.median(finite)) if finite.size else None})

    @staticmethod
    def _inside(pts, w, h, margin=1.0):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin)
                & (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin))

    def _gray_u8(self, frame):
        return np.clip(self._to_gray(frame), 0, 255).astype(np.uint8)
