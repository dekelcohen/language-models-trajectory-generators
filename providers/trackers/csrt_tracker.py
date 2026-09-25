"""CSRT point tracker (opt-in).

OpenCV's CSRT is a *bbox* tracker, so a set of N points is followed with N independent
CSRT instances, each initialised on a small square box centred on its point; the tracked
point is the box centre. That is more expensive than
:class:`providers.trackers.template_tracker.TemplateTracker` but markedly more robust to
rotation and partial occlusion.

Not the default: ``cv2.TrackerCSRT`` ships in ``opencv-contrib-python``, which cannot be
installed alongside the ``opencv_python==4.8.1.78`` pin in ``requirements.txt``. Select it
with ``--tracker-provider csrt`` after replacing that pin with
``opencv-contrib-python==4.8.1.78`` (a strict superset of the base package).
"""

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

try:
    import cv2 as cv
except ImportError:  # pragma: no cover
    cv = None

INSTALL_HINT = (
    "CSRT requires opencv-contrib-python. Replace 'opencv_python==4.8.1.78' with "
    "'opencv-contrib-python==4.8.1.78' in requirements.txt (superset of the base "
    "package) and reinstall, or run with --tracker-provider template."
)


def _csrt_factory():
    """Return a zero-arg CSRT constructor across OpenCV versions, or ``None``."""
    if cv is None:
        return None
    for holder, attr in ((cv, "TrackerCSRT_create"), (cv, "TrackerCSRT"),
                         (getattr(cv, "legacy", None), "TrackerCSRT_create"),
                         (getattr(cv, "legacy", None), "TrackerCSRT")):
        if holder is None:
            continue
        fn = getattr(holder, attr, None)
        if fn is None:
            continue
        create = getattr(fn, "create", None)
        return create if callable(create) else fn
    return None


def csrt_available() -> bool:
    return _csrt_factory() is not None


class CSRTTracker(PointTracker):
    name = "csrt"

    def __init__(self, patch_half=None, conf_min=None, **kwargs):
        super().__init__(**kwargs)
        factory = _csrt_factory()
        if factory is None:
            raise RuntimeError(INSTALL_HINT)
        self._factory = factory
        self.patch_half = int(patch_half if patch_half is not None else config.track_patch_half)
        self.conf_min = float(conf_min if conf_min is not None else config.track_point_conf_min)
        self._trackers = []
        self._points = None
        self._alive = None
        self._miss = None

    def init(self, frame, points, obj_id=None):
        bgr = self._to_bgr(frame)
        h, w = bgr.shape[:2]
        pts = self._as_points(points)
        # One slot per requested point; a box that does not fit starts (and stays) dead so
        # output row j is still seed point j (see PointTracker).
        self._trackers = []
        size = 2 * self.patch_half + 1
        for pt in pts:
            x = float(pt[0]) - self.patch_half
            y = float(pt[1]) - self.patch_half
            if x < 0 or y < 0 or x + size > w or y + size > h:
                self._trackers.append(None)
                continue
            trk = self._factory()
            trk.init(bgr, (x, y, float(size), float(size)))
            self._trackers.append(trk)
        self._alive = np.array([t is not None for t in self._trackers], dtype=bool)
        if not np.any(self._alive):
            raise ValueError("CSRTTracker.init: no point had a full box inside the image")
        self._points = np.array(pts, dtype=float, copy=True)
        self._miss = np.zeros(len(pts), dtype=int)
        self.n_points = len(pts)
        self.initialised = True

    def reset(self):
        super().reset()
        self._trackers, self._points, self._alive, self._miss = [], None, None, None

    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        bgr = self._to_bgr(frame)
        out = np.array(self._points, dtype=float, copy=True)
        visible = np.zeros(self.n_points, dtype=bool)
        scores = np.zeros(self.n_points, dtype=float)
        for i, trk in enumerate(self._trackers):
            if not self._alive[i]:
                continue
            ok, box = trk.update(bgr)
            if not ok:
                self._miss[i] += 1
                self._alive[i] = self._miss[i] < 2
                continue
            x, y, bw, bh = box
            out[i] = [float(x) + float(bw) / 2.0, float(y) + float(bh) / 2.0]
            visible[i] = True
            # CSRT exposes no score; a successful update is full confidence, decayed by
            # any recent misses so the session's health logic still sees degradation.
            scores[i] = max(0.0, 1.0 - 0.3 * self._miss[i])
            self._miss[i] = 0
        self._points = out
        return TrackResult(points=out, visible=visible, scores=scores,
                           meta={"provider": self.name})

    @staticmethod
    def _to_bgr(frame):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
        return np.ascontiguousarray(arr[..., ::-1])
