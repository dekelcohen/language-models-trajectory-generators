"""Default point tracker: normalised-cross-correlation template matching per point.

Chosen as the default because it needs only the pinned base ``opencv_python`` already in
``requirements.txt`` (CSRT lives in ``opencv-contrib-python``, which conflicts with that
pin - see ``providers/trackers/csrt_tracker.py``). For the short, smooth, high-frame-rate
motion of a manipulation rollout, a local NCC search is accurate and costs microseconds
per point.

Each point owns a small template patch. Every frame the patch is matched inside a search
window around the previous position; the NCC peak becomes the new position and doubles as
the confidence. Templates are slowly blended towards the current appearance so gradual
scale/lighting change does not kill the track, while a hard confidence floor stops the
template from drifting onto whatever occluded the object.
"""

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

try:
    import cv2 as cv
except ImportError:  # pragma: no cover - cv2 is a hard requirement of the repo
    cv = None


class TemplateTracker(PointTracker):
    name = "template"

    def __init__(self, patch_half=None, search_scale=None, conf_min=None,
                 template_lr=0.1, **kwargs):
        super().__init__(**kwargs)
        self.patch_half = int(patch_half if patch_half is not None else config.track_patch_half)
        self.search_scale = float(search_scale if search_scale is not None else config.track_search_scale)
        self.conf_min = float(conf_min if conf_min is not None else config.track_point_conf_min)
        self.template_lr = float(template_lr)
        self._templates = []
        self._points = None
        self._alive = None

    # -- lifecycle ---------------------------------------------------------
    def init(self, frame, points, obj_id=None):
        if cv is None:
            raise RuntimeError("opencv (cv2) is required by TemplateTracker")
        gray = self._to_gray(frame)
        pts = self._as_points(points)
        self._templates = []
        keep_pts = []
        for pt in pts:
            patch = self._crop(gray, pt, self.patch_half)
            if patch is None:
                continue
            self._templates.append(patch)
            keep_pts.append(pt)
        if not keep_pts:
            raise ValueError("TemplateTracker.init: no point had a full patch inside the image")
        self._points = np.asarray(keep_pts, dtype=float)
        self._alive = np.ones(len(keep_pts), dtype=bool)
        self.n_points = len(keep_pts)
        self.initialised = True

    def reset(self):
        super().reset()
        self._templates = []
        self._points = None
        self._alive = None

    # -- per-frame ---------------------------------------------------------
    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        gray = self._to_gray(frame)
        h, w = gray.shape[:2]
        out_pts = np.array(self._points, dtype=float, copy=True)
        scores = np.zeros(self.n_points, dtype=float)
        visible = np.zeros(self.n_points, dtype=bool)

        search_half = int(round(self.patch_half * self.search_scale))
        for i in range(self.n_points):
            if not self._alive[i]:
                continue
            prev = self._points[i]
            tmpl = self._templates[i]
            th, tw = tmpl.shape[:2]
            x0 = int(round(prev[0] - search_half))
            y0 = int(round(prev[1] - search_half))
            x1 = int(round(prev[0] + search_half)) + 1
            y1 = int(round(prev[1] + search_half)) + 1
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if (x1 - x0) < tw or (y1 - y0) < th:
                self._alive[i] = False
                continue
            window = gray[y0:y1, x0:x1]
            response = cv.matchTemplate(window, tmpl, cv.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv.minMaxLoc(response)
            score = float(max(0.0, maxv))
            # maxl is the top-left of the best match inside the window; the tracked point
            # is the patch centre.
            new_pt = np.array([x0 + maxl[0] + tw / 2.0 - 0.5,
                               y0 + maxl[1] + th / 2.0 - 0.5], dtype=float)
            scores[i] = score
            if score < self.conf_min:
                self._alive[i] = False
                out_pts[i] = new_pt
                continue
            out_pts[i] = new_pt
            visible[i] = True
            fresh = self._crop(gray, new_pt, self.patch_half)
            if fresh is not None and fresh.shape == tmpl.shape:
                self._templates[i] = ((1.0 - self.template_lr) * tmpl
                                      + self.template_lr * fresh).astype(np.float32)

        self._points = out_pts
        return TrackResult(points=out_pts, visible=visible, scores=scores,
                           meta={"provider": self.name, "alive": int(np.count_nonzero(self._alive))})

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _crop(gray, point, half):
        h, w = gray.shape[:2]
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        x0, y0, x1, y1 = x - half, y - half, x + half + 1, y + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0 or float(np.std(patch)) < 1e-3:
            # A featureless patch matches everywhere; refuse to track it.
            return None
        return np.ascontiguousarray(patch, dtype=np.float32)
