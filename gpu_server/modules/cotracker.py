"""CoTracker3 (online) as a gpu_server module.

The weights are loaded once. Each session (one per camera stream) gets its own copy of the
online predictor, because the predictor keeps the per-stream online state, and wraps it in
the repo's :class:`providers.trackers.cotracker_tracker.CoTrackerTracker`. So window, flush,
visibility and staleness semantics are bit-for-bit those of the local ``cotracker``
provider, and the client (``providers/trackers/remote_tracker.py``) is a thin proxy.

Session methods::

    init    frame (H,W,3) uint8, points (N,2) float; args {obj_id?}  -> {n_points}
    update  frame                                                    -> points, visible, scores + {meta}
    reset                                                            -> {}

Config: ``variant``, ``device`` (``cuda:0`` / ``cpu``; default = first GPU), ``torch_home``.
Session params: ``vis_threshold``, ``stale_decay``, ``warm_start``, ``fp16``.
"""

import copy
import logging

import numpy as np

from gpu_server.modules.base import GpuModule, GpuSession, ModuleError, require

logger = logging.getLogger("gpu_server.cotracker")

SESSION_PARAMS = ("vis_threshold", "stale_decay", "warm_start", "fp16")


def encode_result(res):
    """TrackResult -> (reply, arrays) on the wire."""
    meta = dict(res.meta or {})
    return ({"meta": meta, "confidence": float(res.confidence)},
            {"points": np.asarray(res.points, dtype=np.float64),
             "visible": np.asarray(res.visible, dtype=bool),
             "scores": np.asarray(res.scores, dtype=np.float64)})


class CoTrackerSession(GpuSession):
    def __init__(self, tracker):
        self.tracker = tracker

    def m_init(self, args, arrays):
        frame, points = require(arrays, "frame", "points")
        self.tracker.init(frame, points, obj_id=args.get("obj_id"))
        return {"n_points": int(self.tracker.n_points)}, {}

    def m_update(self, args, arrays):
        (frame,) = require(arrays, "frame")
        return encode_result(self.tracker.update(frame))

    def m_reset(self, args, arrays):
        self.tracker.reset()
        return {}, {}

    def close(self):
        self.tracker.reset()
        self.tracker._model = None


class CoTrackerModule(GpuModule):
    name = "cotracker"
    description = "CoTracker3 online point tracking; one session per camera stream"
    stateful = True

    def __init__(self, **config):
        super().__init__(**config)
        self._model = None
        self.device = None

    def load(self):
        from providers.trackers.cotracker_tracker import load_cotracker_model
        self._model, self.device = load_cotracker_model(self.config.get("variant"),
                                                        self.config.get("device"),
                                                        self.config.get("torch_home"))

    def unload(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def info(self):
        out = {"device": str(self.device), "step": getattr(self._model, "step", None)}
        try:
            import torch
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                out["vram_mb"] = round(torch.cuda.memory_allocated() / 2**20)
        except ImportError:  # pragma: no cover
            pass
        return out

    def open_session(self, **params):
        from providers.trackers.cotracker_tracker import CoTrackerTracker
        unknown = sorted(set(params) - set(SESSION_PARAMS))
        if unknown:
            raise ModuleError(f"cotracker: unknown session param(s) {unknown}; allowed {SESSION_PARAMS}")
        if self._model is None:
            raise ModuleError("cotracker: module not loaded")
        # deepcopy = fresh online state, shared nothing; ~100 MB of weights per stream.
        tracker = CoTrackerTracker(model=copy.deepcopy(self._model), device=self.device, **params)
        return CoTrackerSession(tracker)
