"""Real-time pipeline model in *sim* time: camera clock, tracker latency, drop/queue policy.

The simulator can pause the world while the tracker computes; the real world cannot. This
module makes the tracker pay for its compute in sim time, so a simulated rollout sees what
a real robot would see:

* **Camera clock.** A frame exists every ``1 / camera_fps`` sim seconds, independent of what
  the robot is doing and of the VLM keyframes.
* **Latency.** Processing a frame taken at ``t`` costs ``L`` (measured wall time, a device
  profile, or zero); its result becomes *available* at ``t + L``. Until then consumers
  (monitors, the ``track_objects`` report, the eval scorer) see the previous result.
* **Policy.** ``drop``: a per-frame tracker (template, KLT) that is still busy when a frame
  arrives simply misses it - it takes the next frame once it is free, like a real
  latest-frame loop. ``queue``: a sliding-window tracker (CoTracker) must see every frame, so
  frames queue; results complete in order, and if the device is too slow the queue (and
  the age of the result) grows - which is exactly what would happen on that device.

Rendering is *not* charged: a real camera exposes the next frame while the tracker works.
The simulator's render cost is a wall-clock cost of simulating, reported separately.

Deterministic by construction when the latency is ``zero`` or a ``fixed_*`` profile.
``zero`` + ``camera_fps`` = the scene's motion rate reproduces the historical lock-step
harness exactly.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import config

_EPS = 1e-9


class LatencyModel:
    """Turns one frame's measured compute into the sim seconds it costs."""

    def __init__(self, mode=None, scale=1.0):
        self.mode = str(mode or config.track_latency_mode)
        self.scale = float(scale)
        self.profile = None
        if self.mode not in ("measured", "zero"):
            profiles = getattr(config, "track_latency_profiles", {})
            if self.mode not in profiles:
                raise ValueError(f"unknown latency mode/profile {self.mode!r}; expected "
                                 f"'measured', 'zero' or one of {sorted(profiles)}")
            self.profile = dict(profiles[self.mode])

    @property
    def deterministic(self):
        return self.mode == "zero" or (self.profile is not None and "fixed_s" in self.profile)

    def cost(self, wall_s, timing=None):
        timing = timing or {}
        if self.mode == "zero":
            return 0.0
        if self.profile is None:
            return max(0.0, float(wall_s)) * self.scale
        if "fixed_s" in self.profile:
            return float(self.profile["fixed_s"])
        # Keep the CPU part measured, swap each CoTracker window flush for the device's cost.
        flush_s = float(timing.get("flush_s", 0.0))
        base = max(0.0, float(wall_s) - flush_s) * self.scale
        return base + int(timing.get("flushes", 0)) * float(self.profile.get("cotracker_flush_s", 0.0))

    def describe(self):
        return {"mode": self.mode, "scale": self.scale, "profile": self.profile}


@dataclass
class PipelineResult:
    """One processed frame travelling through the pipeline."""

    frame_idx: int                 # camera-clock index of the frame it was computed on
    frame_t: float                 # sim time the frame was exposed
    start_t: float                 # sim time processing started (after any queue wait)
    available_t: float             # sim time the result becomes usable
    cost_s: float
    wall_s: float
    payload: Any = None            # the TrackFrameReport
    views: Any = None              # the views it was computed on (for scoring / overlays)
    extra: dict = field(default_factory=dict)

    @property
    def queue_wait_s(self):
        return self.start_t - self.frame_t


class RealtimePipeline:
    """The event-driven model. The caller owns the sim clock and calls :meth:`poll` often.

    Typical use (see :meth:`tracking.session.TrackingSession.tick`)::

        pipe.publish(now, commit)             # results whose time has come
        frame = pipe.next_frame(now)          # camera frame exposed at/before now, or None
        if frame is not None and pipe.accepts(frame_t):
            ...compute...; pipe.submit(frame_idx, frame_t, wall_s, timing, payload, views)
            pipe.publish(now, commit)
    """

    def __init__(self, camera_fps=None, latency=None, policy="drop", min_period_s=None):
        fps = float(config.track_camera_fps if camera_fps is None else camera_fps)
        if fps <= 0:
            raise ValueError("RealtimePipeline needs camera_fps > 0 (0 = legacy keyframe cadence)")
        self.camera_fps = fps
        self.frame_dt = 1.0 / fps
        self.latency = latency if isinstance(latency, LatencyModel) else LatencyModel(latency)
        if policy not in ("drop", "queue"):
            raise ValueError(f"policy must be 'drop' or 'queue', not {policy!r}")
        self.policy = policy
        self.min_period_s = float(config.track_period_s if min_period_s is None else min_period_s)
        self.reset()

    def reset(self):
        self._t0 = None
        self._next_idx = 0
        self.busy_until = 0.0
        self._last_accepted_t = -math.inf
        self.pending = deque()
        self.published: Optional[PipelineResult] = None
        self.stats = {"frames_exposed": 0, "frames_missed": 0, "frames_dropped_busy": 0,
                      "frames_dropped_period": 0, "frames_processed": 0, "published": 0,
                      "max_queue_wait_s": 0.0, "max_age_s": 0.0}

    # -- camera clock -------------------------------------------------------
    def frame_time(self, idx):
        return (self._t0 or 0.0) + idx * self.frame_dt

    def next_frame(self, now):
        """Consume the camera frames exposed up to ``now``; return ``(idx, t)`` of the newest.

        The clock starts at the first call (a session may start mid-episode). Only the
        newest frame can still be captured (the past cannot be rendered); older ones the
        caller stepped over count as ``frames_missed`` - poll at least once per frame period
        to avoid them.
        """
        if self._t0 is None:
            self._t0 = float(now)
        newest = None
        while self.frame_time(self._next_idx) <= now + _EPS:
            if newest is not None:
                self.stats["frames_missed"] += 1
            newest = (self._next_idx, self.frame_time(self._next_idx))
            self._next_idx += 1
            self.stats["frames_exposed"] += 1
        return newest

    def accepts(self, frame_t):
        if frame_t - self._last_accepted_t < self.min_period_s - _EPS:
            self.stats["frames_dropped_period"] += 1
            return False
        if self.policy == "drop" and frame_t < self.busy_until - _EPS:
            self.stats["frames_dropped_busy"] += 1
            return False
        return True

    # -- processing -----------------------------------------------------------
    def submit(self, frame_idx, frame_t, wall_s, timing=None, payload=None, views=None):
        cost = self.latency.cost(wall_s, timing)
        start = max(frame_t, self.busy_until) if self.policy == "queue" else frame_t
        done = start + cost
        self.busy_until = done
        self._last_accepted_t = frame_t
        result = PipelineResult(frame_idx=int(frame_idx), frame_t=float(frame_t),
                                start_t=float(start), available_t=float(done),
                                cost_s=float(cost), wall_s=float(wall_s),
                                payload=payload, views=views, extra=dict(timing or {}))
        self.pending.append(result)
        self.stats["frames_processed"] += 1
        self.stats["max_queue_wait_s"] = max(self.stats["max_queue_wait_s"], result.queue_wait_s)
        return result

    def publish(self, now, commit=None):
        """Release every result available by ``now`` (in order); return the newly published."""
        out = []
        while self.pending and self.pending[0].available_t <= now + _EPS:
            result = self.pending.popleft()
            self.published = result
            self.stats["published"] += 1
            if commit is not None:
                commit(result)
            out.append(result)
        return out

    def age(self, now):
        """How old the newest *usable* result is (``None`` before the first one)."""
        if self.published is None:
            return None
        age = float(now - self.published.frame_t)
        self.stats["max_age_s"] = max(self.stats["max_age_s"], age)
        return age

    def describe(self):
        return {"camera_fps": self.camera_fps, "policy": self.policy,
                "min_period_s": self.min_period_s, "latency": self.latency.describe(),
                "stats": dict(self.stats)}


def policy_for(provider):
    """``queue`` for sliding-window trackers that must see every frame, else ``drop``."""
    return "queue" if str(provider).lower() in tuple(config.track_buffering_providers) else "drop"
