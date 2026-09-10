"""Ready-made rollout invariants.

Each factory returns a ``monitor(state) -> dict`` closure, where ``state`` is a
:class:`tracking.types.TrackFrameReport`. They are usable three ways:

  * directly from LLM code - ``track_objects(..., monitor=attached_to_gripper("mug"))``
  * as worked examples the model imitates when it needs something bespoke
  * as the fallback when the agent passes ``monitor=None`` (:func:`default_monitor`)

All of them are stateful closures (they count consecutive violating frames) so a single
noisy frame - a tracker blip, one occluded keyframe - never aborts a rollout.
"""

import numpy as np

import config
from tracking.types import STATUS_ABORT, STATUS_OK, STATUS_RECORD, STATUS_SEVERITY, STATUS_WARN

__all__ = [
    "attached_to_gripper",
    "object_not_lost",
    "stays_within",
    "moved_at_least",
    "combine",
    "default_monitor",
]


def attached_to_gripper(name, max_dist=None, grace_frames=None, status=STATUS_ABORT,
                        engage_dist=None):
    """Object must stay near the gripper once it has ever been grasped.

    The invariant only *arms* itself after the object has come within ``engage_dist`` of
    the gripper (default: ``max_dist``), i.e. after the grasp actually happened - before
    that the object is legitimately far away while the arm is still approaching.
    Once armed, exceeding ``max_dist`` for ``grace_frames`` consecutive frames means the
    object was dropped or never picked up.
    """
    max_dist = config.track_attach_max_dist if max_dist is None else float(max_dist)
    grace = config.track_attach_grace_frames if grace_frames is None else int(grace_frames)
    engage = max_dist if engage_dist is None else float(engage_dist)
    state = {"armed": False, "bad": 0, "min_dist": None}

    def monitor(frame):
        dist = frame.distance(name)
        if dist is None:
            # Handled by object_not_lost; missing measurement is not evidence of a drop.
            return {"status": STATUS_OK}
        state["min_dist"] = dist if state["min_dist"] is None else min(state["min_dist"], dist)
        if not state["armed"]:
            if dist <= engage:
                state["armed"] = True
            return {"status": STATUS_OK, "distance": round(dist, 4), "armed": state["armed"]}
        if dist > max_dist:
            state["bad"] += 1
            if state["bad"] >= grace:
                return {"status": status,
                        "reason": (f"'{name}' is {dist:.3f} m from the gripper "
                                   f"(limit {max_dist:.3f} m) for {state['bad']} frames - "
                                   "it is no longer attached"),
                        "distance": round(dist, 4)}
            return {"status": STATUS_WARN, "reason": f"'{name}' drifting: {dist:.3f} m",
                    "distance": round(dist, 4)}
        state["bad"] = 0
        return {"status": STATUS_OK, "distance": round(dist, 4)}

    monitor.__name__ = f"attached_to_gripper[{name}]"
    return monitor


def object_not_lost(name, patience=None, status=STATUS_RECORD):
    """Escalate when no camera can localise the object for ``patience`` frames.

    Defaults to ``record`` rather than ``abort``: losing sight of an object is often a
    perception problem, not a manipulation failure, so the frames are flagged for the
    reviewer instead of killing the sub-task.
    """
    patience = config.track_lost_patience if patience is None else int(patience)
    state = {"bad": 0, "ever_seen": False}

    def monitor(frame):
        obj = frame.get(name)
        if obj is None:
            return {"status": STATUS_OK}
        if not obj.lost:
            state["ever_seen"] = True
            state["bad"] = 0
            if obj.disagreement is not None and obj.disagreement > config.track_disagree_m:
                return {"status": STATUS_WARN,
                        "reason": (f"cameras disagree on '{name}' by "
                                   f"{obj.disagreement:.3f} m"),
                        "disagreement": round(float(obj.disagreement), 4)}
            return {"status": STATUS_OK}
        if not state["ever_seen"]:
            return {"status": STATUS_OK}
        state["bad"] += 1
        if state["bad"] >= patience:
            return {"status": status,
                    "reason": f"'{name}' not visible in any camera for {state['bad']} frames",
                    "frames_lost": state["bad"]}
        return {"status": STATUS_WARN, "reason": f"'{name}' temporarily lost"}

    monitor.__name__ = f"object_not_lost[{name}]"
    return monitor


def stays_within(name, box, status=STATUS_ABORT, grace_frames=2):
    """Object must stay inside an axis-aligned world box ``((xmin,ymin,zmin),(xmax,ymax,zmax))``."""
    lo = np.asarray(box[0], dtype=float)
    hi = np.asarray(box[1], dtype=float)
    state = {"bad": 0}

    def monitor(frame):
        pos = frame.world_point(name)
        if pos is None:
            return {"status": STATUS_OK}
        if np.all(pos >= lo) and np.all(pos <= hi):
            state["bad"] = 0
            return {"status": STATUS_OK}
        state["bad"] += 1
        if state["bad"] >= grace_frames:
            return {"status": status,
                    "reason": (f"'{name}' left the allowed region at "
                               f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"),
                    "position": [round(float(v), 4) for v in pos]}
        return {"status": STATUS_WARN, "reason": f"'{name}' near the region boundary"}

    monitor.__name__ = f"stays_within[{name}]"
    return monitor


def moved_at_least(name, dist, after_frames=10, status=STATUS_RECORD):
    """Progress check: the object should have moved ``dist`` metres by ``after_frames``.

    Useful for "open the door" style sub-tasks, where the gripper can be perfectly placed
    yet nothing actually moves.
    """
    dist = float(dist)
    state = {"origin": None, "max": 0.0}

    def monitor(frame):
        pos = frame.world_point(name)
        if pos is None:
            return {"status": STATUS_OK}
        if state["origin"] is None:
            state["origin"] = np.array(pos, dtype=float)
            return {"status": STATUS_OK}
        moved = float(np.linalg.norm(pos - state["origin"]))
        state["max"] = max(state["max"], moved)
        if frame.frame_idx >= after_frames and state["max"] < dist:
            return {"status": status,
                    "reason": (f"'{name}' moved only {state['max']:.3f} m after "
                               f"{frame.frame_idx} frames (expected {dist:.3f} m)"),
                    "moved": round(state["max"], 4)}
        return {"status": STATUS_OK, "moved": round(moved, 4)}

    monitor.__name__ = f"moved_at_least[{name}]"
    return monitor


def combine(*monitors):
    """Run several monitors; the worst status wins (abort > record > warn > ok)."""
    monitors = [m for m in monitors if m is not None]

    def monitor(frame):
        from tracking.monitor import normalise  # local import avoids a circular import

        worst = None
        reasons = []
        for m in monitors:
            res = normalise(m(frame))
            if res.status != STATUS_OK and res.reason:
                reasons.append(f"{getattr(m, '__name__', 'monitor')}: {res.reason}")
            if worst is None or STATUS_SEVERITY.get(res.status, 0) > STATUS_SEVERITY.get(worst.status, 0):
                worst = res
        if worst is None:
            return {"status": STATUS_OK}
        return {"status": worst.status, "reason": "; ".join(reasons) or worst.reason}

    monitor.__name__ = "combine"
    return monitor


def default_monitor(names=None, max_dist=None):
    """The invariant used when the agent supplies no monitor.

    ``attached_to_gripper`` (abort on a dropped object) combined with ``object_not_lost``
    (flag perception dropouts for the reviewer), applied to every tracked target. When
    ``names`` is ``None`` the targets are discovered from the first frame.
    """
    per_name = {}

    def monitor(frame):
        from tracking.monitor import normalise

        targets = names if names is not None else list(frame.objects.keys())
        worst, reasons = None, []
        for name in targets:
            if name not in per_name:
                per_name[name] = combine(attached_to_gripper(name, max_dist=max_dist),
                                         object_not_lost(name))
            res = normalise(per_name[name](frame))
            if res.status != STATUS_OK and res.reason:
                reasons.append(res.reason)
            if worst is None or STATUS_SEVERITY.get(res.status, 0) > STATUS_SEVERITY.get(worst.status, 0):
                worst = res
        if worst is None:
            return {"status": STATUS_OK}
        return {"status": worst.status, "reason": "; ".join(reasons) or worst.reason}

    monitor.__name__ = "default_monitor"
    return monitor
