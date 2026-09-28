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
                        engage_dist=None, grace_s=None, unknown_after_s=None):
    """Object must stay near the gripper once it has ever been grasped.

    The invariant only *arms* itself after the object has come within ``engage_dist`` of
    the gripper (default: ``max_dist``), i.e. after the grasp actually happened - before
    that the object is legitimately far away while the arm is still approaching.

    Once armed it keeps a three-way verdict, ``attach_state`` in the result:

    * ``attached`` - the last trusted evidence put the object in the hand;
    * ``detached`` - trusted evidence put it further than ``max_dist`` (+ 2x the fit RMSE)
      for ``grace_frames`` evidence frames *and* ``grace_s`` seconds -> ``status``;
    * ``unknown``  - no trusted evidence for ``unknown_after_s``. Not a failure: tracking
      that is lost or lagging is not a drop.

    Evidence is only (a) a *trusted* measured pose - never a Kalman/FK prediction and never
    a tentative re-acquisition, which may be a wrong lock - and (b) the session's depth test
    of "the object is where the gripper holds it" (``lift_meta['fk_check']``): when the
    cameras' depth confirms the object at the gripper-FK pose, it is attached whatever a
    stale 2D track says. ``grace_frames`` given without ``grace_s`` keeps the historical
    frame-count-only behaviour.
    """
    max_dist = config.track_attach_max_dist if max_dist is None else float(max_dist)
    explicit_frames = grace_frames is not None
    grace = config.track_attach_grace_frames if grace_frames is None else int(grace_frames)
    if grace_s is None:
        grace_s = 0.0 if explicit_frames else config.track_attach_grace_s
    grace_s = float(grace_s)
    unknown_s = config.track_attach_unknown_s if unknown_after_s is None else float(unknown_after_s)
    engage = max_dist if engage_dist is None else float(engage_dist)
    period = 1.0 / float(config.track_kf_ref_hz)
    state = {"armed": False, "bad": 0, "bad_since": None, "min_dist": None, "calls": 0,
             "last_t": None, "last_evidence": None, "verdict": "unarmed"}

    def clock(frame):
        state["calls"] += 1
        t = getattr(frame, "sim_time", None)
        return float(t) if t is not None else state["calls"] * period

    def verdict(t):
        if state["armed"] and state["verdict"] != "detached" and \
                state["last_evidence"] is not None and t - state["last_evidence"] > unknown_s:
            return "unknown"
        return state["verdict"]

    def monitor(frame):
        t = clock(frame)
        step = period if state["last_t"] is None else max(t - state["last_t"], 0.0)
        state["last_t"] = t
        obj = frame.get(name)
        pose = getattr(obj, "pose", None) if obj is not None else None
        fk = ((getattr(obj, "lift_meta", None) or {}).get("fk_check")
              if obj is not None else None)
        fk_support = fk_contra = False
        if fk:
            checkable = fk["support"] + fk["contra"]
            fk_support = fk["support"] >= max(config.track_pose_min_points,
                                              config.track_attach_fk_support_frac * checkable)
            fk_contra = fk["contra"] >= config.track_pose_min_points and \
                fk["contra"] > fk["support"]
        if state["armed"] and fk_support:
            state.update(bad=0, bad_since=None, last_evidence=t, verdict="attached")
            return {"status": STATUS_OK, "attach_state": "attached", "fk_supported": True}
        if obj is not None and getattr(obj, "predicted", False):
            # A Kalman coast or gripper-FK follow is not an observation. FK in particular
            # places the object *in the gripper by construction* - counting it as evidence
            # of attachment would hide exactly the drop this monitor exists to catch. The
            # counter is neither reset nor advanced; the next measured frame decides.
            return {"status": STATUS_OK, "predicted": True, "attach_state": verdict(t)}
        if pose is not None and getattr(pose, "tentative", False):
            return {"status": STATUS_OK, "tentative": True, "attach_state": verdict(t)}
        dist = frame.distance(name)
        if dist is None:
            # Handled by object_not_lost; missing measurement is not evidence of a drop.
            return {"status": STATUS_OK, "attach_state": verdict(t)}
        state["min_dist"] = dist if state["min_dist"] is None else min(state["min_dist"], dist)
        if not state["armed"]:
            if dist <= engage:
                state.update(armed=True, last_evidence=t, verdict="attached")
            return {"status": STATUS_OK, "distance": round(dist, 4), "armed": state["armed"],
                    "attach_state": state["verdict"]}
        rmse = getattr(pose, "rmse_m", None) if pose is not None else None
        limit = max_dist + 2.0 * float(rmse or 0.0)
        state["last_evidence"] = t
        if dist > limit:
            state["bad"] += 1
            if state["bad_since"] is None:
                state["bad_since"] = t - step       # the first bad frame covers its period
            lasted = t - state["bad_since"]
            why = " (depth confirms free space at the grasp pose)" if fk_contra else ""
            if state["bad"] >= grace and lasted >= grace_s - 1e-6:
                state["verdict"] = "detached"
                return {"status": status,
                        "reason": (f"'{name}' is {dist:.3f} m from the gripper "
                                   f"(limit {limit:.3f} m) for {state['bad']} frames / "
                                   f"{lasted:.2f} s{why} - it is no longer attached"),
                        "distance": round(dist, 4), "attach_state": "detached"}
            return {"status": STATUS_WARN, "reason": f"'{name}' drifting: {dist:.3f} m{why}",
                    "distance": round(dist, 4), "attach_state": verdict(t)}
        state.update(bad=0, bad_since=None, verdict="attached")
        return {"status": STATUS_OK, "distance": round(dist, 4), "attach_state": "attached"}

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
        pose_mode = getattr(getattr(obj, "pose", None), "mode", None)
        if getattr(obj, "predicted", False) and pose_mode == "coast":
            # Between two windowed-tracker flushes: no news, not bad news.
            return {"status": STATUS_OK}
        # A red/black Kalman (or FK) prediction keeps a position but nobody *sees* it.
        unseen = obj.lost or getattr(obj, "predicted", False)
        if not unseen:
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
