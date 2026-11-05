"""Guarded pull of a grasped handle (door / drawer / lid) with real-time correction.

Runs in the agent process with the LLM's own tools only (trajectories, gripper, robot state,
the contact guard, optional camera tracking) - no simulator truth. The hand moves in small
position-controlled steps from the MEASURED pose; the contact guard (``guard_motion(on_trip=
"return")``) stops a step the moment the part stops following the hand, and the pull reacts
without an LLM round trip:

* early trip (< ``early_m`` travelled): the hinge hypothesis is wrong -> mirror the hinge across
  the pull line (or turn a straight pull), return to the start, re-grasp if needed, retry
  (``max_flips``);
* late slip: re-grasp where the camera re-acquires the part after a short settle (if trusted)
  or where the hand last held it, projected onto the arc, then continue (``max_regrasps``);
* late jam: the part reached its end stop -> done.

A straight pull (``hinge=None``) fits a circle to the measured hand path once it has moved
``fit_after_m``, and continues on that arc (a door swings even when pulled straight at first).
Top-down grasps only.
"""

import math

import numpy as np

import config


def _regrasp(api, tip, rotation):
    from touch.recovery import regrasp
    return regrasp(api, tip, approach_yaw=None, rotation=rotation, back_off=0.10, tip_depth=0.01, retreat=0.04)


def _unit(v):
    v = np.asarray(v, float)[:2]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _mirror(c, origin, d):
    """Reflect point ``c`` across the line through ``origin`` along unit ``d`` (all xy)."""
    v = c - origin
    return origin + 2 * float(np.dot(v, d)) * d - v


def _arc_sign(c, p, d):
    """+1 (CCW) / -1 so that moving along the circle about ``c`` from ``p`` goes along ``d``."""
    r = p - c
    return 1.0 if (-r[1] * d[0] + r[0] * d[1]) >= 0 else -1.0


def guarded_pull(api, direction, hinge=None, travel=0.35, max_deg=50.0, step=0.03, follow_yaw=True,
                 max_flips=1, max_regrasps=2, part=None, settle_s=0.5, max_force=None,
                 early_m=0.15, fit_after_m=0.06):
    """Pull the held handle along ``direction`` (world, horizontal part used) or around ``hinge``
    ([x, y] of the hinge axis, the door edge FAR from the handle), guarded against slips and jams.

    Returns ``{"outcome": "opened" | "end_stop" | "blocked" | "slipped" | "not_holding" |
    "unsupported", "travel_m", "swept_deg", "hinge", "flips", "regrasps", "events"}``.
    """
    dz = config.gripper_depth_offset_franka
    ev = []
    res = {"outcome": None, "travel_m": 0.0, "swept_deg": 0.0, "hinge": None, "flips": 0, "regrasps": 0,
           "events": ev}

    def log(msg):
        ev.append(msg)
        print(f"[guarded_pull] {msg}")

    st = api.get_robot_state()
    if st.get("rotation") is None or st.get("eef_pos") is None:
        res["outcome"] = "unsupported"
        log("needs a top-down grasp (hand rotation not reported)")
        return res
    if not st.get("gripper_holding"):
        res["outcome"] = "not_holding"
        log("the gripper is not holding anything - grasp the handle first")
        return res
    d0 = _unit(direction)
    if d0 is None:
        raise ValueError("guarded_pull: direction needs a horizontal component")

    from common_utils import Trajectory
    tip0 = np.asarray(st["eef_pos"], float)
    z, yaw0 = float(tip0[2]), float(st["rotation"])
    o = tip0[:2].copy()
    centre = None if hinge is None else np.asarray(hinge, float)[:2]
    line_d = d0.copy()
    sgn = _arc_sign(centre, o, d0) if centre is not None else 0.0
    # early-trip hypotheses: the hinge on the other side; for a straight pull, an arc either way
    nrm = np.array([-d0[1], d0[0]])
    hyps = [_mirror(centre, o, d0)] if centre is not None else [o + 0.5 * nrm, o - 0.5 * nrm]
    flips_allowed = max_flips if centre is not None else max_flips + 1
    max_turn = math.radians(6.0)                 # wrist turn per step: a sudden twist pries the jaws open

    def arm():
        api.guard_motion("hold", part=part, on_trip="return", settle_s=settle_s, max_force=max_force)

    def move(xy, yaw, desc):
        m = api.execute_trajectory(Trajectory([[float(xy[0]), float(xy[1]), z + dz, float(yaw)]], desc))
        if not m or m.get("eef_pos") is None:
            m = dict(m or {})
            m["eef_pos"] = api.get_robot_state()["eef_pos"]
        return m

    def restart(reason, v):
        """Continue the new hypothesis from where the hand is (the part followed it so far - going
        back would re-close it and wear the grip); re-grasp first if the part left the hand."""
        nonlocal p, yaw
        st_ = api.get_robot_state()
        if st_.get("gripper_holding"):
            return
        res["regrasps"] += 1
        where = np.asarray(v.get("last_held_pos") or v.get("slip_pos") or [o[0], o[1], z], float)
        log(f"re-grasp after the {reason} trip at {np.round(where, 3).tolist()} (last held)")
        _regrasp(api, [float(where[0]), float(where[1]), z], yaw)
        st_ = api.get_robot_state()
        p = np.asarray(st_["eef_pos"], float)[:2]

    arm()
    p, yaw, path, s, stalls = o.copy(), yaw0, [o.copy()], 0.0, 0
    try:
        for _ in range(int(3 * travel / step) + 40):
            swept = 0.0
            if centre is not None:
                r0 = float(np.linalg.norm(o - centre))       # fixed: the measured hand lags the arc
                a = math.atan2(p[1] - centre[1], p[0] - centre[0])
                a0 = math.atan2(o[1] - centre[1], o[0] - centre[0])
                swept = abs(math.degrees(math.remainder(a - a0, 2 * math.pi)))
                an = a + sgn * step / max(r0, 0.05)
                tgt = centre + r0 * np.array([math.cos(an), math.sin(an)])
                want = yaw0 + sgn * math.remainder(an - a0, 2 * math.pi) if follow_yaw else yaw0
                yaw_t = yaw + float(np.clip(math.remainder(want - yaw, 2 * math.pi), -max_turn, max_turn))
            else:
                tgt, yaw_t = p + line_d * step, yaw
            res.update(travel_m=round(s, 3), swept_deg=round(swept, 1),
                       hinge=None if centre is None else [round(float(v), 3) for v in centre])
            if s >= travel or swept >= max_deg:
                res["outcome"] = "opened"
                log(f"opened: {s * 100:.0f} cm, {swept:.0f} deg")
                return res

            m = move(tgt, yaw_t, "guarded_pull: step")
            pn = np.asarray(m["eef_pos"], float)[:2]
            v = m.get("guard")
            moved = float(np.linalg.norm(pn - p))
            if v is None:
                s += moved
                if centre is None and moved > 0.3 * step:        # follow the measured direction
                    line_d = _unit(0.5 * line_d + 0.5 * (pn - p) / moved)
                p, yaw = pn, yaw_t
                path.append(p.copy())
                stalls = stalls + 1 if moved < 0.2 * step else 0
                if stalls >= 3:
                    res["outcome"] = "blocked"
                    log(f"blocked: the hand stopped moving after {s * 100:.0f} cm")
                    return res
                if centre is None and s >= fit_after_m and len(path) >= 5:
                    from touch.pull import fit_circle_xy
                    P = np.asarray(path)
                    chord = _unit(P[-1] - P[0])
                    bow = 0.0 if chord is None else float(np.max(np.abs(
                        (P[:, 0] - P[0, 0]) * chord[1] - (P[:, 1] - P[0, 1]) * chord[0])))
                    fit = fit_circle_xy(P) if bow >= 0.003 else None
                    if fit is not None and 0.15 <= fit[1] <= 1.5 and fit[2] < 0.01:
                        centre = np.asarray(fit[0], float)[:2]
                        sgn = _arc_sign(centre, p, line_d)
                        o = path[0]
                        log(f"measured path is an arc: hinge {np.round(centre, 3).tolist()} r={fit[1]:.2f} m")
                continue

            # guard tripped
            p = pn
            log(f"guard {v.get('check')} after {s * 100:.0f} cm: {v.get('reason')}")
            if v.get("check") == "jam" and s >= early_m:
                res["outcome"] = "end_stop"
                return res
            if s < early_m:
                if not hyps or res["flips"] >= flips_allowed:
                    res["outcome"] = "blocked" if v.get("check") == "jam" else "slipped"
                    log("early trip, no hypotheses left")
                    return res
                res["flips"] += 1
                centre = hyps.pop(0)
                sgn = _arc_sign(centre, o, d0)
                log(f"early trip: wrong path, trying the hinge at {np.round(centre, 3).tolist()}")
                restart(v.get("check"), v)
                stalls = 0
                arm()
                continue
            if res["regrasps"] >= max_regrasps:
                res["outcome"] = "slipped"
                return res
            rq = v.get("reacquired") or {}
            if rq.get("trusted") and rq.get("part_pos") is not None:
                where, src = np.asarray(rq["part_pos"], float), "tracking"
            else:
                where, src = np.asarray(v.get("last_held_pos") or v.get("slip_pos"), float), "last held"
            if centre is not None:
                r0 = float(np.linalg.norm(o - centre))
                where[:2] = centre + r0 * _unit(where[:2] - centre)
            res["regrasps"] += 1
            log(f"re-grasp at {np.round(where, 3).tolist()} ({src})")
            _regrasp(api, [float(where[0]), float(where[1]), z], yaw)
            st = api.get_robot_state()
            if not st.get("gripper_holding"):
                res["outcome"] = "slipped"
                log("re-grasp closed on nothing")
                return res
            p = np.asarray(st["eef_pos"], float)[:2]
            path.append(p.copy())
            arm()
        res["outcome"] = "blocked"
        return res
    finally:
        try:
            api.guard_motion("hold")
        except Exception:
            pass
