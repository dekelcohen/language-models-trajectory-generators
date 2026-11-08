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


def _push_around(api, c, sgn, r_grip, a_grip, a_start, z, rot, max_deg, r_push, r_out, go, hand_deg, pt, log,
                 behind_deg=20.0, ahead_deg=17.0, min_open_deg=11.0, hover=0.12, below=0.15, step_deg=5.0):
    """push_open route "around", the geometry that opened the door:hidden panel to 78 deg (r1,
    2026-11-08, LLM-written): hover ``hover`` m above the grasp, go out past the free edge
    (``r_out``) slightly ahead of the panel, arc round to ``behind_deg`` behind it (but at least
    ``min_open_deg`` open, so the hand never reaches the frame), descend to ``below`` m under the
    grasp height (clear of the lever's back-side boss, not so low the hand meets the frame), come in
    to ``r_push``, then push in ``step_deg`` steps. The hand yaw turns with the arc (as during the
    pull), so the hand keeps its slim side toward the panel."""
    # free edge ~r_grip + 0.12 (handle ~12 cm from it); + ~6 cm for the hand's slim side and the panel's
    # thickness. r_grip + 0.12 sat ON the edge and slammed a 38 deg door shut (harness door_around4)
    r_out = float(r_out if r_out is not None else r_grip + 0.30)
    # aim for the grasp radius; the arm column (link6/7) meets the latch spindle (sticks ~10 cm out of the back
    # face at r ~0.41-0.51, lever height) on the way in, and the push then bears on that spindle
    r_push = float(r_push if r_push is not None else r_grip)
    tip = np.asarray(api.get_robot_state()["eef_pos"], float)
    tip = go(tip[:2], z + hover, "push_open: lift off the handle")
    a_h = math.atan2(tip[1] - c[1], tip[0] - c[0])
    yaw_h = rot[0]

    def turn(a):                                        # same yaw-follows-arc rule as guarded_pull
        rot[0] = yaw_h + sgn * math.remainder(a - a_h, 2 * math.pi)
    a_f = a_h + sgn * math.radians(ahead_deg)
    for r in np.linspace(float(np.linalg.norm(tip[:2] - c)), r_out, 5)[1:]:
        a = a_h + (a_f - a_h) * (r - r_grip) / max(r_out - r_grip, 1e-6)
        turn(a)
        tip = go(pt(r, a), z + hover, "push_open: out past the free edge")
    r_got = float(np.linalg.norm(tip[:2] - c))
    if r_got < r_out - 0.03:
        # arcing round from inside the free-edge radius sweeps the panel shut (harness 2026-11-08: 23 -> 3 deg)
        log(f"push_open(around): reached only r={r_got:.2f} of {r_out:.2f} m (arm reach); not going round")
        return "push_blocked", 0.0
    a_b = a_h - sgn * math.radians(behind_deg)
    a_min = a_start + sgn * math.radians(min_open_deg)
    if sgn * (a_b - a_min) < 0:
        a_b = a_min
    for a in np.linspace(a_f, a_b, 5)[1:]:              # round the free edge, beyond its radius
        turn(a)
        tip = go(pt(r_out, a), z + hover, "push_open: round the free edge")
    tip = go(pt(r_out, a_b), z - below, "push_open: down behind the panel")
    log(f"push_open(around): down at r={float(np.linalg.norm(tip[:2] - c)):.2f} hand angle {hand_deg(tip[:2]):.0f} "
        f"deg, z={tip[2]:.3f}")
    if tip[2] > z - below + 0.03:
        # landed on the panel / lever: moving in from here pushes the door SHUT (harness 2026-11-08: 19.5 -> 0 deg)
        log(f"push_open(around): descent stopped at z={tip[2]:.3f} (wanted {z - below:.3f}): something is "
            f"under the hand; not pushing")
        go(tip[:2], z + hover, "push_open: rise clear")
        return "push_blocked", 0.0
    # come in radially in small steps: one long move let the arm swing through the wedge and slam the door
    # shut (harness door_around14: 32 -> 4 deg) or flung it open (door_around11) depending on the IK path
    n_in = max(2, int(math.ceil((float(np.linalg.norm(tip[:2] - c)) - r_push) / 0.03)))
    for r in np.linspace(float(np.linalg.norm(tip[:2] - c)), r_push, n_in + 1)[1:]:
        tip = go(pt(r, a_b), z - below, "push_open: in behind the panel")
        if float(np.linalg.norm(tip[:2] - pt(r, a_b))) > 0.05:
            break
    got = float(np.linalg.norm(tip[:2] - pt(r_push, a_b)))
    if got > 0.05:
        r_here = float(np.linalg.norm(tip[:2] - c))
        if r_here > r_grip + 0.17:
            log(f"push_open(around): blocked {got * 100:.0f} cm short of the spot behind the panel")
            return "push_blocked", 0.0
        # the wrist met the back face on the way in (harness door_around5: 35 -> 55 deg on this move);
        # the hand IS behind the panel, inside its free-edge radius: push from the radius it reached
        log(f"push_open(around): stopped {got * 100:.0f} cm short, at r={r_here:.2f} m behind the panel; "
            f"pushing from there")
        r_push = r_here
    log(f"push_open(around): behind the panel at r={r_push:.2f} m, hand angle {hand_deg(tip[:2]):.0f} deg; "
        f"pushing to {max_deg:.0f} deg")
    a, stalls, prev = a_b, 0, tip[:2]
    a_end = a_start + sgn * math.radians(max_deg)
    while sgn * math.remainder(a_end - a, 2 * math.pi) > 1e-3:
        a += sgn * min(math.radians(step_deg), abs(math.remainder(a_end - a, 2 * math.pi)))
        turn(a)
        cur = go(pt(r_push, a), z - below, "push_open: push")[:2]
        a_prev = math.atan2(prev[1] - c[1], prev[0] - c[0])
        a_cur = math.atan2(cur[1] - c[1], cur[0] - c[0])
        prev = cur
        stalls = stalls + 1 if abs(math.remainder(a_cur - a_prev, 2 * math.pi)) < math.radians(2) else 0
        if stalls >= 2:
            break
    deg = hand_deg(prev)
    out = "pushed" if deg >= max_deg - 5 else "push_stalled"
    log(f"push_open(around): {out}; hand at {deg:.0f} deg about the hinge (the door is at least that open)")
    here = np.asarray(api.get_robot_state()["eef_pos"], float)
    # Retreat without dragging the door: the stalled push presses the arm on the latch spindle (~2 kN, lateral
    # friction 2), so any horizontal slide drags the door shut (harness door_around6-10: 66 -> 28..57 deg).
    # 1) drop straight down 10 cm so the wrist clears the spindle (friction is then vertical: no torque about
    # the hinge), 2) slide RADIALLY out past the free edge at that height, 3) rise outside the panel.
    # 0) unload first: step 2 cm back along the arc (= along the contact normal, so friction cannot drag the
    # panel) at the measured height; the drop under full push load still cost 9 deg (harness door_around29)
    r_now = float(np.linalg.norm(here[:2] - c))
    a_off = math.atan2(here[1] - c[1], here[0] - c[0]) - sgn * 0.02 / max(r_now, 0.1)
    here = go(pt(r_now, a_off), float(here[2]), "push_open: unload")
    z_low = float(here[2]) - 0.10
    tip = go(here[:2], z_low, "push_open: drop below the latch")
    log(f"push_open(around): retreat: dropped to z={tip[2]:.3f} (wanted {z_low:.3f})")
    r_now = float(np.linalg.norm(tip[:2] - c))
    a_cur = math.atan2(tip[1] - c[1], tip[0] - c[0])
    for r in np.linspace(r_now, r_out + 0.04, 4)[1:]:
        go(pt(r, a_cur), z_low, "push_open: slide out past the free edge")
    go(pt(r_out + 0.04, a_cur), z + hover + 0.10, "push_open: rise clear")
    return out, round(deg, 1)


def push_open(api, hinge, sgn, r_grip, a_grip, a_start, z, yaw, max_deg=80.0, step=0.03,
              behind=0.25, r_push=None, lift=0.14, log=print, route="over", r_out=None):
    """Push a partly opened hinged panel further open from its BACK face.

    A pulled door swings toward the arm; past ~20-30 deg the panel meets the robot's own forearm
    and pries the grasp open, so the rest of the arc is pushed instead. Release, lift ``lift`` m
    (over the handle and, for a low panel, over its top), move over the open wedge to ``behind`` m
    (arc length at ``r_push``) behind the handle's current angle ``a_grip``, i.e. behind the panel's
    back face, and come down. If the descent stalls high
    (landed on the panel top), shift 5 deg further back and retry. Then sweep the circle about
    ``hinge`` in the opening direction ``sgn`` until the hand angle reaches ``max_deg`` from
    ``a_start`` (closed grasp angle) or the hand stalls (door end stop / reach). The panel is ahead
    of the hand, so the returned hand angle is a LOWER bound on the door angle. Agent-side inputs only.
    ``route="around"``: instead of over the top, back out radially past the free edge (to ``r_out``),
    swing round it at handle height and come in behind the back face.
    """
    from common_utils import Trajectory
    dz = config.gripper_depth_offset_franka
    c = np.asarray(hinge, float)[:2]
    rot = [float(yaw)]

    def pt(r, a):
        return c + r * np.array([math.cos(a), math.sin(a)])

    def go(xy, zz, desc):
        api.execute_trajectory(Trajectory([[float(xy[0]), float(xy[1]), zz + dz, rot[0]]], desc))
        return np.asarray(api.get_robot_state()["eef_pos"], float)

    def hand_deg(xy):
        return math.degrees(sgn * math.remainder(math.atan2(xy[1] - c[1], xy[0] - c[0]) - a_start, 2 * math.pi))

    api.open_gripper()
    if route == "around":
        return _push_around(api, c, sgn, r_grip, a_grip, a_start, z, rot, max_deg, r_push, r_out, go, hand_deg,
                            pt, log)
    r_push = float(r_push if r_push is not None else max(0.22, r_grip - 0.06))
    zt = z + lift
    tip = np.asarray(api.get_robot_state()["eef_pos"], float)
    for t in np.linspace(0, 1, 5)[1:]:                  # straight up, small steps: clear the handle
        tip = go(tip[:2], z + (zt - z) * t, "push_open: lift off the handle")
    if tip[2] < z + 0.08:
        log(f"push_open: could not lift clear (tip z {tip[2]:.3f}, wanted {zt:.3f})")
        return "push_blocked", 0.0
    zt = min(zt, float(tip[2]))                         # fly at the height the arm really reaches
    api.close_gripper()                                 # compact fingers for pushing
    a_b = a_grip - sgn * behind / r_push
    for k in range(4):
        n = max(2, int(math.ceil(float(np.linalg.norm(pt(r_push, a_b) - tip[:2])) / 0.03)))
        p0 = tip[:2]
        for t in np.linspace(0, 1, n + 1)[1:]:
            tip = go(p0 + (pt(r_push, a_b) - p0) * t, zt, "push_open: over to the back side")
            if tip[2] < zt - 0.04:                      # dragged down by / caught on the panel: stop
                log(f"push_open: hand pulled down to z={tip[2]:.3f} crossing over; aborting")
                go(tip[:2], zt + 0.05, "push_open: up")
                return "push_blocked", 0.0
        for t in np.linspace(0, 1, 5)[1:]:
            tip = go(pt(r_push, a_b), zt + (z - zt) * t, "push_open: down behind the panel")
        if tip[2] < z + 0.03:
            break
        log(f"push_open: landed on something at z={tip[2]:.3f} (hand angle {hand_deg(tip):.0f} deg); 5 deg further back")
        tip = go(tip[:2], zt, "push_open: up")
        a_b -= sgn * math.radians(5)
    else:
        log("push_open: could not get down behind the panel")
        return "push_blocked", 0.0
    log(f"push_open: behind the panel at r={r_push:.2f} m, hand angle {hand_deg(tip):.0f} deg; "
        f"sweeping to {max_deg:.0f} deg")
    a, stalls, prev = a_b, 0, tip[:2]
    a_end = a_start + sgn * math.radians(max_deg)
    while sgn * math.remainder(a_end - a, 2 * math.pi) > 0:
        a += sgn * step / r_push
        cur = go(pt(r_push, a), z, "push_open: push")[:2]
        moved = float(np.linalg.norm(cur - prev))
        prev = cur
        stalls = stalls + 1 if moved < 0.2 * step else 0
        if stalls >= 3:
            break
    deg = hand_deg(prev)
    out = "pushed" if deg >= max_deg - 5 else "push_stalled"
    log(f"push_open: {out}; hand at {deg:.0f} deg about the hinge (the door is at least that open)")
    for t in np.linspace(0, 1, 4)[1:]:                  # straight up: never drag the panel back
        go(prev, z + (zt - z) * t, "push_open: up")
    api.open_gripper()
    return out, round(deg, 1)


def guarded_pull(api, direction, hinge=None, travel=0.55, max_deg=80.0, step=0.03, follow_yaw=True,
                 max_flips=1, max_regrasps=2, part=None, settle_s=0.5, max_force=None,
                 early_m=0.15, fit_after_m=0.06, push_after_deg=15.0, push_kw=None):
    """Pull the held handle along ``direction`` (world, horizontal part used) or around ``hinge``
    ([x, y] of the hinge axis, the door edge FAR from the handle), guarded against slips and jams.

    ``push_after_deg`` (default 15; None = off): a late slip / jam / stall on a hinge arc after that
    many degrees switches to ``push_open(route="around")`` (``push_kw`` overrides): release, go out
    past the free edge, round it and down outside the panel, come in behind it (stopping where the
    arm meets the latch spindle), push the back face along the arc, then unload / drop / slide out
    radially / rise so the retreat does not drag the panel back. A pull alone stops at 18-36 deg
    (the panel meets the forearm and pries the jaws open); with the push the PyBullet door:hidden
    harness reached 60-74 deg in 4 of 5 runs whose pull got past ~29 deg (2026-11-08,
    door_around29-33). Outcome ``"opened"`` needs a hand angle >= 60 deg; the hand angle is a LOWER
    bound (the panel is ahead of the hand), so ``"push_stalled"`` with ``swept_deg`` >= ~50 is often
    already wide open: check the camera. ``"push_blocked"``: the wedge behind the panel was too
    narrow (short pull) - re-grasp and pull again.

    Returns ``{"outcome": "opened" | "end_stop" | "blocked" | "slipped" | "not_holding" |
    "unsupported" | "push_stalled" | "push_blocked", "travel_m", "swept_deg", "hinge", "flips", "regrasps", "events"}``.
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
            return True
        res["regrasps"] += 1
        where = np.asarray(v.get("last_held_pos") or v.get("slip_pos") or [o[0], o[1], z], float)
        log(f"re-grasp after the {reason} trip at {np.round(where, 3).tolist()} (last held)")
        _regrasp(api, [float(where[0]), float(where[1]), z], yaw)
        st_ = api.get_robot_state()
        p = np.asarray(st_["eef_pos"], float)[:2]
        if not st_.get("gripper_holding"):
            # the guard does not arm on an empty hand: going on would sweep air and report "opened"
            log("re-grasp closed on nothing")
            return False
        return True

    arm()
    p, yaw, path, s, stalls = o.copy(), yaw0, [o.copy()], 0.0, 0

    def _push(swept, why):
        """Pull stopped part-way on an arc (slip / jaws pried open / stall): push the rest."""
        api.guard_motion("off")
        r0 = float(np.linalg.norm(o - centre))
        a0 = math.atan2(o[1] - centre[1], o[0] - centre[0])
        ap = math.atan2(p[1] - centre[1], p[0] - centre[0])
        log(f"pull stopped at {swept:.0f} deg ({why}): switching to push_open from the back face")
        out, deg = push_open(api, centre, sgn, r0, ap, a0, z, yaw, max_deg=max_deg, log=log,
                             **{"route": "around", **(push_kw or {})})
        res.update(outcome="opened" if out in ("pushed", "push_stalled") and deg >= 60 else out,
                   swept_deg=max(swept, deg), pushed=True)
        return res
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
                if not api.get_robot_state().get("gripper_holding"):
                    res["outcome"] = "slipped"
                    log(f"swept {s * 100:.0f} cm but the hand holds nothing: the part did not follow")
                    return res
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
                    if centre is not None and push_after_deg is not None and swept >= push_after_deg:
                        return _push(swept, "stall")
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
            if s >= early_m and centre is not None and push_after_deg is not None and swept >= push_after_deg:
                return _push(swept, f"guard {v.get('check')}")
            if s < early_m:
                if not hyps or res["flips"] >= flips_allowed:
                    res["outcome"] = "blocked" if v.get("check") == "jam" else "slipped"
                    log("early trip, no hypotheses left")
                    return res
                res["flips"] += 1
                centre = hyps.pop(0)
                sgn = _arc_sign(centre, o, d0)
                log(f"early trip: wrong path, trying the hinge at {np.round(centre, 3).tolist()}")
                if not restart(v.get("check"), v):
                    res["outcome"] = "slipped"
                    return res
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
