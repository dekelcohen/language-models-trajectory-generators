"""Deterministic hinged-door opening: depth plane -> mid-lever grasp -> guarded pull/push -> depth check.

Small LLMs (door:hidden q1, qwen3.8-27b) ignored ``lever_grasp`` and hand-placed 4 grasps that closed
on nothing. These helpers take the grasp and the retry loop out of the LLM's code: one call opens the
door from pixels + proprioception only (head depth, EE pose, finger width).
"""
import math

import numpy as np

import config

GOAL_DEG = 65.0                  # 60 deg wanted + ~3-5 deg depth error margin


def _rot_about(p, c, ang):
    c_, s_ = math.cos(ang), math.sin(ang)
    d = np.asarray(p, float) - c
    return c + np.array([c_ * d[0] - s_ * d[1], s_ * d[0] + c_ * d[1]])


def _open_sign(plane):
    """+1 / -1: the rotation sense about the hinge that swings the free edge toward the robot."""
    hinge, n = np.asarray(plane["hinge"], float), np.asarray(plane["pull_dir"][:2], float)
    free = np.asarray(plane["free_edge"], float)
    return 1.0 if float(np.dot(_rot_about(free, hinge, 0.1) - free, n)) > 0 else -1.0


def lever_grasp_now(plane, door_deg=0.0):
    """The plane's (closed-door) lever grasp, swung about the hinge by ``door_deg`` toward the robot."""
    g = plane.get("lever_grasp") if plane else None
    if not g:
        return None
    xyz, rot = np.asarray(g["xyz"], float), float(g["rotation"])
    if door_deg and abs(door_deg) > 1.0:
        ang = _open_sign(plane) * math.radians(abs(door_deg))
        xyz[:2] = _rot_about(xyz[:2], np.asarray(plane["hinge"], float), ang)
        rot = (rot + ang + math.pi / 2) % math.pi - math.pi / 2
    return {"xyz": xyz.tolist(), "rotation": rot}


def _line(a, b, step=0.01):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(math.ceil(np.linalg.norm(b[:3] - a[:3]) / step)))
    return [list(a + (b - a) * (i / n)) for i in range(1, n + 1)]


def _approach(api, plane, door_deg, target, rot, desc, front_m=0.15):
    """Hand to ``target`` (fingertip xyz) without crossing the panel: up above the panel top, over to
    ``front_m`` in front of the (swung) face, down, then in horizontally. A diagonal path from high up to
    the hover point under the panel top knocked a 21-31 deg door shut (harness od2, door:hidden q2)."""
    from common_utils import Trajectory
    dz = config.gripper_depth_offset_franka
    n = np.asarray(plane["pull_dir"][:2], float)
    if door_deg and abs(door_deg) > 1.0:
        n = _rot_about(n, np.zeros(2), _open_sign(plane) * math.radians(abs(door_deg)))
    top = plane.get("top_z")
    safe = (float(top) + 0.06) if top is not None else float(target[2]) + 0.30
    tip = np.asarray(api.get_robot_state()["eef_pos"], float)
    x, y, z = (float(v) for v in target)
    fx, fy = x + front_m * n[0], y + front_m * n[1]
    a = [tip[0], tip[1], tip[2] + dz, rot]
    up = [tip[0], tip[1], max(tip[2], safe) + dz, rot]
    over = [fx, fy, max(tip[2], safe) + dz, rot]
    down = [fx, fy, z + dz, rot]
    api.execute_trajectory(Trajectory(_line(a, up, 0.02) + _line(up, over, 0.02) + _line(over, down, 0.02)
                                      + _line(down, [x, y, z + dz, rot], 0.01), desc))


def grasp_door_lever(api, plane, door_deg=0.0, offsets=(0.0, 0.02, -0.02), clear_z=0.12, press_m=0.03):
    """Top-down grasp at the depth-measured lever middle, fingers across the bar; checks
    ``gripper_holding`` and the finger width, and retries +-2 cm along the bar. ``press_m``: press the
    lever down after closing (unlatch). Returns {"holding", "xyz", "rotation", "width", "tries"}."""
    from common_utils import Trajectory
    g = lever_grasp_now(plane, door_deg)
    if not g:
        return {"holding": False, "reason": "no lever_grasp (lever not seen in depth)", "tries": 0}
    dz = config.gripper_depth_offset_franka
    u = _unit2(np.asarray(plane["hinge"], float) - np.asarray(plane.get("rose") or plane["free_edge"], float))
    if door_deg and abs(door_deg) > 1.0:
        u = _rot_about(u, np.zeros(2), _open_sign(plane) * math.radians(abs(door_deg)))
    rot = float(g["rotation"])
    out = {"holding": False, "tries": 0}
    for k, off in enumerate(offsets):
        x, y, z = np.asarray(g["xyz"], float) + np.r_[off * u, 0.0]
        api.open_gripper()
        above = [x, y, z + clear_z + dz, rot]
        down = [x, y, z + dz, rot]
        _approach(api, plane, door_deg, (x, y, z + clear_z), rot, f"door lever: above try {k}")
        api.execute_trajectory(Trajectory(_line(above, down, 0.005), f"door lever: descend try {k}"))
        api.close_gripper()
        st = api.get_robot_state()
        w = float(st.get("gripper_width") or 0.0)
        out = {"holding": bool(st.get("gripper_holding")) and w > 0.005, "xyz": [round(float(v), 3) for v in (x, y, z)],
               "rotation": round(rot, 3), "width": round(w, 4), "tries": k + 1}
        api.logger.info(f"[door_open] grasp try {k} at {out['xyz']} rot {rot:.2f}: holding={out['holding']} width={w:.4f}")
        if out["holding"]:
            if press_m:
                api.execute_trajectory(Trajectory(_line(down, [x, y, z - press_m + dz, rot], 0.005), "door lever: press"))
            return out
        api.open_gripper()
        api.execute_trajectory(Trajectory(_line(down, above, 0.02), f"door lever: lift after miss {k}"))
    return out


def _unit2(v):
    v = np.asarray(v, float)[:2]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _release_and_clear(api, up=0.12):
    from common_utils import Trajectory
    api.open_gripper()
    st = api.get_robot_state()
    tip, rot = np.asarray(st["eef_pos"], float), float(st.get("rotation") or 0.0)
    dz = config.gripper_depth_offset_franka
    a = [tip[0], tip[1], tip[2] + dz, rot]
    api.execute_trajectory(Trajectory(_line(a, [tip[0], tip[1], tip[2] + up + dz, rot], 0.02), "door: release and lift"))


def push_door(api, plane, door_deg, max_deg=85.0, hover=0.12):
    """Push a door that is already >= ~30 deg open further, from its back face, without a grasp: hover over
    the swung lever, then ``push_open(route="around")`` (out past the free edge, in behind the panel, push).
    Re-grasping a swinging lever missed and knocked the door shut (harness od2). Returns (outcome, hand_deg)."""
    from common_utils import Trajectory
    from helpers.guarded_pull import _arc_sign, push_open
    g0, gn = plane["lever_grasp"], lever_grasp_now(plane, door_deg)
    c, o = np.asarray(plane["hinge"], float), np.asarray(g0["xyz"][:2], float)
    sgn = _arc_sign(c, o, np.asarray(plane["pull_dir"][:2], float))
    z, rot = float(g0["xyz"][2]), float(gn["rotation"])
    x, y = gn["xyz"][:2]
    api.open_gripper()
    _approach(api, plane, door_deg, (x, y, z + hover), rot, "door: over the swung lever")
    return push_open(api, c, sgn, float(np.linalg.norm(o - c)), math.atan2(y - c[1], x - c[0]),
                     math.atan2(o[1] - c[1], o[0] - c[0]), z, rot, max_deg=max_deg, route="around",
                     log=lambda m: api.logger.info(f"[door_open] {m}"))


def probe_hinge(api, plane, pull_m=0.18, step=0.02, min_deg=3.0, refit_below_deg=15.0, arc_m=0.10):
    """Find the hinge by moving the door, not by looking at the frame: with the lever held, pull straight out
    along ``pull_dir`` (no hinge-side guess; a lever slides along the bar between the fingers while the
    panel swings), then fit the swung panel in head depth: it meets the closed face at the hinge
    (``touch.door_plane.hinge_from_swung_panel``). Stops early if the hand stalls (the door does not open
    this way) or the contact guard trips. Returns ``{"hinge", "free_edge", "open_deg", "flipped",
    "pulled_m", "source": "swung_panel" | "none", "reason"}``; the hand keeps holding the lever."""
    from common_utils import Trajectory
    dz = config.gripper_depth_offset_franka
    st = api.get_robot_state()
    out = {"hinge": None, "open_deg": 0.0, "pulled_m": 0.0, "source": "none", "flipped": False}
    if not st.get("gripper_holding"):
        out["reason"] = "not holding the lever"
        return out
    n = _unit2(plane["pull_dir"])
    tip0 = np.asarray(st["eef_pos"], float)
    rot = float(st.get("rotation") or 0.0)
    try:
        api.guard_motion("hold", on_trip="stop")
    except Exception:
        pass
    p, stalls, guard = tip0.copy(), 0, None
    for k in range(int(math.ceil(pull_m / step))):
        tgt = tip0[:2] + n * min(pull_m, step * (k + 1))
        m = api.execute_trajectory(Trajectory([[float(tgt[0]), float(tgt[1]), tip0[2] + dz, rot]], "door: probe pull"))
        guard = (m or {}).get("guard") if isinstance(m, dict) else None
        pn = np.asarray(api.get_robot_state()["eef_pos"], float)
        stalls = stalls + 1 if float(np.linalg.norm(pn[:2] - p[:2])) < 0.3 * step else 0
        p = pn
        if guard or stalls >= 2:
            break
    out["pulled_m"] = round(float(np.dot(p[:2] - tip0[:2], n)), 3)
    holding = bool(api.get_robot_state().get("gripper_holding"))
    found = api.find_hinge_from_depth(hand_xy=p[:2]) if callable(getattr(api, "find_hinge_from_depth", None)) else None
    api.logger.info(f"[door_open] probe pull {out['pulled_m'] * 100:.0f} cm (guard={guard and guard.get('check')}, "
                    f"stalls={stalls}, holding={holding}) -> swung panel {found}")
    # The hinge error is ~u*dtheta/theta: at < ~15 deg it is several cm, enough to jam the following arc pull.
    # Continue along the arc about the first fit (small steps, tangent re-computed; short arcs tolerate the
    # error) and re-fit the wider-open panel.
    if found and min_deg <= found["open_deg"] < refit_below_deg and holding and arc_m > 0:
        c = np.asarray(found["hinge"], float)[:2]
        moved = 0.0
        for _ in range(int(math.ceil(arc_m / step))):
            r = p[:2] - c
            t = _unit2(np.array([-r[1], r[0]]))
            if float(np.dot(t, n)) < 0:
                t = -t
            tgt = p[:2] + t * step
            m = api.execute_trajectory(Trajectory([[float(tgt[0]), float(tgt[1]), tip0[2] + dz, rot]],
                                                  "door: probe arc"))
            g2 = (m or {}).get("guard") if isinstance(m, dict) else None
            pn = np.asarray(api.get_robot_state()["eef_pos"], float)
            moved += float(np.linalg.norm(pn[:2] - p[:2]))
            p = pn
            if g2 or float(np.linalg.norm(pn[:2] - tgt)) > 0.6 * step:
                break
        holding = bool(api.get_robot_state().get("gripper_holding"))
        f2 = api.find_hinge_from_depth(hand_xy=p[:2])
        api.logger.info(f"[door_open] probe arc {moved * 100:.0f} cm about {np.round(c, 3).tolist()} "
                        f"(holding={holding}) -> swung panel {f2}")
        if f2 and not f2.get("flipped") and f2["open_deg"] > found["open_deg"]:
            found = f2
        out["pulled_m"] = round(out["pulled_m"] + moved, 3)
    if found and found["open_deg"] >= min_deg:
        out.update(found, source="swung_panel", holding=holding)
    else:
        out["reason"] = (f"door opened < {min_deg:.0f} deg on a {out['pulled_m'] * 100:.0f} cm straight pull"
                         if found or out["pulled_m"] > 0.03 else "the hand could not pull (door stuck / pushes away)")
        out["holding"] = holding
    return out


def open_door(api, handle, goal_deg=GOAL_DEG, max_pulls=3, max_deg=85.0, push_from_deg=30.0):
    """Open a hinged door wide (>= ``goal_deg`` measured from head depth). ``handle`` = world [x, y, z] of
    any point on the handle (from detect_object). Steps: estimate_door_plane -> grasp_door_lever ->
    guarded_pull (pull + push) -> measure_door_angle. Then, until >= ``goal_deg`` (``max_pulls`` moves):
    a door >= ``push_from_deg`` open is pushed from behind (``push_door``), a less open one is re-grasped
    at the swung lever and pulled again (grasp misses do not count, at most 2). Releases and lifts at the
    end. Returns {"outcome": "opened" | "partly_open" | "no_door_face" | "grasp_failed", "door_deg",
    "pulls", "plane"}."""
    plane = api.estimate_door_plane(handle)
    if not plane:
        return {"outcome": "no_door_face", "door_deg": None, "pulls": [], "plane": None,
                "hint": "no door face in depth: clear the occluder / move the arm out of view, then retry"}
    deg = float(plane.get("door_open_deg") or 0.0)
    pulls, misses, moves = [], 0, 0
    res = {"outcome": "partly_open", "door_deg": deg, "pulls": pulls, "plane": plane}

    def measure(prev):
        m = api.measure_door_angle()
        return prev if m is None else float(m)

    while deg < goal_deg and moves < max_pulls and misses <= 2:
        if deg >= push_from_deg:
            out, hand = push_door(api, plane, deg, max_deg=max_deg)
            moves += 1
            _release_and_clear(api)
            deg = measure(deg)
            pulls.append({"push": out, "hand_deg": hand, "door_deg": deg})
            api.logger.info(f"[door_open] push {moves}: {out} hand {hand} deg -> door {deg} deg")
            continue
        partly = deg > 3.0
        g = grasp_door_lever(api, plane, door_deg=deg, offsets=(0.0,) if partly else (0.0, 0.02, -0.02))
        if not g["holding"]:
            misses += 1
            pulls.append({"grasp": g})
            _release_and_clear(api)
            deg = measure(deg)
            api.logger.info(f"[door_open] grasp miss {misses}; door now {deg} deg")
            continue
        # a partly open door with a measured hinge: continue from the measured angle (no wrong-side flips)
        start = deg if partly and plane.get("hinge_source") == "swung_panel" else 0.0
        if plane.get("hinge_source") != "swung_panel" and not partly:
            pr = probe_hinge(api, plane)
            pulls.append({"probe": {k: pr.get(k) for k in ("source", "hinge", "open_deg", "flipped", "pulled_m",
                                                           "reason")}})
            if pr["source"] == "swung_panel":
                plane = dict(plane, hinge=pr["hinge"], free_edge=pr["free_edge"], hinge_source="swung_panel")
                res["plane"] = plane
                start = float(pr["open_deg"])
            if not pr.get("holding"):
                misses += 1
                _release_and_clear(api)
                deg = measure(deg)
                api.logger.info(f"[door_open] lost the lever on the probe pull; door now {deg} deg")
                continue
        r = api.guarded_pull(direction=plane["pull_dir"], hinge=plane["hinge"], max_deg=max_deg, start_deg=start)
        moves += 1
        _release_and_clear(api)
        deg = measure(deg)
        pulls.append({"grasp": g, "pull": {k: r.get(k) for k in ("outcome", "swept_deg", "vision_deg", "pushed")},
                      "door_deg": deg})
        api.logger.info(f"[door_open] pull {moves}: {r.get('outcome')} swept={r.get('swept_deg')} -> door {deg} deg")
    res["door_deg"] = round(deg, 1)
    if deg >= goal_deg:
        res["outcome"] = "opened"
    elif moves == 0:
        res["outcome"] = "grasp_failed"
    api.logger.info(f"[door_open] {res['outcome']} door {res['door_deg']} deg after {moves} move(s), {misses} miss(es)")
    return res
