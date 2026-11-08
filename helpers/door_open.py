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
        tip = np.asarray(api.get_robot_state()["eef_pos"], float)
        cur = [tip[0], tip[1], tip[2] + dz, rot]
        up = [tip[0], tip[1], max(tip[2], z + clear_z) + dz, rot]
        above = [x, y, z + clear_z + dz, rot]
        down = [x, y, z + dz, rot]
        api.execute_trajectory(Trajectory(_line(cur, up, 0.02) + _line(up, above, 0.02), f"door lever: above try {k}"))
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
    z, rot, dz = float(g0["xyz"][2]), float(gn["rotation"]), config.gripper_depth_offset_franka
    x, y = gn["xyz"][:2]
    api.open_gripper()
    tip = np.asarray(api.get_robot_state()["eef_pos"], float)
    up = [tip[0], tip[1], max(tip[2], z + hover) + dz, rot]
    api.execute_trajectory(Trajectory(_line([tip[0], tip[1], tip[2] + dz, rot], up, 0.02)
                                      + _line(up, [x, y, z + hover + dz, rot], 0.02), "door: over the swung lever"))
    return push_open(api, c, sgn, float(np.linalg.norm(o - c)), math.atan2(y - c[1], x - c[0]),
                     math.atan2(o[1] - c[1], o[0] - c[0]), z, rot, max_deg=max_deg, route="around",
                     log=lambda m: api.logger.info(f"[door_open] {m}"))


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
        r = api.guarded_pull(direction=plane["pull_dir"], hinge=plane["hinge"], max_deg=max_deg)
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
