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


def _closed_state(api):
    st = api.get_robot_state()
    w = float(st.get("gripper_width") or 0.0)
    return bool(st.get("gripper_holding")) and w > 0.005, w, st


def _move_tip(api, frm, to, rot, desc, step=0.005):
    from common_utils import Trajectory
    dz = config.gripper_depth_offset_franka
    api.execute_trajectory(Trajectory(_line([frm[0], frm[1], frm[2] + dz, rot], [to[0], to[1], to[2] + dz, rot], step),
                                      desc))


def _recentre_hold(api, out, bar_w, min_ratio, rot):
    """The fingers stalled at a width well under the bar thickness: they pinch a chord of the bar off its
    centre line (above or below it; the pads then roll off under a pull). Re-open and close again higher
    / lower by the chord offset ``sqrt(r^2 - (w/2)^2)``: lower first (a descent that stops on the bar leaves
    the pads above its centre). Keeps the widest hold; returns the updated ``out``."""
    st = api.get_robot_state()
    tip = np.asarray(st.get("eef_pos") or out["xyz"], float)
    r, w0 = 0.5 * bar_w, float(out["width"])
    dzc = max(0.005, math.sqrt(max(r * r - (0.5 * w0) ** 2, 0.0)))
    best, cur = (w0, float(tip[2])), tip.copy()
    for k, zc in enumerate((tip[2] - dzc, tip[2] + dzc)):
        api.open_gripper()
        tgt = np.array([tip[0], tip[1], zc])
        _move_tip(api, cur, tgt, rot, f"door lever: re-centre {k}")
        api.close_gripper()
        hold, w, st = _closed_state(api)
        cur = np.asarray(st.get("eef_pos") or tgt, float)
        api.logger.info(f"[door_open] re-centre {k} at z {zc:.3f}: holding={hold} width={w:.4f} "
                        f"(bar {bar_w * 1000:.0f} mm)")
        if hold and w >= min_ratio * bar_w:
            return dict(out, holding=True, width=round(w, 4), recentred=True,
                        xyz=[round(float(v), 3) for v in (tip[0], tip[1], zc)])
        if hold and w > best[0]:
            best = (w, float(zc))
    api.open_gripper()
    _move_tip(api, cur, [tip[0], tip[1], best[1]], rot, "door lever: back to the best hold")
    api.close_gripper()
    hold, w, _ = _closed_state(api)
    api.logger.info(f"[door_open] re-centre did not reach {min_ratio:.0%} of the bar; best hold at z {best[1]:.3f}: "
                    f"holding={hold} width={w:.4f}")
    return dict(out, holding=hold, width=round(w, 4), off_centre=True,
                xyz=[round(float(v), 3) for v in (tip[0], tip[1], best[1])])


def _press_lever(api, press_m, press_force):
    """Press the held lever down (unlatch) by at most ``press_m``, stopping at ``press_force`` N of
    resistance (move_until_contact = impedance + felt force, as on a real arm: a sprung lever yields, a
    stiff one stops the hand). A position-controlled press into a stiff lever jammed at ~1-4 kN on Genesis
    and peeled the grip, so without force feedback it is only done where the arm is not torque-driven."""
    from common_utils import Trajectory
    mu = getattr(api, "move_until_contact", None)
    res = None
    if callable(mu):
        try:
            res = mu([0.0, 0.0, -1.0], max_travel=float(press_m), force=float(press_force), speed=0.01,
                     grip_force=30.0)
        except Exception as e:  # noqa: BLE001
            api.logger.info(f"[door_open] guarded press failed ({str(e)[:120]})")
    oc = (res or {}).get("outcome")
    if oc in ("contact", "no_contact", "over_force"):
        api.logger.info(f"[door_open] lever press (force-limited {press_force:.0f} N): {oc}, "
                        f"{res.get('travel_mm')} mm, {res.get('contact_N')} N")
        return {"mode": "force", "outcome": oc, "travel_mm": res.get("travel_mm")}
    if oc == "disabled" and getattr(getattr(api, "args", None), "sim", None) == "genesis":
        api.logger.info("[door_open] lever press skipped: no force feedback (--no-touch) on a torque-driven arm")
        return {"mode": "skipped"}
    st = api.get_robot_state()
    tip, rot = np.asarray(st["eef_pos"], float), float(st.get("rotation") or 0.0)
    dz = config.gripper_depth_offset_franka
    api.execute_trajectory(Trajectory(_line([tip[0], tip[1], tip[2] + dz, rot],
                                            [tip[0], tip[1], tip[2] - press_m + dz, rot], 0.005), "door lever: press"))
    return {"mode": "position"}


def grasp_door_lever(api, plane, door_deg=0.0, offsets=(0.0, 0.02, -0.02), clear_z=0.12, press_m=0.03,
                     press_force=20.0, min_width_ratio=0.8):
    """Top-down grasp at the depth-measured lever middle, fingers across the bar; checks
    ``gripper_holding`` and the finger width, and retries +-2 cm along the bar. A hold narrower than
    ``min_width_ratio`` x the depth-measured bar thickness (``lever_grasp["bar_width_m"]``) pinches the bar
    off-centre: re-centred in height (``_recentre_hold``). ``press_m``: press the lever down after closing
    (unlatch), force-limited to ``press_force`` N where the arm has force feedback.
    Returns {"holding", "xyz", "rotation", "width", "tries", "press"?, "recentred"? / "off_centre"?}."""
    from common_utils import Trajectory
    g = lever_grasp_now(plane, door_deg)
    if not g:
        return {"holding": False, "reason": "no lever_grasp (lever not seen in depth)", "tries": 0}
    bar_w = (plane.get("lever_grasp") or {}).get("bar_width_m")
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
        hold, w, _ = _closed_state(api)
        out = {"holding": hold, "xyz": [round(float(v), 3) for v in (x, y, z)],
               "rotation": round(rot, 3), "width": round(w, 4), "tries": k + 1}
        api.logger.info(f"[door_open] grasp try {k} at {out['xyz']} rot {rot:.2f}: holding={out['holding']} width={w:.4f}")
        if out["holding"] and bar_w and w < min_width_ratio * float(bar_w):
            api.logger.info(f"[door_open] off-centre hold: width {w * 1000:.0f} mm < {min_width_ratio:.0%} of the "
                            f"{float(bar_w) * 1000:.0f} mm bar -> re-centre")
            out = _recentre_hold(api, out, float(bar_w), min_width_ratio, rot)
        if out["holding"]:
            if press_m:
                out["press"] = _press_lever(api, press_m, press_force)
            return out
        api.open_gripper()
        st = api.get_robot_state()
        tip = np.asarray(st.get("eef_pos") or [x, y, z], float)
        api.execute_trajectory(Trajectory(_line([tip[0], tip[1], tip[2] + dz, rot], above, 0.02),
                                          f"door lever: lift after miss {k}"))
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


def _to_view(api, plane, view):
    """Hand back to ``view`` (fingertip xyz of a pose the door was already seen from) over the panel top,
    so the arm does not hide the panel from the head camera nor sweep across a swung door."""
    from common_utils import Trajectory
    st = api.get_robot_state()
    tip, rot = np.asarray(st["eef_pos"], float), float(st.get("rotation") or 0.0)
    dz = config.gripper_depth_offset_franka
    top = plane.get("top_z") if plane else None
    safe = max(float(top) + 0.06 if top is not None else tip[2] + 0.15, tip[2], float(view[2]))
    a = [tip[0], tip[1], tip[2] + dz, rot]
    up = [tip[0], tip[1], safe + dz, rot]
    over = [float(view[0]), float(view[1]), safe + dz, rot]
    end = [float(view[0]), float(view[1]), float(view[2]) + dz, rot]
    api.execute_trajectory(Trajectory(_line(a, up, 0.02) + _line(up, over, 0.02) + _line(over, end, 0.02),
                                      "door: back to the clear view"))


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


def _exec_guarded(api, traj):
    """execute_trajectory, but a grasp-slip guard trip (raised as RuntimeError) becomes a guard dict: the
    caller stops the probe and open_door re-grasps, instead of aborting the whole call."""
    try:
        return api.execute_trajectory(traj)
    except RuntimeError as e:
        if "grasp-slip" not in str(e):
            raise
        api.logger.info(f"[door_open] grasp slip during the probe: {str(e)[:160]}")
        return {"guard": {"check": "grasp_slip"}}


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
        m = _exec_guarded(api, Trajectory([[float(tgt[0]), float(tgt[1]), tip0[2] + dz, rot]], "door: probe pull"))
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
            m = _exec_guarded(api, Trajectory([[float(tgt[0]), float(tgt[1]), tip0[2] + dz, rot]],
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


def _swung_dir(plane, door_deg):
    n = np.asarray(plane["pull_dir"][:2], float)
    if door_deg and abs(door_deg) > 1.0:
        n = _rot_about(n, np.zeros(2), _open_sign(plane) * math.radians(abs(door_deg)))
    return [float(n[0]), float(n[1]), 0.0]


def open_door(api, handle, goal_deg=GOAL_DEG, max_pulls=3, max_deg=85.0, push_from_deg=30.0, touch=True):
    """Open a hinged door wide (>= ``goal_deg`` measured from head depth). ``handle`` = world [x, y, z] of
    any point on the handle (from detect_object). Steps: estimate_door_plane -> grasp_door_lever ->
    pull -> measure_door_angle. The pull is the compliant touch pull (``touch.recovery.open_with_touch``:
    force-limited, compliant sideways, the hinge steers the hand; no hinge prior) when the simulator has
    torque control (Genesis) and ``touch``; otherwise probe_hinge + guarded_pull (pull + push). Then,
    until >= ``goal_deg`` (``max_pulls`` moves): without touch a door >= ``push_from_deg`` open is pushed
    from behind (``push_door``); otherwise (and with touch always) the swung lever is re-grasped and pulled
    again (grasp misses do not count, at most 2). Releases and lifts at the end. Returns {"outcome":
    "opened" | "partly_open" | "no_door_face" | "grasp_failed", "door_deg", "pulls", "plane", "method"}."""
    plane = api.estimate_door_plane(handle)
    if not plane:
        return {"outcome": "no_door_face", "door_deg": None, "pulls": [], "plane": None,
                "hint": "no door face in depth: clear the occluder / move the arm out of view, then retry"}
    deg = float(plane.get("door_open_deg") or 0.0)
    pulls, misses, moves = [], 0, 0
    use_touch = bool(touch) and getattr(getattr(api, "args", None), "touch", True) is not False
    touch_hinge = None
    res = {"outcome": "partly_open", "door_deg": deg, "pulls": pulls, "plane": plane}
    try:  # the plane was just measured from this hand pose: a known clear view of the door
        view = list(api.get_robot_state()["eef_pos"])
    except Exception:  # noqa: BLE001
        view = None

    def measure(prev, est=None):
        """Door angle from head depth. When the panel is not seen (arm in view), look again from the clear
        view; still unseen -> ``est`` (the angle the hand swept the door through while holding it), never
        a stale pre-pull reading (door:hidden gs: a 40 deg swing was logged as the old 0.8 deg and the
        re-grasp went to the closed-door lever, knocking the panel shut)."""
        m = api.measure_door_angle()
        if m is None and view is not None:
            _to_view(api, plane, view)
            m = api.measure_door_angle()
        if m is not None:
            return float(m)
        if est is not None:
            api.logger.info(f"[door_open] door not seen in depth; using the pull's swept arc {est:.1f} deg")
            return float(est)
        api.logger.info(f"[door_open] door not seen in depth; keeping the last estimate {prev:.1f} deg")
        return prev

    while deg < goal_deg and moves < max_pulls and misses <= 2:
        if deg >= push_from_deg and not use_touch:
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
        if use_touch:
            from touch.recovery import open_with_touch
            r = open_with_touch(api, direction=_swung_dir(plane, deg), rotation=float(g["rotation"]),
                                target_deg=max(10.0, goal_deg + 5.0 - deg), handle_name="door handle",
                                pull_config={"hinge": touch_hinge} if touch_hinge else None)
            if r.get("outcome") in ("disabled", "unsupported", "error", None):
                api.logger.info(f"[door_open] touch pull {r.get('outcome')} -> probe_hinge + guarded_pull")
                use_touch = False
            else:
                moves += 1
                if r.get("hinge"):
                    touch_hinge = r["hinge"]
                    # the fitted (moved-door) hinge replaces the depth hypothesis for the re-grasp of the swung lever
                    plane = dict(plane, hinge=list(touch_hinge["centre_xy"]), free_edge=list(g["xyz"][:2]),
                                 hinge_source="touch")
                    res["plane"] = plane
                _release_and_clear(api)
                sw = r.get("swept_deg")
                deg = measure(deg, est=(deg + float(sw)) if sw and r.get("hinge") else None)
                pulls.append({"grasp": g, "touch_pull": {k: r.get(k) for k in ("outcome", "swept_deg", "travel_cm",
                                                                                 "hinge")}, "door_deg": deg})
                api.logger.info(f"[door_open] touch pull {moves}: {r.get('outcome')} swept={r.get('swept_deg')} "
                                f"hinge={touch_hinge} -> door {deg} deg")
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
        sw = r.get("vision_deg")
        if sw is None and r.get("swept_deg") is not None:
            sw = start + float(r["swept_deg"])
        deg = measure(deg, est=sw)
        pulls.append({"grasp": g, "pull": {k: r.get(k) for k in ("outcome", "swept_deg", "vision_deg", "pushed")},
                      "door_deg": deg})
        api.logger.info(f"[door_open] pull {moves}: {r.get('outcome')} swept={r.get('swept_deg')} -> door {deg} deg")
    res["door_deg"] = round(deg, 1)
    res["method"] = "touch_pull" if use_touch else "guarded_pull"
    if deg >= goal_deg:
        res["outcome"] = "opened"
    elif moves == 0:
        res["outcome"] = "grasp_failed"
    api.logger.info(f"[door_open] {res['outcome']} door {res['door_deg']} deg after {moves} move(s), {misses} miss(es)")
    return res
