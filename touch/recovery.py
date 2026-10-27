"""Agent-side touch recovery for opening articulated things: pull, and re-grasp after a slip.

``open_with_touch`` runs in the agent process (the LLM's code or a scripted harness) and only
uses the same API calls the LLM has: trajectories, gripper, robot state, ``pull_with_touch``
and (if active) the tracking report. No simulator truth.

After a slip the new handle position comes from proprioception (the fingertip position where the
handle was last held, ``slip_pos``, projected back onto the fitted hinge arc), cross-checked against
the tracked handle (CoTracker) when a tracking session follows it. The re-grasp keeps the hand's
current orientation (turning to the door's new normal is often out of reach once the door has
swung), re-approaches a few cm, and the pull resumes on the earlier hinge fit.
"""

import math
from typing import Optional

import numpy as np

import config

_GRASP_MIN_WIDTH = 0.004


def _log(api, msg):
    print(msg)
    lg = getattr(api, "logger", None)
    if lg is not None:
        lg.info(config.PROGRESS + msg + config.ENDC)


def _state(api):
    api.main_connection.send([config.GET_ROBOT_STATE, {}])
    return api.main_connection.recv()


def _execute(api, points, desc):
    if hasattr(api, "execute_trajectory") and callable(api.execute_trajectory):
        from common_utils import Trajectory
        api.execute_trajectory(Trajectory([list(map(float, p)) for p in points], desc))
        return
    api.main_connection.send([config.EXECUTE_TRAJECTORY, [list(map(float, p)) for p in points]])
    api.main_connection.recv()


def _gripper(api, open_):
    if open_ and hasattr(api, "open_gripper"):
        return api.open_gripper()
    if not open_ and hasattr(api, "close_gripper"):
        return api.close_gripper()
    api.main_connection.send([config.OPEN_GRIPPER if open_ else config.CLOSE_GRIPPER])


def _pull(api, direction, **kw):
    if hasattr(api, "pull_with_touch") and callable(api.pull_with_touch):
        return api.pull_with_touch(direction, **kw)
    if not getattr(getattr(api, "args", None), "touch", True):
        return {"outcome": "disabled"}
    spec = {"direction": [float(v) for v in direction], "max_s": kw.get("max_s", 20.0),
            "grip_force": kw.get("grip_force", 30.0), "config": dict(kw.get("pull_config") or {})}
    api.main_connection.send([config.TOUCH_PULL, spec])
    msg, res = api.main_connection.recv()
    _log(api, str(msg))
    for ev in res.get("events", []):
        _log(api, f"[touch]   {ev}")
    return res


def _tracked_point(api, name: Optional[str]):
    """Latest tracked world point of ``name`` (or the only tracked object), None if unavailable."""
    if not hasattr(api, "get_tracking_report"):
        return None
    try:
        rep = api.get_tracking_report(include_frames=True)
    except Exception:  # noqa: BLE001
        return None
    frames = (rep or {}).get("frames")
    if not isinstance(frames, list):
        return None
    for fr in reversed(frames):
        objs = fr.get("objects") or {}
        keys = [name] if name in objs else (list(objs) if name is None and len(objs) == 1 else [])
        for k in keys:
            o = objs[k]
            if not o.get("lost") and o.get("world_point") is not None:
                return np.asarray(o["world_point"], float)
    return None


def _rotz(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _pose(tip, approach_yaw, rotation):
    """Command pose whose fingertip centre lands on ``tip`` (the pipeline's depth-offset convention)."""
    dz = config.gripper_depth_offset_franka
    if approach_yaw is None:                     # top-down
        return [tip[0], tip[1], tip[2] + dz, rotation]
    from common_utils import side_grasp_pose
    a = np.array([math.cos(approach_yaw), math.sin(approach_yaw), 0.0])
    p = np.asarray(tip, float) - a * dz
    return side_grasp_pose(p[0], p[1], p[2], rotation, approach_yaw)


def _lin(a, b, step=0.01):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(math.ceil(np.linalg.norm(b[:3] - a[:3]) / step)) + 1)
    return [list(a + (b - a) * s) for s in np.linspace(0, 1, n)[1:]]


def regrasp(api, handle_tip, approach_yaw=None, rotation=0.0, hand_yaw=None, back_off=0.15, tip_depth=0.01,
            retreat=0.08):
    """Open, retreat, re-approach ``handle_tip`` (fingertip target) and close. Returns grip width (m).

    ``hand_yaw``: where the hand points NOW (side: its approach yaw; top-down: its rotation), which
    differs from the new ``approach_yaw``/``rotation`` after a hinge turned it. The hand first
    retreats along its current axis without turning (turning at the handle sweeps the fingers
    through the door), turns at the stand-off in free space, then advances onto the handle.
    """
    _gripper(api, True)
    here = np.asarray(_state(api)["eef_pos"], float)
    side = approach_yaw is not None
    now_yaw = (approach_yaw if side else rotation) if hand_yaw is None else hand_yaw

    def axis(yaw):
        return np.array([math.cos(yaw), math.sin(yaw), 0.0]) if side else np.array([0.0, 0.0, -1.0])

    def pose(tip, yaw):
        return _pose(tip, yaw, rotation) if side else _pose(tip, None, yaw)

    new_yaw = approach_yaw if side else rotation
    a_now, a_new = axis(now_yaw), axis(new_yaw)
    handle_tip = np.asarray(handle_tip, float)
    target = handle_tip + a_new * tip_depth
    stand = handle_tip - a_new * back_off
    away = here - a_now * retreat
    _execute(api, _lin(pose(here, now_yaw), pose(away, now_yaw), 0.01), "touch regrasp: retreat")
    _execute(api, _lin(pose(away, now_yaw), pose(stand, new_yaw), 0.015), "touch regrasp: stand-off")
    for p in _lin(pose(stand, new_yaw), pose(target, new_yaw), 0.02):
        _execute(api, [p], "touch regrasp: advance")
    _gripper(api, False)
    _execute(api, [pose(target, new_yaw)], "touch regrasp: close")
    return float(_state(api).get("gripper_width") or 0.0)


def open_with_touch(api, direction, approach_yaw=None, rotation=0.0, target_deg=65.0, max_travel=0.6,
                    max_regrasps=2, handle_name=None, max_s=20.0, grip_force=30.0, pull_config=None):
    """Open a grasped door/drawer: touch pull, and on a slip re-grasp the moved handle and resume.

    Call with the handle already grasped. ``direction``: initial pull direction (e.g. the door
    face normal, toward the robot). ``approach_yaw``/``rotation``: how the handle was grasped
    (``approach_yaw=None`` = top-down with ``rotation``; else a side grasp as in
    ``side_grasp_pose``). Returns ``{"outcome": "opened" | "end_stop" | "slipped" | "blocked" |
    ..., "swept_deg", "travel_cm", "attempts": [...]}``. With ``--no-touch``: ``"disabled"``.
    """
    d = np.asarray(direction, float)
    swept, travel, attempts, pushed, hinge = 0.0, 0.0, [], False, None
    side = approach_yaw is not None
    res = {"outcome": None}
    for k in range(max_regrasps + 1):
        res = _pull(api, d, max_s=max_s, grip_force=grip_force,
                    pull_config={**(pull_config or {}), "open_angle_deg": max(5.0, target_deg - swept),
                                 "max_travel": max(0.05, max_travel - travel),
                                 **({"hinge": hinge} if hinge else {})})
        out = res.get("outcome")
        attempts.append({k2: v for k2, v in res.items() if k2 != "events"})
        if out in ("disabled", "unsupported", "error", None):
            return {"outcome": out, "attempts": attempts}
        swept += abs(float(res.get("swept_deg") or 0.0))
        travel += float(res.get("travel_cm") or 0.0) / 100.0
        d = np.asarray(res.get("direction") or d, float)
        hinge = res.get("hinge") or hinge
        _log(api, f"[touch.open] pull {k}: {out} swept={res.get('swept_deg')} total={swept:.1f}deg "
                  f"travel={travel * 100:.0f}cm hand_yaw={res.get('hand_yaw_deg')} hinge={hinge}")
        if out in ("opened", "end_stop"):
            break
        if out == "blocked" and not pushed and travel < 0.02:
            pushed = True                        # never moved: the other way (push) is the likely opening
            d = -d
            _log(api, "[touch.open] blocked before moving -> trying the opposite direction")
            continue
        if out not in ("slipped", "timeout") or k == max_regrasps:
            break
        # ---- re-grasp the moved handle: proprioception (+ hinge fit), tracking as a cross-check
        slip = res.get("slip_pos") or res.get("end_pos")
        if slip is None:
            break
        hand_yaw = (approach_yaw if side else rotation) + math.radians(float(res.get("hand_yaw_deg") or 0.0))
        a_now = (np.array([math.cos(hand_yaw), math.sin(hand_yaw), 0.0]) if side else np.zeros(3))
        # the handle was sliding out toward the fingertips when the width dropped
        handle = np.asarray(slip, float) + a_now * 0.015
        if hinge is not None:
            c, r = np.asarray(hinge["centre_xy"], float), float(hinge["radius"])
            v = handle[:2] - c
            if np.linalg.norm(v) > 1e-6:
                handle[:2] = c + r * v / np.linalg.norm(v)        # back onto the handle's arc
        tracked = _tracked_point(api, handle_name)
        if tracked is not None:
            off = float(np.linalg.norm((tracked - handle)[:2]))
            _log(api, f"[touch.open] tracked handle {np.round(tracked, 3).tolist()} vs touch estimate "
                      f"{np.round(handle, 3).tolist()} ({off * 100:.1f}cm)")
            if off < 0.03:
                handle = 0.5 * (handle + tracked)
        # Re-approach with the hand's CURRENT orientation: it is reachable (the hand is there), and
        # a bar is still straddled obliquely. Turning to the door's new normal is often out of reach
        # once the door has swung toward the arm.
        if side:
            approach_yaw = hand_yaw
        else:
            rotation = hand_yaw
        w = regrasp(api, handle, approach_yaw, rotation, hand_yaw=hand_yaw, back_off=0.05, retreat=0.03)
        _log(api, f"[touch.open] regrasp {k + 1} at {np.round(handle, 3).tolist()} "
                  f"{'approach_yaw' if side else 'rotation'}={math.degrees(hand_yaw):.1f}deg "
                  f"-> grip width {w * 1000:.1f}mm")
        if w < _GRASP_MIN_WIDTH:
            res = {"outcome": "regrasp_failed"}
            break
    outcome = res.get("outcome")
    return {"outcome": outcome, "swept_deg": round(swept, 1), "travel_cm": round(travel * 100, 1),
            "direction": np.round(d, 3).tolist(), "hinge": hinge, "attempts": attempts,
            "approach_yaw": approach_yaw, "rotation": rotation}
