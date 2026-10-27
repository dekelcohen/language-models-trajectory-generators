"""Run :class:`SpiralInsertController` on a simulated arm through the :class:`SimAdapter` API.

This is the sim stand-in for libfranka's Cartesian impedance controller (1 kHz torque loop) with
the controller's own force estimate. Everything it reads is robot-side: joint positions/velocities,
FK (link pose), the Jacobian and the gravity vector g(q) from the robot model, finger positions.
The scene (port pose, insertion depth, contacts) is never read.

Impedance law at the tool centre point (TCP = EE link + a fixed offset in the hand frame, like
setting the EE frame to the plug tip in Desk/libfranka)::

    tau = J^T [Km (p_des - p) - Dm v ; Kr e_R - Dr w] + N (k_null (q_rest - q) - d_null dq) + g(q)

Felt force (quasi-static ``O_F_ext_hat_K`` analogue): ``F = -(Km (p_des - p) - Dm v) + noise``.
"""

from typing import Callable, Optional, Sequence

import numpy as np

from touch.insertion import InsertConfig, SpiralInsertController
from touch.state import TouchState


def _R_from_xyzw(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _gains(k_axial, k_lat, a, mass=2.0):
    P = np.outer(a, a)
    L = np.eye(3) - P
    return (k_axial * P + k_lat * L,
            2 * np.sqrt(k_axial * mass) * P + 2 * np.sqrt(k_lat * mass) * L)


def run_touch_insert(sim, body, ee_link, arm_joints: Sequence, finger_joints: Sequence,
                     step: Callable[[], None], dt: float, axis, tool_offset=(0.0, 0.0, 0.0),
                     grip_force: float = 20.0, max_s: float = 20.0, period_s: float = 0.02,
                     cfg: Optional[InsertConfig] = None, force_noise: float = 0.5,
                     k_rot: float = 40.0, gravity_every: int = 8, seed: int = 0,
                     log: Optional[Callable[[str], None]] = None,
                     on_tick: Optional[Callable[[TouchState, object], None]] = None) -> dict:
    """Touch-guided insertion of the held part along world ``axis`` from the current pose.

    ``tool_offset``: world vector from the EE link origin to the part's leading tip at the start
    (fixed in the hand frame afterwards). ``step()`` advances the simulator by ``dt``. Fingers are
    squeezed with ``grip_force`` N each (Franka ``grasp(force)``) for the duration. On exit the arm
    and fingers are latched to position control at their current joint values.
    """
    if not sim.supports_torque_control():
        return {"outcome": "unsupported", "reason": f"{type(sim).__name__} has no torque control",
                "events": []}
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    arm, fingers = list(arm_joints), list(finger_joints)
    rng = np.random.default_rng(seed)

    p_link, q_link = sim.get_link_pose(body, ee_link)
    R0 = _R_from_xyzw(q_link)
    tcp_local = R0.T @ np.asarray(tool_offset, float)
    q_rest = np.array([sim.get_joint_state(body, j).position for j in arm])

    ctrl = SpiralInsertController(a, cfg, log=log)
    st = {"p_des": None, "K": ctrl.cfg.k_axial, "K_lat": ctrl.cfg.k_lat_search, "t": 0.0}
    g_cache = {"tau": None, "n": 0}

    def kin():
        p, qx = sim.get_link_pose(body, ee_link)
        R = _R_from_xyzw(qx)
        J = np.asarray(sim.get_link_jacobian(body, ee_link, arm, local_point=tcp_local), float)
        q = np.array([sim.get_joint_state(body, j).position for j in arm])
        dq = np.array([sim.get_joint_state(body, j).velocity for j in arm])
        tw = J @ dq
        return np.asarray(p, float) + R @ tcp_local, tw[:3], tw[3:], R, J, q, dq

    def torque_step():
        p, v, w, R, J, q, dq = kin()
        Km, Dm = _gains(st["K"], st["K_lat"], a)
        e_r = 0.5 * sum(np.cross(R[:, i], R0[:, i]) for i in range(3))
        wrench = np.concatenate([Km @ (st["p_des"] - p) - Dm @ v, k_rot * e_r - 2 * np.sqrt(k_rot * 0.02) * w])
        if g_cache["tau"] is None or g_cache["n"] % gravity_every == 0:
            g_cache["tau"] = np.asarray(sim.get_gravity_torques(body, arm), float)
        g_cache["n"] += 1
        Jt = J.T
        N = np.eye(len(arm)) - Jt @ np.linalg.pinv(Jt)
        tau = Jt @ wrench + N @ (10.0 * (q_rest - q) - 2.0 * dq) + g_cache["tau"]
        sim.set_joint_torques(body, arm, tau)
        if fingers:
            sim.set_joint_torques(body, fingers, [-grip_force] * len(fingers))
        step()
        st["t"] += dt

    def read_state():
        p, v, *_ = kin()
        Km, Dm = _gains(st["K"], st["K_lat"], a)
        f = -(Km @ (st["p_des"] - p) - Dm @ v) + rng.normal(0.0, force_noise, 3)
        width = float(sum(sim.get_joint_state(body, j).position for j in fingers)) if fingers else None
        return TouchState(t=st["t"], ee_pos=p, force=f, gripper_width=width)

    def send(cmd):
        st["p_des"], st["K"] = np.asarray(cmd.ee_target, float), cmd.k_axial
        st["K_lat"] = cmd.k_lateral or ctrl.cfg.k_lat_search

    n_per = max(1, int(round(period_s / dt)))

    def advance(_period):
        for _ in range(n_per):
            torque_step()

    p_start = kin()[0]
    st["p_des"] = p_start.copy()
    if log:
        log(f"[touch.sim] start tcp={np.round(p_start, 4).tolist()} axis={np.round(a, 3).tolist()} "
            f"tool_offset={np.round(tool_offset, 4).tolist()} grip={grip_force}N dt={dt:.4f}s x{n_per}/tick")
    try:
        from touch.insertion import run_reactive
        res = run_reactive(ctrl, read_state, send, advance, max_s=max_s, period_s=n_per * dt, on_tick=on_tick)
    finally:
        # back to position control where we are, so the next trajectory starts from rest
        q_now = [sim.get_joint_state(body, j).position for j in arm]
        sim.set_joint_positions(body, arm, q_now)
        if fingers:
            f_now = [sim.get_joint_state(body, j).position for j in fingers]
            sim.set_joint_positions(body, fingers, f_now)
    end = read_state()
    res.update(advance_mm=round(1000 * float(np.dot(end.ee_pos - p_start, a)), 2),
               tip_pos=np.round(end.ee_pos, 4).tolist(),
               gripper_width_mm=None if end.gripper_width is None else round(1000 * end.gripper_width, 1),
               felt_axial_N=round(end.axial_resistance(a), 1))
    if log:
        log(f"[touch.sim] done outcome={res['outcome']} in {res['duration_s']}s advance={res['advance_mm']}mm "
            f"grip_w={res['gripper_width_mm']}mm")
    return res

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def run_touch_pull(sim, body, ee_link, arm_joints: Sequence, finger_joints: Sequence,
                   step: Callable[[], None], dt: float, direction, grip_force: float = 30.0,
                   max_s: float = 20.0, period_s: float = 0.02, cfg=None, force_noise: float = 0.5,
                   k_rot: float = 60.0, k_yaw: Optional[float] = None, gravity_every: int = 8, seed: int = 0,
                   log: Optional[Callable[[str], None]] = None, on_tick=None) -> dict:
    """Touch-guided pull of the grasped handle (:class:`touch.pull.PullController`).

    Same impedance law as :func:`run_touch_insert`, but the stiffness frame follows the
    controller's current pull direction ``d`` (stiff-ish along d, compliant across it, stiff in
    z) and the hand orientation target turns about world z by the controller's yaw. The TCP is
    the EE link (the fingertip centre). Fingers squeeze with ``grip_force`` N each.
    """
    from touch.pull import PullController, run_reactive_pull

    if not sim.supports_torque_control():
        return {"outcome": "unsupported", "reason": f"{type(sim).__name__} has no torque control",
                "events": []}
    arm, fingers = list(arm_joints), list(finger_joints)
    rng = np.random.default_rng(seed)
    _, q_link = sim.get_link_pose(body, ee_link)
    R0 = _R_from_xyzw(q_link)
    q_rest = np.array([sim.get_joint_state(body, j).position for j in arm])
    ctrl = PullController(direction, cfg, log=log)
    k_yaw = ctrl.cfg.k_yaw if k_yaw is None else k_yaw
    st = {"p_des": None, "Km": None, "Dm": None, "R_des": R0, "t": 0.0}
    g_cache = {"tau": None, "n": 0}
    # Compliant wrist yaw: a hinged handle turns the hand itself (a stiff yaw fights the door
    # through the grasp); roll/pitch stay stiff so the grasp keeps its shape.
    zz = np.outer([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    Kr = k_rot * (np.eye(3) - zz) + k_yaw * zz
    Dr = 2 * np.sqrt(k_rot * 0.02) * (np.eye(3) - zz) + 2 * np.sqrt(k_yaw * 0.02) * zz

    def gains(cmd, mass=2.0):
        d = np.asarray(cmd.direction, float)
        z = np.array([0.0, 0.0, 1.0])
        h = np.cross(z, d)
        P = [(np.outer(d, d), cmd.k_pull), (np.outer(h, h), cmd.k_lat), (np.outer(z, z), cmd.k_z)]
        return sum(k * M for M, k in P), sum(2 * np.sqrt(k * mass) * M for M, k in P)

    def kin():
        p, qx = sim.get_link_pose(body, ee_link)
        R = _R_from_xyzw(qx)
        J = np.asarray(sim.get_link_jacobian(body, ee_link, arm), float)
        q = np.array([sim.get_joint_state(body, j).position for j in arm])
        dq = np.array([sim.get_joint_state(body, j).velocity for j in arm])
        tw = J @ dq
        return np.asarray(p, float), tw[:3], tw[3:], R, J, q, dq

    def torque_step():
        p, v, w, R, J, q, dq = kin()
        e_r = 0.5 * sum(np.cross(R[:, i], st["R_des"][:, i]) for i in range(3))
        wrench = np.concatenate([st["Km"] @ (st["p_des"] - p) - st["Dm"] @ v,
                                 Kr @ e_r - Dr @ w])
        if g_cache["tau"] is None or g_cache["n"] % gravity_every == 0:
            g_cache["tau"] = np.asarray(sim.get_gravity_torques(body, arm), float)
        g_cache["n"] += 1
        Jt = J.T
        N = np.eye(len(arm)) - Jt @ np.linalg.pinv(Jt)
        tau = Jt @ wrench + N @ (10.0 * (q_rest - q) - 2.0 * dq) + g_cache["tau"]
        sim.set_joint_torques(body, arm, tau)
        st["dbg"] = (tau, wrench, e_r)
        if fingers:
            sim.set_joint_torques(body, fingers, [-grip_force] * len(fingers))
        step()
        st["t"] += dt

    def read_state():
        p, v, *_ = kin()
        f = -(st["Km"] @ (st["p_des"] - p) - st["Dm"] @ v) + rng.normal(0.0, force_noise, 3)
        width = float(sum(sim.get_joint_state(body, j).position for j in fingers)) if fingers else None
        return TouchState(t=st["t"], ee_pos=p, force=f, gripper_width=width)

    def send(cmd):
        st["p_des"] = np.asarray(cmd.ee_target, float)
        st["Km"], st["Dm"] = gains(cmd)
        st["R_des"] = _Rz(cmd.yaw) @ R0

    n_per = max(1, int(round(period_s / dt)))

    def advance(_period):
        for _ in range(n_per):
            torque_step()

    p_start = kin()[0]
    st["p_des"] = p_start.copy()
    st["Km"], st["Dm"] = np.eye(3) * 1000.0, np.eye(3) * 90.0
    for _ in range(int(round(0.1 / dt))):        # let the grip force build before pulling
        torque_step()
    if log:
        log(f"[touch.sim] pull start tcp={np.round(p_start, 4).tolist()} d={np.round(ctrl.d, 3).tolist()} "
            f"grip={grip_force}N dt={dt:.4f}s x{n_per}/tick")
    try:
        res = run_reactive_pull(ctrl, read_state, send, advance, max_s=max_s, period_s=n_per * dt,
                                on_tick=on_tick)
    finally:
        q_now = [sim.get_joint_state(body, j).position for j in arm]
        sim.set_joint_positions(body, arm, q_now)
        if fingers:
            f_now = [sim.get_joint_state(body, j).position for j in fingers]
            sim.set_joint_positions(body, fingers, f_now)
    end = read_state()
    R_turn = _R_from_xyzw(sim.get_link_pose(body, ee_link)[1]) @ R0.T
    res.update(gripper_width_mm=None if end.gripper_width is None else round(1000 * end.gripper_width, 1),
               hand_yaw_deg=round(float(np.degrees(np.arctan2(R_turn[1, 0], R_turn[0, 0]))), 1))
    if log:
        log(f"[touch.sim] pull done outcome={res['outcome']} travel={res['travel_cm']}cm "
            f"swept={res['swept_deg']}deg hand_yaw={res['hand_yaw_deg']}deg grip_w={res['gripper_width_mm']}mm "
            f"in {res['duration_s']}s")
    return res
