"""Touch-guided pull of a grasped handle (doors, drawers, cabinet lids...), proprioception only.

DoorBot-style: no LLM in the loop. The hand pulls with a *force* along a direction ``d`` and is
compliant sideways, so the mechanism itself steers the hand along its constraint (hinge arc or
slide). ``d`` is re-estimated from the hand's own motion; once the path bends, a circle fit gives
the hinge (centre/radius) and ``d`` becomes its tangent. The wrist yaw target follows the swept
angle, but the runner keeps yaw compliant so the handle itself turns the hand (a stiff yaw that
lags the door fights it through the grasp).

Inputs per tick (:class:`TouchState`): TCP position, external force estimate, gripper width.
Outcomes: ``opened`` (target swept angle / travel reached), ``end_stop`` (moved, then blocked),
``blocked`` (never moved), ``slipped`` (handle left the fingers), ``timeout``.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from touch.state import TouchState

_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class PullConfig:
    v_target: float = 0.04          # m/s hand speed along d
    f_start: float = 8.0            # N initial pull force
    f_max: float = 45.0             # N max pull force
    f_gain: float = 400.0           # N per (m/s) speed error per second (force ramp)
    k_pull: float = 400.0           # N/m along d (set-point ahead by f/k)
    k_lat: float = 40.0             # N/m horizontal, perpendicular to d (compliant: the mechanism steers)
    k_z: float = 1500.0             # N/m vertical hold
    dir_window: float = 0.02        # m of travel used to re-estimate d
    dir_blend: float = 0.5          # weight of the new estimate
    fit_min_travel: float = 0.05    # m before the circle fit is trusted
    fit_max_radius: float = 1.5     # m; larger -> treat as straight (drawer / slide)
    fit_min_radius: float = 0.12
    open_angle_deg: float = 65.0    # hinge: stop once swept this much (lever grasps slip past ~70)
    max_travel: float = 0.45        # m path length: stop (drawers, or no hinge found)
    slip_width_drop: float = 0.004  # m the fingers closed further since the start -> handle left
    min_width: float = 0.003        # m fingers (nearly) closed on nothing
    stall_s: float = 0.6            # s at f_max without progress
    stall_travel: float = 0.003     # m progress threshold during stall_s
    end_stop_travel: float = 0.03   # m: stalled after this much travel = end stop, else blocked
    yaw_follow: bool = True
    k_yaw: float = 3.0              # Nm/rad wrist yaw stiffness (runner): low = the handle turns the hand
    hinge: Optional[dict] = None    # {"centre_xy", "radius"} from an earlier pull: used, not refitted


@dataclass
class PullCommand:
    ee_target: np.ndarray
    direction: np.ndarray
    k_pull: float
    k_lat: float
    k_z: float
    yaw: float = 0.0                # rad about world z, relative to the start orientation
    phase: str = "pull"


def fit_circle_xy(pts):
    """Least-squares (Kasa) circle through horizontal points -> (centre_xy, radius, rms) or None."""
    P = np.asarray(pts, float)[:, :2]
    if len(P) < 5:
        return None
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = sol[:2]
    r2 = sol[2] + c @ c
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return c, r, rms


def _hz(v):
    v = np.array(v, float)
    v[2] = 0.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _signed_angle_z(a, b):
    return float(np.arctan2(np.cross(a, b)[2], np.dot(a, b)))


class PullController:
    """Velocity-regulated force pull that follows the mechanism; ``step(state) -> PullCommand``."""

    def __init__(self, direction, cfg: Optional[PullConfig] = None, log=None):
        self.cfg = cfg or PullConfig()
        self.d0 = _hz(direction)
        if np.linalg.norm(self.d0) < 0.5:
            raise ValueError("pull direction must have a horizontal component")
        self.d = self.d0.copy()
        self.log = log
        self.f = self.cfg.f_start
        self.path: List[np.ndarray] = []
        self.t_path: List[float] = []
        self.anchor = None          # last point d was updated from
        self.z0 = None
        self.w0 = None
        self.travel = 0.0
        self.hinge = None           # (centre_xy, radius)
        if self.cfg.hinge:          # resumed pull: a short new arc fits worse than the earlier one
            self.hinge = (np.asarray(self.cfg.hinge["centre_xy"], float)[:2], float(self.cfg.hinge["radius"]))
        self.slip_pos = None
        self.last_loaded = None
        self.phase = "pull"
        self.done = False
        self.outcome = None
        self.events: list = []

    # ------------------------------------------------------------------ helpers
    def _event(self, t, what):
        self.events.append((round(t, 3), self.phase, what))
        if self.log:
            self.log(f"[touch.pull] t={t:.2f} {what}")

    def _finish(self, t, outcome, why):
        self.outcome, self.done = outcome, True
        self._event(t, f"{outcome}: {why}")

    @property
    def swept_deg(self) -> float:
        """Angle swept about the fitted hinge (start -> now); before a fit, how far d has turned."""
        if self.hinge is not None and len(self.path) > 1:
            c = self.hinge[0]
            a, b = self.path[0][:2] - c, self.path[-1][:2] - c
            return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))))
        return float(np.degrees(_signed_angle_z(self.d0, self.d)))

    def _speed(self, now):
        """Hand speed along the path over the last ~0.2 s."""
        if len(self.path) < 2:
            return 0.0
        i = len(self.t_path) - 1
        while i > 0 and now - self.t_path[i] < 0.2:
            i -= 1
        dt = now - self.t_path[i]
        return float(np.linalg.norm(self.path[-1][:2] - self.path[i][:2]) / dt) if dt > 1e-6 else 0.0

    def _update_direction(self, p):
        c = self.cfg
        step = p - self.anchor
        step[2] = 0.0
        if np.linalg.norm(step) < c.dir_window:
            return
        self.anchor = p.copy()
        new = _hz(step)
        if c.hinge:
            tan = _hz(np.cross(_Z, np.r_[p[:2] - self.hinge[0], 0.0]))
            new = tan if np.dot(tan, new) >= 0 else -tan
        elif self.travel >= c.fit_min_travel:
            fit = fit_circle_xy(self.path[::max(1, len(self.path) // 40)])
            if fit is not None:
                cen, r, rms = fit
                if c.fit_min_radius <= r <= c.fit_max_radius and rms < 0.01:
                    tan = _hz(np.cross(_Z, np.r_[p[:2] - cen, 0.0]))
                    if np.dot(tan, new) < 0:
                        tan = -tan
                    if self.hinge is None:
                        self._event(self.t_path[-1], f"hinge fit centre={np.round(cen, 3).tolist()} "
                                                     f"r={r:.3f} rms={rms * 1000:.1f}mm")
                    self.hinge = (cen, r)
                    new = tan
        d = _hz((1 - c.dir_blend) * self.d + c.dir_blend * new)
        if np.dot(d, self.d0) > -0.2:       # never flip back toward the start
            self.d = d

    # ------------------------------------------------------------------ main step
    def step(self, s: TouchState) -> PullCommand:
        c = self.cfg
        p = np.asarray(s.ee_pos, float)
        if self.z0 is None:
            self.z0, self.anchor, self.w0 = p[2], p.copy(), s.gripper_width
            self._event(s.t, f"start d={np.round(self.d, 3).tolist()} w0={None if s.gripper_width is None else round(s.gripper_width * 1000, 1)}mm")
        if self.path:
            self.travel += float(np.linalg.norm(p[:2] - self.path[-1][:2]))
        self.path.append(p.copy())
        self.t_path.append(s.t)

        # slip: the fingers closed further (the handle left them) or nothing is held
        w = s.gripper_width
        if w is not None and self.w0 is not None:
            if w < c.min_width or (self.w0 - w) > c.slip_width_drop:
                self.slip_pos = (self.last_loaded if self.last_loaded is not None else p).tolist()
                self._finish(s.t, "slipped", f"grip width {self.w0 * 1000:.1f}->{w * 1000:.1f}mm after "
                                             f"{self.travel * 100:.1f}cm")
                return self._cmd(p)
            self.last_loaded = p.copy()

        self._update_direction(p)
        v = self._speed(s.t)
        dt = s.t - self.t_path[-2] if len(self.t_path) > 1 else 0.0
        self.f = float(np.clip(self.f + c.f_gain * (c.v_target - v) * dt, 0.0, c.f_max))
        if self.log and s.t - getattr(self, "_t_log", -1.0) >= 1.0:
            self._t_log = s.t
            self.log(f"[touch.pull] t={s.t:.1f} travel={self.travel * 100:.1f}cm v={v * 100:.1f}cm/s "
                     f"f={self.f:.1f}N felt={-np.dot(s.force, self.d):.1f}N w={None if w is None else round(w * 1000, 1)}mm "
                     f"swept={self.swept_deg:.1f}deg")

        if self.hinge is not None and abs(self.swept_deg) >= c.open_angle_deg:
            self._finish(s.t, "opened", f"swept {self.swept_deg:.0f} deg, travel {self.travel * 100:.0f}cm")
        elif self.travel >= c.max_travel:
            self._finish(s.t, "opened", f"travel {self.travel * 100:.0f}cm (swept {self.swept_deg:.0f} deg)")
        elif self.f >= c.f_max - 1e-6:
            i = len(self.t_path) - 1
            while i > 0 and s.t - self.t_path[i] < c.stall_s:
                i -= 1
            if s.t - self.t_path[i] >= c.stall_s and np.linalg.norm(p[:2] - self.path[i][:2]) < c.stall_travel:
                kind = "end_stop" if self.travel >= c.end_stop_travel else "blocked"
                self._finish(s.t, kind, f"{c.f_max:.0f}N without progress after {self.travel * 100:.1f}cm")
        return self._cmd(p)

    def _cmd(self, p):
        c = self.cfg
        target = p + self.d * (self.f / c.k_pull)
        target[2] = self.z0
        yaw = np.radians(self.swept_deg) if (c.yaw_follow and self.hinge is not None) else 0.0
        return PullCommand(target, self.d.copy(), c.k_pull, c.k_lat, c.k_z, yaw, self.phase)

    def result(self) -> dict:
        return {"outcome": self.outcome, "travel_cm": round(self.travel * 100, 1),
                "swept_deg": round(self.swept_deg, 1), "direction": np.round(self.d, 3).tolist(),
                "hinge": None if self.hinge is None else {"centre_xy": np.round(self.hinge[0], 3).tolist(),
                                                          "radius": round(self.hinge[1], 3)},
                "slip_pos": None if self.slip_pos is None else np.round(self.slip_pos, 4).tolist(),
                "end_pos": None if not self.path else np.round(self.path[-1], 4).tolist(),
                "pull_force_N": round(self.f, 1)}


def run_reactive_pull(controller: PullController, read_state, send, advance, max_s=20.0, period_s=0.02,
                      on_tick=None) -> dict:
    """Closed loop read -> step -> send at ``period_s`` until the controller is done or ``max_s``."""
    st = read_state()
    t0, n = st.t, 0
    while True:
        cmd = controller.step(st)
        send(cmd)
        if on_tick:
            on_tick(st, cmd)
        n += 1
        if controller.done:
            break
        if st.t - t0 >= max_s:
            controller._finish(st.t, "timeout", f"{max_s}s")
            break
        advance(period_s)
        st = read_state()
    res = controller.result()
    res.update(duration_s=round(st.t - t0, 3), ticks=n, events=list(controller.events))
    return res
