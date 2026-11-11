"""Touch-guided peg / USB insertion: approach -> probe -> spiral search -> insert.

The controller is a pure state machine: ``step(TouchState) -> InsertCommand``. It reads only
proprioception (EE position from FK, estimated external force) and outputs an EE position
set-point for a Cartesian impedance controller with stiffness ``k_axial`` along the insertion
axis (stiff laterally). Pushing "with force F" is a set-point ``F / k_axial`` ahead of the
current EE position, which is how force is commanded on an impedance-controlled Franka.

Signals it relies on (measured on the Genesis USB bench, docs/plans/touch.md section 3):
* free motion ~1-4 N, USB retention ~12-16 N, a jam on the port face 20+ N with no progress;
* the plug drops ~0.5 mm into the opening when it is laterally aligned (retention then stops it);
* a firm push (> retention, < grip-slip ~24 N) seats an aligned plug.

Sim truth (port pose, insertion depth) is never an input; evaluate it outside.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from touch.state import TouchState

APPROACH, PROBE, SEARCH, INSERT, BACKOFF, VERIFY, DONE = ("approach", "probe", "search", "insert", "backoff",
                                                     "verify", "done")


@dataclass
class InsertConfig:
    v_approach: float = 0.01      # m/s set-point ramp along the axis until contact
    max_approach: float = 0.02    # m of travel without contact -> "no_contact"
    f_contact: float = 3.0        # N axial resistance that counts as contact
    f_search: float = 3.0         # N press while spiralling on the face
    f_insert: float = 18.0        # N push to insert: above retention (~12 N), below grip slip (~24 N)
    f_max: float = 35.0           # N any |F_ext| above this aborts
    k_axial: float = 1500.0       # N/m impedance stiffness along the axis (force -> set-point offset)
    probe_s: float = 0.4          # s of firm push to test whether we are in the opening
    drop_probe: float = 0.002     # m advance during a probe = in the opening
    settle_s: float = 0.2         # s at f_search before the spiral reference is taken
    spiral_pitch: float = 0.0004  # m radial gap between turns (< 2x the port clearance)
    spiral_speed: float = 0.005   # m/s tangential speed
    spiral_r_max: float = 0.003   # m search radius -> "search_exhausted"
    drop_search: float = 0.0003   # m sudden advance while spiralling = opening found
    creep_tau: float = 1.0        # s, slow axial drift (plug creeping in the grip) is absorbed
    stall_speed: float = 0.001    # m/s (kept for callers; the insert is judged by `slow_speed`)
    stall_s: float = 0.3
    # A free insert runs at ~20-40 mm/s. Once the plug stops (seated OR wedged) the grasp creeps on at
    # ~2-5 mm/s under full force (Genesis bench), so speed alone cannot tell them apart; the advance made
    # before the motion first turns slow can: ~12 mm when seated, ~2 mm when wedged on an edge.
    slow_speed: float = 0.006     # m/s
    slow_window: float = 0.2      # s
    # Past `in_slot_depth` the tip is already inside the opening (an edge wedge stops at ~2 mm), so slow
    # sliding against the slot walls is progress: call it a jam only below `in_slot_stall_speed`.
    in_slot_depth: float = 0.004  # m
    in_slot_stall_speed: float = 0.002  # m/s
    seat_depth: float = 0.007     # m of advance past the contact point = seated (judged at stall)
    seat_depth_max: Optional[float] = 0.018   # m; far more advance than the 12 mm tip = it slides in the grasp
    k_lat_search: float = 5000.0  # N/m lateral stiffness while spiralling (track the spiral despite face friction)
    k_lat_insert: float = 800.0   # N/m lateral stiffness while inserting (compliant: the walls centre the plug)
    backoff: float = 0.001        # m retreat after a failed insert
    backoff_s: float = 0.3
    max_inserts: int = 4
    # Seat check: a seated plug cannot move sideways more than its clearance (~0.25 mm), but a plug
    # that pivoted / slid back in the grasp while pressed on the face slides with the hand. After a
    # "seated" stall, command verify_shift sideways (stiff, light axial push) for verify_s; moving
    # more than verify_tol = not in the opening -> "slipped". 0 disables.
    verify_shift: float = 0.002   # m
    verify_s: float = 0.4         # s
    verify_tol: float = 0.0018    # m; a plug in the opening moved 0.27 / 0.87 / 1.51 mm (grasp + port
                                  # compliance; 0.8 mm gave false "slipped" for a fully seated plug,
                                  # harness e2e_u4err + usb u4 2026-11-08), a lost one 2.4 mm
    # Line search instead of the spiral, for an opening known only along a line (e.g. its height
    # on a face, but not where along the face): raster passes +-sweep_half along sweep_dir (world,
    # made perpendicular to the axis) at spiral_speed; rows step sweep_pitch across it
    # (0, +p, -p, +2p, ...; sweep_rows rows).
    sweep_dir: Optional[tuple] = None
    sweep_half: float = 0.02      # m
    sweep_pitch: float = 0.0004   # m
    sweep_rows: float = 5
    # Edge finder (line sweep only). A blocked probe that still advanced >= catch_adv is a "partial
    # catch": the tip half overlaps the opening and tilts in. On the Genesis USB bench they occur where
    # the tip centre crosses the opening's two side edges, i.e. ~one opening width apart. Two catches
    # edge_min_sep..edge_max_sep apart along the line (defaults: USB-A, 12 mm wide) bracket the
    # opening: rasterise columns across the line (+-cross_half) at their midpoint, col_pitch apart,
    # out to +-col_half. Up to max_edge_cycles, then the raster resumes.
    catch_adv: float = 0.0004     # m
    edge_min_sep: float = 0.009   # m
    edge_max_sep: float = 0.016   # m
    cross_half: float = 0.0045    # m
    col_pitch: float = 0.0003     # m (< 2x the port clearance)
    col_half: float = 0.0009      # m
    max_edge_cycles: int = 2
    # A one-way pass catches only the edge it is LEAVING (the far edge, seen in the Genesis harness with a
    # 29 mm aim error: catches at +5.5/+7.8 mm past the centre, never at the near edge), so the two-catch
    # bracket may never form. On a lone catch, guess the centre half an opening width back (against the
    # sweep motion; width ~ the edge_*_sep midpoint) and raster columns there, +-guess_col_half along.
    guess_col_half: float = 0.002  # m
    max_guess_cycles: int = 2      # counted apart from max_edge_cycles


@dataclass
class InsertCommand:
    ee_target: np.ndarray
    axis: np.ndarray
    k_axial: float
    phase: str
    force_cmd: float = 0.0        # N commanded along the axis (0 in position phases)
    k_lateral: Optional[float] = None  # N/m across the axis


def _lateral_basis(a):
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, a)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


class SpiralInsertController:
    """Reactive insertion along ``axis`` (unit, world) from the EE's current pose."""

    def __init__(self, axis, cfg: Optional[InsertConfig] = None, log: Optional[Callable[[str], None]] = None):
        a = np.asarray(axis, float)
        self.a = a / np.linalg.norm(a)
        self.u, self.w = _lateral_basis(self.a)
        self.cfg = cfg or InsertConfig()
        self.log = log
        self.phase = None
        self.done = False
        self.outcome = None
        self.events: List[tuple] = []
        self.n_inserts = 0
        self.theta = 0.0
        self.sweep_s = 0.0
        self.offset = np.zeros(3)
        self._hist: List[tuple] = []
        self.d = self.p = None
        self.mode = "raster"          # line sweep: raster -> columns (after two edge catches) -> raster
        self.catches: List[tuple] = []    # (along, across) of partial catches in the current mode
        self.path: List[np.ndarray] = []
        self.n_cycles = 0
        self.sweep_sign = 1.0         # direction of the current raster pass along the line
        self.n_guesses = 0
        self.col_kind = None          # "edges" | "guess"
        if self.cfg.sweep_dir is not None:
            d = np.asarray(self.cfg.sweep_dir, float)
            d = d - self.a * float(d @ self.a)
            self.d = d / np.linalg.norm(d)
            self.p = np.cross(self.a, self.d)

    # ------------------------------------------------------------------ helpers
    def _event(self, t, msg):
        self.events.append((round(t, 3), self.phase, msg))
        if self.log:
            self.log(f"[touch.insert] t={t:.2f}s {self.phase}: {msg}")

    def _enter(self, phase, t, msg=""):
        self.phase, self.t_phase = phase, t
        self._event(t, f"-> {phase} {msg}".strip())

    def _finish(self, t, outcome, msg):
        self.outcome, self.done = outcome, True
        self._enter(DONE, t, f"{outcome}: {msg}")

    def _s(self, p):
        return float(np.dot(self.a, p))

    def _k_lat(self):
        return self.cfg.k_lat_insert if self.phase == INSERT else self.cfg.k_lat_search

    def _push(self, st, force):
        return InsertCommand(self.center + self.offset + self.a * (self._s(st.ee_pos) + force / self.cfg.k_axial),
                             self.a, self.cfg.k_axial, self.phase, force, self._k_lat())

    def _hold_at(self, s_axial):
        return InsertCommand(self.center + self.offset + self.a * s_axial, self.a, self.cfg.k_axial, self.phase,
                             k_lateral=self._k_lat())

    def _speed(self, t, window):
        return self._speed_between(t - window, t, min_frac=0.5)

    def _speed_between(self, t0, t1, min_frac=0.8):
        pts = [h for h in self._hist if t0 - 1e-9 <= h[0] <= t1 + 1e-9]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < min_frac * (t1 - t0):
            return None
        return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])

    def _xy(self, off):
        return float(off @ self.d), float(off @ self.p)

    def _pt(self, along, across):
        return along * self.d + across * self.p

    def _catch(self, t, adv):
        """A blocked push that still advanced: record it as an edge of the opening (line sweep)."""
        c = self.cfg
        if self.d is None or adv < c.catch_adv:
            return
        along, across = self._xy(self.offset)
        self.catches.append((along, across))
        self._event(t, f"partial catch {1000 * adv:.2f}mm at along={1000 * along:.2f} across={1000 * across:.2f}mm")
        if self.mode != "raster" or self.n_cycles >= c.max_edge_cycles:
            return
        lo, hi = min(a for a, _ in self.catches), max(a for a, _ in self.catches)
        col_half = c.col_half
        if c.edge_min_sep <= hi - lo <= c.edge_max_sep:
            ca, self.col_kind = 0.5 * (lo + hi), "edges"
            self._event(t, f"edges along the line at {1000 * lo:.2f} / {1000 * hi:.2f}mm -> centre "
                           f"{1000 * ca:.2f}mm, column raster +-{1000 * c.col_half:.1f} x +-{1000 * c.cross_half:.1f}mm")
        elif len(self.catches) == 1 and c.guess_col_half > 0 and self.n_guesses < c.max_guess_cycles:
            ca = along - self.sweep_sign * 0.25 * (c.edge_min_sep + c.edge_max_sep)
            col_half, self.col_kind = c.guess_col_half, "guess"
            self.n_guesses += 1
            self._event(t, f"lone catch = the edge being left -> centre guess {1000 * ca:.2f}mm (half a width "
                           f"back), column raster +-{1000 * col_half:.1f} x +-{1000 * c.cross_half:.1f}mm")
        else:
            return
        self.mode, self.catches, self.path = "columns", [], []
        n = int(round(col_half / c.col_pitch))
        for i, j in enumerate([0] + [s * k for k in range(1, n + 1) for s in (1, -1)]):
            h = c.cross_half if i % 2 == 0 else -c.cross_half
            self.path += [self._pt(ca + j * c.col_pitch, across + h), self._pt(ca + j * c.col_pitch, across - h)]

    def _follow(self, dt):
        step, off = self.cfg.spiral_speed * dt, self.offset.copy()
        while self.path and step > 0:
            v = self.path[0] - off
            n = float(np.linalg.norm(v))
            if n <= step:
                off, step = self.path.pop(0), step - n
            else:
                off, step = off + v / n * step, 0.0
        return off

    def _spiral_offset(self, dt):
        c = self.cfg
        if self.mode == "columns":
            if self.path:
                return 0.0, self._follow(dt)
            if self.col_kind == "guess":
                self._event(self.t_prev, f"centre guess {self.n_guesses} found nothing, resume the raster")
            else:
                self.n_cycles += 1
                self._event(self.t_prev, f"edge cycle {self.n_cycles} found nothing, resume the raster")
            self.mode, self.catches = "raster", []
        if self.d is not None:
            # 0 -> +L, then rows of 2L alternating direction, each shifted across the line
            L = c.sweep_half
            self.sweep_s += c.spiral_speed * dt
            if self.sweep_s < L:
                self.sweep_sign = 1.0
                return 0.0, self.sweep_s * self.d
            row, q = divmod(self.sweep_s - L, 2 * L)
            row = int(row)
            if row >= int(c.sweep_rows):
                return c.spiral_r_max + 1.0, self.offset
            x = L - q if row % 2 == 0 else -L + q
            self.sweep_sign = -1.0 if row % 2 == 0 else 1.0
            k = 0 if row == 0 else ((row + 1) // 2) * (1 if row % 2 else -1)
            return 0.0, x * self.d + k * c.sweep_pitch * self.p
        r = c.spiral_pitch * self.theta / (2 * np.pi)
        self.theta += c.spiral_speed * dt / max(r, c.spiral_pitch)
        r = c.spiral_pitch * self.theta / (2 * np.pi)
        return r, r * (np.cos(self.theta) * self.u + np.sin(self.theta) * self.w)

    # ------------------------------------------------------------------ main
    def step(self, st: TouchState) -> InsertCommand:
        c, t, s = self.cfg, st.t, self._s(st.ee_pos)
        if self.phase is None:
            p0 = np.asarray(st.ee_pos, float)
            self.center = p0 - self.a * self._s(p0)
            self.s_start, self.t_prev = s, t
            self._enter(APPROACH, t, f"axis={np.round(self.a, 3).tolist()}")
        dt, self.t_prev = max(t - self.t_prev, 0.0), t
        self._hist = [h for h in self._hist if h[0] >= t - 2.0] + [(t, s)]
        if self.done:
            return self._hold_at(s)

        f_axial = st.axial_resistance(self.a)
        if np.linalg.norm(st.force) > c.f_max:
            self._finish(t, "over_force", f"|F|={np.linalg.norm(st.force):.1f}N")
            return self._hold_at(s - c.backoff)

        if self.phase == APPROACH:
            if f_axial > c.f_contact:
                self.s_contact = s
                self._event(t, f"contact F={f_axial:.1f}N after {1000 * (s - self.s_start):.1f}mm")
                self._enter(PROBE, t)
                return self._push(st, c.f_insert)
            if s - self.s_start > c.max_approach:
                self._finish(t, "no_contact", f"travelled {1000 * (s - self.s_start):.1f}mm")
                return self._hold_at(s)
            return self._hold_at(self.s_start + c.v_approach * (t - self.t_phase))

        if self.phase == PROBE:
            adv = s - self.s_contact
            if adv > c.drop_probe:
                self._enter(INSERT, t, f"probe advanced {1000 * adv:.1f}mm")
                return self._push(st, c.f_insert)
            if t - self.t_phase >= c.probe_s:
                self._event(t, f"probe blocked: adv {1000 * adv:.2f}mm F={f_axial:.1f}N "
                               f"lat={np.linalg.norm(st.lateral_force(self.a)):.1f}N")
                self._catch(t, adv)
                self._enter(SEARCH, t, f"theta={self.theta:.1f}")
                return self._push(st, c.f_search)
            return self._push(st, c.f_insert)

        if self.phase == SEARCH:
            if t - self.t_phase < c.settle_s:
                self.s_ref, self.drop_ticks = s, 0
                return self._push(st, c.f_search)
            if s - self.s_ref > c.drop_search:
                self.drop_ticks += 1
                if self.drop_ticks >= 2:
                    self.s_contact = self.s_ref
                    self._event(t, f"drop {1000 * (s - self.s_ref):.2f}mm at r="
                                   f"{1000 * np.linalg.norm(self.offset):.2f}mm")
                    self._enter(PROBE, t)
                    return self._push(st, c.f_insert)
            else:
                self.drop_ticks = 0
                # follow slow creep up, and any retreat at once
                self.s_ref = s if s < self.s_ref else self.s_ref + min(dt / c.creep_tau, 1.0) * (s - self.s_ref)
            r, self.offset = self._spiral_offset(dt)
            if r > c.spiral_r_max:
                done = (f"sweep +-{1000 * c.sweep_half:.0f}mm done" if self.d is not None else f"r={1000 * r:.1f}mm")
                self._finish(t, "search_exhausted", done)
                return self._hold_at(s - c.backoff)
            return self._push(st, c.f_search)

        if self.phase == INSERT:
            depth = s - self.s_contact
            if c.seat_depth_max is not None and depth > c.seat_depth_max:
                self._finish(t, "slipped", f"advance {1000 * depth:.1f}mm exceeds the plug length: in-grasp slip")
                return self._hold_at(s - c.backoff)
            v = self._speed_between(t - c.slow_window, t)
            slow = c.slow_speed if depth < c.in_slot_depth or depth >= c.seat_depth else c.in_slot_stall_speed
            if t - self.t_phase > c.slow_window + 0.05 and v is not None and v < slow:
                if depth >= c.seat_depth:
                    self.seat_msg = f"advance {1000 * depth:.1f}mm then slow {1000 * v:.1f}mm/s F={f_axial:.1f}N"
                    if c.verify_shift <= 0:
                        self._finish(t, "seated", self.seat_msg)
                        return self._hold_at(s)
                    self.verify_p0, self.verify_off0 = np.asarray(st.ee_pos, float), self.offset.copy()
                    self.offset = self.offset + c.verify_shift * self.u
                    self._enter(VERIFY, t, f"{self.seat_msg}; sideways check {1000 * c.verify_shift:.1f}mm")
                    return self._push(st, c.f_search)
                self.n_inserts += 1
                lat = st.lateral_force(self.a)
                self._event(t, f"stopped early at {1000 * depth:.1f}mm ({1000 * v:.1f}mm/s, jam/edge) "
                               f"F={f_axial:.1f}N lat={np.round(lat, 1).tolist()} "
                               f"off={np.round(1000 * self.offset, 2).tolist()}mm (insert #{self.n_inserts})")
                if self.n_inserts >= c.max_inserts:
                    self._finish(t, "jammed", f"{self.n_inserts} stalled inserts")
                    return self._hold_at(s - c.backoff)
                self._catch(t, depth)
                self._enter(BACKOFF, t)
                self.s_backoff = s - c.backoff
                return self._hold_at(self.s_backoff)
            return self._push(st, c.f_insert)

        if self.phase == VERIFY:
            if t - self.t_phase < c.verify_s:
                return self._push(st, c.f_search)
            dp = np.asarray(st.ee_pos, float) - self.verify_p0
            lat = float(np.linalg.norm(dp - self.a * float(dp @ self.a)))
            self.offset = self.verify_off0
            if lat <= c.verify_tol:
                self._finish(t, "seated", f"{self.seat_msg}; sideways check moved {1000 * lat:.2f}mm")
                return self._hold_at(s)
            self._finish(t, "slipped", f"sideways check moved {1000 * lat:.2f}mm: the part is not in the "
                                       f"opening (it slid / pivoted in the grasp)")
            return self._hold_at(s - c.backoff)

        if self.phase == BACKOFF:
            if t - self.t_phase >= c.backoff_s:
                self._enter(SEARCH, t, f"theta={self.theta:.1f}")
                return self._push(st, c.f_search)
            return self._hold_at(self.s_backoff)

        return self._hold_at(s)


def run_reactive(controller, read_state: Callable[[], TouchState], send_command: Callable[[InsertCommand], None],
                 advance: Callable[[float], None], max_s: float = 15.0, period_s: float = 0.02,
                 on_tick: Optional[Callable[[TouchState, InsertCommand], None]] = None) -> dict:
    """Closed loop at ``period_s``: read -> controller.step -> send, until done or ``max_s``.

    ``advance(period_s)`` steps the simulator (or sleeps on a real robot while the 1 kHz
    impedance loop tracks the last command). Exits: the controller's own outcomes
    (seated / jammed / search_exhausted / no_contact / over_force) or "timeout".
    """
    st = read_state()
    t0, n = st.t, 0
    while True:
        cmd = controller.step(st)
        send_command(cmd)
        if on_tick:
            on_tick(st, cmd)
        n += 1
        if controller.done:
            break
        if st.t - t0 >= max_s:
            controller.outcome, controller.done = "timeout", True
            controller.events.append((round(st.t, 3), controller.phase, "timeout"))
            break
        advance(period_s)
        st = read_state()
    return {"outcome": controller.outcome, "duration_s": round(st.t - t0, 3), "ticks": n,
            "events": list(controller.events)}
