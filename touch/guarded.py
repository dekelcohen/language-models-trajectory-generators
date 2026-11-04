"""Guarded move: advance along a direction until a contact force is felt (proprioception only).

The standard real-robot primitive for touch calibration and contact checks (libfranka: Cartesian
impedance + the estimated external wrench ``O_F_ext_hat_K``). The set-point ramps along
``direction`` at ``speed``; contact = the resistance along the direction exceeds ``f_contact`` for 2
ticks. The arm then presses lightly for ``settle_s`` so the reported position is where the part
actually rests on the surface (the impedance set-point is ahead of it), and stops there.

Typical uses: lower a held part onto a flat surface to measure where its bottom is relative to the
hand (in-hand offset calibration); push a door or drawer front until it is touched; check whether a
part is where vision says it is.
"""

from typing import Callable, List, Optional

import numpy as np

from touch.insertion import InsertCommand
from touch.state import TouchState

MOVE, PRESS, DONE = "move", "press", "done"


class GuardedMoveController:
    def __init__(self, direction, max_travel=0.03, f_contact=4.0, speed=0.01, k=1500.0, k_lateral=5000.0,
                 settle_s=0.15, f_max=35.0, log: Optional[Callable[[str], None]] = None):
        d = np.asarray(direction, float)
        self.a = d / np.linalg.norm(d)
        self.max_travel, self.f_contact, self.speed = float(max_travel), float(f_contact), float(speed)
        self.k, self.k_lat, self.settle_s, self.f_max = float(k), float(k_lateral), float(settle_s), float(f_max)
        self.log = log
        self.phase, self.done, self.outcome = None, False, None
        self.events: List[tuple] = []
        self.contact_pos = None
        self.contact_force = None
        self._hits = 0

    def _event(self, t, msg):
        self.events.append((round(t, 3), self.phase, msg))
        if self.log:
            self.log(f"[touch.guarded] t={t:.2f}s {self.phase}: {msg}")

    def _cmd(self, target):
        return InsertCommand(np.asarray(target, float), self.a, self.k, self.phase, k_lateral=self.k_lat)

    def _s(self, p):
        return float(np.dot(self.a, p))

    def step(self, st: TouchState) -> InsertCommand:
        p = np.asarray(st.ee_pos, float)
        if self.phase is None:
            self.p0, self.t0 = p.copy(), st.t
            self.phase = MOVE
            self._event(st.t, f"dir={np.round(self.a, 3).tolist()} max={1000 * self.max_travel:.0f}mm "
                              f"F>{self.f_contact:.1f}N")
        if self.done:
            return self._cmd(self.hold)
        f = st.axial_resistance(self.a)
        if np.linalg.norm(st.force) > self.f_max:
            self.hold = p - self.a * 0.001
            self._finish(st.t, "over_force", f"|F|={np.linalg.norm(st.force):.1f}N", p, f)
            return self._cmd(self.hold)
        travel = self._s(p) - self._s(self.p0)
        if self.phase == MOVE:
            if f > self.f_contact:
                self._hits += 1
                if self._hits >= 2:
                    self.phase, self.t_press = PRESS, st.t
                    self._event(st.t, f"contact F={f:.1f}N after {1000 * travel:.1f}mm")
            else:
                self._hits = 0
            if self.phase == MOVE:
                if travel > self.max_travel:
                    self.hold = p
                    self._finish(st.t, "no_contact", f"travelled {1000 * travel:.1f}mm", p, f)
                    return self._cmd(self.hold)
                ramp = self.speed * (st.t - self.t0)
                return self._cmd(self.p0 + self.a * ramp)
        # PRESS: light push so the hand rests on the surface, then report and stop there
        if st.t - self.t_press >= self.settle_s:
            self.hold = p
            self._finish(st.t, "contact", f"rests at {np.round(p, 4).tolist()} F={f:.1f}N", p, f)
            return self._cmd(self.hold)
        return self._cmd(p + self.a * (self.f_contact / self.k))

    def _finish(self, t, outcome, msg, p, f):
        self.outcome, self.done = outcome, True
        self.contact_pos, self.contact_force = p.copy(), float(f)
        self.phase = DONE
        self._event(t, f"{outcome}: {msg}")
