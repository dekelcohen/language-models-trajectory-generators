"""Grasp-slip guard: stop a position-controlled trajectory as soon as a held part leaves the hand.

Armed by the first trajectory sample after the gripper closes on something; it lasts across
trajectories until the gripper opens or the guard trips. One sample per waypoint, three
independent checks, all available on a real Franka:

* **width** - Franka Hand width: the fingers closed on air (< ``air_width``) or closed
  ``width_drop`` further than at the grasp (the part slid out between them), or the part forced
  them ``width_rise`` open (it is jammed against the motion). Trips alone.
* **tracking** - camera tracking of the held part (only when a trusted, measured track exists):
  the part moved more than ``track_max_dist`` further from the fingers than at the grasp. Trips alone.
* **touch** - external force from the joint torques (``O_F_ext_hat_K``): a load that built up
  (>= ``load_on``) and then fell below ``load_off_frac`` of its peak while the hand kept moving.
  A door that unlatches also loses load, so this trips only when corroborated by a partial finger
  closing (>= ``partial_drop``) or a tracked part drifting (> half ``track_max_dist``);
  otherwise it is only noted.
* **jam** - the same force, median-filtered, staying above ``jam_load``: the hand is dragging
  against a part that does not move along this path (wrong hinge side / straight pull on an arc).

Returns a verdict dict the agent turns into retry feedback (see ``api._handle_grasp_slip``).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SlipGuardConfig:
    air_width: float = 0.005        # m: fingers (nearly) closed on nothing
    width_drop: float = 0.008       # m closed further than at the grasp (fingers rolling on a round
                                    # lever swing the width by ~6 mm while still holding)
    width_rise: float = 0.02        # m the fingers were forced open (the part pries them: jammed path)
    partial_drop: float = 0.0015    # m: corroborates a load loss
    track_max_dist: float = 0.06    # m the tracked part moved further from the fingers since the grasp
    load_on: float = 8.0            # N felt while carrying / pulling: arms the touch check
    load_off_frac: float = 0.3      # load below this fraction of its peak = released
    min_move: float = 0.01          # m the hand moved since the load fell
    confirm: int = 2                # consecutive samples before a check trips
    jam_load: float = 250.0         # N sustained (median-filtered): the hand is pushing against the
                                    # part, which does not move this way (a real FR3 would reflex-stop)
    jam_confirm: int = 3


class GraspSlipGuard:
    def __init__(self, cfg: Optional[SlipGuardConfig] = None):
        self.cfg = cfg or SlipGuardConfig()
        self.armed = False
        self.notes = []

    def arm(self, width, ee_pos):
        self.armed = True
        self.w0 = float(width)
        self.p0 = np.asarray(ee_pos, float)
        self.p_last = self.p0.copy()
        self.peak = 0.0
        self.peak_pos = None
        self.load_lost_at = None
        self.n_width = self.n_track = self.n_load = self.n_jam = 0
        self.loads = []
        self.notes = []

    def disarm(self):
        self.armed = False

    def _verdict(self, check, reason, p, width, load, track_dist, track_name):
        self.armed = False
        moved = p - self.p0
        return {"check": check, "reason": reason,
                "travel_m": round(float(np.linalg.norm(moved)), 4),
                "direction": np.round(moved / max(np.linalg.norm(moved), 1e-9), 3).tolist(),
                "grasp_pos": np.round(self.p0, 4).tolist(), "slip_pos": np.round(p, 4).tolist(),
                "width_at_grasp_mm": round(self.w0 * 1000, 1),
                "width_mm": None if width is None else round(width * 1000, 1),
                "peak_load_N": round(self.peak, 1), "peak_load_pos": self.peak_pos,
                "load_N": None if load is None else round(load, 1),
                "track_name": track_name,
                "track_dist_m": None if track_dist is None else round(track_dist, 3),
                "notes": list(self.notes)}

    def update(self, ee_pos, width=None, force=None, track_dist=None, track_name=None):
        """One sample. Returns a verdict dict when the part left the hand, else None."""
        if not self.armed:
            return None
        c = self.cfg
        p = np.asarray(ee_pos, float)
        load = None
        if force is not None:                   # median of 3: a new set-point gives a one-sample PD spike
            self.loads = (self.loads + [float(np.linalg.norm(np.asarray(force, float)[:3]))])[-3:]
            load = float(np.median(self.loads))

        if width is not None:
            drop = self.w0 - width
            self.n_width = self.n_width + 1 if (width < c.air_width or drop > c.width_drop or -drop > c.width_rise) else 0
            if self.n_width >= c.confirm:
                why = (f"the fingers closed on air ({width * 1000:.1f} mm)" if width < c.air_width else
                       f"the held part forced the fingers {-drop * 1000:.0f} mm open ({self.w0 * 1000:.1f} -> "
                       f"{width * 1000:.1f} mm): it is jammed against this motion" if -drop > c.width_rise else
                       f"the fingers closed {drop * 1000:.1f} mm further than at the grasp "
                       f"({self.w0 * 1000:.1f} -> {width * 1000:.1f} mm)")
                return self._verdict("width", why, p, width, load, track_dist, track_name)

        if track_dist is not None:
            self.n_track = self.n_track + 1 if track_dist > c.track_max_dist else 0
            if self.n_track >= c.confirm:
                return self._verdict("tracking", f"the camera tracks '{track_name}' {track_dist * 100:.1f} cm "
                                     "from the fingers", p, width, load, track_dist, track_name)

        if load is not None:
            if load > self.peak:
                self.peak, self.peak_pos = load, np.round(p, 4).tolist()
            self.n_jam = self.n_jam + 1 if load > c.jam_load else 0
            if self.n_jam >= c.jam_confirm:
                return self._verdict("jam", f"the hand has been pushing {load:.0f} N against the held part "
                                     f"for {self.n_jam} samples: it does not move along this path",
                                     p, width, load, track_dist, track_name)
            if self.peak >= c.load_on and load < c.load_off_frac * self.peak:
                self.n_load += 1
                if self.load_lost_at is None:
                    self.load_lost_at = p.copy()
                moved = float(np.linalg.norm(p - self.load_lost_at))
                partial = width is not None and (self.w0 - width) >= c.partial_drop
                drifting = track_dist is not None and track_dist > c.track_max_dist / 2
                if self.n_load >= c.confirm and moved >= c.min_move:
                    if partial or drifting:
                        what = "the fingers closed in" if partial else f"'{track_name}' drifted from the fingers"
                        return self._verdict("touch", f"the felt load fell from {self.peak:.1f} N to {load:.1f} N "
                                             f"while the hand kept moving, and {what}",
                                             p, width, load, track_dist, track_name)
                    if not self.notes:
                        self.notes.append(f"load fell {self.peak:.1f} -> {load:.1f} N with the grip unchanged "
                                          "(a latch releasing, or the part now moving freely)")
            else:
                self.n_load, self.load_lost_at = 0, None
        self.p_last = p
        return None
