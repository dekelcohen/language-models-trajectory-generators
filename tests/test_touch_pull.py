"""touch.pull.PullController on a toy mechanism (hinge arc or slide), no simulator."""
from types import SimpleNamespace

import numpy as np

from touch.pull import PullConfig, PullController, fit_circle_xy, run_reactive_pull
from touch.state import TouchState

DT = 0.02


class Mechanism:
    """The hand is fixed to a handle that can only move along a hinge arc or a straight slide.

    Speed along the constraint grows with the commanded force projected on it, minus a resisting
    force; ``limit`` is the end stop (rad for hinges, m for slides); ``slip_at`` drops the grip width.
    """

    def __init__(self, hinge=None, radius=0.4, start_angle=0.0, axis=(-1.0, 0.0), resist=6.0,
                 limit=None, slip_at=None, z=0.7):
        self.hinge, self.r, self.a = (None if hinge is None else np.asarray(hinge, float)), radius, start_angle
        self.axis, self.s = np.asarray(axis, float) / np.linalg.norm(axis), 0.0
        self.resist, self.limit, self.slip_at, self.z = resist, limit, slip_at, z
        self.t, self.travel, self.width, self.f = 0.0, 0.0, 0.035, np.zeros(3)

    def pos(self):
        if self.hinge is not None:
            return np.r_[self.hinge + self.r * np.array([np.cos(self.a), np.sin(self.a)]), self.z]
        return np.r_[self.s * self.axis, self.z]

    def tangent(self):
        if self.hinge is not None:
            return np.array([-np.sin(self.a), np.cos(self.a)])
        return self.axis

    def read(self):
        return TouchState(t=self.t, ee_pos=self.pos(), force=self.f, gripper_width=self.width)

    def send(self, cmd):
        p = self.pos()
        F = cmd.k_pull * np.dot(np.asarray(cmd.ee_target) - p, cmd.direction) * np.asarray(cmd.direction)
        self.f = -F
        self.ft = float(np.dot(F[:2], self.tangent()))

    def advance(self, dt):
        v = max(0.0, self.ft - self.resist) * 0.004
        ds = v * dt
        if self.hinge is not None:
            da = ds / self.r
            if self.limit is not None:
                da = min(da, self.limit - self.a)
            self.a += da
            ds = da * self.r
        else:
            if self.limit is not None:
                ds = min(ds, self.limit - self.s)
            self.s += ds
        self.travel += ds
        self.t += dt
        if self.slip_at is not None and self.travel >= self.slip_at:
            self.width = 0.028


def run(mech, direction, **cfg):
    ctrl = PullController(direction, PullConfig(**cfg))
    return run_reactive_pull(ctrl, mech.read, mech.send, mech.advance, max_s=30.0, period_s=DT)


def test_fit_circle_recovers_hinge():
    a = np.linspace(0.0, 0.5, 20)
    pts = np.c_[0.3 + 0.4 * np.cos(a), -0.2 + 0.4 * np.sin(a), np.full(20, 0.7)]
    cen, r, rms = fit_circle_xy(pts)
    assert np.allclose(cen, [0.3, -0.2], atol=1e-6) and abs(r - 0.4) < 1e-6 and rms < 1e-6


def test_hinge_arc_followed_and_fitted_until_open_angle():
    # start pulling along the initial tangent; the arc bends ~70 deg, a fixed direction would bind
    mech = Mechanism(hinge=(0.0, 0.0), radius=0.4, start_angle=0.0)
    res = run(mech, direction=[0.0, 1.0, 0.0], max_travel=1.0)
    assert res["outcome"] == "opened"
    assert np.degrees(mech.a) >= 63
    assert res["hinge"] is not None
    assert np.allclose(res["hinge"]["centre_xy"], [0.0, 0.0], atol=0.03)
    assert abs(res["hinge"]["radius"] - 0.4) < 0.03


def test_skewed_start_direction_reports_true_swept_angle():
    # initial d 30 deg off the tangent: swept must come from the hinge fit, not from d0
    mech = Mechanism(hinge=(0.0, 0.0), radius=0.4, start_angle=0.0)
    d0 = [np.sin(np.radians(30)), np.cos(np.radians(30)), 0.0]
    res = run(mech, direction=d0, max_travel=1.0, open_angle_deg=50)
    assert res["outcome"] == "opened"
    assert abs(res["swept_deg"] - np.degrees(mech.a)) < 3
    assert 48 <= np.degrees(mech.a) <= 56


def test_drawer_slide_opens_by_travel_without_hinge():
    mech = Mechanism(axis=(-1.0, 0.0))
    res = run(mech, direction=[-1.0, 0.0, 0.0], max_travel=0.25)
    assert res["outcome"] == "opened" and res["hinge"] is None
    assert mech.s >= 0.24


def test_slip_detected_from_grip_width():
    mech = Mechanism(hinge=(0.0, 0.0), radius=0.4, slip_at=0.10)
    res = run(mech, direction=[0.0, 1.0, 0.0])
    assert res["outcome"] == "slipped"
    assert res["slip_pos"] is not None
    assert np.linalg.norm(np.asarray(res["slip_pos"]) - mech.pos()) < 0.02


def test_resumed_pull_uses_prior_hinge_from_the_start():
    # after a re-grasp at 50 deg: the earlier fit steers at once and sets the swept angle
    mech = Mechanism(hinge=(0.0, 0.0), radius=0.4, start_angle=np.radians(50))
    prior = {"centre_xy": [0.01, -0.01], "radius": 0.41}
    res = run(mech, direction=[-1.0, 0.0, 0.0], open_angle_deg=15, hinge=prior)
    assert res["outcome"] == "opened" and res["hinge"] == prior
    assert 63 <= np.degrees(mech.a) <= 68


def test_blocked_vs_end_stop():
    blocked = run(Mechanism(axis=(-1.0, 0.0), resist=1e3), direction=[-1.0, 0.0, 0.0])
    assert blocked["outcome"] == "blocked" and blocked["travel_cm"] < 3
    stop = run(Mechanism(axis=(-1.0, 0.0), limit=0.12), direction=[-1.0, 0.0, 0.0])
    assert stop["outcome"] == "end_stop" and 11 <= stop["travel_cm"] <= 13


def test_open_with_touch_disabled_without_touch_flag():
    from touch import recovery

    api = SimpleNamespace(args=SimpleNamespace(touch=False), logger=None)
    assert recovery.open_with_touch(api, direction=[-1, 0, 0])["outcome"] == "disabled"


# ---- grasp-slip guard (touch/slip_guard.py) during position-controlled trajectories
from touch.slip_guard import GraspSlipGuard  # noqa: E402


def _guard_run(samples):
    """samples: (x, width, force_N, track_rise) per waypoint, hand moving along +x."""
    g = GraspSlipGuard()
    g.arm(0.018, [0.0, 0.0, 0.7])
    for x, w, f, tr in samples:
        v = g.update([x, 0.0, 0.7], width=w, force=None if f is None else [f, 0, 0],
                     track_dist=tr, track_name=None if tr is None else "handle")
        if v is not None:
            return g, v
    return g, None


def test_slip_guard_trips_when_fingers_close_on_air():
    pts = [(0.01 * i, 0.018, 12.0, None) for i in range(5)] + [(0.05 + 0.01 * i, 0.001, 0.5, None) for i in range(4)]
    _, v = _guard_run(pts)
    assert v["check"] == "width" and abs(v["slip_pos"][0] - 0.06) < 1e-6   # 2nd closed sample


def test_slip_guard_door_unlatch_load_drop_alone_does_not_trip():
    pts = [(0.01 * i, 0.018, 15.0, None) for i in range(5)] + [(0.05 + 0.01 * i, 0.0178, 2.0, None) for i in range(10)]
    g, v = _guard_run(pts)
    assert v is None and g.armed and g.notes


def test_slip_guard_load_drop_with_partial_closing_trips_and_feedback_says_it_resisted():
    from api import API
    pts = [(0.01 * i, 0.018, 15.0, None) for i in range(5)] + [(0.05 + 0.01 * i, 0.016, 1.5, None) for i in range(5)]
    _, v = _guard_run(pts)
    assert v["check"] == "touch" and v["peak_load_N"] == 15.0
    assert "resisted" in API.grasp_slip_feedback(v)


def test_slip_guard_tracked_part_left_the_fingers():
    pts = [(0.01 * i, 0.018, None, 0.005 * i) for i in range(20)]
    _, v = _guard_run(pts)
    assert v["check"] == "tracking" and v["track_dist_m"] > 0.06


def test_slip_guard_jam_on_sustained_load_but_not_on_setpoint_spikes():
    from api import API
    spikes = [(0.01 * i, 0.018, 900.0 if i % 4 == 0 else 85.0, None) for i in range(20)]
    assert _guard_run(spikes)[1] is None
    jam = [(0.001 * i, 0.018, 85.0 if i < 3 else 500.0, None) for i in range(10)]
    _, v = _guard_run(jam)
    assert v["check"] == "jam" and "still held" in API.grasp_slip_feedback(v)


def test_grasp_slip_handler_fails_attempt_but_not_on_end_stop_jam():
    import logging
    import pytest
    from api import API
    stub = SimpleNamespace(grasp_slip_feedback=API.grasp_slip_feedback, logger=logging.getLogger("t"), task=SimpleNamespace(
        failed_task=False, review_reason=None, review_improvement_steps=None))
    with pytest.raises(RuntimeError, match="grasp-slip guard"):
        API._handle_grasp_slip(stub, {"check": "width", "reason": "air", "travel_m": 0.02})
    assert stub.task.failed_task and "GRASP SLIP" in stub.task.review_reason
    stub.task.failed_task = False
    with pytest.raises(RuntimeError, match="end stop"):
        API._handle_grasp_slip(stub, {"check": "jam", "reason": "stuck", "travel_m": 0.4, "peak_load_N": 900})
    assert not stub.task.failed_task
