"""Unit tests for touch.insertion with a quasi-static mock port (no simulator, no sim truth to the controller)."""

import numpy as np
import pytest

from touch import InsertConfig, SpiralInsertController, TouchState, run_reactive


class MockPort:
    """EE pushes along +x into a face at x=FACE with an opening at (y, z) = hole (+-tol).
    Aligned: the tip drops 0.5 mm in, then retention friction (12 N) resists until the bottom."""

    FACE, DROP, BOTTOM, RETENTION = 0.0, 0.0005, 0.012, 12.0  # BOTTOM: housing hits the face

    def __init__(self, hole=(0.0, 0.0), tol=0.00025, start_x=-0.005, wall=True):
        self.hole, self.tol, self.wall = np.asarray(hole), tol, wall
        self.p = np.array([start_x, 0.0, 0.0])
        self.target, self.k, self.t = self.p.copy(), 1500.0, 0.0

    def aligned(self):
        return np.all(np.abs(self.p[1:] - self.hole) <= self.tol)

    def read(self):
        f_axial = max(self.k * (self.target[0] - self.p[0]), 0.0)
        return TouchState(t=self.t, ee_pos=self.p.copy(), force=np.array([-f_axial, 0.0, 0.0]))

    def send(self, cmd):
        self.target, self.k = np.asarray(cmd.ee_target, float), cmd.k_axial

    def advance(self, dt):
        self.t += dt
        x_prev = self.p[0]
        self.p[1:] = self.target[1:] if self.p[0] <= self.FACE else self.p[1:]
        x = self.target[0]
        if self.wall:
            if not self.aligned() and x_prev <= self.FACE:
                x = min(x, self.FACE)
            elif x > self.FACE + self.DROP:
                x = max(min(self.target[0] - self.RETENTION / self.k, self.BOTTOM), min(x_prev, x),
                        self.FACE + self.DROP)
            x = min(x, self.BOTTOM)
        self.p[0] = x


def _run(port, **cfg):
    ctrl = SpiralInsertController([1.0, 0.0, 0.0], InsertConfig(**cfg))
    res = run_reactive(ctrl, port.read, port.send, port.advance, max_s=30.0, period_s=0.02)
    return res, port


def test_aligned_plug_seats_without_search():
    res, port = _run(MockPort())
    assert res["outcome"] == "seated"
    assert port.p[0] > 0.009
    assert not any("drop" in e[2] for e in res["events"])


@pytest.mark.parametrize("hole", [(0.0012, 0.0), (-0.0008, 0.0009)])
def test_spiral_finds_offset_opening(hole):
    res, port = _run(MockPort(hole=hole))
    assert res["outcome"] == "seated", res["events"]
    assert any("drop" in e[2] for e in res["events"])
    assert port.p[0] > 0.009


def test_far_opening_exhausts_search():
    res, _ = _run(MockPort(hole=(0.006, 0.0)))
    assert res["outcome"] == "search_exhausted"


def test_free_space_reports_no_contact():
    res, _ = _run(MockPort(wall=False))
    assert res["outcome"] == "no_contact"


def test_advance_beyond_plug_length_reports_slip():
    port = MockPort()
    port.BOTTOM = 0.025   # the hand keeps going: plug sliding in the grasp
    res, _ = _run(port)
    assert res["outcome"] == "slipped"


class WedgePort(MockPort):
    """Face at x=0 held by 12 N retention; past it the plug enters 2.5 mm freely, then wedges and the
    hand only creeps on at 3 mm/s (grasp sliding)."""

    def advance(self, dt):
        self.t += dt
        tx, x0 = self.target[0], self.p[0]
        if tx <= 0.0 or tx < x0:
            x = min(tx, 0.0) if x0 <= 0.0 else tx
        elif tx - self.RETENTION / self.k <= 0.0:
            x = min(tx, 0.0) if x0 <= 0.0 else x0
        else:
            cap = 0.0025 if x0 < 0.0025 else x0 + 0.003 * dt
            x = max(min(tx - self.RETENTION / self.k, cap), min(x0, cap))
        self.p[0] = x


def test_wedged_creep_is_not_seated():
    port = WedgePort()
    res, _ = _run(port, max_inserts=2)
    assert any("stopped early" in e[2] for e in res["events"]), res["events"]
    assert res["outcome"] != "seated"


def test_over_force_aborts():
    port = MockPort(hole=(0.006, 0.0))
    res, _ = _run(port, f_search=40.0, f_max=30.0)
    assert res["outcome"] == "over_force"


