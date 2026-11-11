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




@pytest.mark.parametrize("hole", [(0.015, 0.0), (-0.012, 0.0006)])
def test_line_sweep_finds_far_opening_along_the_face(hole):
    port = MockPort(hole=hole)
    ctrl = SpiralInsertController([1.0, 0.0, 0.0], InsertConfig(sweep_dir=(0.0, 1.0, 0.0), sweep_half=0.02))
    res = run_reactive(ctrl, port.read, port.send, port.advance, max_s=60.0, period_s=0.02)
    assert res["outcome"] == "seated", res["events"]
    assert port.p[0] > 0.009


def test_line_sweep_exhausts_beyond_its_half_length():
    port = MockPort(hole=(0.03, 0.0))
    ctrl = SpiralInsertController([1.0, 0.0, 0.0], InsertConfig(sweep_dir=(0.0, 1.0, 0.0), sweep_half=0.01))
    res = run_reactive(ctrl, port.read, port.send, port.advance, max_s=60.0, period_s=0.02)
    assert res["outcome"] == "search_exhausted"


@pytest.mark.parametrize("rows,expected", [(5, "search_exhausted"), (11, "seated")])
def test_line_sweep_rows_cover_height_error(rows, expected):
    # opening 1.6 mm off across the line: outside the default +-0.8 mm band, inside +-2 mm
    port = MockPort(hole=(0.004, 0.0016))
    cfg = InsertConfig(sweep_dir=(0.0, 1.0, 0.0), sweep_half=0.01, sweep_rows=rows)
    res = run_reactive(SpiralInsertController([1.0, 0.0, 0.0], cfg), port.read, port.send, port.advance,
                       max_s=120.0, period_s=0.02)
    assert res["outcome"] == expected, res["events"]


class EdgePort(MockPort):
    """Like MockPort, but a tip centred within 0.6 mm of an opening edge (while the tip still overlaps
    the opening) tilts and dips 1 mm in without entering: the partial catch seen on the Genesis bench."""

    HALF = (0.00625, 0.00255)   # opening half-size (y, z)

    def dip(self):
        d = np.abs(self.p[1:] - self.hole)
        on_y_edge = abs(d[0] - self.HALF[0]) < 0.0006 and d[1] < 0.0045
        on_z_edge = abs(d[1] - self.HALF[1]) < 0.0006 and d[0] < 0.012
        return 0.001 if on_y_edge or on_z_edge else 0.0

    def advance(self, dt):
        if self.aligned() or self.p[0] > self.FACE + 0.0015:
            return MockPort.advance(self, dt)
        self.t += dt
        self.p[1:] = self.target[1:]
        self.p[0] = min(self.target[0], self.FACE + self.dip())


@pytest.mark.parametrize("max_edge_cycles,expected", [(2, "seated"), (0, "search_exhausted")])
def test_edge_catches_locate_an_off_row_opening(max_edge_cycles, expected):
    # opening 2 mm off the sweep line (outside the +-0.8 mm rows): only its edge catches find it
    cfg = InsertConfig(sweep_dir=(0.0, 1.0, 0.0), sweep_half=0.012, max_edge_cycles=max_edge_cycles)
    port = EdgePort(hole=(0.0021, 0.002))
    res = run_reactive(SpiralInsertController([1.0, 0.0, 0.0], cfg), port.read, port.send, port.advance,
                       max_s=150.0, period_s=0.02)
    assert res["outcome"] == expected, res["events"]
    if expected == "seated":
        assert any("edges along the line" in e[2] for e in res["events"]), res["events"]


class SlipPort(MockPort):
    """The plug never finds the opening: above SLIP N it slides back in the grasp, so the hand advances
    ~12 mm until the fingers hit the face, like a seated plug. Pressed on the face, it still slides sideways."""

    SLIP = 15.0

    def advance(self, dt):
        self.t += dt
        self.p[1:] = self.target[1:]
        x = self.FACE if self.p[0] <= self.FACE else self.p[0]
        if self.k * (self.target[0] - x) > self.SLIP:
            x = self.target[0] - self.SLIP / self.k
        self.p[0] = min(x, self.BOTTOM)


def test_grasp_slip_that_looks_seated_is_caught_by_the_sideways_check():
    res, _ = _run(SlipPort(hole=(0.0, 0.05)))
    assert res["outcome"] == "slipped", res["events"]
    res, _ = _run(SlipPort(hole=(0.0, 0.05)), verify_shift=0.0)
    assert res["outcome"] == "seated"   # without the check the slip passes as seated

def test_insert_with_touch_with_tuning_args_does_not_shadow_the_config_module():
    """USB run 2026-11-11 p14: the plug seated, then insert_with_touch crashed because its local
    tuning dict was named `config` and hid the config module (config.FINGER_PAD_HALF_WIDTH)."""
    import logging
    import types
    from api import API
    sent = []

    class _Conn:
        def send(self, m):
            sent.append(m)

        def recv(self):
            return "ok", {"outcome": "seated", "advance_mm": 12.0, "events": []}

    api = API.__new__(API)
    api.args = types.SimpleNamespace(touch=True)
    api.main_connection = _Conn()
    api.logger = logging.getLogger("test")
    api._record_touch = lambda *a: None
    api._budget_tick = lambda *a: None
    res = api.insert_with_touch([1, 0, 0], tool_offset=[0.025, 0, 0], insert_force=18.0,
                                opening_width=0.012, search_line=[0, 1, 0], search_half=0.04)
    assert res["outcome"] == "seated" and "finger_clearance_m" in res
    cfg = sent[0][1]["config"]
    assert cfg["edge_min_sep"] == 0.75 * 0.012 and cfg["sweep_half"] == 0.04
