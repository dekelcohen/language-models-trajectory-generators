import numpy as np

from touch.guarded import GuardedMoveController
from touch.state import TouchState


def _run(ctrl, wall_z=None, wall_k=2e4, extra_force=None, dt=0.005, steps=2000):
    """Point hand that follows the set-point; a floor at ``wall_z`` pushes back (+z) when penetrated."""
    p = np.array([0.0, 0.0, 0.10])
    for i in range(steps):
        f = np.zeros(3)
        if wall_z is not None and p[2] < wall_z:
            f[2] = wall_k * (wall_z - p[2])
        if extra_force is not None:
            f = f + extra_force
        cmd = ctrl.step(TouchState(t=i * dt, ee_pos=p.copy(), force=f))
        if ctrl.done:
            break
        target = np.asarray(cmd.ee_target, float)
        if wall_z is not None and target[2] < wall_z:
            # equilibrium between the impedance spring and the wall
            target[2] = (ctrl.k * target[2] + wall_k * wall_z) / (ctrl.k + wall_k)
        p = target
    return ctrl


def test_contact_reports_surface_position():
    c = _run(GuardedMoveController([0, 0, -1], max_travel=0.05, f_contact=4.0, speed=0.02), wall_z=0.08)
    assert c.outcome == "contact"
    assert abs(c.contact_pos[2] - 0.08) < 0.001
    assert c.contact_force > 4.0


def test_no_contact_stops_after_max_travel():
    c = _run(GuardedMoveController([0, 0, -1], max_travel=0.02, speed=0.02))
    assert c.outcome == "no_contact"
    assert 0.02 <= 0.10 - c.contact_pos[2] < 0.022


def test_over_force_aborts():
    c = _run(GuardedMoveController([0, 0, -1], f_max=20.0), extra_force=np.array([30.0, 0, 0]))
    assert c.outcome == "over_force"


def _api_with_conn(replies):
    import types
    from unittest import mock
    from api import API
    api = API.__new__(API)
    api.args = types.SimpleNamespace(touch=True)
    api.logger = mock.MagicMock()
    api.trajectory_step = 1
    api.task = types.SimpleNamespace(touch_results=[])
    conn = mock.MagicMock()
    conn.recv.side_effect = list(replies)
    api.main_connection = conn
    return api


def test_grasp_info_keeps_closed_hand_top_touch_across_subtasks():
    import config
    touch = ("ok", {"outcome": "contact", "eef_pos": [-0.111, 0.45, 0.0651], "advance_mm": 13.8,
                    "contact_N": 2.7, "gripper_width_mm": -1.3})
    api = _api_with_conn([touch, {"eef_pos": [-0.1107, 0.449, 0.0523]}])
    r = api.move_until_contact([0, 0, -1])
    assert abs(r["finger_bottom_z"] - (0.0651 - config.FINGER_BOTTOM_BELOW_EEF)) < 1e-6
    api.open_gripper()
    api.close_gripper()
    api.task = None                                   # next subtask: the record survives
    info = api.get_grasp_info()
    assert info["touched_top_z"] == r["finger_bottom_z"]
    assert info["grasp_eef_pos"][2] == 0.0523
    assert abs(info["grasp_depth_below_top"] - (r["finger_bottom_z"] - 0.0523)) < 1e-4


def test_holding_hand_touch_is_not_a_top_touch():
    touch = ("ok", {"outcome": "contact", "eef_pos": [0.0, 0.0, 0.06], "gripper_width_mm": 15.6})
    api = _api_with_conn([touch, {"eef_pos": [0.0, 0.0, 0.06]}])
    api.move_until_contact([0, 0, -1])
    api.close_gripper()
    assert api.get_grasp_info()["touched_top_z"] is None

def test_stalled_trajectory_reports_measured_pose(capsys):
    from api import API
    api = API.__new__(API)
    api.logger = __import__("unittest").mock.MagicMock()
    stalled = api._motion_feedback({"desc": "press lever", "cmd_end": [0.1, 0.0, 0.60], "cmd_pose_end": [0.1, 0.0, 0.66],
                                    "ee_end": [0.1, 0.0, 0.69], "ee_pre_settle": [0.1, 0.0, 0.691],
                                    "holding_end": True})
    assert stalled["stalled"] is True and stalled["eef_pos"] == [0.1, 0.0, 0.69]
    assert stalled["pose"] == [0.1, 0.0, 0.75]   # trajectory frame: fingertips + the 6 cm depth offset
    assert "STALLED" in capsys.readouterr().out
    lagging = api._motion_feedback({"cmd_end": [0, 0, 0.5], "ee_end": [0, 0, 0.6], "ee_pre_settle": [0, 0, 0.7]})
    assert lagging["reached"] is False and lagging["stalled"] is False
    assert "STALLED" not in capsys.readouterr().out
    assert api._motion_feedback({"cmd_end": [0, 0, 0.5], "ee_end": [0, 0, 0.505]})["reached"] is True

def test_closed_gripper_caps_unthrottled_moves():
    import types
    from api import API
    import config
    sent = []
    api = API.__new__(API)
    api.main_connection = types.SimpleNamespace(send=sent.append, recv=lambda: ["ok", 1])
    api.args = types.SimpleNamespace(vis_traj=False)
    api.task = types.SimpleNamespace(trajectory_length=0)
    api.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    api._gripper_closed = True
    api.execute_trajectory([[0, 0, 0.1, 0], [0.6, 0, 0.1, 0]])          # 0.6 m in one 8 ms step
    pts = sent[-1][1].points
    step = max(abs(b[0] - a[0]) for a, b in zip(pts, pts[1:]))
    assert len(pts) > 2 and step <= config.CLOSED_GRIPPER_MAX_SPEED * 0.01


def test_repeated_unreachable_target_stops_loop():
    import types
    import pytest
    from api import API
    api = API.__new__(API)
    api.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    still = {"reached": False, "shortfall_m": 0.0026, "eef_pos": [-0.617, 0.19, 0.536], "pose": [-0.617, 0.19, 0.596]}
    for _ in range(API.REPEAT_NO_PROGRESS_CALLS - 1):
        api._check_repeated_no_progress(types.SimpleNamespace(points=[[0, 0, 0, 0], [-0.6196, 0.1907, 0.596, 0.3]]), still)
    with pytest.raises(RuntimeError, match="did not move"):
        api._check_repeated_no_progress(types.SimpleNamespace(points=[[-0.6196, 0.1907, 0.596, 0.3]]), still)
    # a changing target (e.g. stepwise tilt in place) never trips it
    for i in range(3 * API.REPEAT_NO_PROGRESS_CALLS):
        api._check_repeated_no_progress(types.SimpleNamespace(points=[[0, 0, 0.5, 0.01 * i]]), still)
