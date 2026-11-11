import numpy as np

from touch.door_plane import estimate_door_plane, hinge_from_swung_panel, measure_door_angle


def _panel(hinge, width, deg, z0=0.4, z1=0.9, step=0.01):
    """Points on a door face hinged at ``hinge`` (closed face along +x, opening toward -y)."""
    a = np.radians(-deg)
    u = np.array([np.cos(a), np.sin(a)])
    pts = [[*(np.asarray(hinge) + s * u), z] for s in np.arange(0.0, width, step)
           for z in np.arange(z0, z1, step)]
    return np.asarray(pts, float)


def _jamb(x0, x1, z0=0.4, z1=0.9, step=0.01):
    return np.asarray([[x, 0.0, z] for x in np.arange(x0, x1, step) for z in np.arange(z0, z1, step)], float)


def test_door_plane_puts_the_hinge_at_the_far_panel_end_not_across_the_gap():
    # panel x 0..0.5 (hinge x=0), latch at x=0.4, 4 cm gap, then a coplanar jamb x 0.54..0.64
    pts = np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64)])
    plane = estimate_door_plane(pts, [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    assert plane and plane["confident"]
    assert np.allclose(plane["pull_dir"][:2], [0, -1], atol=0.02)
    assert abs(plane["hinge"][0]) < 0.03 and abs(plane["free_edge"][0] - 0.5) < 0.03


def _swung(axis, s0, s1, deg, sign=1.0, z0=0.4, z1=0.9, step=0.01):
    """Panel points from ``s0`` to ``s1`` m along the panel from a hinge ``axis``; the closed panel runs
    along ``sign`` * x, opened ``deg`` toward -y (the camera / robot side)."""
    a = np.radians(deg)
    u = np.array([sign * np.cos(a), -np.sin(a)])
    return np.asarray([[*(np.asarray(axis) + s * u), z] for s in np.arange(s0, s1, step)
                       for z in np.arange(z0, z1, step)], float)


def test_door_plane_hinge_is_only_the_visible_panel_end():
    """No frame-gap model: the hinge hypothesis is the visible end, whatever lies past it."""
    pts = np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(-0.14, -0.04)])
    plane = estimate_door_plane(pts, [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    assert plane and abs(plane["hinge"][0]) < 0.03 and plane["hinge_source"] == "panel_end"


def test_hinge_from_swung_panel_finds_an_axis_past_a_frame_gap():
    """The axis 8 cm past the visible panel end (PyBullet door): measured from the panel's motion."""
    closed = np.vstack([_swung([-0.08, 0.0], 0.08, 0.58, 0), _jamb(-0.2, -0.1)])
    plane = estimate_door_plane(closed, [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    opened = np.vstack([_swung([-0.08, 0.0], 0.08, 0.58, 15), _jamb(-0.2, -0.1)])
    h = hinge_from_swung_panel(opened, plane, 0.65)
    assert h and abs(h["hinge"][0] + 0.08) < 0.02 and abs(h["hinge"][1]) < 0.02
    assert abs(h["open_deg"] - 15) < 1.5 and not h["flipped"]


def test_hinge_from_swung_panel_flips_a_wrong_hinge_side():
    """The plane guessed the hinge at x=0, but the door turns about the other end (x=0.52)."""
    closed = _swung([0.52, 0.0], 0.02, 0.52, 0, sign=-1.0)
    plane = estimate_door_plane(closed, [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    assert plane and plane["hinge"][0] < 0.1
    h = hinge_from_swung_panel(_swung([0.52, 0.0], 0.02, 0.52, 12, sign=-1.0), plane, 0.65)
    assert h and h["flipped"] and abs(h["hinge"][0] - 0.52) < 0.03 and h["free_edge"][0] < 0.1


def test_door_plane_finds_the_hinge_from_the_rose_when_given_the_lever_tip():
    """door:hidden r5: the LLM passed the lever's tip (mid-panel); both panel ends were ~equally far."""
    rose = np.asarray([[x, -0.03, z] for x in np.arange(0.40, 0.45, 0.01) for z in np.arange(0.63, 0.68, 0.01)])
    pts = np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64), rose])
    plane = estimate_door_plane(pts, [0.26, -0.07, 0.65], cam_xy=[0.3, -1.0])
    assert plane and abs(plane["hinge"][0]) < 0.03 and plane["confident"]


def test_door_plane_gives_a_mid_lever_grasp_across_the_bar():
    """Lever bar 0.11 m proud of the face, from the rose (x 0.42) 0.2 m toward the hinge, centre z 0.65."""
    rose = np.asarray([[x, -0.03, z] for x in np.arange(0.40, 0.45, 0.01) for z in np.arange(0.63, 0.68, 0.01)])
    bar = np.asarray([[x, -0.11, 0.665] for x in np.arange(0.22, 0.42, 0.005) for _ in range(2)])
    pts = np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64), rose, bar])
    g = estimate_door_plane(pts, [0.3, -0.11, 0.66], cam_xy=[0.3, -1.0])["lever_grasp"]
    assert g and np.allclose(g["xyz"], [0.32, -0.11, 0.65], atol=0.02)
    assert abs(abs(g["rotation"]) - np.pi / 2) < 0.05          # closing axis along y, across the x bar


def test_measure_door_angle_sees_a_door_that_swung_back_shut():
    plane = estimate_door_plane(np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64)]),
                                [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    deg, _ = measure_door_angle(np.vstack([_panel([0.0, 0.0], 0.5, 60), _jamb(0.54, 0.64)]), plane, 0.65)
    assert abs(deg - 60) < 4
    deg, _ = measure_door_angle(np.vstack([_panel([0.0, 0.0], 0.5, 2), _jamb(0.54, 0.64)]), plane, 0.65)
    assert deg < 6


def test_api_keeps_the_wider_closed_face_when_the_arm_occludes_a_re_estimate(monkeypatch):
    """door:hidden r8: a re-estimate with the arm in view saw a 0.26 m panel and moved the hinge in."""
    import logging
    from types import SimpleNamespace
    import touch.door_plane as dp
    from api import API
    full = np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64)])
    shadowed = full[(full[:, 0] > 0.25) | (full[:, 2] > 0.85)]          # arm shadow over x < 0.25
    clouds = iter([full, shadowed])
    monkeypatch.setattr(dp, "depth_to_world", lambda *a, **k: next(clouds))
    fake = SimpleNamespace(_capture_head_image_and_depth=lambda: (None, None), head_camera_position=[0.3, -1.0, 1.0],
                           head_camera_orientation_q=None, cam_info=None, _load_head_robot_mask=lambda: None,
                           logger=logging.getLogger("t"))
    monkeypatch.setattr(dp, "measure_door_angle", lambda *a, **k: (0.0, 300))
    first = API.estimate_door_plane(fake, [0.4, -0.06, 0.65])
    again = API.estimate_door_plane(fake, [0.4, -0.06, 0.65])
    assert abs(first["hinge"][0]) < 0.03 and again["hinge"] == first["hinge"]


def test_lever_grasp_follows_the_door_about_the_hinge():
    """A partly open door: the re-grasp target is the closed-door lever point swung about the hinge toward
    the robot (pull_dir), and the hand rotation turns with it."""
    import math
    from helpers.door_open import lever_grasp_now
    plane = {"hinge": [0.0, 0.0], "free_edge": [0.5, 0.0], "pull_dir": [0.0, -1.0, 0.0],
             "lever_grasp": {"xyz": [0.3, -0.1, 0.65], "rotation": -math.pi / 2}}
    g = lever_grasp_now(plane, 90.0)
    assert np.allclose(g["xyz"], [-0.1, -0.3, 0.65], atol=1e-6)        # swung toward -y (the robot)
    assert abs(abs(g["rotation"])) < 1e-6 or abs(abs(g["rotation"]) - math.pi) < 1e-6


def _open_door_with(monkeypatch, touch_result):
    """Run helpers.door_open.open_door on a fake API; returns (result, calls)."""
    import math
    from types import SimpleNamespace
    from helpers import door_open
    from touch import recovery
    plane = {"hinge": [0.0, 0.0], "free_edge": [0.5, 0.0], "pull_dir": [0.0, -1.0, 0.0], "top_z": 1.0,
             "lever_grasp": {"xyz": [0.3, -0.1, 0.65], "rotation": -math.pi / 2}}
    calls, angle = [], {"deg": 0.0}
    monkeypatch.setattr(door_open, "grasp_door_lever", lambda api, p, door_deg=0.0, offsets=(): calls.append("grasp")
                        or {"holding": True, "xyz": [0.3, -0.1, 0.65], "rotation": -math.pi / 2})
    monkeypatch.setattr(door_open, "_release_and_clear", lambda api: None)
    monkeypatch.setattr(door_open, "probe_hinge", lambda api, p: calls.append("probe")
                        or {"source": "none", "holding": True})

    def touch_pull(api, **kw):
        calls.append(("touch", kw["direction"]))
        if touch_result["outcome"] == "opened":
            angle["deg"] = 70.0
        return touch_result
    monkeypatch.setattr(recovery, "open_with_touch", touch_pull)

    def guarded(direction, hinge=None, **kw):
        calls.append("guarded")
        angle["deg"] = 70.0
        return {"outcome": "opened"}
    api = SimpleNamespace(estimate_door_plane=lambda h: plane, measure_door_angle=lambda: angle["deg"],
                          guarded_pull=guarded, logger=SimpleNamespace(info=lambda m: None),
                          args=SimpleNamespace(touch=True))
    return door_open.open_door(api, [0.3, -0.1, 0.65]), calls


def test_open_door_pulls_with_touch_and_no_hinge_probe(monkeypatch):
    res, calls = _open_door_with(monkeypatch, {"outcome": "opened", "swept_deg": 66.0,
                                               "hinge": {"centre_xy": [0.0, 0.0], "radius": 0.3}})
    assert res["outcome"] == "opened" and res["method"] == "touch_pull"
    assert "probe" not in calls and "guarded" not in calls
    assert calls[1] == ("touch", [0.0, -1.0, 0.0])          # starts along the door normal, toward the robot


def test_open_door_falls_back_to_guarded_pull_without_torque_control(monkeypatch):
    res, calls = _open_door_with(monkeypatch, {"outcome": "unsupported"})
    assert res["outcome"] == "opened" and res["method"] == "guarded_pull"
    assert calls == ["grasp", ("touch", [0.0, -1.0, 0.0]), "probe", "guarded"]   # same grasp, no re-grasp

def test_probe_hinge_turns_a_grasp_slip_into_a_lost_lever_not_a_crash():
    from types import SimpleNamespace
    from helpers import door_open
    state = {"eef_pos": [0.3, -0.1, 0.65], "rotation": 0.0, "gripper_holding": True}

    def execute(traj):
        state["gripper_holding"] = False
        raise RuntimeError("Trajectory stopped by the grasp-slip guard. GRASP SLIP (width check): ...")
    api = SimpleNamespace(get_robot_state=lambda: dict(state), execute_trajectory=execute,
                          guard_motion=lambda *a, **k: None, find_hinge_from_depth=lambda hand_xy: None,
                          logger=SimpleNamespace(info=lambda m: None))
    out = door_open.probe_hinge(api, {"pull_dir": [0.0, -1.0, 0.0]})
    assert out["source"] == "none" and out["holding"] is False      # open_door then re-grasps
