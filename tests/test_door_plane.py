import numpy as np

from touch.door_plane import estimate_door_plane, measure_door_angle


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
