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


def test_measure_door_angle_sees_a_door_that_swung_back_shut():
    plane = estimate_door_plane(np.vstack([_panel([0.0, 0.0], 0.5, 0), _jamb(0.54, 0.64)]),
                                [0.4, -0.06, 0.65], cam_xy=[0.3, -1.0])
    deg, _ = measure_door_angle(np.vstack([_panel([0.0, 0.0], 0.5, 60), _jamb(0.54, 0.64)]), plane, 0.65)
    assert abs(deg - 60) < 4
    deg, _ = measure_door_angle(np.vstack([_panel([0.0, 0.0], 0.5, 2), _jamb(0.54, 0.64)]), plane, 0.65)
    assert deg < 6
