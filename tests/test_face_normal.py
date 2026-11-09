import math

import numpy as np

from touch.face_normal import face_normal_from_touches, fit_vertical_face


def _turned_box_points(heading_deg, port=(0.1, 0.5, 0.07), step=0.002):
    """Side face through ``port`` with outward normal at ``heading_deg``, plus the device's top deck 5 mm above."""
    n = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))])
    t = np.array([-n[1], n[0]])
    p0 = np.asarray(port[:2])
    face = [[*(p0 + s * t), z] for s in np.arange(-0.08, 0.08, step) for z in np.arange(0.05, 0.075, step)]
    deck = [[*(p0 + s * t - d * n), 0.0755] for s in np.arange(-0.08, 0.08, step) for d in np.arange(0, 0.1, step)]
    return np.asarray(face + deck), n


def test_fit_vertical_face_finds_a_turned_side_and_points_outward():
    pts, n = _turned_box_points(155.0)
    cam = np.array([0.1, 0.5]) + 0.6 * np.array([math.cos(math.radians(185)), math.sin(math.radians(185))])
    res = fit_vertical_face(pts, [0.1, 0.5, 0.07], cam_xy=cam)
    assert res is not None
    assert np.dot(res["normal"][:2], n) > 0.999
    assert np.allclose(res["axis"][:2], -np.asarray(res["normal"][:2]))


def test_face_normal_from_touches_orients_outward():
    n = face_normal_from_touches([0.1, 0.5, 0.07], [0.1 + 0.03 * 0.5, 0.5 + 0.03 * 0.866, 0.07], [-1.0, 0.0, 0.0])
    assert n[0] < 0 and abs(n[0] * 0.5 + n[1] * 0.866) < 1e-3
