"""Outward normal of a vertical device face (laptop side, socket plate) near a point, from depth.

A device may stand at any heading, so the insertion axis of a port on its side is the face normal,
not a world axis. Pure functions of camera-derived points / touch contacts (Sim2Real safe).
"""
import numpy as np


def fit_vertical_face(points, near, cam_xy=None, r=0.06, z_below=0.03, z_above=0.004, inlier_m=0.003,
                      max_tilt=0.3, min_inliers=30, iters=300, seed=0):
    """RANSAC plane through depth ``points`` (N x 3 world) within ``r`` m (horizontally) of ``near`` and
    from ``z_below`` under to ``z_above`` over its height (the device top is excluded), keeping only
    near-vertical planes (|n_z| <= ``max_tilt``).

    Returns ``{"normal": [nx, ny, 0] (outward, toward ``cam_xy`` / the viewer), "axis": -normal
    (insertion direction), "along": horizontal direction along the face, "point": ``near`` projected
    onto the face, "n_inliers", "rms_m"}`` or None (no vertical face there)."""
    P = np.asarray(points, float).reshape(-1, 3)
    c = np.asarray(near, float).ravel()[:3]
    m = (np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < r) & (P[:, 2] > c[2] - z_below) & (P[:, 2] < c[2] + z_above)
    P = P[m]
    if len(P) < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, d = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, d - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) > max_tilt:
            continue
        inl = np.abs((P - a) @ n) < inlier_m
        k = int(inl.sum())
        if best is None or k > best[0]:
            best = (k, inl)
    if best is None or best[0] < min_inliers:
        return None
    Q = P[best[1]]
    mu = Q.mean(axis=0)
    # least squares on the inliers, constrained vertical: a 2D line fit in xy
    xy = Q[:, :2] - mu[:2]
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    n2 = vt[1] / np.linalg.norm(vt[1])
    toward = (np.asarray(cam_xy, float)[:2] - mu[:2]) if cam_xy is not None else -mu[:2]
    if n2 @ toward < 0:
        n2 = -n2
    rms = float(np.sqrt(np.mean((xy @ n2) ** 2)))
    along = np.array([-n2[1], n2[0]])
    if along[np.argmax(np.abs(along))] < 0:
        along = -along
    p = c[:2] - ((c[:2] - mu[:2]) @ n2) * n2
    r3 = lambda v: [round(float(x), 4) for x in v]  # noqa: E731
    return {"normal": r3([n2[0], n2[1], 0.0]), "axis": r3([-n2[0], -n2[1], 0.0]),
            "along": r3([along[0], along[1], 0.0]), "point": r3([p[0], p[1], c[2]]),
            "n_inliers": int(best[0]), "rms_m": round(rms, 4)}


def face_normal_from_touches(p1, p2, outward_hint):
    """Outward face normal from two touch contacts on the same vertical face (world [x, y, z] each,
    a few cm apart along it); ``outward_hint`` = any vector pointing out of the face (e.g. the
    approach direction reversed). Returns [nx, ny, 0]."""
    d = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    if np.linalg.norm(d) < 1e-3:
        raise ValueError("touch points must be >= 1 mm apart along the face")
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if n @ np.asarray(outward_hint, float)[:2] < 0:
        n = -n
    return [round(float(n[0]), 4), round(float(n[1]), 4), 0.0]
