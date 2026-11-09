"""Door plane, pull normal and hinge side from a depth point cloud around the handle.

Pixels + depth only (Sim2Real): the head camera's depth is back-projected to world points
(calibrated intrinsics/extrinsics), and a vertical plane is fitted to the points around the
handle. The door is the largest vertical surface right behind the handle:

* pull direction = the plane normal pointing toward the camera (the robot's side);
* the panel's extent along the plane is grown outward from the handle until a gap in the
  points (so a coplanar wall far away is not merged in);
* hinge = the panel end FARTHER from the rose (the bump 1-6 cm in front of the face near the
  given handle point; handles sit near the free edge). Without a visible rose, the given point.

door:hidden r3 (2026-11-08): the LLM guessed the door plane ~35 deg off and put the hinge on
the wrong side; perception's "door" box was 5 cm wide and the "hinge" box was the handle.
"""

import math

import numpy as np


def depth_to_world(depth, cam_pos, cam_q, cam_info=None, stride=2, robot_mask=None, camera="head"):
    """Back-project every ``stride``-th depth pixel to world points (N, 3) with the calibrated
    camera (``utils.get_world_point_world_frame``); robot-masked pixels are skipped."""
    import utils
    H, W = depth.shape[:2]
    out = []
    for v in range(0, H, stride):
        for u in range(0, W, stride):
            if robot_mask is not None and robot_mask[v, u]:
                continue
            d = float(depth[v, u])
            if not np.isfinite(d) or d <= 0:
                continue
            out.append(np.asarray(utils.get_world_point_world_frame(
                cam_pos, cam_q, camera, (W, H), [u, v, d], cam_info=cam_info), float).ravel()[:3])
    return np.asarray(out, float).reshape(-1, 3)


def estimate_door_plane(points, handle, cam_xy, radius=0.7, z_band=0.25, floor_z=0.08,
                        inlier_m=0.015, max_face_m=0.20, bin_m=0.02, max_gap_m=0.02, min_density=0.3,
                        min_width_m=0.2, iters=300, seed=0, rose_off_m=(0.01, 0.06), rose_radius_m=0.3,
                        min_rose_pts=10):
    """Fit the vertical door plane behind ``handle`` in world ``points`` (N, 3).

    Returns a dict (all xy in world metres) or ``None`` when no vertical plane is found:
    ``pull_dir`` [x, y, 0] (unit, toward ``cam_xy``), ``tangent``, ``hinge`` [x, y] (panel end
    farther from the handle), ``free_edge`` [x, y], ``handle_to_hinge_m``, ``handle_to_free_m``,
    ``width_m``, ``face_offset_m`` (handle in front of the face), ``n_inliers``, ``side_ratio``
    (far / near extent; < ~1.3 means the hinge side is ambiguous) and ``confident``.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    h = np.asarray(handle, float).ravel()[:3]
    hxy = h[:2]
    d_xy = np.linalg.norm(P[:, :2] - hxy, axis=1)
    keep = (d_xy < radius) & (np.abs(P[:, 2] - h[2]) < z_band) & (P[:, 2] > floor_z)
    keep &= np.linalg.norm(P - h, axis=1) > 0.04          # not the lever itself
    P3 = P[keep]
    Q = P3[:, :2]
    if len(Q) < 50:
        return None

    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(Q), 2, replace=False)
        t = Q[j] - Q[i]
        n_t = float(np.linalg.norm(t))
        if n_t < 0.05:
            continue
        t /= n_t
        nrm = np.array([-t[1], t[0]])
        if float(np.dot(np.asarray(cam_xy, float)[:2] - Q[i], nrm)) < 0:
            nrm = -nrm                                     # toward the camera
        off = float(np.dot(hxy - Q[i], nrm))
        if not -0.08 < off <= max_face_m:                  # the face is right behind (or at) the handle
            continue
        n_in = int(np.sum(np.abs((Q - Q[i]) @ nrm) < inlier_m))
        if n_in > best_n:
            best, best_n = (Q[i].copy(), t), n_in
    if best is None or best_n < 30:
        return None

    q0, t = best
    nrm = np.array([-t[1], t[0]])
    inl = Q[np.abs((Q - q0) @ nrm) < inlier_m]
    c = inl.mean(0)                                        # refine: principal direction of inliers
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    t = vt[0] / np.linalg.norm(vt[0])
    nrm = np.array([-t[1], t[0]])
    inl = Q[np.abs((Q - c) @ nrm) < inlier_m]

    to_cam = np.asarray(cam_xy, float)[:2] - c
    if float(np.dot(to_cam, nrm)) < 0:
        nrm = -nrm
    foot = c + float(np.dot(hxy - c, t)) * t               # handle projected onto the face line
    face_off = float(np.dot(hxy - foot, nrm))

    s = (inl - foot) @ t
    kb = np.floor(s / bin_m).astype(int)
    counts = {}
    for k in kb.tolist():
        counts[k] = counts.get(k, 0) + 1
    near_c = [counts.get(k, 0) for k in range(-5, 5)]          # +-10 cm around the handle
    ref = float(np.median([c for c in near_c if c > 0])) if any(near_c) else 1.0
    low = max(1.0, min_density * ref)

    def grow(sign):
        # the panel ends at the first sparse bin: the door-frame gap shows as a density dip even
        # when a coplanar jamb continues past it (PyBullet door: 8 vs ~50 points per 4 cm)
        k, last, miss = 0, 0, 0
        while abs(k) < 200:
            k += sign
            if counts.get(k, 0) >= low:
                last, miss = k, 0
            else:
                miss += 1
                if miss * bin_m >= max_gap_m:
                    break
        return last
    k_pos, k_neg = grow(+1), grow(-1)
    e_pos, e_neg = (k_pos + 1) * bin_m, k_neg * bin_m          # outer bin edges

    if abs(e_pos) + abs(e_neg) < min_width_m:
        return None
    # The rose (where the handle meets the door) is the bump just proud of the face; it sits near the
    # free edge whatever point on the handle was given (door:hidden r5: an anchor at the lever's tip,
    # mid-panel, picked the wrong side; the rose stays ~6 cm from the free edge, harness 2026-11-08).
    off3 = (P3[:, :2] - foot) @ nrm
    s3 = (P3[:, :2] - foot) @ t
    bump = ((off3 > rose_off_m[0]) & (off3 < rose_off_m[1]) & (np.abs(P3[:, 2] - h[2]) < 0.12)
            & (np.linalg.norm(P3[:, :2] - hxy, axis=1) < rose_radius_m) & (s3 > e_neg) & (s3 < e_pos))
    rose_s = float(np.median(s3[bump])) if int(bump.sum()) >= min_rose_pts else None
    ref_s = rose_s if rose_s is not None else 0.0
    far, near = (e_pos, e_neg) if abs(e_pos - ref_s) >= abs(e_neg - ref_s) else (e_neg, e_pos)
    ratio = abs(far - ref_s) / max(abs(near - ref_s), 1e-3)
    # The hinge is only a HYPOTHESIS here: the visible panel end far from the rose. Where the axis really is
    # (frame gaps differ: 6 cm on the PyBullet door, a few mm on real doors) is measured after a short pull
    # by hinge_from_swung_panel (helpers/door_open.probe_hinge).
    hinge_s = far
    hinge = foot + hinge_s * t
    free = foot + near * t
    rose = foot + ref_s * t
    # Panel top (approach height over a partly open door): face points over the panel span, any height.
    on_face = (np.abs((P[:, :2] - foot) @ nrm) < inlier_m) & (((P[:, :2] - foot) @ t - e_neg) * ((P[:, :2] - foot) @ t - e_pos) < 0)
    top_z = float(np.percentile(P[on_face, 2], 98)) if int(on_face.sum()) >= 20 else None
    # Lever grasp: the bar 7-16 cm proud of the face on the hinge side of the rose. Points crowd near
    # the rose, so take the middle of the seen span (offline door:hidden: 1.5 cm from the true middle);
    # the head camera sees the bar's top, so drop ~1.5 cm to its centre line.
    grasp = None
    if rose_s is not None:
        u = np.sign(far - ref_s) * t
        offr = (P3[:, :2] - rose) @ nrm
        sr = (P3[:, :2] - rose) @ u
        lev = (offr > 0.07) & (offr < 0.16) & (sr > 0.03) & (sr < 0.35) & (np.abs(P3[:, 2] - h[2]) < 0.1)
        if int(lev.sum()) >= min_rose_pts:
            lo, hi = np.percentile(sr[lev], [5, 95])
            gxy = rose + 0.5 * (lo + hi) * u + float(np.median(offr[lev])) * nrm
            rot = math.atan2(u[1], u[0]) + math.pi / 2           # fingers close across the bar
            rot = (rot + math.pi / 2) % math.pi - math.pi / 2
            grasp = {"xyz": [round(float(gxy[0]), 3), round(float(gxy[1]), 3),
                             round(float(np.median(P3[lev, 2])) - 0.015, 3)],
                     "rotation": round(rot, 3), "lever_len_m": round(float(hi - lo), 3), "n": int(lev.sum())}
    return {
        "pull_dir": [round(float(nrm[0]), 3), round(float(nrm[1]), 3), 0.0],
        "tangent": [round(float(t[0]), 3), round(float(t[1]), 3), 0.0],
        "hinge": [round(float(hinge[0]), 3), round(float(hinge[1]), 3)],
        "free_edge": [round(float(free[0]), 3), round(float(free[1]), 3)],
        "handle_to_hinge_m": round(abs(hinge_s - ref_s), 3),
        "hinge_source": "panel_end",
        "handle_to_free_m": round(abs(near - ref_s), 3),
        "width_m": round(abs(far - near), 3),
        "face_offset_m": round(face_off, 3),
        "rose": [round(float(rose[0]), 3), round(float(rose[1]), 3)] if rose_s is not None else None,
        "lever_grasp": grasp,
        "top_z": None if top_z is None else round(top_z, 3),
        "n_inliers": int(len(inl)),
        "side_ratio": round(float(ratio), 2),
        "confident": bool(ratio >= 1.3 and len(inl) >= 80),
    }


def measure_door_angle(points, plane, z, z_band=0.15, r_min=0.12, bin_deg=3.0, min_pts=30, min_bin_pts=25):
    """Door opening (deg) about ``plane["hinge"]`` from world ``points`` seen AFTER a pull/push.

    Points at the handle height ``z`` (+-``z_band``) within the panel's radius of the hinge are
    binned by their angle from the closed face (0 deg = the face fitted by
    :func:`estimate_door_plane`, + toward ``pull_dir``); the panel is the densest bin. Returns
    ``(deg, n_pts)`` or ``(None, n)`` when too few points are visible.

    The hand path alone can't tell a door that swung shut behind the hand (harness door_vis3/5,
    2026-11-08: the push "reached 64 deg" while the panel was back at 1-3 deg)."""
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    hg = np.asarray(plane["hinge"], float)[:2]
    u = np.asarray(plane["free_edge"], float)[:2] - hg
    width = float(np.linalg.norm(u))
    u /= max(width, 1e-6)
    n = np.asarray(plane["pull_dir"], float)[:2]
    D = P[:, :2] - hg
    r = np.linalg.norm(D, axis=1)
    keep = (np.abs(P[:, 2] - float(z)) < z_band) & (r > r_min) & (r < width - 0.02)
    D = D[keep]
    ang = np.degrees(np.arctan2(D @ n, D @ u))
    ang = ang[(ang > -5.0) & (ang < 125.0)]
    if len(ang) < min_pts:
        return None, int(len(ang))
    k = np.floor((ang + 5.0) / bin_deg).astype(int)
    cnt = np.bincount(k)
    best = int(np.argmax(cnt))
    if cnt[best] < min_bin_pts:              # door:hidden r5: 8-16 stray points read as 27-54 deg
        return None, int(cnt[best])
    sel = ang[(k >= best - 1) & (k <= best + 1)]
    return round(float(np.median(sel)), 1), int(cnt[best])

def hinge_from_swung_panel(points, plane, z, top_z=None, z_off=(0.05, 0.60), inlier_m=0.012, min_deg=3.0,
                           max_deg=70.0, min_span_m=0.12, iters=400, seed=0, hand_xy=None, hand_clear_m=0.10,
                           near_hinge_m=0.08):
    """Measure the hinge axis from depth seen AFTER a short pull opened the door a little.

    The swung panel is a vertical line (in xy) that meets the CLOSED face line (``plane``: ``free_edge``
    / ``hinge`` / ``tangent`` / ``pull_dir``, measured before the pull) at the hinge: so the hinge comes
    from the panel's own motion, not from how the frame / jamb looks (frame gaps differ between doors).
    Points in front of the closed face (+``pull_dir``), above / below the handle height ``z`` by
    ``z_off`` (skips the lever and rose) and away from the hand are RANSAC-fitted with a line that must
    cross the closed face within the panel's span (+-15 cm) at ``min_deg``..``max_deg``.

    Returns ``{"hinge": [x, y], "open_deg", "free_edge": [x, y] (the other panel end on the closed face),
    "flipped": bool (the hinge is at the end ``plane`` called free), "n", "span_m"}`` or None.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    t = np.asarray(plane["tangent"], float)[:2]
    t /= np.linalg.norm(t)
    n = np.asarray(plane["pull_dir"], float)[:2]
    n /= np.linalg.norm(n)
    f0 = np.asarray(plane["free_edge"], float)[:2]
    s_h = float((np.asarray(plane["hinge"], float)[:2] - f0) @ t)
    lo_s, hi_s = min(0.0, s_h) - 0.15, max(0.0, s_h) + 0.15
    width = abs(s_h)
    top = top_z if top_z is not None else plane.get("top_z")
    dz = P[:, 2] - float(z)
    zk = (np.abs(dz) > z_off[0]) & (np.abs(dz) < z_off[1]) & (P[:, 2] > 0.08)
    if top is not None:
        zk &= P[:, 2] < float(top) - 0.02
    D = P[:, :2] - f0
    s, off = D @ t, D @ n
    k = zk & (off > 0.015) & (off < width * math.sin(math.radians(max_deg)) + 0.05) \
        & (s > lo_s - width) & (s < hi_s + width)
    if hand_xy is not None:
        k &= np.linalg.norm(P[:, :2] - np.asarray(hand_xy, float)[:2], axis=1) > hand_clear_m
    Q = np.c_[s[k], off[k]]                      # (s, off) coordinates: closed face = off 0
    if len(Q) < 30:
        return None
    rng = np.random.default_rng(seed)
    best, best_score = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(Q), 2, replace=False)
        d = Q[j] - Q[i]
        L = float(np.linalg.norm(d))
        if L < 0.05:
            continue
        d /= L
        if abs(d[1]) < 1e-6:
            continue
        s0 = float(Q[i, 0] - Q[i, 1] * d[0] / d[1])         # where the line crosses off = 0
        ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
        if not (lo_s <= s0 <= hi_s and min_deg <= ang <= max_deg):
            continue
        nr = np.array([-d[1], d[0]])
        inl = np.abs((Q - Q[i]) @ nr) < inlier_m
        u = (Q[inl] - [s0, 0.0]) @ (d if d[1] > 0 else -d)   # distance from the hinge along the panel
        u = u[u > -0.03]
        if len(u) < 20:
            continue
        span = float(np.percentile(u, 95) - max(0.0, np.percentile(u, 5)))
        if span < min_span_m:
            continue
        if len(u) > best_score:
            best, best_score = (s0, d if d[1] > 0 else -d), len(u)
    if best is None:
        return None
    s0, d = best
    nr = np.array([-d[1], d[0]])
    inl = Q[np.abs((Q - [s0, 0.0]) @ nr) < inlier_m]
    # refine from the panel away from the hinge: frame / jamb faces near the hinge sit just in front of the
    # closed face, flatten the line and (pivoting about the far panel) push the hinge several cm off
    far = inl[(inl - [s0, 0.0]) @ d > near_hinge_m]
    if len(far) >= 20:
        inl = far
    c = inl.mean(0)                                          # refine: principal direction of the inliers
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    d = vt[0] if vt[0][1] > 0 else -vt[0]
    if abs(d[1]) > 1e-6:
        s_ref = float(c[0] - c[1] * d[0] / d[1])
        if lo_s <= s_ref <= hi_s:
            s0 = s_ref
    ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
    toward = float(np.sign(d[0]))                            # the panel runs from the hinge toward +s or -s
    hinge = f0 + s0 * t
    free_s = s0 + toward * width
    flipped = (abs(s0 - 0.0) < abs(s0 - s_h))                # the hinge is at the end called "free" before
    u = (inl - [s0, 0.0]) @ d
    return {"hinge": [round(float(hinge[0]), 3), round(float(hinge[1]), 3)],
            "open_deg": round(ang, 1),
            "free_edge": [round(float((f0 + free_s * t)[0]), 3), round(float((f0 + free_s * t)[1]), 3)],
            "flipped": bool(flipped), "n": int(len(inl)),
            "span_m": round(float(np.percentile(u, 95) - np.percentile(u, 5)), 3)}
