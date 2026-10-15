"""Seed-point selection: a spread grid over the object's mask, affordance points first.

Why not just the affordance point? Five points within +/-2 px of one pixel (the historical
seeding) span ~1 cm, so

* rotation is unobservable (``pose_tracker`` falls back to translation-only), and
* on a textureless part - a uniform grey door handle - the whole cluster can slide along
  the part as one, and no multi-camera check can see a slide that every camera shares.

A grid spread across the mask gives the rigid fit a real lever arm, and points on textured
regions anchor the ones on textureless regions (they must stay rigidly consistent).

Affordance points (where the robot will act: a grasp point, a handle tip) come first, so a
caller that truncates the list keeps them, and the per-object world point the session
reports is still dominated by the part that matters. Farthest-point sampling then spreads
the rest over the eroded mask (erosion keeps seeds off silhouette edges, where one pixel of
drift lands the depth sample on the background - the grasp failure mode in the videos).

The mask is caller-provided: the simulator's instance segmentation today
(:func:`sim_link_mask`), a segmentation model on real RGB-D.
"""

import logging

import numpy as np

import config
from tracking import geometry

log = logging.getLogger(__name__)

__all__ = ["sim_link_mask", "erode", "mask_grid_seed", "farthest_point_order", "region_seed"]

_UID_BITS = 24


def sim_link_mask(view, body_id, link_index=-1):
    """Boolean mask of one body/link from a PyBullet link-level segmentation buffer.

    ``link_index=-1`` is the base link; ``None`` selects every link of the body.
    Returns ``None`` when the view carries no segmentation.
    """
    seg = getattr(view, "segmentation", None)
    if seg is None:
        return None
    seg = np.asarray(seg).astype(np.int64)
    valid = seg >= 0
    uid = seg & ((1 << _UID_BITS) - 1)
    mask = valid & (uid == int(body_id))
    if link_index is not None:
        link = (seg >> _UID_BITS) - 1
        mask &= link == int(link_index)
    return mask


def erode(mask, px=1):
    """Binary erosion by ``px`` pixels (4-neighbourhood), no OpenCV needed."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(px))):
        m = out
        out = m.copy()
        out[1:, :] &= m[:-1, :]
        out[:-1, :] &= m[1:, :]
        out[:, 1:] &= m[:, :-1]
        out[:, :-1] &= m[:, 1:]
        out[0, :] = out[-1, :] = False
        out[:, 0] = out[:, -1] = False
    return out


def farthest_point_order(candidates, first, n):
    """Indices into ``candidates`` (K, 2): ``first`` (list of indices) then FPS to ``n``."""
    candidates = np.asarray(candidates, dtype=float)
    chosen = list(dict.fromkeys(int(i) for i in first))[:n]
    if not len(candidates):
        return chosen
    if not chosen:
        chosen = [0]
    d = np.min(np.linalg.norm(candidates[:, None, :] - candidates[chosen][None, :, :], axis=2),
               axis=1)
    while len(chosen) < min(n, len(candidates)):
        j = int(np.argmax(d))
        if d[j] <= 0.0:
            break
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(candidates - candidates[j], axis=1))
    return chosen


def mask_grid_seed(view, mask, affordance_px=(), n_points=None, erode_px=None, min_points=None):
    """World seed points on ``mask``: affordance pixels first, then a spread grid.

    ``affordance_px``: pixels ``(x, y)`` that must be tracked; each is snapped to the nearest
    usable mask pixel. Returns ``(world_points (N, 3), pixels (N, 2), info dict)`` or
    ``(None, None, info)`` when fewer than ``min_points`` usable pixels exist - the caller
    should then fall back to affordance-only seeding. Defaults: ``config.track_seed_*``.
    """
    n_points = int(config.track_seed_points if n_points is None else n_points)
    erode_px = int(config.track_seed_erode_px if erode_px is None else erode_px)
    min_points = int(config.track_seed_min_points if min_points is None else min_points)
    info = {"n_mask_px": 0, "n_usable_px": 0, "n_affordance": 0}
    if mask is None:
        info["why"] = "no mask"
        return None, None, info
    mask = np.asarray(mask, dtype=bool)
    info["n_mask_px"] = int(mask.sum())
    core = erode(mask, erode_px)
    if core.sum() < min_points:                 # thin parts: erosion would erase them
        core = mask
    ys, xs = np.nonzero(core)
    pix = np.stack([xs, ys], axis=1).astype(float)
    depth_ok = np.array([geometry.sample_depth(view, p, window=0) is not None for p in pix],
                        dtype=bool) if len(pix) else np.zeros(0, dtype=bool)
    pix = pix[depth_ok]
    info["n_usable_px"] = int(len(pix))
    if len(pix) < min_points:
        info["why"] = f"only {len(pix)} usable mask pixel(s)"
        return None, None, info

    first = []
    for a in ([] if affordance_px is None else list(affordance_px)):
        a = np.asarray(a, dtype=float).reshape(2)
        first.append(int(np.argmin(np.linalg.norm(pix - a, axis=1))))
    info["n_affordance"] = len(set(first))
    order = farthest_point_order(pix, first, int(n_points))
    pixels = pix[order]
    world = []
    for p in pixels:
        depth = geometry.sample_depth(view, p, window=0)
        world.append(geometry.deproject_pixel_to_world(view, p, depth))
    world = np.asarray(world, dtype=float)
    extent = float(np.max(np.linalg.norm(world - world.mean(axis=0), axis=1)))
    info.update({"n_points": int(len(world)), "extent_m": round(extent, 4),
                 "spread_px": round(float(np.ptp(pixels, axis=0).max()), 1)})
    log.info("[seeding] mask grid: %d point(s) (%d affordance) over %d usable px, "
             "extent %.3f m", len(world), info["n_affordance"], len(pix), extent)
    return world, pixels, info


def _near_depth(view, px, radius=2):
    """Metric depth of the nearest surface in a small window (the part, not what is behind)."""
    x, y = int(round(float(px[0]))), int(round(float(px[1])))
    d = np.asarray(view.depth, dtype=float)
    win = d[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].reshape(-1)
    win = win[np.isfinite(win) & (win > 0)]
    return float(np.percentile(win, 10)) if win.size else None


def _protruding(view, sub, x0, y0, protrude_m, info, iters=200, inlier_m=0.006, min_frac=0.25,
                seed=0):
    """Mask of ``sub`` (a depth window at ``(x0, y0)``) standing >= ``protrude_m`` in front of
    the window's dominant plane (towards the camera), or ``None`` when no plane holds
    ``min_frac`` of the pixels."""
    hh, ww = sub.shape
    ys, xs = np.nonzero(np.isfinite(sub) & (sub > 0))
    if len(xs) < 20:
        return None
    pix = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    world, valid = geometry.points_to_world(view, pix)
    world, ys, xs = world[valid], ys[valid], xs[valid]
    if len(world) < 20:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = world[rng.choice(len(world), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        cnt = int(np.count_nonzero(np.abs((world - a) @ n) <= inlier_m))
        if cnt > best_n:
            best, best_n = (n, a), cnt
    if best is None or best_n < min_frac * len(world):
        return None
    n, a = best
    if float((np.asarray(view.position, dtype=float) - a) @ n) < 0:
        n = -n                                         # normal towards the camera
    out = np.zeros((hh, ww), dtype=bool)
    out[ys, xs] = (world - a) @ n >= protrude_m
    info.update(part="plane", plane_frac=round(best_n / len(world), 2))
    return out


def region_seed(view, affordance_px, n_points=None, radius_px=None, depth_tol_m=None,
                corner_frac=None, erode_px=None, min_points=None, protrude_m=None):
    """World seed points spread over the *part* around an affordance pixel, no mask needed.

    The part is the connected component of the affordance pixel among the window pixels that
    stand at least ``protrude_m`` in front of the window's dominant plane (RANSAC: the door
    face behind a handle, the table under a box). Without such a plane, it is the pixels
    within ``depth_tol_m`` of the affordance pixel's near depth. Within the part, only pixels
    with 2D texture (Shi-Tomasi min-eigenvalue >= ``corner_frac`` x the part's maximum) are
    candidates: on a uniform bar a patch cannot see motion along the bar (aperture problem)
    and slides; corners and ends cannot. Affordance pixels first, then farthest-point
    sampling. Returns ``(world (N, 3), pixels (N, 2), info)`` or ``(None, None, info)``.
    """
    import cv2

    n_points = int(config.track_seed_points if n_points is None else n_points)
    radius_px = int(config.track_seed_region_radius_px if radius_px is None else radius_px)
    depth_tol_m = float(config.track_seed_region_depth_tol_m if depth_tol_m is None else depth_tol_m)
    protrude_m = float(config.track_seed_region_protrude_m if protrude_m is None else protrude_m)
    corner_frac = float(config.track_seed_region_corner_frac if corner_frac is None else corner_frac)
    erode_px = int(config.track_seed_erode_px if erode_px is None else erode_px)
    min_points = int(config.track_seed_min_points if min_points is None else min_points)
    info = {}
    aff = [np.asarray(a, dtype=float).reshape(2) for a in ([] if affordance_px is None else list(affordance_px))]
    if not aff:
        info["why"] = "no affordance pixel"
        return None, None, info
    z0 = _near_depth(view, aff[0])
    if z0 is None:
        info["why"] = "no depth at the affordance pixel"
        return None, None, info
    depth = np.asarray(view.depth, dtype=float)
    H, W = depth.shape
    cx, cy = int(round(aff[0][0])), int(round(aff[0][1]))
    x0, y0 = max(0, cx - radius_px), max(0, cy - radius_px)
    x1, y1 = min(W, cx + radius_px + 1), min(H, cy + radius_px + 1)
    sub = depth[y0:y1, x0:x1]
    near = _protruding(view, sub, x0, y0, protrude_m, info)
    if near is None:            # no support plane: the near-depth band around the pixel
        near = np.isfinite(sub) & (np.abs(sub - z0) <= depth_tol_m)
        info["part"] = "depth band"
    _n, labels = cv2.connectedComponents(near.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(near)
    if not len(xs):
        info["why"] = "empty part"
        return None, None, info
    # the component the affordance pixel (or its nearest part pixel) belongs to
    j = int(np.argmin((xs - (cx - x0)) ** 2 + (ys - (cy - y0)) ** 2))
    part = labels == labels[ys[j], xs[j]]
    core = erode(part, erode_px)
    if core.sum() < min_points:
        core = part
    info["n_part_px"] = int(part.sum())

    rgb = np.asarray(view.rgb)[..., :3].astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    corner = cv2.cornerMinEigenVal(gray, blockSize=7, ksize=3)[y0:y1, x0:x1]
    cvals = corner[core]
    if not cvals.size or float(cvals.max()) <= 0:
        info["why"] = "textureless part"
        return None, None, info
    textured = core & (corner >= corner_frac * float(cvals.max()))
    ys, xs = np.nonzero(textured)
    pix = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    info["n_textured_px"] = int(len(pix))
    if len(pix) < min_points:
        info["why"] = f"only {len(pix)} textured part pixel(s)"
        return None, None, info
    first = [int(np.argmin(np.linalg.norm(pix - a, axis=1))) for a in aff]
    order = farthest_point_order(pix, first, n_points)
    pixels = pix[order]
    world = np.asarray([geometry.deproject_pixel_to_world(view, p, depth[int(p[1]), int(p[0])])
                        for p in pixels], dtype=float)
    extent = float(np.max(np.linalg.norm(world - world.mean(axis=0), axis=1)))
    info.update({"n_points": int(len(world)), "extent_m": round(extent, 4),
                 "spread_px": round(float(np.ptp(pixels, axis=0).max()), 1)})
    log.info("[seeding] region: %d point(s) over %d textured / %d part px, extent %.3f m",
             len(world), len(pix), info["n_part_px"], extent)
    return world, pixels, info
