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

__all__ = ["sim_link_mask", "erode", "mask_grid_seed", "farthest_point_order"]

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
