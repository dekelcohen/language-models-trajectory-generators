"""Search-based re-acquisition (docs/plans/3d_tracking.md §13, fix 6).

When no camera measures the object for a while, the per-point trackers cannot recover on
their own: their search windows are small, and re-seeding them from the Kalman prediction
only helps while that prediction is still close. After ~1 s of an unmeasured swinging door
it is 10-25 cm off, so predicted re-seeds land on the door leaf and every later measurement
is gated out - the track stays black for good (measured: door, head occluded, wrist framed
off the handle).

This module *looks for the object*, independent of the 2D tracks:

1. **Reference bank.** While the pose is trusted, every ``track_reacq_ref_every`` frames each
   camera stores an appearance patch around every depth-verified seed point, together with
   that pose. The newest bank is used, so the appearance gap is only the motion since the
   object was last seen, not since seeding.
2. **Hypotheses -> search windows.** Each pose hypothesis (Kalman prediction, gripper FK,
   last measured pose) projects every seed point into every camera; NCC with the banked
   patch, rotated over ``track_reacq_angles``, runs in a window around it. The prediction's
   window grows with the filter's uncertainty, clamped to ``[win_min_px, win_max_px]``.
3. **Candidates -> 3D.** The ``topk`` NCC peaks per point above ``ncc_min`` are lifted
   through the camera's own depth.
4. **RANSAC rigid fit** (Kabsch, no scale) of the seed template against the candidates,
   pooled over cameras: a wrong peak per point is fine as long as enough points agree on
   one rigid pose. Accepted only with ``min_frac`` of the banked points (and at least
   ``track_reacq_min_inliers``), spanning ``min_spread`` of the template, RMSE below
   ``track_pose_green_rmse_m``.
5. **Plausibility + verification.** The found centroid must lie within
   ``max_jump_m + max_speed * t_unmeasured`` of a hypothesis, and the banked patches must
   NCC-match at the found pose's projections (:func:`verify_pose`).

The caller (``TrackingSession``) re-seeds the cameras from the found pose and re-initialises
the pose filter on it as *tentative* - a wrong lock still has to be confirmed by the normal
tracking before it counts (pose_tracker's wrong-lock guards).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

import config
from tracking import geometry


def _gray(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    return cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _crop(gray, pt, half):
    x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
    if x - half < 0 or y - half < 0 or x + half + 1 > gray.shape[1] or y + half + 1 > gray.shape[0]:
        return None
    return gray[y - half:y + half + 1, x - half:x + half + 1].copy()


def kabsch(src, dst):
    """``R, t`` minimising ``|R src + t - dst|`` (no scale)."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


@dataclass
class RefBank:
    """One camera's appearance snapshot: ``{seed_idx: rotated template set}``."""
    frame: int
    templates: Dict[int, List[np.ndarray]] = field(default_factory=dict)


def rotated_templates(gray, pt, half, angles):
    """The patch at ``pt`` rotated by each angle (a crop twice the size, rotated, re-cropped)."""
    big = _crop(gray, pt, 2 * half)
    if big is None:
        return []
    c = (2 * half, 2 * half)
    out = []
    for ang in angles:
        if abs(float(ang)) < 1e-9:
            out.append(big[half:3 * half + 1, half:3 * half + 1].copy())
            continue
        M = cv2.getRotationMatrix2D(c, float(ang), 1.0)
        r = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR)
        out.append(r[half:3 * half + 1, half:3 * half + 1].copy())
    return out


def capture_bank(view, pose, seed_points_world, frame, half=None, angles=None):
    """Snapshot the depth-verified seed points' appearance under a trusted ``pose``."""
    half = int(config.track_patch_half if half is None else half)
    angles = tuple(config.track_reacq_angles if angles is None else angles)
    gray = _gray(view.rgb)
    bank = RefBank(frame=int(frame))
    for i, wp in enumerate(pose.apply(seed_points_world)):
        ok, px, _detail = geometry.is_visible(view, wp)
        if not ok:
            continue
        tm = rotated_templates(gray, px, half, angles)
        if tm:
            bank.templates[i] = tm
    return bank


def _peaks(score, k, min_score, suppress):
    """Top-``k`` local maxima of ``score`` (greedy non-max suppression)."""
    s = score.copy()
    out = []
    for _ in range(k):
        _mn, mx, _l, loc = cv2.minMaxLoc(s)
        if mx < min_score:
            break
        out.append((float(mx), loc))
        x, y = loc
        s[max(0, y - suppress):y + suppress + 1, max(0, x - suppress):x + suppress + 1] = -1.0
    return out


def search_candidates(view, bank, centres, radius, topk, ncc_min):
    """``{seed_idx: [(score, pixel), ...]}`` - NCC peaks of each banked patch near its centre."""
    gray = _gray(view.rgb)
    H, W = gray.shape
    out = {}
    for i, tmpls in bank.templates.items():
        c = centres.get(i)
        if c is None or not np.all(np.isfinite(c)):
            continue
        h = tmpls[0].shape[0] // 2
        x, y = int(round(float(c[0]))), int(round(float(c[1])))
        X0, Y0 = max(0, x - radius - h), max(0, y - radius - h)
        X1, Y1 = min(W, x + radius + h + 1), min(H, y + radius + h + 1)
        sub = gray[Y0:Y1, X0:X1]
        if sub.shape[0] <= 2 * h or sub.shape[1] <= 2 * h:
            continue
        best = None
        for t in tmpls:
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            best = r if best is None else np.maximum(best, r)
        peaks = _peaks(best, topk, ncc_min, suppress=max(2, h // 2))
        if peaks:
            out[i] = [(s, (X0 + loc[0] + h, Y0 + loc[1] + h)) for s, loc in peaks]
    return out


def ransac_pose(template, cands, iters, inlier_m, min_inliers, rng=None):
    """Best rigid ``(R, t, inlier_idx, rmse)`` mapping ``template[i]`` onto one of ``cands[i]``.

    ``cands``: ``{seed_idx: (M, 3) world candidates}``. Hypotheses come from 3 random seed
    indices with one random candidate each; a seed index is an inlier when *any* of its
    candidates lies within ``inlier_m`` of the transformed template point.
    """
    rng = rng or np.random.default_rng(0)
    idx = [i for i, c in cands.items() if len(c)]
    if len(idx) < 3:
        return None
    best = None
    for _ in range(int(iters)):
        pick = rng.choice(len(idx), 3, replace=False)
        ii = [idx[j] for j in pick]
        src = template[ii]
        dst = np.array([cands[i][rng.integers(len(cands[i]))] for i in ii])
        if np.linalg.matrix_rank(src - src.mean(axis=0), tol=1e-4) < 2:
            continue
        # the 3 samples must be as far apart as their template points (rigid)
        if np.max(np.abs(np.linalg.norm(src[:, None] - src[None], axis=2)
                         - np.linalg.norm(dst[:, None] - dst[None], axis=2))) > 2.0 * inlier_m:
            continue
        R, t = kabsch(src, dst)
        inl, match = [], []
        for i in idx:
            p = R @ template[i] + t
            d = np.linalg.norm(cands[i] - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= inlier_m:
                inl.append(i)
                match.append(cands[i][j])
        if best is None or len(inl) > len(best[0]):
            best = (inl, np.array(match))
    if best is None or len(best[0]) < max(3, min_inliers):
        return None
    inl, match = best
    R, t = kabsch(template[inl], match)
    res = np.linalg.norm(template[inl] @ R.T + t - match, axis=1)
    return R, t, inl, float(np.sqrt(np.mean(res ** 2)))


def verify_pose(views, banks, tmpl, R, t, radius=3):
    """``(verified, checkable)``: seed points that, under ``R, t``, are depth-verified visible
    in a camera with a banked patch *and* whose patch matches (NCC >= ncc_min) within
    ``radius`` px of the projection. RANSAC consensus alone is not enough: with several
    candidates per point and 3-point hypotheses, a few far-off peaks can agree rigidly by
    chance (measured: a door "found" 63 cm away, which then confirmed itself)."""
    world = tmpl @ np.asarray(R).T + np.asarray(t)
    ok_idx, chk_idx = set(), set()
    ncc_min = float(config.track_reacq_ncc_min)
    for cam, bank in banks.items():
        view = views.get(cam)
        if view is None or bank is None:
            continue
        gray = None
        for i, tmpls in bank.templates.items():
            vis, px, _d = geometry.is_visible(view, world[i])
            if not vis:
                continue
            chk_idx.add(i)
            if gray is None:
                gray = _gray(view.rgb)
            found = search_candidates(_GrayView(gray), RefBank(0, {i: tmpls}),
                                      {i: px}, radius, 1, ncc_min)
            if found.get(i):
                ok_idx.add(i)
    return len(ok_idx), len(chk_idx)


class _GrayView:
    """Minimal stand-in so :func:`search_candidates` can reuse an already-converted image."""
    def __init__(self, gray):
        self.rgb = gray


def reacquire(views, banks, seed_points_world, hypotheses, frame_idx=None, max_dev_m=None):
    """Look for the object; returns ``(R, t, info)`` or ``(None, None, info)``.

    ``banks``: ``{cam: RefBank}``. ``hypotheses``: list of ``(name, pose, radius_px)``.
    ``max_dev_m``: the found centroid must lie this close to some hypothesis' centroid
    (how far the object can have moved while unseen).
    """
    tmpl = np.asarray(seed_points_world, dtype=float).reshape(-1, 3)
    info = {"cands": {}, "hyps": [h[0] for h in hypotheses]}
    pooled: Dict[int, list] = {}
    for cam, bank in banks.items():
        view = views.get(cam)
        if view is None or bank is None or not bank.templates:
            continue
        n_c = 0
        for _name, pose, radius in hypotheses:
            centres = {}
            for i, wp in enumerate(pose.apply(tmpl)):
                px, z = geometry.project_world_to_pixel(view, wp)
                if px is not None and z is not None and z > 0:
                    centres[i] = px
            found = search_candidates(view, bank, centres, int(radius),
                                      int(config.track_reacq_topk),
                                      float(config.track_reacq_ncc_min))
            for i, peaks in found.items():
                pix = np.array([p for _s, p in peaks], dtype=float)
                world, valid = geometry.points_to_world(view, pix)
                for w in world[valid]:
                    lst = pooled.setdefault(i, [])
                    if all(np.linalg.norm(w - q) > 1e-3 for q in lst):
                        lst.append(w)
                        n_c += 1
        info["cands"][cam] = n_c
    cands = {i: np.array(v) for i, v in pooled.items()}
    n_banked = len(set().union(*[set(b.templates) for b in banks.values() if b])) if banks else 0
    need = max(int(config.track_pose_min_points), int(config.track_reacq_min_inliers),
               int(np.ceil(float(config.track_reacq_min_frac) * max(n_banked, 1))))
    fit = ransac_pose(tmpl, cands, config.track_reacq_iters,
                      float(getattr(config, "track_reacq_inlier_m", config.track_pose_inlier_m)), need)
    if fit is None:
        info["why"] = f"no rigid consensus (need {need} of {n_banked} banked points)"
        return None, None, info
    R, t, inl, rmse = fit
    info.update(inliers=len(inl), need=need, rmse_m=round(rmse, 4))
    if rmse > float(config.track_pose_green_rmse_m):
        info["why"] = f"rmse {rmse:.4f}"
        return None, None, info
    full = float(np.max(np.linalg.norm(tmpl - tmpl.mean(axis=0), axis=1)))
    if full >= float(config.track_pose_min_rot_extent_m):
        sub = tmpl[inl]
        spread = float(np.max(np.linalg.norm(sub - sub.mean(axis=0), axis=1)))
        if spread < float(config.track_reacq_min_spread) * full:
            info["why"] = f"spread {spread:.3f}/{full:.3f} m"
            return None, None, info
    centre = R @ tmpl.mean(axis=0) + t
    if max_dev_m is not None and hypotheses:
        dev = min(float(np.linalg.norm(centre - pose.apply(tmpl.mean(axis=0))[0]))
                  for _n, pose, _r in hypotheses)
        info["dev_m"] = round(dev, 3)
        if dev > float(max_dev_m):
            info["why"] = f"implausible jump {dev:.3f} > {float(max_dev_m):.3f} m"
            return None, None, info
    verified, checkable = verify_pose(views, banks, tmpl, R, t)
    info["verified"] = f"{verified}/{checkable}"
    if verified < need or verified < float(config.track_reacq_verify_frac) * checkable:
        info["why"] = f"verification {verified}/{checkable} (need {need})"
        return None, None, info
    return R, t, info
