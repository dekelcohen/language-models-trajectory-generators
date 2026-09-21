"""Hand-written per-view weighting on top of DLT triangulation - no network, no GPU.

Why this exists
---------------
LAPA pays for a DINOv2 ViT-B/14 forward pass (measured: 221 MB / 194 ms per frame, versus
24 MB / 23 ms for everything else it does) to produce a *learned* per-view weight. Audited,
that head is a constant function: logit ``+6.91`` for every view, and sweeping the
reprojection residual from 0 to 200 px moves it by 0.01. Its whole learned contribution to
the 3D point is a residual clamped to ~8 mm.

This module is the control that replaces it: the same IRLS DLT, but with a weight written
by hand from signals that cost microseconds. The interesting claim is not that a hand-made
weight is better in principle - it is that the *informative* signal at two cameras is one
LAPA structurally cannot use.

The two-camera problem, and the signal that survives it
-------------------------------------------------------
At exactly two views the DLT is exactly determined (4 equations, 4 homogeneous unknowns),
so *any* pair of pixels reprojects near-perfectly: a 47 px drift moves the 3D point by more
than a centimetre while the residual stays at 0.2 px. Reprojection-residual weighting -
which is all LAPA's head could ever learn from, since that is its only geometric input - is
therefore structurally blind exactly where head+wrist lives. Residual weighting is enabled
here only from ``residual_min_views`` (default 3) cameras up, and that is stated honestly
rather than hidden behind an IRLS loop that silently does nothing.

What does work at two views is **metric depth**. Each camera's tracked pixel has a rendered
depth behind it; deprojecting it gives that camera's own opinion of where the point is in
3D, entirely independently of the other camera. When the arm swings in front of the head
camera, the head's depth sample jumps to the occluder's surface - tens of centimetres - in
a single frame, while the object itself moves ~1 cm per frame. Comparing each view's
depth-deprojected point against the previous accepted 3D position (and, on the first frame,
against the cross-view median) detects that immediately, names the guilty camera, and costs
one depth lookup per point per view.

When only one camera survives that test, triangulation is impossible and the provider falls
back to that camera's depth deprojection - which is exactly the right answer, because the
surviving camera is looking at the object and knows how far away it is.

Signals used, in order of measured usefulness
---------------------------------------------
1. **depth consistency** - the only one that bites at two cameras (see above);
2. **tracker confidence / health** - cheap, already computed per camera upstream;
3. **staleness** - a camera whose tracker has not genuinely re-measured for several frames
   (``meta['stale_frames']`` where the 2D provider reports it) is coasting, not observing;
4. **reprojection residual** - genuinely useful from 3 views, inert at 2;
5. **cross-view patch NCC** - optional (``use_ncc``), off by default: it is the most
   expensive of the five and the depth test already catches the "template latched onto the
   occluder" case it is aimed at.
"""

from dataclasses import dataclass, field

import numpy as np

from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D
from tracking import camera_convert, geometry


@dataclass
class ViewWeight:
    """One camera's weight for one point, with the reason it ended up that way."""

    cam: str
    weight: float
    reason: str = "ok"
    terms: dict = field(default_factory=dict)

    @property
    def rejected(self):
        return self.weight <= 0.0


def view_weight(cam, confidence=1.0, stale_frames=0, residual_px=None, depth_gap_m=None,
                ncc=None, sigma_px=2.0, depth_tol_m=0.05, depth_reject_m=0.08,
                stale_halflife=3.0, floor=1e-3):
    """Per-view weight from cheap, DINOv2-free signals. Never raises.

    ``depth_gap_m`` is *signed*: ``sampled_depth - predicted_depth`` of the reference point
    in this view. Negative means the camera sees a surface **in front of** where the point
    should be, i.e. an occluder; positive means the point is floating in front of whatever
    the camera images, i.e. the track slid off the object into free space. Both are
    disqualifying past ``depth_reject_m``, but they are reported separately because they
    have different causes and the distinction is what makes the log readable.

    ``residual_px`` should be passed as ``None`` at two views - see the module docstring;
    the residual is not merely weak there, it is uninformative by construction.
    """
    terms = {}
    reason = "ok"

    conf = float(np.clip(confidence if confidence is not None else 1.0, 0.0, 1.0))
    terms["conf"] = 0.1 + 0.9 * conf

    stale = max(0.0, float(stale_frames or 0.0))
    terms["stale"] = float(0.5 ** (stale / max(stale_halflife, 1e-6)))

    if residual_px is None or not np.isfinite(residual_px):
        terms["residual"] = 1.0
    else:
        terms["residual"] = float(1.0 / (1.0 + (float(residual_px) / max(sigma_px, 1e-6)) ** 2))

    if depth_gap_m is None or not np.isfinite(depth_gap_m):
        terms["depth"] = 1.0
    else:
        gap = abs(float(depth_gap_m))
        if gap >= depth_reject_m:
            terms["depth"] = 0.0
            reason = "occluder" if float(depth_gap_m) < 0 else "depth_mismatch"
        elif gap <= depth_tol_m:
            terms["depth"] = 1.0
        else:
            terms["depth"] = float(1.0 / (1.0 + ((gap - depth_tol_m) / max(depth_tol_m, 1e-6)) ** 2))

    if ncc is None or not np.isfinite(ncc):
        terms["ncc"] = 1.0
    else:
        terms["ncc"] = float(0.25 + 0.75 * np.clip(ncc, 0.0, 1.0))

    weight = float(np.prod([terms[k] for k in ("conf", "stale", "residual", "depth", "ncc")]))
    if terms["depth"] <= 0.0:
        return ViewWeight(cam=cam, weight=0.0, reason=reason, terms=terms)
    if weight < floor:
        # A view is never allowed to fade to *exactly* zero on soft signals alone: only the
        # hard depth rejection above may remove a camera, because at two views removing the
        # wrong one is unrecoverable.
        weight = floor
        reason = "floored"
    return ViewWeight(cam=cam, weight=weight, reason=reason, terms=terms)


def depth_gap(view, pixel_xy, reference_point):
    """Signed ``sampled_depth - predicted_depth`` (metres) of ``reference_point`` in a view.

    ``None`` when the comparison cannot be made (no depth, off-image, behind the camera) -
    which must be read as "no information", never as "bad view".
    """
    if view is None or reference_point is None:
        return None
    predicted, z_eye = geometry.project_world_to_pixel(view, np.asarray(reference_point, float))
    if predicted is None or z_eye <= 0.0:
        return None
    sampled = geometry.sample_depth(view, pixel_xy)
    if sampled is None:
        return None
    return float(sampled) - float(z_eye)


def patch_ncc(view_a, uv_a, view_b, uv_b, window=5):
    """Zero-mean normalised cross-correlation of two grayscale patches, in ``[-1, 1]``.

    Detects "one camera's template latched onto a different surface" from raw pixels only.
    Optional because the depth test already catches that case far more cheaply.
    """
    patch_a = _gray_patch(view_a, uv_a, window)
    patch_b = _gray_patch(view_b, uv_b, window)
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return None
    a = patch_a - patch_a.mean()
    b = patch_b - patch_b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return None
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)


def triangulate_weighted(projections, pixels, prior_weights=None, iters=10, sigma_px=2.0,
                         use_residual=True):
    """Prior-weighted DLT, optionally reweighted by the reprojection residual (Cauchy).

    Identical to :func:`providers.tracker3d.triangulate.triangulate_irls` except that each
    view enters with a prior weight and the residual reweighting can be switched off - the
    honest configuration at two views, where the residual is structurally zero.

    Deliberately *not* row-normalised. LAPA scales its DLT rows by ``sqrt(w)`` and then
    divides every row by its own norm, which cancels the weight exactly; measured, 8 orders
    of magnitude of weight change there move the output by 6e-07 m.
    """
    projections = [np.asarray(P, dtype=float).reshape(3, 4) for P in projections]
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(projections) != len(pixels) or len(projections) < 2:
        return None, float("inf")

    prior = (np.ones(len(projections), dtype=float) if prior_weights is None
             else np.asarray(prior_weights, dtype=float).reshape(-1))
    if prior.shape[0] != len(projections) or not np.all(np.isfinite(prior)):
        prior = np.ones(len(projections), dtype=float)
    prior = np.clip(prior, 0.0, None)
    if not np.any(prior > 0.0):
        prior = np.ones(len(projections), dtype=float)

    weights = prior.copy()
    point = None
    rounds = max(1, int(iters)) if use_residual else 1
    for _ in range(rounds):
        rows = []
        for P, (x, y), w in zip(projections, pixels, weights):
            rows.append(w * (x * P[2] - P[0]))
            rows.append(w * (y * P[2] - P[1]))
        try:
            _u, _s, vt = np.linalg.svd(np.asarray(rows, dtype=float))
        except np.linalg.LinAlgError:
            return None, float("inf")
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-12:
            return None, float("inf")
        point = homogeneous[:3] / homogeneous[3]
        if not np.all(np.isfinite(point)):
            return None, float("inf")
        if not use_residual:
            break
        errors = _errors(projections, pixels, point)
        weights = prior * (1.0 / (1.0 + (errors / max(sigma_px, 1e-6)) ** 2))

    return point, float(np.median(_errors(projections, pixels, point)))


class WeightedTriangulateTracker3D(MultiCamTracker3D):
    """DLT triangulation with hand-written per-view weights and a depth-consistency gate."""

    name = "weighted_triangulate"
    needs_depth = True                  # for *weighting* only; the lift itself is geometric
    min_cameras = 2

    def __init__(self, iters=10, sigma_px=2.0, max_reproj_px=25.0, depth_tol_m=0.05,
                 depth_reject_m=0.08, residual_min_views=3, stale_halflife=3.0,
                 use_depth_gate=True, use_confidence=True, use_stale=True, use_ncc=False,
                 ncc_window=5, depth_fallback=True, prev_max_gap_m=0.25, **kwargs):
        super().__init__(iters=iters, sigma_px=sigma_px, max_reproj_px=max_reproj_px,
                         depth_tol_m=depth_tol_m, depth_reject_m=depth_reject_m,
                         residual_min_views=residual_min_views, stale_halflife=stale_halflife,
                         use_depth_gate=use_depth_gate, use_confidence=use_confidence,
                         use_stale=use_stale, use_ncc=use_ncc, ncc_window=ncc_window,
                         depth_fallback=depth_fallback, prev_max_gap_m=prev_max_gap_m, **kwargs)
        self.iters = int(iters)
        self.sigma_px = float(sigma_px)
        self.max_reproj_px = float(max_reproj_px)
        self.depth_tol_m = float(depth_tol_m)
        self.depth_reject_m = float(depth_reject_m)
        self.residual_min_views = int(residual_min_views)
        self.stale_halflife = float(stale_halflife)
        self.use_depth_gate = bool(use_depth_gate)
        self.use_confidence = bool(use_confidence)
        self.use_stale = bool(use_stale)
        self.use_ncc = bool(use_ncc)
        self.ncc_window = int(ncc_window)
        self.depth_fallback = bool(depth_fallback)
        self.prev_max_gap_m = float(prev_max_gap_m)
        self._prev = {}                 # obj -> {seed index: last accepted world point}

    def reset(self):
        self._prev.clear()

    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None):
        try:
            return self._lift(obj_name, views, cam_tracks, point_index)
        except Exception as exc:                            # never raise: ABC contract
            self._log(f"[wtri] '{obj_name}' lift error {type(exc).__name__}: {exc}")
            return Lift3DResult(meta={"provider": self.name,
                                      "why": f"{type(exc).__name__}: {exc}"})

    # -- implementation ----------------------------------------------------
    def _lift(self, obj_name, views, cam_tracks, point_index):
        usable = self._usable(cam_tracks)
        if len(usable) < self.min_cameras:
            return Lift3DResult(meta={"provider": self.name,
                                      "why": f"need {self.min_cameras} cams, have {len(usable)}"})

        obs = {}                                  # seed index -> [(cam, uv), ...]
        cams_seen = set()
        for cam, (points, visible) in usable.items():
            if views.get(cam) is None:
                continue
            indices = self._indices_for(cam, points, point_index)
            for j, idx in enumerate(indices):
                if idx < 0 or not visible[j]:
                    continue
                obs.setdefault(int(idx), []).append((cam, points[j]))
            cams_seen.add(cam)

        cameras = {cam: camera_convert.camera_matrices(views[cam]) for cam in cams_seen}
        confidence = {cam: self._confidence(cam_tracks.get(cam)) for cam in cams_seen}
        stale = {cam: self._stale_frames(cam_tracks.get(cam)) for cam in cams_seen}
        previous = self._prev.get(obj_name, {})

        world_points, kept_index, residuals, worst = [], [], [], []
        used_cams = set()
        cam_weight_sum, cam_weight_n = {}, {}
        rejected = {}                             # cam -> [(seed index, gap, reason), ...]
        n_fallback = 0
        fresh = {}

        for idx in sorted(obs):
            entries = [(c, uv) for c, uv in obs[idx] if c in cameras]
            if len(entries) < self.min_cameras:
                continue

            depth_points = {c: self._depth_point(views[c], uv) for c, uv in entries}
            reference = self._reference(previous.get(idx), depth_points)
            weights = self._weigh(entries, views, reference, depth_points, confidence, stale)

            for wv in weights:
                cam_weight_sum[wv.cam] = cam_weight_sum.get(wv.cam, 0.0) + wv.weight
                cam_weight_n[wv.cam] = cam_weight_n.get(wv.cam, 0) + 1
                if wv.rejected:
                    rejected.setdefault(wv.cam, []).append(
                        (idx, round(float(wv.terms.get("gap_m", 0.0)), 4), wv.reason))

            keep = [i for i, wv in enumerate(weights) if not wv.rejected]
            point, residual, cams_for_point, how = self._solve(
                entries, weights, keep, cameras, depth_points)
            if point is None:
                continue
            if how == "triangulated":
                if residual > self.max_reproj_px:
                    continue
                if not self._cheirality(cams_for_point, cameras, point):
                    continue
                projections = [camera_convert.projection_matrix_3x4(*cameras[c])
                               for c in cams_for_point]
                pixels = np.asarray([uv for c, uv in entries if c in cams_for_point], dtype=float)
                worst.append(float(np.max(_errors(
                    [np.asarray(P, float).reshape(3, 4) for P in projections], pixels, point))))
            else:
                n_fallback += 1

            world_points.append(point)
            kept_index.append(idx)
            residuals.append(residual)
            used_cams.update(cams_for_point)
            fresh[idx] = np.asarray(point, dtype=float)

        self._prev[obj_name] = fresh
        if not world_points:
            return Lift3DResult(meta={"provider": self.name,
                                      "why": "no seed index usable in >=2 cameras",
                                      "n_candidates": len(obs)})

        points = np.asarray(world_points, dtype=float)
        centroid, n_used = geometry.robust_centroid(points, np.ones(len(points), dtype=bool))
        if centroid is None:
            return Lift3DResult(meta={"provider": self.name, "why": "robust centroid rejected all"})

        weights_out = {c: round(cam_weight_sum[c] / max(cam_weight_n[c], 1), 4)
                       for c in sorted(cam_weight_sum)}
        if rejected or n_fallback:
            self._log(f"[wtri] '{obj_name}' depth gate: "
                      + "; ".join(f"{cam} rejected {len(items)} pt(s) "
                                  f"({items[0][2]}, gap={items[0][1]:+.3f}m)"
                                  for cam, items in rejected.items())
                      + (f"; {n_fallback} point(s) fell back to single-view depth"
                         if n_fallback else "")
                      + f" | weights={weights_out}")

        return Lift3DResult(
            world_point=centroid,
            disagreement=_spread(points),
            used_cams=sorted(used_cams),
            per_point=points,
            per_point_index=np.asarray(kept_index, dtype=int),
            weights=weights_out,
            meta={"provider": self.name, "n_points": int(n_used),
                  "median_reproj_px": round(float(np.median(residuals)), 3),
                  "worst_reproj_px": round(float(np.max(worst)), 3) if worst else None,
                  "n_depth_rejected": int(sum(len(v) for v in rejected.values())),
                  "rejected_cams": sorted(rejected),
                  "n_depth_fallback": int(n_fallback)},
        )

    def _weigh(self, entries, views, reference, depth_points, confidence, stale):
        """Per-view weights for one seed point, with the residual left out at <3 views."""
        ncc = self._ncc_scores(entries, views) if self.use_ncc else {}
        out = []
        for cam, uv in entries:
            gap = (depth_gap(views[cam], uv, reference)
                   if (self.use_depth_gate and reference is not None) else None)
            wv = view_weight(cam,
                             confidence=confidence.get(cam, 1.0) if self.use_confidence else 1.0,
                             stale_frames=stale.get(cam, 0) if self.use_stale else 0,
                             residual_px=None,
                             depth_gap_m=gap,
                             ncc=ncc.get(cam),
                             sigma_px=self.sigma_px, depth_tol_m=self.depth_tol_m,
                             depth_reject_m=self.depth_reject_m,
                             stale_halflife=self.stale_halflife)
            wv.terms["gap_m"] = 0.0 if gap is None else float(gap)
            out.append(wv)
        return out

    def _solve(self, entries, weights, keep, cameras, depth_points):
        """Triangulate the surviving views, or degrade in a documented order.

        Order: weighted triangulation over survivors -> single surviving view's depth
        deprojection -> uniform triangulation over *everything* (nothing survived, so the
        gate has no opinion worth acting on and suppressing the point would be worse).
        """
        if len(keep) >= 2:
            cams_for_point = [entries[i][0] for i in keep]
            projections = [camera_convert.projection_matrix_3x4(*cameras[entries[i][0]])
                           for i in keep]
            pixels = np.asarray([entries[i][1] for i in keep], dtype=float)
            prior = np.asarray([weights[i].weight for i in keep], dtype=float)
            point, residual = triangulate_weighted(
                projections, pixels, prior, iters=self.iters, sigma_px=self.sigma_px,
                use_residual=len(keep) >= self.residual_min_views)
            return point, residual, cams_for_point, "triangulated"

        if len(keep) == 1 and self.depth_fallback:
            cam = entries[keep[0]][0]
            point = depth_points.get(cam)
            if point is not None:
                return np.asarray(point, dtype=float), 0.0, [cam], "depth_fallback"

        cams_for_point = [c for c, _uv in entries]
        projections = [camera_convert.projection_matrix_3x4(*cameras[c]) for c in cams_for_point]
        pixels = np.asarray([uv for _c, uv in entries], dtype=float)
        point, residual = triangulate_weighted(
            projections, pixels, None, iters=self.iters, sigma_px=self.sigma_px,
            use_residual=len(entries) >= self.residual_min_views)
        return point, residual, cams_for_point, "triangulated"

    def _reference(self, previous_point, depth_points):
        """The 3D position every view's depth sample is judged against.

        The previous frame's accepted point is preferred: it is independent of *this*
        frame's measurements, so a camera that just jumped onto an occluder cannot vote on
        its own innocence. It is only trusted while some camera still agrees with it, so a
        genuinely fast-moving object drops the gate rather than rejecting every view.
        """
        candidates = [np.asarray(p, dtype=float) for p in depth_points.values() if p is not None]
        if previous_point is not None and candidates:
            gaps = [float(np.linalg.norm(c - np.asarray(previous_point, float)))
                    for c in candidates]
            if min(gaps) <= self.prev_max_gap_m:
                return np.asarray(previous_point, dtype=float)
            return None
        if len(candidates) >= 2:
            return np.median(np.stack(candidates), axis=0)
        return None

    def _ncc_scores(self, entries, views):
        """Each view's mean NCC against the other views at the same seed point."""
        scores = {}
        for i, (cam_a, uv_a) in enumerate(entries):
            values = []
            for j, (cam_b, uv_b) in enumerate(entries):
                if i == j:
                    continue
                value = patch_ncc(views[cam_a], uv_a, views[cam_b], uv_b, self.ncc_window)
                if value is not None:
                    values.append(value)
            if values:
                scores[cam_a] = float(np.mean(values))
        return scores

    @staticmethod
    def _depth_point(view, pixel_xy):
        depth = geometry.sample_depth(view, pixel_xy)
        if depth is None:
            return None
        return geometry.deproject_pixel_to_world(view, pixel_xy, depth)

    @staticmethod
    def _confidence(track):
        if track is None:
            return 1.0
        return float(np.clip(float(track.confidence) * max(float(track.health), 0.05), 0.0, 1.0))

    @staticmethod
    def _stale_frames(track):
        meta = getattr(track, "meta", None) or {}
        try:
            return float(meta.get("stale_frames", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _cheirality(cams, cameras, point):
        for cam in cams:
            row = camera_convert.mat4(cameras[cam][1])[2]
            if float(np.dot(row[:3], point) + row[3]) <= 0.0:
                return False
        return True


def _gray_patch(view, pixel_xy, window):
    if view is None or view.rgb is None:
        return None
    x, y = int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))
    half = max(1, int(window) // 2)
    if not (half <= x < view.width - half and half <= y < view.height - half):
        return None
    rgb = np.asarray(view.rgb)[y - half:y + half + 1, x - half:x + half + 1]
    if rgb.size == 0:
        return None
    if rgb.ndim == 3:
        rgb = rgb[..., :3].astype(float).mean(axis=2)
    return rgb.astype(float)


def _errors(projections, pixels, point):
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    errors = []
    for P, uv in zip(projections, pixels):
        projected = np.asarray(P, dtype=float).reshape(3, 4) @ homogeneous
        if abs(projected[2]) < 1e-12:
            errors.append(float("inf"))
            continue
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - uv)))
    return np.asarray(errors, dtype=float)


def _spread(points):
    if len(points) < 2:
        return None
    return float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
