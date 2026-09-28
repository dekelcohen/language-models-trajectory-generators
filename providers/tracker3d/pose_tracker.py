"""``pose_tracker``: the full point-tracking pipeline as a 3D lift provider.

Per frame, per object (step numbers follow the design in ``docs/plans/3d_tracking.md`` §9):

1.2  **Epipolar gate.** For every seed point seen by two cameras, the distance of its pixel
     in one camera to the epipolar line of its pixel in the other. At two cameras this is
     the *only* purely geometric consistency check there is (one number per point), and it
     only catches drift *across* the line - drift along it is invisible to it.
2    **Per-point depth lift** + **jump gate**: each camera deprojects each point through its
     own depth buffer; a point further than ``jump_m`` (+ the filter's uncertainty) from
     where the Kalman filter predicts that seed point is rejected - an occluder stepped in
     front of *that point*.
3    **Per-point fusion across cameras** by seed index, weighted ``confidence / z^2``
     (depth noise of real RGB-D grows ~z^2, so the close wrist camera should dominate).
     A point the cameras disagree on keeps the reading nearer the prediction.
4    **Rigid fit** (Kabsch, *no* scale - the object does not change size) of the fused
     points against the seed-time template -> relative 6-DoF pose ``R, t``.
5    **Occlusion hierarchy**:
       green  - fused fit ok -> update the Kalman filter, output a measured pose;
       yellow - fused fit failed, one camera's own fit ok -> use it, and ask the session to
                force-correct the other cameras from that pose (``meta['correct_cams']``);
       red    - no camera fits -> Kalman prediction, ``predicted=True``, and ask the session
                to re-acquire (``meta['reacquire']``: re-seed where the predicted points are
                depth-verified visible);
       black  - red for more than ``black_after`` frames. The session may then follow the
                gripper's FK if the object is grasped; still ``predicted=True``.
6    **Wrong-lock guards.** A filter re-init needs a fit supported by ``reset_min_frac`` of
     the template spanning ``reset_min_spread`` of its extent (a few points that slid onto
     the same wrong surface agree perfectly, but cannot cover the object). The first fits
     after a re-init or after ``confirm_after`` unmeasured frames are *tentative* for
     ``confirm_frames``: reported yellow with ``pose.tentative``, and they do not
     force-correct other cameras - so a wrong lock cannot copy itself into every camera.

Rotation is only fitted when the tracked points span at least ``min_rot_extent_m`` and are
not collinear. With a tight seed cluster (5 points within a few pixels, today's affordance
seeding) the rotation would be noise, so the pose is translation-only and says so
(``meta['rotation_observable'] = False``) - see ``tracking.seeding`` for spread seeding.

The pose is **relative to seed time** (there is no object model): ``current_i = R @ seed_i + t``.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import copy

import numpy as np

import config
from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D
from tracking import geometry
from tracking.kalman import KFConfig, RigidBodyKF
from tracking.motion import fit_rigid_motion, is_collinear
from tracking.types import ObjectPose

MODE_GREEN, MODE_YELLOW, MODE_RED, MODE_BLACK = "green", "yellow", "red", "black"
#: No fresh 2D measurement this frame because the windowed tracker has not flushed yet.
MODE_COAST = "coast"


@dataclass
class _ObjState:
    template: np.ndarray                     # (N, 3) seed-time world points
    centroid: np.ndarray                     # (3,)  template centroid
    kf: RigidBodyKF
    unmeasured_frames: float = 0             # in reference frames (see tracking/kalman.py)
    last_mode: str = ""
    history: Dict[str, int] = field(default_factory=dict)
    last_time: Optional[float] = None        # sim time of the last lift (real-time mode)
    confirm_left: float = 0.0                # ref frames of agreeing fits still needed (tentative)


def epipolar_distance(view_a, px_a, view_b, px_b, near=0.05, far=4.0):
    """Pixel distance of ``px_b`` to the epipolar line of ``px_a`` in ``view_b``.

    The line is the image in B of A's viewing ray, built from two depths on that ray, so it
    needs nothing but the cameras' own view/projection matrices (no fundamental matrix to
    get wrong across the OpenGL/OpenCV convention boundary). ``None`` when degenerate.
    """
    ends = []
    for depth in (near, far):
        world = geometry.deproject_pixel_to_world(view_a, px_a, depth)
        pixel, z = geometry.project_world_to_pixel(view_b, world)
        if pixel is None:
            return None
        ends.append(pixel)
    a, b = ends
    d = b - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    p = np.asarray(px_b, dtype=float) - a
    return float(abs(d[0] * p[1] - d[1] * p[0]) / norm)


class PoseTracker3D(MultiCamTracker3D):
    name = "pose_tracker"
    needs_depth = True
    #: Holds a Kalman filter per object - the session must not advance it twice a frame.
    stateful = True

    def __init__(self, epipolar_px=None, jump_m=None, fuse_tol_m=None, green_rmse_m=None,
                 min_inlier_frac=None, min_points=None, min_rot_extent_m=None, black_after=None,
                 max_jump_gate_m=None, kf_config=None, inlier_m=None, inlier_median_k=None,
                 reset_min_frac=None, reset_min_spread=None, confirm_frames=None,
                 confirm_after=None, px_noise=None, max_rot_std=None, **kwargs):
        super().__init__(**kwargs)

        def pick(value, key):
            return getattr(config, key) if value is None else value

        self.epipolar_px = float(pick(epipolar_px, "track_pose_epipolar_px"))
        self.jump_m = float(pick(jump_m, "track_pose_jump_m"))
        self.max_jump_gate_m = float(pick(max_jump_gate_m, "track_pose_max_jump_gate_m"))
        self.fuse_tol_m = float(pick(fuse_tol_m, "track_pose_fuse_tol_m"))
        self.green_rmse_m = float(pick(green_rmse_m, "track_pose_green_rmse_m"))
        self.inlier_m = float(pick(inlier_m, "track_pose_inlier_m"))
        self.inlier_median_k = float(pick(inlier_median_k, "track_pose_inlier_median_k"))
        self.min_inlier_frac = float(pick(min_inlier_frac, "track_pose_min_inlier_frac"))
        self.min_points = int(pick(min_points, "track_pose_min_points"))
        self.min_rot_extent_m = float(pick(min_rot_extent_m, "track_pose_min_rot_extent_m"))
        self.black_after = int(pick(black_after, "track_pose_black_after"))
        self.reset_min_frac = float(pick(reset_min_frac, "track_pose_reset_min_frac"))
        self.reset_min_spread = float(pick(reset_min_spread, "track_pose_reset_min_spread"))
        self.confirm_frames = float(pick(confirm_frames, "track_pose_confirm_frames"))
        self.confirm_after = float(pick(confirm_after, "track_pose_confirm_after"))
        self.px_noise = float(pick(px_noise, "track_pose_px_noise"))
        self.max_rot_std = float(pick(max_rot_std, "track_pose_max_rot_std"))
        self.kf_config = kf_config or KFConfig()
        self._objects: Dict[str, _ObjState] = {}
        self._last_fit_reject = None

    def reset(self):
        self._objects.clear()

    def set_time(self, sim_time):
        """Real-time mode: the sim time of the frame the next :meth:`lift` measures.

        The filter then steps ``elapsed * track_kf_ref_hz`` reference frames instead of 1,
        so its tuning holds at any camera rate. ``None`` (the default) = 1 per lift."""
        self._time = None if sim_time is None else float(sim_time)

    def _step_dt(self, st):
        now = getattr(self, "_time", None)
        if now is None:
            return 1.0
        last, st.last_time = st.last_time, now
        if last is None:
            return 1.0
        return max(0.0, (now - last) * float(config.track_kf_ref_hz))

    # -- state ----------------------------------------------------------------
    def _state(self, obj_name, seed_world_points):
        st = self._objects.get(obj_name)
        if st is None and seed_world_points is not None:
            tmpl = np.asarray(seed_world_points, dtype=float).reshape(-1, 3)
            st = _ObjState(template=tmpl, centroid=tmpl.mean(axis=0),
                           kf=RigidBodyKF(self.kf_config))
            self._objects[obj_name] = st
        return st

    def extend_template(self, obj_name, seed_frame_points):
        """Append points (in the seed-time frame) to ``obj_name``'s template - a camera that
        joins seeing a different face tracks points the seed camera never saw. The centroid
        (the pose's reference point) is unchanged. Returns the first new seed index."""
        st = self._objects.get(obj_name)
        if st is None:
            return None
        pts = np.asarray(seed_frame_points, dtype=float).reshape(-1, 3)
        first = int(st.template.shape[0])
        st.template = np.vstack([st.template, pts])
        return first

    def inject_pose(self, obj_name, R, t, pos_std=None, rot_std=None, reason="external"):
        """Re-initialise ``obj_name``'s filter on an externally found pose (search-based
        re-acquisition, gripper FK). ``current_i = R @ seed_i + t``. The next fits are
        tentative, so a wrong external pose must still be confirmed by the trackers."""
        st = self._objects.get(obj_name)
        if st is None:
            return False
        R = np.asarray(R, dtype=float).reshape(3, 3)
        p = R @ st.centroid + np.asarray(t, dtype=float).reshape(3)
        st.kf.init(p, R, pos_std=pos_std, rot_std=rot_std)
        st.unmeasured_frames = 0
        st.confirm_left = self.confirm_frames
        if self.logger is not None:
            self._log(f"[pose_tracker] '{obj_name}' filter re-initialised from {reason} pose")
        return True

    def _rotation_observable(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if pts.shape[0] < self.min_points:
            return False
        extent = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        return extent >= self.min_rot_extent_m and not is_collinear(pts)

    # -- per-camera measurement ------------------------------------------------
    def _measure_cams(self, views, cam_tracks, point_index, st, predicted_pts, gate_m, meta):
        """``{cam: {seed_idx: (P (3,), pixel (2,), z, conf)}}`` after the jump gate."""
        per_cam = {}
        rejected_jump = {}
        stale = []
        n_template = st.template.shape[0]
        for cam, (pts, vis) in self._usable(cam_tracks).items():
            view = views.get(cam)
            if view is None:
                continue
            if int(getattr(cam_tracks[cam], "stale_frames", 0) or 0) > 0:
                # Points computed frames ago, sampled against *this* frame's depth: a
                # measurement of nothing. Between flushes the filter's prediction is better.
                stale.append(cam)
                continue
            idx = self._indices_for(cam, pts, point_index)
            world, valid = geometry.points_to_world(view, pts, vis)
            conf = float(np.clip(getattr(cam_tracks[cam], "confidence", 1.0) or 0.0, 0.05, 1.0))
            entries = {}
            for j in range(pts.shape[0]):
                i = int(idx[j])
                if not valid[j] or i < 0 or i >= n_template:
                    continue
                if predicted_pts is not None and \
                        np.linalg.norm(world[j] - predicted_pts[i]) > gate_m:
                    rejected_jump[cam] = rejected_jump.get(cam, 0) + 1
                    continue
                _px, z = geometry.project_world_to_pixel(view, world[j])
                entries[i] = (world[j], pts[j], max(float(z or 1.0), 0.05), conf)
            if entries:
                per_cam[cam] = entries
        meta["rejected_jump"] = rejected_jump
        if stale:
            meta["stale_cams"] = stale
        return per_cam

    def _epipolar_gate(self, views, per_cam, predicted_pts, meta):
        cams = sorted(per_cam)
        rejected = {}
        worst = 0.0
        for a_i in range(len(cams)):
            for b_i in range(a_i + 1, len(cams)):
                a, b = cams[a_i], cams[b_i]
                for i in sorted(set(per_cam[a]) & set(per_cam[b])):
                    Pa, pxa, _za, _ca = per_cam[a][i]
                    Pb, pxb, _zb, _cb = per_cam[b][i]
                    d1 = epipolar_distance(views[a], pxa, views[b], pxb)
                    d2 = epipolar_distance(views[b], pxb, views[a], pxa)
                    if d1 is None or d2 is None:
                        continue
                    d = 0.5 * (d1 + d2)
                    worst = max(worst, d)
                    if d <= self.epipolar_px:
                        continue
                    if predicted_pts is not None:
                        blame = [a if np.linalg.norm(Pa - predicted_pts[i])
                                 > np.linalg.norm(Pb - predicted_pts[i]) else b]
                    else:
                        blame = [a, b]
                    for cam in blame:
                        if i in per_cam[cam]:
                            del per_cam[cam][i]
                            rejected[cam] = rejected.get(cam, 0) + 1
        meta["epipolar_worst_px"] = round(worst, 2)
        meta["rejected_epipolar"] = rejected
        for cam in [c for c, e in per_cam.items() if not e]:
            del per_cam[cam]

    def _fuse_points(self, per_cam, predicted_pts):
        fused, disagreements = {}, []
        for i in sorted(set().union(*[set(e) for e in per_cam.values()])) if per_cam else []:
            readings = [(cam, per_cam[cam][i]) for cam in per_cam if i in per_cam[cam]]
            if len(readings) == 1:
                fused[i] = readings[0][1][0]
                continue
            pts = np.array([r[1][0] for r in readings])
            spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) * 2.0)
            disagreements.append(spread)
            if spread <= self.fuse_tol_m:
                w = np.array([r[1][3] / (r[1][2] ** 2) for r in readings])
                fused[i] = (w[:, None] * pts).sum(axis=0) / w.sum()
            elif predicted_pts is not None:
                fused[i] = pts[int(np.argmin(np.linalg.norm(pts - predicted_pts[i], axis=1)))]
            else:
                # no prediction yet: trust the nearer camera (smaller depth noise)
                fused[i] = pts[int(np.argmin([r[1][2] for r in readings]))]
        return fused, (float(np.median(disagreements)) if disagreements else None)

    def _lateral_sigma(self, views, per_cam):
        """``{seed_idx: sigma_m}``: the 3D error one ``px_noise`` of 2D tracking noise makes
        at that point (``px_noise * z / f``), from its best camera."""
        out = {}
        if self.px_noise <= 0:
            return out
        for cam, entries in per_cam.items():
            view = views.get(cam)
            if view is None:
                continue
            P = geometry._mat4(view.projection_matrix)
            f_px = abs(float(P[1, 1])) * float(view.height) / 2.0
            if f_px <= 0:
                continue
            for i, (_P, _px, z, _c) in entries.items():
                s = self.px_noise * float(z) / f_px
                out[i] = min(out.get(i, s), s)
        return out

    def _rotation_std(self, tmpl, idx, sigma):
        """Std (rad) the 2D noise alone puts on a rotation fitted to ``tmpl`` rows ``idx``
        (``sigma``: per-point 3D noise, :meth:`_lateral_sigma`), floored at the filter's
        ``rot_meas_std``. A small, far cluster cannot measure rotation (256 px: 1 px ~ 4 mm
        at 0.85 m ~ 10 deg on a 2 cm lever), and the fit hides it in a *small* RMSE."""
        s = [sigma[i] for i in idx if i in sigma]
        r_rms = float(np.sqrt(np.mean(np.sum((tmpl - tmpl.mean(axis=0)) ** 2, axis=1))))
        if not s or r_rms <= 1e-9:
            return self.kf_config.rot_meas_std
        rs = float(np.sqrt(np.mean(np.square(s)))) / (r_rms * np.sqrt(len(s)))
        return max(rs, self.kf_config.rot_meas_std)

    def _fit(self, st, points_by_idx, sigma=None):
        """Relative pose from ``{seed_idx: P}``.

        Returns ``(R, p_centroid, rmse, n, rot_std, used_idx)``: ``rot_std`` (rad) is the
        rotation's measurement noise, ``None`` when rotation is unobservable (then R is held);
        ``used_idx`` are the seed ids that survived trimming (see :meth:`_support_ok`)."""
        if not points_by_idx:
            return None
        idx = np.array(sorted(points_by_idx), dtype=int)
        cur = np.array([points_by_idx[i] for i in idx])
        tmpl = st.template[idx]
        rot_std = self._rotation_std(tmpl, idx, sigma or {})
        if len(idx) >= self.min_points and self._rotation_observable(tmpl) and \
                rot_std <= self.max_rot_std:
            fit = fit_rigid_motion(tmpl, cur, min_points=self.min_points,
                                   trim_m=self.inlier_m, trim_median_k=self.inlier_median_k)
            if not np.isfinite(fit.rmse):
                return None
            if fit.n_used / float(len(idx)) < self.min_inlier_frac:
                self._last_fit_reject = f"inliers {fit.n_used}/{len(idx)} rmse={fit.rmse:.4f}"
                return None
            used = idx[np.asarray(fit.inliers, dtype=bool)]
            return fit.R, fit.R @ st.centroid + fit.t, fit.rmse, fit.n_used, rot_std, used
        else:
            # Translation only: rotation is unobservable from this point set, so hold the
            # filter's rotation (identity before the first rotation-observable frame).
            R = st.kf.rotation if st.kf.initialized else np.eye(3)
            shifted = cur - (tmpl - st.centroid) @ R.T
            p = np.median(shifted, axis=0)
            rmse = float(np.sqrt(np.mean(np.sum((shifted - p) ** 2, axis=1))))
            return R, p, rmse, len(idx), None, idx

    def _support_ok(self, st, used_idx):
        """Is a fit supported widely enough to *replace* the track (a filter re-init)?

        A handful of points that slid onto the same wrong surface (the door frame, an
        occluder) agree with each other perfectly, so a low RMSE proves nothing; what they
        cannot fake is covering most of the object. Needs ``reset_min_frac`` of the template
        and, when the template has a real extent, ``reset_min_spread`` of that extent.
        """
        used = np.asarray(used_idx, dtype=int).reshape(-1)
        n_tmpl = st.template.shape[0]
        need = max(self.min_points, int(np.ceil(self.reset_min_frac * n_tmpl)))
        if used.size < need:
            return False, f"support {used.size}/{n_tmpl} < {need}"
        full = float(np.max(np.linalg.norm(st.template - st.centroid, axis=1)))
        if full >= self.min_rot_extent_m:
            sub = st.template[used]
            extent = float(np.max(np.linalg.norm(sub - sub.mean(axis=0), axis=1)))
            if extent < self.reset_min_spread * full:
                return False, f"spread {extent:.3f}/{full:.3f} m"
        return True, ""

    # -- main -----------------------------------------------------------------
    def lift(self, obj_name, views, cam_tracks, point_index=None, seed_world_points=None,
             commit=True):
        if commit:
            return self._lift(obj_name, views, cam_tracks, point_index, seed_world_points)
        saved = copy.deepcopy(self._objects.get(obj_name))
        logger, self.logger = self.logger, None          # a dry run must not log transitions
        try:
            return self._lift(obj_name, views, cam_tracks, point_index, seed_world_points)
        finally:
            self.logger = logger
            if saved is None:
                self._objects.pop(obj_name, None)
            else:
                self._objects[obj_name] = saved

    def _lift(self, obj_name, views, cam_tracks, point_index, seed_world_points):
        meta = {"provider": self.name}
        st = self._state(obj_name, seed_world_points)
        if st is None:
            meta["why"] = "no seed template"
            return Lift3DResult(meta=meta)

        predicted_pts = None
        gate_m = None
        dt = self._step_dt(st)
        meta["dt"] = round(dt, 4)
        if st.kf.initialized:
            st.kf.predict(dt)
            if not st.kf.stale:
                R_pred, p_pred = st.kf.rotation, st.kf.position
                predicted_pts = (st.template - st.centroid) @ R_pred.T + p_pred
                gate_m = min(self.jump_m + 3.0 * st.kf.pos_std, self.max_jump_gate_m)
                meta["jump_gate_m"] = round(gate_m, 4)

        per_cam = self._measure_cams(views, cam_tracks, point_index, st, predicted_pts,
                                     gate_m, meta)
        self._epipolar_gate(views, per_cam, predicted_pts, meta)
        fused, disagreement = self._fuse_points(per_cam, predicted_pts)
        meta["n_fused"] = len(fused)
        sigma = self._lateral_sigma(views, per_cam)

        mode, fit, fit_cams, correct = None, None, list(per_cam), []
        # With fewer than ``min_points`` points a fit is trivially perfect (1 point -> rmse 0)
        # and so proves nothing: it would let one point sliding off the object drag the
        # whole estimate. Once the filter has a track, such frames are not measurements.
        need = self.min_points if st.kf.initialized else 1
        if len(fused) >= need:
            self._last_fit_reject = None
            fit = self._fit(st, fused, sigma)
            if self._last_fit_reject:
                meta["fit_reject"] = self._last_fit_reject
            if fit is not None:
                meta["fit_rmse_m"] = round(float(fit[2]), 4)
                meta["fit_n"] = int(fit[3])
            if fit is not None and fit[2] <= self.green_rmse_m:
                mode = MODE_GREEN
        if mode is None and len(per_cam) >= 2:
            best = None
            for cam, entries in per_cam.items():
                if len(entries) < need:
                    continue
                single = self._fit(st, {i: e[0] for i, e in entries.items()}, sigma)
                if single is not None and single[2] <= self.green_rmse_m and \
                        (best is None or single[2] < best[1][2]):
                    best = (cam, single)
            if best is not None:
                mode, fit, fit_cams = MODE_YELLOW, best[1], [best[0]]
                correct = [c for c in cam_tracks if c != best[0]]
        # NOTE: a "degraded" level (accept 1.5-3.5 cm fits and re-anchor every camera at the
        # fitted pose) was tried on the door scene and made it worse (median 2.4 -> 5.6 cm):
        # re-seeding from a pose that is already ~2 cm off writes the error into fresh
        # tracks. Fits above ``green_rmse_m`` are therefore not measurements.

        result = Lift3DResult(meta=meta, disagreement=disagreement)
        if mode is not None:
            R, p, rmse, n, rot_std, used = fit
            support_ok, why_weak = self._support_ok(st, used)
            if not support_ok:
                meta["weak_support"] = why_weak
            was_unmeasured = st.unmeasured_frames
            # An unobservable rotation is not a measurement of "no rotation": inflate its
            # noise so the filter's rotation estimate is not falsely tightened.
            if rot_std is not None:
                meta["rot_std_deg"] = round(float(np.degrees(rot_std)), 1)
            kf_res = st.kf.update(p, R, pos_std=max(rmse, 1e-4),
                                  rot_std=1.0 if rot_std is None else rot_std,
                                  allow_reset=support_ok)
            meta["kf"] = kf_res.reason or "update"
            meta["kf_reset"] = bool(kf_res.reset)
            if kf_res.mahalanobis2 is not None:
                meta["kf_d2"] = round(kf_res.mahalanobis2, 2)
            if not kf_res.accepted:
                mode = None           # inconsistent with the motion model: treat as unmeasured
            else:
                st.unmeasured_frames = 0
                # The first fits after the filter was replaced, or after it coasted, are
                # tentative: a wrong lock (points on the door frame, one camera on the wrong
                # surface) looks exactly like a re-acquisition for a frame or two, and green
                # would let it force-correct the other cameras onto itself.
                if (kf_res.reset and kf_res.reason != "init") or \
                        was_unmeasured >= self.confirm_after - 1e-6:
                    st.confirm_left = self.confirm_frames
                tentative = st.confirm_left > 1e-6
                if tentative:
                    st.confirm_left = max(0.0, st.confirm_left - max(dt, 1e-3))
                    meta["tentative"] = True
                    if mode == MODE_GREEN:
                        mode = MODE_YELLOW
                    correct = []
                meta["rotation_observable"] = rot_std is not None
                pose_R, pose_p = (R, p) if kf_res.reset else (st.kf.rotation, st.kf.position)
                result.pose = ObjectPose(
                    R=pose_R, t=pose_p - pose_R @ st.centroid,
                    source="measured_single_cam" if mode == MODE_YELLOW and len(fit_cams) == 1
                    else "measured",
                    mode=mode, rmse_m=rmse, n_points=n,
                    velocity=st.kf.velocity, angular_velocity=st.kf.angular_velocity,
                    tentative=tentative)
                result.world_point = pose_p
                result.used_cams = list(fit_cams)
                result.per_point = np.array([fused[i] for i in sorted(fused)]) if fused else None
                result.per_point_index = np.array(sorted(fused), dtype=int) if fused else None
                if correct:
                    meta["correct_cams"] = correct

        if mode is None:
            coasting = bool(meta.get("stale_cams")) and not per_cam
            if fused and len(fused) < need:
                meta["too_few_points"] = len(fused)
            if not coasting:
                st.unmeasured_frames += dt
            if not st.kf.initialized or st.kf.stale:
                meta["why"] = "no measurement and no usable prediction"
                meta["mode"] = "lost"
                st.last_mode = "lost"
                return result
            if coasting:
                # Between two windowed-tracker flushes: nothing is wrong, there is simply no
                # new measurement yet. Predict, but do not escalate or ask for re-seeding.
                mode = MODE_COAST
            else:
                mode = MODE_BLACK if st.unmeasured_frames > self.black_after + 1e-6 else MODE_RED
            R, p = st.kf.rotation, st.kf.position
            result.pose = ObjectPose(R=R, t=p - R @ st.centroid, source="predicted", mode=mode,
                                     n_points=0, velocity=st.kf.velocity,
                                     angular_velocity=st.kf.angular_velocity)
            result.world_point = p
            result.predicted = True
            if not coasting:
                meta["reacquire"] = True
            meta["unmeasured_frames"] = st.unmeasured_frames

        meta["mode"] = mode
        if mode != st.last_mode and self.logger is not None:
            self._log(f"[pose_tracker] '{obj_name}' {st.last_mode or 'init'} -> {mode} "
                      f"(fused={meta['n_fused']}, epi_rej={meta.get('rejected_epipolar')}, "
                      f"jump_rej={meta.get('rejected_jump')})")
        st.last_mode = mode
        st.history[mode] = st.history.get(mode, 0) + 1
        return result
