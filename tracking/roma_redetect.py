"""RoMa v2 re-detection of a tracked object after its pose is lost (gated re-seed).

While the pose is trusted (green), the session banks a **reference** per camera: the full
RGB frame plus the pixels of the depth-verified seed points (:meth:`RomaRedetector.capture`).
When the pose goes unmeasured (red/black), :meth:`RomaRedetector.redetect` runs:

1. RoMa v2 matches each camera's reference frame (A) against the current frame (B) on the
   gpu_server (module ``romav2``). It reads the dense warp at the seed pixels, using the
   "full frame A, warp read at the query point" mode. That mode had a 2.2 px median on the
   door scene (docs/plans/3d_tracking.md §13.12); crops and NCC/SIFT failed.
2. Each warped pixel with RoMa overlap >= ``track_roma_overlap_min`` is lifted with this
   frame's depth. This gives 3D candidates per seed index, pooled over cameras.
3. The pose is a RANSAC rigid fit of the seed template onto those candidates
   (:func:`tracking.reacquire.ransac_pose`), with the same gates as the NCC search: the
   inlier count, RMSE, spread, a plausible jump from the hypotheses, and free-space.

The session adopts a hit as an external pose (``source="roma"``). It re-seeds the 2D
trackers there and re-initialises the pose filter.

RoMa needs torch >= 2.4, so it normally runs in its own env, on its own server
(``config.track_roma_url``, default ``ws://127.0.0.1:8775/v1/ws``). The launcher starts that
server with the ``romav2`` module profile (``config.gpu_server_module_profiles``). If the
server cannot be reached, re-detection is disabled for the session with one warning, and
tracking continues without it.
"""

import logging
import os
import time

import numpy as np

import config
from tracking import geometry

logger = logging.getLogger(__name__)


def roma_url():
    return os.environ.get("GPU_SERVER_ROMA_URL") or getattr(
        config, "track_roma_url", "ws://127.0.0.1:8775/v1/ws")


class RomaRef:
    """One camera's reference: full RGB at a trusted pose and the seed pixels it images."""

    def __init__(self, rgb, pixels, indices, frame):
        self.rgb = rgb
        self.pixels = np.asarray(pixels, float).reshape(-1, 2)
        self.indices = np.asarray(indices, int)
        self.frame = frame


class RomaRedetector:
    """Client-side RoMa re-detection. ``client`` may be injected (tests)."""

    def __init__(self, url=None, token=None, codec=None, client=None, launch=True):
        self.url = url or roma_url()
        self.token = token
        self.codec = codec or os.environ.get("GPU_SERVER_CODEC") or "raw"
        self._client = client
        self._launch = launch and client is None
        self.disabled = None             # reason string once disabled
        self.calls = 0
        self.hits = 0
        self.total_s = 0.0

    # -- server ------------------------------------------------------------
    def prewarm(self):
        """Start (or find) the RoMa server without waiting, so the model loads early."""
        if self._client is not None or not self._launch:
            return
        from gpu_server import launcher
        launcher.prewarm(self.url, "romav2")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.disabled:
            return None
        try:
            if self._launch:
                from gpu_server import launcher
                launcher.ensure_server(self.url, "romav2")
            from gpu_server.client import GpuClient
            self._client = GpuClient(self.url, timeout=float(getattr(config, "track_roma_timeout_s", 120.0)),
                                     token=self.token, image_codec=self.codec)
            try:
                from gpu_server import launcher
                launcher.register_client(self._client)
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001 - tracking must survive a missing RoMa
            self.disabled = f"{type(exc).__name__}: {exc}"
            logger.warning("[roma] re-detection disabled for this session: %s", self.disabled)
            return None
        return self._client

    def close(self):
        if self._client is not None and self._launch:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None

    # -- references --------------------------------------------------------
    @staticmethod
    def capture(view, pose, seed_points_world, frame):
        """A :class:`RomaRef` for ``view`` at ``pose`` or ``None`` (too few visible points)."""
        if view is None or getattr(view, "rgb", None) is None:
            return None
        world = pose.apply(seed_points_world)
        pixels, indices = [], []
        for i, wp in enumerate(world):
            visible, px, _d = geometry.is_visible(view, wp)
            if visible:
                pixels.append(px)
                indices.append(i)
        if len(indices) < int(config.track_pose_min_points):
            return None
        return RomaRef(np.array(view.rgb[..., :3], dtype=np.uint8, copy=True), pixels, indices, frame)

    # -- re-detection ------------------------------------------------------
    def match(self, ref, rgb):
        """``(points_b (N,2), overlap (N,), match_s)`` of ``ref``'s seed pixels in ``rgb``."""
        client = self._get_client()
        if client is None:
            return None
        reply, out = client.call("romav2", "match", img_a=ref.rgb,
                                 img_b=np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8),
                                 points_a=ref.pixels)
        return out["points_b"], out["overlap"], float(reply.get("match_s") or 0.0)

    def redetect(self, views, refs, seed_points_world, hypotheses, max_dev_m=None):
        """``(R, t, info)`` or ``(None, None, info)``; ``hypotheses`` as in
        :func:`tracking.reacquire.reacquire` (only their centroids are used, for the jump gate)."""
        from tracking.reacquire import free_space_counts, ransac_pose

        tmpl = np.asarray(seed_points_world, float).reshape(-1, 3)
        info = {"cams": {}}
        pooled = {}
        t0 = time.perf_counter()
        ov_min = float(getattr(config, "track_roma_overlap_min", 0.3))
        for cam, ref in refs.items():
            view = views.get(cam)
            if view is None or ref is None or getattr(view, "rgb", None) is None:
                continue
            try:
                res = self.match(ref, view.rgb)
            except Exception as exc:  # noqa: BLE001 - one failed call is not fatal
                info["cams"][cam] = f"error: {type(exc).__name__}: {exc}"
                continue
            if res is None:
                info["why"] = f"disabled: {self.disabled}"
                return None, None, info
            pb, ov, match_s = res
            self.calls += 1
            keep = np.isfinite(pb).all(axis=1) & (ov >= ov_min)
            n_lift = 0
            if keep.any():
                world, valid = geometry.points_to_world(view, pb[keep])
                for i, w, ok in zip(ref.indices[keep], world, valid):
                    if ok:
                        pooled.setdefault(int(i), []).append(w)
                        n_lift += 1
            info["cams"][cam] = {"ref_frame": ref.frame, "n": int(len(ref.indices)),
                                 "overlap_ok": int(keep.sum()), "lifted": n_lift,
                                 "overlap_med": round(float(np.median(ov)), 3) if len(ov) else None,
                                 "match_s": round(match_s, 3)}
        dt = time.perf_counter() - t0
        self.total_s += dt
        info["wall_s"] = round(dt, 3)
        cands = {i: np.array(v) for i, v in pooled.items()}
        n_ref = len(set().union(*[set(r.indices.tolist()) for r in refs.values() if r is not None])) \
            if refs else 0
        need = max(int(config.track_pose_min_points),
                   int(np.ceil(float(getattr(config, "track_roma_min_frac", 0.5)) * max(n_ref, 1))))
        inlier_m = float(getattr(config, "track_roma_inlier_m", 0.015))
        fit = ransac_pose(tmpl, cands, int(getattr(config, "track_roma_iters", 200)), inlier_m, need)
        if fit is None:
            info["why"] = f"no rigid consensus ({len(cands)} matched of {n_ref}, need {need})"
            return None, None, info
        R, t, inl, rmse = fit
        info.update(inliers=len(inl), need=need, rmse_m=round(rmse, 4))
        if rmse > float(getattr(config, "track_roma_max_rmse_m", config.track_pose_green_rmse_m)):
            info["why"] = f"rmse {rmse:.4f}"
            return None, None, info
        centre = R @ tmpl.mean(axis=0) + t
        if max_dev_m is not None and hypotheses:
            dev = min(float(np.linalg.norm(centre - pose.apply(tmpl.mean(axis=0))[0]))
                      for _n, pose, _r in hypotheses)
            info["dev_m"] = round(dev, 3)
            if dev > float(max_dev_m):
                info["why"] = f"implausible jump {dev:.3f} > {float(max_dev_m):.3f} m"
                return None, None, info
        support, contra = free_space_counts(views, tmpl @ np.asarray(R).T + np.asarray(t))
        info["free_space"] = f"{support}+/{contra}-"
        max_contra = float(getattr(config, "track_reacq_max_contra_frac", 1.0))
        if contra > 0 and contra > max_contra * (support + contra):
            info["why"] = f"free space in {contra} of {support + contra} camera views"
            return None, None, info
        self.hits += 1
        return R, t, info

    def summary(self):
        return {"url": self.url, "calls": self.calls, "hits": self.hits,
                "total_s": round(self.total_s, 2), "disabled": self.disabled}
