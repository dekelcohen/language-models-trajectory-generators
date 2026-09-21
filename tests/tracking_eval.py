"""Provider-agnostic ground-truth scoring harness for the rollout tracker.

This is a *library* module (import it; pytest does not collect it). It runs a scripted
scene through the real :class:`tracking.session.TrackingSession` and scores the fused 3D
estimate against the simulator's ground truth, so that any combination of

    2D tracker provider  x  3D lift provider (``depth_fusion`` / ``triangulate`` / ``lapa``)

can be compared on identical frames with identical metrics.

Two things in here are easy to get wrong and would silently invalidate every number:

**1. The seed offset.** The tracker is seeded from points on the object's *visible
surface* (deprojected from the depth buffer at the affordance pixel), while the simulator
reports the object's *origin/COM*. The constant vector between them is not tracking error.
It is captured once at seed time and expressed in the object's **local frame**
(``R_seed^T @ (seed_centroid - origin_seed)``), so it rotates with the body:
``expected(t) = position(t) + R(t) @ offset_local``. When the scene driver reports no
orientation the offset degenerates to a world-frame constant, which is only correct while
the object does not rotate - :meth:`SceneDriver.ground_truth` should return a quaternion
whenever it can. ``test_tracking_eval.py`` pins this down by feeding ground-truth-derived
predictions through the scorer and asserting ~0 error.

**2. Lost frames.** Dropping lost frames from the error statistics would reward a tracker
that gives up: reporting ``lost`` every frame would leave an empty error list and a
flattering (or undefined) median. Instead *every* frame is scored: a frame with no usable
prediction contributes ``lost_penalty_m`` (default 1.0 m, i.e. "wrong by a scene") to the
error series. ``median_l2_valid_m`` is also emitted for diagnosis, clearly labelled as the
optimistic, valid-frames-only view.

Nothing here raises from scoring: a failed run still yields a payload, with the failure
recorded under ``"error"``.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from sim_adapter import camera_math, transforms  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.session import TrackingSession  # noqa: E402
from tracking.types import CameraView  # noqa: E402
from tracking_scenario import paint_occluder, surface_seed_points  # noqa: E402

log = logging.getLogger("tracking_eval")

SCHEMA = "tracking_eval/v1"

#: Error charged to a frame with no usable prediction (lost / dropped). Roughly the size
#: of the whole workspace, so "always lost" can never look accurate.
LOST_PENALTY_M = 1.0
#: Error below which the object counts as re-acquired after an occlusion window.
RECOVERY_THRESHOLD_M = 0.05


# -- ground-truth bookkeeping ---------------------------------------------
def quat_matrix(quat_xyzw):
    """3x3 rotation matrix for an xyzw quaternion (identity when ``quat`` is ``None``)."""
    if quat_xyzw is None:
        return np.eye(3)
    return np.asarray(transforms.matrix_from_quat(list(quat_xyzw)), dtype=float).reshape(3, 3)


@dataclass
class SeedOffset:
    """The constant surface-vs-origin bias captured when a target is seeded.

    ``local`` is the offset in the body frame; ``rotating`` records whether the scene
    driver supplied an orientation. With ``rotating=False`` the correction is a world-frame
    constant and is only valid while the object keeps its seed orientation.
    """

    local: np.ndarray
    rotating: bool = True

    def expected(self, position, quat=None):
        """Where a perfect tracker's fused point should sit for this ground-truth pose."""
        rot = quat_matrix(quat) if self.rotating else np.eye(3)
        return np.asarray(position, dtype=float) + rot @ np.asarray(self.local, dtype=float)

    def to_dict(self):
        return {"local": [round(float(v), 6) for v in self.local], "rotating": bool(self.rotating)}


def capture_seed_offset(seed_points, position, quat=None):
    """Offset from the object origin to the seed centroid, in the body frame."""
    centroid = np.asarray(seed_points, dtype=float).reshape(-1, 3).mean(axis=0)
    delta = centroid - np.asarray(position, dtype=float)
    if quat is None:
        return SeedOffset(local=delta, rotating=False)
    return SeedOffset(local=quat_matrix(quat).T @ delta, rotating=True)


@dataclass
class FrameSample:
    """One frame of one object: what was predicted, what was true, and at what cost."""

    frame: int
    obj: str
    predicted: Optional[np.ndarray] = None
    expected: Optional[np.ndarray] = None
    lost: bool = True
    occluded: bool = False
    dropped: bool = False               # the session returned no report for this frame
    disagreement: Optional[float] = None
    latency_ms: float = 0.0
    lift_meta: dict = field(default_factory=dict)

    @property
    def error_m(self) -> Optional[float]:
        """Measured L2 error, or ``None`` when this frame carries no usable prediction."""
        if self.lost or self.predicted is None or self.expected is None:
            return None
        err = float(np.linalg.norm(np.asarray(self.predicted, dtype=float)
                                   - np.asarray(self.expected, dtype=float)))
        return err if np.isfinite(err) else None

    def to_dict(self):
        return {
            "frame": self.frame,
            "obj": self.obj,
            "predicted": None if self.predicted is None
            else [round(float(v), 6) for v in np.asarray(self.predicted).reshape(-1)],
            "expected": None if self.expected is None
            else [round(float(v), 6) for v in np.asarray(self.expected).reshape(-1)],
            "error_m": None if self.error_m is None else round(self.error_m, 6),
            "lost": bool(self.lost),
            "occluded": bool(self.occluded),
            "dropped": bool(self.dropped),
            "disagreement": None if self.disagreement is None else round(float(self.disagreement), 6),
            "latency_ms": round(float(self.latency_ms), 3),
            "lift": dict(self.lift_meta or {}),
        }


# -- metrics ---------------------------------------------------------------
def _percentile(values, pct):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _median(values):
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def peak_vram_mb():
    """Peak CUDA allocation in MB, or ``None`` on a CPU-only install."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0), 3)
    except Exception:
        return None


def reset_vram_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def scored_errors(samples, lost_penalty_m=LOST_PENALTY_M):
    """Per-frame error with lost/dropped frames charged ``lost_penalty_m``.

    This is the series every headline metric is computed from; see the module docstring
    for why lost frames are penalised rather than dropped.
    """
    out = []
    for sample in samples:
        err = sample.error_m
        out.append(lost_penalty_m if err is None else err)
    return out


def max_jump_m(samples):
    """Largest prediction displacement between two *consecutive* scored frames.

    Frames with no prediction break the chain instead of being bridged: a tracker that
    goes lost and comes back somewhere else is caught by ``frames_to_recover``, not by a
    fake jump across the gap.
    """
    worst, previous = None, None
    for sample in samples:
        point = None if sample.lost else sample.predicted
        if point is not None and previous is not None and sample.frame == previous[0] + 1:
            jump = float(np.linalg.norm(np.asarray(point, dtype=float) - previous[1]))
            if np.isfinite(jump) and (worst is None or jump > worst):
                worst = jump
        previous = None if point is None else (sample.frame, np.asarray(point, dtype=float))
    return worst


def frames_to_recover(samples, errors, threshold_m=RECOVERY_THRESHOLD_M):
    """Frames from the end of the occlusion window until the error drops below threshold.

    ``0`` means the very first post-occlusion frame was already accurate; ``None`` means
    there was no occlusion window, or the tracker never recovered within the run.
    """
    occluded_idx = [i for i, s in enumerate(samples) if s.occluded]
    if not occluded_idx:
        return None
    last = occluded_idx[-1]
    for i in range(last + 1, len(samples)):
        if errors[i] <= threshold_m:
            return i - last - 1
    return None


def compute_metrics(samples, lost_penalty_m=LOST_PENALTY_M,
                    recovery_threshold_m=RECOVERY_THRESHOLD_M):
    """Score one object's frame samples. Never raises - failures land in ``"error"``."""
    try:
        samples = sorted(samples, key=lambda s: s.frame)
        n_frames = len(samples)
        if n_frames == 0:
            return {"n_frames": 0, "n_valid_frames": 0, "error": "no frames"}

        errors = scored_errors(samples, lost_penalty_m)
        valid = [s.error_m for s in samples if s.error_m is not None]
        occl_errors = [e for e, s in zip(errors, samples) if s.occluded]
        latencies = [float(s.latency_ms) for s in samples]
        n_lost = sum(1 for s in samples if s.lost or s.predicted is None)

        return {
            "n_frames": n_frames,
            "n_valid_frames": len(valid),
            "median_l2_m": _median(errors),
            "p95_l2_m": _percentile(errors, 95),
            "max_l2_m": max(errors) if errors else None,
            # Optimistic view: ignores the frames the tracker gave up on. Diagnosis only -
            # never compare providers on this one.
            "median_l2_valid_m": _median(valid),
            "pct_lost": round(100.0 * n_lost / n_frames, 3),
            "n_dropped_frames": sum(1 for s in samples if s.dropped),
            "err_during_occlusion_m": _median(occl_errors),
            "n_occluded_frames": len(occl_errors),
            "frames_to_recover": frames_to_recover(samples, errors, recovery_threshold_m),
            "max_jump_m": max_jump_m(samples),
            "median_disagreement_m": _median([float(s.disagreement) for s in samples
                                              if s.disagreement is not None]),
            "latency_ms_mean": round(float(np.mean(latencies)), 3) if latencies else None,
            "latency_ms_p95": round(_percentile(latencies, 95), 3) if latencies else None,
            "lost_penalty_m": float(lost_penalty_m),
            "recovery_threshold_m": float(recovery_threshold_m),
        }
    except Exception as exc:                        # scoring must never break a run
        log.warning("[eval] metric computation failed: %s: %s", type(exc).__name__, exc)
        return {"n_frames": len(samples), "n_valid_frames": 0,
                "error": f"{type(exc).__name__}: {exc}"}


#: Metrics that aggregate across objects by taking the worst (rather than the mean) value.
_WORST_KEYS = ("p95_l2_m", "max_l2_m", "max_jump_m", "pct_lost", "latency_ms_p95")
_MEAN_KEYS = ("median_l2_m", "median_l2_valid_m", "err_during_occlusion_m", "latency_ms_mean",
              "median_disagreement_m")


def aggregate_metrics(per_object):
    """Roll per-object metrics into one scene-level summary.

    Central tendencies are averaged; tail metrics take the worst object, because a scene is
    only as good as the object the tracker handled worst.
    """
    metrics = [m for m in per_object.values() if isinstance(m, dict) and m.get("n_frames")]
    if not metrics:
        return {"n_objects": 0}
    out = {"n_objects": len(metrics),
           "n_frames": int(sum(m.get("n_frames", 0) for m in metrics)),
           "n_valid_frames": int(sum(m.get("n_valid_frames", 0) for m in metrics))}
    for key in _MEAN_KEYS:
        values = [m[key] for m in metrics if m.get(key) is not None]
        out[key] = round(float(np.mean(values)), 6) if values else None
    for key in _WORST_KEYS:
        values = [m[key] for m in metrics if m.get(key) is not None]
        out[key] = round(float(np.max(values)), 6) if values else None
    recoveries = [m["frames_to_recover"] for m in metrics if m.get("frames_to_recover") is not None]
    out["frames_to_recover"] = int(max(recoveries)) if recoveries else None
    return out


# -- scene drivers ---------------------------------------------------------
class SceneDriver:
    """What the runner needs from a scene: motion, ground truth, and an occlusion window.

    Implementations own the scripted motion; the runner owns timing, scoring and reporting.
    A driver is simulator-agnostic in exactly the same way ``tracking_scenario`` is: it is
    handed an already-booted sim/env/robot.
    """

    name = "scene"
    objects: Sequence[str] = ()
    n_frames = 0

    def reset(self):
        """Put the scene back at its start pose. Called once before seeding."""

    def select_cameras(self, names):
        """Restrict rendering to the cameras the session tracks.

        A driver that renders a camera the session has no tracker for would feed the
        session an unseeded view it then tries to re-seed, which is an avoidable error path.
        """

    def seed_points(self, obj):
        """``(N, 3)`` world points on the object's visible surface, as detection produces."""
        raise NotImplementedError

    def ground_truth(self, obj):
        """``(position (3,), quat_xyzw (4,) or None)`` for the object's *origin* right now."""
        raise NotImplementedError

    def advance(self, frame_idx):
        """Move the scene into the state frame ``frame_idx`` should be scored against."""

    def views(self, frame_idx):
        """Optional pre-rendered/edited views for this frame; ``None`` = let the session render."""
        return None

    def is_occluded(self, frame_idx, obj):
        return False

    def gripper_pose(self):
        return None

    def describe(self):
        return {"scene": self.name, "n_frames": int(self.n_frames),
                "objects": list(self.objects)}


class GraspSceneDriver(SceneDriver):
    """The scripted grasp scene from ``tracking_scenario``, with an occlusion window.

    Phase 1 (frames ``0..occlusion_start``): the object translates smoothly.
    Phase 2 (``occlusion_start..occlusion_end``): the object keeps moving while the head
    camera's depth/RGB buffers are painted over, exactly as
    ``test_occluded_camera_is_reseeded_from_the_other`` does - a deterministic stand-in for
    the arm swinging through the line of sight.
    Phase 3: clear line of sight again, which is what ``frames_to_recover`` measures.
    """

    name = "grasp_scripted"

    def __init__(self, sim, env, robot, object_id, obj_name="cube", n_frames=30,
                 start=None, step_vector=(0.010, 0.0, 0.003), occlusion=(12, 19),
                 occluder_cam="head", settle=3):
        self.sim = sim
        self.env = env
        self.robot = robot
        self.object_id = object_id
        self.objects = (obj_name,)
        self.n_frames = int(n_frames)
        from tracking_scenario import OBJECT_START

        self.start = np.asarray(start if start is not None else OBJECT_START, dtype=float)
        self.step_vector = np.asarray(step_vector, dtype=float)
        self.occlusion = (int(occlusion[0]), int(occlusion[1])) if occlusion else None
        self.occluder_cam = occluder_cam
        self.settle = int(settle)
        self.cameras = tuple(config.tracking_cameras)
        self._seed_offset = None

    def select_cameras(self, names):
        self.cameras = tuple(names or config.tracking_cameras)

    # -- scene control ----------------------------------------------------
    def _place(self, position):
        self.sim.set_base_pose(self.object_id, list(position), [0.0, 0.0, 0.0, 1.0])
        for _ in range(self.settle):
            self.sim.step()

    def reset(self):
        self._place(self.start)
        log.info("[eval] scene '%s' reset, object at %s", self.name, list(np.round(self.start, 4)))

    def seed_points(self, obj):
        view = self.robot.capture_camera_view("head", self.env)
        position, _quat = self.ground_truth(obj)
        points, _pixels = surface_seed_points(view, position)
        self._seed_offset = points.mean(axis=0) - position
        return points

    def ground_truth(self, obj):
        position, quat = self.sim.get_base_pose(self.object_id)
        return np.asarray(position, dtype=float), np.asarray(quat, dtype=float)

    def advance(self, frame_idx):
        self._place(self.start + self.step_vector * float(frame_idx))

    def views(self, frame_idx):
        """Always render here, so the measured latency is the *tracking* step only.

        Leaving the session to render would fold the simulator's camera capture (tens of
        milliseconds) into every provider's latency and drown the difference between them.
        """
        views = {}
        for cam in self.cameras:
            try:
                views[cam] = self.robot.capture_camera_view(cam, self.env)
            except Exception as exc:
                log.warning("[eval] camera '%s' capture failed: %s", cam, exc)
        if self._occluding(frame_idx):
            target = views.get(self.occluder_cam)
            if target is not None:
                position, _quat = self.ground_truth(self.objects[0])
                offset = self._seed_offset if self._seed_offset is not None else np.zeros(3)
                paint_occluder(target, position + offset)
        return views or None

    def _occluding(self, frame_idx):
        return bool(self.occlusion) and self.occlusion[0] <= frame_idx < self.occlusion[1]

    def is_occluded(self, frame_idx, obj):
        return self._occluding(frame_idx)

    def gripper_pose(self):
        pos, quat = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return {"position": list(pos), "orientation_q": list(quat)}

    def describe(self):
        info = super().describe()
        info.update({"start": [round(float(v), 4) for v in self.start],
                     "step_vector": [round(float(v), 4) for v in self.step_vector],
                     "occlusion_window": list(self.occlusion) if self.occlusion else None,
                     "occluder_cam": self.occluder_cam})
        return info


# -- simulator-free reference scene ---------------------------------------
def _make_view(name, eye, target, width, height, fov=60.0):
    view_matrix = camera_math.gl_view_matrix(eye, target, (0.0, 0.0, 1.0))
    projection = camera_math.gl_projection_matrix(fov, float(width) / float(height),
                                                  config.near_plane, config.far_plane)
    return CameraView(name=name,
                      rgb=np.zeros((height, width, 3), dtype=np.uint8),
                      depth=np.full((height, width), float(config.far_plane), dtype=np.float32),
                      view_matrix=geometry.mat4(view_matrix),
                      projection_matrix=geometry.mat4(projection),
                      near=float(config.near_plane), far=float(config.far_plane),
                      position=list(eye))


def _pixel_rays(view, xs, ys):
    """Vectorised inverse projection: world ``origin`` and ``direction`` per unit depth.

    A pixel's world point at metric depth ``d`` is ``origin + d * direction``, which is
    what makes an analytic ray/sphere intersection possible without a renderer.
    """
    ndc_x = (2.0 * xs / view.width) - 1.0
    ndc_y = 1.0 - (2.0 * ys / view.height)
    inv = np.linalg.inv(geometry.view_projection(view))

    def at(depth):
        ndc_z = float(camera_math.metric_to_ndc_z(depth, view.near, view.far))
        homog = np.stack([ndc_x, ndc_y, np.full_like(ndc_x, ndc_z), np.ones_like(ndc_x)], axis=-1)
        world = homog @ inv.T
        return world[..., :3] / world[..., 3:4]

    near_pt, far_pt = at(1.0), at(2.0)
    direction = far_pt - near_pt
    return near_pt - direction, direction


def paint_sphere(view, centre, radius, texture=None):
    """Render a matte sphere into ``view``'s depth (and optionally RGB) buffers.

    Analytic rather than rasterised so the scene needs no simulator, yet the depth buffer
    it produces is exactly what the tracker consumes: metric depth along the optical axis
    of a surface that is *physically consistent between cameras*, which is what makes
    cross-camera re-seeding behave as it does in the real scene.
    """
    pixel, z_eye = geometry.project_world_to_pixel(view, centre)
    if pixel is None or z_eye <= 0.0:
        return None
    focal = (view.height / 2.0) / np.tan(np.radians(30.0))
    span = int(np.ceil(radius * focal / max(z_eye, 1e-3))) + 2
    cx, cy = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    x0, x1 = max(0, cx - span), min(view.width, cx + span + 1)
    y0, y1 = max(0, cy - span), min(view.height, cy + span + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    origin, direction = _pixel_rays(view, xs, ys)
    delta = origin - np.asarray(centre, dtype=float)
    a = np.einsum("...i,...i->...", direction, direction)
    b = 2.0 * np.einsum("...i,...i->...", delta, direction)
    c = np.einsum("...i,...i->...", delta, delta) - radius ** 2
    disc = b ** 2 - 4.0 * a * c
    hit = disc >= 0.0
    depth = np.full(disc.shape, np.inf)
    root = np.sqrt(np.where(hit, disc, 0.0))
    depth[hit] = ((-b[hit] - root[hit]) / (2.0 * a[hit]))
    hit &= depth > 0.0

    patch = view.depth[y0:y1, x0:x1]
    closer = hit & (depth < patch)
    patch[closer] = depth[closer].astype(np.float32)
    if texture is not None:
        tile = np.asarray(texture)
        tex = tile[(ys.astype(int) - cy) % tile.shape[0], (xs.astype(int) - cx) % tile.shape[1]]
        rgb_patch = view.rgb[y0:y1, x0:x1]
        rgb_patch[closer] = tex[closer][:, None]
    return pixel


class SyntheticSphereScene(SceneDriver):
    """A textured sphere translating in front of two cameras - no simulator required.

    Exists so the harness itself can be exercised (and providers smoke-compared) on any
    machine, including CI without PyBullet. The object surface is analytic, so ground
    truth is exact and the surface-vs-origin seed offset is a genuine ``radius``-sized
    bias - precisely the thing the seed-offset correction has to remove.
    """

    name = "synthetic_sphere"

    def __init__(self, n_frames=24, resolution=256, radius=0.08, start=(0.0, 0.40, 0.17),
                 step_vector=(0.008, 0.0, 0.003), occlusion=(9, 15), occluder_cam="head",
                 cameras=None, seed=0):
        self.objects = ("ball",)
        self.n_frames = int(n_frames)
        self.resolution = int(resolution)
        self.radius = float(radius)
        self.start = np.asarray(start, dtype=float)
        self.step_vector = np.asarray(step_vector, dtype=float)
        self.occlusion = (int(occlusion[0]), int(occlusion[1])) if occlusion else None
        self.occluder_cam = occluder_cam
        self.cam_eyes = dict(cameras or {"head": (0.45, 0.95, 0.55), "wrist": (-0.42, 0.92, 0.48)})
        rng = np.random.default_rng(seed)
        # High-contrast, object-anchored texture: NCC template matching needs something to
        # lock onto, and anchoring it to the sphere centre makes it translate with the body.
        self.texture = rng.integers(20, 236, size=(64, 64), dtype=np.uint8)
        self.robot = None
        self.env = None
        self._position = self.start.copy()
        self._seed_offset = np.zeros(3)

    def reset(self):
        self._position = self.start.copy()
        log.info("[eval] synthetic scene reset | centre=%s radius=%.3f",
                 list(np.round(self._position, 4)), self.radius)

    def select_cameras(self, names):
        wanted = [c for c in (names or self.cam_eyes) if c in self.cam_eyes]
        if wanted:
            self.cam_eyes = {c: self.cam_eyes[c] for c in wanted}

    def advance(self, frame_idx):
        self._position = self.start + self.step_vector * float(frame_idx)

    def ground_truth(self, obj):
        return self._position.copy(), np.array([0.0, 0.0, 0.0, 1.0])

    def render(self, frame_idx=None):
        views = {}
        for cam, eye in self.cam_eyes.items():
            view = _make_view(cam, eye, tuple(self._position), self.resolution, self.resolution)
            paint_sphere(view, self._position, self.radius, self.texture)
            views[cam] = view
        if frame_idx is not None and self.is_occluded(frame_idx, self.objects[0]):
            target = views.get(self.occluder_cam)
            if target is not None:
                paint_occluder(target, self._position + self._seed_offset,
                               radius=int(self.resolution / 6))
        return views

    def seed_points(self, obj):
        view = self.render()["head"]
        points, _pixels = surface_seed_points(view, self._position, spread_px=6)
        self._seed_offset = points.mean(axis=0) - self._position
        return points

    def views(self, frame_idx):
        return self.render(frame_idx)

    def is_occluded(self, frame_idx, obj):
        return bool(self.occlusion) and self.occlusion[0] <= frame_idx < self.occlusion[1]

    def describe(self):
        info = super().describe()
        info.update({"radius": self.radius, "resolution": self.resolution,
                     "step_vector": [round(float(v), 4) for v in self.step_vector],
                     "occlusion_window": list(self.occlusion) if self.occlusion else None,
                     "occluder_cam": self.occluder_cam})
        return info


# -- runner ----------------------------------------------------------------
class TrackingEvalRunner:
    """Runs one (2D provider, 3D provider, cameras) configuration over one scene."""

    def __init__(self, scene, tracker_provider=None, tracker3d=None, cameras=None,
                 tracker_kwargs=None, tracker3d_kwargs=None, monitor=None,
                 lost_penalty_m=LOST_PENALTY_M, recovery_threshold_m=RECOVERY_THRESHOLD_M,
                 keep_frames=True, label=None):
        self.scene = scene
        self.tracker_provider = tracker_provider or config.tracker_provider_default
        self.tracker3d = tracker3d or getattr(config, "tracker3d_provider_default", "depth_fusion")
        self.cameras = tuple(cameras or config.tracking_cameras)
        self.tracker_kwargs = dict(tracker_kwargs or {})
        self.tracker3d_kwargs = dict(tracker3d_kwargs or {})
        self.monitor = monitor
        self.lost_penalty_m = float(lost_penalty_m)
        self.recovery_threshold_m = float(recovery_threshold_m)
        self.keep_frames = bool(keep_frames)
        self.label = label or f"{self.tracker_provider}+{self.tracker3d}"
        self.samples: Dict[str, List[FrameSample]] = {}
        self.offsets: Dict[str, SeedOffset] = {}
        self.session = None

    # -- run --------------------------------------------------------------
    def run(self):
        """Execute the scene and return the JSON-serialisable result payload."""
        started = time.time()
        reset_vram_peak()
        log.info("[eval] run '%s' | scene=%s cameras=%s frames=%d", self.label,
                 self.scene.name, list(self.cameras), self.scene.n_frames)
        failure = None
        try:
            self._seed()
            self._loop()
        except Exception as exc:                    # a broken run still produces a payload
            failure = f"{type(exc).__name__}: {exc}"
            log.error("[eval] run '%s' aborted: %s", self.label, failure)
        payload = self.report(failure=failure, wall_s=time.time() - started)
        summary = payload.get("aggregate", {})
        log.info("[eval] run '%s' done | median=%s p95=%s pct_lost=%s latency_ms=%s",
                 self.label, summary.get("median_l2_m"), summary.get("p95_l2_m"),
                 summary.get("pct_lost"), summary.get("latency_ms_mean"))
        return payload

    def _seed(self):
        self.scene.select_cameras(self.cameras)
        self.scene.reset()
        self.session = TrackingSession(robot=self.scene.robot, env=self.scene.env,
                                       cameras=self.cameras, provider=self.tracker_provider,
                                       tracker3d=self.tracker3d,
                                       tracker_kwargs=self.tracker_kwargs,
                                       tracker3d_kwargs=self.tracker3d_kwargs,
                                       monitor=self.monitor, write_jsonl=False)
        for obj in self.scene.objects:
            points = np.asarray(self.scene.seed_points(obj), dtype=float).reshape(-1, 3)
            position, quat = self.scene.ground_truth(obj)
            offset = capture_seed_offset(points, position, quat)
            self.offsets[obj] = offset
            self.samples[obj] = []
            self.session.add_target(obj, points)
            log.info("[eval] seeded '%s' with %d point(s) | origin=%s seed_offset_local=%s "
                     "(|offset|=%.4f m, rotating=%s)", obj, points.shape[0],
                     list(np.round(position, 4)), list(np.round(offset.local, 4)),
                     float(np.linalg.norm(offset.local)), offset.rotating)

    def _loop(self):
        for frame in range(self.scene.n_frames):
            self.scene.advance(frame)
            truth = {obj: self.scene.ground_truth(obj) for obj in self.scene.objects}
            views = self.scene.views(frame)
            gripper = self.scene.gripper_pose()

            start = time.perf_counter()
            report = self.session.on_frame(views=views, gripper_pose=gripper,
                                           trajectory_step=frame)
            latency_ms = (time.perf_counter() - start) * 1000.0

            for obj in self.scene.objects:
                position, quat = truth[obj]
                self.samples[obj].append(
                    self._sample(frame, obj, report, position, quat, latency_ms))

            if self.session.aborted:
                log.warning("[eval] session aborted at frame %d: %s", frame,
                            self.session.abort_reason)
                break

    def _sample(self, frame, obj, report, position, quat, latency_ms):
        expected = self.offsets[obj].expected(position, quat)
        sample = FrameSample(frame=frame, obj=obj, expected=expected, latency_ms=latency_ms,
                             occluded=bool(self.scene.is_occluded(frame, obj)))
        if report is None:
            sample.dropped = True
            sample.lost = True
            return sample
        state = report.objects.get(obj)
        if state is None:
            sample.lost = True
            return sample
        sample.lost = bool(state.lost) or state.world_point is None
        sample.predicted = (None if state.world_point is None
                            else np.asarray(state.world_point, dtype=float))
        sample.disagreement = state.disagreement
        sample.lift_meta = dict(state.lift_meta or {})
        return sample

    # -- reporting --------------------------------------------------------
    def config_payload(self):
        return {
            "label": self.label,
            "tracker_provider": self.tracker_provider,
            "tracker3d": self.tracker3d,
            "cameras": list(self.cameras),
            "image_size": [int(config.image_width), int(config.image_height)],
            "track_interval": int(config.track_interval),
            "lost_penalty_m": self.lost_penalty_m,
            "recovery_threshold_m": self.recovery_threshold_m,
            "seed_offset_frame": "local",
            **self.scene.describe(),
        }

    def report(self, failure=None, wall_s=None):
        per_object = {obj: compute_metrics(samples, self.lost_penalty_m,
                                           self.recovery_threshold_m)
                      for obj, samples in self.samples.items()}
        payload = {
            "schema": SCHEMA,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_s": None if wall_s is None else round(float(wall_s), 3),
            "config": self.config_payload(),
            "seed_offsets": {obj: off.to_dict() for obj, off in self.offsets.items()},
            "objects": per_object,
            "aggregate": aggregate_metrics(per_object),
            "peak_vram_mb": peak_vram_mb(),
            "session": {
                "errors": getattr(self.session, "errors", None),
                "lift_errors": getattr(self.session, "lift_errors", None),
                "aborted": bool(getattr(self.session, "aborted", False)),
                "abort_reason": getattr(self.session, "abort_reason", None),
            },
        }
        if failure:
            payload["error"] = failure
        if self.keep_frames:
            payload["frames"] = {obj: [s.to_dict() for s in samples]
                                 for obj, samples in self.samples.items()}
        return payload


def run_eval(scene, **kwargs):
    """Convenience wrapper: build a :class:`TrackingEvalRunner` and run it."""
    return TrackingEvalRunner(scene, **kwargs).run()


# -- output helpers --------------------------------------------------------
def write_report(payload, path):
    """Write ``payload`` as JSON (creating parent directories). Returns the path."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    log.info("[eval] wrote %s", path)
    return path


_TABLE_COLUMNS = (
    ("config", "config"),
    ("median_l2_m", "median_m"),
    ("p95_l2_m", "p95_m"),
    ("max_l2_m", "max_m"),
    ("pct_lost", "%lost"),
    ("err_during_occlusion_m", "occl_m"),
    ("frames_to_recover", "recov_f"),
    ("max_jump_m", "jump_m"),
    ("latency_ms_mean", "lat_ms"),
    ("latency_ms_p95", "lat_p95"),
    ("peak_vram_mb", "vram_mb"),
)


def _cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)


def render_comparison_table(payloads, obj=None):
    """Fixed-width table comparing several run payloads.

    ``obj`` selects one object's metrics; by default the scene-level aggregate is used.
    """
    rows = []
    for payload in payloads or []:
        metrics = (payload.get("objects", {}).get(obj) if obj
                   else payload.get("aggregate", {})) or {}
        label = payload.get("config", {}).get("label", "?")
        row = {"config": label, "peak_vram_mb": payload.get("peak_vram_mb")}
        row.update({key: metrics.get(key) for key, _ in _TABLE_COLUMNS if key != "config"})
        rows.append(row)
    if not rows:
        return "(no results)"

    widths = {}
    for key, header in _TABLE_COLUMNS:
        widths[key] = max(len(header), *(len(_cell(row.get(key))) for row in rows))
    header = " | ".join(h.ljust(widths[k]) for k, h in _TABLE_COLUMNS)
    rule = "-+-".join("-" * widths[k] for k, _ in _TABLE_COLUMNS)
    lines = [header, rule]
    for row in rows:
        lines.append(" | ".join(_cell(row.get(k)).ljust(widths[k]) for k, _ in _TABLE_COLUMNS))
    return "\n".join(lines)
