"""Data types shared by the rollout tracking subsystem.

Everything here is plain data (dataclasses + numpy), free of simulator or agent
imports, so the geometry/fusion/monitor logic can be unit-tested without booting a
simulator.

Conventions (identical to the rest of the repo, see ``sim_adapter/base.py``):
  * world coordinates are metres, Z-up
  * quaternions are xyzw
  * 2D points are ``(x=column, y=row)`` in pixels, image origin top-left
  * depth handed to this module is always **metric metres along the optical axis**
    (callers run ``sim_adapter.camera_math.depth_to_metric`` first)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sim_adapter.transforms import quat_from_matrix, rotation_angle

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_RECORD = "record"
STATUS_ABORT = "abort"

#: Ordered by increasing severity - ``combine`` and the session take the worst.
STATUS_SEVERITY = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_RECORD: 2, STATUS_ABORT: 3}

GRIPPER_NAME = "gripper"


def _as_list(value):
    """Return a JSON-serialisable copy of ``value`` (numpy -> python)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return [round(float(v), 6) for v in arr.reshape(-1)]


@dataclass
class CameraView:
    """One camera's rendered frame plus everything needed to (un)project points."""

    name: str
    rgb: np.ndarray                     # (H, W, 3) uint8
    depth: np.ndarray                   # (H, W) float32, metric metres
    view_matrix: np.ndarray             # 4x4
    projection_matrix: np.ndarray       # 4x4
    near: float
    far: float
    position: Optional[Sequence[float]] = None
    orientation_q: Optional[Sequence[float]] = None
    #: Optional (H, W) int instance mask from the simulator (PyBullet: object uid in the
    #: low 24 bits, link index + 1 above). Only used to seed; never needed to track.
    segmentation: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    @property
    def image_size(self):
        return (self.width, self.height)


@dataclass
class CamTrack:
    """Per-object, per-camera track state for a single frame."""

    cam: str
    seeded: bool = False
    points_2d: Optional[np.ndarray] = None      # (N, 2) float
    visible: Optional[np.ndarray] = None        # (N,) bool
    confidence: float = 0.0
    world_point: Optional[np.ndarray] = None    # (3,) fused over this camera's points
    depth_valid: bool = False
    health: float = 0.0
    status: str = "unseeded"                    # unseeded|ok|low_confidence|occluded|lost|jumped|rejected
    reseeded_reason: Optional[str] = None
    #: Frames since the 2D tracker last computed these points (windowed trackers such as
    #: online CoTracker only refresh every ``step`` frames; 0 = fresh this frame).
    stale_frames: int = 0

    @property
    def n_visible(self) -> int:
        if self.visible is None:
            return 0
        return int(np.count_nonzero(self.visible))

    def to_dict(self):
        return {
            "cam": self.cam,
            "seeded": self.seeded,
            "points_2d": None if self.points_2d is None
            else [[round(float(x), 2), round(float(y), 2)] for x, y in np.asarray(self.points_2d)],
            "visible": None if self.visible is None else [bool(v) for v in np.asarray(self.visible)],
            "confidence": round(float(self.confidence), 4),
            "world_point": _as_list(self.world_point),
            "depth_valid": bool(self.depth_valid),
            "health": round(float(self.health), 4),
            "status": self.status,
            "reseeded_reason": self.reseeded_reason,
        }


@dataclass
class ReseedEvent:
    cam: str
    obj: str
    reason: str
    donor: Optional[str] = None
    applied: bool = False
    detail: str = ""

    def to_dict(self):
        return {
            "cam": self.cam, "obj": self.obj, "reason": self.reason,
            "donor": self.donor, "applied": self.applied, "detail": self.detail,
        }


@dataclass
class ObjectPose:
    """6-DoF pose of a tracked object **relative to its seed-time configuration**.

    ``R, t`` map the seed-time template points to where they are now:
    ``current_i = R @ template_i + t``. There is no object model, so this is a *relative*
    pose - "rotated 40 deg about this axis since tracking started" - not an absolute one.

    ``source`` says where the pose came from, and matters to anything that reasons about it:
    ``measured`` (fused cameras), ``measured_single_cam``, ``predicted`` (Kalman coast) or
    ``fk`` (following the gripper while grasped and occluded). Only ``measured*`` poses are
    evidence about the world; monitors must not treat a prediction as an observation.
    """

    R: np.ndarray
    t: np.ndarray
    source: str = "measured"
    mode: str = "green"                   # green | yellow | red | black
    rmse_m: Optional[float] = None
    n_points: int = 0
    velocity: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None

    @property
    def measured(self) -> bool:
        return self.source.startswith("measured")

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(rotation_angle(self.R)))

    def apply(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ np.asarray(self.R).T + np.asarray(self.t)

    def to_dict(self):
        return {
            "quat_xyzw": [round(float(v), 5) for v in quat_from_matrix(self.R)],
            "t": _as_list(self.t),
            "rotation_deg": round(self.rotation_deg, 3),
            "source": self.source,
            "mode": self.mode,
            "rmse_m": None if self.rmse_m is None else round(float(self.rmse_m), 5),
            "n_points": int(self.n_points),
            "velocity": _as_list(self.velocity),
            "angular_velocity": _as_list(self.angular_velocity),
        }


@dataclass
class TrackedObjectState:
    """Fused, per-frame state of one tracked target."""

    name: str
    world_point: Optional[np.ndarray] = None
    cams: Dict[str, CamTrack] = field(default_factory=dict)
    confidence: float = 0.0
    lost: bool = True
    disagreement: Optional[float] = None     # metres between per-camera world points
    reseeds: List[ReseedEvent] = field(default_factory=list)
    lift_meta: dict = field(default_factory=dict)   # which 3D provider produced world_point
    pose: Optional[ObjectPose] = None               # only providers that fit a pose set this
    #: ``world_point`` is a prediction (Kalman coast / gripper FK), not a measurement.
    predicted: bool = False

    @property
    def visible_cams(self) -> List[str]:
        return [c for c, t in self.cams.items() if t.seeded and t.world_point is not None]

    def to_dict(self):
        return {
            "name": self.name,
            "world_point": _as_list(self.world_point),
            "confidence": round(float(self.confidence), 4),
            "lost": bool(self.lost),
            "predicted": bool(self.predicted),
            "disagreement": None if self.disagreement is None else round(float(self.disagreement), 5),
            "cams": {c: t.to_dict() for c, t in self.cams.items()},
            "reseeds": [r.to_dict() for r in self.reseeds],
            **({"pose": self.pose.to_dict()} if self.pose is not None else {}),
            **({"lift": self.lift_meta} if self.lift_meta else {}),
        }


@dataclass
class GripperState:
    """Gripper pose from forward kinematics (truth) and, optionally, from vision."""

    world_pos: Optional[np.ndarray] = None          # FK - authoritative
    world_quat: Optional[np.ndarray] = None
    opening: Optional[float] = None
    visual_world_pos: Optional[np.ndarray] = None   # visual cross-check
    visual_confidence: float = 0.0

    @property
    def discrepancy(self) -> Optional[float]:
        if self.world_pos is None or self.visual_world_pos is None:
            return None
        return float(np.linalg.norm(np.asarray(self.world_pos, dtype=float)
                                    - np.asarray(self.visual_world_pos, dtype=float)))

    def to_dict(self):
        return {
            "world_pos": _as_list(self.world_pos),
            "world_quat": _as_list(self.world_quat),
            "opening": None if self.opening is None else round(float(self.opening), 5),
            "visual_world_pos": _as_list(self.visual_world_pos),
            "visual_confidence": round(float(self.visual_confidence), 4),
            "discrepancy": None if self.discrepancy is None else round(self.discrepancy, 5),
        }


@dataclass
class MonitorResult:
    status: str = STATUS_OK
    reason: str = ""
    data: dict = field(default_factory=dict)

    @property
    def severity(self) -> int:
        return STATUS_SEVERITY.get(self.status, 0)

    def to_dict(self):
        return {"status": self.status, "reason": self.reason, "data": self.data}


@dataclass
class TrackFrameReport:
    """One frame of tracking output.

    This object is what the user/LLM monitor receives as ``state``; keep the public
    attribute names stable - the prompt documents them.
    """

    frame_idx: int = 0
    trajectory_step: int = 0
    objects: Dict[str, TrackedObjectState] = field(default_factory=dict)
    gripper: GripperState = field(default_factory=GripperState)
    monitor: MonitorResult = field(default_factory=MonitorResult)
    rgb_paths: Dict[str, str] = field(default_factory=dict)
    # Real-time pipeline (tracking/realtime.py); None on the legacy keyframe cadence.
    sim_time: Optional[float] = None       # sim time the frame was exposed
    camera_frame: Optional[int] = None     # camera-clock index of that frame
    available_at: Optional[float] = None   # sim time the result became usable
    latency_s: Optional[float] = None      # available_at - sim_time (queue wait + compute)
    timing: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Optional[TrackedObjectState]:
        return self.objects.get(name)

    def world_point(self, name: str) -> Optional[np.ndarray]:
        """World position of ``name``; ``"gripper"`` resolves to the FK pose."""
        if name == GRIPPER_NAME:
            return None if self.gripper.world_pos is None else np.asarray(self.gripper.world_pos, dtype=float)
        obj = self.objects.get(name)
        if obj is None or obj.world_point is None:
            return None
        return np.asarray(obj.world_point, dtype=float)

    def distance(self, name_a: str, name_b: str = GRIPPER_NAME) -> Optional[float]:
        """Euclidean distance in metres, or ``None`` if either point is unavailable."""
        pa, pb = self.world_point(name_a), self.world_point(name_b)
        if pa is None or pb is None:
            return None
        return float(np.linalg.norm(pa - pb))

    def is_lost(self, name: str) -> bool:
        obj = self.objects.get(name)
        return True if obj is None else bool(obj.lost)

    def to_dict(self):
        out = {
            "frame_idx": self.frame_idx,
            "trajectory_step": self.trajectory_step,
            "objects": {n: o.to_dict() for n, o in self.objects.items()},
            "gripper": self.gripper.to_dict(),
            "monitor": self.monitor.to_dict(),
            "rgb_paths": dict(self.rgb_paths),
        }
        if self.sim_time is not None:
            out.update({"sim_time": round(self.sim_time, 5), "camera_frame": self.camera_frame,
                        "available_at": round(self.available_at, 5),
                        "latency_s": round(self.latency_s, 5), "timing": dict(self.timing)})
        return out
