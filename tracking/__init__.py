"""Rollout tracking: per-frame 2D tracks fused into 3D world coordinates.

Public entry points:
  * :class:`tracking.session.TrackingSession` - drives everything from the simulator's
    per-keyframe hook (``Robot.step_env_and_record``).
  * :mod:`tracking.monitors` - ready-made invariants (``attached_to_gripper`` etc.).
  * :mod:`tracking.types` - the dataclasses a monitor receives.
"""

from tracking.types import (  # noqa: F401
    STATUS_ABORT,
    STATUS_OK,
    STATUS_RECORD,
    STATUS_WARN,
    CameraView,
    CamTrack,
    GripperState,
    MonitorResult,
    ReseedEvent,
    TrackedObjectState,
    TrackFrameReport,
)
