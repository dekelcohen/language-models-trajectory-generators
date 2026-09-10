"""Per-camera track health and the cross-camera re-seed policy.

The head camera sees the object from the start; the wrist camera only later. During a
reach the robot arm regularly occludes the head camera exactly when the wrist camera has
the cleanest view - and vice versa when the wrist camera dives past the object. So
re-seeding is *continuous*, not a one-off bootstrap: every frame each camera is scored,
and an unhealthy camera is re-initialised from the healthy camera's world estimate.

Re-seed triggers (in priority order):
  1. ``unseeded``      - camera never had a track (wrist at the start)
  2. ``lost``          - all points dropped or left the frame
  3. ``low_confidence``- confidence below threshold for N frames while another camera is healthy
  4. ``disagreement``  - per-camera world points differ by more than the tolerance;
                         the lower-scoring camera is re-seeded from the higher-scoring one

Guards: a re-seed is only applied when the donor's world point is genuinely imaged by the
recipient camera (``geometry.is_visible``, which includes the depth/occlusion test), and
never more often than ``config.track_reseed_cooldown`` tracked frames per camera. Without
those two guards the policy would happily re-seed the head camera onto the robot arm that
is occluding the object, and then ping-pong between cameras every frame.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import config
from tracking.types import CamTrack


@dataclass
class CamHealthState:
    """Rolling, per-(object, camera) bookkeeping used by the policy."""

    low_conf_frames: int = 0
    lost_frames: int = 0
    last_reseed_frame: int = -10 ** 9
    last_world_point: Optional[np.ndarray] = None
    history: list = field(default_factory=list)

    def can_reseed(self, frame_idx, cooldown=None):
        cooldown = config.track_reseed_cooldown if cooldown is None else cooldown
        return (frame_idx - self.last_reseed_frame) >= cooldown


def temporal_continuity(previous, current, max_jump=None):
    """1.0 when the world point barely moved, decaying to 0 at ``max_jump`` metres.

    A track that slips onto the background teleports; this term is what makes the
    disagreement tie-break pick the camera that stayed consistent with itself.
    """
    max_jump = config.track_max_jump_m if max_jump is None else float(max_jump)
    if previous is None or current is None:
        return 1.0
    jump = float(np.linalg.norm(np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)))
    if jump <= 0.0:
        return 1.0
    return float(max(0.0, 1.0 - jump / max(max_jump, 1e-6)))


def score_camera(track: CamTrack, state: CamHealthState, z_eye=None):
    """Health in [0, 1]: confidence x depth validity x 1/z x temporal continuity."""
    if not track.seeded or track.world_point is None or not track.depth_valid:
        return 0.0
    if track.status in ("jumped", "rejected"):
        # Already known to be tracking the wrong thing; it must not be fused or donate.
        return 0.0
    conf = float(np.clip(track.confidence, 0.0, 1.0))
    if conf <= 0.0:
        return 0.0
    continuity = temporal_continuity(state.last_world_point, track.world_point)
    # Distance term normalised so ~0.3 m (a typical wrist stand-off) scores ~1.0.
    z = max(float(z_eye) if z_eye else 0.3, 1e-3)
    distance_term = float(np.clip(0.3 / z, 0.2, 1.0))
    support = min(track.n_visible, 4) / 4.0
    return float(conf * continuity * distance_term * (0.4 + 0.6 * support))


def decide_reseeds(obj_name, tracks: Dict[str, CamTrack], states: Dict[str, CamHealthState],
                   frame_idx, fused_world_point=None):
    """Decide which cameras to re-seed this frame.

    Returns ``[(cam, reason, donor_cam, world_point)]``. ``donor_cam`` is ``None`` when the
    seed comes from the fused estimate rather than one specific camera.
    """
    healthy = {c: t for c, t in tracks.items()
               if t.seeded and t.health >= config.track_health_min and t.world_point is not None}
    decisions = []

    for cam, track in tracks.items():
        state = states[cam]

        # Book-keeping first, so counters keep advancing even during a cooldown.
        if track.seeded and track.confidence < config.track_reseed_conf:
            state.low_conf_frames += 1
        elif track.seeded:
            state.low_conf_frames = 0
        if track.seeded and track.world_point is None:
            state.lost_frames += 1
        elif track.seeded:
            state.lost_frames = 0

        donor_cam, donor_point = _pick_donor(cam, healthy, fused_world_point)
        if donor_point is None:
            continue

        reason = None
        if not track.seeded:
            reason = "unseeded"
        elif track.status == "jumped":
            reason = "jumped"
        elif state.lost_frames >= 1:
            reason = "lost"
        elif state.low_conf_frames >= config.track_reseed_patience:
            reason = "low_confidence"

        if reason is None:
            continue
        if not state.can_reseed(frame_idx):
            continue
        decisions.append((cam, reason, donor_cam, np.asarray(donor_point, dtype=float)))

    if not decisions:
        decisions.extend(_disagreement_reseeds(tracks, states, frame_idx))
    return decisions


def _pick_donor(cam, healthy, fused_world_point):
    """Best world point to re-seed ``cam`` from: the healthiest *other* camera."""
    others = {c: t for c, t in healthy.items() if c != cam}
    if others:
        donor = max(others.items(), key=lambda kv: kv[1].health)
        return donor[0], donor[1].world_point
    if fused_world_point is not None:
        return None, fused_world_point
    return None, None


def _disagreement_reseeds(tracks, states, frame_idx):
    """Re-seed the loser when two seeded cameras localise the object far apart.

    Only fires when there is a clear winner: if both cameras are equally healthy the
    frame is fused and flagged instead (the caller records ``disagreement``), because
    re-seeding an equally-good camera just moves the error around.
    """
    seeded = [(c, t) for c, t in tracks.items() if t.seeded and t.world_point is not None]
    if len(seeded) < 2:
        return []

    out = []
    for i in range(len(seeded)):
        for j in range(i + 1, len(seeded)):
            (cam_a, ta), (cam_b, tb) = seeded[i], seeded[j]
            gap = float(np.linalg.norm(np.asarray(ta.world_point, dtype=float)
                                       - np.asarray(tb.world_point, dtype=float)))
            if gap <= config.track_disagree_m:
                continue
            better, worse = (ta, tb) if ta.health >= tb.health else (tb, ta)
            better_cam, worse_cam = (cam_a, cam_b) if ta.health >= tb.health else (cam_b, cam_a)
            # Require a clear winner (>25% healthier) to avoid ping-pong.
            if better.health < 1.25 * max(worse.health, 1e-6):
                continue
            if not states[worse_cam].can_reseed(frame_idx):
                continue
            out.append((worse_cam, f"disagreement({gap:.3f}m)", better_cam,
                        np.asarray(better.world_point, dtype=float)))
    return out
