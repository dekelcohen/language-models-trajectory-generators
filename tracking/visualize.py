"""Render annotated tracking frames so a human can *see* what the tracker is doing.

The eval harness (``tests/tracking_eval.py``) answers "how many millimetres wrong"; this
module answers "wrong how, and where". Per camera it paints, on top of the RGB the tracker
actually saw:

* every tracked 2D point - green when the provider calls it visible, orange when not;
* the fused 3D estimate reprojected into that camera (cyan cross);
* the ground-truth point reprojected into that camera (red circle), with a magenta segment
  joining the two, so the 3D error is legible as a pixel gap;
* a header line per camera (status, confidence, health, points visible) and a frame banner
  (3D error in mm, lost/occluded flags, the 3D provider's ``lift_meta``).

Drawing never raises: a visualisation bug must not break an eval run, so every entry point
is defensive and returns the best image it can.
"""

import logging
import os

import numpy as np

from tracking import geometry

log = logging.getLogger("tracking.visualize")

try:                                             # cv2 is already a hard dep of the sim path
    import cv2
except Exception:                                # pragma: no cover - defensive
    cv2 = None

# BGR, because that is what cv2 draws in.
COLOR_VISIBLE = (0, 220, 0)
COLOR_HIDDEN = (0, 165, 255)
COLOR_PREDICTED = (255, 255, 0)
COLOR_TRUTH = (0, 0, 255)
COLOR_ERROR = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BAD = (0, 0, 255)
COLOR_WARN = (0, 200, 255)

_FONT = 0 if cv2 is None else cv2.FONT_HERSHEY_SIMPLEX


def available():
    """True when OpenCV is importable, i.e. overlays/videos can be produced."""
    return cv2 is not None


def _to_bgr(rgb, scale):
    img = np.asarray(rgb)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    img = np.ascontiguousarray(img[:, :, :3].astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if scale and scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return img


def _text(img, lines, origin=(4, 4), color=COLOR_TEXT, scale=0.38, line_h=13):
    x, y = origin
    for i, line in enumerate(lines):
        pos = (x, y + line_h * (i + 1))
        cv2.putText(img, str(line), pos, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(line), pos, _FONT, scale, color, 1, cv2.LINE_AA)


def _cross(img, xy, color, size=6, thickness=1):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _project(view, world):
    if world is None:
        return None
    try:
        pixel, _z = geometry.project_world_to_pixel(view, np.asarray(world, dtype=float))
    except Exception:
        return None
    if pixel is None or not np.all(np.isfinite(pixel)):
        return None
    return pixel


def draw_camera_overlay(view, cam_track=None, predicted_world=None, expected_world=None,
                        scale=2, header_lines=(), occluded=False):
    """Annotate one camera's RGB frame. Returns a BGR uint8 image (never raises)."""
    if cv2 is None:
        return None
    try:
        img = _to_bgr(view.rgb, scale)
    except Exception as exc:                     # pragma: no cover - defensive
        log.warning("[viz] could not convert '%s' rgb: %s", getattr(view, "name", "?"), exc)
        return None

    try:
        if cam_track is not None and cam_track.points_2d is not None:
            points = np.asarray(cam_track.points_2d, dtype=float).reshape(-1, 2)
            visible = (np.ones(len(points), dtype=bool) if cam_track.visible is None
                       else np.asarray(cam_track.visible).reshape(-1).astype(bool))
            for idx, point in enumerate(points):
                if not np.all(np.isfinite(point)):
                    continue
                centre = (int(round(point[0] * scale)), int(round(point[1] * scale)))
                is_vis = bool(visible[idx]) if idx < len(visible) else True
                cv2.circle(img, centre, 3, COLOR_VISIBLE if is_vis else COLOR_HIDDEN,
                           -1 if is_vis else 1, cv2.LINE_AA)

        pred_px = _project(view, predicted_world)
        true_px = _project(view, expected_world)
        if pred_px is not None and true_px is not None:
            cv2.line(img, (int(pred_px[0] * scale), int(pred_px[1] * scale)),
                     (int(true_px[0] * scale), int(true_px[1] * scale)),
                     COLOR_ERROR, 1, cv2.LINE_AA)
        if true_px is not None:
            cv2.circle(img, (int(round(true_px[0] * scale)), int(round(true_px[1] * scale))),
                       7, COLOR_TRUTH, 1, cv2.LINE_AA)
        if pred_px is not None:
            _cross(img, pred_px * scale, COLOR_PREDICTED, size=7, thickness=2)

        lines = list(header_lines)
        if cam_track is not None:
            lines.append(f"{cam_track.cam}: {cam_track.status} conf={cam_track.confidence:.2f} "
                         f"health={cam_track.health:.2f} vis={cam_track.n_visible}")
            if cam_track.reseeded_reason:
                lines.append(f"reseed: {cam_track.reseeded_reason}")
        elif getattr(view, "name", None):
            lines.append(f"{view.name}: no track")
        _text(img, lines)

        if occluded:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), COLOR_WARN, 2)
            _text(img, ["OCCLUDED"], origin=(4, img.shape[0] - 24), color=COLOR_WARN, scale=0.5)
    except Exception as exc:                     # pragma: no cover - defensive
        log.warning("[viz] overlay failed on '%s': %s", getattr(view, "name", "?"), exc)
    return img


def _banner(width, lines, height=None, color=COLOR_TEXT):
    height = height or (16 * len(lines) + 8)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    _text(strip, lines, origin=(6, 0), color=color, scale=0.45, line_h=16)
    return strip


def compose_frame(views, state=None, expected_world=None, cameras=None, frame_idx=0,
                  scale=2, occluded=False, extra_lines=()):
    """Stack every camera's annotated frame side by side under a summary banner.

    ``state`` is a :class:`tracking.types.TrackedObjectState` (or ``None`` when the session
    produced no report for this frame). Returns a BGR image, or ``None`` if nothing could
    be drawn.
    """
    if cv2 is None or not views:
        return None
    names = list(cameras or views.keys())
    predicted = None if state is None else state.world_point
    tiles = []
    for name in names:
        view = views.get(name)
        if view is None:
            continue
        cam_track = None if state is None else state.cams.get(name)
        tile = draw_camera_overlay(view, cam_track=cam_track, predicted_world=predicted,
                                   expected_world=expected_world, scale=scale,
                                   occluded=occluded and _is_painted(cam_track, name, state))
        if tile is not None:
            tiles.append(tile)
    if not tiles:
        return None
    height = max(tile.shape[0] for tile in tiles)
    tiles = [tile if tile.shape[0] == height else
             cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for tile in tiles]
    grid = np.hstack(tiles)

    error_mm = None
    if predicted is not None and expected_world is not None:
        error_mm = float(np.linalg.norm(np.asarray(predicted, dtype=float)
                                        - np.asarray(expected_world, dtype=float))) * 1000.0
    status = "LOST" if (state is None or state.lost) else "tracking"
    head = [f"frame {frame_idx:03d}  {status}"
            + (f"  err={error_mm:6.1f} mm" if error_mm is not None else "  err=n/a")
            + (f"  disagree={state.disagreement * 1000:.1f} mm"
               if state is not None and state.disagreement is not None else "")
            + ("  OCCLUSION" if occluded else "")]
    if state is not None and state.lift_meta:
        head.append("lift: " + ", ".join(f"{k}={_fmt(v)}" for k, v in
                                         list(state.lift_meta.items())[:6]))
    head.extend(extra_lines)
    colour = COLOR_BAD if (state is None or state.lost) else COLOR_TEXT
    return np.vstack([_banner(grid.shape[1], head, color=colour), grid])


def _is_painted(cam_track, name, state):
    """Only flag the camera the scene actually occluded, when that is knowable."""
    del state
    return name == "head" if cam_track is None else cam_track.status in ("occluded", "lost",
                                                                         "rejected", "low_confidence")


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt(v) for v in value[:4]) + "]"
    return str(value)


def write_video(frames, path, fps=6):
    """Encode BGR frames to ``path`` (mp4). Returns the path, or ``None`` on failure."""
    if cv2 is None or not frames:
        log.warning("[viz] nothing to encode for %s", path)
        return None
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        log.error("[viz] could not open video writer for %s", path)
        return None
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    log.info("[viz] wrote %s (%d frames, %dx%d)", path, len(frames), width, height)
    return path


def write_frames(frames, folder, prefix="frame"):
    """Dump each BGR frame as a PNG (useful when a video codec is unavailable)."""
    if cv2 is None or not frames:
        return []
    os.makedirs(folder, exist_ok=True)
    paths = []
    for idx, frame in enumerate(frames):
        path = os.path.join(folder, f"{prefix}_{idx:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths
