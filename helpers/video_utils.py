"""Trajectory video helpers: per-attempt review clips + incrementally grown full video.

Why: the VLM reviewer only needs the frames of the CURRENT attempt, while a human
wants one full-session video to scrub through mid-run. Re-encoding every frame from
step 0 on each review is O(n^2) over a long run (the door bug log re-encoded 589
frames twice per attempt). Instead:

    1. encode ONLY the new attempt's frames  -> <video_folder>/<base>_attempt_<start>_<end>.mp4  (review input)
    2. append that clip to the running full video with `ffmpeg -c copy` (stream copy,
       no re-encode) -> <video_folder>/<base>_full.mp4

Fallback: if ffmpeg is unavailable, the full video is re-encoded from frame 0 (old
behaviour); the review clip is still produced.
"""
import os
import shutil
import subprocess
import tempfile

import config
from debug.dbg_utils import create_video_from_images


def ffmpeg_available():
    """True if an `ffmpeg` binary is on PATH (needed for no-re-encode concat)."""
    return shutil.which("ffmpeg") is not None


def concat_videos(base_video, clip, out_path):
    """Append `clip` to `base_video` into `out_path` using ffmpeg stream copy.

    Uses the concat demuxer with `-c copy`: no re-encoding, so cost is proportional
    to the new clip only. Both inputs must share codec/size/fps (they do - both come
    from create_video_from_images).

    Returns True on success, False if ffmpeg is missing or the call failed.
    """
    if not ffmpeg_available():
        return False
    list_file = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            list_file = f.name
            for path in (base_video, clip):
                f.write(f"file '{os.path.abspath(path)}'\n")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
        return proc.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False
    finally:
        if list_file and os.path.exists(list_file):
            os.remove(list_file)


def build_attempt_clip(base_name, start_idx, fps=None, folder_path=None, out_dir=None):
    """Encode the frames of the CURRENT attempt (start_idx -> end of sequence) into
    `<video_folder>/<base_name>_attempt_<start>_inf.mp4`.

    Returns the clip path, or None when the attempt produced no frames.
    """
    folder_path = folder_path or config.trajectory_folder
    out_dir = out_dir or config.video_folder
    os.makedirs(out_dir, exist_ok=True)
    return create_video_from_images(
        folder_path=folder_path,
        output_video_folder_path=out_dir,
        base_name=base_name,
        start_idx=start_idx,
        end_idx=float("inf"),
        fps=fps or config.trajectory_video_fps,
        output_filename=f"{base_name}_attempt_{start_idx}_inf.mp4",
    )


def update_full_video(base_name, clip_path, folder_path=None, out_dir=None):
    """Append `clip_path` to the running `<video_folder>/<base_name>_full.mp4`.

    First call simply copies the clip. Subsequent calls concat without re-encoding.
    If ffmpeg is missing, falls back to re-encoding the whole sequence from frame 0.

    Returns the full-video path, or None on failure.
    """
    folder_path = folder_path or config.trajectory_folder
    out_dir = os.path.abspath(out_dir or config.video_folder)
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, f"{base_name}_full.mp4")

    if clip_path and not os.path.exists(full_path):
        shutil.copy2(clip_path, full_path)
        return full_path

    if clip_path and concat_videos(full_path, clip_path, full_path + ".tmp.mp4"):
        os.replace(full_path + ".tmp.mp4", full_path)
        return full_path

    # Fallback: no ffmpeg (or concat failed) -> re-encode everything from the start.
    if os.path.exists(full_path + ".tmp.mp4"):
        os.remove(full_path + ".tmp.mp4")
    return create_video_from_images(
        folder_path=folder_path,
        output_video_folder_path=out_dir,
        base_name=base_name,
        start_idx=0,
        end_idx=float("inf"),
        fps=config.trajectory_video_fps,
        output_filename=f"{base_name}_full.mp4",
    )


def build_review_clips(logger, start_idx, cam_bases=None):
    """Per-camera: encode the current attempt's clip and grow that camera's full video.

    Args:
        start_idx: first trajectory step of the attempt under review
                   (task.start_attempt_trajectory_step).
        cam_bases: image base names, default head + wrist.

    Returns: list of clip paths (attempt clips only, in head-then-wrist order); empty
    when the attempt produced no frames (robot never moved).
    """
    from contextlib import redirect_stdout
    import sys as _sys

    cam_bases = cam_bases or [config.trajectory_image_base, config.trajectory_wrist_image_base]
    clips = []
    for base_name in cam_bases:
        try:
            with redirect_stdout(_sys.stderr):
                clip = build_attempt_clip(base_name, start_idx)
                if clip:
                    update_full_video(base_name, clip)
        except Exception as e:
            logger.info(f"Warning: could not build trajectory video for '{base_name}': {e}")
            continue
        if clip:
            clips.append(clip)
    return clips
