"""Scene perception (perception VLM) + target-object affordance pointing.

Runs the `--planner-perception-vlm` model on the current head-camera image before every
planner LLM call and returns a free-text scene analysis that is injected into the planner
prompt's SCENE ANALYSIS section.

When `--affordance-points` is enabled (default), the perception prompt also asks for a
ranked JSON block of 2D grasp-affordance points on the target object. Those 2D points are
parsed out here, converted to 3D world coordinates via `API.convert_2d_point_to_3d_world`,
and spliced back into the analysis text as 3D coords - the raw 2D points are removed so the
planner never reasons over pixel coordinates.
"""
import re
import json
import os
import shutil

import numpy as np
from PIL import Image

import config
import models
from config import OK, PROGRESS, WARNING, ENDC


def _capture_fresh_head_image(ctx):
    """Re-render the head camera so perception analyses the CURRENT world state.

    Without this the perception VLM (and the planner image + the reviewer's
    start-of-attempt scene snapshot) saw `rgb_image_head.png` as left by the PREVIOUS
    subtask's detect_object - i.e. the scene BEFORE that subtask executed. Symptom: after
    the occluding cylinder had already been moved away, the analysis still reported it
    standing in front of the door.

    Best-effort: on failure the stale image is used rather than aborting the run.
    """
    if ctx.api is None:
        return False
    try:
        ctx.api._capture_head_image_and_depth()
        return True
    except Exception as e:
        ctx.logger.info(WARNING + f"Perception: head re-capture failed ({e}); using last head image." + ENDC)
        return False


def _save_scene_analysis_image(ctx):
    """Snapshot the head image the perception VLM analyzed.

    Kept as its own file (not a trajectory frame) so the reviewer VLM can be shown the
    start-of-attempt scene separately from the many trajectory frames. Returns the saved
    path, or the live head-image path if the copy failed.
    """
    src = config.rgb_image_head_path
    try:
        step = getattr(ctx.api, "trajectory_step", 0) if ctx.api is not None else 0
        dst = config.scene_analysis_image_path.format(step=step)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        ctx.logger.info(PROGRESS + f"Perception: saved scene-analysis image to {dst}" + ENDC)
        return dst
    except Exception as e:
        ctx.logger.info(WARNING + f"Warning: failed to save scene-analysis image: {e}" + ENDC)
        return src


def _parse_affordance_points_block(text):
    """Extract the AFFORDANCE_POINTS JSON block from the perception VLM response.

    Returns (analysis_text_without_block, points_list). points_list is a list of
    {"point": [...], "label": str}; empty if none found. The block is stripped from
    the returned analysis text (its 2D points are replaced later by 3D coords).
    """
    marker = "AFFORDANCE_POINTS:"
    idx = text.find(marker)
    if idx == -1:
        return text, []
    analysis = text[:idx].rstrip()
    tail = text[idx + len(marker):]
    # Prefer a fenced ```json ... ``` block; fall back to first [...] array.
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", tail, re.DOTALL)
    if not m:
        m = re.search(r"(\[.*\])", tail, re.DOTALL)
    points = []
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                points = [p for p in parsed if isinstance(p, dict)]
        except Exception:
            points = []
    return analysis, points


def _process_affordance_points(ctx, command, analysis_text, points, captured_fresh=False):
    """Convert parsed 2D affordance points to 3D world coords and splice them into
    the scene-analysis text (replacing the raw 2D points so the planner never sees
    them). Stores the 2D->3D mapping on ctx for debugging. Returns updated text.

    `captured_fresh`: the head image was just re-captured by run_scene_perception, so the
    2D points refer to it - skip the redundant re-capture inside the conversion."""
    args = ctx.args
    logger = ctx.logger
    if not points or ctx.api is None:
        return analysis_text

    is_gemini = "gemini" in args.planner_perception_vlm.lower()
    try:
        rgb = Image.open(config.rgb_image_head_path)
        width, height = rgb.size
    except Exception:
        width = height = None

    points_xy = []
    labels = []
    for item in points:
        p = item.get("point") or item.get("pt")
        if not p or not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        labels.append(item.get("label") or item.get("name") or str(command))
        if is_gemini:
            if width is None:
                continue
            from features_markers.bbox_providers.gemini_bbox_provider import denormalize_yx_point_to_xy_pixels
            points_xy.append(denormalize_yx_point_to_xy_pixels(p, width, height))
        else:
            # Non-gemini VLMs return [x, y] pixel coordinates directly.
            points_xy.append([float(p[0]), float(p[1])])

    if not points_xy:
        return analysis_text

    object_name = labels[0] if labels else str(command)
    try:
        world_points = ctx.api.convert_2d_point_to_3d_world(points_xy, object_name, capture=not captured_fresh)
    except Exception as e:
        logger.info(WARNING + f"Affordance 2D->3D conversion failed: {e}." + ENDC)
        return analysis_text

    ctx.affordance_points.append({
        "object": object_name,
        "points_2d": points_xy,
        "points_3d": [None if wp is None else list(np.array(wp).flatten()) for wp in world_points],
    })

    lines = ["AFFORDANCE POINTS (target-object grasp affordances, 3D world coords, best-first):"]
    for i, wp in enumerate(world_points):
        label = labels[i] if i < len(labels) else object_name
        if wp is None:
            lines.append(f"  {i+1}. {label}: (unavailable)")
        else:
            coords = list(np.around(np.array(wp).flatten(), 3))
            lines.append(f"  {i+1}. {label}: {coords}")
    return (analysis_text.rstrip() + "\n\n" + "\n".join(lines)).strip()


def run_scene_perception(ctx, command):
    """Run the perception VLM (--planner-perception-vlm) on the head image.

    First re-captures the head camera (`_capture_fresh_head_image`) so the analysis, the
    saved scene-analysis snapshot and the planner's attached image all reflect the CURRENT
    world state - not the state left over from the previous subtask's detect_object.

    Returns its free-text scene analysis, injected into the planner prompt's
    SCENE ANALYSIS section. The prompt also asks for a ranked JSON block of 2D
    grasp-affordance points on the target object; these are parsed out, converted to
    3D world coordinates (via API.convert_2d_point_to_3d_world), and spliced back into
    the analysis text as 3D coords (the raw 2D points are removed to avoid confusing
    the planner). Disable that part with --no-affordance-points. Best-effort: on any
    failure returns a short fallback string so the planner still runs (it can fall back
    to detect_object)."""
    from prompts.scene_perception_prompt import SCENE_PERCEPTION_PROMPT, AFFORDANCE_POINTING_SECTION
    args = ctx.args
    logger = ctx.logger
    affordance_enabled = args.affordance_points
    if affordance_enabled:
        is_gemini = "gemini" in args.planner_perception_vlm.lower()
        coords_format = (
            config.affordance_coords_format_gemini
            if is_gemini else
            config.affordance_coords_format_pixels
        )
        affordance_section = AFFORDANCE_POINTING_SECTION.replace("COORDINATES_FORMAT_PLACEHOLDER", coords_format)
    else:
        affordance_section = ""
    prompt = (
        SCENE_PERCEPTION_PROMPT
        .replace("[INSERT USER COMMAND TASK]", str(command))
        .replace("[INSERT AFFORDANCE POINTING SECTION]", affordance_section)
    )
    image_paths = [config.rgb_image_head_path] if args.lm_images else None
    captured_fresh = _capture_fresh_head_image(ctx)
    ctx.scene_analysis_image_path = _save_scene_analysis_image(ctx)
    try:
        logger.info(PROGRESS + f"Perception: analyzing scene with {args.planner_perception_vlm}..." + ENDC)
        messages = models.call_llm_cached(
            ctx.main_connection, ctx.client, args.planner_perception_vlm, prompt, [], role="system",
            image_paths=image_paths,
            options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": ctx.llm_cache},
        )
        text = messages[-1]["content"] if messages and isinstance(messages[-1], dict) else ""
        text = (text or "").strip()
        analysis, points = _parse_affordance_points_block(text)
        if affordance_enabled:
            analysis = _process_affordance_points(ctx, command, analysis, points, captured_fresh=captured_fresh)
        logger.info(OK + "Perception: scene analysis ready." + ENDC)
        return analysis or "(perception returned no analysis)"
    except Exception as e:
        logger.info(WARNING + f"Perception VLM failed: {e}. Proceeding without scene analysis." + ENDC)
        return "(scene analysis unavailable; inspect the scene yourself with detect_object(...))"
