"""Did a detected object leave its place? Pixels + depth only (fixed head camera).

At a subtask's first ``detect_object`` of a label the head RGB, metric depth and the
segmentation mask are kept (``snapshot``). At review time a fresh head capture is compared
with that snapshot on the object's ORIGINAL pixels (``compare``):

- depth: share of the mask's pixels that now see something farther away (the background
  behind a removed object) or nearer (something - usually the arm - in front of it);
- appearance: normalised cross-correlation of the original bbox crop, searched in a small
  window around the original place (CPU template match, OpenCV).

The verdict tells the reviewer what the low-res clips often get wrong: a pole that was
toppled away is gone from its spot ("left_place"), a handle whose door swung open left its
spot, an object nobody touched is "in_place". Real-robot inputs only: a calibrated fixed
camera's RGB-D; no simulator state is read.
"""
import numpy as np

#: Depth change (m) that counts as "this pixel now sees something else".
DEPTH_DELTA_M = 0.03
#: Smallest usable mask (pixels).
MIN_MASK_PX = 25
#: Camera may not move between snapshot and comparison (m).
MAX_CAMERA_SHIFT_M = 0.005


def metric_depth(depth_raw, cam_info, camera="head"):
    """Raw depth buffer -> metres, using the encoding/near/far the sim adapter reports."""
    import config
    from sim_adapter import camera_math
    info = {}
    if isinstance(cam_info, dict):
        entry = cam_info.get(camera)
        info = dict(entry) if isinstance(entry, dict) else {}
        if "depth_encoding" not in info and "depth_encoding" in cam_info:
            info["depth_encoding"] = cam_info["depth_encoding"]
    enc = info.get("depth_encoding", camera_math.DEPTH_OPENGL)
    near = float(info.get("znear", config.near_plane))
    far = float(info.get("zfar", config.far_plane))
    return np.asarray(camera_math.depth_to_metric(np.asarray(depth_raw, dtype=np.float32), enc, near, far),
                      dtype=np.float32)


def snapshot(label, rgb, depth_m, mask, camera_position=None, step=None):
    """Keep what is needed to compare later; None if the mask is unusable."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(bool)
    rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if m.shape != depth_m.shape or m.shape != rgb.shape[:2] or int(m.sum()) < MIN_MASK_PX:
        return None
    ys, xs = np.nonzero(m)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return {"label": str(label), "rgb": rgb.copy(), "depth": depth_m.copy(), "mask": m, "bbox": bbox,
            "camera_position": None if camera_position is None else [float(v) for v in camera_position],
            "step": step}


def _gray(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _local_ncc(snap, gray_now):
    """Best NCC of the original bbox crop inside a small window around its original place."""
    import cv2
    x0, y0, x1, y1 = snap["bbox"]
    tmpl = _gray(snap["rgb"])[y0:y1, x0:x1]
    if tmpl.shape[0] < 3 or tmpl.shape[1] < 3 or float(tmpl.std()) < 1.0:
        return None
    h, w = gray_now.shape
    mx = max(4, int(0.15 * (x1 - x0)))
    my = max(4, int(0.15 * (y1 - y0)))
    wx0, wy0, wx1, wy1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    window = gray_now[wy0:wy1, wx0:wx1]
    if window.shape[0] < tmpl.shape[0] or window.shape[1] < tmpl.shape[1]:
        return None
    res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(np.nanmax(res))


def compare(snap, rgb_now, depth_now_m, camera_position=None):
    """Original place of ``snap`` now -> facts dict (``verdict`` + measurements) or None."""
    if snap is None:
        return None
    if (camera_position is not None and snap.get("camera_position") is not None
            and float(np.linalg.norm(np.subtract(camera_position, snap["camera_position"]))) > MAX_CAMERA_SHIFT_M):
        return None                        # camera moved: pixel comparison is meaningless
    rgb_now = np.asarray(rgb_now)[..., :3]
    depth_now_m = np.asarray(depth_now_m, dtype=np.float32)
    m = snap["mask"]
    if rgb_now.shape[:2] != m.shape or depth_now_m.shape != m.shape:
        return None
    d0, d1 = snap["depth"][m], depth_now_m[m]
    ok = np.isfinite(d0) & np.isfinite(d1) & (d0 > 0) & (d1 > 0)
    n = int(ok.sum())
    if n < MIN_MASK_PX:
        return None
    diff = d1[ok] - d0[ok]
    farther = float(np.mean(diff > DEPTH_DELTA_M))
    nearer = float(np.mean(diff < -DEPTH_DELTA_M))
    col = np.abs(rgb_now[m].astype(np.int16) - snap["rgb"][m].astype(np.int16)).sum(axis=1)
    changed = float(np.mean(col > 60))
    ncc = _local_ncc(snap, _gray(rgb_now))

    if nearer >= 0.5:
        verdict = "hidden"                 # something closer (usually the arm) covers the spot
    elif farther >= 0.5 or (changed >= 0.6 and (ncc is None or ncc < 0.5)):
        verdict = "left_place"
    elif farther < 0.15 and changed < 0.25 and (ncc is None or ncc >= 0.8):
        verdict = "in_place"
    else:
        verdict = "partly_moved"
    return {"label": snap["label"], "verdict": verdict, "farther": round(farther, 2),
            "nearer": round(nearer, 2), "changed": round(changed, 2),
            "ncc": None if ncc is None else round(ncc, 2), "n_px": n,
            "median_depth_shift_m": round(float(np.median(diff)), 3)}


_VERDICT_TEXT = {
    "left_place": "LEFT ITS ORIGINAL PLACE: its original spot is now empty",
    "in_place": "STILL IN ITS ORIGINAL PLACE (unchanged)",
    "partly_moved": "PARTLY MOVED or changed appearance at its original place",
    "hidden": ("its original spot is now COVERED by something nearer to the camera (the arm, or the part "
               "itself swung toward the camera) - this measurement cannot tell"),
}


def mark_shared_regions(snaps, facts_list, min_overlap=0.8):
    """Flag facts whose original masks are (nearly) the same pixels as another label's.

    A weak detector can return one blob for two names (a pole standing in front of a door ->
    'gray cylinder' and 'door' both = pole + door); the verdict then belongs to the REGION,
    not to each name. Sets ``f["same_region_as"]``.
    """
    by_label = {s["label"]: s["mask"] for s in snaps if s is not None}
    for f in facts_list:
        if not f:
            continue
        m = by_label.get(f["label"])
        same = []
        for lbl, other in by_label.items():
            if lbl == f["label"] or m is None or other.shape != m.shape:
                continue
            inter = float(np.logical_and(m, other).sum())
            if inter / max(1.0, float(min(m.sum(), other.sum()))) >= min_overlap:
                same.append(lbl)
        f["same_region_as"] = same
    return facts_list


def format_facts(facts_list, already=None):
    """Prompt section for the reviewer. ``already``: labels that had left their place before
    this attempt started (measured at the end of an earlier attempt)."""
    facts_list = [f for f in (facts_list or []) if f]
    if not facts_list:
        return None
    already = set(already or ())
    lines = ["OBJECT DISPLACEMENT (MEASURED by the fixed head camera: each object's ORIGINAL pixels, where it "
             "was first detected in this subtask, compared with a fresh RGB-D capture now; template match + "
             "depth, no guessing):"]
    for f in facts_list:
        detail = (f"{int(f['farther'] * 100)}% of those pixels now see something farther away, "
                  f"{int(f['nearer'] * 100)}% something nearer; appearance match "
                  f"{'n/a' if f['ncc'] is None else f['ncc']} (1.0 = unchanged)")
        line = f"  - '{f['label']}': {_VERDICT_TEXT[f['verdict']]} ({detail})."
        if f["label"] in already:
            line += (" It had ALREADY left its place before this attempt started - a goal state reached "
                     "in an earlier attempt still counts: judge the END STATE, not which attempt caused it.")
        if f.get("same_region_as"):
            line += (" CAUTION: the detector returned the SAME image region for "
                     + ", ".join(f"'{o}'" for o in f["same_region_as"])
                     + " - this verdict covers that whole region: it says SOMETHING there moved, not "
                     "which of these objects it was; use the clips to tell which.")
        lines.append(line)
    lines.append("  Judge the subtask GOAL from this: an object that had to be moved / cleared away and has "
                 "LEFT ITS ORIGINAL PLACE is no longer there (how it lies now does not matter unless the task "
                 "says so); a part that had to open (door, drawer) and LEFT ITS ORIGINAL PLACE has moved.")
    return "\n".join(lines)


def format_run_facts(facts_list):
    """Prompt block for baselines taken before the robot first moved in this run (earlier
    subtasks). Only objects that changed are listed; None if nothing did."""
    moved = [f for f in (facts_list or []) if f and f["verdict"] in ("left_place", "partly_moved")]
    if not moved:
        return None
    lines = ["SINCE THE START OF THE WHOLE COMMAND (same measurement, against head-camera pixels captured "
             "BEFORE THE ROBOT FIRST MOVED, i.e. before earlier subtasks):"]
    for f in moved:
        line = (f"  - '{f['label']}': {_VERDICT_TEXT[f['verdict']]} ({int(f['farther'] * 100)}% farther, "
                f"{int(f['nearer'] * 100)}% nearer; appearance match {'n/a' if f['ncc'] is None else f['ncc']}).")
        if f.get("same_region_as"):
            line += (" CAUTION: same image region as " + ", ".join(f"'{o}'" for o in f["same_region_as"])
                     + " - something there moved, not necessarily this object.")
        lines.append(line)
    lines.append("  A part that had to open (door, drawer) and appears here was ALREADY moved before this "
                 "subtask (e.g. pushed open by an earlier step): judge the END STATE - if it is open enough "
                 "now, the goal is met even if this subtask's own grasp/pull failed.")
    return "\n".join(lines)


def retry_hint(facts_list, task_text):
    """Improvement step for a failed relocate/clear subtask whose object already left its place."""
    from helpers.pull_proprio import RELOCATE_TASK_RE, is_articulated_task
    if not RELOCATE_TASK_RE.search(str(task_text or "")) or is_articulated_task(task_text):
        return None
    gone = [f["label"] for f in (facts_list or []) if f and f["verdict"] == "left_place"]
    if not gone:
        return None
    names = ", ".join(f"'{g}'" for g in gone)
    return (f"MEASURED (head camera RGB-D): {names} is no longer where it was first detected. Do NOT act at "
            "its old position. Detect it again first; if the subtask goal (e.g. clearing it out of the way) "
            "is already met, do not move it again - only move the arm clear of the camera view, then call "
            "task_completed().")
