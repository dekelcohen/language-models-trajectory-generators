"""Key-frame selection for review models that take images, not video (e.g. copilot CLI models).

A fixed stride from the attempt start often skips the FINAL frame (the one that decides
success) and spends most of the budget on frames where nothing moves. Here:

- frames are the files that exist for this attempt (``<base>_<idx>.png``, idx >= start);
- the first and the last frame are always kept, plus a frame shortly before the end (the
  settled final state vs. the last motion);
- the rest of the budget is spread evenly along the attempt's cumulative *visual change*
  (mean abs grey-level difference between consecutive low-res frames, with a small floor so
  long static stretches still get some coverage). More frames go where things move: grasp,
  slip, door swing, pour.
"""
import os
import re

import numpy as np


def existing_indices(root, base_name, start_idx=0, end_idx=None, ext="png"):
    if not os.path.isdir(root):
        return []
    pat = re.compile(rf"^{re.escape(base_name)}_(\d+)\.{ext}$")
    out = []
    for f in os.listdir(root):
        m = pat.match(f)
        if m:
            i = int(m.group(1))
            if i >= int(start_idx) and (end_idx is None or i <= int(end_idx)):
                out.append(i)
    return sorted(out)


def _thumb(path, size=48):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size)), dtype=np.float32)


def change_profile(paths):
    """Per-frame visual change vs. the previous frame (0 for the first)."""
    diffs = [0.0]
    prev = _thumb(paths[0]) if paths else None
    for p in paths[1:]:
        cur = _thumb(p)
        diffs.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    return np.asarray(diffs, dtype=np.float64)


def pick_by_change(diffs, budget, floor_frac=0.15, tail_gap=3):
    """Positions (0..n-1) to keep: first, last, one ~``tail_gap`` before the end, and the rest
    evenly spaced in cumulative change. Pure function (testable without images)."""
    n = len(diffs)
    if n <= budget:
        return list(range(n))
    must = {0, n - 1, max(0, n - 1 - tail_gap)}
    d = np.asarray(diffs, dtype=np.float64).copy()
    floor = floor_frac * (float(d[1:].mean()) if n > 1 and d[1:].mean() > 0 else 1.0)
    d[1:] += floor
    cum = np.cumsum(d)
    k = max(0, budget - len(must))
    targets = np.linspace(0, cum[-1], k + 2)[1:-1] if k else []
    chosen = set(must)
    for t in targets:
        pos = int(np.searchsorted(cum, t))
        pos = min(max(pos, 0), n - 1)
        # nearest free position
        for off in range(n):
            for cand in (pos + off, pos - off):
                if 0 <= cand < n and cand not in chosen:
                    chosen.add(cand)
                    break
            else:
                continue
            break
    return sorted(chosen)[:budget] if len(chosen) > budget else sorted(chosen)


def select_key_frames(root, base_name, start_idx, budget, end_idx=None, ext="png"):
    """-> list of (frame_idx, path), chronological, at most ``budget`` long."""
    idx = existing_indices(root, base_name, start_idx, end_idx, ext)
    if not idx or budget <= 0:
        return []
    paths = [os.path.join(root, f"{base_name}_{i}.{ext}") for i in idx]
    if len(paths) <= budget:
        return list(zip(idx, paths))
    try:
        keep = pick_by_change(change_profile(paths), budget)
    except Exception:
        keep = sorted(set(np.linspace(0, len(paths) - 1, budget).round().astype(int).tolist()))
    return [(idx[k], paths[k]) for k in keep]


def frames_listing(head, wrist):
    """Prompt lines for the attached frames, in attachment order; marks each camera's final frame."""
    lines = []
    for cam, frames in (("head cam", head), ("wrist cam", wrist)):
        for j, (i, p) in enumerate(frames):
            tag = "  <- FINAL STATE" if j == len(frames) - 1 else ""
            lines.append(f"  {cam} step {i}: {p}{tag}")
    return "\n".join(lines)
