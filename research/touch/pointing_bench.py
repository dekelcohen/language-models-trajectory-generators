"""Offline wrist-camera pointing benchmark (port opening on the usb scene).

Runs the exact ``API.locate_with_wrist_camera`` prompt on saved wrist views
(``research/touch/capture_wrist_views.py``, or any ``wrist_locate_*.png`` with its ``_cam.npz``
sidecar) and scores each model's best point against the eval-only truth: pixel error, 3D error of
the depth-lifted point, hit rate (3D error < --hit-mm) and latency. Answers are cached in
``<views>/answers_<model>_<effort>.json`` so reruns cost nothing.

    conda run -n vlm_traj python research/touch/pointing_bench.py outputs/_bench/wrist_views \
        --models copilot-gpt-6.1-terra:low copilot-gpt-6.1-terra:medium or-google/gemini-3.8-flash:low
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))
except ImportError:
    pass
import config  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402
from llm_providers.model_registry import get_model_info  # noqa: E402
from api import wrist_locate_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("views")
ap.add_argument("--models", nargs="+", required=True, help="model[:effort] ... (gpu-molmo[:url] supported)")
ap.add_argument("--max-calls", type=int, default=50, help="new (uncached) calls per model")
ap.add_argument("--hit-mm", type=float, default=5.0)
ap.add_argument("--max-points", type=int, default=4)
args = ap.parse_args()
if models.logger is None:
    import logging
    logging.basicConfig(level=logging.WARNING)
    models.logger = logging.getLogger("pointing_bench")
if getattr(utils, "args", None) is None:
    from types import SimpleNamespace
    utils.args = SimpleNamespace(save_grasp_inputs=False)

truth = json.load(open(os.path.join(args.views, "truth.json")))["views"]


def lift(cam, u, v):
    z, _ = utils.sample_surface_depth(cam["depth"], int(u), int(v))
    if z is None or not np.isfinite(z):
        return None
    return np.asarray(utils.get_world_point_world_frame(cam["pos"], cam["quat"], "wrist", cam["size"], [int(u), int(v), z],
                                                        cam_info=cam["info"])).ravel()[:3]


def load(rec):
    stem = os.path.join(args.views, rec["view"])
    d = np.load(stem + "_cam.npz")
    img = Image.open(stem + ".png")
    cam = {"depth": d["depth"], "pos": d["cam_pos"].tolist(), "quat": d["cam_quat"].tolist(),
           "info": json.loads(str(d["cam_info"])), "size": img.size, "desc": str(d["description"])}
    port = np.asarray(rec["truth_port_pos"], float)
    px = utils.project_3d_world_pos_to_2d_pixel(cam["pos"], cam["quat"], "wrist", img.size, port, cam["info"])
    visible = False
    if px and 0 <= px[0] < img.size[0] and 0 <= px[1] < img.size[1]:
        wp = lift(cam, px[0], px[1])
        visible = wp is not None and np.linalg.norm(wp - port) < 0.012   # not hidden by the gripper / plug
    return stem + ".png", cam, port, px, visible


def ask(model, effort, image_path, desc, w, h):
    if model.startswith("gpu-molmo"):
        from providers.pointing.molmo_remote import point_with_molmo, url_from_model
        return [{"point": p} for p in point_with_molmo(image_path, desc, args.max_points, url=url_from_model(model))], "xy_pixels"
    key = get_model_info(model).pointing_coords_format
    prompt = wrist_locate_prompt(desc, args.max_points, config.affordance_coords_format_by_key[key])
    msgs = models.call_llm_cached(None, None, model, prompt, [], role="user", image_paths=[image_path],
                                  options={"max_tokens": 60000, "reasoning_effort": effort, "cache": None})
    text = (msgs[-1]["content"] if msgs and isinstance(msgs[-1], dict) else "") or ""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        items = json.loads(m.group(0)) if m else []
    except Exception:
        items = []
    return (items if isinstance(items, list) else [items]), key


def to_px(p, key, w, h):
    if key == "yx_norm_1000":
        from features_markers.bbox_providers.gemini_bbox_provider import denormalize_yx_point_to_xy_pixels
        return denormalize_yx_point_to_xy_pixels(p, w, h)
    return float(p[0]), float(p[1])


views = [(rec, *load(rec)) for rec in truth]
print(f"{len(views)} views, port visible in {sum(v[-1] for v in views)}")
rows = []
for spec in args.models:
    model, _, effort = spec.partition(":") if not spec.startswith("gpu-molmo") else (spec, "", "")
    effort = effort or None
    cache_path = os.path.join(args.views, f"answers_{re.sub(r'[^A-Za-z0-9.-]+', '_', model)}_{effort}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    calls = 0
    px_err, err3, hits, hits10, lat, n_vis, empty = [], [], 0, 0, [], 0, 0
    for rec, img_path, cam, port, tpx, visible in views:
        if not visible:
            continue
        n_vis += 1
        w, h = cam["size"]
        if rec["view"] not in cache:
            if calls >= args.max_calls:
                continue
            t0 = time.time()
            try:
                items, key = ask(model, effort, img_path, cam["desc"], w, h)
            except Exception as e:
                print(f"  {spec} {rec['view']}: error {e}")
                calls += 1
                continue
            cache[rec["view"]] = {"items": items, "key": key, "latency_s": round(time.time() - t0, 2)}
            calls += 1
            json.dump(cache, open(cache_path, "w"), indent=1)
        a = cache[rec["view"]]
        lat.append(a["latency_s"])
        pts = [it.get("point") for it in a["items"] if isinstance(it, dict) and isinstance(it.get("point"), (list, tuple))]
        if not pts or a["key"] is None:
            empty += 1
            continue
        u, v = to_px(pts[0], a["key"], w, h)
        px_err.append(float(np.hypot(u - tpx[0], v - tpx[1])))
        wp = lift(cam, u, v) if 0 <= u < w and 0 <= v < h else None
        e = float(np.linalg.norm(wp - port) * 1000) if wp is not None else float("inf")
        err3.append(e)
        hits += e < args.hit_mm
        hits10 += e < 2 * args.hit_mm
        print(f"  {spec:40s} {rec['view']} hold={rec['holding']} px_err={px_err[-1]:6.1f} 3d={e:7.1f}mm "
              f"lat={a['latency_s']:5.1f}s n={len(pts)}")
    fin = [e for e in err3 if np.isfinite(e)]
    rows.append({"model": spec, "views": n_vis, "hit_rate": round(hits / max(1, n_vis), 2), "hit_rate_2x": round(hits10 / max(1, n_vis), 2), "empty": empty,
                 "px_err_median": round(float(np.median(px_err)), 1) if px_err else None,
                 "err3d_mm_median": round(float(np.median(fin)), 1) if fin else None,
                 "latency_s_median": round(float(np.median(lat)), 1) if lat else None, "new_calls": calls})

print("\n| model | views | hit rate (<%.0f mm) | hit rate (<%.0f mm) | no answer | px err (median) | 3D err mm (median) | latency s (median) |" % (args.hit_mm, 2 * args.hit_mm))
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['model']} | {r['views']} | {r['hit_rate']} | {r['hit_rate_2x']} | {r['empty']} | {r['px_err_median']} | "
          f"{r['err3d_mm_median']} | {r['latency_s_median']} |")
json.dump(rows, open(os.path.join(args.views, "bench_results.json"), "w"), indent=1)
