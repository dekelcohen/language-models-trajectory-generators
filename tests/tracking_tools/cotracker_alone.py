"""CoTracker3 alone on the dumped door clip: does the 2D tracker drift on the handle?

Runs CoTracker3 (online and/or offline, with/without the 6x6 support grid) on the clip from
``dump_door_clip.py`` - no 3D lift, no gates, no re-seeding - and scores every seed point
against its ground-truth pixel. Clean and occluded (disc painted on the head) variants.

GPU env (TORCH_HOME defaults to ~/.cache/torch):
    & $lapa_py tests/tracking_tools/cotracker_alone.py outputs/tracking_probes/door_clip/door_region12_fps30.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_TORCH_HOME = os.environ.get("TORCH_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "torch")
HUB_DIR = os.path.join(_TORCH_HOME, "hub", "facebookresearch_co-tracker_main")


def load(variant, device):
    os.environ.setdefault("TORCH_HOME", _TORCH_HOME)
    model = torch.hub.load(HUB_DIR, variant, source="local")
    return model.to(device).eval()


def run_online(model, video, queries, support, fp16):
    step = model.step
    T = video.shape[1]
    pad = (-(T - step)) % step + step           # make sure the last real frame is covered
    vid = torch.cat([video, video[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        model(video_chunk=vid, is_first_step=True, queries=queries, grid_size=0,
              add_support_grid=support)
        tracks = vis = None
        for ind in range(0, vid.shape[1] - step, step):
            tracks, vis = model(video_chunk=vid[:, ind:ind + 2 * step],
                                add_support_grid=support)
    return tracks[0, :T, :queries.shape[1]].float().cpu().numpy(), \
        vis[0, :T, :queries.shape[1]].float().cpu().numpy()


def run_offline(model, video, queries, support, fp16):
    """Whole clip at once; the 120-frame feature volume does not fit 4 GB -> CPU fallback."""
    dev = next(model.model.parameters()).device
    try:
        if dev.type == "cpu":
            raise torch.cuda.OutOfMemoryError
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            # predictor adds the support grid itself when grid_size == 0 (queries given)
            tracks, vis = model(video, queries=queries, backward_tracking=False)
    except torch.cuda.OutOfMemoryError:
        if dev.type != "cpu":
            print("    offline: CUDA OOM -> CPU")
        torch.cuda.empty_cache()
        model.cpu()
        with torch.no_grad():
            tracks, vis = model(video.cpu(), queries=queries.cpu(), backward_tracking=False)
    return tracks[0].float().cpu().numpy(), vis[0].float().cpu().numpy()


def score(tracks, vis, gt, gt_vis, mf, occ, mask):
    """Per-segment median / p90 px error over GT-visible points, plus 'on handle' rate."""
    err = np.linalg.norm(tracks - gt, axis=-1)                    # (T, N)
    err = np.where(gt_vis, err, np.nan)
    on_handle = np.zeros_like(err)
    H, W = mask.shape[1:]
    for t in range(len(tracks)):
        xy = np.round(tracks[t]).astype(int)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        on_handle[t, ok] = mask[t, xy[ok, 1], xy[ok, 0]]
    segs = {"pre": mf < occ[0], "occ": (mf >= occ[0]) & (mf < occ[1]),
            "post": mf >= occ[1], "last10": mf >= mf.max() - 3.4}
    out = {}
    for name, sel in segs.items():
        e = err[sel]
        e = e[np.isfinite(e)]
        out[name] = {"med_px": round(float(np.median(e)), 2) if e.size else None,
                     "p90_px": round(float(np.percentile(e, 90)), 2) if e.size else None,
                     "on_handle": round(float(np.mean(on_handle[sel])), 2),
                     "pred_vis": round(float(np.mean(vis[sel] > 0.5)), 2)}
    per_frame = np.nanmedian(err, axis=1)
    return out, per_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--cams", nargs="+", default=["head", "wrist"])
    p.add_argument("--variants", nargs="+", default=["online", "online+grid", "offline"])
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args()
    d = np.load(args.clip)
    mf, occ = d["mf"], d["occlusion_mf"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda" and not args.no_fp16
    print(f"[cotracker_alone] {args.clip} T={len(mf)} device={device} fp16={fp16} occ_mf={occ}")
    models = {}
    results = {}
    curves = {}
    tracks_out = {}
    for cam in args.cams:
        gt, gt_vis, mask = d[f"{cam}_gt_px"], d[f"{cam}_gt_vis"], d[f"{cam}_mask"]
        q0 = gt[0]
        keep = gt_vis[0] & np.isfinite(q0).all(axis=1)
        if not keep.any():
            print(f"  {cam}: no seed point visible at frame 0, skipped")
            continue
        q = torch.tensor(np.concatenate([np.zeros((keep.sum(), 1)), q0[keep]], 1),
                         dtype=torch.float32, device=device)[None]
        for clip in ("rgb", "rgb_occ"):
            if clip == "rgb_occ" and not d[f"{cam}_occ"].any():
                continue
            frames = d[f"{cam}_{clip}"].astype(np.float32)
            video = torch.tensor(frames).permute(0, 3, 1, 2)[None].to(device)
            for var in args.variants:
                kind = "online" if var.startswith("online") else "offline"
                name = "cotracker3_" + kind
                if name not in models:
                    models[name] = load(name, device)
                torch.cuda.empty_cache() if device == "cuda" else None
                t0 = time.time()
                if kind == "online":
                    tr, vi = run_online(models[name], video, q, var.endswith("+grid"), fp16)
                else:
                    tr, vi = run_offline(models[name], video, q, True, fp16)
                dt = time.time() - t0
                s, curve = score(tr, vi, gt[:, keep], gt_vis[:, keep], mf, occ, mask)
                key = f"{cam}/{'occluded' if clip == 'rgb_occ' else 'clean'}/{var}"
                results[key] = dict(s, n_points=int(keep.sum()), wall_s=round(dt, 2))
                tracks_out[key.replace("/", "__")] = tr
                curves[key] = [None if not np.isfinite(v) else round(float(v), 2) for v in curve]
                print(f"  {key:32s} n={keep.sum():2d} " + "  ".join(
                    f"{k}: {v['med_px']}/{v['p90_px']}px hdl={v['on_handle']} vis={v['pred_vis']}"
                    for k, v in s.items()) + f"  ({dt:.1f}s)")
    out = os.path.splitext(args.clip)[0] + "_cotracker_alone.json"
    with open(out, "w") as f:
        json.dump({"results": results, "per_frame_median_px": curves,
                   "mf": [round(float(v), 2) for v in mf]}, f, indent=1)
    print(f"[cotracker_alone] wrote {out}")
    np.savez_compressed(os.path.splitext(out)[0] + "_tracks.npz", **tracks_out)


if __name__ == "__main__":
    main()
