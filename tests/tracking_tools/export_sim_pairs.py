"""Export sim re-detection image pairs (PNG) and optionally upload them to imgbb.

Feeds ``romav2_redetect_colab.ipynb``. Uses the door clip from ``dump_door_clip.py``. Settings
match ``redetect_probe.py``:
  A      = last camera frame before the occlusion (full frame + a ``--crop`` px square crop
           centred on the GT handle centroid);
  B      = query frames (default: first / middle / last post-occlusion frame);
  gt_B   = GT handle centroid in each query frame, so the notebook can score the error in px.

Only sim renders are exported/uploaded. The imgbb key is read from ``IMGBB_API_KEY`` and is
never written to disk. The script prints a ``PAIRS = [...]`` snippet to paste into the notebook
and writes ``manifest.json`` next to the PNGs.

    python tests/tracking_tools/export_sim_pairs.py outputs/tracking_probes/door_clip/door_region12_fps30.npz
    $env:IMGBB_API_KEY="..."; python tests/tracking_tools/export_sim_pairs.py <clip> --upload
"""
import argparse
import base64
import json
import os
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


def gt_centroid(gt_px, k):
    return np.nanmean(gt_px[k], axis=0)


def crop_box(c, size, W, H):
    """Square box centred on c, shifted (not padded) to stay inside the image."""
    size = min(size, W, H)
    x0 = int(round(c[0])) - size // 2
    y0 = int(round(c[1])) - size // 2
    x0 = max(0, min(W - size, x0))
    y0 = max(0, min(H - size, y0))
    return [x0, y0, x0 + size, y0 + size]


def imgbb_upload(path, key, expiration=None):
    with open(path, "rb") as f:
        payload = {"image": base64.b64encode(f.read()).decode(), "name": os.path.splitext(os.path.basename(path))[0]}
    q = {"key": key}
    if expiration:
        q["expiration"] = str(expiration)
    url = "https://api.imgbb.com/1/upload?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, data=urllib.parse.urlencode(payload).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    if not resp.get("success"):
        raise RuntimeError(f"imgbb upload failed for {path}: {resp}")
    return resp["data"]["url"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--cam", default="head")
    p.add_argument("--crop", type=int, default=100)
    p.add_argument("--queries", type=int, nargs="*", default=None,
                   help="query frame indices (default: first/middle/last post-occlusion frame)")
    p.add_argument("--occluded", action="store_true", help="export the occluded RGB stream")
    p.add_argument("--upscale", type=int, default=1, help="integer bicubic upscale of all PNGs (coords in PAIRS are scaled too)")
    p.add_argument("--out", default=None, help="default: <clip dir>/../romav2_pairs/<cam>")
    p.add_argument("--upload", action="store_true", help="upload PNGs to imgbb (needs IMGBB_API_KEY)")
    p.add_argument("--expiration", type=int, default=None, help="imgbb auto-delete after N seconds")
    args = p.parse_args()

    d = np.load(args.clip)
    mf, occ = d["mf"], d["occlusion_mf"]
    rgb = d[f"{args.cam}_rgb_occ" if args.occluded else f"{args.cam}_rgb"]
    gt = d[f"{args.cam}_gt_px"]
    H, W = rgb.shape[1:3]
    k_ref = int(np.where(mf < occ[0])[0].max())
    post = [k for k in range(len(mf)) if mf[k] >= occ[1]]
    queries = args.queries if args.queries else [post[0], post[len(post) // 2], post[-1]]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.clip)), "..", "romav2_pairs", args.cam)
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    s = max(1, args.upscale)

    def save(arr, name):
        im = Image.fromarray(arr)
        if s > 1:
            im = im.resize((arr.shape[1] * s, arr.shape[0] * s), Image.BICUBIC)
        path = os.path.join(out, name)
        im.save(path)
        return path

    c_ref = gt_centroid(gt, k_ref)
    box = crop_box(c_ref, args.crop, W, H)
    tag = f"door_{args.cam}{'_occ' if args.occluded else ''}"
    path_A = save(rgb[k_ref], f"{tag}_A_f{k_ref:03d}.png")
    x0, y0, x1, y1 = box
    path_crop = save(np.ascontiguousarray(rgb[k_ref, y0:y1, x0:x1]), f"{tag}_A_f{k_ref:03d}_crop{args.crop}.png")
    items = []
    for k in queries:
        items.append({"frame": int(k), "mf": round(float(mf[k]), 1), "path": save(rgb[k], f"{tag}_B_f{k:03d}.png"),
                      "gt_B": [round(float(v) * s, 1) for v in gt_centroid(gt, k)]})
    print(f"[export] cam={args.cam} ref f{k_ref} (mf {mf[k_ref]:.1f}) crop box {box} (x{s}) "
          f"queries {[it['frame'] for it in items]} -> {out}")

    urls = {}
    if args.upload:
        key = os.environ.get("IMGBB_API_KEY")
        if not key:
            raise SystemExit("[export] set IMGBB_API_KEY to upload")
        for path in [path_A, path_crop] + [it["path"] for it in items]:
            urls[path] = imgbb_upload(path, key, args.expiration)
            print(f"[export] uploaded {os.path.basename(path)} -> {urls[path]}")

    def ref(path):
        return urls.get(path, os.path.basename(path))

    pairs = [{"name": f"{tag}_f{k_ref}->f{it['frame']}", "A": ref(path_A), "B": ref(it["path"]),
              "crop_A": [v * s for v in box], "A_crop_img": ref(path_crop), "gt_B": it["gt_B"],
              "gt_A": [round(float(v) * s, 1) for v in c_ref]} for it in items]
    manifest = {"clip": os.path.basename(args.clip), "cam": args.cam, "ref_frame": k_ref, "crop": args.crop,
                "upscale": s, "occluded": args.occluded, "pairs": pairs}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\n# ---- paste into the notebook's PAIRS cell ----")
    print("PAIRS = [\n" + ",\n".join("    " + json.dumps(p) for p in pairs) + "\n]")


if __name__ == "__main__":
    main()
