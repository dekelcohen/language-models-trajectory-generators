"""Render RoMa v2 door-handle re-detection panels + video through gpu_server's ``romav2`` module.

For each pair of an ``export_sim_pairs.py`` manifest (A = last pre-occlusion frame, B = a
post-occlusion frame) and each mode:

* ``fullA_box`` - match the whole A to B, read the warp at the handle anchor;
* ``crop_only`` - match only the stored 100 px crop (the re-detection-from-memory setting);

it calls ``romav2/match`` (dense) on the server, then draws five panels: A + box + anchor;
box pixels mapped into B coloured by overlap with the direct / local-median / box-RANSAC /
GT markers; the overlap map; B warped into A; and match lines A|B - thick lines from pixels
around the handle (the area that matters), thin ones from the rest of the box (global context),
and the anchor's own match. See docs/plans/3d_tracking.md section 13.12, "How to read".

    # server (RoMa env): python -m gpu_server.server --modules romav2 --tcp 127.0.0.1:8776 \
    #     --set romav2.root=<RoMaV2 checkout> --preload romav2
    python tests/tracking_tools/render_romav2_redetect.py outputs/tracking_probes/romav2_pairs/head_sweep/manifest.json \
        --url tcp://127.0.0.1:8776 --out outputs/tracking_video
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpu_server.client import GpuClient  # noqa: E402

OVERLAP_THR = 0.3
BOX_STEP = 2
RANSAC_PX = 3.0
HANDLE_R_PX = 12        # A pixels around the anchor whose matches are drawn thick
N_HANDLE_LINES = 10
N_CONTEXT_LINES = 8
CONTEXT_R_PX = (20, 45)  # ring around the anchor that samples the door for global context
CONTEXT_OVERLAP = 0.6
MODES = {"fullA_box": "fullA+box", "crop_only": "crop-only"}


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(img[..., ::-1])


def sample_dense(dense, pts_src, src_size):
    """Bilinear lookup of a dense (h,w,C) map at source-image pixels (RoMa's pixel-centre grid)."""
    h, w = dense.shape[:2]
    ws, hs = src_size
    mx = (pts_src[:, 0] * w / ws - 0.5).astype(np.float32).reshape(-1, 1)
    my = (pts_src[:, 1] * h / hs - 0.5).astype(np.float32).reshape(-1, 1)
    out = cv2.remap(dense.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(len(pts_src), -1)


def box_grid(box, step):
    xs = np.arange(box[0] + 0.5, box[2], step)
    ys = np.arange(box[1] + 0.5, box[3], step)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def spread_sample(pts, n, rng):
    """n points spread over ``pts`` (farthest-point sampling from a random start)."""
    if len(pts) <= n:
        return np.arange(len(pts))
    sel = [int(rng.integers(len(pts)))]
    d = np.hypot(*(pts - pts[sel[0]]).T)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.hypot(*(pts - pts[i]).T))
    return np.array(sel)


def redetect(client, A, B, pair, mode):
    box = [float(v) for v in pair["crop_A"]]
    anchor = np.asarray(pair["gt_A"], float)
    kw = {"box": box} if mode == "crop_only" else {}
    t0 = time.perf_counter()
    reply, out = client.call("romav2", "match", img_a=A, img_b=B, points_a=anchor[None], dense=True,
                             local_radius_px=8, overlap_thr=OVERLAP_THR, **kw)
    rtt = time.perf_counter() - t0
    origin = np.asarray(reply["origin"], float)
    src_size = tuple(reply["src_size"])
    pA = box_grid(box, BOX_STEP)                                        # full-A pixels
    pB = sample_dense(out["warp_b"], pA - origin, src_size)
    ov = sample_dense(out["overlap_a"][..., None], pA - origin, src_size)[:, 0]
    ransac = None
    keep = ov >= OVERLAP_THR
    if keep.sum() >= 6:
        M, _ = cv2.estimateAffinePartial2D(pA[keep].astype(np.float32), pB[keep].astype(np.float32),
                                           method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        if M is not None:
            ransac = M[:, :2] @ anchor + M[:, 2]
    local = out["points_b_local"][0]
    return dict(box=box, anchor=anchor, origin=origin, src_size=src_size, pA=pA, pB=pB, ov=ov,
                direct=out["points_b"][0], ov_direct=float(out["overlap"][0]),
                local=None if np.isnan(local).any() else local, n_local=int(out["n_local"][0]),
                ransac=ransac, warp_b=out["warp_b"], overlap_a=out["overlap_a"],
                match_s=reply["match_s"], rtt_s=rtt, setting=reply["setting"], amp=reply["amp"])


def err(p, gt):
    return None if p is None or gt is None else round(float(np.hypot(*(np.asarray(p) - gt))), 2)


def warp_b_into_src(v, B):
    wb = v["warp_b"]
    img = cv2.remap(B.astype(np.float32) / 255, wb[..., 0] - 0.5, wb[..., 1] - 0.5, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    ov = v["overlap_a"][..., None]
    return np.clip(ov * img + (1 - ov), 0, 1)


def draw_matches(ax, A, B, v, gt_B, rng):
    """A's box (zoomed to B's height) | B side by side; handle lines thick, door context thin."""
    x0, y0, x1, y1 = [int(round(c)) for c in v["box"]]
    crop = A[y0:y1, x0:x1]
    s = B.shape[0] / crop.shape[0]
    A_s = cv2.resize(crop, (int(round(crop.shape[1] * s)), B.shape[0]), interpolation=cv2.INTER_NEAREST)
    gap = 12
    canvas = np.full((B.shape[0], A_s.shape[1] + gap + B.shape[1], 3), 255, np.uint8)
    canvas[:, :A_s.shape[1]] = A_s
    canvas[:, A_s.shape[1] + gap:] = B
    xoff = A_s.shape[1] + gap
    ax.imshow(canvas)

    def a_xy(p):           # full-A pixel -> canvas
        return (np.asarray(p, float) - [x0, y0]) * s

    def b_xy(p):
        return np.asarray(p, float) + [xoff, 0]

    conf = v["ov"] >= OVERLAP_THR
    dist = np.hypot(*(v["pA"] - v["anchor"]).T)
    hi = np.where((dist <= HANDLE_R_PX) & conf)[0]
    # context from the door around the handle: a ring, confident pixels only (box corners are floor/sky)
    ci = np.where((dist >= CONTEXT_R_PX[0]) & (dist <= CONTEXT_R_PX[1]) & (v["ov"] >= CONTEXT_OVERLAP))[0]
    if len(ci) < N_CONTEXT_LINES:
        ci = np.where((dist > HANDLE_R_PX) & conf)[0]
    hsel = hi[spread_sample(v["pA"][hi], N_HANDLE_LINES, rng)] if len(hi) else []
    csel = ci[spread_sample(v["pA"][ci], N_CONTEXT_LINES, rng)] if len(ci) else []
    for i in csel:
        a, b = a_xy(v["pA"][i]), b_xy(v["pB"][i])
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="deepskyblue", lw=0.9, alpha=0.75)
        ax.plot(*a, "o", color="deepskyblue", ms=3)
        ax.plot(*b, "o", color="deepskyblue", ms=3)
    for i in hsel:
        a, b = a_xy(v["pA"][i]), b_xy(v["pB"][i])
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="orangered", lw=1.6, alpha=0.9)
        ax.plot(*a, "o", color="orangered", ms=4)
        ax.plot(*b, "o", color="orangered", ms=4)
    a, b = a_xy(v["anchor"]), b_xy(v["direct"])
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="white", lw=4.0)
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="orange", lw=2.2)
    ax.plot(*a, "+", color="orange", ms=14, mew=2.5)
    ax.plot(*b, "+", color="orange", ms=14, mew=2.5)
    if gt_B is not None:
        ax.plot(*b_xy(gt_B), "o", mfc="none", mec="lime", ms=16, mew=2)
    ax.set_title(f"matches A box (zoomed) | B: {len(hsel)} on handle (r<={HANDLE_R_PX}px, red), "
                 f"{len(csel)} door context (r {CONTEXT_R_PX[0]}-{CONTEXT_R_PX[1]}px, blue), anchor (orange)",
                 fontsize=9)


def render(pair, A, B, v, mode, rng):
    gt_B = np.asarray(pair["gt_B"], float) if pair.get("gt_B") else None
    x0, y0 = v["origin"]
    ws, hs = v["src_size"]
    A_src = A[int(y0):int(y0) + hs, int(x0):int(x0) + ws]
    fig = plt.figure(figsize=(30, 5.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1.15, 1.15, 1, 2.1])
    ax = [fig.add_subplot(gs[0, i]) for i in range(5)]
    bx = v["box"]
    ax[0].imshow(A)
    ax[0].add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec="cyan", lw=1.5))
    ax[0].plot(*v["anchor"], "r+", ms=14, mew=2)
    ax[0].set_title(f"A (input): {'matched crop' if mode == 'crop_only' else 'full frame, box shown'} + anchor")
    ax[1].imshow(B)
    sc = ax[1].scatter(v["pB"][:, 0], v["pB"][:, 1], c=v["ov"], s=4, cmap="viridis", vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(sc, ax=ax[1], fraction=0.046, label="overlap")
    if v["ransac"] is not None:
        ax[1].plot(*v["ransac"], "rx", ms=14, mew=3, label=f"RANSAC box ({err(v['ransac'], gt_B)} px)")
    if v["local"] is not None:
        ax[1].plot(*v["local"], "x", color="magenta", ms=12, mew=2, label=f"local median ({err(v['local'], gt_B)} px)")
    ax[1].plot(*v["direct"], "+", color="orange", ms=14, mew=2.5, label=f"direct warp ({err(v['direct'], gt_B)} px)")
    if gt_B is not None:
        ax[1].plot(*gt_B, "o", mfc="none", mec="lime", ms=16, mew=2, label="GT")
    ax[1].set_xlim(0, B.shape[1]); ax[1].set_ylim(B.shape[0], 0)
    ax[1].legend(loc="lower right", fontsize=8)
    ax[1].set_title("RESULT in B: correct when orange + is inside the green ring")
    ax[2].imshow(A_src, alpha=0.35)
    im = ax[2].imshow(v["overlap_a"], cmap="magma", vmin=0, vmax=1, alpha=0.75, extent=(0, ws, hs, 0))
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    ax[2].set_title("overlap_AB over A (dark = not visible in B)")
    ax[3].imshow(warp_b_into_src(v, B), extent=(0, ws, hs, 0))
    ax[3].set_title("B warped into A (white = no overlap)")
    draw_matches(ax[4], A, B, v, gt_B, rng)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{pair['name']}  [{MODES[mode]}, RoMa v2 {v['setting']}, {v['amp']}, "
                 f"match {v['match_s']:.2f}s via gpu_server]  direct err {err(v['direct'], gt_B)} px, "
                 f"overlap {v['ov_direct']:.2f}")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--url", default="tcp://127.0.0.1:8776")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--out", default="outputs/tracking_video")
    ap.add_argument("--prefix", default="door_redetect_romav2")
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(args.manifest))
    man = json.load(open(args.manifest))
    cam = man.get("cam", "cam")
    os.makedirs(args.out, exist_ok=True)
    results = []
    with GpuClient(args.url, timeout=600) as client:
        print("server:", client.ping())
        for mode in args.modes:
            writer, last = None, None
            for pair in man["pairs"]:
                A = load_rgb(os.path.join(base, pair["A"]))
                B = load_rgb(os.path.join(base, pair["B"]))
                v = redetect(client, A, B, pair, mode)
                gt_B = np.asarray(pair["gt_B"], float) if pair.get("gt_B") else None
                row = {"pair": pair["name"], "mode": MODES[mode], "match_s": v["match_s"],
                       "rtt_s": round(v["rtt_s"], 3), "err_direct_px": err(v["direct"], gt_B),
                       "err_local_px": err(v["local"], gt_B), "err_ransac_px": err(v["ransac"], gt_B),
                       "ov_direct": round(v["ov_direct"], 3), "pred_direct": np.round(v["direct"], 1).tolist()}
                results.append(row)
                print(json.dumps(row))
                frame = render(pair, A, B, v, mode, np.random.default_rng(0))
                if writer is None:
                    size = (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2)
                    path = os.path.join(args.out, f"{args.prefix}_{mode}_{cam}.mp4")
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
                writer.write(cv2.resize(frame, size)[..., ::-1])
                last = frame
            writer.release()
            png = os.path.join(args.out, f"{args.prefix}_{mode}_{cam}.png")
            if not cv2.imwrite(png, last[..., ::-1]):
                print(f"WARNING: could not write {png} (open in a viewer?)")
            print("wrote", path, png)
    with open(os.path.join(base, "results_gpu_server.json"), "w") as f:
        json.dump(results, f, indent=1)
    for mode in args.modes:
        e = [r["err_direct_px"] for r in results if r["mode"] == MODES[mode] and r["err_direct_px"] is not None]
        if e:
            print(f"{MODES[mode]:10s} direct: median {np.median(e):.2f} px, max {np.max(e):.2f} px, "
                  f"<5px {np.mean(np.asarray(e) < 5):.0%}")


if __name__ == "__main__":
    main()
