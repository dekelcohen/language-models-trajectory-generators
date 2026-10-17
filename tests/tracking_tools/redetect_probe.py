"""Re-detection probe: find the handle after the occlusion from a pre-occlusion context crop.

Uses the clip from ``dump_door_clip.py``. Reference = the last camera frame before the
occlusion window; a ``--crop`` px square around the GT handle centroid (handle + door context).
Targets = post-occlusion frames. Each matcher maps the reference handle centroid into the
target image; error = px distance to the GT centroid there.

    python tests/tracking_tools/redetect_probe.py outputs/tracking_probes/door_clip/door_region12_fps30.npz
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


def gt_centroid(gt_px, k):
    return np.nanmean(gt_px[k], axis=0)


def crop_at(img, c, size):
    h = size // 2
    pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REFLECT)
    x, y = int(round(c[0])), int(round(c[1]))
    return pad[y:y + size, x:x + size].copy(), np.array([x - h, y - h], float)


def ncc_search(ref, origin, cref, tgt, scales=(0.8, 0.9, 1.0, 1.1, 1.25),
               angles=(-30, -20, -10, 0, 10, 20, 30)):
    """Brute-force NCC over scale x in-plane rotation; returns mapped centroid + score."""
    g_t = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    g_r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    size = g_r.shape[0]
    centre = np.array([size / 2.0, size / 2.0])
    off = cref - origin - centre                      # handle centroid relative to crop centre
    best = (-2, None)
    for s in scales:
        for a in angles:
            M = cv2.getRotationMatrix2D(tuple(centre), a, s)
            warped = cv2.warpAffine(g_r, M, (size, size), borderMode=cv2.BORDER_REFLECT)
            # only the central disc survives rotation without border artefacts
            r = int(size * 0.35)
            tpl = warped[size // 2 - r:size // 2 + r, size // 2 - r:size // 2 + r]
            if tpl.shape[0] >= g_t.shape[0]:
                continue
            res = cv2.matchTemplate(g_t, tpl, cv2.TM_CCOEFF_NORMED)
            _mn, mx, _l, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                rot = M[:, :2]
                c = np.array(loc, float) + r + rot @ off
                best = (mx, c)
    return best[1], best[0]


def ncc_detail(ref, origin, cref, tgt, gt_c, scales=(0.8, 0.9, 1.0, 1.1, 1.25),
               angles=(-30, -20, -10, 0, 10, 20, 30), topk=3, nms_px=8):
    """Like ncc_search, but returns the ranked candidate list and the best score *at the truth*.

    Candidates: top-``topk`` peaks per (scale, angle) map, mapped to a handle-centroid pixel,
    merged with NMS on that pixel. ``at_gt``: best score over transforms of the template
    placement that maps the reference centroid within 2 px of the GT centroid.
    """
    g_t = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    g_r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    size = g_r.shape[0]
    centre = np.array([size / 2.0, size / 2.0])
    off = cref - origin - centre
    cands, at_gt = [], (-2.0, None)
    for s in scales:
        for a in angles:
            M = cv2.getRotationMatrix2D(tuple(centre), a, s)
            warped = cv2.warpAffine(g_r, M, (size, size), borderMode=cv2.BORDER_REFLECT)
            r = int(size * 0.35)
            tpl = warped[size // 2 - r:size // 2 + r, size // 2 - r:size // 2 + r]
            if tpl.shape[0] >= g_t.shape[0]:
                continue
            res = cv2.matchTemplate(g_t, tpl, cv2.TM_CCOEFF_NORMED)
            shift = r + M[:, :2] @ off
            loc_gt = np.round(gt_c - shift).astype(int)
            y0, x0 = max(0, loc_gt[1] - 2), max(0, loc_gt[0] - 2)
            win = res[y0:loc_gt[1] + 3, x0:loc_gt[0] + 3]
            if win.size and float(win.max()) > at_gt[0]:
                at_gt = (float(win.max()), (s, a))
            work = res.copy()
            for _ in range(topk):
                _mn, mx, _l, loc = cv2.minMaxLoc(work)
                cands.append((float(mx), np.array(loc, float) + shift, (s, a)))
                cv2.circle(work, loc, nms_px, -2.0, -1)
    cands.sort(key=lambda c: -c[0])
    merged = []
    for sc, c, tf in cands:
        if all(np.linalg.norm(c - m[1]) > nms_px for m in merged):
            merged.append((sc, c, tf))
    return merged, at_gt


def sim_map(ref, origin, cref, tgt, scales=(0.8, 0.9, 1.0, 1.1, 1.25),
            angles=(-30, -20, -10, 0, 10, 20, 30)):
    """Best NCC over (scale, angle) per *handle-centroid pixel* of the target image (-1 = no
    template placement maps the reference centroid there)."""
    g_t = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    g_r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    size = g_r.shape[0]
    centre = np.array([size / 2.0, size / 2.0])
    off = cref - origin - centre
    out = np.full(g_t.shape, -1.0, np.float32)
    for s in scales:
        for a in angles:
            M = cv2.getRotationMatrix2D(tuple(centre), a, s)
            warped = cv2.warpAffine(g_r, M, (size, size), borderMode=cv2.BORDER_REFLECT)
            r = int(size * 0.35)
            tpl = warped[size // 2 - r:size // 2 + r, size // 2 - r:size // 2 + r]
            if tpl.shape[0] >= g_t.shape[0]:
                continue
            res = cv2.matchTemplate(g_t, tpl, cv2.TM_CCOEFF_NORMED)
            dx, dy = np.round(r + M[:, :2] @ off).astype(int)
            h, w = res.shape
            y0, x0 = max(0, dy), max(0, dx)
            y1, x1 = min(out.shape[0], dy + h), min(out.shape[1], dx + w)
            if y1 > y0 and x1 > x0:
                np.maximum(out[y0:y1, x0:x1], res[y0 - dy:y1 - dy, x0 - dx:x1 - dx],
                           out=out[y0:y1, x0:x1])
    return out


def render_sim(ref, tgt, smap, gt_c, merged, size, k, mf, scale=3):
    """Target frame with the NCC map (jet, alpha) + GT (green ring) + top-5 (numbered)."""
    valid = smap > -1
    norm = np.clip((smap - 0.2) / 0.6, 0, 1)                    # 0.2 .. 0.8 -> full colormap
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    img = tgt.copy()
    img[valid] = (0.45 * img[valid] + 0.55 * heat[valid]).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def P(c):
        return tuple(int(v * scale) for v in c)
    cv2.circle(img, P(gt_c), 5 * scale, (0, 255, 0), 2)
    for i, (sc, c, _tf) in enumerate(merged[:5]):
        col = (255, 255, 255) if i else (255, 0, 255)
        cv2.drawMarker(img, P(c), col, cv2.MARKER_TILTED_CROSS, 4 * scale, 2)
        cv2.putText(img, f"{i + 1}:{sc:.2f}", (P(c)[0] + 6, P(c)[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    refz = cv2.resize(ref, (img.shape[0] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    r = int(size * 0.35) * refz.shape[0] // size
    cz = refz.shape[0] // 2
    cv2.rectangle(refz, (cz - r, cz - r), (cz + r, cz + r), (255, 255, 0), 1)
    side = np.zeros((img.shape[0], refz.shape[1], 3), np.uint8)
    side[:refz.shape[0]] = refz
    cv2.putText(side, f"ref crop {size}px", (6, refz.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    cv2.putText(side, "yellow box = template", (6, refz.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 0), 1)
    cv2.putText(side, "green ring = truth", (6, refz.shape[0] + 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1)
    cv2.putText(side, "magenta X = top-1", (6, refz.shape[0] + 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 0, 255), 1)
    cv2.putText(side, "jet: NCC 0.2 (blue)..0.8 (red)", (6, refz.shape[0] + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    banner = f"frame {k} mf {mf:.1f}  crop {size}"
    cv2.putText(img, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.hstack([side, img])


def feature_match(ref, origin, cref, tgt, kind):
    g_t = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    g_r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    if kind == "sift":
        det, norm = cv2.SIFT_create(), cv2.NORM_L2
    elif kind == "akaze":
        det, norm = cv2.AKAZE_create(threshold=1e-4), cv2.NORM_HAMMING
    else:
        det, norm = cv2.ORB_create(1000, fastThreshold=5, edgeThreshold=8, patchSize=15), cv2.NORM_HAMMING
    k1, d1 = det.detectAndCompute(g_r, None)
    k2, d2 = det.detectAndCompute(g_t, None)
    if d1 is None or d2 is None or len(k1) < 3 or len(k2) < 3:
        return None, 0
    m = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [a for a, *b in m if b and a.distance < 0.8 * b[0].distance]
    if len(good) < 3:
        return None, len(good)
    p1 = np.float32([k1[g.queryIdx].pt for g in good])
    p2 = np.float32([k2[g.trainIdx].pt for g in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if A is None:
        return None, 0
    c = A[:, :2] @ (cref - origin) + A[:, 2]
    return c, int(inl.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("--cam", default="head")
    p.add_argument("--crops", type=int, nargs="+", default=[30, 60, 100, 140])
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--detail", type=int, nargs="*", default=None, metavar="CROP",
                   help="NCC similarity detail for these crop sizes: top-5 candidates vs score at GT")
    args = p.parse_args()
    d = np.load(args.clip)
    mf, occ = d["mf"], d["occlusion_mf"]
    rgb, gt = d[f"{args.cam}_rgb"], d[f"{args.cam}_gt_px"]
    mask = d[f"{args.cam}_mask"]
    k_ref = int(np.where(mf < occ[0])[0].max())
    targets = [k for k in range(len(mf)) if mf[k] >= occ[1]][::args.every]
    cref = gt_centroid(gt, k_ref)
    if args.detail is not None:
        detail = {}
        for size in args.detail or [100, 140]:
            ref, origin = crop_at(rgb[k_ref], cref, size)
            print(f"[redetect] NCC detail, crop {size} (template {2 * int(size * 0.35)} px), "
                  f"ref frame {k_ref}: top-5 = score/err_px/on_handle_mask; at_gt = best score "
                  f"of a placement within 2 px of the truth; rank = first candidate < 5 px")
            rows = []
            frames = []
            for k in targets:
                g = gt_centroid(gt, k)
                merged, (s_gt, tf_gt) = ncc_detail(ref, origin, cref, rgb[k], g)
                errs = [float(np.linalg.norm(c - g)) for _s, c, _t in merged]
                rank = next((i + 1 for i, e in enumerate(errs) if e < 5), None)

                def on_mask(c):
                    x, y = int(round(c[0])), int(round(c[1]))
                    return bool(0 <= x < mask.shape[2] and 0 <= y < mask.shape[1] and mask[k, y, x])
                top = [(round(sc, 3), round(e, 1), on_mask(c)) for (sc, c, _t), e in
                       zip(merged[:5], errs[:5])]
                rows.append({"frame": k, "mf": round(float(mf[k]), 1), "at_gt": round(s_gt, 3),
                             "at_gt_tf": tf_gt, "top1": top[0][0], "margin": round(top[0][0] - s_gt, 3),
                             "rank_gt": rank, "top5": top})
                print(f"  f{k:3d} mf{mf[k]:5.1f} at_gt={s_gt:.3f} top1={top[0][0]:.3f} "
                      f"margin={top[0][0] - s_gt:+.3f} rank_gt={rank}  top5={top}")
                frame = render_sim(ref, rgb[k], sim_map(ref, origin, cref, rgb[k]), g, merged,
                                   size, k, mf[k])
                frames.append(frame)
            vid = os.path.join(os.path.dirname(os.path.abspath(args.clip)), "..", "..",
                               "tracking_video", f"door_redetect_ncc{size}_{args.cam}.mp4")
            vid = os.path.abspath(vid)
            os.makedirs(os.path.dirname(vid), exist_ok=True)
            wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, 30.0 / args.every / 2),
                                 (frames[0].shape[1], frames[0].shape[0]))
            for fr in frames:
                wr.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            wr.release()
            pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 3).astype(int)]
            cv2.imwrite(vid[:-4] + ".png", cv2.cvtColor(np.vstack(pick), cv2.COLOR_RGB2BGR))
            print(f"[redetect] wrote {vid} (+ .png)")
            detail[size] = rows
        out = os.path.splitext(args.clip)[0] + f"_redetect_ncc_detail_{args.cam}.json"
        json.dump(detail, open(out, "w"), indent=1)
        print(f"[redetect] wrote {out}")
        return
    print(f"[redetect] ref frame {k_ref} (mf {mf[k_ref]:.1f}), {len(targets)} targets, "
          f"GT centroid moved {np.linalg.norm(gt_centroid(gt, targets[-1]) - cref):.0f} px by the end")
    rows = {}
    for size in args.crops:
        ref, origin = crop_at(rgb[k_ref], cref, size)
        for name in ("ncc", "sift", "akaze", "orb"):
            errs, t0 = [], time.time()
            for k in targets:
                if name == "ncc":
                    c, sc = ncc_search(ref, origin, cref, rgb[k])
                else:
                    c, sc = feature_match(ref, origin, cref, rgb[k], name)
                errs.append(np.inf if c is None else float(np.linalg.norm(c - gt_centroid(gt, k))))
            e = np.asarray(errs)
            key = f"{name}{size}"
            rows[key] = {"med_px": round(float(np.median(e)), 1),
                         "ok5": round(float(np.mean(e < 5)), 2), "ok10": round(float(np.mean(e < 10)), 2),
                         "ms": round(1000 * (time.time() - t0) / len(targets), 1)}
            print(f"  {key:10s} {rows[key]}")
    out = os.path.splitext(args.clip)[0] + f"_redetect_{args.cam}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"[redetect] wrote {out}")


if __name__ == "__main__":
    main()
