"""Video of a 2D tracker run on a dumped door clip (``cotracker_alone.py`` output).

Tiles = the given track keys (e.g. head__clean__online head__occluded__online), each the full
camera frame x``--scale`` plus a x6 zoom on the handle. Marks: green ring = GT pixel of a seed
point (red ring = GT point self-occluded), filled cyan dot = tracked point, magenta line = error,
yellow cross = tracked centroid, green cross = GT centroid. Banner: frame, motion frame, median px
error over GT-visible points, OCCLUSION.

    python tests/tracking_tools/render_clip_tracks.py CLIP.npz TRACKS.npz --keys head__clean__online ...
"""
import argparse
import os

import cv2
import numpy as np


def zoom(img, c, r=24, f=6):
    pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)
    c = np.nan_to_num(np.asarray(c, float), nan=img.shape[1] / 2)
    x = int(np.clip(round(c[0]), 0, img.shape[1] - 1))
    y = int(np.clip(round(c[1]), 0, img.shape[0] - 1))
    return cv2.resize(pad[y:y + 2 * r, x:x + 2 * r], None, fx=f, fy=f,
                      interpolation=cv2.INTER_NEAREST), np.array([x - r, y - r], float)


def draw(img, gt, vis, tr, origin, s):
    def P(p):
        return tuple(int(v) for v in ((np.asarray(p) - origin) * s))
    for g, v, t in zip(gt, vis, tr):
        cv2.line(img, P(g), P(t), (255, 0, 255), 1)
        cv2.circle(img, P(g), max(3, s), (0, 255, 0) if v else (255, 0, 0), 1)
        cv2.circle(img, P(t), max(2, s // 2), (0, 255, 255), -1)
    cv2.drawMarker(img, P(gt.mean(0)), (0, 255, 0), cv2.MARKER_CROSS, 6 * s, 2)
    cv2.drawMarker(img, P(tr.mean(0)), (255, 255, 0), cv2.MARKER_CROSS, 6 * s, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("tracks")
    p.add_argument("--keys", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args()
    d, t = np.load(args.clip), np.load(args.tracks)
    mf, occ = d["mf"], d["occlusion_mf"]
    writer = None
    for k in range(len(mf)):
        tiles = []
        for key in args.keys:
            cam, clip, _var = key.split("__")
            rgb = d[f"{cam}_{'rgb_occ' if clip == 'occluded' else 'rgb'}"][k]
            gt_all, vis_all = d[f"{cam}_gt_px"], d[f"{cam}_gt_vis"]
            keep = vis_all[0] & np.isfinite(gt_all[0]).all(1)       # the queried points
            gt, vis, tr = gt_all[k, keep], vis_all[k, keep], t[key][k]
            s = args.scale
            full = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            draw(full, gt, vis, tr, np.zeros(2), s)
            z, o = zoom(rgb, gt.mean(0), r=full.shape[0] // 12, f=6)
            draw(z, gt, vis, tr, o, 6)
            z = cv2.resize(z, (full.shape[0], full.shape[0]), interpolation=cv2.INTER_NEAREST)
            tile = np.hstack([full, z])
            err = np.linalg.norm(tr - gt, axis=1)[vis]
            occl = bool(d[f"{cam}_occ"][k]) if clip == "occluded" else False
            banner = np.zeros((40, tile.shape[1], 3), np.uint8)
            cv2.putText(banner, f"{key}  frame {k:3d}  mf {mf[k]:4.1f}  med err "
                        f"{np.median(err) if err.size else float('nan'):4.1f}px  "
                        f"centroid {np.linalg.norm(tr.mean(0) - gt.mean(0)):4.1f}px"
                        + ("  OCCLUSION" if occl else ""), (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 80, 80) if occl else (255, 255, 255), 1)
            tiles.append(np.vstack([banner, tile]))
        frame = np.vstack(tiles)
        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[render_clip_tracks] wrote {args.out} ({len(mf)} frames)")


if __name__ == "__main__":
    main()
