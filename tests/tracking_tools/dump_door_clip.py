"""Dump the door scene as a clip + per-point 2D ground truth, for offline 2D-tracker studies.

Runs the same scripted door motion as ``tracking_eval`` (hinge + latch), sampled like the
real-time harness (``--fps`` camera frames per second, motion at 10 motion frames / s), and
stores for every camera frame and camera: clean RGB, RGB with the disc occluder painted in
motion frames [occ0, occ1), depth, handle-link mask, and the ground-truth pixel + visibility
of every seed point (body-frame offsets carried by the latch link pose).

No tracking runs here. Output: outputs/tracking_probes/door_clip/<tag>.npz

    python tests/tracking_tools/dump_door_clip.py --seeding region --seed-points 12 --fps 30
"""
import argparse
import logging
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))
os.chdir(_REPO)

import tracking_eval as te  # noqa: E402
from tracking import geometry  # noqa: E402
from tracking.seeding import sim_link_mask  # noqa: E402
from tracking_scenario import paint_occluder  # noqa: E402

log = logging.getLogger("dump_door_clip")


def main():
    p = argparse.ArgumentParser()
    te.add_eval_args(p)
    p.add_argument("--fps", type=float, default=30.0, help="camera frames per second")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    args.scene = "door"
    if not args.occlusion:
        args.occlusion = [15, 25]
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    occ = tuple(args.occlusion)
    args.occlusion = None                     # render clean; the occluder is painted here
    scene, _kw = te.build_scene(args)
    cams = tuple(args.cameras or te.config.tracking_cameras)
    scene.select_cameras(cams)
    scene.reset()
    obj = scene.objects[0]
    seeds = np.asarray(scene.seed_points(obj), dtype=float).reshape(-1, 3)
    pos0, quat0 = scene.ground_truth(obj)
    local = (seeds - pos0) @ te.rotation_matrix(quat0)
    seed_off = seeds.mean(axis=0) - pos0
    log.info("[dump] %d seed points, cameras %s, occlusion mf %s on %s",
             len(seeds), cams, occ, args.occluder_cam)

    n_cam = int(round(scene.n_frames * args.fps / te.DEFAULT_MOTION_RATE_HZ))
    out = {c: {"rgb": [], "rgb_occ": [], "depth": [], "mask": [], "gt_px": [], "gt_vis": [],
               "occ": []} for c in cams}
    mfs = []
    for k in range(n_cam):
        mf = k * te.DEFAULT_MOTION_RATE_HZ / args.fps
        scene.advance(mf)
        pos, quat = scene.ground_truth(obj)
        R = te.rotation_matrix(quat)
        world = pos + local @ R.T
        occluded = occ[0] <= mf < occ[1]
        mfs.append(mf)
        for c in cams:
            view = scene.robot.capture_camera_view(c, scene.env)
            o = out[c]
            o["rgb"].append(view.rgb[..., :3].copy())
            o["depth"].append(view.depth.astype(np.float32).copy())
            o["mask"].append(sim_link_mask(view, scene.door_id, scene.handle_link))
            px, vis = [], []
            for w in world:
                ok, pix, _det = geometry.is_visible(view, w)
                if pix is None:
                    pix, _z = geometry.project_world_to_pixel(view, w)
                px.append(pix if pix is not None else (np.nan, np.nan))
                vis.append(bool(ok))
            occ_here = occluded and (args.occluder_cam == ["all"] or c in args.occluder_cam)
            if occ_here:
                # same placement as SimSceneDriver.views: GT origin + seed offset (world frame)
                paint_occluder(view, pos + seed_off, radius=args.occluder_radius)
            o["rgb_occ"].append(view.rgb[..., :3].copy())
            o["gt_px"].append(np.asarray(px, dtype=float))
            o["gt_vis"].append(np.asarray(vis))
            o["occ"].append(occ_here)
        if k % 30 == 0:
            log.info("[dump] cam frame %d / %d (mf %.2f)", k, n_cam, mf)

    tag = args.tag or f"door_{args.seeding}{len(seeds)}_fps{int(args.fps)}"
    dst = os.path.join(_REPO, "outputs", "tracking_probes", "door_clip")
    os.makedirs(dst, exist_ok=True)
    payload = {"mf": np.asarray(mfs), "cams": np.asarray(cams), "occlusion_mf": np.asarray(occ),
               "seeds_world": seeds, "seeds_local": local}
    for c in cams:
        for key, vals in out[c].items():
            payload[f"{c}_{key}"] = np.asarray(vals)
    path = os.path.join(dst, tag + ".npz")
    np.savez_compressed(path, **payload)
    log.info("[dump] wrote %s (%d frames)", path, n_cam)


if __name__ == "__main__":
    main()
