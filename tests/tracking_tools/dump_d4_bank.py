"""D4 diagnosis: are the re-acquisition bank patches on the handle? (cause 5)

Re-runs D4 (door, head occluded 15-25, default wrist, affordance seeding, pose_tracker,
30 fps real-time) at HEAD with two wrappers:
  * capture_bank: per banked seed point -> pixel under the trusted pose vs GT pixel,
    handle-mask coverage of the 11x11 patch at both, NCC(bank patch, GT patch).
  * reacquire: why each search attempt failed.
Writes <out>/bank.csv, search.csv, crops/*.png and a summary to stdout.
"""
import csv, os, sys, logging
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_OUT = os.path.join(_REPO, "outputs", "tracking_probes")
import numpy as np
import cv2

ROOT = _REPO
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
OUT = os.path.join(_OUT, "d4_bank")
os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)

import argparse
import tracking_eval as te
import tracking.reacquire as rq
from tracking import geometry
from tracking.seeding import sim_link_mask

logging.basicConfig(level=logging.WARNING, format="%(message)s")
p = argparse.ArgumentParser(); te.add_eval_args(p)
args = p.parse_args(["--scene", "door", "--occlusion", "15", "25", "--occluder-cam", "head",
                     "--tracker3d", "pose_tracker"])
scene, kw = te.build_scene(args)
runner = te.TrackingEvalRunner(scene, tracker_provider="template", tracker3d="pose_tracker",
                               cameras=args.cameras, keep_frames=True,
                               tracker_kwargs=kw.get("template"), **te.timing_kwargs(args))
OBJ = scene.objects[0]
bank_rows, search_rows = [], []
_orig_capture, _orig_reacq = rq.capture_bank, rq.reacquire


def gt_world():
    pos, quat = scene.ground_truth(OBJ)
    return np.asarray(pos) + runner.seed_local[OBJ] @ te.rotation_matrix(quat).T


def ncc(a, b):
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    a = a.astype(float) - a.mean(); b = b.astype(float) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def capture_wrap(view, pose, seed_points_world, frame, half=None, angles=None):
    bank = _orig_capture(view, pose, seed_points_world, frame, half=half, angles=angles)
    h = int(half or 5)
    gray = rq._gray(view.rgb)
    m_handle = sim_link_mask(view, scene.door_id, scene.handle_link)
    m_door = sim_link_mask(view, scene.door_id, None)
    est_w, gt_w = pose.apply(seed_points_world), gt_world()
    pe, pg = [], []
    for i in range(len(seed_points_world)):
        px_e, _ = geometry.project_world_to_pixel(view, est_w[i])
        px_g, _ = geometry.project_world_to_pixel(view, gt_w[i])
        if px_e is None or px_g is None:
            continue
        def cov(m, px):
            if m is None:
                return float("nan")
            x, y = int(round(px[0])), int(round(px[1]))
            sub = m[max(0, y - h):y + h + 1, max(0, x - h):x + h + 1]
            return float(sub.mean()) if sub.size else float("nan")
        _ok, _px, vis_detail = geometry.is_visible(view, est_w[i])
        bank_rows.append(dict(vis=str(vis_detail)[:40],
            frame=frame, mf=round(float(scene.current_frame), 2), cam=view.name, seed=i,
            banked=i in bank.templates, err3d_mm=round(1000 * float(np.linalg.norm(est_w[i] - gt_w[i])), 1),
            px_err=round(float(np.hypot(px_e[0] - px_g[0], px_e[1] - px_g[1])), 1),
            handle_cov_bank=round(cov(m_handle, px_e), 2), handle_cov_gt=round(cov(m_handle, px_g), 2),
            door_cov_bank=round(cov(m_door, px_e), 2),
            ncc_bank_vs_gt=round(ncc(rq._crop(gray, px_e, h), rq._crop(gray, px_g, h)), 2)))
        pe.append(px_e); pg.append(px_g)
    if pe:   # crop around GT, x6, red = bank pixels, green = GT pixels
        c = np.mean(pg, axis=0); R = 24
        x0, y0 = int(c[0]) - R, int(c[1]) - R
        img = cv2.cvtColor(np.ascontiguousarray(view.rgb[..., :3]), cv2.COLOR_RGB2BGR)
        img = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT)
        crop = cv2.resize(img[y0 + R:y0 + 3 * R, x0 + R:x0 + 3 * R], None, fx=6, fy=6,
                          interpolation=cv2.INTER_NEAREST)
        for pts, col in ((pg, (0, 255, 0)), (pe, (0, 0, 255))):
            for q in pts:
                qq = (int((q[0] - x0) * 6), int((q[1] - y0) * 6))
                cv2.circle(crop, qq, 5, col, 2)
                if col == (0, 0, 255):
                    cv2.rectangle(crop, (qq[0] - 6 * h, qq[1] - 6 * h), (qq[0] + 6 * h, qq[1] + 6 * h), col, 1)
        cv2.imwrite(os.path.join(OUT, "crops", f"bank_f{frame:03d}_{view.name}.png"), crop)
    return bank


_orig_search = rq.search_candidates
_cap = {"on": False, "found": []}


def search_wrap(view, bank, centres, radius, topk, ncc_min):
    out = _orig_search(view, bank, centres, radius, topk, ncc_min)
    if _cap["on"]:
        _cap["found"].append((view, out))
    return out


def reacq_wrap(views, banks, seed_points_world, hypotheses, frame_idx=None, max_dev_m=None):
    _cap.update(on=True, found=[])
    R, t, info = _orig_reacq(views, banks, seed_points_world, hypotheses, frame_idx, max_dev_m)
    _cap["on"] = False
    gtw = gt_world()
    gt = gtw.mean(axis=0)
    # per seed: closest candidate (3D) to the truth, and best banked-NCC right at the GT pixel
    near, ncc_gt, gt_in_view = {}, {}, {}
    for view, out in _cap["found"][:len(hypotheses) * len(banks)]:
        for i, peaks in out.items():
            pix = np.array([q for _s, q in peaks], dtype=float)
            w, valid = geometry.points_to_world(view, pix)
            for ww in w[valid]:
                d = 1000 * float(np.linalg.norm(ww - gtw[i]))
                near[i] = round(min(near.get(i, 1e9), d), 1)
    for cam, bank in banks.items():
        view = views.get(cam)
        if view is None:
            continue
        gray = rq._gray(view.rgb)
        for i, tm in bank.templates.items():
            ok, px, det = geometry.is_visible(view, gtw[i])
            gt_in_view[i] = str(det)[:12] if not ok else "vis"
            if px is None:
                continue
            h = tm[0].shape[0] // 2
            sub = rq._crop(gray, px, h + 2)
            if sub is None:
                continue
            ncc_gt[i] = round(max(float(cv2.matchTemplate(sub, t_, cv2.TM_CCOEFF_NORMED).max())
                                  for t_ in tm), 2)
    found_err = None if R is None else round(1000 * float(np.linalg.norm(
        R @ np.asarray(seed_points_world).mean(axis=0) + t - gt)), 1)
    search_rows.append(dict(mf=round(float(scene.current_frame), 2),
                            banks={c: (b.frame, len(b.templates)) for c, b in banks.items()},
                            cands=info.get("cands"), inliers=info.get("inliers"),
                            verified=info.get("verified"), why=info.get("why", "ACCEPTED"),
                            found_err_mm=found_err, gt_vis=gt_in_view,
                            nearest_cand_to_gt_mm=near, best_ncc_at_gt=ncc_gt))
    return R, t, info


rq.capture_bank, rq.reacquire, rq.search_candidates = capture_wrap, reacq_wrap, search_wrap
payload = runner.run()
agg = payload["aggregate"]
print("D4 @HEAD: median %.3f p95 %.3f max %.3f recover=%s" % (
    agg["median_l2_m"], agg["p95_l2_m"], agg["max_l2_m"], agg["frames_to_recover"]))
te.write_report(payload, os.path.join(OUT, "d4_head.json"))
for name, rows in (("bank.csv", bank_rows), ("search.csv", search_rows)):
    if rows:
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("bank captures:")
for (fr, cam) in sorted({(r["frame"], r["cam"]) for r in bank_rows}):
    rs = [r for r in bank_rows if r["frame"] == fr and r["cam"] == cam]
    print(f"  f{fr:3d} mf={rs[0]['mf']:5.1f} {cam:5s} banked={sum(r['banked'] for r in rs)}/{len(rs)} "
          f"err3d={np.median([r['err3d_mm'] for r in rs]):5.1f}mm px_err={np.median([r['px_err'] for r in rs]):4.1f} "
          f"handle_cov bank={np.mean([r['handle_cov_bank'] for r in rs]):.2f} gt={np.mean([r['handle_cov_gt'] for r in rs]):.2f} "
          f"door_cov bank={np.mean([r['door_cov_bank'] for r in rs]):.2f} ncc(bank,gt)={np.nanmean([r['ncc_bank_vs_gt'] for r in rs]):.2f} "
          f"vis={sorted({r['vis'] for r in rs})}")
print("search attempts:")
for r in search_rows:
    print("  ", r)
