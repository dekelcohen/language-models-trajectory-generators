"""Per-phase drift probe (pre / during / post occlusion) for one 2D x 3D configuration.

Per frame and camera it compares every tracked 2D point with the reprojection of *its own*
seed point's ground truth, and classifies points:
  stuck   - moved < 30 % of the GT motion since the camera's last (re)seed (GT moved > 10 px)
  latched - reported visible inside an occluded camera's disc during occlusion
Usage: same args as run_tracking_eval.py (first --provider / --tracker3d are used).
"""
import argparse, collections, logging, os, sys
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_OUT = os.path.join(_REPO, "outputs", "tracking_probes")
import numpy as np
ROOT = os.environ.get("TRK_ROOT", _REPO)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import tracking_eval as te
from tracking import geometry
from sim_adapter.transforms import rotation_matrix

p = argparse.ArgumentParser()
te.add_eval_args(p)
p.add_argument("--trace", action="store_true", help="print kf re-inits and mode changes")
p.add_argument("--every", type=int, default=0, help="print every Nth frame (0 = auto)")
p.add_argument("--detail", nargs=2, type=int, default=None, help="dump full meta for frames k in [lo, hi)")
p.add_argument("--cfg", nargs="*", default=[], help="config overrides key=value (python literal)")
a = p.parse_args()
import ast, config
for kv in a.cfg:
    k, v = kv.split("=", 1)
    try:
        v = ast.literal_eval(v)
    except Exception:
        pass
    setattr(config, k, v)
logging.basicConfig(level=logging.WARNING)
scene, kw_by = te.build_scene(a)
provider, lift = a.provider[0], a.tracker3d[0]
timing = te.timing_kwargs(a)
occ_cams = set(scene.occluder_cams()) if hasattr(scene, "occluder_cams") else set()
rows, seed_ref = [], {}


def hook(k, views, report, samples):
    obj = scene.objects[0]
    st = report.objects.get(obj) if report else None
    smp = samples.get(obj)
    if st is None or smp is None:
        return
    pos, quat = scene.ground_truth(obj)
    gt = runner.seed_local[obj] @ rotation_matrix(quat).T + pos
    tgt = runner.session.targets[obj]
    row = {"k": k, "occl": smp.occluded, "err": smp.error_m, "pred": bool(st.predicted),
           "mode": (st.lift_meta or {}).get("mode"), "reseeds": len(st.reseeds), "cams": {}, "meta": {q: (st.lift_meta or {}).get(q) for q in ("kf", "fit_n", "fit_rmse_m", "n_fused", "rejected_jump", "reanchored", "tentative", "kf_reset", "weak_support", "reanchor_skipped", "external_pose", "search", "fk_anchor", "fk_check")}}
    row["full"] = {q: v for q, v in (st.lift_meta or {}).items() if q not in ("per_cam",)}
    centre = smp.expected
    for cam, tr in st.cams.items():
        v = views.get(cam)
        c = {"status": tr.status}
        row["cams"][cam] = c
        idx = tgt.point_index.get(cam)
        if v is None or tr.points_2d is None or not tr.seeded:
            continue
        pts = np.asarray(tr.points_2d, float).reshape(-1, 2)
        idx = np.arange(len(pts)) if idx is None else np.asarray(idx, int).reshape(-1)
        visr = np.asarray(tr.visible if tr.visible is not None else np.ones(len(pts)), bool).reshape(-1)
        if len(idx) != len(pts) or len(visr) != len(pts):
            c["mismatch"] = True
            continue
        ok = (idx >= 0) & (idx < len(gt))
        gpx = np.full((len(pts), 2), np.nan)
        for j in np.flatnonzero(ok):
            px, _z = geometry.project_world_to_pixel(v, gt[idx[j]])
            if px is not None:
                gpx[j] = px
        vis = np.asarray(tr.visible, bool).reshape(-1) & ok
        e = np.linalg.norm(pts - gpx, axis=1)
        key = (cam, tgt.health[cam].last_reseed_frame)
        if key not in seed_ref:
            seed_ref[key] = (pts.copy(), gpx.copy())
        sp, sg = seed_ref[key]
        if sp.shape != pts.shape:
            seed_ref[key] = (pts.copy(), gpx.copy()); sp, sg = pts, gpx
        dg = np.linalg.norm(gpx - sg, axis=1); dt = np.linalg.norm(pts - sp, axis=1)
        stuck = vis & (dg > 10) & (dt < 0.3 * dg)
        c.update(n=len(pts), vis=int(vis.sum()),
                 err_px=float(np.nanmedian(e[vis])) if vis.any() else None,
                 stuck=int(stuck.sum()))
        if smp.occluded and cam in occ_cams:
            cpx, _z = geometry.project_world_to_pixel(v, centre)
            if cpx is not None:
                inside = np.linalg.norm(pts - cpx, axis=1) <= scene.occluder_radius
                c["latched"] = int((vis & inside).sum())
    if os.environ.get("KFDBG"):
        kst = runner.session.tracker3d._objects.get(obj)
        if kst is not None and kst.kf.initialized:
            kf = kst.kf
            prev = getattr(hook, "_prev", None)
            hook._prev = (gt.mean(0), k)
            tv = (gt.mean(0) - prev[0]) if prev else np.zeros(3)
            print(f"  KF k={k:3d} err={smp.error_m:.3f} lat={getattr(smp, 'latency_ms', 0):.0f}ms drop={getattr(smp, 'dropped', None)} coast={kf.coast:.2f} bend={kf._bend:.2f} "
                  f"v={np.round(kf.velocity * 1000, 1)} w={np.round(kf.angular_velocity, 3)} "
                  f"true_dc/frame={np.round(tv * 1000, 1)} p-c={np.round((kf.position - gt.mean(0)) * 1000, 0)}")
    rows.append(row)


runner = te.TrackingEvalRunner(scene, tracker_provider=provider, tracker3d=lift,
                               cameras=a.cameras, tracker_kwargs=kw_by.get(provider),
                               frame_hook=hook, **timing)
payload = runner.run()

occ_idx = [i for i, r in enumerate(rows) if r["occl"]]
for i, r in enumerate(rows):
    r["phase"] = ("pre" if not occ_idx or i < occ_idx[0] else
                  "occl" if r["occl"] else "post")
cams = sorted({c for r in rows for c in r["cams"]})
print(f"\n=== {a.scene} {provider}+{lift} cams={cams} occl_cams={sorted(occ_cams)} "
      f"rot={getattr(a, 'rot_step', 0)} fps={timing.get('camera_fps', 'lockstep')} ===")
for ph in ("pre", "occl", "post"):
    rs = [r for r in rows if r["phase"] == ph]
    if not rs:
        continue
    errs = [r["err"] if r["err"] is not None else 1.0 for r in rs]
    modes = collections.Counter(r["mode"] for r in rs)
    line = (f"{ph:5s} n={len(rs):3d} err med={np.median(errs):.3f} max={np.max(errs):.3f} "
            f"pred={np.mean([r['pred'] for r in rs]):.2f} reseeds={sum(r['reseeds'] for r in rs)} "
            f"modes={dict(modes)} reanch={sum(sum((r['meta'].get('reanchored') or {}).values()) for r in rs)} tent={sum(bool(r['meta'].get('tentative')) for r in rs)} reinit={sum(bool(r['meta'].get('kf_reset')) for r in rs)} refused={sum('refused' in str(r['meta'].get('kf')) for r in rs)} ext={dict(collections.Counter(r['meta'].get('external_pose') for r in rs if r['meta'].get('external_pose')))} search={sum(1 for r in rs if r['meta'].get('search'))}")
    print(line)
    for cam in cams:
        cs = [r["cams"].get(cam, {}) for r in rs]
        ep = [c["err_px"] for c in cs if c.get("err_px") is not None]
        stat = collections.Counter(c.get("status") for c in cs)
        print(f"      {cam:8s} vis={np.mean([c.get('vis', 0) / max(c.get('n', 1), 1) for c in cs]):.2f} "
              f"err_px med={np.median(ep) if ep else float('nan'):5.1f} "
              f"stuck={np.mean([c.get('stuck', 0) for c in cs]):4.1f} "
              f"latched={np.mean([c.get('latched', 0) for c in cs]):4.1f} status={dict(stat)}")
post = [r for r in rows if r["phase"] == "post"]
rec = next((j for j, r in enumerate(post) if r["err"] is not None and r["err"] < 0.02), None)
print(f"recover(<2cm) after {rec} frames post-occlusion" if post else "")
every = a.every or max(1, len(rows) // 16)
for r in rows[::every]:
    cs = " ".join(f"{c}:{r['cams'][c].get('status')}/{r['cams'][c].get('vis')}/"
                  f"{(r['cams'][c].get('err_px') or float('nan')):.0f}px"
                  f"/s{r['cams'][c].get('stuck', 0)}" for c in cams if c in r["cams"])
    print(f"  k={r['k']:3d} {r['phase']:4s} err={(r['err'] if r['err'] is not None else float('nan')):.3f} "
          f"{r['mode']} {cs}")
if a.trace:
    last = None
    for r in rows:
        m = r["meta"]
        if "init" in str(m.get("kf")) or "refused" in str(m.get("kf")):
            print(f"  T k={r['k']:3d} {r['phase']:4s} err={(r['err'] or float('nan')):.3f} {r['mode']} {m}")
        last = r["mode"]
if a.detail:
    lo, hi = a.detail
    for r in rows:
        if lo <= r["k"] < hi:
            cs = " ".join(f"{c}:{r['cams'][c].get('status')}/{r['cams'][c].get('vis')}/{(r['cams'][c].get('err_px') or float('nan')):.0f}px" for c in cams if c in r["cams"])
            print(f"  D k={r['k']:3d} {r['phase']:4s} err={(r['err'] if r['err'] is not None else float('nan')):.3f} {r['mode']} {cs} {r['full']}")
agg = payload["aggregate"]
print("aggregate median/p95/max", agg.get("median_l2_m"), agg.get("p95_l2_m"), agg.get("max_l2_m"),
      "occl", agg.get("err_during_occlusion_m"), "recov_f", agg.get("frames_to_recover"))





