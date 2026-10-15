"""Pre-occlusion slide probe (D4): do tracked handle points slide toward the latch axle?

Per frame (head camera): tracked 2D points vs the GT projection of their seed point, and
both vs the projected latch pivot (handle link origin). Also splits the fused 3D error into
the head camera's image-plane and depth (along-ray) components.
"""
import os, sys, logging, argparse
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_OUT = os.path.join(_REPO, "outputs", "tracking_probes")
import numpy as np
ROOT = os.environ.get("TRK_ROOT", _REPO)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); os.chdir(ROOT)
import tracking_eval as te
from tracking import geometry
from sim_adapter.transforms import rotation_matrix

logging.basicConfig(level=logging.WARNING)
p = argparse.ArgumentParser(); te.add_eval_args(p)
p.add_argument("--cfg", nargs="*", default=[])
extra = sys.argv[1:]
a = p.parse_args(["--scene", "door", "--occlusion", "15", "25", "--occluder-cam", "head",
                  "--tracker3d", "pose_tracker", *extra])
import ast, config
for kv in a.cfg:
    k, v = kv.split("=", 1)
    try: v = ast.literal_eval(v)
    except Exception: pass
    setattr(config, k, v)
scene, kw = te.build_scene(a)
rows = []


def hook(k, views, report, samples):
    obj = scene.objects[0]
    st = report.objects.get(obj) if report else None
    smp = samples.get(obj)
    v = views.get("head") if views else None
    if st is None or smp is None or v is None or smp.occluded or k > 44:
        return
    pos, quat = scene.ground_truth(obj)
    R = rotation_matrix(quat)
    gt = runner.seed_local[obj] @ R.T + pos
    tgt = runner.session.targets[obj]
    tr = st.cams.get("head")
    piv, _ = geometry.project_world_to_pixel(v, pos)
    if tr is None or tr.points_2d is None or piv is None:
        return
    pts = np.asarray(tr.points_2d, float).reshape(-1, 2)
    idx = np.asarray(tgt.point_index["head"], int).reshape(-1)
    d_tr, d_gt, e = [], [], []
    for j, i in enumerate(idx):
        if i < 0 or i >= len(gt):
            continue
        g, _ = geometry.project_world_to_pixel(v, gt[i])
        d_tr.append(np.linalg.norm(pts[j] - piv)); d_gt.append(np.linalg.norm(g - piv))
        e.append(pts[j] - g)
    e = np.asarray(e)
    # 3D error split along the head camera ray
    cam = np.asarray(v.position, float)
    if st.world_point is not None and smp.expected is not None:
        err = np.asarray(st.world_point) - np.asarray(smp.expected)
        ray = np.asarray(smp.expected) - cam; ray /= np.linalg.norm(ray)
        along = float(err @ ray); lateral = float(np.linalg.norm(err - along * ray))
    else:
        along = lateral = float("nan")
    lat = (st.lift_meta or {}).get("rotation_observable")
    rows.append((k, smp.error_m, np.mean(d_tr), np.mean(d_gt), np.linalg.norm(e.mean(0)),
                 e.mean(0), along, lateral, lat))


runner = te.TrackingEvalRunner(scene, tracker_provider=a.provider[0], tracker3d="pose_tracker",
                               cameras=a.cameras, tracker_kwargs=kw.get(a.provider[0]),
                               frame_hook=hook, **te.timing_kwargs(a))
runner.run()
print(" k   err3d  |pts-pivot|px tracked/gt   2D mean err px (dx,dy)   3D err along-ray / lateral mm  rotObs")
for r in rows[::3]:
    k, err, dtr, dgt, en, ev, al, la, ro = r
    print(f"{k:3d} {err:6.3f}   {dtr:5.1f} / {dgt:5.1f}        {en:4.1f} ({ev[0]:+.1f},{ev[1]:+.1f})"
          f"        {1000*al:+6.1f} / {1000*la:5.1f}   {ro}")
