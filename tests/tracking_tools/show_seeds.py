"""Draw affordance (5-pt) vs region seeds on the head (+wrist) view of the door / grasp scene."""
import os, sys, logging, argparse
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_OUT = os.path.join(_REPO, "outputs", "tracking_probes")
import numpy as np, cv2
ROOT = os.environ.get("TRK_ROOT", _REPO)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); os.chdir(ROOT)
import tracking_eval as te
from tracking import geometry
from tracking.seeding import region_seed
logging.basicConfig(level=logging.WARNING)
p = argparse.ArgumentParser(); te.add_eval_args(p)
a = p.parse_args(sys.argv[1:] or ["--scene", "door"])
scene, kw = te.build_scene(a)
scene.select_cameras(a.cameras or ("head", "wrist")); scene.reset()
obj = scene.objects[0]
base = scene.seed_points(obj)                      # affordance seeding (default)
view = scene.robot.capture_camera_view(scene.seed_camera(), scene.env)
aff_px = [geometry.project_world_to_pixel(view, w)[0] for w in base]
reg, px, info = region_seed(view, [aff_px[0]], n_points=a.seed_points)
print("region info:", info); print("dist-to-centroid cm", np.round(100*np.linalg.norm(reg-reg.mean(0),axis=1),1), "px", px.astype(int).tolist(), "aff world", np.round(base.mean(0),3), "reg", np.round(reg,3).tolist())
img = cv2.cvtColor(np.ascontiguousarray(np.asarray(view.rgb)[..., :3]), cv2.COLOR_RGB2BGR)
S = 4
img = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
for q in aff_px:
    cv2.circle(img, (int(q[0] * S), int(q[1] * S)), 4, (0, 0, 255), -1)
if px is not None:
    for k, q in enumerate(px):
        cv2.circle(img, (int(q[0] * S), int(q[1] * S)), 6, (0, 255, 0), 2)
c = np.mean(aff_px, axis=0).astype(int) * S
R = 60 * S
crop = img[max(0, c[1] - R):c[1] + R, max(0, c[0] - R):c[0] + R]
os.makedirs(_OUT, exist_ok=True)
out = os.path.join(_OUT, f"seeds_{a.scene}.png")
cv2.imwrite(out, crop); print(out)
