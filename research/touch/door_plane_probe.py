"""Probe touch.door_plane on a live PyBullet door render (eval truth printed for comparison).

    python research/touch/door_plane_probe.py [--task door:hidden] [--noise 0.02]
"""
import argparse
import os
import sys
import threading
import time
from multiprocessing import Pipe

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="door")
ap.add_argument("--noise", type=float, default=0.0, help="handle position error (m), applied in +x,+y")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--anchor", default="lever", choices=("lever", "latch"))
ap.add_argument("--clear-pole", action="store_true", help="door:hidden: lay the pole on the floor (as after clearing)")
ap.add_argument("--door-deg", default="", help="comma list: set the hinge joint (deg) and measure the angle from depth")
args = ap.parse_args()

import config  # noqa: E402
from sim_envs.genesis import genesis_env  # noqa: E402
from sim_adapter import get_adapter  # noqa: E402
import env as envmod  # noqa: E402
from touch.door_plane import depth_to_world, estimate_door_plane  # noqa: E402
import utils  # noqa: E402

utils.args = argparse.Namespace(save_grasp_inputs=False)

eargs = genesis_env.build_arg_parser().parse_args(["--task", args.task])
eargs.sim = "pybullet"
sim = get_adapter("pybullet")
parent, child = Pipe()
threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True).start()
parent.recv()


def call(m):
    parent.send(m)
    return parent.recv()


S = call([config.GET_STATE])["sim_state"]
import pybullet as pb  # noqa: E402
DOOR, HINGE = S["door_id"], S["door_hinge_index"]
hinge = np.asarray(pb.getLinkState(DOOR, sim.get_joint_child_link(DOOR, HINGE))[4])
if args.clear_pole and S.get("pole_id") is not None:
    pb.resetBasePositionAndOrientation(S["pole_id"], [0.54, 0.15, 0.15], pb.getQuaternionFromEuler([1.5708, 0, 0]))
    for _ in range(50):
        pb.stepSimulation()
print(f"[probe] (eval) hinge {np.round(hinge[:2], 3).tolist()}  lever grasp ~[-0.277, 0.009, 0.672] "
      f"normal ~[-0.95, 0.312]")

t0 = time.time()
r = call([config.CAPTURE_IMAGES])
cam_pos, cam_q, cam_info = r[0], r[1], (r[5] if len(r) > 5 else None)
depth = np.load(os.path.splitext(config.depth_image_head_path)[0] + ".npy").astype(np.float32)
mpath = os.path.splitext(config.depth_image_head_path)[0] + "_robot_mask.npy"
mask = np.load(mpath) if os.path.exists(mpath) else None
P = depth_to_world(depth, cam_pos, cam_q, cam_info, stride=args.stride, robot_mask=mask)
t1 = time.time()
np.savez(os.path.join("outputs", "touch", f"door_points_{args.task.replace(':', '_')}.npz"), P=P, cam=np.asarray(cam_pos))
handle = np.array([-0.277 + args.noise, 0.009 + args.noise, 0.672]) if args.anchor == "lever" else \
    np.array([-0.098 + args.noise, -0.013 + args.noise, 0.672])
g = estimate_door_plane(P, handle, cam_pos[:2])
if g:
    t_ = np.asarray(g["tangent"][:2]); n_ = np.asarray(g["pull_dir"][:2])
    foot = np.asarray(g["hinge"]) - g["handle_to_hinge_m"] * (np.asarray(g["hinge"]) - np.asarray(g["free_edge"])) / g["width_m"]
    near = P[(np.abs((P[:, :2] - foot) @ n_ - 0) < 0.03) & (np.abs(P[:, 2] - handle[2]) < 0.25) & (P[:, 2] > 0.08)]
    s_ = (near[:, :2] - foot) @ t_
    hist, edges = np.histogram(s_, bins=np.arange(-0.6, 0.6, 0.04))
    print("[probe] face-line histogram (s along tangent, 4 cm bins):")
    for c_, e_ in zip(hist, edges):
        zs = near[(s_ >= e_) & (s_ < e_ + 0.04), 2]
        print(f"[probe]   s={e_:+.2f} n={c_:4d} z=[{zs.min() if len(zs) else 0:.2f},{zs.max() if len(zs) else 0:.2f}]")
print(f"[probe] {len(P)} pts in {t1 - t0:.1f}s, fit {time.time() - t1:.2f}s")
print(f"[probe] result {g}")
if g:
    err = np.linalg.norm(np.asarray(g["hinge"]) - hinge[:2])
    ang = np.degrees(np.arccos(abs(np.dot(g["pull_dir"][:2], [-0.95, 0.312]) / np.linalg.norm([-0.95, 0.312]))))
    print(f"[probe] hinge error {err:.3f} m, normal error {ang:.1f} deg, "
          f"side {'RIGHT' if err < 0.25 else 'WRONG'}")
if g and args.door_deg:
    from touch.door_plane import measure_door_angle
    for dd in [float(v) for v in args.door_deg.split(",")]:
        pb.resetJointState(DOOR, HINGE, np.radians(dd))
        r = call([config.CAPTURE_IMAGES])
        depth = np.load(os.path.splitext(config.depth_image_head_path)[0] + ".npy").astype(np.float32)
        mask = np.load(mpath) if os.path.exists(mpath) else None
        P2 = depth_to_world(depth, r[0], r[1], r[5] if len(r) > 5 else None, stride=args.stride, robot_mask=mask)
        np.savez(os.path.join("outputs", "touch", f"door_points_{args.task.replace(':', '_')}_deg{dd:.0f}.npz"),
                 P=P2, cam=np.asarray(cam_pos), latch=np.asarray(pb.getLinkState(DOOR, 2)[0]))  # latch: eval only
        g2 = estimate_door_plane(P2, handle, cam_pos[:2])
        print(f"[probe] joint {dd:+.0f} deg: measure_door_angle {measure_door_angle(P2, g, handle[2])}, "
              f"re-estimate pull_dir {g2 and g2['pull_dir']}", flush=True)
sys.stdout.flush()
os._exit(0)
