"""Capture wrist-camera views of the usb scene's port for the offline pointing benchmark.

Genesis usb scene. Views are top-down gripper poses in front of the port face (like the
touch-guided-insertion skill's step 2: in front of the face, ~10 cm above, offset sideways), half of
them holding the plug (it can hide / be mistaken for the port). For each view it saves
``view_<n>.png`` + ``view_<n>_cam.npz`` (depth, wrist camera pose, cam_info - the same sidecar as
``API.locate_with_wrist_camera``) and the eval-only truth (``truth.json``: port position and its
projected pixel is computed by pointing_bench.py). The truth is only for scoring; it is never shown to a model.

    conda run -n vlm_genesis python research/touch/capture_wrist_views.py --out outputs/_bench/wrist_views
"""
import argparse
import json
import os
import shutil
import sys
import threading
from multiprocessing import Pipe

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join("outputs", "_bench", "wrist_views"))
ap.add_argument("--n", type=int, default=16, help="views (half empty-handed, half holding the plug)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from sim_envs.genesis import genesis_env  # noqa: E402

eargs = genesis_env.build_arg_parser().parse_args(["--task", "usb"])
eargs.sim = "genesis"
sim = genesis_env.make_adapter(eargs)
import env as envmod  # noqa: E402

parent, child = Pipe()
threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True).start()
parent.recv()


def call(msg, reply=True):
    parent.send(msg)
    return parent.recv() if reply else None


def truth():
    return call([config.GET_STATE])["sim_state"]


def go(p, rot):
    call([config.EXECUTE_TRAJECTORY, [list(map(float, p)) + [float(rot)]]])
    return call([config.GET_ROBOT_STATE])


def line(a, b, n):
    return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s for s in np.linspace(0, 1, n)[1:]]


def go_line(b, rot, n=25):
    a = np.asarray(call([config.GET_ROBOT_STATE])["eef_pos"]) + DZ
    for p in line(a, b, n):
        st = go(p, rot)
    return st


DZ = np.array([0.0, 0.0, config.gripper_depth_offset_franka])
s0 = truth()
port, plug = np.asarray(s0["port_pos"], float), np.asarray(s0["plug_pos"], float)
rng = np.random.default_rng(args.seed)


def capture(n, holding, rot):
    payload = call([config.CAPTURE_IMAGES])
    cam_pos, cam_q, cam_info = payload[2], payload[3], payload[5]
    depth = np.load(os.path.splitext(config.depth_image_wrist_path)[0] + ".npy").astype(np.float32)

    stem = os.path.join(args.out, f"view_{n:02d}")
    shutil.copy(config.rgb_image_wrist_path, stem + ".png")
    np.savez_compressed(stem + "_cam.npz", depth=depth, cam_pos=np.asarray(cam_pos, float),
                        cam_quat=np.asarray(cam_q, float), cam_info=json.dumps(cam_info, default=str),
                        description="USB-A port opening on the laptop side")

    eef = call([config.GET_ROBOT_STATE])
    rec = {"view": os.path.basename(stem), "holding": bool(holding), "rot": round(float(rot), 3),
           "eef_pos": [round(float(v), 4) for v in eef["eef_pos"]],
           "gripper_holding": bool(eef.get("gripper_holding")),
           "truth_port_pos": port.round(4).tolist()}
    print(f"[capture] {rec}", flush=True)
    return rec


def view_poses(k):
    out = []
    for _ in range(k):
        d = rng.uniform(0.06, 0.12)             # in front of the face (port axis is +x: front is -x)
        s = rng.choice([-1, 1]) * rng.uniform(0.04, 0.11)
        hgt = rng.uniform(0.07, 0.12)
        out.append(port + np.array([-d, s, hgt]))
    return out


_body = [h for h in sim._bodies if "panda" in str(sim._bodies[h].source)][0]
_names = sim.list_link_names(_body)
_fa = (np.asarray(sim.get_link_pose(_body, _names["panda_leftfinger"])[0])
       - np.asarray(sim.get_link_pose(_body, _names["panda_rightfinger"])[0]))
rot_g = 0.0 if abs(_fa[1]) / np.linalg.norm(_fa) > 0.9 else np.pi / 2   # fingers close across the housing (y)
records = []
half = args.n // 2
call([config.OPEN_GRIPPER], reply=False)
for i, p in enumerate(view_poses(half)):
    rot = rng.choice([0.0, 0.0, np.pi / 2, rng.uniform(-0.5, 0.5)])
    go_line(p, rot)
    records.append(capture(i, False, rot))

# Grasp the plug (top-down at its centre; the fingers close across the housing width).
above = plug + np.array([0.003, 0, 0.10])
go_line(above, rot_g)
go_line(plug + np.array([0.003, 0, 0.0]), rot_g, 20)
call([config.CLOSE_GRIPPER], reply=False)
go_line(above, rot_g, 15)
for i, p in enumerate(view_poses(args.n - half)):
    go_line(p, rot_g)
    records.append(capture(half + i, True, rot_g))

with open(os.path.join(args.out, "truth.json"), "w") as f:
    json.dump({"note": "EVAL ONLY - never give to a model", "views": records}, f, indent=1)
print(f"[capture] {len(records)} views -> {args.out}",
      flush=True)
sys.stdout.flush()
os._exit(0)
