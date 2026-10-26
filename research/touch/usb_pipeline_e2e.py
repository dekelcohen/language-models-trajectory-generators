"""Scripted end-to-end check of the Genesis USB task through the real IPC app layer.

Drives ``env.run_simulation_environment`` (the same loop the agent talks to) with the commands an
LLM-written program would send: trajectories, open/close gripper, TOUCH_INSERT. The script plays
the role of perception: it reads the plug / port pose from the scene **once** and adds a fixed
perception error (default ~1.5 mm, like a camera-to-robot calibration error), so the insertion
starts misaligned and must be solved by touch. Success is judged from sim truth (eval only).

    <vlm_genesis>/python.exe research/touch/usb_pipeline_e2e.py [--no-touch] [--err 0.0012,0.0008]
"""

import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from multiprocessing import Pipe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=("scene", "all"))
ap.add_argument("--touch", action=argparse.BooleanOptionalAction, default=True,
                help="--no-touch: push straight in with position control (the baseline)")
ap.add_argument("--err", default="0.0012,0.0008", help="perception error of the port (dy,dz) m")
ap.add_argument("--out", default="outputs/touch/e2e")
ap.add_argument("--rot", type=float, default=None, help="gripper yaw for the grasp (auto if omitted)")
ap.add_argument("--grasp-off", default="0,0,0",
                help="grasp position error dx,dy,dz m (an imperfect pick, like an LLM-planned grasp)")
ap.add_argument("--step", type=float, default=0.002,
                help="max waypoint spacing m (1 sim step each); LLM trajectories use ~0.015 (20 points / 30 cm)")
ap.add_argument("--llm-z", action="store_true",
                help="send targets like the LLM does (pose z = object z, no depth-offset compensation)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from sim_envs.genesis import genesis_env  # noqa: E402

eargs = genesis_env.build_arg_parser().parse_args(["--task", "usb"])
eargs.sim = "genesis"
sim = genesis_env.make_adapter(eargs)
import env as envmod  # noqa: E402

parent, child = Pipe()
th = threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True)
t0 = time.time()
th.start()
eef0, _coords, state0, msg = parent.recv()
print(f"[e2e] env up in {time.time() - t0:.0f}s eef={np.round(eef0, 3).tolist()}", flush=True)
print(f"[e2e] sim_state (eval) {json.dumps(state0)}", flush=True)


def call(msg, reply=True):
    parent.send(msg)
    return parent.recv() if reply else None


def robot_state():
    return call([config.GET_ROBOT_STATE])


def truth():
    return call([config.GET_STATE])["sim_state"]


call([config.CAPTURE_IMAGES])
shutil.copy(config.rgb_image_head_path, os.path.join(args.out, "head.png"))
shutil.copy(config.rgb_image_wrist_path, os.path.join(args.out, "wrist.png"))
print(f"[e2e] head/wrist images -> {args.out}", flush=True)

# Which world axis do the fingers close along at trajectory rotation 0? (robot kinematics, not scene)


def finger_axis():
    body = [h for h in sim._bodies if "panda" in str(sim._bodies[h].source)][0]
    names = sim.list_link_names(body)
    l = np.asarray(sim.get_link_pose(body, names["panda_leftfinger"])[0])
    r = np.asarray(sim.get_link_pose(body, names["panda_rightfinger"])[0])
    d = l - r
    return d / np.linalg.norm(d), body, names


fa, ROBOT, NAMES = finger_axis()
print(f"[e2e] finger closing axis at start (rot 0): {np.round(fa, 3).tolist()}", flush=True)
if args.stage == "scene":
    os._exit(0)

# ---------------------------------------------------------------- "perception" (one look, with error)
plug = np.asarray(state0["plug_pos"])
port = np.asarray(state0["port_pos"])
ey, ez = (float(v) for v in args.err.split(","))
port_seen = port + np.array([0.0, ey, ez])
# Fingers must close across the housing width (world y): rotate if they close along x.
rot = args.rot if args.rot is not None else (0.0 if abs(fa[1]) > 0.9 else math.pi / 2)
print(f"[e2e] grasp rot {rot:.2f} rad; plug at {np.round(plug, 4).tolist()} port seen at "
      f"{np.round(port_seen, 4).tolist()} (err dy {ey*1000:+.1f} dz {ez*1000:+.1f} mm)", flush=True)

TIP_X = 0.0245            # housing centre -> tip end (known plug model, like a CAD part)
STANDOFF = 0.004


#: Pipeline convention: a top-down trajectory point is gripper_depth_offset above the fingertip centre
#: (the reported eef_pos). Targets below are fingertip-centre positions.
DZ = np.array([0.0, 0.0, 0.0 if args.llm_z else config.gripper_depth_offset_franka])


def go(points, desc):
    r = call([config.EXECUTE_TRAJECTORY, [list(map(float, np.asarray(p) + DZ)) + [rot] for p in points]])
    st = robot_state()
    print(f"[e2e] {desc}: {r[0].strip()[:60]} eef={np.round(st['eef_pos'], 4).tolist()} "
          f"width={st.get('gripper_width')} holding={st.get('gripper_holding')}", flush=True)
    return st


def go_exact(target, desc, tol=0.0005, iters=3):
    """Closed loop on the measured EE pose (proprioception): the sim's position control settles a few
    mm off (gravity sag); re-command target + error until within tol."""
    cmd = np.asarray(target, float)
    for _ in range(iters):
        st = go([cmd], desc)
        err = np.asarray(target) - np.asarray(st["eef_pos"])
        if np.linalg.norm(err) < tol:
            break
        cmd = cmd + err
    return st


def line(a, b, n, step=None):
    """Waypoints a->b; at most `step` m apart (1 sim step per point: 1 mm = 0.24 m/s, gentle on a held part)."""
    step = step or args.step
    n = n if args.step <= 0.002 else 2
    n = max(n, int(np.ceil(np.linalg.norm(np.asarray(b) - np.asarray(a)) / step)) + 1)
    return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s for s in np.linspace(0, 1, n)[1:]]


# Grasp at the balance point (3 mm toward the metal tip, from the plug model): sim finger contacts are
# near-points with little torsional friction, so an off-centre grasp lets the plug pitch in the fingers.
BALANCE = np.array([0.003, 0.0, 0.0])
seen_plug = plug.copy()
plug = plug + BALANCE + np.array([float(v) for v in args.grasp_off.split(",")])
above = plug + np.array([0, 0, 0.10])
call([config.OPEN_GRIPPER], reply=False)
st = robot_state()
go(line(np.asarray(st["eef_pos"]), above, 30), "above plug")
go(line(above, plug, 25), "down to plug")
st = go([plug], "at LLM grasp z") if args.llm_z else go_exact(plug, "centred on plug")
print(f"[e2e] (eval) plug before close: {truth().get('plug_pos')}", flush=True)
call([config.CLOSE_GRIPPER], reply=False)
st = go([st["eef_pos"]], "closed")
print(f"[e2e] (eval) plug after close: {truth().get('plug_pos')}", flush=True)

lift = plug + np.array([0, 0, 0.08])
go(line(plug, lift, 15), "lifted")
_t = truth()
print(f"[e2e] (eval) plug after lift: {_t.get('plug_pos')} axis {_t.get('plug_axis')}", flush=True)
# In-hand offset = plug seen in the hand (wrist-camera stand-in: true pose + ~1 mm error; closing the
# fingers can shift the plug a few mm, so the pre-grasp view is not enough) - measured fingertip centre.
st = robot_state()
in_hand = np.asarray(_t["plug_pos"]) + np.array([0.0, 0.0005, 0.0008]) - np.asarray(st["eef_pos"])
print(f"[e2e] in-hand plug offset {np.round(in_hand * 1000, 1).tolist()} mm", flush=True)

# Fingertip-centre target = seen port - axis * (tip length + standoff) - in-hand offset.
pre = port_seen - np.array([TIP_X + STANDOFF, 0.0, 0.0]) - in_hand
go(line(lift, pre + np.array([-0.03, 0, 0.05]), 40), "over the port side")
go(line(pre + np.array([-0.03, 0, 0.05]), pre + np.array([-0.03, 0, 0]), 20), "lowered")
go(line(pre + np.array([-0.03, 0, 0]), pre, 20), "at standoff")
st = go_exact(pre, "standoff corrected")
tr = truth()
print(f"[e2e] (eval) before insert: tip {tr.get('plug_tip_pos')} depth {tr.get('insertion_depth')} "
      f"lat_err {tr.get('lateral_error')} plug_axis {tr.get('plug_axis')}", flush=True)

t1 = time.time()
if args.touch:
    tool = np.array([TIP_X, 0.0, 0.0]) + in_hand
    reply = call([config.TOUCH_INSERT, {"axis": [1.0, 0.0, 0.0], "tool_offset": tool.tolist(), "max_s": 20.0}])
    print(f"[e2e] TOUCH_INSERT -> {reply[0].strip()} {json.dumps({k: v for k, v in reply[1].items() if k != 'events'})}",
          flush=True)
    for ev in reply[1].get("events", []):
        print("       ", ev, flush=True)
else:
    go(line(pre, pre + np.array([0.030, 0, 0]), 30), "naive push 30 mm")
print(f"[e2e] insert wall time {time.time() - t1:.0f}s", flush=True)
tr = truth()
ok = bool(tr.get("insertion_depth", 0) > 0.010 and tr.get("lateral_error", 1) < 0.002)
print(f"[e2e] RESULT {'touch' if args.touch else 'naive'}: depth {1000 * tr.get('insertion_depth', float('nan')):.1f}mm "
      f"lat_err {1000 * tr.get('lateral_error', float('nan')):.2f}mm -> {'INSERTED' if ok else 'not inserted'}", flush=True)
call([config.CAPTURE_IMAGES])
shutil.copy(config.rgb_image_head_path, os.path.join(args.out, f"head_after_{'touch' if args.touch else 'naive'}.png"))
call([config.OPEN_GRIPPER], reply=False)
os._exit(0 if ok else 2)
