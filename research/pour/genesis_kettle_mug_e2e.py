"""Scripted end-to-end check of the Genesis kettle -> mug pour (SPH water) through the real IPC app layer.

Drives ``env.run_simulation_environment`` (the loop the agent talks to) with the commands an LLM program
would send: open/close gripper and trajectories, the pour built with ``pour_pose`` exactly as the
``pour-from-held-container`` skill does. The script plays perception: it reads the kettle / mug pose once
from sim truth (eval only). Prints step time and water counts; saves head-camera frames.

    <vlm_genesis>/python.exe research/pour/genesis_kettle_mug_e2e.py [--stage scene] [--tilt 0.8]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("scene", "all"))
    ap.add_argument("--out", default="outputs/pour/genesis_e2e")
    ap.add_argument("--tilt", type=float, default=0.8)
    ap.add_argument("--past", type=float, default=0.03, help="hand this far past the mug centre along pour_yaw")
    ap.add_argument("--height", type=float, default=0.30, help="hand height above the mug top")
    ap.add_argument("--hold", type=int, default=60, help="hold-the-pour waypoints")
    ap.add_argument("--max-speed", type=float, default=None, help="hand speed cap m/s (common_utils.limit_speed)")
    ap.add_argument("--max-rot-speed", type=float, default=None, help="hand rotation speed cap rad/s")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from common_utils import pour_pose
    from sim_envs.genesis import genesis_env
    from sim_envs.genesis import kettle_mug as km

    eargs = genesis_env.build_arg_parser().parse_args(["--task", "kettle_mug"])
    eargs.sim = "genesis"
    sim = genesis_env.make_adapter(eargs)
    import env as envmod

    parent, child = Pipe()
    th = threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True)
    t0 = time.time()
    th.start()
    eef0, _coords, state0, _msg = parent.recv()
    print(f"[e2e] env up in {time.time() - t0:.0f}s eef={np.round(eef0, 3).tolist()}", flush=True)
    print(f"[e2e] sim_state (eval) {json.dumps(state0)}", flush=True)

    def call(msg, reply=True):
        parent.send(msg)
        return parent.recv() if reply else None

    def truth():
        return call([config.GET_STATE])["sim_state"]

    shot = [0]

    def snap(tag):
        call([config.CAPTURE_IMAGES])
        shot[0] += 1
        shutil.copy(config.rgb_image_head_path, os.path.join(args.out, f"{shot[0]:02d}_head_{tag}.png"))
        shutil.copy(config.rgb_image_wrist_path, os.path.join(args.out, f"{shot[0]:02d}_wrist_{tag}.png"))

    snap("start")
    if args.stage == "scene":
        os._exit(0)

    DZ = np.array([0.0, 0.0, config.gripper_depth_offset_franka])
    rot = 0.0                            # fingers close along world x, across the bar handle (along y)
    pour_yaw = rot + math.pi / 2         # spout (+y) toward the mug

    last = [None]

    def go(points, desc, limit=True):
        t = time.time()
        pts = [list(map(float, p)) for p in points]
        if limit and (args.max_speed or args.max_rot_speed):
            from common_utils import limit_speed
            # prepend the previous commanded pose so the first segment is capped too
            if last[0] is not None:
                pts = limit_speed([last[0]] + pts, args.max_speed, args.max_rot_speed)[1:]
            else:
                pts = limit_speed(pts, args.max_speed, args.max_rot_speed)
        last[0] = pts[-1]
        r = call([config.EXECUTE_TRAJECTORY, pts])
        st = call([config.GET_ROBOT_STATE])
        tr = truth()
        w = tr.get("water")
        print(f"[e2e] {desc}: {r[0].strip()[:50]} {time.time() - t:.0f}s eef={np.round(st['eef_pos'], 3).tolist()} "
              f"width={st.get('gripper_width')} holding={st.get('gripper_holding')} water={w} kettle_tilt={tr.get('kettle_tilt_deg')} kettle_pos={tr.get('kettle_pos')} mug_pos={tr.get('mug_pos')}", flush=True)
        return st

    def line(a, b, n=20):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return [a + (b - a) * s for s in np.linspace(0, 1, n)[1:]]

    def top(p):
        return list(np.asarray(p) + DZ) + [rot]

    kettle = np.asarray(state0["kettle_pos"])
    mug = np.asarray(state0["mug_pos"])
    handle = kettle + np.array([0, 0, km.HANDLE_Z])
    st = call([config.GET_ROBOT_STATE])
    call([config.OPEN_GRIPPER], reply=False)
    above = handle + np.array([0, 0, 0.10])
    go([top(p) for p in line(st["eef_pos"], above, 30)], "above handle")
    go([top(p) for p in line(above, handle, 25)], "down to handle")
    call([config.CLOSE_GRIPPER], reply=False)
    go([top(handle)], "closed")
    snap("grasped")

    top_z = mug[2] + km.MUG_H
    d = np.array([math.cos(pour_yaw), math.sin(pour_yaw), 0])
    pz = top_z + args.height
    px, py = mug[0] + args.past * d[0], mug[1] + args.past * d[1]
    st = call([config.GET_ROBOT_STATE])
    hand = np.asarray(st["eef_pos"])
    up = np.array([hand[0], hand[1], max(hand[2], pz)])
    carry = np.array([px, py, pz])
    go([top(p) for p in line(hand, up, 20)], "rise")
    go([top(p) for p in line(up, carry, 20)], "carry")
    snap("carried")
    c_dz = carry + DZ
    poured = pour_pose(c_dz[0], c_dz[1], c_dz[2], rot, args.tilt, pour_yaw)
    tilt_pts = [pour_pose(c_dz[0], c_dz[1], c_dz[2], rot, args.tilt * s, pour_yaw) for s in np.linspace(0, 1, 30)[1:]]
    go(tilt_pts, "tilt")
    snap("tilted")
    go([poured] * args.hold, "hold")
    snap("held")
    go(list(reversed(tilt_pts[:-1])) + [top(carry)], "tilt back")
    for _ in range(3):
        go([top(carry)] * 20, "settle")
    snap("end")
    tr = truth()
    print(f"[e2e] (eval) final water={tr.get('water')} task_error={tr.get('task_error')} "
          f"success={tr.get('success')}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
