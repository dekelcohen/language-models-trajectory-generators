"""Wall-time profile of the touch torque loop (run_touch_insert) on the Genesis usb scene.

Wraps the sim-adapter calls the loop makes with timers, runs a free-space guarded move
(`TOUCH_INSERT` mode "contact", ~5 s sim) and prints ms per sim step per call and the real-time factor.

    conda run -n vlm_genesis python research/touch/profile_touch_loop.py [--sim-s 5]
"""
import argparse
import collections
import os
import sys
import threading
import time
from multiprocessing import Pipe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sim-s", type=float, default=5.0)
ap.add_argument("--cprofile", action="store_true")
args = ap.parse_args()

from sim_envs.genesis import genesis_env  # noqa: E402

eargs = genesis_env.build_arg_parser().parse_args(["--task", "usb"])
eargs.sim = "genesis"
sim = genesis_env.make_adapter(eargs)
import env as envmod  # noqa: E402

T = collections.defaultdict(float)
N = collections.defaultdict(int)
ACTIVE = {"on": False}


def wrap(name):
    fn = getattr(sim, name, None)
    if fn is None:
        return

    def timed(*a, **k):
        if not ACTIVE["on"]:
            return fn(*a, **k)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t
            N[name] += 1
    setattr(sim, name, timed)


for name in ("step", "get_joint_state", "get_joint_positions_velocities", "get_link_pose", "get_link_jacobian",
             "get_gravity_torques", "set_joint_torques"):
    wrap(name)

if args.cprofile:
    import cProfile
    import pstats
    import touch.sim_runner as sr
    _orig = sr.run_touch_insert

    def _profiled(*a, **k):
        prof = cProfile.Profile()
        try:
            return prof.runcall(_orig, *a, **k)
        finally:
            pstats.Stats(prof).sort_stats("tottime").print_stats(25)
    sr.run_touch_insert = _profiled

parent, child = Pipe()
threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True).start()
parent.recv()


def call(msg):
    parent.send(msg)
    return parent.recv()


speed = 0.01
ACTIVE["on"] = True
t0 = time.perf_counter()
r = call([config.TOUCH_INSERT, {"mode": "contact", "axis": [0.0, 0.0, 1.0], "tool_offset": [0, 0, 0],
                                "max_s": args.sim_s + 1.0, "grip_force": 0.0,
                                "guarded": {"max_travel": speed * args.sim_s, "f_contact": 50.0, "speed": speed}}])
wall = time.perf_counter() - t0
ACTIVE["on"] = False
res = r[1]
sim_s = float(res.get("duration_s") or 0.0)
steps = max(1, N["step"])
print(f"outcome={res.get('outcome')} sim={sim_s:.2f}s wall={wall:.2f}s RTF={sim_s / wall:.3f} steps={steps}")
total = 0.0
for name in sorted(T, key=T.get, reverse=True):
    total += T[name]
    print(f"  {name:32s} {1000 * T[name] / steps:7.3f} ms/step  ({N[name] / steps:5.2f} calls/step)")
print(f"  {'other (numpy, controller, IPC)':32s} {1000 * (wall - total) / steps:7.3f} ms/step")
sys.stdout.flush()
os._exit(0)
