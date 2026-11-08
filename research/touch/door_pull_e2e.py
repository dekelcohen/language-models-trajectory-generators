"""Scripted end-to-end check of the door task in Genesis through the real IPC app layer.

Plays the LLM-written program: approach the lever handle top-down, close, then open the door either
open-loop (``--no-touch``: a straight / guessed-arc pull, like today's LLM trajectories) or with the
touch-guided ``open_with_touch`` policy (``touch/recovery.py``) through the same API object the agent
uses. "Perception" reads the handle pose from the scene ONCE and adds an error (like a detection);
the door angle is read for **evaluation only**.

    <vlm_genesis>/python.exe research/touch/door_pull_e2e.py [--no-touch] [--err 0.01,0.0,0.0] [--dir normal]
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from multiprocessing import Pipe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

import config  # noqa: E402
from common_utils import side_grasp_pose, top_down_euler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=("scene", "probe", "all"))
ap.add_argument("--touch", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--err", default="0,0,0", help="handle perception error dx,dy,dz m")
ap.add_argument("--dir", default="normal", choices=("normal", "tangent", "skew"),
                help="initial pull direction: door face normal (an LLM's straight pull), the true tangent, "
                     "or normal rotated 30 deg away from the hinge")
ap.add_argument("--pull", type=float, default=0.35, help="open-loop pull length m (--no-touch)")
ap.add_argument("--grip-x", type=float, default=0.10, help="grasp point along the lever from its root, m")
ap.add_argument("--approach", default="side", choices=("side", "top"))
ap.add_argument("--max-regrasps", type=int, default=2)
ap.add_argument("--k-yaw", type=float, default=3.0, help="touch pull wrist yaw stiffness Nm/rad")
ap.add_argument("--probe-series", action="store_true")
ap.add_argument("--depth", type=float, default=0.01, help="fingertip centre past the lever centre, m")
ap.add_argument("--grip", type=float, default=30.0, help="touch grip force per finger, N")
ap.add_argument("--no-cap", action="store_true", help="A/B: disable the actuator force caps")
ap.add_argument("--out", default="outputs/touch/door_e2e")
ap.add_argument("--sim", default="genesis", choices=("genesis", "pybullet"),
                help="pybullet: open-loop only (touch control is Genesis-only)")
ap.add_argument("--press", type=float, default=0.0, help="open-loop: press the lever down this much (m) first")
ap.add_argument("--arc", type=float, default=0.0, help="open-loop: pull along the (eval) hinge arc this many deg")
ap.add_argument("--arc-yaw", action="store_true", help="--arc: turn the wrist with the door (yaw += swept)")
ap.add_argument("--arc-wrong", action="store_true", help="--arc: swing the WRONG way (wrong hinge side) -> grasp-slip guard test")
ap.add_argument("--hinge-err", type=float, default=0.0, help="--arc: hinge estimate pushed this far (m) further from the handle")
ap.add_argument("--debug", action="store_true", help="trace each regrasp phase (eval door angle + image)")
ap.add_argument("--gp-kw", default="", help="--guarded: extra guarded_pull kwargs k=json,k=json, e.g. follow_yaw=false,step=0.015")
ap.add_argument("--hinge-shift", default="", help="--guarded: add dx,dy (m) to the hinge hint (a bad LLM guess)")
ap.add_argument("--vision", action="store_true",
                help="--guarded: pull normal + hinge from touch.door_plane on the head depth (rosette anchor)")
ap.add_argument("--guarded", default="off", choices=("off", "right", "wrong", "line"),
                help="open with helpers.guarded_pull: hinge hint on the (eval) right side, mirrored to the "
                     "wrong side, or none (straight pull along --dir, hinge fitted from the hand path)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from sim_envs.genesis import genesis_env  # noqa: E402

eargs = genesis_env.build_arg_parser().parse_args(["--task", "door"])
eargs.sim = args.sim
if args.sim == "pybullet":
    from sim_adapter import get_adapter
    sim = get_adapter("pybullet")
    args.touch = False
else:
    sim = genesis_env.make_adapter(eargs)
if args.no_cap:                                      # A/B: pre-5fcde9d behaviour (1200 Nm force ranges)
    class _NoCaps(dict):
        def setdefault(self, k, d=None):
            return {}
    sim._force_caps = _NoCaps()
import env as envmod  # noqa: E402

parent, child = Pipe()
th = threading.Thread(target=envmod.run_simulation_environment, args=(eargs, child, None, sim), daemon=True)
t0 = time.time()
th.start()
eef0, _coords, state0, msg = parent.recv()
print(f"[door] env up in {time.time() - t0:.0f}s eef={np.round(eef0, 3).tolist()}", flush=True)


def call(m, reply=True):
    parent.send(m)
    return parent.recv() if reply else None


def truth():
    return call([config.GET_STATE])["sim_state"]


S = truth()
DOOR, HINGE = S["door_id"], S["door_hinge_index"]


def door_angle():                                    # eval only
    return float(sim.get_joint_state(DOOR, HINGE).position)


def R_of(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame(link):                                     # URDF link origin (eval/scripted perception only)
    if args.sim == "pybullet":
        import pybullet as pb
        ls = pb.getLinkState(DOOR, int(link), computeForwardKinematics=True)
        return np.asarray(ls[4], float), R_of(ls[5])
    gl = sim._body(DOOR).links[int(link)]
    p = np.asarray(gl.get_pos().cpu()).ravel()
    w, x, y, z = (float(v) for v in np.asarray(gl.get_quat().cpu()).ravel())
    return p, R_of([x, y, z, w])


latch_link = S["door_handle_latch"]
lp, RL = frame(latch_link)
lever_dir = RL[:, 0]                                 # lever cylinders lie along the latch-frame x
base = np.asarray(config.base_start_position_franka, float)
levers = [lp + RL @ np.array([0.1 + (args.grip_x - 0.1), -0.15, 0.0]), lp + RL @ np.array([-0.1, 0.10, 0.0])]
lever_mid = min(levers, key=lambda p: np.linalg.norm((p - base)[:2]))
hinge, _ = frame(sim.get_joint_child_link(DOOR, HINGE))
radial = lever_mid - hinge
radial[2] = 0
tangent = np.cross([0, 0, 1.0], radial)
tangent /= np.linalg.norm(tangent)
if np.dot(tangent, (lever_mid - lp)) < 0:            # open toward the robot side (the lever's side)
    tangent = -tangent
normal = tangent.copy()                              # at angle 0 the tangent IS the face normal
print(f"[door] (eval) latch {np.round(lp, 3).tolist()} levers {[np.round(p, 3).tolist() for p in levers]}",
      flush=True)
print(f"[door] (eval) hinge {np.round(hinge, 3).tolist()} lever grasp {np.round(lever_mid, 3).tolist()} "
      f"lever_dir {np.round(lever_dir, 3).tolist()} radius {np.linalg.norm(radial):.3f} "
      f"tangent {np.round(tangent, 3).tolist()} angle {door_angle():.3f}", flush=True)

cap = call([config.CAPTURE_IMAGES])
shutil.copy(config.rgb_image_head_path, os.path.join(args.out, "head.png"))
shutil.copy(config.rgb_image_wrist_path, os.path.join(args.out, "wrist.png"))
VISION = None
if args.vision:                                      # pixels + depth only; the anchor is the rosette (+ --err)
    import utils as _u
    from touch.door_plane import depth_to_world, estimate_door_plane
    _u.args = _u.args or argparse.Namespace(save_grasp_inputs=False)
    _dep = np.load(os.path.splitext(config.depth_image_head_path)[0] + ".npy").astype(np.float32)
    _mp = os.path.splitext(config.depth_image_head_path)[0] + "_robot_mask.npy"
    _pts = depth_to_world(_dep, cap[0], cap[1], cap[5] if len(cap) > 5 else None,
                          robot_mask=np.load(_mp) if os.path.exists(_mp) else None)
    _anchor = lp + np.array([float(v) for v in args.err.split(",")])
    VISION = estimate_door_plane(_pts, _anchor, np.asarray(cap[0])[:2])
    print(f"[door] vision door plane: {VISION}", flush=True)
if args.stage == "scene":
    os._exit(0)
if args.stage == "probe":                            # eval: how much hinge torque does the door need?
    ent = sim._body(DOOR).entity
    hd = sim._dofs(DOOR, HINGE)
    st = call([config.GET_ROBOT_STATE])
    hold = [list(np.asarray(st["eef_pos"]) + [0, 0, config.gripper_depth_offset_franka]) + [0.0]]
    for tau in ((8, 15) if args.probe_series else (2, 5, 10, 20, 40, 80)):
        def f(t=tau):
            ent.set_dofs_force_range(np.array([-1e3]), np.array([1e3]), dofs_idx_local=hd)
            ent.control_dofs_force(np.array([float(t)]), dofs_idx_local=hd)
        sim._defer("probe", f)
        for _ in range(8 if args.probe_series else 1):
            call([config.EXECUTE_TRAJECTORY, hold])
            print(f"[door] probe tau={tau} Nm -> angle {math.degrees(door_angle()):.2f} deg", flush=True)
    os._exit(0)

# ------------------------------------------------------------- agent side: the same API the LLM uses
from types import SimpleNamespace  # noqa: E402


class _Log:
    def info(self, m):
        print("   ", str(m).replace("\x1b", "")[:400], flush=True)

    warning = error = debug = info


api = SimpleNamespace(main_connection=parent, logger=_Log(), args=SimpleNamespace(touch=args.touch))
seen = lever_mid + np.array([float(v) for v in args.err.split(",")])
DZ = config.gripper_depth_offset_franka
# rotation that closes the fingers across the bar: the closing axis at rot 0 is the hand's x
rot = math.atan2(lever_dir[1], lever_dir[0]) + math.pi / 2
rot = (rot + math.pi / 2) % math.pi - math.pi / 2
d0 = {"normal": normal, "tangent": tangent}.get(args.dir)
if d0 is None:                                       # skew: 30 deg away from the hinge side
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    d0 = np.array([c * normal[0] - s * normal[1], s * normal[0] + c * normal[1], 0.0])
    if np.dot(d0, radial) < 0:
        d0 = np.array([c * normal[0] + s * normal[1], -s * normal[0] + c * normal[1], 0.0])


SLIPS = []


def _slip(r, desc):
    if len(r) > 2 and isinstance(r[2], dict) and r[2].get("grasp_slip"):
        print(f"[door] GRASP SLIP at {desc}: {json.dumps(r[2]['grasp_slip'])}", flush=True)
        SLIPS.append(desc)
    return bool(SLIPS)


def exe(points, desc, settle=6, tip=None):
    tip = np.asarray(points[-1][:3]) - [0, 0, DZ] if tip is None else np.asarray(tip)
    r = call([config.EXECUTE_TRAJECTORY, [list(map(float, p)) for p in points]])
    for _ in range(0 if _slip(r, desc) else settle):  # hold the last set-point until the arm settles
        st = call([config.GET_ROBOT_STATE])
        if np.linalg.norm(tip - st["eef_pos"]) < 0.004:
            break
        if _slip(call([config.EXECUTE_TRAJECTORY, [list(map(float, points[-1]))]]), desc + " (settle)"):
            break
    st = call([config.GET_ROBOT_STATE])
    print(f"[door] {desc}: {r[0].strip()[:50]} eef={np.round(st['eef_pos'], 3).tolist()} "
          f"w={st.get('gripper_width')} hold={st.get('gripper_holding')} angle={door_angle():.3f}", flush=True)
    return st


def lin(a, b, n=None, step=0.01):
    n = max(n or 2, int(np.ceil(np.linalg.norm(np.subtract(b[:3], a[:3])) / step)) + 1)
    return [list(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s) for s in np.linspace(0, 1, n)[1:]]


def finger_axis():
    if args.sim == "pybullet":
        return "n/a"
    body = [h for h in sim._bodies if "panda" in str(sim._bodies[h].source)][0]
    names = sim.list_link_names(body)
    d = np.asarray(sim.get_link_pose(body, names["panda_leftfinger"])[0]) - \
        np.asarray(sim.get_link_pose(body, names["panda_rightfinger"])[0])
    return np.round(d / np.linalg.norm(d), 3).tolist()


call([config.OPEN_GRIPPER], reply=False)
st = call([config.GET_ROBOT_STATE])
cur = list(np.asarray(st["eef_pos"]) + [0, 0, DZ]) + [rot]
if args.approach == "top":
    above = list(seen + [0, 0, DZ + 0.12]) + [rot]
    grasp = list(seen + [0, 0, DZ]) + [rot]
    exe(lin(cur, above), "above handle")
    print(f"[door] finger closing axis {finger_axis()} (lever {np.round(lever_dir, 3).tolist()}) rot={rot:.2f}",
          flush=True)
    exe(lin(above, grasp, step=0.005), "down to handle")
else:
    # Side approach (skill 2b, rotation pi/2 = jaws close vertically across the horizontal lever):
    # the lever sits only ~0.10 m off the door face, so a top-down hand (~0.2 m wide along its
    # closing axis) hits the face. Commanded pose = fingertip target - DZ * approach axis.
    a = -normal                                       # toward the door
    yaw = math.atan2(a[1], a[0])
    tip = seen + a * args.depth                             # fingertips just past the lever's centre line

    def side(p):
        return side_grasp_pose(*(np.asarray(p) - a * DZ), math.pi / 2, yaw)

    stand = side(tip - a * 0.20)
    grasp = side(tip)
    cur6 = cur[:3] + list(top_down_euler(rot))
    exe(lin(cur6, stand, step=0.02), "stand-off (side)", tip=tip - a * 0.20)
    print(f"[door] finger closing axis {finger_axis()} (want vertical)", flush=True)
    for k, pt in enumerate(lin(stand, grasp, step=0.02)):
        exe([pt], f"advance {k}", tip=tip - a * 0.20 * (1 - (k + 1) / 10))
call([config.CLOSE_GRIPPER], reply=False)
st = exe([grasp], "closed", tip=seen)

t1 = time.time()
if args.guarded != "off":
    from helpers.guarded_pull import guarded_pull

    class _GuardApi:
        """The agent API surface guarded_pull uses, over the raw IPC pipe."""
        main_connection, logger = parent, _Log()

        def get_robot_state(self):
            return call([config.GET_ROBOT_STATE])

        def guard_motion(self, mode="hold", **kw):
            return call([config.SET_MOTION_GUARD, {"mode": mode, **kw}])[1]

        def open_gripper(self):
            call([config.OPEN_GRIPPER], reply=False)

        def close_gripper(self):
            call([config.CLOSE_GRIPPER], reply=False)

        def execute_trajectory(self, traj):
            pts = [list(map(float, p)) for p in traj.points]
            r = call([config.EXECUTE_TRAJECTORY, pts])
            g = r[2].get("grasp_slip") if len(r) > 2 and isinstance(r[2], dict) else None
            tip = np.asarray(pts[-1][:3]) - [0, 0, DZ]
            for _ in range(0 if g else 3):
                if np.linalg.norm(tip - call([config.GET_ROBOT_STATE])["eef_pos"]) < 0.004:
                    break
                r = call([config.EXECUTE_TRAJECTORY, [pts[-1]]])
                g = r[2].get("grasp_slip") if len(r) > 2 and isinstance(r[2], dict) else None
                if g:
                    break
            if g:
                print(f"[door] guard trip {g['check']} at {g['slip_pos']} door={math.degrees(door_angle()):.1f}deg "
                      f"reacq={g.get('reacquired')}", flush=True)
            out = {"eef_pos": call([config.GET_ROBOT_STATE])["eef_pos"]}
            if os.environ.get("DOOR_DIAG"):            # eval-only per-step trace: lever vs hand
                lp_, RL_ = frame(latch_link)
                ld = RL_[:, 0]
                grip_pt = lp_ + RL_ @ np.array([0.1 + (args.grip_x - 0.1), -0.15, 0.0])
                st_ = call([config.GET_ROBOT_STATE])
                k = getattr(self, "k", 0)
                self.k = k + 1
                lever_yaw = math.degrees(math.atan2(ld[1], ld[0]))
                print(f"[diag] k={k} door={math.degrees(door_angle()):.1f} cmd_rot={math.degrees(pts[-1][3]):.1f} "
                      f"meas_rot={None if st_.get('rotation') is None else round(math.degrees(st_['rotation']), 1)} "
                      f"lever_yaw={lever_yaw:.1f} w={st_.get('gripper_width')} "
                      f"tip-lever={np.round((np.asarray(st_['eef_pos']) - grip_pt) * 1000, 0).tolist()}mm", flush=True)
                if args.sim == "pybullet":
                    import pybullet as pb
                    cs = {}
                    for b in range(pb.getNumBodies()):
                        if b == DOOR:
                            continue
                        for c in pb.getContactPoints(bodyA=b, bodyB=DOOR):
                            nm = pb.getJointInfo(b, c[3])[12].decode() if c[3] >= 0 else "base"
                            dl = pb.getJointInfo(DOOR, c[4])[12].decode() if c[4] >= 0 else "base"
                            key = f"{nm}->{dl}"
                            cs[key] = round(cs.get(key, 0) + c[9], 1)
                    print(f"[diag]    contacts {cs}", flush=True)
                if k % 2 == 0:
                    call([config.CAPTURE_IMAGES])
                    shutil.copy(config.rgb_image_head_path, os.path.join(args.out, f"diag_head_{k:02d}.png"))
                    shutil.copy(config.rgb_image_wrist_path, os.path.join(args.out, f"diag_wrist_{k:02d}.png"))
            if g:
                out["guard"] = g
            return out

    h = None
    if args.press:
        pressed = list(np.asarray(grasp[:3]) - [0, 0, args.press]) + list(grasp[3:])
        exe(lin(grasp, pressed, step=0.005), "press", settle=3)
    if args.guarded != "line":
        h = hinge[:2] if args.guarded == "right" else 2 * (seen[:2] + np.dot(hinge[:2] - seen[:2], normal[:2]) * normal[:2]) - hinge[:2]
        if args.hinge_shift:
            h = np.asarray(h, float) + np.asarray([float(v) for v in args.hinge_shift.split(",")])
    if VISION:
        h, d0 = np.asarray(VISION["hinge"], float), np.asarray(VISION["pull_dir"], float)
    print(f"[door] guarded_pull hinge hint {None if h is None else np.round(h, 3).tolist()} dir {np.round(d0, 3).tolist()}", flush=True)
    res = guarded_pull(_GuardApi(), direction=list(d0), hinge=None if h is None else list(h), **{k: json.loads(v) for k, v in (kv.split("=", 1) for kv in re.split(r",(?=\w+=)", args.gp_kw) if kv)})
    print(f"[door] guarded_pull -> {json.dumps({k: v for k, v in res.items() if k != 'events'})}", flush=True)
elif args.touch:
    from touch import recovery
    _orig_exec = recovery._execute

    def _dbg_exec(api_, points, desc):  # harness debug: door angle (eval) + image after each regrasp phase
        _orig_exec(api_, points, desc)
        if "regrasp" in desc:
            st_ = call([config.GET_ROBOT_STATE])
            print(f"[door] {desc}: eef={np.round(st_['eef_pos'], 3).tolist()} w={st_.get('gripper_width')} "
                  f"door(eval)={math.degrees(door_angle()):.1f}deg", flush=True)
            if not desc.endswith("advance"):  # rg_*.png
                call([config.CAPTURE_IMAGES])
                shutil.copy(config.rgb_image_head_path, os.path.join(args.out, f"rg_{desc.split(': ')[-1]}.png"))

    if args.debug:
        recovery._execute = _dbg_exec
    res = recovery.open_with_touch(api, direction=list(d0), approach_yaw=None if args.approach == "top" else yaw, rotation=rot if args.approach == "top" else math.pi / 2, max_regrasps=args.max_regrasps, grip_force=args.grip, pull_config={"k_yaw": args.k_yaw})
    print(f"[door] open_with_touch -> {json.dumps({k: v for k, v in res.items() if k != 'attempts'})}", flush=True)
else:
    start = list(grasp)
    if args.press:
        start = list(np.asarray(grasp[:3]) - [0, 0, args.press]) + list(grasp[3:])
        exe(lin(grasp, start, step=0.005), "press", settle=3)
    if args.arc:                                     # diagnosis only: the true arc (what a perfect plan does)
        u_h = (hinge[:2] - np.asarray(start[:2])) / np.linalg.norm(hinge[:2] - np.asarray(start[:2]))
        c0 = hinge[:2] + u_h * args.hinge_err
        r0 = np.linalg.norm(np.asarray(start[:2]) - c0)
        a0 = math.atan2(start[1] - c0[1], start[0] - c0[0])
        sgn = 1.0 if np.dot(np.cross([0, 0, 1.0], [math.cos(a0), math.sin(a0), 0]), tangent) > 0 else -1.0
        sgn = -sgn if args.arc_wrong else sgn
        pts = []
        for t in np.linspace(0, args.arc, int(args.arc / 3) + 1)[1:]:
            th = sgn * math.radians(t) if args.arc_yaw else 0.0
            if args.approach == "top" or not args.arc_yaw:
                pt = [c0[0] + r0 * math.cos(a0 + sgn * math.radians(t)), c0[1] + r0 * math.sin(a0 + sgn * math.radians(t))] \
                    + list(start[2:])
                if args.approach == "top":
                    pt[3] = start[3] + th
            else:                                    # side: rotate the fingertip point and the approach axis
                tip0 = np.asarray(start[:3]) + a * DZ
                v = tip0[:2] - c0
                cs, sn = math.cos(th), math.sin(th)
                tip_t = np.r_[c0 + [cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]], tip0[2]]
                a_t = np.array([cs * a[0] - sn * a[1], sn * a[0] + cs * a[1], 0.0])
                pt = side_grasp_pose(*(tip_t - a_t * DZ), math.pi / 2, yaw + th)
            pts.append(pt)
    else:
        endp = np.asarray(start[:3]) + d0 * args.pull
        pts = lin(start, list(endp) + list(start[3:]), step=0.02)
    for k, pt in enumerate(pts):
        exe([pt], f"pull {k}", settle=3, tip=np.asarray(st["eef_pos"]) + d0 * args.pull * (k + 1) / len(pts))
        if SLIPS:
            break
lv, RLv = frame(latch_link)
lever_now = lv + RLv @ np.array([args.grip_x, -0.15, 0.0])
st_end = call([config.GET_ROBOT_STATE])
print(f"[door] (eval) lever grasp point now {np.round(lever_now, 3).tolist()} tcp {np.round(st_end['eef_pos'], 3).tolist()} "
      f"along-lever offset {float(np.dot(np.asarray(st_end['eef_pos']) - lever_now, RLv[:, 0])) * 100:.1f}cm", flush=True)
print(f"[door] open wall time {time.time() - t1:.0f}s", flush=True)
ang = door_angle()
print(f"[door] RESULT {'touch' if args.touch else 'open-loop'} dir={args.dir} err={args.err}: "
      f"door angle {math.degrees(ang):.1f} deg -> {'OPEN' if ang > math.radians(20) else 'not open'}", flush=True)
cap2 = call([config.CAPTURE_IMAGES])
shutil.copy(config.rgb_image_head_path, os.path.join(args.out, f"head_after_{'touch' if args.touch else 'ol'}.png"))
if VISION:                                           # pixels-only door angle vs the eval truth
    from touch.door_plane import measure_door_angle
    _dep = np.load(os.path.splitext(config.depth_image_head_path)[0] + ".npy").astype(np.float32)
    _mp = os.path.splitext(config.depth_image_head_path)[0] + "_robot_mask.npy"
    _pts = depth_to_world(_dep, cap2[0], cap2[1], cap2[5] if len(cap2) > 5 else None,
                          robot_mask=np.load(_mp) if os.path.exists(_mp) else None)
    print(f"[door] vision door angle {measure_door_angle(_pts, VISION, lp[2])} (true {math.degrees(ang):.1f})", flush=True)
os._exit(0 if ang > math.radians(20) else 2)
