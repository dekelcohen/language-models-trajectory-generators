"""Ablation: does 3D-tracking evidence help the VLM reviewer judge a door pull?

Stage ``render`` (PyBullet, headless, no LLM):
  For each scripted scenario the Franka arm is driven *kinematically* (IK + joint reset per
  frame, no contact physics) and the Adroit door hinge/latch are set to a scripted ground
  truth, so every outcome - wide open, slightly open, slip mid-arc, latched slip, wrong
  direction - is exactly known. Per frame (10 fps = one tracking reference frame):
    * head + wrist RGB -> ``head.mp4`` / ``wrist.mp4`` (the reviewer's video evidence),
    * the REAL ``tracking.session.TrackingSession`` (template 2D tracker + pose_tracker 3D
      lift) with the production monitors running live (``attached_to_gripper``,
      ``object_not_lost``, ``moved_at_least``) -> per-frame monitor status = the
      real-time-monitor test,
    * an "oracle" tracker: the same seed points carried by the true door pose + 2 mm noise,
      hidden when the head camera does not see them.
  Both point streams go through ``tracking.review_evidence.summarize_pull`` (window = the
  pull, i.e. tracking started just before the opening trajectory) -> evidence text. No hinge
  prior is given (sim and real do not know the hinge); the true hinge only scores the
  recommended next pull direction.

Stage ``review`` (VLM): sends each clip to the reviewer under several arms
  A video only (production today) | B video + real-tracker evidence |
  O video + oracle evidence | C video + MISLEADING evidence | D evidence text only
and scores success / door_state / grasp_kept / pull_direction against ground truth.
Arm A runs first on every scenario; when it already gets every hard case right the run
stops (tracking cannot improve a perfect reviewer) unless ``--force``.

Usage::

    python tests/tracking_tools/review_ablation.py render
    python tests/tracking_tools/review_ablation.py evidence      # re-derive evidence, no GPU
    python tests/tracking_tools/review_ablation.py review --model or-google/gemini-3.8-flash
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

OUT_ROOT = os.path.join(REPO, "outputs", "review_ablation")
CLIPS = os.path.join(OUT_ROOT, "clips")
OBJECT = "door_handle"
FPS = 10
N_FRAMES = 110
# phase boundaries (frame indices)
F_APPROACH, F_INSERT, F_CLOSE, F_PULL, F_PULL_END = 10, 30, 40, 45, 95
PRE_GRASP_M = 0.12
FINGER_STANDOFF_M = 0.02          # finger centre sits this far out from the axle cap
#: latched_slip grasps the visible lever bar (latch-link frame; bar = r 0.02 m along x at
#: y=-0.15, the robot-facing axle end) and closes to the bar's width.
LEVER_GRASP_LOCAL = (0.13, -0.15, 0.0)
LEVER_HOLD_WIDTH = 0.019           # per-finger joint opening that pinches the 4 cm bar
LEVER_SLIP_M = 0.035               # pull distance after which the bar has left the fingertips
ORACLE_NOISE_M = 0.002
#: 3x3 pixel grid around the handle (px) used to seed both trackers - region seeding.
SEED_GRID_PX = (-6, 0, 6)
#: Door-panel rings around the lever root (px radii, points per ring): visible while the
#: fingers cover the lever. Kept only if they deproject onto the door near the handle.
PANEL_RING_PX = (20, 32)
PANEL_RING_N = 10
PANEL_FACE_OFFSET_M = 0.05         # door panel half-thickness: hinge plane -> front face
PANEL_MAX_DEPTH_M = 0.02
PANEL_MAX_DIST_M = 0.25
#: Point streams turned into reviewer evidence (see ``render``).
EVIDENCE_SOURCES = ("real", "online_bf", "offline", "oracle")

TASK = "Open the door."
PLAN_CONTEXT = (
    "Plan that was executed for this attempt (from the planner):\n"
    "  T1 approach: move the open gripper in front of the door handle lever.\n"
    "  T2 grasp: move in and close the gripper on the lever.\n"
    "  T3 open: press the lever and pull the door open along its hinge arc (~50 deg).\n"
    "  T4 hold the final pose.\n"
)

#: Ground truth per scenario. ``hard`` = the cases a reviewer tends to get wrong.
SCENARIOS = {
    "wide_open": dict(success=True, door_state="wide_open", grasp_kept=True,
                      pull_direction_correct=True, hard=False,
                      desc="grasp, pull along the arc, door opens ~50 deg"),
    "slight_open": dict(success=False, door_state="slightly_open", grasp_kept=True,
                        pull_direction_correct=True, hard=True,
                        desc="grasp, pull along the arc, door stops at ~8 deg"),
    "slip_mid_arc": dict(success=False, door_state="slightly_open", grasp_kept=False,
                         pull_direction_correct=True, hard=True,
                         desc="door follows to ~12 deg then the lever slips out; gripper "
                              "keeps following the planned arc empty"),
    "latched_slip": dict(success=False, door_state="closed", grasp_kept=False,
                         pull_direction_correct=True, hard=True,
                         desc="gripper pinches the lever bar but never presses it; the door "
                              "stays latched, and pulling straight back the bar slides out of "
                              "the fingertips"),
    "wrong_dir_lost": dict(success=False, door_state="closed", grasp_kept=False,
                           pull_direction_correct=False, hard=True,
                           desc="gripper drags sideways along the door face (wrong direction) "
                                "and loses the lever; door stays closed"),
}


def log(msg):
    print(f"[review-ablation] {msg}", flush=True)


def gpu_stats():
    """One-line GPU snapshot (the A1000 is shared): VRAM used/total, util, other procs, torch peak."""
    import subprocess
    parts = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10).stdout.strip().splitlines()[0]
        used, total, util = (int(x) for x in q.split(","))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=10).stdout.strip().splitlines()
        others = [a.strip() for a in apps if a.strip() and int(a.split(",")[0]) != os.getpid()]
        parts.append(f"vram {used}/{total} MiB util {util}% other_procs={others or 'none'}")
    except Exception as exc:                                       # noqa: BLE001
        parts.append(f"nvidia-smi unavailable ({exc})")
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"torch peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    except ImportError:
        pass
    return "; ".join(parts)


# =====================================================================================
# render
# =====================================================================================
def _smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def _phase(f, a, b):
    return _smooth((f - a) / float(max(1, b - a)))


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_from_matrix(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()          # xyzw, like pybullet


class DoorRig:
    """Kinematic arm + scripted door on the production door scene."""

    def __init__(self):
        import pybullet as p
        import test_tracking_door_pybullet as T
        self.p, self.T = p, T
        self.sim, self.env, self.robot = T._boot()
        st = self.env.simenv.get_state()
        self.door, self.hinge, self.latch = st["door_id"], st["door_hinge_index"], st["latch_index"]
        self.handle_link = st["door_handle_latch"]
        self.door_link = self.hinge                        # hinge child link == door panel
        ls = p.getLinkState(self.door, self.hinge)
        self.hinge_point = np.array(ls[4], dtype=float)
        self.hinge_axis = np.array([0.0, 0.0, 1.0])
        movable = self.sim.get_movable_joints(self.robot.id)
        self.arm_joints = [j.index for j in movable][:7]
        self.limits = ([j.lower_limit for j in movable], [j.upper_limit for j in movable])
        self.set_door(0.0, 0.0)
        self.home_pos, home_q = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        self.home_pos = np.asarray(self.home_pos, dtype=float)
        self.home_R = np.asarray(p.getMatrixFromQuaternion(home_q)).reshape(3, 3)
        self.home_joints = [self.sim.get_joint_state(self.robot.id, j).position
                            for j in self.arm_joints]
        # Door normal pointing to the robot side, at hinge angle 0.
        lat = np.asarray(p.getJointInfo(self.door, self.latch)[13], dtype=float)
        R_l = self.link_rot(self.latch)
        n = R_l @ lat
        n[2] = 0.0
        n /= np.linalg.norm(n)
        handle0 = self.handle_pos()
        to_robot = np.asarray(self.robot.base_start_position, dtype=float) - handle0
        self.n0 = n if n @ to_robot > 0 else -n
        ls = p.getLinkState(self.door, self.handle_link, computeForwardKinematics=True)
        self.lever0 = (np.asarray(ls[4], dtype=float)
                       + np.asarray(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
                       @ np.asarray(LEVER_GRASP_LOCAL))
        log(f"scene: hinge_point={np.round(self.hinge_point, 3).tolist()} "
            f"handle0={np.round(handle0, 3).tolist()} n_robot={np.round(self.n0, 3).tolist()} "
            f"radius={np.linalg.norm((handle0 - self.hinge_point)[:2]):.3f} m")

    def init_cap0(self):
        """Lever root (axle cap facing the robot) at hinge 0 = the finger target."""
        self.set_door(0.0, 0.0)
        head0 = self.robot.capture_camera_view("head", self.env)

        class _Aim(self.T.TestTrackingDoorPyBullet):
            pass
        aimer = _Aim("test_affordance_pixel_is_derived_and_seeds_on_the_handle")
        aimer.sim, aimer.door_id, aimer.handle_link = self.sim, self.door, self.handle_link
        self.cap0 = aimer.handle_aim_point(head0) + FINGER_STANDOFF_M * self.n0

    # -- door ---------------------------------------------------------------
    def set_door(self, hinge_rad, latch_rad):
        self.sim.reset_joint_state(self.door, self.hinge, float(hinge_rad))
        self.sim.reset_joint_state(self.door, self.latch, float(latch_rad))

    def link_rot(self, link):
        _pos, q = self.sim.get_link_pose(self.door, link)
        return np.asarray(self.p.getMatrixFromQuaternion(q)).reshape(3, 3)

    def handle_pos(self):
        pos, _q = self.sim.get_link_pose(self.door, self.handle_link)
        return np.asarray(pos, dtype=float)

    def door_pose(self):
        pos, q = self.sim.get_link_pose(self.door, self.door_link)
        return np.asarray(pos, dtype=float), np.asarray(self.p.getMatrixFromQuaternion(q)).reshape(3, 3)

    def hinge_angle(self):
        return float(self.sim.get_joint_state(self.door, self.hinge).position)

    def grasp_point(self, hinge_rad):
        """Finger target on the lever root for a given hinge angle (rotates with the door)."""
        h0 = self.cap0
        R = _rot_z(hinge_rad)
        return self.hinge_point + R @ (h0 - self.hinge_point)

    def normal(self, hinge_rad):
        return _rot_z(hinge_rad) @ self.n0

    def grasp_R(self, hinge_rad):
        z = -self.normal(hinge_rad)               # approach: into the door
        y = np.array([0.0, 0.0, 1.0])             # fingers pinch the horizontal lever top/bottom
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z])

    # -- arm ----------------------------------------------------------------
    def set_arm(self, finger_pos, R, gripper_open):
        euler = self.sim.euler_from_quat(_quat_from_matrix(R))
        target = self.robot._apply_gripper_depth_offset(
            list(finger_pos), euler, 0.0)          # panda_grasptarget = finger centre
        q = _quat_from_matrix(R)
        lo, hi = self.limits
        ranges = [abs(b - a) for a, b in zip(lo, hi)]
        for _ in range(4):
            sol = self.sim.inverse_kinematics(self.robot.id, self.robot.ee_index, target,
                                              orientation_q=q, lower_limits=lo, upper_limits=hi,
                                              joint_ranges=ranges, rest_poses=self.home_joints + [0.02, 0.02],
                                              max_iterations=500)
            for j, v in zip(self.arm_joints, sol[:7]):
                self.sim.reset_joint_state(self.robot.id, j, v)
        if isinstance(gripper_open, bool):
            width = 0.04 if gripper_open else 0.0005
        else:
            width = float(gripper_open)            # explicit per-finger opening (m)
        for j in self.robot.gripper_joint_indices:
            self.sim.reset_joint_state(self.robot.id, j, width)
        pos, _q = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return float(np.linalg.norm(np.asarray(pos) - np.asarray(target)))

    def ee_pose(self):
        pos, q = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return {"position": list(pos), "orientation_q": list(q)}


def scenario_frame(name, f, rig):
    """(hinge_rad, latch_rad, finger_pos, R, gripper_open) for frame ``f`` of a scenario."""
    deg = np.radians
    lever = name == "latched_slip"
    g0 = rig.lever0 if lever else rig.grasp_point(0.0)
    pre = lambda th: rig.grasp_point(th) + PRE_GRASP_M * rig.normal(th)
    pre0 = g0 + PRE_GRASP_M * rig.n0

    if f < F_APPROACH:
        return 0.0, 0.0, rig.home_pos, rig.home_R, True
    if f < F_INSERT:
        u = _phase(f, F_APPROACH, F_INSERT)
        pos = (1 - u) * rig.home_pos + u * pre0
        R = rig.home_R if u < 0.5 else rig.grasp_R(0.0)
        return 0.0, 0.0, pos, R, True
    if f < F_CLOSE:
        u = _phase(f, F_INSERT, F_CLOSE)
        return 0.0, 0.0, (1 - u) * pre0 + u * g0, rig.grasp_R(0.0), True
    if f < F_PULL:
        press = name in ("wide_open", "slight_open", "slip_mid_arc")
        lat = 0.6 * _phase(f, F_CLOSE, F_PULL) if press else (0.15 * _phase(f, F_CLOSE, F_PULL)
                                                              if name == "wrong_dir_lost" else 0.0)
        return 0.0, lat, g0, rig.grasp_R(0.0), (LEVER_HOLD_WIDTH if lever else False)

    u = _phase(min(f, F_PULL_END), F_PULL, F_PULL_END)
    if name == "wide_open":
        th = deg(50) * u
        return th, 0.6, rig.grasp_point(th), rig.grasp_R(th), False
    if name == "slight_open":
        th = deg(8) * u
        return th, 0.6, rig.grasp_point(th), rig.grasp_R(th), False
    if name == "slip_mid_arc":
        th_g = deg(50) * u
        th_d = min(th_g, deg(12))
        slipped = th_g > deg(12)
        return th_d, (0.0 if slipped else 0.6), rig.grasp_point(th_g), rig.grasp_R(th_g), False
    if name == "latched_slip":
        # door is latched: the hand pulls straight back, the pinched bar slides out of the
        # fingertips after LEVER_SLIP_M, then the empty fingers snap shut
        d = 0.20 * u
        width = LEVER_HOLD_WIDTH if d < LEVER_SLIP_M else 0.0005
        return 0.0, 0.0, rig.lever0 + d * rig.n0, rig.grasp_R(0.0), width
    if name == "wrong_dir_lost":
        side = np.cross(rig.hinge_axis, rig.n0)
        if side @ (rig.grasp_point(0.0) - rig.hinge_point) < 0:
            side = -side
        return 0.0, 0.15 * (1 - u), rig.grasp_point(0.0) + 0.20 * u * side, rig.grasp_R(0.0), False
    raise ValueError(name)


def _seed_points(rig, view):
    """Region seed: 3x3 pixel grid on the lever root + rings on the door panel around it.

    The lever root is exactly where the fingers close, so during the pull those points are
    hidden (and their depth is the gripper's). Panel points around the handle stay visible
    and are rigid with the door, so they carry the opening angle on their own. Returns
    ``(world_points (N,3), pixels (N,2), centre_px, n_lever)``.
    """
    import utils
    from tracking import geometry
    aim = rig.cap0
    pix, _z = geometry.project_world_to_pixel(view, aim)
    cand = [(dx, dy) for dy in SEED_GRID_PX for dx in SEED_GRID_PX]
    n_grid = len(cand)
    for r in PANEL_RING_PX:
        for a in np.linspace(0, 2 * np.pi, PANEL_RING_N, endpoint=False):
            cand.append((r * np.cos(a), r * np.sin(a)))
    pts, pixels, n_lever = [], [], 0
    for i, (dx, dy) in enumerate(cand):
        x, y = int(round(pix[0] + dx)), int(round(pix[1] + dy))
        if not geometry.in_bounds(view, (x, y)):
            continue
        d, _info = utils.sample_surface_depth(view.depth, x, y)
        if d is None:
            continue
        w = np.asarray(geometry.deproject_pixel_to_world(view, (x, y), d), dtype=float)
        if i >= n_grid:
            # panel ring: keep only points on the door's front face near the handle (not the
            # lever/axle sticking out of it, nor the wall / floor behind)
            off = w - aim
            face = (w - rig.hinge_point) @ rig.n0 - PANEL_FACE_OFFSET_M
            lateral = np.linalg.norm(off - (off @ rig.n0) * rig.n0)
            if abs(face) > PANEL_MAX_DEPTH_M or lateral > PANEL_MAX_DIST_M:
                continue
        else:
            n_lever += 1
        pts.append(w)
        pixels.append((x, y))
    return (np.asarray(pts, dtype=float), np.asarray(pixels, dtype=float),
            [int(round(pix[0])), int(round(pix[1]))], n_lever)


_OFFLINE = {}


def _offline_tracks(frames_rgb, query_px):
    """CoTracker3 *offline* over the whole clip (queries on frame 0). GPU fp16 first; the
    A1000 is shared and a 110-frame volume may not fit, so OOM falls back to CPU (and stays
    there for the rest of the run). Returns ``(tracks (T,N,2), vis (T,N), device)``."""
    import torch
    from providers.trackers import cotracker_tracker as ct
    if "model" not in _OFFLINE:
        ct.ensure_torch_home()
        hub_dir = os.path.join(ct.DEFAULT_TORCH_HOME, "hub", "facebookresearch_co-tracker_main")
        model = (torch.hub.load(hub_dir, "cotracker3_offline", source="local")
                 if os.path.isdir(hub_dir) else ct._hub_load("cotracker3_offline"))
        dev = ct.resolve_device()
        _OFFLINE["model"], _OFFLINE["device"] = model.to(dev).eval(), dev
    model, dev = _OFFLINE["model"], _OFFLINE["device"]
    video = torch.tensor(np.stack(frames_rgb).astype(np.float32)).permute(0, 3, 1, 2)[None]
    q = torch.tensor(np.concatenate([np.zeros((len(query_px), 1)), query_px], axis=1),
                     dtype=torch.float32)[None]
    try:
        if not str(dev).startswith("cuda"):
            raise torch.cuda.OutOfMemoryError
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tracks, vis = model(video.to(dev), queries=q.to(dev), backward_tracking=False)
    except torch.cuda.OutOfMemoryError:
        if str(dev).startswith("cuda"):
            log(f"  offline: CUDA OOM on the shared GPU ({gpu_stats()}) -> CPU from now on")
            torch.cuda.empty_cache()
            model.cpu()
            _OFFLINE["device"] = dev = "cpu"
        with torch.no_grad():
            tracks, vis = model(video, queries=q, backward_tracking=False)
    tracks = tracks[0, :, :len(query_px)].float().cpu().numpy()
    vis = vis[0, :, :len(query_px)].float().cpu().numpy()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    return tracks, vis, str(dev)


def gt_pull_dir(rig, name, rows):
    """SCORING ONLY (never fed to the evidence code): the true opening direction of the
    grasped point at the end of the clip = the arc tangent ``z x (p - hinge)`` at the handle
    grasp point rotated by the final hinge angle (the handle sticks out of the door, so this
    is ~20 deg off the door normal). Returns ``(pull_dir, door_normal_at_0)``."""
    th = np.radians(rows[-1]["hinge_deg"])
    p0 = rig.lever0 if name == "latched_slip" else rig.grasp_point(0.0)
    p = rig.hinge_point + _rot_z(th) @ (p0 - rig.hinge_point)
    t = np.cross(rig.hinge_axis, p - rig.hinge_point)
    t[2] = 0.0
    t /= np.linalg.norm(t)
    if t @ rig.normal(th) < 0:
        t = -t
    return t, rig.n0


def _angle_between_deg(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))), 1)


def compute_evidence(rows, seeds, gt, name):
    """Hinge-free reviewer evidence per point source. ``seeds`` = the tracker's seed points
    (measured on the door front from depth before the pull: what production has at
    ``track_objects`` time); the robot's home gripper position orients the face normal."""
    from tracking.review_evidence import format_evidence, summarize_pull
    win = slice(F_CLOSE, N_FRAMES)       # tracking window for the reviewer = the pull
    grip = [r["gripper"] for r in rows[win]]
    view = rows[0]["gripper"]            # robot side of the door (EE at home)
    true_deg = float(rows[-1]["hinge_deg"])
    gt_dir, gt_n0 = gt
    ev = {}
    for src in EVIDENCE_SOURCES:
        if src not in rows[0]:
            continue
        frames = [np.asarray(r[src], dtype=float) for r in rows[win]]
        facts = summarize_pull(frames, grip, dt=1.0 / FPS, face_points=seeds, view_point=view,
                               logger=lambda m: log("    " + m))
        score = {"true_angle_deg": round(true_deg, 1),
                 "angle_err_deg": (None if facts["angle_deg"] is None
                                   else round(abs(facts["angle_deg"] - true_deg), 1)),
                 "gt_pull_dir": np.round(gt_dir, 3).tolist()}
        if facts.get("recommended_pull_dir") is not None:
            score["rec_err_deg"] = _angle_between_deg(facts["recommended_pull_dir"], gt_dir)
        if facts.get("face_normal") is not None:
            score["normal_err_deg"] = _angle_between_deg(facts["face_normal"], gt_n0)
        ev[src] = {"facts": facts, "text": format_evidence(facts), "score": score}
        log(f"  {name} evidence[{src}]: angle {facts['angle_deg']} vs true {true_deg:.1f} "
            f"({facts['angle_source']}) measured {facts['handle_measured_frames']}/"
            f"{facts['n_frames']} flags={facts['flags']} align={facts.get('pull_alignment_cos')}"
            f"/{facts.get('pull_alignment_ref')} rec={facts.get('recommended_pull_dir')} "
            f"({facts.get('recommended_pull_dir_source')}) rec_err={score.get('rec_err_deg')} deg "
            f"normal_err={score.get('normal_err_deg')} deg")
    return ev


def evidence(args):
    """Recompute the (hinge-free) evidence from saved ``rollout.json`` files - no GPU, no
    re-render. The rig is booted only to get the ground-truth pull direction for scoring."""
    rig = DoorRig()
    rig.init_cap0()
    path = os.path.join(CLIPS, "index.json")
    with open(path) as fh:
        index = json.load(fh)
    rows_out = []
    for name in args.scenarios or list(index):
        with open(os.path.join(CLIPS, name, "rollout.json")) as fh:
            roll = json.load(fh)
        rows = roll["frames"]
        seeds = roll.get("seeds")
        if seeds is None:   # older renders: the oracle's first frame = seeds + 2 mm noise
            seeds = rows[0]["oracle"]
        seeds = np.asarray(seeds, dtype=float)
        ev = compute_evidence(rows, seeds, gt_pull_dir(rig, name, rows), name)
        index[name]["evidence"] = ev
        index[name]["gt_pull_dir"] = np.round(gt_pull_dir(rig, name, rows)[0], 3).tolist()
        roll["evidence"] = ev
        with open(os.path.join(CLIPS, name, "rollout.json"), "w") as fh:
            json.dump(roll, fh)
        for src, e in ev.items():
            f, sc = e["facts"], e["score"]
            rows_out.append(f"| {name} | {src} | {f['angle_deg']} / {sc['true_angle_deg']} "
                            f"| {f['handle_measured_frames']}/{f['n_frames']} "
                            f"| {f.get('pull_alignment_cos')} ({f.get('pull_alignment_ref')}) "
                            f"| {', '.join(f['flags']) or '-'} "
                            f"| {f.get('recommended_pull_dir_source')} | {sc.get('rec_err_deg')} |")
    with open(path, "w") as fh:
        json.dump(index, fh, indent=2)
    table = "\n".join(["| scenario | src | angle meas/true | seen | align cos (ref) | flags "
                       "| rec source | rec err deg |", "|---" * 8 + "|"] + rows_out)
    with open(os.path.join(CLIPS, "evidence_report.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)


def render(args):
    import cv2
    from PIL import Image
    from tracking import monitors
    from tracking.session import TrackingSession
    from tracking.types import STATUS_RECORD
    from tracking import geometry

    rig = DoorRig()
    rig.init_cap0()
    T = rig.T

    names = args.scenarios or list(SCENARIOS)
    log(f"tracker={args.tracker_provider}; gpu before: {gpu_stats()}")
    index = {}
    if os.path.isfile(os.path.join(CLIPS, "index.json")):     # re-rendering a subset keeps the rest
        with open(os.path.join(CLIPS, "index.json")) as fh:
            index = json.load(fh)
    for name in names:
        out = os.path.join(CLIPS, name)
        os.makedirs(out, exist_ok=True)
        rig.set_door(0.0, 0.0)
        # IK seeds from the current joints: start every scenario from the true home joints,
        # not from wherever the previous scenario left the arm (that gave 0.3 m IK misses).
        for j, v in zip(rig.arm_joints, rig.home_joints):
            rig.sim.reset_joint_state(rig.robot.id, j, v)
        rig.set_arm(rig.home_pos, rig.home_R, True)
        head = rig.robot.capture_camera_view("head", rig.env)
        seeds, seed_pix, seed_px, n_lever = _seed_points(rig, head)
        door_t0, door_R0 = rig.door_pose()
        seeds_local = (seeds - door_t0) @ door_R0          # door-panel frame (rigid oracle)
        Image.fromarray(np.asarray(head.rgb)[..., :3].astype(np.uint8)).save(
            os.path.join(out, "start_head.png"))

        mon = monitors.combine(
            monitors.attached_to_gripper(OBJECT, status=STATUS_RECORD),   # would-be abort
            monitors.object_not_lost(OBJECT),
            monitors.moved_at_least(OBJECT, 0.05, after_frames=F_PULL_END),
        )
        session = TrackingSession(robot=rig.robot, env=rig.env, provider=args.tracker_provider,
                                  tracker3d="pose_tracker", monitor=mon, write_jsonl=False,
                                  tracker_kwargs=dict(T.TRACKER_KWARGS),
                                  run_id=f"ablation_{name}")
        session.add_target(OBJECT, seeds)
        log(f"{name}: seeded {len(seeds)} pts ({n_lever} lever + {len(seeds) - n_lever} door "
            f"panel) at head px {seed_px}; {SCENARIOS[name]['desc']}")

        writers, rng = {}, np.random.default_rng(abs(hash(name)) % 2**31)
        rows, ik_err = [], []
        sess_to_f, backfill = {}, {"head": {}, "wrist": {}}
        head_rgb, head_views = [], []
        for f in range(N_FRAMES):
            th, lat, finger, R, g_open = scenario_frame(name, f, rig)
            rig.set_door(th, lat)
            ik_err.append(rig.set_arm(finger, R, g_open))
            views = session.capture_views()
            for cam in ("head", "wrist"):
                v = views.get(cam) or rig.robot.capture_camera_view(cam, rig.env)
                bgr = cv2.cvtColor(np.asarray(v.rgb)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                if cam not in writers:
                    writers[cam] = cv2.VideoWriter(os.path.join(out, f"{cam}.mp4"),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                                   (bgr.shape[1], bgr.shape[0]))
                writers[cam].write(bgr)
                if f == N_FRAMES - 1 and cam == "head":
                    cv2.imwrite(os.path.join(out, "final_head.png"), bgr)
            gp = rig.ee_pose()
            head_views.append(views["head"])
            head_rgb.append(np.asarray(views["head"].rgb)[..., :3].astype(np.uint8))
            sess_to_f[session.frame_idx] = f
            rep = session.on_frame(views=views, gripper_pose=gp, trajectory_step=f)

            # oracle: seed points carried by the true door pose + noise, hidden if occluded
            dt_, dR = rig.door_pose()
            truth = seeds_local @ dR.T + dt_
            oracle = truth + rng.normal(0.0, ORACLE_NOISE_M, truth.shape)
            hv = views.get("head")
            for i in range(len(oracle)):
                if hv is not None and not geometry.is_visible(hv, truth[i])[0]:
                    oracle[i] = np.nan

            real = np.full_like(seeds, np.nan)
            mon_status, mon_reason, pose_src = "none", "", None
            if rep is not None:
                obj = rep.objects.get(OBJECT)
                mon_status, mon_reason = rep.monitor.status, rep.monitor.reason
                if obj is not None and obj.pose is not None:
                    pose_src = f"{obj.pose.source}/{obj.pose.mode}"
                    if obj.pose.trusted:
                        real = obj.pose.apply(seeds)
                for cam, store in backfill.items():
                    track = obj.cams.get(cam) if obj is not None else None
                    for rec in (getattr(track, "backfill", None) or []):
                        if rec["frame_idx"] in sess_to_f:
                            store[sess_to_f[rec["frame_idx"]]] = rec["points_world"]
            rows.append({"f": f, "hinge_deg": round(float(np.degrees(rig.hinge_angle())), 2),
                         "latch": round(float(lat), 3), "gripper": gp["position"],
                         "monitor": mon_status, "reason": mon_reason, "pose": pose_src,
                         "real": real.tolist(), "oracle": oracle.tolist()})
            if f % 10 == 0 or (mon_status not in ("ok", "none") and rows[-2]["monitor"] in ("ok", "none")):
                log(f"  f={f:3d} hinge={rows[-1]['hinge_deg']:5.1f} deg ik_err={ik_err[-1]:.3f} "
                    f"pose={pose_src} monitor={mon_status} {mon_reason[:110]}")
        for w in writers.values():
            w.release()
        session_summary = session.summary()
        session.close() if hasattr(session, "close") else None
        log(f"  gpu after {name}: {gpu_stats()}")

        # online + backfill: the trusted live pose where there is one, else the windowed
        # tracker's fresh per-frame rows (head first, wrist second) lifted with that frame's depth
        n_bf = 0
        for r in rows:
            if np.isfinite(np.asarray(r["real"], dtype=float)).any():
                r["online_bf"] = r["real"]
                continue
            pts = backfill["head"].get(r["f"])
            if pts is None or not np.isfinite(pts).any():
                pts = backfill["wrist"].get(r["f"])
            if pts is not None and np.isfinite(pts).any():
                n_bf += 1
                r["online_bf"] = np.asarray(pts, dtype=float).tolist()
            else:
                r["online_bf"] = np.full_like(seeds, np.nan).tolist()

        # offline CoTracker3 over the recorded head clip, lifted with each frame's depth
        t0 = time.time()
        try:
            tr, vi, odev = _offline_tracks(head_rgb, seed_pix)
            for r, v, pts2, vis in zip(rows, head_views, tr, vi):
                world, valid = geometry.points_to_world(v, pts2, vis > 0.5)
                r["offline"] = np.where(valid[:, None], world, np.nan).tolist()
            log(f"  offline cotracker3 on {odev}: {time.time() - t0:.1f} s; {gpu_stats()}")
        except Exception as exc:                                   # noqa: BLE001
            log(f"  offline cotracker3 FAILED: {type(exc).__name__}: {exc}")
            for r in rows:
                r["offline"] = np.full_like(seeds, np.nan).tolist()

        ev = compute_evidence(rows, seeds, gt_pull_dir(rig, name, rows), name)
        log(f"  backfilled {n_bf} frames beyond the trusted live poses")

        first = {}
        for r in rows:
            for key in ("warn", "record"):
                if r["monitor"] == key and key not in first:
                    first[key] = {"f": r["f"], "t_s": round(r["f"] / FPS, 1), "reason": r["reason"]}
            if "attached_to_gripper" in r["reason"] and "detach" not in first \
                    and r["monitor"] == "record":
                first["detach"] = {"f": r["f"], "t_s": round(r["f"] / FPS, 1), "reason": r["reason"]}
        result = {"scenario": name, "gt": SCENARIOS[name], "seed_px": seed_px,
                  "n_seeds": len(seeds), "n_lever_seeds": n_lever,
                  "tracker_provider": args.tracker_provider,
                  "final_hinge_deg": rows[-1]["hinge_deg"],
                  "ik_err_max_m": round(float(max(ik_err)), 4),
                  "monitor_first": first, "evidence": ev,
                  "session": {k: session_summary.get(k) for k in
                              ("status", "abort_reason", "num_frames", "flagged_frames")},
                  "videos": {c: os.path.join(out, f"{c}.mp4") for c in ("head", "wrist")},
                  "start_image": os.path.join(out, "start_head.png")}
        with open(os.path.join(out, "rollout.json"), "w") as fh:
            json.dump({**result, "seeds": seeds.tolist(), "frames": rows}, fh)
        index[name] = result
        log(f"{name}: final hinge {result['final_hinge_deg']} deg, ik_err_max "
            f"{result['ik_err_max_m']} m, monitor first={json.dumps(first)}")
    with open(os.path.join(CLIPS, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    log(f"wrote {CLIPS}\\index.json")


# =====================================================================================
# review
# =====================================================================================
EXTRA_KEYS = (
    "In addition to those keys, the JSON object MUST also contain:\n"
    "  door_state: one of \"closed\", \"slightly_open\" (a gap of a few degrees, up to ~15 deg), "
    "\"partly_open\" (~15-35 deg), \"wide_open\" (> ~35 deg) - the door at the END of the clip;\n"
    "  handle_grasp_kept: true/false - did the gripper keep holding the handle for the whole pull;\n"
    "  pull_direction_correct: true/false - did the gripper move along the door's opening arc "
    "(away from the door face, around the hinge) rather than sideways/up/down;\n"
    "  next_pull_direction: [x, y, z] - unit vector in the world frame (metres, z up) the "
    "gripper should pull the handle along on the NEXT attempt (at the handle's latest "
    "position).\n"
)

ARMS = ("A", "B", "O", "C", "D")
ARM_DESC = {"A": "video only (production)", "B": "video + real-tracker evidence",
            "O": "video + oracle-tracker evidence", "C": "video + MISLEADING evidence",
            "D": "real-tracker evidence text only (no video)"}
MISLEAD = {"wide_open": "wrong_dir_lost", "slight_open": "wide_open", "slip_mid_arc": "wide_open",
           "latched_slip": "wide_open", "wrong_dir_lost": "wide_open"}


def build_prompt(arm, name, index):
    from prompts.review_prompt import (REVIEW_PROMPT, REVIEW_MEDIA_VIDEO_SECTION,
                                       REVIEW_SCENE_ANALYSIS_SECTION, REMAINING_MEDIA_SENTENCE_VIDEO)
    clip = index[name]
    scene = (REVIEW_SCENE_ANALYSIS_SECTION
             .replace("[INSERT SCENE ANALYSIS IMAGE PATH]", "start_head.png")
             .replace("[INSERT REMAINING MEDIA SENTENCE]",
                      REMAINING_MEDIA_SENTENCE_VIDEO if arm != "D" else
                      "No trajectory video is available for this attempt.")
             .replace("[INSERT SCENE ANALYSIS]",
                      "A wooden hinged door with a horizontal lever handle; a Franka arm stands "
                      "in front of it with the gripper open."))
    if arm == "D":
        media = ""
    else:
        media = (REVIEW_MEDIA_VIDEO_SECTION
                 .replace("[INSERT VIDEO LIST]", "  1. head cam: head.mp4\n  2. wrist cam: wrist.mp4")
                 .replace("[INSERT VIDEO FPS]", str(FPS)))
    evidence = ""
    if arm in ("B", "D"):
        evidence = clip["evidence"]["real"]["text"]
    elif arm == "O":
        evidence = clip["evidence"]["oracle"]["text"]
    elif arm == "C":
        evidence = index[MISLEAD[name]]["evidence"]["oracle"]["text"]
    if evidence:
        media += evidence + "\n\n"
    prompt = (REVIEW_PROMPT.replace("[INSERT TASK]", TASK)
              .replace("[INSERT 3D COORDINATES PROMPT SECTION]",
                       "World frame in metres, z up. The door hinge is vertical.")
              .replace("[INSERT SCENE ANALYSIS SECTION]", scene)
              .replace("[INSERT MEDIA SECTION]", media))
    return prompt + "\n" + EXTRA_KEYS


def _parse(raw):
    s = (raw or "").strip()
    l, r = s.find("{"), s.rfind("}")
    if l < 0 or r <= l:
        return None
    try:
        return json.loads(s[l:r + 1])
    except json.JSONDecodeError:
        return None


class Caller:
    """Free Gemini key first (throttled), OpenRouter once the free quota is gone."""

    def __init__(self, model, fallback, min_gap_s):
        self.model, self.fallback, self.min_gap = model, fallback, float(min_gap_s)
        self.last = 0.0
        self.free_dead = not model.startswith("gemini-")

    def __call__(self, messages):
        import models
        if not self.free_dead:
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                return self.model, models.call_llm_provider(None, self.model, messages, 8000, "low")
            except Exception as exc:                       # 429 daily quota etc.
                log(f"free Gemini failed ({str(exc)[:120]}) -> switching to {self.fallback}")
                self.free_dead = True
        return self.fallback, models.call_llm_provider(None, self.fallback, messages, 8000, "low")


def review(args):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    from llm_providers.message_media import append_to_messages

    with open(os.path.join(CLIPS, "index.json")) as fh:
        index = json.load(fh)
    res_path = os.path.join(OUT_ROOT, "review_results.jsonl")
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[(r["arm"], r["scenario"], r["rep"])] = r
    call = Caller(args.model, args.fallback, args.min_gap_s)
    names = args.scenarios or list(index)

    def run_arm(arm):
        for rep in range(args.reps):
            for name in names:
                key = (arm, name, rep)
                if key in done:
                    continue
                clip = index[name]
                prompt = build_prompt(arm, name, index)
                videos = [] if arm == "D" else [clip["videos"]["head"], clip["videos"]["wrist"]]
                msgs = append_to_messages(PLAN_CONTEXT + "\n" + prompt, [clip["start_image"]], [],
                                          "user", attach_videos=videos)
                t0 = time.time()
                try:
                    used, raw = call(msgs)
                except Exception as exc:
                    log(f"{arm}/{name}/{rep}: call failed: {exc}")
                    continue
                parsed = _parse(raw) or {}
                gt = clip["gt"]
                score = {
                    "success": parsed.get("success") == gt["success"],
                    "door_state": parsed.get("door_state") == gt["door_state"],
                    "grasp_kept": parsed.get("handle_grasp_kept") == gt["grasp_kept"],
                    "pull_dir": parsed.get("pull_direction_correct") == gt["pull_direction_correct"],
                }
                rec_err = None
                nd = parsed.get("next_pull_direction")
                try:
                    nd = np.asarray(nd, dtype=float).reshape(3)
                    if clip.get("gt_pull_dir") is not None and np.linalg.norm(nd) > 1e-6:
                        rec_err = _angle_between_deg(nd, clip["gt_pull_dir"])
                except (TypeError, ValueError):
                    pass
                row = {"arm": arm, "scenario": name, "rep": rep, "model": used,
                       "next_pull_err_deg": rec_err,
                       "latency_s": round(time.time() - t0, 1), "parsed": parsed,
                       "score": score, "raw": raw}
                done[key] = row
                with open(res_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"{arm}/{name}/rep{rep} [{used}] success={parsed.get('success')} "
                    f"door={parsed.get('door_state')} grasp={parsed.get('handle_grasp_kept')} "
                    f"dir={parsed.get('pull_direction_correct')} next_dir_err={rec_err} -> {score}")

    run_arm("A")
    hard = [n for n in names if index[n]["gt"]["hard"]]
    a_rows = [done[k] for k in done if k[0] == "A" and k[1] in hard]
    a_perfect = bool(a_rows) and all(all(r["score"].values()) for r in a_rows)
    if a_perfect and not args.force:
        log("STOP: video-only reviewer (arm A) is already correct on every hard case - "
            "tracking evidence cannot improve it. Inspect the clips under " + CLIPS)
    else:
        for arm in [a for a in (args.arms or ARMS) if a != "A"]:
            run_arm(arm)
    report(done, index, names)


def report(done, index, names):
    lines = ["| arm | n | success | door_state | grasp_kept | pull_dir | all-4 | false-success "
             "| next-pull err median deg (<=30 deg) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rows = [r for k, r in done.items() if k[0] == arm and k[1] in names]
        if not rows:
            continue
        acc = lambda key: f"{np.mean([r['score'][key] for r in rows]):.0%}"
        allok = np.mean([all(r["score"].values()) for r in rows])
        fs = sum(1 for r in rows if r["parsed"].get("success") is True
                 and not index[r["scenario"]]["gt"]["success"])
        errs = [r.get("next_pull_err_deg") for r in rows]
        errs = [e if e is not None else 180.0 for e in errs]        # missing = useless
        nxt = f"{np.median(errs):.0f} ({np.mean([e <= 30 for e in errs]):.0%})"
        lines.append(f"| {arm} {ARM_DESC[arm]} | {len(rows)} | {acc('success')} | {acc('door_state')} "
                     f"| {acc('grasp_kept')} | {acc('pull_dir')} | {allok:.0%} | {fs} | {nxt} |")
    lines += ["", "Per scenario (all-4 correct / n):", ""]
    hdr = "| scenario | " + " | ".join(a for a in ARMS) + " |"
    lines += [hdr, "|---" * (len(ARMS) + 1) + "|"]
    for n in names:
        cells = []
        for arm in ARMS:
            rows = [r for k, r in done.items() if k[0] == arm and k[1] == n]
            cells.append(f"{sum(all(r['score'].values()) for r in rows)}/{len(rows)}" if rows else "-")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    with open(os.path.join(OUT_ROOT, "review_report.md"), "w") as fh:
        fh.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--scenarios", nargs="*")
    r.add_argument("--tracker-provider", default="cotracker",
                   help="2D tracker behind the live session: cotracker (CoTracker3 online, "
                        "run under the CUDA env env_lapa_tracking on the A1000) | template | ...")
    v = sub.add_parser("review")
    v.add_argument("--model", default="gemini-3.8-flash", help="free Gemini key first")
    v.add_argument("--fallback", default="or-google/gemini-3.8-flash")
    v.add_argument("--min-gap-s", type=float, default=15.0, help="throttle for the free key")
    v.add_argument("--reps", type=int, default=2)
    v.add_argument("--arms", nargs="*", choices=ARMS)
    v.add_argument("--scenarios", nargs="*")
    v.add_argument("--force", action="store_true", help="run all arms even if A is perfect")
    for p in (r, v):
        p.add_argument("--clips", default=None, help=f"clip directory (default {CLIPS})")
    e = sub.add_parser("evidence", help="recompute hinge-free evidence from rollout.json")
    e.add_argument("--scenarios", nargs="*")
    e.add_argument("--clips", default=None, help=f"clip directory (default {CLIPS})")
    a = ap.parse_args()
    if a.clips:
        globals()["CLIPS"] = os.path.abspath(a.clips)
    {"render": render, "review": review, "evidence": evidence}[a.cmd](a)


if __name__ == "__main__":
    main()
