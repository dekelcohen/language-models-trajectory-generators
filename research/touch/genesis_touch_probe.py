"""Genesis touch-sensing probe (no LLM): can a Franka "feel" contact, port misalignment and slip?

Scene: Franka, a laptop (textured mesh, visual only) on a 5 cm riser with a USB-A port on its side
that faces the robot (collision = 5 boxes leaving a real-size 12.5 x 5.1 x 14 mm USB-A pocket), a USB-A
plug (URDF: 25x16x8 mm housing + 12x12x4.5 mm tip) and a small table block. USB retention springs are
modelled as 12 N Coulomb friction on the plug while its tip is in the port (env physics only).

Experiments (each prints RESULT lines; see plans/touch.md for the conclusions):
  press  - closed fingers pressed onto the table block. Compares the true contact force on hand+
           fingers with the end-effector wrench estimated from the 7 joint-torque sensors, the way
           libfranka estimates O_F_ext_hat_K from tau_J (motor-current analogue).
  slot   - the plug, held horizontally, is pushed towards the port along +x at several (dy, dz)
           offsets. Does the wrench tell whether it went in, and the offset direction?
  push   - Cartesian-impedance pushes into the port: light / medium / firm force, off-axis push
           direction, yawed plug axis, lateral offset. Only a firm, well-aligned push inserts.
  grip   - external axial pull ramp on the held plug: where does the 20 N grasp slip?
  slip   - the held plug is pushed down onto the table with a strong vs weak squeeze. Which
           signals show it sliding in the fingers (wrench, finger taxels, gripper width)?
  render - saves a camera image of the plug in front of the port (asset sanity check).

Run (vlm_genesis env, CPU; set PYTHONIOENCODING=utf-8 on Windows):
  python research/touch/genesis_touch_probe.py --exp press --noise 0.1
  python research/touch/genesis_touch_probe.py --exp slot --offsets 0,0 0.002,0 0,0.0015
  python research/touch/genesis_touch_probe.py --exp slip
  python research/touch/genesis_touch_probe.py --exp push
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

ap = argparse.ArgumentParser()
ap.add_argument("--exp", choices=("press", "slot", "slip", "grip", "creep", "push", "search", "render", "all"),
                default="all")
ap.add_argument("--offsets", nargs="*", default=["0,0", "0.002,0", "-0.002,0", "0,0.0015", "0,-0.0015", "0.0007,0"],
                help="plug (dy,dz) offsets from the port centre, metres")
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--substeps", type=int, default=2)
ap.add_argument("--noslip", type=int, default=5, help="Genesis noslip solver iterations (0 = friction creeps)")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--armature", type=float, default=0.5, help="plug translational armature kg (conditioning, not weight)")
ap.add_argument("--stiff_fric", type=int, default=1)
ap.add_argument("--noise", type=float, default=0.0, help="joint torque sensor noise std (Nm)")
ap.add_argument("--img", default="research/touch/usb_probe.png")
ap.add_argument("--trials", type=int, default=8, help="search: random misalignment trials")
ap.add_argument("--grip", type=float, default=20.0, help="per-finger squeeze force, N")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--methods", nargs="*", default=["naive", "touch"])
ap.add_argument("--err_r", type=float, default=0.002, help="search: max lateral alignment error, m")
ap.add_argument("--err_yaw", type=float, default=2.0, help="search: max plug yaw error, deg")
ap.add_argument("--case", default="", help="search: fixed 'dy,dz,yaw' (m,m,deg) for the first trial")
ap.add_argument("--standoff", type=float, default=0.004, help="search: start gap tip -> port face, m")
ap.add_argument("--k_lat", type=float, default=5000.0, help="search: lateral impedance stiffness N/m")
ap.add_argument("--kr", type=float, default=40.0, help="search: rotational impedance stiffness Nm/rad")
ap.add_argument("--tcp", type=int, default=1, help="search: impedance/EE frame at the nominal plug tip (1) or hand (0)")
ap.add_argument("--fnoise", type=float, default=0.5, help="search: force-estimate noise std, N")
ap.add_argument("--max_s", type=float, default=20.0, help="search: controller timeout, s")
ap.add_argument("--f_search", type=float, default=3.0, help="search: press force while spiralling, N")
ap.add_argument("--trace", type=int, default=0, help="search: print an eval-only sim-truth trace (debug)")
ap.add_argument("--verbose", type=int, default=1, help="search: print controller events for the first N trials")
args = ap.parse_args()

gs.init(backend=gs.cpu, precision="32", logging_level="warning")

ASSETS = "sim_envs/assets/usb_laptop"
TIP_LEN, TIP_W, TIP_H = 0.012, 0.012, 0.0045   # USB-A metal tip (x along plug axis, y wide, z thin)
HOUSING = (0.025, 0.016, 0.008)                 # gripped plastic housing
TIP_X = HOUSING[0] / 2 + TIP_LEN                # housing centre -> tip end, along +x
CLEAR_Y, CLEAR_Z = 0.00025, 0.0003             # per side: USB-A receptacle 12.5 x 5.1 mm vs plug 12.0 x 4.5 mm
# USB retention springs: Coulomb friction on the plug while its tip is in the port (env physics, never
# exposed to the robot). USB-A spec: insertion <= 35 N, extraction >= 10 N; typical ~10-20 N.
PORT_FRICTION_N = 12.0
PORT_FRICTION_NM = 0.05
HAND_DOWN = gu.euler_to_quat((0.0, 180.0, 0.0))
TIP = 0.1034                                    # hand origin -> finger pad centre (approx)

# Laptop (41.6 x 28.2 x 2.4 cm base) on a 5 cm riser; its -x side faces the robot and carries the port.
LAP_C = np.array([0.62, 0.0])
RISER = 0.05
LAP = (0.416, 0.2824, 0.0244)
FACE_X = LAP_C[0] - LAP[0] / 2                  # port face plane
PORT = np.array([FACE_X, LAP_C[1] - 0.0541, RISER + 0.0190])     # port opening centre = the textured USB port
POCKET = 0.014                                  # port depth

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=args.dt, substeps=args.substeps),
    rigid_options=gs.options.RigidOptions(box_box_detection=True, noslip_iterations=args.noslip),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
                          material=gs.materials.Rigid(gravity_compensation=1.0))  # like the real controller's g(q) model


def add_fixed_box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    scene.add_entity(gs.morphs.Box(pos=tuple((lo + hi) / 2), size=tuple(hi - lo), fixed=True, visualization=False))


# Collision: riser + laptop base as 5 boxes leaving a (TIP_W+2c) x (TIP_H+2c) x POCKET pocket.
x0, x1 = FACE_X, FACE_X + LAP[0]
y0, y1 = LAP_C[1] - LAP[1] / 2, LAP_C[1] + LAP[1] / 2
z0, z1 = RISER, RISER + LAP[2]
hy, hz = TIP_W / 2 + CLEAR_Y, TIP_H / 2 + CLEAR_Z
scene.add_entity(gs.morphs.Box(pos=(LAP_C[0], LAP_C[1], RISER / 2), size=(LAP[0], LAP[1], RISER), fixed=True),
                 surface=gs.surfaces.Default(color=(0.35, 0.25, 0.18)))
add_fixed_box((x0 + POCKET, y0, z0), (x1, y1, z1))
add_fixed_box((x0, y0, z0), (x0 + POCKET, PORT[1] - hy, z1))
add_fixed_box((x0, PORT[1] + hy, z0), (x0 + POCKET, y1, z1))
add_fixed_box((x0, PORT[1] - hy, z0), (x0 + POCKET, PORT[1] + hy, PORT[2] - hz))
add_fixed_box((x0, PORT[1] - hy, PORT[2] + hz), (x0 + POCKET, PORT[1] + hy, z1))
# Visual: the textured laptop mesh (no collision) + a dark port opening drawn on its side.
scene.add_entity(gs.morphs.Mesh(file=f"{ASSETS}/laptop.obj", pos=(LAP_C[0], LAP_C[1], RISER), fixed=True,
                                collision=False))
scene.add_entity(gs.morphs.Box(pos=(FACE_X - 0.0004, PORT[1], PORT[2]), size=(0.0006, 2 * hy, 2 * hz), fixed=True,
                               collision=False), surface=gs.surfaces.Default(color=(0.02, 0.02, 0.02)))
table = scene.add_entity(gs.morphs.Box(pos=(0.5, 0.25, 0.05), size=(0.15, 0.15, 0.1), fixed=True))
plug = scene.add_entity(gs.morphs.URDF(file=f"{ASSETS}/usb_plug.urdf", pos=(0.3, -0.3, HOUSING[2] / 2)),
                        material=gs.materials.Rigid(friction=0.6))
cam = scene.add_camera(res=(640, 480), pos=(0.2, -0.2, 0.3), lookat=tuple(PORT), fov=40, GUI=False)
cam_side = scene.add_camera(res=(640, 480), pos=(FACE_X - 0.22, -0.12, 0.14), lookat=(FACE_X, 0.02, RISER + 0.01),
                            fov=50, GUI=False)

motors = np.arange(7)
fingers = np.arange(7, 9)
jt = scene.add_sensor(gs.sensors.JointTorque(entity_idx=franka.idx, dofs_idx_local=tuple(motors),
                                             noise=args.noise))
taxel_pos = gu.generate_grid_points_on_plane(lo=(-0.008, 0.0, 0.045), hi=(0.008, 0.0, 0.055),
                                             normal=(0.0, -1.0, 0.0), nx=4, ny=4)
taxels = [scene.add_sensor(gs.sensors.KinematicTaxel(
    entity_idx=franka.idx, link_idx_local=franka.get_link(n).idx_local,
    probe_local_pos=taxel_pos.reshape(-1, 3), probe_radius=0.003)) for n in ("left_finger", "right_finger")]

t0 = time.time()
scene.build()
print(f"build {time.time() - t0:.1f}s", flush=True)
franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 10, 10])

hand = franka.get_link("hand")
robot_links = [franka.get_link(n) for n in ("hand", "left_finger", "right_finger")]
GRIP_N = args.grip                # per-finger squeeze force (Franka Hand: up to 70 N continuous)


def np_(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def ik(pos):
    return franka.inverse_kinematics(link=hand, pos=pos, quat=HAND_DOWN)


_port_fric_on = [None]
# Genesis (like MuJoCo) scales constraint stiffness by the inertia of the bodies involved, so everything touching
# the 12 g plug (port walls, retention friction, finger contacts) is soft relative to a 2 kg arm: misaligned
# plugs squeeze through walls and frictionloss degrades to ~0.3 N of viscous drag. Standard fix: dof armature
# (extra inertia, NOT weight - the robot still feels 12 g) + a stiff joint constraint for the retention friction.
if args.armature > 0:
    plug.set_dofs_armature(np.array([args.armature] * 3 + [args.armature * 4e-4] * 3))
if args.stiff_fric:
    plug.joints[0].set_sol_params(np.array([2 * args.dt / args.substeps, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0]))


def port_physics():
    """USB retention: dry friction on the plug's free-joint dofs while the tip is inside the port."""
    tip_in = np_(plug.get_pos())[0] + TIP_X - FACE_X > 0.0005
    if tip_in != _port_fric_on[0]:
        f = [PORT_FRICTION_N] * 3 + [PORT_FRICTION_NM] * 3 if tip_in else [0.0] * 6
        plug.set_dofs_frictionloss(np.array(f))
        _port_fric_on[0] = tip_in


def drive(q_arm, grip, steps, force=None):
    """Position-control the arm. Fingers: position `grip`, or squeeze with `force` N if given."""
    for _ in range(steps):
        port_physics()
        franka.control_dofs_position(q_arm[:7], motors)
        if force is None:
            franka.control_dofs_position(np.array([grip, grip]), fingers)
        else:
            franka.control_dofs_force(np.array([-force, -force]), fingers)
        scene.step()


def true_contact_force():
    f = np_(franka.get_links_net_contact_force())
    return sum(f[l.idx_local] for l in robot_links)


def wrench_from_torque(tau, tau0):
    """libfranka-style: F_ext = -pinv(J^T) (tau - tau_model). tau_model = the quasi-static
    reading in free space just before contact (stands in for the gravity model). Returns
    [Fx, Fy, Fz, Mx, My, Mz] in the world frame, moment about the hand origin."""
    J = np_(franka.get_jacobian(hand))[:, :7]
    return -np.linalg.pinv(J.T) @ (tau - tau0)


def taxel_summary():
    out = []
    for s in taxels:
        r = s.read()
        f = np_(r.force).reshape(-1, 3)
        out.append((round(float(np.linalg.norm(f, axis=1).max()), 2), np.round(f.sum(0), 2)))
    return out


def grip_width():
    return float(np_(franka.get_dofs_position(fingers)).sum())


def set_arm(pos, grip):
    q = np_(ik(pos))
    q[7:] = grip
    franka.set_qpos(q)
    drive(q, grip, 20)
    return q


def exp_press():
    """Descend in 0.5 mm steps from 2 cm above the table: when does each signal see contact?"""
    z0 = 0.1 + TIP + 0.02
    set_arm([0.5, 0.25, z0], 0.0)
    drive(np_(ik([0.5, 0.25, z0])), 0.0, 100)
    tau0 = np_(jt.read()).copy()
    first_true = first_est = None
    for k in range(80):
        z = z0 - 0.0005 * (k + 1)
        drive(np_(ik([0.5, 0.25, z])), 0.0, 20)
        F = true_contact_force()
        W = wrench_from_torque(np_(jt.read()), tau0)
        if first_true is None and abs(F[2]) > 0.5:
            first_true = k
        if first_est is None and W[2] > 3.0:
            first_est = k
        if first_true is not None and (k - first_true) % 2 == 0 and k - first_true <= 12:
            print(f"  press {0.5*(k-first_true):.1f}mm past contact: true Fz {F[2]:6.2f}  est Fz {W[2]:6.2f} "
                  f"(est Fxy {np.round(W[:2], 2)})", flush=True)
        if first_true is not None and k - first_true > 12:
            break
    print(f"RESULT press: first true contact at step {first_true}, est Fz>3N at step {first_est} "
          f"(0.5 mm steps; tau noise {args.noise} Nm)", flush=True)


def grasp_plug(force=GRIP_N):
    """Teleport the plug (axis +x) between the pre-set fingers in free space, then squeeze."""
    start = np.array([0.3, -0.15, 0.25])
    # Fingers close along world y (hand pointing down); pre-set them to the housing width so the
    # teleported plug is held at once (kp-100 fingers are too slow to catch a falling plug).
    q = set_arm(start, HOUSING[1] / 2 + 0.0003)
    finger_mid = (np_(franka.get_link("left_finger").get_pos()) + np_(franka.get_link("right_finger").get_pos())) / 2
    plug.set_pos(finger_mid - np.array([0, 0, 0.045]))
    plug.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    plug.set_dofs_velocity(np.zeros(6))
    drive(q, 0.0, 80, force=force)
    rel = np_(plug.get_pos()) - np_(hand.get_pos())
    print(f"  grasp {force:.0f}N: width {grip_width()*1000:.1f}mm plug rel hand {np.round(rel, 4)}", flush=True)
    return q, rel


def hand_target_for_tip(tip_xyz, rel):
    """Hand position that puts the plug's tip end at tip_xyz (nominal in-hand offset `rel`)."""
    return np.asarray(tip_xyz) - rel - np.array([TIP_X, 0, 0])


def tip_end_pose():
    R = gu.quat_to_R(np_(plug.get_quat()))
    return np_(plug.get_pos()) + R @ np.array([TIP_X, 0, 0]), R


def drive_impedance(p_des, R_des, steps, K=1500.0, Kr=40.0, q_rest=None, log_every=0, push_dir=None, K_lat=2000.0,
                    tcp_local=None):
    """Cartesian impedance at the hand (gravity is compensated by the robot model, like libfranka's
    cartesian_impedance example): tau = J^T [K dp - D v ; Kr dR - Dr w] + null-space posture hold.
    The contact force the robot feels is the spring deflection K (p_des - p) at equilibrium.
    With push_dir, stiffness is K along it and K_lat across it (compliant push, stiff lateral hold).
    tcp_local (hand frame) moves the compliance centre and p/p_des to the tool tip, like setting
    libfranka's EE frame to the TCP: a push at the tip then does not tilt the hand."""
    if push_dir is None:
        Km, Dm = K * np.eye(3), 2 * np.sqrt(K * 2.0) * np.eye(3)
    else:
        P = np.outer(push_dir, push_dir)
        Km = K * P + K_lat * (np.eye(3) - P)
        Dm = 2 * np.sqrt(K * 2.0) * P + 2 * np.sqrt(K_lat * 2.0) * (np.eye(3) - P)
    Dr = 2 * np.sqrt(Kr * 0.02)
    out = []
    for k in range(steps):
        port_physics()
        p, R = np_(hand.get_pos()), gu.quat_to_R(np_(hand.get_quat()))
        v, w = np_(hand.get_vel()), np_(hand.get_ang())
        if tcp_local is not None:
            r = R @ tcp_local
            p, v = p + r, v + np.cross(w, r)
        e_r = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
        wrench = np.concatenate([Km @ (p_des - p) - Dm @ v, Kr * e_r - Dr * w])
        J = np_(franka.get_jacobian(hand, local_point=None if tcp_local is None else
                                    torch.tensor(tcp_local, dtype=torch.float32)))[:, :7]
        q, dq = np_(franka.get_dofs_position(motors)), np_(franka.get_dofs_velocity(motors))
        tau = J.T @ wrench
        if q_rest is not None:
            N = np.eye(7) - J.T @ np.linalg.pinv(J.T)
            tau += N @ (10.0 * (q_rest - q) - 2.0 * dq)
        franka.control_dofs_force(tau, motors)
        franka.control_dofs_force(np.array([-GRIP_N, -GRIP_N]), fingers)
        scene.step()
        if log_every and k % log_every == 0:
            out.append((k, K * (p_des - p)))
    return out


def exp_push(cases):
    """Impedance pushes into the port. Each case: (label, K N/m, push distance past the aim point
    along the push direction, push-direction yaw deg, plug-axis yaw deg, lateral dy m)."""
    R_down = gu.quat_to_R(np_(HAND_DOWN))
    for label, K, dist, dir_yaw, axis_yaw, dy in cases:
        q, rel = grasp_plug()
        # Perfect-perception alignment (probe baseline only; the robot gets no sim truth): put the
        # real tip end 3 mm in front of the port, nudging the hand by the measured tip error.
        Ryaw = gu.quat_to_R(np_(gu.euler_to_quat((0.0, 0.0, axis_yaw))))
        R_des = Ryaw @ R_down
        aim = PORT + np.array([-0.003, dy, 0.0])
        hand_t = hand_target_for_tip(aim, rel)
        for _ in range(6):
            qd = np_(franka.inverse_kinematics(link=hand, pos=hand_t, quat=gu.R_to_quat(R_des), pos_tol=1e-5,
                                               rot_tol=1e-4, max_solver_iters=100))
            drive(qd, 0.0, 120, force=GRIP_N)
            tip, _ = tip_end_pose()
            hand_t = hand_t + (aim - tip)
        tip0, Rp = tip_end_pose()
        q_rest = np_(franka.get_dofs_position(motors))
        d = np.array([np.cos(np.radians(dir_yaw)), np.sin(np.radians(dir_yaw)), 0.0])
        p0 = np_(hand.get_pos())
        def trace(t):
            tp, _ = tip_end_pose()
            hp = np_(hand.get_pos())
            slip = (np_(plug.get_pos()) - hp)[0] - rel[0]
            print(f"    t {t:4.2f}s depth {(tp[0]-FACE_X)*1000:6.2f}mm  hand dx {(hp-p0)[0]*1000:6.2f}mm  "
                  f"felt Fx {K*(sp[0]-hp[0]):6.1f}N  plug slip-in-grip {slip*1000:+.2f}mm", flush=True)
        for i, s in enumerate(np.linspace(0, 1, 9)[1:]):   # ramp the set-point over 0.8 s, then hold 1.2 s
            sp = p0 + d * dist * s
            drive_impedance(sp, R_des, 20, K=K, q_rest=q_rest, push_dir=d)
            if i % 2 == 1:
                trace((i + 1) * 0.1)
        for j in range(3):
            drive_impedance(sp, R_des, 80, K=K, q_rest=q_rest, push_dir=d)
            trace(0.8 + (j + 1) * 0.4)
        tip, Rp = tip_end_pose()
        depth = tip[0] - FACE_X
        P = np.outer(d, d)
        F_felt = (K * P + 2000.0 * (np.eye(3) - P)) @ (p0 + d * dist - np_(hand.get_pos()))
        print(f"RESULT push {label}: K {K:.0f} dist {dist*1000:.0f}mm dir_yaw {dir_yaw} axis_yaw {axis_yaw} dy "
              f"{dy*1000:+.1f}mm | start tip err {np.round((tip0 - aim)*1000, 2)}mm | depth {depth*1000:5.1f}mm "
              f"({'IN' if depth > 0.010 else 'not in'}) | felt F {np.round(F_felt, 1)} N", flush=True)
        # release and back off so the next case starts clean
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers)
        drive(np_(ik(np_(hand.get_pos()) + np.array([-0.05, 0, 0.05]))), 0.04, 100)


def imp_gains(K, push_dir, K_lat):
    P = np.outer(push_dir, push_dir)
    return K * P + K_lat * (np.eye(3) - P), 2 * np.sqrt(K * 2.0) * P + 2 * np.sqrt(K_lat * 2.0) * (np.eye(3) - P)


def place_tip(aim, R_des, rel):
    """Scenario set-up only: put the real tip end at `aim` (which already contains the perception
    error) using sim truth, so every method starts from the same, known misalignment."""
    hand_t = hand_target_for_tip(aim, rel)
    for _ in range(6):
        qd = np_(franka.inverse_kinematics(link=hand, pos=hand_t, quat=gu.R_to_quat(R_des), pos_tol=1e-5,
                                           rot_tol=1e-4, max_solver_iters=100))
        drive(qd, 0.0, 120, force=GRIP_N)
        tip, _ = tip_end_pose()
        hand_t = hand_t + (aim - tip)


def search_trial(method, ey, ez, yaw, rng, verbose=False):
    """One insertion attempt from a misaligned start. The robot only believes 'the port is straight
    ahead along +x'; (ey, ez, yaw) is the unknown error of its vision-based alignment."""
    from touch import InsertConfig, SpiralInsertController, TouchState, run_reactive

    q, rel = grasp_plug()
    R_des = gu.quat_to_R(np_(gu.euler_to_quat((0.0, 0.0, yaw)))) @ gu.quat_to_R(np_(HAND_DOWN))
    place_tip(PORT + np.array([-args.standoff, ey, ez]), R_des, rel)
    q_rest = np_(franka.get_dofs_position(motors))
    axis = np.array([1.0, 0.0, 0.0])
    # TCP = nominal plug tip from the planned grasp (fingertip centre + plug length along the believed
    # axis), in the hand frame. The robot does not know the yaw error, so it uses the unyawed geometry.
    tcp_local = gu.quat_to_R(np_(HAND_DOWN)).T @ np.array([TIP_X, 0.0, -TIP]) if args.tcp else None

    def ee():
        p, R = np_(hand.get_pos()), gu.quat_to_R(np_(hand.get_quat()))
        v, w = np_(hand.get_vel()), np_(hand.get_ang())
        if tcp_local is None:
            return p, v, R
        r = R @ tcp_local
        return p + r, v + np.cross(w, r), R

    p0, _, R0 = ee()
    rel0 = R0.T @ (np_(plug.get_pos()) - np_(hand.get_pos()))
    st = {"t": 0.0, "p_des": p0.copy(), "K": 1500.0, "K_lat": args.k_lat}

    def advance(period):
        n = max(1, int(round(period / args.dt)))
        drive_impedance(st["p_des"], R_des, n, K=st["K"], Kr=args.kr, q_rest=q_rest, push_dir=axis, K_lat=st["K_lat"],
                        tcp_local=tcp_local)
        st["t"] += n * args.dt

    def read_state():
        # Proprioception only: EE pose (FK) and the external force the impedance controller is resisting
        # (what tau_J-based O_F_ext_hat_K reports quasi-statically), plus sensor noise.
        p, v, _ = ee()
        Km, Dm = imp_gains(st["K"], axis, st["K_lat"])
        f = -(Km @ (st["p_des"] - p) - Dm @ v) + rng.normal(0.0, args.fnoise, 3)
        return TouchState(t=st["t"], ee_pos=p, force=f, gripper_width=grip_width())

    if method == "naive":
        for s in np.linspace(0, 1, 9)[1:]:
            st["p_des"] = p0 + axis * 0.030 * s
            advance(0.1)
        advance(1.2)
        res = {"outcome": "pushed", "duration_s": st["t"], "events": []}
    else:
        def send(cmd):
            st["p_des"], st["K"] = np.asarray(cmd.ee_target, float), cmd.k_axial
            st["K_lat"] = cmd.k_lateral or args.k_lat
        ctrl = SpiralInsertController(axis, InsertConfig(f_search=args.f_search), log=(lambda m: print("   ", m, flush=True)) if verbose else None)
        def debug_tick(ts, cmd):   # eval-only trace (sim truth), printed for debugging, never fed back
            if verbose and args.trace and int(round(ts.t / 0.02)) % 10 == 0:
                tp, _ = tip_end_pose()
                Rh = gu.quat_to_R(np_(hand.get_quat()))
                sl = 1000 * np.linalg.norm(Rh.T @ (np_(plug.get_pos()) - np_(hand.get_pos())) - rel0)
                print(f"      trace t {ts.t:5.2f} {cmd.phase:8s} F_ax {ts.axial_resistance(axis):5.1f} "
                      f"ee_adv {1000 * (ts.ee_pos - p0)[0]:6.2f} lat_cmd {np.round(1000 * (cmd.ee_target - p0)[1:], 2)} "
                      f"| true depth {1000 * (tp[0] - FACE_X):6.2f} tip_yz_err {np.round(1000 * (tp[1:] - PORT[1:]), 2)} "
                      f"slip {sl:.2f}", flush=True)
        res = run_reactive(ctrl, read_state, send, advance, max_s=args.max_s, period_s=0.02, on_tick=debug_tick)
    tip, _ = tip_end_pose()
    depth = tip[0] - FACE_X                                               # eval only (sim truth)
    R1 = gu.quat_to_R(np_(hand.get_quat()))
    slip = 1000 * np.linalg.norm(R1.T @ (np_(plug.get_pos()) - np_(hand.get_pos())) - rel0)
    tilt = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
    ok = depth > 0.010
    print(f"RESULT search {method:5s} err dy {ey*1000:+.2f} dz {ez*1000:+.2f}mm yaw {yaw:+.1f}deg | "
          f"outcome {res['outcome']:16s} t {res['duration_s']:5.2f}s | true depth {depth*1000:5.1f}mm "
          f"({'IN' if ok else 'not in'}) | in-grip slip {slip:.1f}mm hand tilt {tilt:.1f}deg", flush=True)
    franka.control_dofs_position(np.array([0.04, 0.04]), fingers)
    drive(np_(ik(np_(hand.get_pos()) + np.array([-0.05, 0, 0.05]))), 0.04, 100)
    return ok, res


def exp_search(n, seed, methods):
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1000)   # separate stream: trial errors stay fixed
    stats = {m: [] for m in methods}
    for i in range(n):
        # Relative plug-to-port alignment error after wrist-camera servoing: uniform in a 2 mm disc, +-2 deg yaw.
        r, ang = args.err_r * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        ey, ez, yaw = r * np.cos(ang), r * np.sin(ang), rng.uniform(-args.err_yaw, args.err_yaw)
        if i == 0 and args.case:
            ey, ez, yaw = (float(v) for v in args.case.split(","))
        for m in methods:
            ok, res = search_trial(m, ey, ez, yaw, noise_rng, verbose=(i < args.verbose))
            stats[m].append((ok, res["duration_s"], res["outcome"]))
    for m, s in stats.items():
        ok = [x for x in s if x[0]]
        print(f"SUMMARY {m}: {len(ok)}/{len(s)} inserted; mean time (successes) "
              f"{np.mean([x[1] for x in ok]) if ok else float('nan'):.2f}s; outcomes "
              f"{sorted({o: sum(1 for x in s if x[2] == o) for o in {x[2] for x in s}}.items())}", flush=True)


def exp_creep(loads):
    """Hold a constant axial push (-x, like the port face pushing back) on the held plug for 2 s, arm under
    position control. A Coulomb grasp would not move below ~2*mu*F_grip; soft contacts creep."""
    pl = plug.get_link("plug")
    for F in loads:
        q, rel0 = grasp_plug()
        trace = []
        for k in range(400):
            pl.apply_external_force(np.array([-F, 0.0, 0.0]))
            franka.control_dofs_position(q[:7], motors)
            franka.control_dofs_force(np.array([-GRIP_N, -GRIP_N]), fingers)
            scene.step()
            if k % 100 == 99:
                trace.append(round(1000 * ((np_(plug.get_pos()) - np_(hand.get_pos()))[0] - rel0[0]), 2))
        print(f"RESULT creep grip {GRIP_N:.0f}N load {F:4.1f}N: slip mm at 0.5/1/1.5/2 s {trace}; "
              f"width {grip_width()*1000:.1f}mm", flush=True)


def exp_slot(offsets, render_only=False):
    for off in offsets:
        dy, dz = (float(v) for v in off.split(","))
        q, rel = grasp_plug()
        # approach from 6 cm, then hover with the tip 8 mm in front of the face
        aim = PORT + np.array([0.0, dy, dz])
        for d in (0.06, 0.008):
            drive(np_(ik(hand_target_for_tip(aim - np.array([d, 0, 0]), rel))), 0.0, 120, force=GRIP_N)
        hover = hand_target_for_tip(aim - np.array([0.008, 0, 0]), rel)
        if render_only:
            import imageio.v2 as iio
            rgb = np.concatenate([np.asarray(c.render(rgb=True)[0]) for c in (cam, cam_side)], axis=1)
            iio.imwrite(args.img, rgb)
            print(f"saved {args.img}", flush=True)
            return
        rel_h = np_(plug.get_pos()) - np_(hand.get_pos())
        R = gu.quat_to_R(np_(plug.get_quat()))
        tip_end = np_(plug.get_pos()) + R @ np.array([TIP_X, 0, 0])
        print(f"  hover: hand err {np.round((np_(hand.get_pos()) - hover) * 1000, 2)}mm; tip end - aim "
              f"{np.round((tip_end - (aim - np.array([0.008, 0, 0]))) * 1000, 2)}mm; plug axis {np.round(R[:, 0], 4)} "
              f"(tilt {np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1))):.2f} deg, yaw {np.degrees(np.arctan2(R[1, 0], R[0, 0])):.2f} deg)",
              flush=True)
        tau0 = np_(jt.read()).copy()
        trace = []
        for k in range(26):  # advance 1 mm per command along +x, 25 steps (0.125 s) each
            drive(np_(ik(hover + np.array([0.001 * (k + 1), 0, 0]))), 0.0, 25, force=GRIP_N)
            W = wrench_from_torque(np_(jt.read()), tau0)
            depth = np_(plug.get_pos())[0] + TIP_X - FACE_X          # eval only (sim truth)
            trace.append((k + 1, depth, W))
        Fx = np.array([t[2][0] for t in trace])
        k_contact = next((t[0] for t in trace if t[2][0] < -2.0), None)
        last = trace[-1]
        # moment about the (nominal) tip end, from the hand-origin wrench: M_tip = M_hand - r x F
        r = rel + np.array([TIP_X, 0, 0])
        M_tip = last[2][3:] - np.cross(r, last[2][:3])
        slide = 1000 * ((np_(plug.get_pos()) - np_(hand.get_pos()))[0] - rel_h[0])
        print(f"RESULT slot dy {dy*1000:+.1f} dz {dz*1000:+.1f}mm: depth {last[1]*1000:5.1f}mm (in>=10); "
              f"first Fx<-2N at advance {k_contact}mm; final F {np.round(last[2][:3], 1)} "
              f"M_tip {np.round(M_tip, 3)}; min Fx {Fx.min():.1f}; plug slid in fingers {slide:.1f}mm", flush=True)
        for k, depth, W in trace[::3]:
            print(f"    adv {k:2d}mm depth {depth*1000:5.1f} F {np.round(W[:3], 1)} M {np.round(W[3:], 2)}", flush=True)


def exp_grip():
    """Ramp an external pull (-x, along the plug axis) on the held plug: at what force does it slip?"""
    q, rel0 = grasp_plug()
    pl = plug.get_link("plug")
    fl = np_(franka.get_links_net_contact_force())
    print(f"  finger contact F: left {np.round(fl[franka.get_link('left_finger').idx_local], 1)} right "
          f"{np.round(fl[franka.get_link('right_finger').idx_local], 1)}; plug friction {plug.geoms[0].friction:.2f}, "
          f"finger friction {franka.get_link('left_finger').geoms[0].friction:.2f}", flush=True)
    slip_at = None
    for k in range(41):
        F = float(k)
        for _ in range(10):
            pl.apply_external_force(np.array([-F, 0.0, 0.0]))
            franka.control_dofs_position(q[:7], motors)
            franka.control_dofs_force(np.array([-GRIP_N, -GRIP_N]), fingers)
            scene.step()
        dx = 1000 * ((np_(plug.get_pos()) - np_(hand.get_pos()))[0] - rel0[0])
        if slip_at is None and dx < -1.0:
            slip_at = F
        if k % 5 == 0:
            print(f"  pull {F:4.0f}N: plug moved {dx:6.1f}mm in fingers", flush=True)
    print(f"RESULT grip {GRIP_N:.0f}N/finger: plug slips at ~{slip_at} N axial pull", flush=True)


def exp_slip():
    for grip in (GRIP_N, 2.0):  # strong vs weak squeeze (N per finger)
        q, rel0 = grasp_plug(grip)
        hover = np.array([0.5, 0.25, 0.1 + HOUSING[2] / 2 - rel0[2] + 0.005])
        drive(np_(ik(hover)), 0.0, 150, force=grip)
        tau0 = np_(jt.read()).copy()
        for k in range(20):
            target = hover - np.array([0, 0, 0.001 * (k + 1)])
            drive(np_(ik(target)), 0.0, 25, force=grip)
            rel = np_(plug.get_pos()) - np_(hand.get_pos())
            W = wrench_from_torque(np_(jt.read()), tau0)
            if k % 4 == 3:
                print(f"  grip {grip:.0f}N push {k+1}mm: plug slid {1000*(rel[2]-rel0[2]):.1f}mm in fingers, "
                      f"est Fz {W[2]:.1f}, width {1000*grip_width():.1f}mm, taxels {taxel_summary()}", flush=True)
        print(f"RESULT slip grip {grip:.0f}N: slid {1000*(rel[2]-rel0[2]):.1f}mm; final est Fz {W[2]:.1f}", flush=True)


t0 = time.time()
if args.exp == "render":
    exp_slot(["0,0"], render_only=True)
if args.exp in ("press", "all"):
    exp_press()
if args.exp in ("slot", "all"):
    exp_slot(args.offsets)
if args.exp in ("push", "all"):
    exp_push([("light", 250, 0.030, 0, 0, 0.0)] if args.quick else [("light", 250, 0.030, 0, 0, 0.0), ("medium", 600, 0.030, 0, 0, 0.0), ("firm", 1500, 0.030, 0, 0, 0.0),
              ("firm_dir15", 1500, 0.030, 15, 0, 0.0), ("firm_dir30", 1500, 0.030, 30, 0, 0.0),
              ("firm_axis4", 1500, 0.030, 0, 4, 0.0), ("firm_axis4_dir4", 1500, 0.030, 4, 4, 0.0),
              ("firm_axis8", 1500, 0.030, 0, 8, 0.0), ("firm_dy1.0", 1500, 0.030, 0, 0, 0.0010),
              ("firm_dy1.5", 1500, 0.030, 0, 0, 0.0015)])
if args.exp == "grip":
    exp_grip()
if args.exp == "creep":
    exp_creep([4.0, 8.0, 12.0, 18.0])
if args.exp == "search":
    exp_search(args.trials, args.seed, args.methods)
if args.exp in ("slip", "all"):
    exp_slip()
print(f"sim wall time {time.time() - t0:.1f}s (dt {args.dt}, substeps {args.substeps})", flush=True)
