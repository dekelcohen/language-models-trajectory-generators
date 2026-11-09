"""Plug-insertion task (Genesis): pick a plug off its pedestal and insert it into a port on a laptop.

One code path for several connector types (``CONNECTORS``: USB-A, USB-C, HDMI, a square peg) and a
randomisable layout, so skills can't overfit one geometry. Default (no env vars) = the original
bench scene: USB-A port on the laptop's left (-x) face, plug on a pedestal pointing at it.

Perturbation knobs (environment variables, read when the env is created):
  PLUG_TYPE         usb_a | usb_c | hdmi | peg                     (default usb_a)
  PORT_FACE         left | right | back  (laptop's own sides)       (default left)
  PORT_DIR_DEG      world heading of the port face's outward normal (default 180 = -x); the
                    laptop is turned to match, the port stays in the robot's reach
  PORT_U            port position along its face, m from the face centre (default per face)
  PORT_DISTRACTOR   a second, EMPTY port of another type beside the target (e.g. hdmi); a wrong
                    plug enters it only ~3 mm (its backstop)
  PLUG_YAW_DEG      plug heading offset from "tip toward the port" (default 0)
  PLUG_SCENE_SEED   if set: random face, heading, position along the face, distractor and plug
                    heading (explicit knobs above still win). Random headings face sideways (reach)
                    and toward the fixed head camera (config.head_camera_position), >= 15 deg off
                    edge-on, so the port face is in its depth; the wrist camera sees it up close.

Geometry and contact physics: the ones measured on the bench (research/touch/genesis_touch_probe.py,
docs/plans/touch.md section 3). The robot gets nothing from here except camera images; port pose
and depth are evaluation only. Genesis-only: needs ``sim.native_entity`` / ``run_after_build``.
"""

import math
import os
import tempfile

import numpy as np

import config
from sim_envs.pybullet.base import SimEnvBase

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "usb_laptop")

#: Solver settings this scene needs (passed to GenesisAdapter(scene_options=...) before the scene
#: exists): 2 substeps, box-box contacts and noslip, or a 12 g plug squeezes through a 0.3 mm gap.
SCENE_OPTIONS = {"substeps": 2, "rigid": {"box_box_detection": True, "noslip_iterations": 5}}

#: Connector models: tip (len, width, height) inserted along the plug's +x; housing (x, y, z) gripped;
#: clearance (width, height) per side; retention (N, N*m) dry friction while the tip is inside;
#: port_z = port centre above the laptop's bottom; seat = depth (m) that counts as seated.
CONNECTORS = {
    "usb_a": dict(label="USB-A", tip=(0.012, 0.012, 0.0045), housing=(0.025, 0.016, 0.008),
                  clear=(0.00025, 0.0003), retention=(12.0, 0.05), port_z=0.019, seat=0.010, mass=0.012),
    "usb_c": dict(label="USB-C", tip=(0.0065, 0.0084, 0.0026), housing=(0.022, 0.012, 0.0065),
                  clear=(0.0002, 0.0002), retention=(8.0, 0.03), port_z=0.0125, seat=0.005, mass=0.008),
    "hdmi": dict(label="HDMI", tip=(0.009, 0.014, 0.0045), housing=(0.030, 0.020, 0.010),
                 clear=(0.0003, 0.0003), retention=(15.0, 0.06), port_z=0.0125, seat=0.007, mass=0.020),
    "peg": dict(label="square peg", tip=(0.020, 0.010, 0.010), housing=(0.030, 0.016, 0.016),
                clear=(0.0004, 0.0004), retention=(4.0, 0.02), port_z=0.0122, seat=0.015, mass=0.020),
}

# Backward-compatible module constants (USB-A, default layout)
TIP_LEN, TIP_W, TIP_H = CONNECTORS["usb_a"]["tip"]
HOUSING = CONNECTORS["usb_a"]["housing"]
TIP_X = HOUSING[0] / 2 + TIP_LEN                # plug origin (housing centre) -> tip end, along plug +x
CLEAR_Y, CLEAR_Z = CONNECTORS["usb_a"]["clear"]
PORT_FRICTION_N, PORT_FRICTION_NM = CONNECTORS["usb_a"]["retention"]
PLUG_FRICTION = 0.6
#: Finger-pad patch torsional friction (~mu * F_grip * r_patch = 0.6 * 20 N * 4 mm). The URDF Franka's mesh
#: fingers touch the housing at near-points, so without this the held plug pivots freely about the finger
#: axis and creeps 10-20 deg during a carry. Far below jam torques (5 N x 25 mm), so the port still aligns it.
PAD_TORSION_NM = 0.05
FREE_FRICTION = [0.0] * 3 + [PAD_TORSION_NM] * 3
GRIP_KP = 2500.0                                # finger position gain for this task (see configure_robot)
PLUG_ARMATURE = 0.5                             # conditioning only (not weight), see the bench notes

LAP_C = np.array([0.30, 0.55])                  # default laptop centre
RISER = 0.05
LAP = (0.416, 0.2824, 0.0244)                   # laptop (local x, y, z); screen at local +y
FACE_X = LAP_C[0] - LAP[0] / 2
PORT = np.array([FACE_X, LAP_C[1] - 0.0541, RISER + 0.0190])
PORT_XY0 = PORT[:2].copy()                      # every layout puts the port opening here (+ PORT_SHIFT)
POCKET = 0.014
PED_H = 0.05
PLUG_STANDOFF = (0.212, 0.0459)                 # plug pedestal: along the port normal, then toward the robot (-y)
SEATED_DEPTH = CONNECTORS["usb_a"]["seat"]
DISTRACTOR_BACKSTOP = 0.003                     # a distractor port admits a foreign plug only this deep

#: laptop faces: (normal axis, outward sign, along-face axis, default along-face offset of the port)
FACES = {"left": (0, -1, 1, -0.0541), "right": (0, 1, 1, -0.0541), "back": (1, 1, 0, 0.10)}


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _env(name, cast=str, default=None):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


def scene_layout(env=None, seed=None):
    """Resolve the knobs (module docstring) into a layout dict: connector, face, laptop pose, port pose
    (world), insertion axis, distractor, plug pose. Pure; ``env`` defaults to os.environ."""
    env = os.environ if env is None else env
    get = lambda k, cast=str, d=None: d if env.get(k) in (None, "") else cast(env.get(k))  # noqa: E731
    seed = get("PLUG_SCENE_SEED", int) if seed is None else seed
    rng = np.random.default_rng(seed) if seed is not None else None
    ctype = get("PLUG_TYPE", str, "usb_a").lower().replace("-", "_")
    if ctype not in CONNECTORS:
        raise ValueError(f"PLUG_TYPE {ctype!r}: one of {sorted(CONNECTORS)}")
    face = get("PORT_FACE", str, None) or (str(rng.choice(list(FACES))) if rng is not None else "left")
    k, s, j, u0 = FACES[face]
    half_u = LAP[j] / 2
    if rng is not None:
        # sideways (reach) AND turned toward the fixed head camera (>= 15 deg off edge-on)
        lo, hi = (130.0, 172.0) if rng.random() < 0.5 else (22.0, 60.0)
        dir_deg = get("PORT_DIR_DEG", float, float(rng.uniform(lo, hi)))
        u = get("PORT_U", float, float(rng.uniform(-half_u + 0.05, half_u - 0.05)))
        others = [t for t in CONNECTORS if t != ctype]
        distractor = get("PORT_DISTRACTOR", str, str(rng.choice(others)) if rng.random() < 0.5 else None)
        plug_yaw = get("PLUG_YAW_DEG", float, float(rng.uniform(-60, 60)))
    else:
        dir_deg = get("PORT_DIR_DEG", float, 180.0)
        u = get("PORT_U", float, u0)
        distractor = get("PORT_DISTRACTOR", str, None)
        plug_yaw = get("PLUG_YAW_DEG", float, 0.0)
    distractor = (distractor or "").lower().replace("-", "_") or None
    if distractor and (distractor not in CONNECTORS or distractor == ctype):
        raise ValueError(f"PORT_DISTRACTOR {distractor!r}: another of {sorted(CONNECTORS)}")
    spec = CONNECTORS[ctype]

    # laptop-local port (centre of the opening on the face) and outward normal
    n_loc = np.zeros(2); n_loc[k] = s
    t_loc = np.zeros(2); t_loc[j] = 1.0
    port_loc = n_loc * LAP[k] / 2 + t_loc * u
    # laptop yaw so the local normal points along dir_deg; laptop centre so the port lands at PORT_XY0
    yaw = math.radians(dir_deg) - math.atan2(n_loc[1], n_loc[0])
    R = _rot(yaw)
    port_xy = PORT_XY0.copy()
    centre = port_xy - R @ port_loc
    n_w = R @ n_loc
    t_w = R @ t_loc
    ports = [dict(type=ctype, u=u, target=True)]
    if distractor:
        dw = CONNECTORS[distractor]["tip"][1] + spec["tip"][1]
        du = 0.5 * dw + 0.012
        du = du if u + du + 0.02 < half_u else -du        # beside the target, inside the face
        ports.append(dict(type=distractor, u=u + du, target=False))
    for pt in ports:
        sp = CONNECTORS[pt["type"]]
        pl = n_loc * LAP[k] / 2 + t_loc * pt["u"]
        pt["xyz"] = [*(centre + R @ pl), RISER + sp["port_z"]]
        pt["label"] = sp["label"]
    cam = np.asarray(config.head_camera_position, float)
    v = cam[:2] - port_xy
    head_view_deg = math.degrees(math.asin(max(-1.0, min(1.0, float(v @ n_w) / np.linalg.norm(v)))))
    plug_xy = port_xy + PLUG_STANDOFF[0] * n_w + np.array([0.0, -PLUG_STANDOFF[1]])
    plug_heading = math.atan2(-n_w[1], -n_w[0]) + math.radians(plug_yaw)
    return dict(type=ctype, spec=spec, face=face, dir_deg=dir_deg, u=u, laptop_xy=centre, laptop_yaw=yaw,
                k=k, s=s, j=j, n_world=np.array([*n_w, 0.0]), t_world=np.array([*t_w, 0.0]),
                port=np.array(ports[0]["xyz"]), ports=ports, distractor=distractor,
                plug_xy=plug_xy, plug_yaw=plug_heading, seed=seed, head_view_deg=head_view_deg)


def _box(name, lo, hi, rgba=None, collision=True, visual=True):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, s = (lo + hi) / 2, hi - lo
    geo = (f'<origin xyz="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}"/>'
           f'<geometry><box size="{s[0]:.5f} {s[1]:.5f} {s[2]:.5f}"/></geometry>')
    out = ""
    if visual:
        col = rgba or (0.35, 0.25, 0.18, 1)
        out += (f'<visual name="{name}">{geo}<material name="m_{name}"><color rgba="'
                f'{" ".join(f"{v:.3f}" for v in col)}"/></material></visual>')
    if collision:
        out += f'<collision name="{name}">{geo}</collision>'
    return out


def _urdf(name, parts):
    return (f'<?xml version="1.0"?>\n<robot name="{name}"><link name="{name}">'
            '<inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
            + "".join(parts) + "</link></robot>\n")


def scene_urdf(layout=None):
    """Laptop in its OWN frame (load at ``laptop_xy`` / ``laptop_yaw``): riser, body collision boxes
    with a pocket slab cut around each port on the chosen face, laptop mesh (visual), dark openings."""
    L = layout or scene_layout(env={})
    k, s, j = L["k"], L["s"], L["j"]
    half = np.array([LAP[0] / 2, LAP[1] / 2])
    z0, z1 = RISER, RISER + LAP[2]
    pd = max(POCKET, max(CONNECTORS[p["type"]]["tip"][0] for p in L["ports"]) + 0.002)

    def box3(name, ka, ja, za, **kw):     # ranges along the normal axis k, the face axis j, and z
        lo, hi = [0.0, 0.0, za[0]], [0.0, 0.0, za[1]]
        lo[k], hi[k] = ka
        lo[j], hi[j] = ja
        return _box(name, lo, hi, **kw)

    face = s * half[k]
    slab = (face - pd, face) if s > 0 else (face, face + pd)
    rest = (-half[k], face - pd) if s > 0 else (face + pd, half[k])
    full_j = (-half[j], half[j])
    parts = [_box("riser", (-half[0], -half[1], 0), (half[0], half[1], RISER)),
             box3("body", rest, full_j, (z0, z1), visual=False)]
    holes = []
    for p in L["ports"]:
        sp = CONNECTORS[p["type"]]
        hw, hh = sp["tip"][1] / 2 + sp["clear"][0], sp["tip"][2] / 2 + sp["clear"][1]
        holes.append((p["u"] - hw, p["u"] + hw, RISER + sp["port_z"] - hh, RISER + sp["port_z"] + hh, p))
    holes.sort(key=lambda h: h[0])
    edge = -half[j]
    for i, (a, b, lo, hi, p) in enumerate(holes):
        parts.append(box3(f"seg{i}", slab, (edge, a), (z0, z1), visual=False))
        parts.append(box3(f"below{i}", slab, (a, b), (z0, lo), visual=False))
        parts.append(box3(f"above{i}", slab, (a, b), (hi, z1), visual=False))
        out = (face + 0.0001, face + 0.0007) if s > 0 else (face - 0.0007, face - 0.0001)
        parts.append(box3(f"port{i}", out, (a, b), (lo, hi), rgba=(0.02, 0.02, 0.02, 1), collision=False))
        if not p["target"]:              # tongue / keyed shape: a foreign plug stops after a few mm
            stop = (face - pd, face - DISTRACTOR_BACKSTOP) if s > 0 else (face + DISTRACTOR_BACKSTOP, face + pd)
            parts.append(box3(f"backstop{i}", stop, (a, b), (lo, hi), visual=False))
        edge = b
    parts.append(box3("seg_end", slab, (edge, half[j]), (z0, z1), visual=False))
    mesh = os.path.join(ASSETS, "laptop.obj").replace("\\", "/")
    parts.append(f'<visual name="laptop"><origin xyz="0 0 {RISER:.5f}"/>'
                 f'<geometry><mesh filename="{mesh}"/></geometry></visual>')
    return _urdf("plug_scene", parts)


def pedestal_urdf(spec):
    """Narrow pedestal (local x = plug axis); narrower than the housing so the fingers pass on both sides."""
    hx, hy, _ = spec["housing"]
    px, py = min(0.02, hx - 0.004), max(0.004, hy - 0.006)
    return _urdf("pedestal", [_box("pedestal", (-px / 2, -py / 2, 0), (px / 2, py / 2, PED_H),
                                   rgba=(0.55, 0.55, 0.6, 1))])


def plug_urdf(spec):
    """Plug: housing (gripped, link origin at its centre) + tip pointing +x."""
    hx, hy, hz = spec["housing"]
    tl, tw, th = spec["tip"]
    m = spec["mass"]
    ixx, iyy, izz = (max(4e-7, m * (b * b + c * c) / 12) for b, c in ((hy, hz), (hx, hz), (hx, hy)))
    tx = hx / 2 + tl / 2
    return (f'<?xml version="1.0"?>\n<robot name="plug"><link name="plug">'
            f'<inertial><origin xyz="{tx * tl / (hx + tl) / 2:.4f} 0 0"/><mass value="{m}"/>'
            f'<inertia ixx="{ixx:.3g}" ixy="0" ixz="0" iyy="{iyy:.3g}" iyz="0" izz="{izz:.3g}"/></inertial>'
            f'<visual name="housing"><geometry><box size="{hx} {hy} {hz}"/></geometry>'
            f'<material name="black"><color rgba="0.08 0.08 0.08 1"/></material></visual>'
            f'<collision name="housing"><geometry><box size="{hx} {hy} {hz}"/></geometry></collision>'
            f'<visual name="tip"><origin xyz="{tx:.5f} 0 0"/><geometry><box size="{tl} {tw} {th}"/></geometry>'
            f'<material name="metal"><color rgba="0.75 0.75 0.78 1"/></material></visual>'
            f'<collision name="tip"><origin xyz="{tx:.5f} 0 0"/><geometry><box size="{tl} {tw} {th}"/></geometry>'
            f'</collision></link></robot>\n')


def _R(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_z(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


class SimEnvUSB(SimEnvBase):
    """Insert a plug (USB-A / USB-C / HDMI / peg) into a laptop port (contact-rich; touch-guided insertion).
    Layout from ``scene_layout()`` (env knobs, module docstring)."""

    def __init__(self, sim=None, layout=None):
        super().__init__(sim)
        self.scene_id = None
        self.plug_id = None
        self.pedestal_id = None
        self._fric_on = None
        self.layout = layout or scene_layout()
        sp = self.layout["spec"]
        self.tip_x = sp["housing"][0] / 2 + sp["tip"][0]
        print(f"[Env] plug scene: {sp['label']} port on the laptop's {self.layout['face']} face, normal "
              f"{self.layout['dir_deg']:.0f} deg, port {np.round(self.layout['port'], 4).tolist()}, distractor "
              f"{self.layout['distractor']}, plug heading {math.degrees(self.layout['plug_yaw']):.0f} deg "
              f"(seed {self.layout['seed']}); head camera sees the port face at {self.layout['head_view_deg']:.0f} deg",
              flush=True)
        if self.layout["head_view_deg"] < 5.0:
            print("[Env] WARNING: the port face is edge-on / turned away from the head camera "
                  "(only the wrist camera can see it)", flush=True)

    def apply(self, env):
        # GUI / spherical-view camera (the default headless head camera is the fixed
        # config.head_camera_position): on the port face's outward side, 30 deg off its normal toward
        # the robot (-y), looking down 30 deg at a point 9 cm in front of the port and the pedestal.
        L = self.layout
        n = L["n_world"][:2]
        target = L["port"][:2] + 0.092 * n + np.array([0.0, -0.016])
        phi = math.atan2(n[1], n[0])
        beta = min((phi + math.radians(30), phi - math.radians(30)), key=math.sin)
        config.camera_target_position = [float(target[0]), float(target[1]), 0.07]
        config.camera_distance = 0.6
        config.camera_yaw = (math.degrees(beta) + 90.0) % 360.0
        config.camera_pitch = -30

    def load_assets(self, env):
        L = self.layout
        tmp = tempfile.gettempdir()
        files = {}
        for name, text in (("scene", scene_urdf(L)), ("pedestal", pedestal_urdf(L["spec"])),
                           ("plug", plug_urdf(L["spec"]))):
            files[name] = os.path.join(tmp, f"vlm_plug_{name}_{L['type']}_{os.getpid()}.urdf")
            with open(files[name], "w", encoding="utf-8") as f:
                f.write(text)
        self.scene_id = self.sim.load_urdf(files["scene"], [*L["laptop_xy"], 0.0], _quat_z(L["laptop_yaw"]),
                                           fixed_base=True)
        self.pedestal_id = self.sim.load_urdf(files["pedestal"], [*L["plug_xy"], 0.0], _quat_z(L["plug_yaw"]),
                                              fixed_base=True)
        hz = L["spec"]["housing"][2]
        self.plug_id = self.sim.load_urdf(files["plug"], [*L["plug_xy"], PED_H + hz / 2 + 0.0005],
                                          _quat_z(L["plug_yaw"]), fixed_base=False)

    def tune_physics(self):
        self.sim.change_dynamics(self.plug_id, -1, lateralFriction=PLUG_FRICTION)
        dt = config.control_dt / SCENE_OPTIONS["substeps"]

        def apply():
            plug = self.sim.native_entity(self.plug_id)
            plug.set_dofs_armature(np.array([PLUG_ARMATURE] * 3 + [PLUG_ARMATURE * 4e-4] * 3))
            # Armature is inertia without weight: alone it cuts the plug's free-fall to ~g/40, so a
            # dropped plug floats and drifts metres over the floor. Give that inertia its weight too.
            plug.control_dofs_force(np.array([-PLUG_ARMATURE * 9.81]), dofs_idx_local=[2])
            plug.set_dofs_frictionloss(np.array(FREE_FRICTION))
            plug.joints[0].set_sol_params(np.array([2 * dt, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0]))
        self.sim.run_after_build("plug conditioning", apply)

    def configure_robot(self, robot):
        """Franka Hand grasps with 20-70 N; the profile's finger kp 100 squeezes < 1 N, which
        cannot lift the (armature-conditioned) plug. kp 2500 ~ 20 N at the 8 mm closing error."""
        if robot.robot == "franka":
            self.sim.configure_motor_gains(robot.id, robot.gripper_joint_indices, kp=[GRIP_KP] * 2)

    def _tip(self):
        pos, q = self.sim.get_base_pose(self.plug_id)
        return np.asarray(pos, float) + _R(q) @ np.array([self.tip_x, 0.0, 0.0]), _R(q)

    def _port_frame(self, tip):
        """(depth past the target port's face along the insertion axis, lateral offset vector)."""
        d = tip - self.layout["port"]
        ax = -self.layout["n_world"]
        depth = float(d @ ax)
        return depth, d - depth * ax

    def step_hook(self):
        """Retention: dry friction on the plug while its tip is past the TARGET port's face."""
        if self.plug_id is None:
            return
        try:
            tip, _ = self._tip()
            depth, lat = self._port_frame(tip)
            inside = depth > 0.0005 and abs(lat[2]) < 0.01 and float(np.linalg.norm(lat[:2])) < 0.01
            if inside != self._fric_on:
                rn, rm = self.layout["spec"]["retention"]
                f = [rn] * 3 + [rm] * 3 if inside else FREE_FRICTION
                self.sim.native_entity(self.plug_id).set_dofs_frictionloss(np.array(f))
                self._fric_on = inside
        except Exception:
            pass

    def get_state(self):
        """Evaluation only (sim truth)."""
        L = self.layout
        st = {"plug_id": self.plug_id, "connector": L["type"], "port_face": L["face"],
              "port_pos": L["port"].round(4).tolist(), "insertion_axis": (-L["n_world"]).round(4).tolist(),
              "ports": [{k: p[k] for k in ("type", "target", "xyz")} for p in L["ports"]]}
        try:
            tip, R = self._tip()
            depth, lat = self._port_frame(tip)
            st.update(plug_pos=[round(float(v), 4) for v in self.sim.get_base_pose(self.plug_id)[0]],
                      plug_tip_pos=tip.round(4).tolist(), insertion_depth=round(depth, 4),
                      lateral_error=round(float(np.linalg.norm(lat)), 4), plug_axis=R[:, 0].round(3).tolist())
        except Exception as e:
            st["error"] = str(e)
        return st

    def get_success_criteria(self):
        return {"insertion_depth_gt": self.layout["spec"]["seat"], "lateral_error_lt": 0.002}

    def check_success(self):
        st = self.get_state()
        if "insertion_depth" not in st:
            return None
        return bool(st["insertion_depth"] > self.layout["spec"]["seat"] and st["lateral_error"] < 0.002)
