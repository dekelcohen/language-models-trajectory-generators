"""Kettle -> mug pour task (Genesis, native particle water).

Scene (world frame, robot base at the origin facing +y): a wooden board on the floor with a hollow
kettle (top bar handle along y, spout toward +y) and a red mug on its spout side (+y). The kettle
holds real SPH liquid particles; tilting it about the finger axis (x) pours them out of the spout.

Physics honesty: the kettle and the mug are hollow compound bodies (box walls) whose geoms couple
two-way with the SPH solver (particles push on the walls, walls push on the particles). Nothing is
spawned or teleported; the kettle is held by the fingers through contact + friction only.

The robot gets nothing from here except camera images and proprioception; water counts, mug and
kettle poses are evaluation / logging only. Genesis-only (needs the SPH solver).
"""

import math
import os

import numpy as np

import config
from sim_envs.pybullet.base import SimEnvBase

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "kettle_mug")

# --- liquid (SPH) -----------------------------------------------------------------------------
PARTICLE_SIZE = 0.008                   # SPH particle diameter (m): ~0.5 g each at rho 1000
#: Simulation domain for the liquid; particles cannot leave it (they stop at the box faces).
SPH_LOWER = (-0.35, 0.08, 0.0)
SPH_UPPER = (0.35, 0.90, 0.85)
#: Solver settings this scene needs (GenesisAdapter(scene_options=...) before the scene exists).
#: 4 substeps = 1 ms SPH step. PBD liquid was tried too: its rigid coupling blew the kettle away.
#: Torsional friction: Genesis contacts are points, so without it a handle squeezed between the
#: pads spins freely about the finger axis (the pour tilt axis); a real rubber pad resists that
#: over its contact patch (HANDLE_PATCH_R).
SCENE_OPTIONS = {"substeps": 4,
                 "rigid": {"enable_torsional_friction": True, "noslip_iterations": 5},
                 "sph": {"particle_size": PARTICLE_SIZE, "pressure_solver": "DFSPH",
                         "lower_bound": SPH_LOWER, "upper_bound": SPH_UPPER}}
HANDLE_PATCH_R = 0.012   # pad torsional patch (m): ~2.4 cm rubber pad, mu ~1.3
WATER_COLOR = (0.25, 0.55, 0.95, 0.75)
CAPTURE_RES = 512                       # head + wrist capture size (px)

# --- board -------------------------------------------------------------------------------------
BOARD = ((-0.25, 0.14, 0.0), (0.25, 0.74, 0.02))    # (lo, hi): thin wooden board on the floor
TOP = BOARD[1][2]

# --- kettle (local frame: origin at the bottom centre, spout toward +x, handle bar along x) ------
K_R = 0.075                 # outer radius
K_T = 0.01                  # wall / bottom / lid thickness
K_H = 0.15                  # body height
K_SEG = 16
K_HOLE = (0.02, 0.075)      # spout opening in the +x wall segment (z range)
SPOUT_ANGLE = math.radians(38.0)
SPOUT_START = (K_R - K_T / 2 - 0.01, 0.035)   # (x, z) of the duct axis start (inside the wall)
SPOUT_LEN = 0.065
SPOUT_IN = (0.04, 0.036)    # inner (width y, height) of the duct
HANDLE_Z = 0.20             # bar centre height: tilted 0.8 rad the spout tip is ~1 cm behind, ~0.17 m below it
HANDLE = (0.13, 0.026, 0.03)
POST = (0.02, 0.02)
KETTLE_MASS = 0.3           # thin aluminium kettle; + ~0.3 kg water
KETTLE_COLOR = (0.82, 0.84, 0.86, 1.0)
HANDLE_COLOR = (0.15, 0.15, 0.16, 1.0)
KETTLE_XY = (0.0, 0.25)
#: The kettle is placed yawed +90 deg: spout toward world +y (the mug), handle bar along y. The
#: natural top-down grasp across the bar is then rotation 0 (fingers close along x), whose pour
#: tilt axis (pour_yaw = +-pi/2) dips the spout - the subtask prompt always reports rotation 0
#: for the held object, so a pi/2 grasp would be lost at the subtask boundary.
KETTLE_YAW = math.pi / 2
#: Water fill: cylinder inside the kettle (local frame), sampled as SPH particles.
FILL_R, FILL_Z = K_R - K_T - 0.008, (K_T + 0.004, K_T + 0.034)

# --- mug -----------------------------------------------------------------------------------------
MUG_R = 0.05                # outer radius
MUG_T = 0.008
MUG_H = 0.09
MUG_SEG = 20
MUG_MASS = 0.3
MUG_COLOR = (0.85, 0.2, 0.2, 1.0)
MUG_XY = (0.0, 0.52)    # pour hand ~0.6 m from the base: farther, the tilted pose is out of reach
MUG_YAW = math.pi / 2       # mug handle (local +y) toward the -x head camera
#: Success: at least this fraction of the water inside the mug's inner cylinder.
MIN_FRACTION = 0.5


def _yaw_q(a):
    """xyzw quaternion of a rotation by ``a`` about world z."""
    return [0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _geom(name, xyz, size, rpy=(0.0, 0.0, 0.0), rgba=None, collision=True, visual=True, mesh=None):
    o = (f'<origin xyz="{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}" '
         f'rpy="{rpy[0]:.5f} {rpy[1]:.5f} {rpy[2]:.5f}"/>')
    g = f'<geometry><box size="{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}"/></geometry>'
    out = ""
    if visual:
        vg = f'<geometry><mesh filename="{mesh}"/></geometry>' if mesh else g
        col = " ".join(f"{v:.3f}" for v in (rgba or (0.8, 0.8, 0.8, 1)))
        out += f'<visual name="{name}">{o}{vg}<material name="m_{name}"><color rgba="{col}"/></material></visual>'
    if collision:
        out += f'<collision name="{name}">{o}{g}</collision>'
    return out


def _ring(prefix, r_out, t, h, n, z0=0.0, hole=None):
    """Box wall segments of a hollow cylinder (collision only). ``hole`` = (z_lo, z_hi) cut out of
    the segment that faces +x."""
    r_mid = r_out - t / 2
    w = 2 * math.pi * r_out / n * 1.3     # overlap so no slit between segments
    parts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        c, s = math.cos(a), math.sin(a)
        spans = [(z0, z0 + h)]
        if hole is not None and i == 0:
            spans = [(z0, hole[0]), (hole[1], z0 + h)]
        for j, (lo, hi) in enumerate(spans):
            parts.append(_geom(f"{prefix}{i}_{j}", (r_mid * c, r_mid * s, (lo + hi) / 2), (t, w, hi - lo),
                               rpy=(0, 0, a), visual=False))
    return parts


def _spout_frame():
    d = np.array([math.cos(SPOUT_ANGLE), 0.0, math.sin(SPOUT_ANGLE)])
    n = np.array([-math.sin(SPOUT_ANGLE), 0.0, math.cos(SPOUT_ANGLE)])
    s = np.array([SPOUT_START[0], 0.0, SPOUT_START[1]])
    return s, d, n


def spout_tip_local():
    """Kettle-local centre of the spout opening."""
    s, d, _ = _spout_frame()
    return s + d * SPOUT_LEN


def _inertia(m, sx, sy, sz):
    return (m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12)


def _urdf(name, mass, inertial_xyz, inertia, parts):
    ixx, iyy, izz = inertia
    return (f'<?xml version="1.0"?>\n<robot name="{name}"><link name="{name}">'
            f'<inertial><origin xyz="{inertial_xyz[0]:.4f} {inertial_xyz[1]:.4f} {inertial_xyz[2]:.4f}"/>'
            f'<mass value="{mass}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" '
            f'izz="{izz:.6f}"/></inertial>' + "".join(parts) + "</link></robot>\n")


def kettle_urdf(body_mesh=None, spout_mesh=None):
    """Hollow kettle: bottom, 16 wall boxes (one with the spout opening), lid, a 4-plate spout duct
    and a bar handle on two posts. ``*_mesh`` = optional smooth visual meshes (else box visuals)."""
    parts = [_geom("bottom", (0, 0, K_T / 2), (2 * K_R * 0.98, 2 * K_R * 0.98, K_T), visual=False),
             _geom("lid", (0, 0, K_H - K_T / 2), (2 * K_R * 0.98, 2 * K_R * 0.98, K_T), visual=False)]
    parts += _ring("wall", K_R, K_T, K_H, K_SEG, hole=K_HOLE)
    s, d, n = _spout_frame()
    pitch = -SPOUT_ANGLE
    w, h = SPOUT_IN
    mid = s + d * SPOUT_LEN / 2
    for nm, off, size in (("sp_bot", -n * (h / 2 + K_T / 2), (SPOUT_LEN, w + 2 * K_T, K_T)),
                          ("sp_top", n * (h / 2 + K_T / 2), (SPOUT_LEN, w + 2 * K_T, K_T)),
                          ("sp_l", np.array([0, w / 2 + K_T / 2, 0]), (SPOUT_LEN, K_T, h)),
                          ("sp_r", np.array([0, -w / 2 - K_T / 2, 0]), (SPOUT_LEN, K_T, h))):
        parts.append(_geom(nm, mid + off, size, rpy=(0, pitch, 0), visual=False))
    for i, x in enumerate((-HANDLE[0] / 2 + POST[0] / 2, HANDLE[0] / 2 - POST[0] / 2)):
        z0, z1 = K_H, HANDLE_Z - HANDLE[2] / 2
        parts.append(_geom(f"post{i}", (x, 0, (z0 + z1) / 2), (POST[0], POST[1], z1 - z0 + 0.002),
                           rgba=HANDLE_COLOR))
    parts.append(_geom("handle", (0, 0, HANDLE_Z), HANDLE, rgba=HANDLE_COLOR))
    if body_mesh:
        parts.append(_geom("body_vis", (0, 0, 0), (1, 1, 1), rgba=KETTLE_COLOR, collision=False, mesh=body_mesh))
    if spout_mesh:
        parts.append(_geom("spout_vis", (0, 0, 0), (1, 1, 1), rgba=KETTLE_COLOR, collision=False, mesh=spout_mesh))
    return _urdf("kettle", KETTLE_MASS, (0, 0, K_H / 2), _inertia(KETTLE_MASS, 2 * K_R, 2 * K_R, K_H), parts)


def mug_urdf(mesh=None):
    parts = [_geom("bottom", (0, 0, MUG_T / 2), (2 * MUG_R * 0.9, 2 * MUG_R * 0.9, MUG_T), visual=False)]
    parts += _ring("wall", MUG_R, MUG_T, MUG_H, MUG_SEG)
    if mesh:
        parts.append(_geom("mug_vis", (0, 0, 0), (1, 1, 1), rgba=MUG_COLOR, collision=False, mesh=mesh))
    return _urdf("mug", MUG_MASS, (0, 0, MUG_H / 3), _inertia(MUG_MASS, 2 * MUG_R, 2 * MUG_R, MUG_H), parts)


def board_urdf():
    lo, hi = np.array(BOARD[0]), np.array(BOARD[1])
    return _urdf("board", 1.0, (0, 0, 0), (1, 1, 1),
                 [_geom("board", (lo + hi) / 2, hi - lo, rgba=(0.55, 0.38, 0.22, 1.0))])


def write_meshes(out_dir=ASSETS):
    """Smooth visual meshes (OBJ) for the kettle body, spout and mug. Needs trimesh (Genesis env)."""
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    body = trimesh.creation.cylinder(radius=K_R, height=K_H - 0.004, sections=48)
    body.apply_translation([0, 0, (K_H - 0.004) / 2])
    knob = trimesh.creation.cylinder(radius=K_R * 0.55, height=0.008, sections=32)
    knob.apply_translation([0, 0, K_H])
    body = trimesh.util.concatenate([body, knob])
    s, d, _ = _spout_frame()
    r_out = max(SPOUT_IN) / 2 + K_T * 0.8
    spout = trimesh.creation.annulus(r_min=max(SPOUT_IN) / 2 + 0.001, r_max=r_out, height=SPOUT_LEN, sections=32)
    # annulus axis is z; rotate z -> d (pitch about y by 90 deg - spout angle), then move to the duct mid.
    spout.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2 - SPOUT_ANGLE, [0, 1, 0]))
    spout.apply_translation(s + d * SPOUT_LEN / 2)
    # annulus / cylinder are centred on z=0; lift them so the mug stands on its base.
    mug = trimesh.util.concatenate([
        trimesh.creation.annulus(r_min=MUG_R - MUG_T * 0.6, r_max=MUG_R, height=MUG_H, sections=48)
        .apply_translation([0, 0, MUG_H / 2]),
        trimesh.creation.cylinder(radius=MUG_R - 0.001, height=MUG_T, sections=48)
        .apply_translation([0, 0, MUG_T / 2])])
    handle = trimesh.creation.annulus(r_min=0.022, r_max=0.03, height=0.012, sections=32)
    handle.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    handle.apply_translation([0, MUG_R + 0.012, MUG_H * 0.55])
    mug = trimesh.util.concatenate([mug, handle])
    paths = {}
    for name, m in (("kettle_body", body), ("kettle_spout", spout), ("mug", mug)):
        paths[name] = os.path.join(out_dir, f"{name}.obj").replace("\\", "/")
        m.export(paths[name])
    return paths


def write_assets(out_dir=ASSETS):
    """Write the scene URDFs (+ visual meshes) and return their paths."""
    meshes = write_meshes(out_dir)
    files = {"kettle": kettle_urdf(meshes["kettle_body"], meshes["kettle_spout"]),
             "mug": mug_urdf(meshes["mug"]), "board": board_urdf()}
    paths = {}
    for name, text in files.items():
        paths[name] = os.path.join(out_dir, f"{name}.urdf")
        with open(paths[name], "w", encoding="utf-8") as f:
            f.write(text)
    return paths


def _R(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def count_water(positions, mug_pos, mug_q, kettle_pos=None, kettle_q=None):
    """Classify particles: inside the mug's inner cylinder / inside the kettle body / elsewhere."""
    p = np.asarray(positions, float).reshape(-1, 3)
    local = (p - np.asarray(mug_pos, float)) @ _R(mug_q)
    in_mug = ((np.hypot(local[:, 0], local[:, 1]) < MUG_R - MUG_T + 0.002)
              & (local[:, 2] > -0.005) & (local[:, 2] < MUG_H + 0.01))
    in_kettle = np.zeros(len(p), bool)
    if kettle_pos is not None:
        kl = (p - np.asarray(kettle_pos, float)) @ _R(kettle_q)
        in_kettle = (np.hypot(kl[:, 0], kl[:, 1]) < K_R) & (kl[:, 2] > -0.005) & (kl[:, 2] < K_H + 0.005)
        tip = spout_tip_local()
        s, d, _ = _spout_frame()
        t = (kl - s) @ d
        radial = np.linalg.norm((kl - s) - np.outer(t, d), axis=1)
        in_kettle |= (t > 0) & (t < SPOUT_LEN) & (radial < max(SPOUT_IN)) & (kl[:, 2] < tip[2] + 0.02)
    n_mug = int(in_mug.sum())
    n_kettle = int((in_kettle & ~in_mug).sum())
    return {"total": int(len(p)), "in_kettle": n_kettle, "in_mug": n_mug,
            "outside_mug": int(len(p)) - n_mug - n_kettle}


class SimEnvKettleMug(SimEnvBase):
    """Pick up the kettle by its top handle and pour its (SPH) water into the mug."""

    def __init__(self, sim=None):
        super().__init__(sim)
        self.kettle_id = None
        self.mug_id = None
        self.board_id = None
        self.water = None

    def apply(self, env):
        # Head camera: side view from -x looking +x (as the franka_kitchen scenes: image right =
        # -y), so kettle, stream and mug sit side by side; framed up to z~0.85 so a kettle lifted
        # by the planner's default pick-up height (50 cm) stays in view.
        config.camera_target_position = [0.0, 0.42, 0.35]
        config.camera_distance = 1.0
        config.camera_yaw = 270
        config.camera_pitch = -20
        config.head_camera_use_debug_view = False
        config.head_camera_use_spherical_view = True
        # 512 px so the thin SPH stream is visible (pixel->3D code reads the image size).
        config.image_width = config.image_height = CAPTURE_RES
        self.sim.reserve_camera(CAPTURE_RES, CAPTURE_RES, fov=config.fov, near=config.near_plane,
                                far=config.far_plane)

    def load_assets(self, env):
        import genesis as gs

        paths = write_assets()
        self.board_id = self.sim.load_urdf(paths["board"], [0, 0, 0], [0, 0, 0, 1], fixed_base=True)
        self.kettle_id = self.sim.load_urdf(paths["kettle"], [KETTLE_XY[0], KETTLE_XY[1], TOP + 0.001],
                                            _yaw_q(KETTLE_YAW), fixed_base=False)
        self.mug_id = self.sim.load_urdf(paths["mug"], [MUG_XY[0], MUG_XY[1], TOP + 0.001],
                                         _yaw_q(MUG_YAW), fixed_base=False)
        scene = self.sim._ensure_scene()
        zc = TOP + 0.001 + (FILL_Z[0] + FILL_Z[1]) / 2
        # mu 0.002 keeps the SPH viscosity step stable at the 1 ms substep (0.005 needs 0.5 ms).
        material = gs.materials.SPH.Liquid(rho=1000.0, stiffness=50000.0, exponent=7.0, mu=0.002,
                                           gamma=0.01, sampler="regular")
        self.water = scene.add_entity(
            material=material,
            morph=gs.morphs.Cylinder(pos=(KETTLE_XY[0], KETTLE_XY[1], zc), radius=FILL_R,
                                     height=FILL_Z[1] - FILL_Z[0]),
            surface=gs.surfaces.Default(color=WATER_COLOR, vis_mode=os.environ.get("LMTG_WATER_VIS", "particle")),
        )

    def tune_physics(self):
        self.sim.change_dynamics(self.kettle_id, -1, lateralFriction=1.0)
        self.sim.change_dynamics(self.mug_id, -1, lateralFriction=1.0)
        kettle = self.sim.native_entity(self.kettle_id)
        self.sim.run_after_build("kettle torsional friction",
                                 lambda: kettle.set_friction_torsional(HANDLE_PATCH_R))

    def configure_robot(self, robot):
        """Squeeze the handle at the Franka Hand's 70 N continuous grasp force (the profile caps the
        finger force at 70 N; kp 6000 x ~13 mm closing error saturates it). The tilt torque about
        the finger axis is held by pad friction only, so the default ~30 N lets the kettle swing."""
        if robot.robot == "franka":
            self.sim.configure_motor_gains(robot.id, robot.gripper_joint_indices, kp=[6000.0] * 2)

    def water_positions(self):
        from sim_adapter.genesis_adapter import _to_numpy

        return _to_numpy(self.water.get_particles_pos()).reshape(-1, 3)

    def water_counts(self):
        mp, mq = self.sim.get_base_pose(self.mug_id)
        kp, kq = self.sim.get_base_pose(self.kettle_id)
        return count_water(self.water_positions(), mp, mq, kp, kq)

    def task_error(self):
        """Fraction of the water NOT in the mug."""
        try:
            c = self.water_counts()
        except Exception:
            return None
        return 1.0 - c["in_mug"] / float(max(c["total"], 1))

    def get_success_criteria(self):
        return {"goal_pour": {"min_fraction": MIN_FRACTION}, "threshold": 1.0 - MIN_FRACTION + 1e-9}

    def check_success(self):
        err = self.task_error()
        return None if err is None else bool(err < 1.0 - MIN_FRACTION + 1e-9)

    def get_state(self):
        """Evaluation / logging only (sim truth): never goes into a prompt."""
        st = {"sim_env": "genesis_kettle_mug", "task": "kettle_mug", "kettle_id": self.kettle_id,
              "mug_id": self.mug_id, "success_criteria": self.get_success_criteria()}
        try:
            st["kettle_pos"] = [round(float(v), 4) for v in self.sim.get_base_pose(self.kettle_id)[0]]
            st["kettle_tilt_deg"] = round(math.degrees(math.acos(min(1.0, max(-1.0, float(
                _R(self.sim.get_base_pose(self.kettle_id)[1])[2, 2]))))), 1)
            st["mug_pos"] = [round(float(v), 4) for v in self.sim.get_base_pose(self.mug_id)[0]]
            st["water"] = self.water_counts()
            st["task_error"] = round(1.0 - st["water"]["in_mug"] / float(max(st["water"]["total"], 1)), 4)
            st["success"] = bool(st["task_error"] < 1.0 - MIN_FRACTION + 1e-9)
        except Exception as e:
            st["error"] = str(e)
        return st
