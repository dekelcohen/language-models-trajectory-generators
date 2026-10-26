"""USB task (Genesis): pick the USB plug off its pedestal and insert it into the laptop's side port.

Scene (world frame, robot base at the origin facing +y): a laptop on a 5 cm riser with its USB-A
port on the side that faces -x, and a USB plug lying on a narrow pedestal (fingers pass on both
sides of it), its tip pointing +x, i.e. toward the port side. Geometry and contact physics are the
ones measured on the bench (research/touch/genesis_touch_probe.py, plans/touch.md section 3).

The robot gets nothing from here except camera images; port pose and depth are evaluation only.
Genesis-only: needs ``sim.native_entity`` / ``run_after_build`` and the solver options below.
"""

import os
import tempfile

import numpy as np

import config
from sim_envs.pybullet.base import SimEnvBase

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "usb_laptop")

#: Solver settings this scene needs (passed to GenesisAdapter(scene_options=...) before the scene
#: exists): 2 substeps, box-box contacts and noslip, or a 12 g plug squeezes through a 0.3 mm gap.
SCENE_OPTIONS = {"substeps": 2, "rigid": {"box_box_detection": True, "noslip_iterations": 5}}

TIP_LEN, TIP_W, TIP_H = 0.012, 0.012, 0.0045
HOUSING = (0.025, 0.016, 0.008)
TIP_X = HOUSING[0] / 2 + TIP_LEN                # plug origin (housing centre) -> tip end, along plug +x
CLEAR_Y, CLEAR_Z = 0.00025, 0.0003
PORT_FRICTION_N, PORT_FRICTION_NM = 12.0, 0.05  # USB retention while the tip is inside (env physics)
PLUG_FRICTION = 0.6
#: Finger-pad patch torsional friction (~mu * F_grip * r_patch = 0.6 * 20 N * 4 mm). The URDF Franka's mesh
#: fingers touch the housing at near-points, so without this the held plug pivots freely about the finger
#: axis and creeps 10-20 deg during a carry. Far below jam torques (5 N x 25 mm), so the port still aligns it.
PAD_TORSION_NM = 0.05
FREE_FRICTION = [0.0] * 3 + [PAD_TORSION_NM] * 3
GRIP_KP = 2500.0                                # finger position gain for this task (see configure_robot)
PLUG_ARMATURE = 0.5                             # conditioning only (not weight), see the bench notes

LAP_C = np.array([0.30, 0.55])
RISER = 0.05
LAP = (0.416, 0.2824, 0.0244)
FACE_X = LAP_C[0] - LAP[0] / 2
PORT = np.array([FACE_X, LAP_C[1] - 0.0541, RISER + 0.0190])
POCKET = 0.014
PED = (0.02, 0.010, 0.05)                       # pedestal (x, y, z); narrower than the 16 mm housing
PLUG_XY = np.array([-0.12, 0.45])
SEATED_DEPTH = 0.010


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


def scene_urdf():
    """Fixed scene as one URDF: riser, laptop collision boxes around the port pocket, laptop mesh
    (visual), dark port opening (visual), plug pedestal."""
    x0, x1 = FACE_X, FACE_X + LAP[0]
    y0, y1 = LAP_C[1] - LAP[1] / 2, LAP_C[1] + LAP[1] / 2
    z0, z1 = RISER, RISER + LAP[2]
    hy, hz = TIP_W / 2 + CLEAR_Y, TIP_H / 2 + CLEAR_Z
    parts = [
        _box("riser", (x0, y0, 0), (x1, y1, RISER)),
        _box("body", (x0 + POCKET, y0, z0), (x1, y1, z1), visual=False),
        _box("left", (x0, y0, z0), (x0 + POCKET, PORT[1] - hy, z1), visual=False),
        _box("right", (x0, PORT[1] + hy, z0), (x0 + POCKET, y1, z1), visual=False),
        _box("below", (x0, PORT[1] - hy, z0), (x0 + POCKET, PORT[1] + hy, PORT[2] - hz), visual=False),
        _box("above", (x0, PORT[1] - hy, PORT[2] + hz), (x0 + POCKET, PORT[1] + hy, z1), visual=False),
        _box("port", (FACE_X - 0.0007, PORT[1] - hy, PORT[2] - hz), (FACE_X - 0.0001, PORT[1] + hy, PORT[2] + hz),
             rgba=(0.02, 0.02, 0.02, 1), collision=False),
        _box("pedestal", (PLUG_XY[0] - PED[0] / 2, PLUG_XY[1] - PED[1] / 2, 0),
             (PLUG_XY[0] + PED[0] / 2, PLUG_XY[1] + PED[1] / 2, PED[2]), rgba=(0.55, 0.55, 0.6, 1)),
    ]
    mesh = os.path.join(ASSETS, "laptop.obj").replace("\\", "/")
    parts.append(f'<visual name="laptop"><origin xyz="{LAP_C[0]:.5f} {LAP_C[1]:.5f} {RISER:.5f}"/>'
                 f'<geometry><mesh filename="{mesh}"/></geometry></visual>')
    return ('<?xml version="1.0"?>\n<robot name="usb_scene"><link name="scene">'
            '<inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
            + "".join(parts) + "</link></robot>\n")


def _R(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SimEnvUSB(SimEnvBase):
    """Plug a USB stick into a laptop's side port (contact-rich; meant for touch-guided insertion)."""

    def __init__(self, sim=None):
        super().__init__(sim)
        self.scene_id = None
        self.plug_id = None
        self._fric_on = None

    def apply(self, env):
        # Head camera: front-left, facing the laptop's port side (-x) with the plug in view.
        config.camera_target_position = [0.0, 0.48, 0.07]
        config.camera_distance = 0.6
        config.camera_yaw = 300
        config.camera_pitch = -30

    def load_assets(self, env):
        path = os.path.join(tempfile.gettempdir(), "vlm_usb_scene.urdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(scene_urdf())
        self.scene_id = self.sim.load_urdf(path, [0, 0, 0], [0, 0, 0, 1], fixed_base=True)
        self.plug_id = self.sim.load_urdf(os.path.join(ASSETS, "usb_plug.urdf"),
                                          [PLUG_XY[0], PLUG_XY[1], PED[2] + HOUSING[2] / 2 + 0.0005],
                                          [0, 0, 0, 1], fixed_base=False)

    def tune_physics(self):
        self.sim.change_dynamics(self.plug_id, -1, lateralFriction=PLUG_FRICTION)
        dt = config.control_dt / SCENE_OPTIONS["substeps"]

        def apply():
            plug = self.sim.native_entity(self.plug_id)
            plug.set_dofs_armature(np.array([PLUG_ARMATURE] * 3 + [PLUG_ARMATURE * 4e-4] * 3))
            plug.set_dofs_frictionloss(np.array(FREE_FRICTION))
            plug.joints[0].set_sol_params(np.array([2 * dt, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0]))
        self.sim.run_after_build("usb plug conditioning", apply)

    def configure_robot(self, robot):
        """Franka Hand grasps with 20-70 N; the profile's finger kp 100 squeezes < 1 N, which
        cannot lift the (armature-conditioned) plug. kp 2500 ~ 20 N at the 8 mm closing error."""
        if robot.robot == "franka":
            self.sim.configure_motor_gains(robot.id, robot.gripper_joint_indices, kp=[GRIP_KP] * 2)

    def _tip(self):
        pos, q = self.sim.get_base_pose(self.plug_id)
        return np.asarray(pos, float) + _R(q) @ np.array([TIP_X, 0.0, 0.0]), _R(q)

    def step_hook(self):
        """USB retention springs: dry friction on the plug while its tip is past the port face."""
        if self.plug_id is None:
            return
        try:
            tip, _ = self._tip()
            inside = (tip[0] - FACE_X > 0.0005 and abs(tip[1] - PORT[1]) < 0.01 and abs(tip[2] - PORT[2]) < 0.01)
            if inside != self._fric_on:
                f = [PORT_FRICTION_N] * 3 + [PORT_FRICTION_NM] * 3 if inside else FREE_FRICTION
                self.sim.native_entity(self.plug_id).set_dofs_frictionloss(np.array(f))
                self._fric_on = inside
        except Exception:
            pass

    def get_state(self):
        """Evaluation only (sim truth)."""
        st = {"plug_id": self.plug_id, "port_pos": PORT.round(4).tolist(), "insertion_axis": [1.0, 0.0, 0.0]}
        try:
            tip, R = self._tip()
            st.update(plug_pos=[round(float(v), 4) for v in self.sim.get_base_pose(self.plug_id)[0]],
                      plug_tip_pos=tip.round(4).tolist(),
                      insertion_depth=round(float(tip[0] - FACE_X), 4),
                      lateral_error=round(float(np.hypot(tip[1] - PORT[1], tip[2] - PORT[2])), 4),
                      plug_axis=R[:, 0].round(3).tolist())
        except Exception as e:
            st["error"] = str(e)
        return st

    def get_success_criteria(self):
        return {"insertion_depth_gt": SEATED_DEPTH, "lateral_error_lt": 0.002}

    def check_success(self):
        st = self.get_state()
        if "insertion_depth" not in st:
            return None
        return bool(st["insertion_depth"] > SEATED_DEPTH and st["lateral_error"] < 0.002)
