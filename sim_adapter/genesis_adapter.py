"""Genesis implementation of :class:`~sim_adapter.base.SimAdapter`.

This file exists so that ``env.py``, ``robot.py`` and the sim-env profiles can stay exactly
as they are. Everywhere Genesis disagrees with PyBullet, the disagreement is resolved
*here*:

===========================  ====================================================
Genesis                      What the adapter does
===========================  ====================================================
quaternions ``wxyz``         converts at every boundary (``sim_adapter.transforms``)
Euler in degrees             never used; rotations go through PyBullet-convention
                             quaternion math so both sims frame identical pictures
DOF indices, not joints      keeps a per-body joint -> local-DOF map
torch tensors, maybe on GPU  ``_scalar`` / ``_vector`` convert at the boundary
``scene.build()`` barrier    pre-build mutations are **queued** and replayed in
                             :meth:`build`, so callers keep PyBullet's ordering
no implicit PD gains         :meth:`configure_motor_gains`, fed by ``RobotProfile``
``forces=[F]``               becomes ``set_dofs_force_range(-F, +F)``
merges fixed links           loads with ``merge_fixed_links=False`` so link names
                             match PyBullet's exactly
cameras must pre-exist       :meth:`reserve_camera` before build; the projection
                             matrix passed to :meth:`render_camera` is validated
                             against the camera that was actually created
depth is metric              ``depth_encoding = "linear_metric"``
===========================  ====================================================

Body handles are **ints**, not ``RigidEntity`` objects, because ``simenv.get_state()``
puts them straight into an IPC payload that has to be JSON-serialisable.
"""

import math
import os
import threading

import numpy as np

import config
from sim_adapter import transforms
from sim_adapter.base import (
    DEPTH_LINEAR_METRIC,
    JOINT_FIXED,
    JOINT_OTHER,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    CameraFrame,
    JointInfo,
    JointState,
    SimAdapter,
)
from sim_adapter.camera_math import spherical_camera_pose

_GS_INIT_LOCK = threading.Lock()
_GS_INITIALISED = False

#: Anything matching this is PyBullet's finite ground plate; Genesis' infinite
#: ``gs.morphs.Plane`` is the closer semantic match and avoids depending on
#: pybullet_data being importable in the Genesis interpreter.
_PLANE_URDFS = ("plane.urdf", "plane_implicit.urdf", "plane100.urdf")

#: Exposure, tuned so a Genesis capture has roughly the same brightness as the PyBullet
#: one for the same scene and camera. Override per run with LMTG_GENESIS_AMBIENT.
VIS_AMBIENT_LIGHT = tuple(
    float(v) for v in os.environ.get("LMTG_GENESIS_AMBIENT", "0.30,0.30,0.30").split(","))
VIS_BACKGROUND_COLOR = (0.85, 0.88, 0.92)
VIS_LIGHT_INTENSITY = float(os.environ.get("LMTG_GENESIS_LIGHT_INTENSITY", "4.0"))
#: Ground colour, chosen to sit near PyBullet's light-blue checker plate.
PLANE_COLOR = tuple(
    float(v) for v in os.environ.get("LMTG_GENESIS_PLANE_COLOR", "0.62,0.66,0.74").split(","))


def _log(message):
    """Single funnel for adapter diagnostics.

    Genesis runs in a child process whose stdout is the only channel back, so these go to
    print rather than to a logger that may not be configured yet.
    """
    print(f"[GenesisAdapter] {message}", flush=True)


def _to_numpy(value):
    """torch tensor / list / scalar -> numpy, always on the CPU."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value):
    arr = _to_numpy(value).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _vector(value, size=None):
    arr = _to_numpy(value).reshape(-1)
    out = [float(v) for v in arr]
    return out[:size] if size is not None else out


class _Body:
    """Bookkeeping for one loaded entity, keyed by the int handle callers see."""

    __slots__ = ("handle", "entity", "source", "joints", "links",
                 "joint_names", "link_names")

    def __init__(self, handle, entity, source):
        self.handle = handle
        self.entity = entity
        self.source = source
        self.joints = []
        self.links = []
        self.joint_names = {}
        self.link_names = {}

    def refresh(self):
        """(Re)build the joint/link tables. Genesis exposes them once the morph is parsed."""
        self.joints = list(getattr(self.entity, "joints", []) or [])
        self.links = list(getattr(self.entity, "links", []) or [])
        self.joint_names = {j.name: i for i, j in enumerate(self.joints)}
        self.link_names = {l.name: i for i, l in enumerate(self.links)}


_JOINT_TYPE_NAMES = {
    "FIXED": JOINT_FIXED,
    "REVOLUTE": JOINT_REVOLUTE,
    "PRISMATIC": JOINT_PRISMATIC,
}


class GenesisAdapter(SimAdapter):

    name = "genesis"
    depth_encoding = DEPTH_LINEAR_METRIC

    def __init__(self, backend=None, dt=None, gravity=(0.0, 0.0, -9.81)):
        self.scene = None
        self.viewer_options = None
        self._backend = backend
        self._dt = float(dt) if dt is not None else float(getattr(config, "control_dt", 1.0 / 240.0))
        self._gravity = tuple(float(v) for v in gravity)
        self._gui = False
        self._connected = False
        self._built = False

        self._bodies = {}
        self._next_handle = 1
        self._asset_root = None

        # Mutations issued before build(); Genesis forbids them on an unbuilt scene but
        # PyBullet allows them, so the app layer is written that way and we replay here.
        self._deferred = []

        # (width, height) -> genesis Camera. Cameras are entities-in-spirit: they must
        # exist before scene.build().
        self._cameras = {}
        self._reserved = set()
        self._camera_fov = float(getattr(config, "fov", 60.0))
        self._camera_near = float(getattr(config, "near_plane", 0.01))
        self._camera_far = float(getattr(config, "far_plane", 100.0))
        self._warned_projection = False

        self._debug_nodes = {}
        self._next_debug_handle = 1
        self._warned_texture = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self, gui=False):
        global _GS_INITIALISED
        import genesis as gs

        self._gui = bool(gui)
        with _GS_INIT_LOCK:
            if not _GS_INITIALISED:
                backend = self._backend
                if backend is None:
                    backend = gs.cpu
                elif isinstance(backend, str):
                    backend = getattr(gs, backend)
                gs.init(backend=backend, logging_level="warning")
                _GS_INITIALISED = True
        self._connected = True
        _log(f"connected (gui={self._gui}, dt={self._dt}, backend={self._backend or 'cpu'})")
        return 0

    def _ensure_scene(self):
        """Create the Scene on first use.

        Deliberately lazy: ``env.py`` calls ``set_gravity`` *after* ``connect``, and Genesis
        can only take gravity in the ``SimOptions`` handed to the Scene constructor.
        """
        if self.scene is not None:
            return self.scene
        import genesis as gs

        kwargs = {
            "sim_options": gs.options.SimOptions(dt=self._dt, gravity=self._gravity),
            "show_viewer": self._gui,
            # Genesis defaults to a dark, near-black studio (ambient 0.1, background
            # 0.04/0.08/0.12). PyBullet's TinyRenderer is much brighter, and the whole
            # downstream pipeline is a VLM looking at these frames - underexposed
            # captures measurably degrade segmentation. Match PyBullet's exposure.
            "vis_options": gs.options.VisOptions(
                ambient_light=VIS_AMBIENT_LIGHT,
                background_color=VIS_BACKGROUND_COLOR,
                shadow=True,
                lights=[{"type": "directional", "dir": (-1.0, -1.0, -1.0),
                         "color": (1.0, 1.0, 1.0), "intensity": VIS_LIGHT_INTENSITY}],
            ),
        }
        if self._gui:
            eye, _ = spherical_camera_pose(
                config.camera_target_position, config.camera_distance,
                config.camera_yaw, config.camera_pitch,
            )
            kwargs["viewer_options"] = gs.options.ViewerOptions(
                camera_pos=tuple(eye),
                camera_lookat=tuple(float(v) for v in config.camera_target_position),
                camera_fov=float(getattr(config, "fov", 60.0)),
            )
        self.scene = gs.Scene(**kwargs)
        _log(f"scene created (gravity={self._gravity}, viewer={self._gui})")
        return self.scene

    def disconnect(self):
        try:
            import genesis as gs
            gs.destroy()
        except Exception as exc:
            _log(f"disconnect: gs.destroy() failed ({exc})")
        self.scene = None
        self._connected = False
        self._built = False
        self._bodies.clear()
        self._cameras.clear()

    def is_connected(self):
        return bool(self._connected)

    def is_gui(self):
        return bool(self._gui)

    def set_gravity(self, x, y, z):
        gravity = (float(x), float(y), float(z))
        if self.scene is not None and gravity != self._gravity:
            _log(f"WARNING: gravity {gravity} requested after the scene was created; "
                 f"Genesis fixes it at construction, keeping {self._gravity}.")
            return
        self._gravity = gravity

    def set_asset_search_path(self, path=None):
        """Record a fallback root for asset paths that are not relative to the cwd.

        Genesis resolves relative paths against the cwd and its own asset bundle, so this
        only matters for ``pybullet_data`` assets. The parent process passes its
        ``pybullet_data.getDataPath()`` down via ``LMTG_ASSET_ROOT`` because pybullet is
        deliberately not installed in the Genesis env.
        """
        if path is None:
            path = os.environ.get("LMTG_ASSET_ROOT")
        self._asset_root = path
        _log(f"asset search path = {path}")

    def _plane_surface(self):
        """Ground surface that looks like PyBullet's, not Genesis' dark default.

        Everything downstream of a capture is a vision model, so the floor is not
        cosmetic: Genesis' near-black default checkerboard drags the exposure of the
        whole frame down. Reuse PyBullet's own ``checker_blue.png`` when it is reachable
        through ``LMTG_ASSET_ROOT``, and fall back to a flat light colour when it is not.
        """
        import genesis as gs

        if self._asset_root:
            texture_path = os.path.join(self._asset_root, "checker_blue.png")
            if os.path.exists(texture_path):
                _log(f"plane textured from '{texture_path}'")
                return gs.surfaces.Default(
                    diffuse_texture=gs.textures.ImageTexture(image_path=texture_path))
        _log(f"plane using flat colour {PLANE_COLOR} (checker_blue.png not found)")
        return gs.surfaces.Default(color=PLANE_COLOR)

    def step(self):
        self._ensure_scene().step()

    def build(self):
        if self._built:
            return
        scene = self._ensure_scene()
        self._create_reserved_cameras()
        scene.build()
        self._built = True
        for body in self._bodies.values():
            body.refresh()
        _log(f"scene built: {len(self._bodies)} bodies, {len(self._cameras)} cameras, "
             f"{len(self._deferred)} deferred ops")
        self._replay_deferred()

    def _defer(self, description, fn):
        """Run ``fn`` now if the scene is built, otherwise queue it for :meth:`build`."""
        if self._built:
            fn()
            return
        self._deferred.append((description, fn))

    def _replay_deferred(self):
        pending, self._deferred = self._deferred, []
        for description, fn in pending:
            try:
                fn()
            except Exception as exc:
                # One bad deferred op must not abort the whole build; report it loudly and
                # keep going so the rest of the scene is still inspectable.
                _log(f"WARNING: deferred op '{description}' failed after build: "
                     f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ bodies
    def _body(self, handle):
        try:
            return self._bodies[int(handle)]
        except (KeyError, TypeError, ValueError):
            raise KeyError(f"Unknown body handle {handle!r}. Known: {sorted(self._bodies)}") from None

    def _register(self, entity, source):
        handle = self._next_handle
        self._next_handle += 1
        body = _Body(handle, entity, source)
        body.refresh()
        self._bodies[handle] = body
        return handle

    def _resolve_asset(self, path):
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        if self._asset_root:
            candidate = os.path.join(self._asset_root, path)
            if os.path.exists(candidate):
                return candidate
        # Let Genesis raise its own error, which names its asset bundle too.
        return path

    def load_urdf(self, path, position=None, orientation_q=None, fixed_base=False,
                  scaling=1.0, links_to_keep=None, merge_fixed_links=False):
        import genesis as gs

        scene = self._ensure_scene()
        if os.path.basename(path).lower() in _PLANE_URDFS:
            # PyBullet's plane.urdf is a large textured plate; Genesis' Plane morph is the
            # semantic equivalent and needs no pybullet_data on this interpreter.
            _log(f"'{path}' mapped to gs.morphs.Plane()")
            return self._register(
                scene.add_entity(gs.morphs.Plane(), surface=self._plane_surface()), path)

        kwargs = {
            "file": self._resolve_asset(path),
            "fixed": bool(fixed_base),
            # Genesis merges fixed-joint links by default, which deletes exactly the link
            # names the app layer looks up ('panda_grasptarget', 'panda_hand'). Keeping
            # them makes get_link_index_by_name behave like PyBullet.
            "merge_fixed_links": bool(merge_fixed_links),
        }
        if links_to_keep:
            kwargs["links_to_keep"] = list(links_to_keep)
        if position is not None:
            kwargs["pos"] = tuple(float(v) for v in position)
        if orientation_q is not None:
            kwargs["quat"] = tuple(transforms.xyzw_to_wxyz(orientation_q))
        if scaling is not None and float(scaling) != 1.0:
            kwargs["scale"] = float(scaling)

        entity = scene.add_entity(gs.morphs.URDF(**kwargs))
        handle = self._register(entity, path)
        body = self._body(handle)
        _log(f"loaded '{path}' as body {handle}: "
             f"{len(body.links)} links, {len(body.joints)} joints, {entity.n_dofs} dofs")
        return handle

    def remove_body(self, body):
        # Genesis has no entity removal; the scene topology is frozen at build().
        _log(f"WARNING: remove_body({body}) ignored - Genesis cannot remove entities. "
             f"Use debug markers for anything that has to disappear.")

    def get_base_pose(self, body):
        entity = self._body(body).entity
        pos = _vector(entity.get_pos(), 3)
        quat_wxyz = _vector(entity.get_quat(), 4)
        return pos, transforms.wxyz_to_xyzw(quat_wxyz)

    def get_aabb(self, body, link=-1):
        entity = self._body(body).entity
        getter = getattr(entity, "get_AABB", None)
        if getter is not None:
            try:
                aabb = _to_numpy(getter()).reshape(2, 3)
                return [float(v) for v in aabb[0]], [float(v) for v in aabb[1]]
            except Exception as exc:
                _log(f"get_aabb: entity.get_AABB() failed ({exc}); falling back to link positions")
        # Fallback: the bounding box of the link origins. Coarser than PyBullet's mesh
        # AABB, but this is only used for debug bounding-box drawing.
        pts = np.asarray([_vector(l.get_pos(), 3) for l in self._body(body).links], dtype=float)
        if pts.size == 0:
            pos = np.asarray(self.get_base_pose(body)[0], dtype=float)
            pts = pos.reshape(1, 3)
        return [float(v) for v in pts.min(axis=0)], [float(v) for v in pts.max(axis=0)]

    def change_dynamics(self, body, link, **kwargs):
        """Genesis sets friction per entity, not per link.

        ``lateralFriction`` therefore applies to the whole body. That is coarser than
        PyBullet, and it is why the door's latch friction is logged: if the door ever
        becomes ungrippable on Genesis, this is the first place to look.
        """
        friction = kwargs.get("lateralFriction")
        mass = kwargs.get("mass")
        if friction is None and mass is None:
            return

        def apply():
            entity = self._body(body).entity
            if friction is not None and hasattr(entity, "set_friction"):
                entity.set_friction(float(friction))
                _log(f"body {body}: friction={friction} (entity-wide; PyBullet asked for "
                     f"link {link} only)")
            if mass is not None and hasattr(entity, "set_mass"):
                entity.set_mass(float(mass))

        self._defer(f"change_dynamics(body={body}, link={link})", apply)

    def load_texture(self, path):
        if not self._warned_texture:
            _log(f"textures are applied at load time in Genesis; '{path}' ignored (cosmetic)")
            self._warned_texture = True
        return None

    def set_visual(self, body, link, texture=None, rgba=None):
        return None

    # -- primitive shapes.
    # PyBullet builds a body from separate collision + visual shape handles; Genesis takes
    # one morph. So the *_shape calls just record a spec and create_body assembles it.
    def create_visual_shape(self, shape, **kwargs):
        return {"shape": shape, "visual": True, **kwargs}

    def create_collision_shape(self, shape, **kwargs):
        return {"shape": shape, "visual": False, **kwargs}

    def create_body(self, mass=0.0, collision_shape=None, visual_shape=None,
                    position=None, orientation_q=None):
        import genesis as gs

        scene = self._ensure_scene()
        spec = collision_shape or visual_shape
        if spec is None:
            raise ValueError("create_body needs a collision or visual shape spec")
        merged = dict(spec)
        if visual_shape:
            merged.setdefault("rgba", visual_shape.get("rgba"))

        shape = merged["shape"]
        common = {}
        if position is not None:
            common["pos"] = tuple(float(v) for v in position)
        if orientation_q is not None:
            common["quat"] = tuple(transforms.xyzw_to_wxyz(orientation_q))
        # PyBullet's mass=0 means "static"; Genesis expresses that as a fixed morph.
        common["fixed"] = float(mass) <= 0.0

        if shape == "cylinder":
            morph = gs.morphs.Cylinder(radius=float(merged["radius"]),
                                       height=float(merged["length"]), **common)
        elif shape == "box":
            half = [float(v) for v in merged["half_extents"]]
            morph = gs.morphs.Box(size=tuple(2.0 * v for v in half), **common)
        elif shape == "sphere":
            morph = gs.morphs.Sphere(radius=float(merged["radius"]), **common)
        else:
            raise ValueError(f"Unsupported primitive shape '{shape}'")

        add_kwargs = {"morph": morph}
        rgba = merged.get("rgba")
        if rgba is not None:
            try:
                add_kwargs["surface"] = gs.surfaces.Default(
                    color=tuple(float(c) for c in list(rgba)[:3]))
            except Exception:
                pass
        entity = scene.add_entity(**add_kwargs)

        handle = self._register(entity, f"primitive:{shape}")
        if float(mass) > 0.0 and hasattr(entity, "set_mass"):
            self._defer(f"set_mass(body={handle})", lambda: entity.set_mass(float(mass)))
        _log(f"created {shape} body {handle} (mass={mass}, fixed={common['fixed']})")
        return handle

    def create_fixed_constraint(self, parent_body, parent_link, child_body, child_link,
                                parent_frame_position, child_frame_position,
                                child_frame_orientation_q=None):
        """Weld two links together (sawyer's robotiq gripper attachment).

        Genesis welds by *scene-level* link index and offers no frame offsets, so the
        PyBullet frame arguments are dropped. Only used by the sawyer path, which is out
        of scope for the Genesis port; implemented so it degrades loudly, not silently.
        """
        def apply():
            parent = self._body(parent_body).links[int(parent_link)]
            child = self._body(child_body).links[int(child_link)]
            solver = getattr(self.scene, "rigid_solver", None)
            if solver is None or not hasattr(solver, "add_weld_constraint"):
                _log("WARNING: this Genesis build has no add_weld_constraint; skipping weld")
                return
            solver.add_weld_constraint(parent.idx, child.idx)
            _log(f"welded link {parent.name} to {child.name} "
                 f"(frame offsets {parent_frame_position}/{child_frame_position} ignored)")

        self._defer("create_fixed_constraint", apply)
        return None

    # ------------------------------------------------------------------ joints & links
    def num_joints(self, body):
        return len(self._body(body).joints)

    def get_joint_info(self, body, joint):
        entry = self._body(body)
        gs_joint = entry.joints[int(joint)]
        type_name = getattr(getattr(gs_joint, "type", None), "name", "")
        joint_type = _JOINT_TYPE_NAMES.get(type_name, JOINT_OTHER)

        lower, upper = -math.inf, math.inf
        try:
            limits = _to_numpy(gs_joint.dofs_limit).reshape(-1, 2)
            if limits.size:
                lower, upper = float(limits[0, 0]), float(limits[0, 1])
        except Exception:
            pass

        # A joint's "child link" is the link it drives. Genesis exposes it directly, which
        # is what get_link_index_by_name relies on for PyBullet parity.
        child = getattr(gs_joint, "link", None)
        child_name = getattr(child, "name", "") or ""

        return JointInfo(
            index=int(joint),
            name=str(gs_joint.name),
            joint_type=joint_type,
            lower_limit=lower,
            upper_limit=upper,
            child_link_name=str(child_name),
        )

    def get_joint_index_by_name(self, body, name):
        return self._body(body).joint_names.get(name)

    def get_link_index_by_name(self, body, name):
        """Resolve against the **link** table, not via joints.

        Genesis drops fixed joints from ``entity.joints`` entirely, so the base class's
        joint-walking implementation cannot see a link like ``panda_grasptarget``.
        """
        return self._body(body).link_names.get(name)

    def get_joint_child_link(self, body, joint):
        """Genesis joint index != link index, so map through the joint's child link."""
        gs_joint = self._body(body).joints[int(joint)]
        return self._body(body).link_names.get(gs_joint.link.name)

    def list_joint_names(self, body):
        return dict(self._body(body).joint_names)

    def list_link_names(self, body):
        return dict(self._body(body).link_names)

    def _dofs(self, body, joint):
        gs_joint = self._body(body).joints[int(joint)]
        return [int(d) for d in _to_numpy(gs_joint.dofs_idx_local).reshape(-1)]

    def _dofs_many(self, body, joints):
        out = []
        for joint in joints:
            out.extend(self._dofs(body, joint))
        return out

    def get_joint_state(self, body, joint):
        entity = self._body(body).entity
        dofs = self._dofs(body, joint)
        if not dofs:
            return JointState(0.0, 0.0, 0.0)
        position = _scalar(entity.get_dofs_position(dofs_idx_local=dofs))
        velocity = _scalar(entity.get_dofs_velocity(dofs_idx_local=dofs))
        try:
            # PD control force, the closest analogue of PyBullet's applied motor torque.
            # get_dofs_force() would also include gravity and contacts.
            torque = _scalar(entity.get_dofs_control_force(dofs_idx_local=dofs))
        except Exception:
            torque = 0.0
        return JointState(position=position, velocity=velocity, applied_torque=torque)

    def reset_joint_state(self, body, joint, position):
        def apply():
            entity = self._body(body).entity
            dofs = self._dofs(body, joint)
            if dofs:
                entity.set_dofs_position(np.array([float(position)] * len(dofs)),
                                         dofs_idx_local=dofs)

        self._defer(f"reset_joint_state(body={body}, joint={joint})", apply)

    def get_link_pose(self, body, link):
        """Match PyBullet's ``getLinkState(...)[0]`` / ``[1]`` exactly.

        PyBullet reports the link's **centre-of-mass** (inertial) frame there, while
        Genesis' ``link.get_pos()`` reports the URDF link origin - PyBullet's ``[4]``.
        On the adroit door the two differ by up to 29 cm, so the wrong choice would
        silently move every perceived handle/hinge position on Genesis. Measured
        bit-identical to PyBullet once ``link_COM`` is requested.
        """
        gs_link = self._body(body).links[int(link)]
        pos, quat_wxyz = self._link_com_pose(gs_link)
        return _vector(pos, 3), transforms.wxyz_to_xyzw(_vector(quat_wxyz, 4))

    @staticmethod
    def _link_com_pose(gs_link):
        """``(pos, quat_wxyz)`` of the link's inertial frame, with a safe fallback."""
        try:
            from genesis.engine.solvers.rigid.rigid_solver import link_ref_frame
        except Exception:
            # Older/renamed Genesis: fall back to the link origin rather than crashing.
            # Positions will be off by the inertial offset - loudly, not silently.
            _log("WARNING link_ref_frame unavailable; link poses use the URDF origin "
                 "frame and will not match PyBullet's centre-of-mass convention")
            return gs_link.get_pos(), gs_link.get_quat()
        solver = gs_link._solver
        pos = solver.get_links_pos(gs_link._idx, ref=link_ref_frame.link_COM)
        # There is no ``ref`` for orientation: PyBullet's centre-of-mass frame and URDF
        # link frame have the same orientation unless the URDF's <inertial><origin> has
        # a non-zero rpy, which neither the panda nor the adroit door use (verified).
        quat = solver.get_links_quat(gs_link._idx, relative=False)
        return _to_numpy(pos).ravel(), _to_numpy(quat).ravel()

    def set_joint_position(self, body, joint, target, force=None, position_gain=None):
        def apply():
            entity = self._body(body).entity
            dofs = self._dofs(body, joint)
            if not dofs:
                return
            if force is not None:
                self._apply_force_range(entity, dofs, force)
            if target is not None:
                entity.control_dofs_position(np.array([float(target)] * len(dofs)),
                                             dofs_idx_local=dofs)

        self._defer(f"set_joint_position(body={body}, joint={joint})", apply)

    def set_joint_positions(self, body, joints, targets, forces=None, position_gains=None):
        joints = list(joints)
        targets = [float(t) for t in targets]

        def apply():
            entity = self._body(body).entity
            dofs, expanded = [], []
            for joint, target in zip(joints, targets):
                joint_dofs = self._dofs(body, joint)
                dofs.extend(joint_dofs)
                expanded.extend([target] * len(joint_dofs))
            if not dofs:
                return
            if forces is not None:
                self._apply_force_range(entity, dofs, forces)
            entity.control_dofs_position(np.asarray(expanded, dtype=np.float64),
                                         dofs_idx_local=dofs)

        self._defer(f"set_joint_positions(body={body}, n={len(joints)})", apply)

    def set_joint_velocity(self, body, joint, target_velocity, force=None):
        def apply():
            entity = self._body(body).entity
            dofs = self._dofs(body, joint)
            if not dofs:
                return
            if force is not None:
                self._apply_force_range(entity, dofs, force)
            entity.control_dofs_velocity(
                np.array([float(target_velocity)] * len(dofs)), dofs_idx_local=dofs)

        self._defer(f"set_joint_velocity(body={body}, joint={joint})", apply)

    @staticmethod
    def _apply_force_range(entity, dofs, force):
        """PyBullet's ``forces=[F, ...]`` maps to a symmetric Genesis force range."""
        if np.isscalar(force):
            magnitudes = np.full(len(dofs), abs(float(force)))
        else:
            values = [abs(float(f)) for f in force]
            if len(values) == len(dofs):
                magnitudes = np.asarray(values)
            else:
                # set_joint_positions gets one force per *joint*; a multi-DOF joint
                # expands to several DOFs, so fall back to a uniform value.
                magnitudes = np.full(len(dofs), values[0] if values else 0.0)
        entity.set_dofs_force_range(-magnitudes, magnitudes, dofs_idx_local=dofs)

    def configure_motor_gains(self, body, joints, kp=None, kv=None, force_range=None):
        """Genesis position control does nothing without explicit PD gains.

        Values come from ``RobotProfile.genesis_kp`` / ``genesis_kv`` /
        ``genesis_force_range``. PyBullet has implicit gains and no-ops this.
        """
        joints = list(joints)

        def apply():
            entity = self._body(body).entity
            dofs = self._dofs_many(body, joints)
            if not dofs:
                _log(f"WARNING: configure_motor_gains found no DOFs for body {body}")
                return
            n = len(dofs)

            def sized(values):
                arr = np.asarray([float(v) for v in values], dtype=np.float64)
                if arr.size == n:
                    return arr
                _log(f"WARNING: gain vector of length {arr.size} for {n} DOFs; "
                     f"{'truncating' if arr.size > n else 'padding with the last value'}")
                if arr.size > n:
                    return arr[:n]
                return np.concatenate([arr, np.full(n - arr.size, arr[-1] if arr.size else 0.0)])

            if kp is not None:
                entity.set_dofs_kp(sized(kp), dofs_idx_local=dofs)
            if kv is not None:
                entity.set_dofs_kv(sized(kv), dofs_idx_local=dofs)
            if force_range is not None:
                limits = sized(force_range)
                entity.set_dofs_force_range(-np.abs(limits), np.abs(limits),
                                            dofs_idx_local=dofs)
            # PyBullet arms do not collapse under gravity when nothing has commanded
            # them yet: every joint has a default motor enabled. In Genesis, setting
            # kp/kv alone leaves the DOFs uncontrolled, so the arm free-falls out of
            # its start posture before the first move(). Latch the current positions
            # as the initial target to reproduce PyBullet's "motors are on" default.
            hold = _to_numpy(entity.get_dofs_position(dofs_idx_local=dofs)).ravel()
            entity.control_dofs_position(hold, dofs_idx_local=dofs)
            _log(f"body {body}: PD gains set on {n} DOFs "
                 f"(kp={kp is not None}, kv={kv is not None}, force={force_range is not None}); "
                 f"holding at {np.round(hold, 4).tolist()}")

        self._defer(f"configure_motor_gains(body={body})", apply)

    def inverse_kinematics(self, body, link, position, orientation_q=None,
                           lower_limits=None, upper_limits=None, joint_ranges=None,
                           rest_poses=None, max_iterations=500):
        """Return targets for the movable joints, in ``get_movable_joints`` order.

        Genesis returns a full qpos over all entity DOFs, which for a URDF-loaded arm is
        exactly the movable-joint DOFs in declaration order - the same thing PyBullet's
        ``calculateInverseKinematics`` returns. The PyBullet joint-limit arguments are
        dropped: Genesis reads limits from the URDF via ``respect_joint_limit``.
        """
        entry = self._body(body)
        gs_link = entry.links[int(link)]
        kwargs = {"link": gs_link, "pos": np.asarray([float(v) for v in position])}
        if orientation_q is not None:
            kwargs["quat"] = np.asarray(transforms.xyzw_to_wxyz(orientation_q))
        qpos = entry.entity.inverse_kinematics(**kwargs)
        return _vector(qpos)

    # ------------------------------------------------------------------ transforms
    # Delegated to PyBullet-convention numpy math (see sim_adapter/transforms.py) rather
    # than Genesis' degree-based helpers, so both simulators agree exactly.
    def quat_from_euler(self, euler):
        return transforms.quat_from_euler(euler)

    def euler_from_quat(self, quat_xyzw):
        return transforms.euler_from_quat(quat_xyzw)

    def matrix_from_quat(self, quat_xyzw):
        return transforms.matrix_from_quat(quat_xyzw)

    def quat_from_axis_angle(self, axis, angle):
        return transforms.quat_from_axis_angle(axis, angle)

    # ------------------------------------------------------------------ cameras
    def compute_projection_matrix(self, fov, aspect, near, far):
        from sim_adapter.camera_math import gl_projection_matrix
        return gl_projection_matrix(fov, aspect, near, far)

    def compute_view_matrix(self, eye, target, up):
        from sim_adapter.camera_math import gl_view_matrix
        return gl_view_matrix(eye, target, up)

    def compute_view_matrix_from_yaw_pitch_roll(self, target, distance, yaw_deg, pitch_deg,
                                                roll_deg=0.0, up_axis_index=2):
        eye, _ = spherical_camera_pose(target, distance, yaw_deg, pitch_deg)
        return self.compute_view_matrix(eye, target, [0.0, 0.0, 1.0])

    def reserve_camera(self, width, height, fov=None, near=None, far=None):
        """Declare a render resolution **before** :meth:`build`.

        ``scene.add_camera`` is ``@gs.assert_unbuilt``, so every resolution the app will
        ever render at has to be known up front. ``genesis_env.py`` reserves the one that
        ``robot.py`` uses; this is a no-op on PyBullet.
        """
        if fov is not None:
            self._camera_fov = float(fov)
        if near is not None:
            self._camera_near = float(near)
        if far is not None:
            self._camera_far = float(far)
        self._reserved.add((int(width), int(height)))

    def _create_reserved_cameras(self):
        import genesis as gs

        scene = self._ensure_scene()
        if not self._reserved:
            self._reserved.add((int(config.image_width), int(config.image_height)))
        for width, height in sorted(self._reserved):
            if (width, height) in self._cameras:
                continue
            self._cameras[(width, height)] = scene.add_camera(
                res=(width, height),
                pos=(1.0, 1.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov=self._camera_fov,
                near=self._camera_near,
                far=self._camera_far,
                GUI=False,
                # Without this, scene.draw_debug_* is invisible to offscreen renders
                # (vis/rasterizer.py: skip_markers = not camera.debug) and every debug
                # overlay the agent asks for would silently vanish from the captures.
                debug=True,
            )
            _log(f"camera reserved {width}x{height} fov={self._camera_fov} "
                 f"near={self._camera_near} far={self._camera_far}")

    @staticmethod
    def _decompose_projection(projection_matrix):
        """Recover (fov_deg, aspect, near, far) from a flat 16 column-major GL matrix."""
        m = np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")
        f = float(m[1, 1])
        fov = 2.0 * math.degrees(math.atan(1.0 / f)) if f else float("nan")
        aspect = f / float(m[0, 0]) if m[0, 0] else float("nan")
        c, d = float(m[2, 2]), float(m[2, 3])
        near = d / (c - 1.0) if c != 1.0 else float("nan")
        far = d / (c + 1.0) if c != -1.0 else float("nan")
        return fov, aspect, near, far

    def _check_projection(self, projection_matrix):
        """Warn once if the caller's projection disagrees with the built camera.

        Genesis bakes fov/near/far into the camera at creation, so a mismatch cannot be
        honoured - it would show up as subtly wrong 3D reconstruction rather than an
        error, which is exactly the failure mode worth shouting about.
        """
        if self._warned_projection:
            return
        try:
            fov, _aspect, near, far = self._decompose_projection(projection_matrix)
        except Exception:
            return
        if (abs(fov - self._camera_fov) > 1e-3
                or abs(near - self._camera_near) > 1e-6
                or abs(far - self._camera_far) > 1e-3):
            _log(f"WARNING: requested projection (fov={fov:.4f}, near={near:.6f}, "
                 f"far={far:.4f}) differs from the built camera "
                 f"(fov={self._camera_fov}, near={self._camera_near}, far={self._camera_far}). "
                 f"Genesis cannot change these after build(); rendering with the camera's.")
            self._warned_projection = True

    def render_camera(self, width, height, view_matrix, projection_matrix):
        if not self._built:
            raise RuntimeError("render_camera() before build(); Genesis cannot render an "
                               "unbuilt scene.")
        key = (int(width), int(height))
        camera = self._cameras.get(key)
        if camera is None:
            raise RuntimeError(
                f"No Genesis camera reserved at {width}x{height}. Cameras must be created "
                f"before scene.build(); call sim.reserve_camera({width}, {height}) first. "
                f"Reserved: {sorted(self._cameras)}"
            )
        self._check_projection(projection_matrix)

        # view (world->camera, column-major flat) -> transform (camera->world), which is
        # what Genesis' set_pose takes. Verified against PyBullet in
        # tests/test_genesis_camera_semantics.py.
        view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
        camera.set_pose(transform=np.linalg.inv(view))

        rgb, depth, seg, _normal = camera.render(rgb=True, depth=True)
        rgb_array = np.asarray(rgb)
        if rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array * 255.0, 0, 255).astype(np.uint8)
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.asarray(depth, dtype=np.float32)
        if depth_array.ndim == 3:
            depth_array = depth_array[..., 0]

        return CameraFrame(width=key[0], height=key[1], rgb=rgb_array,
                           depth=depth_array, segmentation=seg)

    def get_camera_view_matrix(self, width, height):
        """Current view matrix of a built camera, recomputed (never the stale cache).

        ``camera.extrinsics`` is a ``@cached_property`` that does not refresh after
        ``set_pose`` - see ``tests/test_genesis_camera_semantics.py``.
        """
        camera = self._cameras.get((int(width), int(height)))
        if camera is None:
            return None
        view = np.linalg.inv(np.asarray(camera.transform, dtype=np.float64))
        return [float(v) for v in view.flatten(order="F")]

    # ------------------------------------------------------------------ viewer & debug draw
    def _viewer(self):
        return getattr(self.scene, "viewer", None) if self.scene is not None else None

    def reset_viewer_camera(self, distance, yaw, pitch, target):
        viewer = self._viewer()
        if viewer is None:
            return
        eye, _ = spherical_camera_pose(target, distance, yaw, pitch)
        try:
            viewer.set_camera_pose(pos=tuple(eye),
                                   lookat=tuple(float(v) for v in target))
        except Exception as exc:
            _log(f"reset_viewer_camera failed: {exc}")

    def get_viewer_camera(self):
        viewer = self._viewer()
        if viewer is None:
            return {"available": False, "tuple_len": None}
        try:
            eye = _vector(viewer.camera_pos, 3)
            target = _vector(viewer.camera_lookat, 3)
        except Exception:
            return {"available": False, "tuple_len": None}

        delta = np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)
        distance = float(np.linalg.norm(delta))
        # Invert spherical_camera_pose: eye = target - distance * forward, with
        # forward = (-cos(p)sin(y), cos(p)cos(y), sin(p)).
        forward = -delta / distance if distance > 1e-9 else np.array([0.0, 1.0, 0.0])
        pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
        yaw = math.degrees(math.atan2(-float(forward[0]), float(forward[1])))
        return {
            "available": True,
            "tuple_len": 12,  # matches PyBullet's getDebugVisualizerCamera arity
            "view_matrix": self.compute_view_matrix(eye, target, [0.0, 0.0, 1.0]),
            "projection_matrix": self.compute_projection_matrix(
                self._camera_fov, config.aspect, self._camera_near, self._camera_far),
            "yaw": yaw,
            "pitch": pitch,
            "distance": distance,
            "target": target,
        }

    def set_viewer_options(self, show_gui=None, shadows=None):
        return None

    def set_real_time(self, enabled):
        # Genesis' viewer always renders on its own thread but physics only advances on
        # scene.step(); the interactive demo loop steps explicitly instead.
        return None

    def viewer_is_open(self):
        viewer = self._viewer()
        if viewer is None:
            return False
        checker = getattr(viewer, "is_alive", None)
        try:
            return bool(checker()) if callable(checker) else True
        except Exception:
            return False

    def get_click_events(self):
        return []

    # -- debug primitives. Genesis returns pyrender nodes; we hand out int handles so the
    # app layer keeps treating marker ids the way PyBullet taught it to.
    def _add_debug(self, node):
        if node is None:
            return None
        handle = self._next_debug_handle
        self._next_debug_handle += 1
        self._debug_nodes[handle] = node
        return handle

    @staticmethod
    def _rgba(color, default_alpha=1.0):
        values = [float(c) for c in list(color)[:4]]
        while len(values) < 3:
            values.append(0.0)
        if len(values) == 3:
            values.append(default_alpha)
        return tuple(values)

    def draw_debug_line(self, start, end, color, line_width=1.0, life_time=0.0):
        def draw():
            # Genesis draws a thin cylinder, so line_width becomes a radius in metres.
            return self.scene.draw_debug_line(
                start=tuple(float(v) for v in start),
                end=tuple(float(v) for v in end),
                radius=max(0.001, float(line_width) * 0.002),
                color=self._rgba(color),
            )

        if not self._built:
            self._defer("draw_debug_line", draw)
            return None
        return self._add_debug(draw())

    def draw_debug_points(self, points, colors, point_size=5.0, life_time=0.0):
        def draw():
            poss = np.asarray([[float(v) for v in pt] for pt in points], dtype=np.float64)
            first = colors[0] if isinstance(colors, (list, tuple)) and colors else (1, 0, 0)
            if np.isscalar(first):
                first = colors
            return self.scene.draw_debug_points(poss=poss, colors=self._rgba(first))

        if not self._built:
            self._defer("draw_debug_points", draw)
            return None
        return self._add_debug(draw())

    def draw_marker_sphere(self, position, radius, color):
        """Native Genesis debug sphere.

        No massless-MultiBody trick needed: cameras are created with ``debug=True`` so
        markers appear in offscreen captures.
        """
        if not self._built:
            self._defer("draw_marker_sphere", lambda: self.draw_marker_sphere(position, radius, color))
            return None
        try:
            return self._add_debug(self.scene.draw_debug_sphere(
                pos=tuple(float(v) for v in position),
                radius=float(radius),
                color=self._rgba(color),
            ))
        except Exception as exc:
            _log(f"WARNING: draw_marker_sphere failed: {exc}")
            return None

    def draw_marker_cylinder(self, start, end, radius, color):
        if not self._built:
            self._defer("draw_marker_cylinder",
                        lambda: self.draw_marker_cylinder(start, end, radius, color))
            return None
        a = np.asarray([float(v) for v in start], dtype=float)
        b = np.asarray([float(v) for v in end], dtype=float)
        if not np.isfinite(np.linalg.norm(b - a)) or np.linalg.norm(b - a) < 1e-6:
            return self.draw_marker_sphere(a.tolist(), radius * 1.5, color)
        try:
            # draw_debug_line already renders a cylinder of the given radius between the
            # two points, so this needs none of PyBullet's quaternion gymnastics.
            return self._add_debug(self.scene.draw_debug_line(
                start=tuple(a), end=tuple(b), radius=float(radius), color=self._rgba(color)))
        except Exception as exc:
            _log(f"WARNING: draw_marker_cylinder failed: {exc}")
            return None

    def remove_marker(self, handle):
        if handle is None:
            return
        node = self._debug_nodes.pop(handle, None)
        if node is None:
            return
        try:
            self.scene.clear_debug_object(node)
        except Exception as exc:
            _log(f"WARNING: remove_marker({handle}) failed: {exc}")

    def clear_debug_markers(self):
        """Drop every debug object at once (used by the interactive demo)."""
        self._debug_nodes.clear()
        try:
            self.scene.clear_debug_objects()
        except Exception as exc:
            _log(f"WARNING: clear_debug_markers failed: {exc}")
