"""Simulator abstraction: the only layer that knows which physics engine is running.

Design rules
------------
* **App logic never lives here.** ``env.py`` keeps every IPC handler
  (``EXECUTE_TRAJECTORY``, ``CAPTURE_IMAGES``, ...) so an app-level change is made in
  exactly one place regardless of simulator.
* **This layer owns only primitives**: load a body, read a joint, drive a motor, render a
  camera, draw a debug marker.
* **Signatures mirror PyBullet** where the semantics genuinely match, so the port of
  ``env.py`` / ``robot.py`` is mechanical (``p.X(...)`` -> ``self.sim.X(...)``) and therefore
  reviewable against the golden traces.
* **Conventions are normalised at this boundary, not above it.** Every adapter returns
  quaternions as ``xyzw``, Euler angles in **radians**, plain Python floats/lists (never
  torch tensors), and resolves joints and links **by name** where the caller has a choice.
  Genesis natively uses ``wxyz``, degrees and GPU tensors; that conversion is the adapter's
  job and must never leak upward.

Adding a simulator = implement :class:`SimAdapter` + add sim-env profiles + register it.
Nothing in ``env.py`` or ``robot.py`` should need to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Joint types, named so callers never import a simulator's enum.
JOINT_REVOLUTE = "revolute"
JOINT_PRISMATIC = "prismatic"
JOINT_FIXED = "fixed"
JOINT_OTHER = "other"

MOVABLE_JOINT_TYPES = (JOINT_REVOLUTE, JOINT_PRISMATIC)

# Values for ``SimAdapter.depth_encoding``, consumed by ``utils.get_world_point_world_frame``.
DEPTH_OPENGL = "opengl"                 # non-linear [0, 1] z-buffer (PyBullet)
DEPTH_LINEAR_METRIC = "linear_metric"   # metres along the optical axis (Genesis)


@dataclass
class JointInfo:
    """Simulator-neutral joint description.

    ``index`` is whatever handle the adapter needs; callers must treat it as opaque and
    look joints up **by name**. PyBullet joint indices and Genesis DOF indices do not
    correspond (a fixed joint has 0 DOFs in Genesis, a free joint has 6).
    """

    index: Any
    name: str
    joint_type: str
    lower_limit: float
    upper_limit: float
    child_link_name: str

    @property
    def is_movable(self) -> bool:
        return self.joint_type in MOVABLE_JOINT_TYPES


@dataclass
class JointState:
    position: float
    velocity: float
    applied_torque: float


@dataclass
class CameraFrame:
    """One render. ``depth`` is raw, in the adapter's ``depth_encoding``."""

    width: int
    height: int
    rgb: Any          # (H, W, 3) uint8
    depth: Any        # (H, W) float32
    segmentation: Any = None


@dataclass
class CameraParams:
    """Everything the agent needs to unproject a pixel, in PyBullet's layout.

    ``view_matrix`` and ``projection_matrix`` are **flat 16-element** sequences in
    column-major order, exactly as ``p.computeViewMatrix`` returns them, because
    ``utils.get_intrinsics_extrinsics`` does ``reshape(4, 4, order='F')``. Genesis
    matrices are row-major 4x4 and must be flattened with ``order='F'`` here.
    """

    position: List[float]
    orientation_q: List[float]
    view_matrix: Sequence[float]
    projection_matrix: Sequence[float]
    near: float
    far: float
    extra: Dict[str, Any] = field(default_factory=dict)


class SimAdapter(ABC):
    """Primitives every simulator provider must supply."""

    name = "abstract"

    #: How this simulator encodes its depth buffer. Sent to the agent in ``cam_info`` so the
    #: unprojection math picks the right branch. Measured, not assumed - see
    #: ``tests/test_genesis_camera_semantics.py``.
    depth_encoding = DEPTH_OPENGL

    # ------------------------------------------------------------------ lifecycle
    @abstractmethod
    def connect(self, gui: bool = False) -> None:
        """Start the simulator. ``gui=True`` opens an interactive viewer."""

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def is_gui(self) -> bool:
        ...

    def build(self) -> None:
        """Finalise the scene. No-op on PyBullet.

        Genesis requires ``scene.build()`` before stepping and forbids adding entities
        afterwards, so every provider gets an explicit build phase and callers must load
        all bodies before calling it.
        """
        return None

    @abstractmethod
    def set_gravity(self, x: float, y: float, z: float) -> None:
        ...

    @abstractmethod
    def set_asset_search_path(self, path: str) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance one control timestep (``config.control_dt``)."""

    # ------------------------------------------------------------------ bodies
    @abstractmethod
    def load_urdf(self, path: str, position=None, orientation_q=None,
                  fixed_base: bool = False, scaling: float = 1.0) -> Any:
        """Load a URDF and return an opaque body handle. ``orientation_q`` is xyzw."""

    @abstractmethod
    def remove_body(self, body: Any) -> None:
        ...

    @abstractmethod
    def get_base_pose(self, body: Any) -> Tuple[List[float], List[float]]:
        """Return ``(position, orientation_xyzw)``."""

    @abstractmethod
    def get_aabb(self, body: Any, link: int = -1) -> Tuple[List[float], List[float]]:
        """Return ``(aabb_min, aabb_max)`` in world coordinates."""

    @abstractmethod
    def change_dynamics(self, body: Any, link: int, **kwargs) -> None:
        """Set physical properties (``lateralFriction``, ``spinningFriction``, ``mass``...)."""

    def load_texture(self, path: str) -> Any:
        """Return a texture handle, or ``None`` if unsupported (purely cosmetic)."""
        return None

    def set_visual(self, body: Any, link: int, texture: Any = None, rgba=None) -> None:
        """Apply a texture and/or colour. Cosmetic; safe to no-op."""
        return None

    @abstractmethod
    def create_visual_shape(self, shape: str, **kwargs) -> Any:
        """``shape`` is one of 'sphere', 'cylinder', 'box'."""

    @abstractmethod
    def create_collision_shape(self, shape: str, **kwargs) -> Any:
        ...

    @abstractmethod
    def create_body(self, mass: float = 0.0, collision_shape: Any = None,
                    visual_shape: Any = None, position=None, orientation_q=None) -> Any:
        ...

    @abstractmethod
    def create_fixed_constraint(self, parent_body: Any, parent_link: int,
                                child_body: Any, child_link: int,
                                parent_frame_position, child_frame_position,
                                child_frame_orientation_q=None) -> Any:
        ...

    # ------------------------------------------------------------------ joints & links
    @abstractmethod
    def num_joints(self, body: Any) -> int:
        ...

    @abstractmethod
    def get_joint_info(self, body: Any, joint: Any) -> JointInfo:
        ...

    def get_movable_joints(self, body: Any) -> List[JointInfo]:
        """Movable joints in declaration order. Shared: derived from ``get_joint_info``."""
        out = []
        for j in range(self.num_joints(body)):
            info = self.get_joint_info(body, j)
            if info.is_movable:
                out.append(info)
        return out

    def get_joint_index_by_name(self, body: Any, name: str) -> Optional[Any]:
        for j in range(self.num_joints(body)):
            info = self.get_joint_info(body, j)
            if info.name == name:
                return info.index
        return None

    def get_link_index_by_name(self, body: Any, name: str) -> Optional[Any]:
        for j in range(self.num_joints(body)):
            info = self.get_joint_info(body, j)
            if info.child_link_name == name:
                return info.index
        return None

    def list_joint_names(self, body: Any) -> Dict[str, Any]:
        return {i.name: i.index for i in
                (self.get_joint_info(body, j) for j in range(self.num_joints(body)))}

    def list_link_names(self, body: Any) -> Dict[str, Any]:
        return {i.child_link_name: i.index for i in
                (self.get_joint_info(body, j) for j in range(self.num_joints(body)))}

    @abstractmethod
    def get_joint_state(self, body: Any, joint: Any) -> JointState:
        ...

    @abstractmethod
    def reset_joint_state(self, body: Any, joint: Any, position: float) -> None:
        ...

    @abstractmethod
    def get_link_pose(self, body: Any, link: Any) -> Tuple[List[float], List[float]]:
        """Return ``(position, orientation_xyzw)`` with forward kinematics applied."""

    @abstractmethod
    def set_joint_position(self, body: Any, joint: Any, target: float,
                           force: Optional[float] = None,
                           position_gain: Optional[float] = None) -> None:
        ...

    @abstractmethod
    def set_joint_positions(self, body: Any, joints: Sequence[Any], targets: Sequence[float],
                            forces=None, position_gains=None) -> None:
        ...

    @abstractmethod
    def set_joint_velocity(self, body: Any, joint: Any, target_velocity: float,
                           force: Optional[float] = None) -> None:
        ...

    @abstractmethod
    def inverse_kinematics(self, body: Any, link: Any, position,
                           orientation_q=None, lower_limits=None, upper_limits=None,
                           joint_ranges=None, rest_poses=None,
                           max_iterations: int = 500) -> List[float]:
        """Target positions **for the movable joints, in order** - never a full qpos."""

    # ------------------------------------------------------------------ transforms
    @abstractmethod
    def quat_from_euler(self, euler) -> List[float]:
        """Euler (radians, PyBullet convention) -> quaternion xyzw."""

    @abstractmethod
    def euler_from_quat(self, quat_xyzw) -> List[float]:
        ...

    @abstractmethod
    def matrix_from_quat(self, quat_xyzw) -> List[float]:
        """Return the row-major 9-element rotation matrix."""

    @abstractmethod
    def quat_from_axis_angle(self, axis, angle: float) -> List[float]:
        ...

    # ------------------------------------------------------------------ cameras
    @abstractmethod
    def compute_projection_matrix(self, fov: float, aspect: float,
                                  near: float, far: float) -> Sequence[float]:
        """Flat 16, column-major. Same dims and meaning across all providers."""

    @abstractmethod
    def compute_view_matrix(self, eye, target, up) -> Sequence[float]:
        """Flat 16, column-major."""

    @abstractmethod
    def compute_view_matrix_from_yaw_pitch_roll(self, target, distance: float, yaw_deg: float,
                                                pitch_deg: float, roll_deg: float = 0.0,
                                                up_axis_index: int = 2) -> Sequence[float]:
        ...

    @abstractmethod
    def render_camera(self, width: int, height: int, view_matrix, projection_matrix) -> CameraFrame:
        ...

    # ------------------------------------------------------------------ viewer & debug draw
    def reset_viewer_camera(self, distance: float, yaw: float, pitch: float, target) -> None:
        return None

    def get_viewer_camera(self) -> Optional[dict]:
        """``{'yaw','pitch','distance','target','view_matrix','projection_matrix'}`` or None."""
        return None

    def set_viewer_options(self, show_gui: Optional[bool] = None,
                           shadows: Optional[bool] = None) -> None:
        return None

    def set_real_time(self, enabled: bool) -> None:
        """Let the viewer advance physics on its own clock so joints can be dragged.

        Interactive-debug only; the production loop always steps explicitly.
        """
        return None

    def viewer_is_open(self) -> bool:
        """True while an interactive window is up - the manual-inspection loop condition."""
        return self.is_gui() and self.is_connected()

    def get_click_events(self) -> List[Any]:
        """Mouse-button-down events since the last call, for click-to-print debugging.

        Empty list when the simulator exposes no such hook.
        """
        return []

    @abstractmethod
    def draw_debug_line(self, start, end, color, line_width: float = 1.0,
                        life_time: float = 0.0) -> Any:
        """Overlay line. May be invisible to offscreen cameras (it is on PyBullet)."""

    @abstractmethod
    def draw_debug_points(self, points, colors, point_size: float = 5.0,
                          life_time: float = 0.0) -> Any:
        ...

    @abstractmethod
    def draw_marker_sphere(self, position, radius: float, color) -> Any:
        """A sphere that **is** visible to offscreen cameras.

        PyBullet needs a massless MultiBody for this (overlay debug draws do not appear in
        ``getCameraImage``); Genesis renders native debug markers into ``debug=True``
        cameras. Both return an opaque handle for :meth:`remove_marker`.
        """

    @abstractmethod
    def draw_marker_cylinder(self, start, end, radius: float, color) -> Any:
        """A camera-visible cylinder spanning two world points."""

    @abstractmethod
    def remove_marker(self, handle: Any) -> None:
        ...
