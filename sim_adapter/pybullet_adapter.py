"""PyBullet implementation of :class:`~sim_adapter.base.SimAdapter`.

Deliberately thin: every method is a 1:1 wrapper around the ``p.*`` call the pre-refactor
code made, with the same arguments in the same order. That is what makes the port
diff-reviewable against ``tests/golden/pybullet/*.jsonl`` at ``atol=1e-9``.

The only logic that lives here is buffer reshaping and the massless-MultiBody marker trick,
both of which are PyBullet quirks that must not leak into ``env.py``.
"""

import math
import traceback

import numpy as np
import pybullet as p
import pybullet_data

from sim_adapter.base import (
    DEPTH_OPENGL,
    JOINT_FIXED,
    JOINT_OTHER,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    CameraFrame,
    JointInfo,
    JointState,
    SimAdapter,
)

_JOINT_TYPE_NAMES = {
    p.JOINT_REVOLUTE: JOINT_REVOLUTE,
    p.JOINT_PRISMATIC: JOINT_PRISMATIC,
    p.JOINT_FIXED: JOINT_FIXED,
}

_SHAPE_TYPES = {
    "sphere": p.GEOM_SPHERE,
    "cylinder": p.GEOM_CYLINDER,
    "box": p.GEOM_BOX,
}


def _decode(value):
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


class PyBulletAdapter(SimAdapter):

    name = "pybullet"
    depth_encoding = DEPTH_OPENGL

    def __init__(self):
        self.client_id = None
        self._gui = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self, gui=False):
        self._gui = bool(gui)
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        return self.client_id

    def disconnect(self):
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass
        self.client_id = None

    def is_connected(self):
        try:
            return bool(p.isConnected())
        except Exception:
            return False

    def is_gui(self):
        try:
            return bool(p.isConnected()) and p.getConnectionInfo()[1] == p.GUI
        except Exception:
            return False

    def set_gravity(self, x, y, z):
        p.setGravity(x, y, z)

    def set_asset_search_path(self, path=None):
        p.setAdditionalSearchPath(path if path is not None else pybullet_data.getDataPath())

    def step(self):
        p.stepSimulation()

    # ------------------------------------------------------------------ bodies
    def load_urdf(self, path, position=None, orientation_q=None, fixed_base=False, scaling=1.0):
        kwargs = {}
        if fixed_base:
            kwargs["useFixedBase"] = True
        if scaling is not None and float(scaling) != 1.0:
            kwargs["globalScaling"] = float(scaling)
        if position is None:
            return p.loadURDF(path, **kwargs)
        if orientation_q is None:
            return p.loadURDF(path, position, **kwargs)
        return p.loadURDF(path, position, orientation_q, **kwargs)

    def remove_body(self, body):
        p.removeBody(int(body))

    def get_base_pose(self, body):
        pos, quat = p.getBasePositionAndOrientation(body)
        return list(map(float, pos)), list(map(float, quat))

    def get_aabb(self, body, link=-1):
        aabb_min, aabb_max = p.getAABB(body, link)
        return list(map(float, aabb_min)), list(map(float, aabb_max))

    def change_dynamics(self, body, link, **kwargs):
        p.changeDynamics(body, link, **kwargs)

    def load_texture(self, path):
        return p.loadTexture(path)

    def set_visual(self, body, link, texture=None, rgba=None):
        kwargs = {}
        if texture is not None:
            kwargs["textureUniqueId"] = texture
        if rgba is not None:
            kwargs["rgbaColor"] = list(rgba)
        if kwargs:
            p.changeVisualShape(body, link, **kwargs)

    def create_visual_shape(self, shape, **kwargs):
        return p.createVisualShape(_SHAPE_TYPES[shape], **self._shape_kwargs(shape, kwargs, visual=True))

    def create_collision_shape(self, shape, **kwargs):
        return p.createCollisionShape(_SHAPE_TYPES[shape], **self._shape_kwargs(shape, kwargs, visual=False))

    @staticmethod
    def _shape_kwargs(shape, kwargs, visual):
        """Translate the adapter's neutral shape kwargs to PyBullet's names.

        PyBullet spells a cylinder's extent ``length`` for visual shapes but ``height`` for
        collision shapes; hiding that asymmetry here is the whole point of the adapter.
        """
        out = {}
        if "radius" in kwargs:
            out["radius"] = float(kwargs["radius"])
        if "length" in kwargs:
            out["length" if visual else "height"] = float(kwargs["length"])
        if "half_extents" in kwargs:
            out["halfExtents"] = list(map(float, kwargs["half_extents"]))
        if visual and kwargs.get("rgba") is not None:
            out["rgbaColor"] = list(kwargs["rgba"])
        return out

    def create_body(self, mass=0.0, collision_shape=None, visual_shape=None,
                    position=None, orientation_q=None):
        kwargs = {"baseMass": float(mass)}
        if collision_shape is not None:
            kwargs["baseCollisionShapeIndex"] = collision_shape
        if visual_shape is not None:
            kwargs["baseVisualShapeIndex"] = visual_shape
        if position is not None:
            kwargs["basePosition"] = list(map(float, position))
        if orientation_q is not None:
            kwargs["baseOrientation"] = list(orientation_q)
        return p.createMultiBody(**kwargs)

    def create_fixed_constraint(self, parent_body, parent_link, child_body, child_link,
                                parent_frame_position, child_frame_position,
                                child_frame_orientation_q=None):
        return p.createConstraint(
            parent_body,
            parent_link,
            child_body,
            child_link,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(parent_frame_position),
            childFramePosition=list(child_frame_position),
            childFrameOrientation=(list(child_frame_orientation_q)
                                   if child_frame_orientation_q is not None
                                   else p.getQuaternionFromEuler([0, 0, 0])),
        )

    # ------------------------------------------------------------------ joints & links
    def num_joints(self, body):
        return int(p.getNumJoints(body))

    def get_joint_info(self, body, joint):
        info = p.getJointInfo(body, int(joint))
        return JointInfo(
            index=int(info[0]),
            name=_decode(info[1]),
            joint_type=_JOINT_TYPE_NAMES.get(info[2], JOINT_OTHER),
            lower_limit=float(info[8]),
            upper_limit=float(info[9]),
            child_link_name=_decode(info[12]),
        )

    def get_joint_state(self, body, joint):
        state = p.getJointState(body, int(joint))
        return JointState(position=state[0], velocity=state[1], applied_torque=state[3])

    def reset_joint_state(self, body, joint, position):
        p.resetJointState(body, int(joint), position)

    def get_link_pose(self, body, link):
        state = p.getLinkState(body, int(link), computeForwardKinematics=True)
        return state[0], state[1]

    def set_joint_position(self, body, joint, target, force=None, position_gain=None):
        kwargs = {}
        if target is not None:
            kwargs["targetPosition"] = target
        if force is not None:
            kwargs["force"] = force
        if position_gain is not None:
            kwargs["positionGain"] = position_gain
        p.setJointMotorControl2(body, int(joint), p.POSITION_CONTROL, **kwargs)

    def set_joint_positions(self, body, joints, targets, forces=None, position_gains=None):
        kwargs = {}
        if forces is not None:
            kwargs["forces"] = forces
        if position_gains is not None:
            kwargs["positionGains"] = position_gains
        p.setJointMotorControlArray(body, list(joints), p.POSITION_CONTROL,
                                    targetPositions=targets, **kwargs)

    def set_joint_velocity(self, body, joint, target_velocity, force=None):
        kwargs = {"targetVelocity": target_velocity}
        if force is not None:
            kwargs["force"] = force
        p.setJointMotorControl2(body, int(joint), controlMode=p.VELOCITY_CONTROL, **kwargs)

    def inverse_kinematics(self, body, link, position, orientation_q=None,
                           lower_limits=None, upper_limits=None, joint_ranges=None,
                           rest_poses=None, max_iterations=500):
        return p.calculateInverseKinematics(
            body,
            int(link),
            position,
            targetOrientation=orientation_q,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=max_iterations,
        )

    # ------------------------------------------------------------------ transforms
    def quat_from_euler(self, euler):
        return p.getQuaternionFromEuler(euler)

    def euler_from_quat(self, quat_xyzw):
        return p.getEulerFromQuaternion(quat_xyzw)

    def matrix_from_quat(self, quat_xyzw):
        return p.getMatrixFromQuaternion(quat_xyzw)

    def quat_from_axis_angle(self, axis, angle):
        return p.getQuaternionFromAxisAngle(list(axis), float(angle))

    # ------------------------------------------------------------------ cameras
    def compute_projection_matrix(self, fov, aspect, near, far):
        return p.computeProjectionMatrixFOV(fov, aspect, near, far)

    def compute_view_matrix(self, eye, target, up):
        return p.computeViewMatrix(cameraEyePosition=eye,
                                   cameraTargetPosition=target,
                                   cameraUpVector=up)

    def compute_view_matrix_from_yaw_pitch_roll(self, target, distance, yaw_deg, pitch_deg,
                                                roll_deg=0.0, up_axis_index=2):
        # Positional args: some PyBullet builds require 'upAxisIndex' positionally.
        return p.computeViewMatrixFromYawPitchRoll(
            target, distance, yaw_deg, pitch_deg, roll_deg, up_axis_index,
        )

    def render_camera(self, width, height, view_matrix, projection_matrix):
        image = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        img_w, img_h = image[0], image[1]
        rgb_buffer = image[2]
        depth_buffer = image[3]

        # PyBullet may return a flattened tuple/list (W*H*4 RGBA) or an already-shaped
        # array depending on platform/renderer; normalise to (H, W, 3) uint8 here so no
        # caller has to know.
        try:
            rgb_array = np.array(rgb_buffer, dtype=np.uint8).reshape(img_h, img_w, 4)
        except Exception:
            rgb_array = np.asarray(rgb_buffer, dtype=np.uint8)
            if rgb_array.ndim == 1:
                rgb_array = rgb_array.reshape(img_h, img_w, 4)
        rgb_array = rgb_array[:, :, :3]

        depth_array = np.array(depth_buffer, dtype=np.float32).reshape(img_h, img_w)
        return CameraFrame(width=img_w, height=img_h, rgb=rgb_array, depth=depth_array,
                           segmentation=image[4] if len(image) > 4 else None)

    # ------------------------------------------------------------------ viewer & debug draw
    def reset_viewer_camera(self, distance, yaw, pitch, target):
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target)

    def get_viewer_camera(self):
        try:
            dbg = p.getDebugVisualizerCamera()
        except Exception:
            return None
        tuple_len = len(dbg) if isinstance(dbg, (list, tuple)) else None
        if tuple_len != 12:
            return {"available": False, "tuple_len": tuple_len}
        return {
            "available": True,
            "tuple_len": tuple_len,
            "view_matrix": dbg[2],
            "projection_matrix": dbg[3],
            "yaw": float(dbg[8]),
            "pitch": float(dbg[9]),
            "distance": float(dbg[10]),
            "target": dbg[11],
        }

    def set_viewer_options(self, show_gui=None, shadows=None):
        if show_gui is not None:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(bool(show_gui)))
        if shadows is not None:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(bool(shadows)))

    def set_real_time(self, enabled):
        p.setRealTimeSimulation(1 if enabled else 0)

    def viewer_is_open(self):
        return self.is_connected()

    def get_click_events(self):
        # ev = (eventType, mousePosX, mousePosY, buttonIndex, buttonState);
        # eventType 2 = button event, KEY_WAS_TRIGGERED = pressed down this frame.
        try:
            return [ev for ev in p.getMouseEvents()
                    if ev[0] == 2 and ev[4] & p.KEY_WAS_TRIGGERED]
        except Exception:
            return []

    def draw_debug_line(self, start, end, color, line_width=1.0, life_time=0.0):
        return p.addUserDebugLine(start, end, list(color)[:3], lifeTime=life_time)

    def draw_debug_points(self, points, colors, point_size=5.0, life_time=0.0):
        return p.addUserDebugPoints(points, colors, pointSize=point_size, lifeTime=life_time)

    def draw_marker_sphere(self, position, radius, color):
        """Massless visual-only MultiBody.

        ``addUserDebugLine``/``Points`` are viewer overlays and do **not** appear in
        ``getCameraImage``, so anything that must show up in a captured frame has to be a
        real (but massless, collision-free) body.
        """
        try:
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=float(radius),
                rgbaColor=list(color),
            )
            return p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=list(map(float, position)),
            )
        except Exception as e:
            print("[Sim] Warning: draw_marker_sphere failed:", e)
            traceback.print_exc()
            return None

    def draw_marker_cylinder(self, start, end, radius, color):
        try:
            a = np.array(list(map(float, start)), dtype=float)
            b = np.array(list(map(float, end)), dtype=float)
            v = b - a
            length = float(np.linalg.norm(v))
            if not np.isfinite(length) or length < 1e-6:
                return self.draw_marker_sphere(a.tolist(), radius * 1.5, color)
            mid = ((a + b) * 0.5).tolist()
            # Orientation: align local +Z with direction v
            z = np.array([0.0, 0.0, 1.0], dtype=float)
            u = v / length
            c = float(np.dot(z, u))
            if c > 0.999999:
                quat = [0, 0, 0, 1]
            elif c < -0.999999:
                quat = self.quat_from_axis_angle([1, 0, 0], math.pi)
            else:
                axis = np.cross(z, u)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm < 1e-8:
                    quat = [0, 0, 0, 1]
                else:
                    quat = self.quat_from_axis_angle((axis / axis_norm).tolist(), math.acos(c))
            vis_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=float(radius),
                length=length,
                rgbaColor=list(color),
            )
            return p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=vis_id,
                basePosition=mid,
                baseOrientation=quat,
            )
        except Exception as e:
            print("[Sim] Warning: draw_marker_cylinder failed:", e)
            traceback.print_exc()
            return None

    def remove_marker(self, handle):
        try:
            if handle is not None:
                p.removeBody(int(handle))
        except Exception as e:
            print("[Sim] Warning: remove_marker failed:", e)
