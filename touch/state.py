"""Touch state: what the robot feels, from proprioception only.

Real-robot sources (Franka): ``O_T_EE`` (EE pose, FK), ``O_F_ext_hat_K`` (external wrench estimated
from the joint-torque sensors), Franka Hand width. Resolution of the wrench estimate is ~1-2 N.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TouchState:
    """One proprioceptive sample.

    ``force`` / ``torque`` are the **external** wrench the environment applies to the end
    effector, world frame (estimated, noisy). With an insertion axis ``a`` the resistance the
    robot feels while pushing along ``a`` is ``-force . a``.
    """

    t: float
    ee_pos: np.ndarray
    force: np.ndarray
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_R: Optional[np.ndarray] = None
    gripper_width: Optional[float] = None
    holding: Optional[bool] = None

    def axial_resistance(self, axis) -> float:
        return float(-np.dot(self.force, axis))

    def lateral_force(self, axis) -> np.ndarray:
        a = np.asarray(axis, float)
        return self.force - np.dot(self.force, a) * a

    def describe(self, axis=None) -> str:
        """Short text for prompts / logs (proprioception only)."""
        parts = [f"t={self.t:.2f}s", f"ee={np.round(self.ee_pos, 4).tolist()}",
                 f"F_ext={np.round(self.force, 1).tolist()}N"]
        if axis is not None:
            parts.append(f"axial_resist={self.axial_resistance(axis):.1f}N "
                         f"lateral={np.linalg.norm(self.lateral_force(axis)):.1f}N")
        if self.gripper_width is not None:
            parts.append(f"grip_w={self.gripper_width * 1000:.1f}mm")
        return " ".join(parts)


def wrench_from_joint_torques(J, tau, tau_model):
    """libfranka-style external wrench: ``W = -pinv(J^T) (tau - tau_model)``.

    ``J`` is the 6xN EE Jacobian (world frame), ``tau_model`` the torque the model predicts
    without contact (gravity/dynamics, or a free-space reading at the same pose). Returns
    ``[Fx, Fy, Fz, Mx, My, Mz]``: the wrench the environment applies to the EE.
    """
    J = np.asarray(J, float)
    return -np.linalg.pinv(J.T) @ (np.asarray(tau, float) - np.asarray(tau_model, float))
