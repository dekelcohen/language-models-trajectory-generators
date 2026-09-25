"""Constant-velocity Kalman filter for one rigid body's 6-DoF pose.

One filter per *object*, not per point: all N tracked points share one pose, so every point's
measurement strengthens the same estimate, and a point the cameras lose still has a
predicted position (``R @ template_i + t``).

State (error-state form, 12-D)::

    x = [p (3), v (3), theta (3), omega (3)]

``p`` is the template centroid's world position and ``v`` its velocity. Rotation is kept as a
reference matrix ``R_ref`` plus a small rotation vector ``theta`` on top of it; after every
update ``theta`` is folded into ``R_ref`` and reset to zero. That keeps the linearisation
exact to first order at any orientation - a plain Euler/quaternion state would wrap or become
singular exactly when an object tumbles.

Units: ``dt`` is in *reference frames* - 1 unit = ``1 / config.track_kf_ref_hz`` seconds (0.1 s,
the lock-step harness's motion rate the noise defaults were tuned at). The session passes
``dt = 1`` per tracked frame when it has no clock and ``elapsed_s * ref_hz`` in real-time mode,
so a 30 fps camera steps ``dt = 1/3`` and the filter's per-second behaviour (process noise,
coast budget, reset patience) does not change with the frame rate.

Outlier gating is deliberately **not** permanent. A measurement beyond the Mahalanobis gate
is rejected, but ``reset_after`` consecutive rejections that agree with each other re-initialise
the filter on the measurement. Without that, a *real* abrupt change - a dropped ball, the
case the drop monitor exists for - would be rejected as an outlier forever and the filter
would coast confidently along the old trajectory.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from sim_adapter.transforms import exp_so3, log_so3

#: chi-square 99.9 % quantile for 6 dof (position + rotation measurement)
CHI2_6_999 = 22.46


def _cfg(key):
    return field(default_factory=lambda: getattr(config, key))


@dataclass
class KFConfig:
    """Noise model and gating. Defaults are ``config.track_kf_*`` (read at construction)."""

    accel_std: float = _cfg("track_kf_accel_std")            # m / ref-frame^2
    ang_accel_std: float = _cfg("track_kf_ang_accel_std")    # rad / ref-frame^2
    pos_meas_std: float = _cfg("track_kf_pos_meas_std")      # m   - measurement noise floor
    rot_meas_std: float = _cfg("track_kf_rot_meas_std")      # rad - measurement noise floor
    gate_chi2: float = CHI2_6_999
    reset_after: int = _cfg("track_kf_reset_after")          # ref-frames of agreeing rejections -> re-init
    max_coast: int = _cfg("track_kf_max_coast")              # ref-frames of prediction before giving up


@dataclass
class KFResult:
    accepted: bool
    reset: bool = False
    mahalanobis2: Optional[float] = None
    reason: str = ""


@dataclass
class RigidBodyKF:
    """See module docstring. ``initialized`` is False until the first :meth:`init`."""

    cfg: KFConfig = field(default_factory=KFConfig)

    def __post_init__(self):
        self.x = np.zeros(12)
        self.P = np.eye(12)
        self.R_ref = np.eye(3)
        self.initialized = False
        self.coast = 0
        self.rejects = 0
        self.last_rejected = None      # (p, R) of the last rejected measurement

    # -- lifecycle ----------------------------------------------------------
    def init(self, p, R, pos_std=None, rot_std=None):
        c = self.cfg
        self.x = np.zeros(12)
        self.x[0:3] = np.asarray(p, dtype=float).reshape(3)
        self.R_ref = np.asarray(R, dtype=float).reshape(3, 3).copy()
        ps = pos_std or c.pos_meas_std
        rs = rot_std or c.rot_meas_std
        self.P = np.diag([ps ** 2] * 3 + [0.02 ** 2] * 3 + [rs ** 2] * 3 + [0.1 ** 2] * 3)
        self.initialized = True
        self.coast = 0
        self.rejects = 0
        self.last_rejected = None

    @property
    def position(self):
        return self.x[0:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()

    @property
    def rotation(self):
        return exp_so3(self.x[6:9]) @ self.R_ref

    @property
    def angular_velocity(self):
        return self.x[9:12].copy()

    @property
    def pos_std(self):
        return float(np.sqrt(max(np.trace(self.P[0:3, 0:3]) / 3.0, 0.0)))

    @property
    def stale(self):
        return self.coast > self.cfg.max_coast + 1e-6

    # -- filter -------------------------------------------------------------
    def _F(self, dt):
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt
        F[6:9, 9:12] = np.eye(3) * dt
        return F

    def _Q(self, dt):
        c = self.cfg
        q = np.zeros((12, 12))
        for block, std in ((0, c.accel_std), (6, c.ang_accel_std)):
            s2 = std ** 2
            q[block:block + 3, block:block + 3] = np.eye(3) * s2 * dt ** 4 / 4.0
            q[block:block + 3, block + 3:block + 6] = np.eye(3) * s2 * dt ** 3 / 2.0
            q[block + 3:block + 6, block:block + 3] = np.eye(3) * s2 * dt ** 3 / 2.0
            q[block + 3:block + 6, block + 3:block + 6] = np.eye(3) * s2 * dt ** 2
        return q

    def predict(self, dt=1.0):
        if not self.initialized:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.coast += dt
        self._last_dt = dt

    def predicted_pose(self, dt=1.0):
        """Pose ``dt`` ahead, without mutating the filter."""
        p = self.x[0:3] + self.x[3:6] * dt
        theta = self.x[6:9] + self.x[9:12] * dt
        return p, exp_so3(theta) @ self.R_ref

    def _innovation(self, p, R):
        z_rot = log_so3(np.asarray(R, dtype=float) @ self.R_ref.T)
        return np.concatenate([np.asarray(p, dtype=float).reshape(3) - self.x[0:3],
                               z_rot - self.x[6:9]])

    def update(self, p, R, pos_std=None, rot_std=None):
        """Fuse a measured pose. Returns :class:`KFResult`; never raises on bad input."""
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(R)):
            return KFResult(False, reason="non-finite measurement")
        if not self.initialized:
            self.init(p, R, pos_std, rot_std)
            return KFResult(True, reset=True, reason="init")

        c = self.cfg
        ps = max(float(pos_std or 0.0), c.pos_meas_std)
        rs = max(float(rot_std or 0.0), c.rot_meas_std)
        H = np.zeros((6, 12))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        Rm = np.diag([ps ** 2] * 3 + [rs ** 2] * 3)
        y = self._innovation(p, R)
        S = H @ self.P @ H.T + Rm
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return KFResult(False, reason="singular innovation covariance")
        d2 = float(y @ S_inv @ y)

        if d2 > c.gate_chi2:
            consistent = False
            if self.last_rejected is not None:
                lp, lR = self.last_rejected
                consistent = (np.linalg.norm(np.asarray(p) - lp) < 3.0 * ps + 0.02
                              and np.linalg.norm(log_so3(np.asarray(R) @ lR.T)) < 3.0 * rs + 0.1)
            step = float(getattr(self, "_last_dt", 1.0))
            self.rejects = self.rejects + step if consistent or self.rejects == 0 else step
            self.last_rejected = (np.asarray(p, dtype=float).copy(),
                                  np.asarray(R, dtype=float).copy())
            if self.rejects >= c.reset_after - 1e-6:
                self.init(p, R, pos_std, rot_std)
                return KFResult(True, reset=True, mahalanobis2=d2,
                                reason=f"{c.reset_after} consistent rejections -> re-init")
            return KFResult(False, mahalanobis2=d2, reason="outside gate")

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P
        # fold the rotation error into the reference so theta stays small
        self.R_ref = exp_so3(self.x[6:9]) @ self.R_ref
        self.x[6:9] = 0.0
        self.coast = 0
        self.rejects = 0
        self.last_rejected = None
        return KFResult(True, mahalanobis2=d2)
