"""touch.sim_runner on a fake torque-controlled Cartesian arm (no simulator needed)."""
from types import SimpleNamespace

import numpy as np

from touch.sim_runner import run_touch_insert

FACE, BOTTOM, HOLE_YZ, HOLE_TOL = 0.004, 0.016, (0.001, 0.0), 0.0005


class FakeArm:
    """3 prismatic joints = tip position (J = I), 2 fingers; a wall at x=FACE with a hole at HOLE_YZ."""

    def __init__(self, torque=True):
        self.torque = torque
        self.q, self.dq, self.tau = np.zeros(3), np.zeros(3), np.zeros(3)
        self.grip, self.latched = None, {}

    def supports_torque_control(self):
        return self.torque

    def get_link_pose(self, body, link):
        return self.q.copy(), [0, 0, 0, 1]

    def get_link_jacobian(self, body, link, joints, local_point=None):
        return np.vstack([np.eye(3), np.zeros((3, 3))])

    def get_gravity_torques(self, body, joints):
        return np.zeros(3)

    def get_joint_state(self, body, j):
        if j < 3:
            return SimpleNamespace(position=self.q[j], velocity=self.dq[j])
        return SimpleNamespace(position=0.008, velocity=0.0)

    def set_joint_torques(self, body, joints, tau):
        if list(joints) == [0, 1, 2]:
            self.tau = np.asarray(tau, float)
        else:
            self.grip = list(tau)

    def set_joint_positions(self, body, joints, pos):
        self.latched[tuple(joints)] = list(pos)

    def step(self, dt=0.002, m=2.0):
        lateral = np.hypot(self.q[1] - HOLE_YZ[0], self.q[2] - HOLE_YZ[1])
        wall = BOTTOM if lateral < HOLE_TOL else FACE
        f = self.tau.copy()
        if self.q[0] > wall:
            f[0] -= 2e4 * (self.q[0] - wall) + 50 * self.dq[0]
            f[1:] -= 0.3 * abs(f[0]) * np.sign(self.dq[1:])      # face friction while pressed
        self.dq += f / m * dt
        self.q += self.dq * dt


def run(sim):
    return run_touch_insert(sim, 0, 0, [0, 1, 2], [3, 4], sim.step, 0.002, axis=[1, 0, 0],
                            max_s=20.0, force_noise=0.0, k_rot=0.0)


def test_unsupported_without_torque_control():
    assert run(FakeArm(torque=False))["outcome"] == "unsupported"


def test_finds_offset_hole_and_seats_then_latches_position_control():
    sim = FakeArm()
    res = run(sim)
    assert res["outcome"] == "seated", res
    assert abs(sim.q[1] - HOLE_YZ[0]) < HOLE_TOL and sim.q[0] > FACE + 0.007
    assert sim.grip == [-20.0, -20.0]
    assert (0, 1, 2) in sim.latched and (3, 4) in sim.latched
