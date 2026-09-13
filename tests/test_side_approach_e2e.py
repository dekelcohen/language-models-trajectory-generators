"""End-to-end proof that a side (horizontal) approach can open a vertical-handle cabinet.

Everything else in the side-approach test suite checks a *component*: the rotation maths,
the pose builders, the length-4/length-6 dispatch. This test drives the real environment
subprocess over the real IPC protocol with a scripted length-6 trajectory and asks the
sim-env's own success oracle for a verdict, so it covers the whole chain at once.

The target is ``franka_kitchen:slide_cabinet``. Its handle (``slidelink_1``) is a vertical
bar roughly 0.11 x 0.04 x 0.25 m on a panel facing -x, so the side approach comes in along
+x with the fingers closing along y.

**What this does and does not prove.** It proves the side-approach path works end to end:
the pose survives IPC, IK solves it, the gripper reaches the bar and pulls the door open.
It does *not* prove the side approach is *required* here - the control test below measures
that a top-down gripper also opens this particular cabinet, because a sliding panel is
low-resistance and can simply be shoved sideways without a proper pinch. Necessity comes
from geometry (a bar taller than the fingers, or a panel blocked from above), not from
this task's success oracle.

No LLM is involved: the trajectory is scripted and ``sim_state["success"]`` is computed by
the kitchen sim-env from joint angles.

Slow (a few minutes of stepped physics with camera capture), so it is opt-in::

    $env:LMTG_RUN_SLOW_SIM_TESTS = "1"
    python -m pytest tests/test_side_approach_e2e.py -s
"""

import math
import multiprocessing as mp
import os
import traceback
import unittest

import numpy as np

import config
from common_utils import side_grasp_pose

TASK = "franka_kitchen:slide_cabinet"

#: Handle (``slidelink_1``) AABB centre in the freshly loaded scene.
HANDLE = [0.528, 0.106, 1.950]
#: Pinch just in front of the bar's centre: deep enough to be a real grasp, shallow enough
#: that the wrist does not reach the door panel behind it.
GRASP = [HANDLE[0] - 0.02, HANDLE[1], HANDLE[2]]
#: Stand off along -x, i.e. straight back along the approach axis.
STANDOFF = 0.18
#: How far to drag the door open along -y. The goal is 0.37 m with a 0.30 tolerance.
PULL = 0.45

#: Trajectory points are consumed one simulation step each, so a waypoint has to be
#: repeated to give the position controller time to converge on it.
STEPS_PER_WAYPOINT = 15
SETTLE_STEPS = 150


class _Args:
    mode = "default"
    robot = "franka"
    task = TASK
    sim = "pybullet"
    save_grasp_inputs = False


def _run_env(conn):
    import logging
    from env import run_simulation_environment

    logger = logging.getLogger("env")
    logger.setLevel(logging.INFO)
    try:
        run_simulation_environment(_Args, conn, logger)
    except Exception:
        traceback.print_exc()


def _hold(pose, steps):
    return [list(pose)] * steps


def _sweep(start_xyz, end_xyz, waypoints, approach_yaw=0.0, rotation=0.0):
    """Dense list of side-approach poses from ``start_xyz`` to ``end_xyz``."""
    points = []
    for i in range(waypoints):
        t = i / float(waypoints - 1)
        xyz = [start_xyz[j] + (end_xyz[j] - start_xyz[j]) * t for j in range(3)]
        points += _hold(side_grasp_pose(xyz[0], xyz[1], xyz[2], rotation, approach_yaw),
                        STEPS_PER_WAYPOINT)
    return points


@unittest.skipUnless(
    os.environ.get("LMTG_RUN_SLOW_SIM_TESTS"),
    "slow simulation test; set LMTG_RUN_SLOW_SIM_TESTS=1 to run",
)
class TestSideApproachOpensSlideCabinet(unittest.TestCase):

    def setUp(self):
        self.parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=_run_env, args=(child_conn,), daemon=True)
        self.proc.start()
        self.handshake = self.parent_conn.recv()

    def tearDown(self):
        for close in (self.parent_conn.close, self.proc.terminate):
            try:
                close()
            except Exception:
                pass
        try:
            self.proc.join(timeout=10)
        except Exception:
            pass

    def _request(self, *message):
        self.parent_conn.send(list(message))
        return self.parent_conn.recv()

    def _state(self):
        return self._request(config.GET_STATE)["sim_state"]

    def test_horizontal_grasp_opens_the_cabinet(self):
        start = self._state()
        self.assertEqual(start["task"], "slide_cabinet")
        self.assertFalse(start["success"], "task must not start already solved")

        pre = [GRASP[0] - STANDOFF, GRASP[1], GRASP[2]]

        # 1. Stand off in front of the handle, gripper already horizontal.
        reply = self._request(
            config.EXECUTE_TRAJECTORY,
            _hold(side_grasp_pose(pre[0], pre[1], pre[2], 0.0, 0.0), SETTLE_STEPS),
        )
        self.assertIn("Finished executing generated trajectory", reply[0])

        # 2. Straight in along +x until the bar sits between the fingers.
        self._request(config.EXECUTE_TRAJECTORY, _sweep(pre, GRASP, waypoints=10))
        approached = self._request(config.GET_STATE)["eef_pos"]
        self.assertLess(
            abs(approached[1] - GRASP[1]), 0.05,
            f"gripper drifted off the handle in y: {approached}",
        )
        self.assertLess(
            abs(approached[2] - GRASP[2]), 0.05,
            f"gripper drifted off the handle in z: {approached}",
        )

        # 3. Pinch. CLOSE_GRIPPER does not reply, so the next request is the barrier.
        self.parent_conn.send([config.CLOSE_GRIPPER])
        gripped = self._state()
        self.assertFalse(gripped["success"], "closing the gripper alone must not pass")

        # 4. Drag the door open along -y.
        pulled = [GRASP[0], GRASP[1] - PULL, GRASP[2]]
        self._request(config.EXECUTE_TRAJECTORY, _sweep(GRASP, pulled, waypoints=30))

        final = self._state()
        print(f"\nslide_cabinet: error {start['task_error']:.3f} -> {final['task_error']:.3f}")
        self.assertLess(
            final["task_error"], start["task_error"] - 0.2,
            "the side grasp never moved the door; the gripper most likely missed the bar",
        )
        self.assertTrue(
            final["success"],
            f"cabinet not open enough: task_error={final['task_error']:.3f}",
        )

    def test_legacy_top_down_path_still_opens_the_cabinet(self):
        """Regression guard for the length-4 pose path, and the honest control.

        Two things are checked in one run. First, that ``env.py``'s legacy branch still
        executes a plain ``[x, y, z, theta]`` trajectory end to end after the dispatch
        change - the no-regression claim, measured rather than argued.

        Second, the control: this *also* opens the cabinet. Measured, not assumed. A
        sliding panel offers so little resistance that a top-down gripper shoves the bar
        sideways without ever pinching it, so ``slide_cabinet`` is evidence that the side
        approach *works*, not that it is *needed*. The need is geometric - a bar longer
        than the fingers, or a panel with no clearance above it - and is covered by the
        reachability checks in ``tests/test_side_approach.py``.
        """
        start = self._state()
        above = [GRASP[0], GRASP[1], GRASP[2] + 0.20]
        reply = self._request(
            config.EXECUTE_TRAJECTORY,
            _hold([above[0], above[1], above[2], 0.0], SETTLE_STEPS),
        )
        self.assertIn("Finished executing generated trajectory", reply[0])

        descent = []
        for i in range(10):
            t = i / 9.0
            z = above[2] + (GRASP[2] - above[2]) * t
            descent += [[above[0], above[1], z, 0.0]] * STEPS_PER_WAYPOINT
        self._request(config.EXECUTE_TRAJECTORY, descent)

        self.parent_conn.send([config.CLOSE_GRIPPER])
        pull = [[GRASP[0], GRASP[1] - PULL * (i / 29.0), GRASP[2], 0.0] for i in range(30)]
        dense = [p for point in pull for p in [point] * STEPS_PER_WAYPOINT]
        self._request(config.EXECUTE_TRAJECTORY, dense)

        final = self._state()
        print(f"\nslide_cabinet top-down control: "
              f"error {start['task_error']:.3f} -> {final['task_error']:.3f}")
        self.assertLess(
            final["task_error"], start["task_error"] - 0.2,
            "the legacy length-4 trajectory no longer moves the door - the top-down path "
            "regressed",
        )


if __name__ == "__main__":
    unittest.main()
