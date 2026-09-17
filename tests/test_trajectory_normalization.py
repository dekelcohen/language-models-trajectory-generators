"""execute_trajectory must accept only things the simulator process can rebuild.

The trajectory is sent over multiprocessing.Pipe (PyBullet) or JSON (Genesis), so a class
defined inside the model's own code block cannot travel: pickle stores it by reference as
``<module>.<name>``, and neither interpreter can resolve that. Observed in a door rollout as

    _pickle.PicklingError: Can't pickle <class 'agent_runner._Arc'>:
    attribute lookup _Arc on agent_runner failed

These tests run entirely in-process: no sim, no LLM.
"""

import pickle
import types
import unittest

import numpy as np

import common_utils
from common_utils import Trajectory


def _llm_defined_arc_class():
    """A class built the way exec'd LLM code builds one: __module__ points at the module
    whose globals seeded the exec namespace, but the class is not an attribute there."""
    ns = {"__name__": "agent_runner"}
    exec(
        "class _Arc:\n"
        "    def __init__(self, points, desc):\n"
        "        self.points = points\n"
        "        self.desc = desc\n",
        ns,
    )
    return ns["_Arc"]


class ReproducesTheOriginalFailure(unittest.TestCase):
    def test_llm_defined_class_is_not_picklable(self):
        arc_cls = _llm_defined_arc_class()
        arc = arc_cls([[0.1, 0.2, 0.3, 0.0]], "arc pull")
        self.assertEqual(arc_cls.__module__, "agent_runner")
        with self.assertRaises((pickle.PicklingError, AttributeError)):
            pickle.dumps(arc)

    def test_normalize_makes_it_picklable(self):
        arc_cls = _llm_defined_arc_class()
        arc = arc_cls([[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 1.0]], "arc pull")
        out = Trajectory.normalize(arc)
        self.assertIsInstance(out, Trajectory)
        self.assertEqual(out.points, [[0.1, 0.2, 0.3, 0.0], [0.4, 0.5, 0.6, 1.0]])
        self.assertEqual(out.desc, "arc pull")
        # The whole point: it now survives the transport.
        self.assertEqual(pickle.loads(pickle.dumps(out)).points, out.points)


class AcceptedInputs(unittest.TestCase):
    def test_real_trajectory_passes_through(self):
        traj = Trajectory([[0.0, 0.1, 0.2, 0.0], [0.3, 0.4, 0.5, 1.0]], "move to handle")
        out = Trajectory.normalize(traj)
        self.assertEqual(out.points, traj.points)
        self.assertEqual(out.desc, "move to handle")

    def test_bare_list_of_poses_is_wrapped(self):
        out = Trajectory.normalize([[0.0, 0.1, 0.2, 0.0]])
        self.assertIsInstance(out, Trajectory)
        self.assertEqual(out.points, [[0.0, 0.1, 0.2, 0.0]])
        self.assertEqual(out.desc, "")

    def test_length_six_poses_are_preserved(self):
        pose = [0.1, 0.2, 0.3, 0.0, 1.5, -0.7]
        self.assertEqual(Trajectory.normalize([pose]).points, [pose])

    def test_numpy_points_become_plain_floats(self):
        """Numpy survives pickle but is rejected by the JSON transport, so strip it here."""
        out = Trajectory.normalize(Trajectory(np.array([[0.1, 0.2, 0.3, 0.0]]), "np"))
        self.assertEqual(out.points, [[0.1, 0.2, 0.3, 0.0]])
        for pose in out.points:
            for v in pose:
                self.assertIsInstance(v, float)
                self.assertNotIsInstance(v, np.generic)

    def test_numpy_scalars_inside_a_python_list(self):
        out = Trajectory.normalize([[np.float64(0.1), 0.2, 0.3, np.float32(0.0)]])
        for v in out.points[0]:
            self.assertIsInstance(v, float)
            self.assertNotIsInstance(v, np.generic)

    def test_tuples_and_generators_are_accepted(self):
        out = Trajectory.normalize(((0.0, 0.1, 0.2, 0.0) for _ in range(2)))
        self.assertEqual(out.points, [[0.0, 0.1, 0.2, 0.0], [0.0, 0.1, 0.2, 0.0]])

    def test_non_string_desc_is_coerced(self):
        arc_cls = _llm_defined_arc_class()
        out = Trajectory.normalize(arc_cls([[0.0, 0.0, 0.0, 0.0]], 42))
        self.assertEqual(out.desc, "42")

    def test_missing_desc_defaults_to_empty(self):
        obj = types.SimpleNamespace(points=[[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(Trajectory.normalize(obj).desc, "")


class RejectedInputs(unittest.TestCase):
    def _assert_actionable(self, ctx):
        msg = str(ctx.exception)
        self.assertIn("execute_trajectory()", msg)
        self.assertIn("generate_linear_trajectory", msg)

    def test_opaque_object_raises_actionable_typeerror(self):
        class Opaque:
            pass

        with self.assertRaises(TypeError) as ctx:
            Trajectory.normalize(Opaque())
        self._assert_actionable(ctx)
        self.assertIn("Opaque", str(ctx.exception))

    def test_pose_that_is_not_a_sequence(self):
        with self.assertRaises(TypeError) as ctx:
            Trajectory.normalize([[0.0, 0.1, 0.2, 0.0], 5.0])
        self._assert_actionable(ctx)
        self.assertIn("Pose 1", str(ctx.exception))

    def test_pose_with_a_non_numeric_value(self):
        with self.assertRaises(TypeError) as ctx:
            Trajectory.normalize([[0.0, 0.1, "left", 0.0]])
        self._assert_actionable(ctx)
        self.assertIn("Pose 0", str(ctx.exception))

    def test_string_is_not_a_trajectory(self):
        with self.assertRaises(TypeError) as ctx:
            Trajectory.normalize("go left")
        self._assert_actionable(ctx)


class WiredIntoTheApi(unittest.TestCase):
    """execute_trajectory must normalize BEFORE the first send(), or the pipe still dies."""

    def test_execute_trajectory_normalizes_before_sending(self):
        import api as api_module

        sent = []

        class FakeConnection:
            def send(self, payload):
                # Mirrors what multiprocessing.Pipe does to the payload.
                pickle.dumps(payload)
                sent.append(payload)

            def recv(self):
                return ["ok", 7]

        class FakeArgs:
            vis_traj = False
            tracking = False

        class FakeTask:
            trajectory_length = 0

        inst = api_module.API.__new__(api_module.API)
        inst.main_connection = FakeConnection()
        inst.args = FakeArgs()
        inst.task = FakeTask()
        inst.logger = types.SimpleNamespace(info=lambda *a, **k: None)

        arc_cls = _llm_defined_arc_class()
        inst.execute_trajectory(arc_cls([[0.1, 0.2, 0.3, 0.0]], "arc"))

        self.assertEqual(len(sent), 1)
        payload = sent[0]
        self.assertIsInstance(payload[1], common_utils.Trajectory)
        self.assertEqual(payload[1].points, [[0.1, 0.2, 0.3, 0.0]])
        self.assertEqual(inst.task.trajectory_length, 1)


class ConstructorEnforcesTheInvariant(unittest.TestCase):
    """Coercion lives in __init__, so a Trajectory that cannot travel cannot be built."""

    def test_ctor_coerces_numpy_points(self):
        traj = Trajectory(np.array([[0.1, 0.2, 0.3, 0.0]]), "np")
        self.assertEqual(traj.points, [[0.1, 0.2, 0.3, 0.0]])
        for v in traj.points[0]:
            self.assertIsInstance(v, float)
            self.assertNotIsInstance(v, np.generic)

    def test_ctor_coerces_ints_and_tuples(self):
        traj = Trajectory([(0, 1, 2, 0)], "ints")
        self.assertEqual(traj.points, [[0.0, 1.0, 2.0, 0.0]])
        for v in traj.points[0]:
            self.assertIsInstance(v, float)

    def test_ctor_coerces_desc(self):
        self.assertEqual(Trajectory([[0.0, 0.0, 0.0, 0.0]], 42).desc, "42")
        self.assertEqual(Trajectory([[0.0, 0.0, 0.0, 0.0]], None).desc, "")
        self.assertEqual(Trajectory([[0.0, 0.0, 0.0, 0.0]]).desc, "")

    def test_ctor_rejects_bad_points(self):
        with self.assertRaises(TypeError):
            Trajectory([[0.0, 0.1, "left", 0.0]], "bad")

    def test_every_constructed_trajectory_is_picklable(self):
        traj = Trajectory(np.array([[0.1, 0.2, 0.3, 0.0]]), "np")
        self.assertEqual(pickle.loads(pickle.dumps(traj)).points, traj.points)

    def test_normalize_returns_a_real_trajectory_unchanged(self):
        traj = Trajectory([[0.0, 0.1, 0.2, 0.0]], "keep")
        self.assertIs(Trajectory.normalize(traj), traj)


if __name__ == "__main__":
    unittest.main()

