"""Contract tests for the JSON-lines IPC transport.

These run entirely in-process (a real localhost socket pair, no simulator) so they are
fast and can gate every change to the wire format. The payloads mirror what
``env.run_simulation_environment`` and ``agent_runner`` actually exchange.
"""

import threading
import unittest

import numpy as np

from common_utils import Trajectory
from providers.json_ipc import (
    JsonIpcConnection,
    JsonIpcServer,
    decode,
    dumps,
    encode,
    loads,
)


def _roundtrip(payload):
    return loads(dumps(payload).decode("utf-8").rstrip("\n"))


class TestCodec(unittest.TestCase):
    def test_scalars_and_containers_survive(self):
        payload = [3, "ok", None, True, 1.5, [1, 2], {"a": {"b": [1.0]}}]
        self.assertEqual(_roundtrip(payload), payload)

    def test_nan_and_inf_survive(self):
        out = _roundtrip([float("inf"), float("-inf")])
        self.assertEqual(out[0], float("inf"))
        self.assertEqual(out[1], float("-inf"))
        self.assertTrue(np.isnan(_roundtrip([float("nan")])[0]))

    def test_tuples_stay_tuples(self):
        out = _roundtrip({"pose": (1.0, 2.0, 3.0)})
        self.assertIsInstance(out["pose"], tuple)
        self.assertEqual(out["pose"], (1.0, 2.0, 3.0))

    def test_ndarray_roundtrip_is_bit_exact(self):
        # VISUALIZE_GRASP_POSE sends an (N, 4, 4) float64 array.
        arr = np.random.RandomState(0).rand(5, 4, 4)
        out = _roundtrip(arr)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, arr.shape)
        self.assertEqual(out.dtype, arr.dtype)
        np.testing.assert_array_equal(out, arr)

    def test_ndarray_dtypes_are_preserved(self):
        for dtype in ("float32", "float64", "int32", "int64", "uint8", "bool"):
            arr = np.arange(6).reshape(2, 3).astype(dtype)
            out = _roundtrip(arr)
            self.assertEqual(out.dtype, np.dtype(dtype), dtype)
            np.testing.assert_array_equal(out, arr)

    def test_decoded_ndarray_is_writable(self):
        out = _roundtrip(np.zeros((2, 2)))
        out[0, 0] = 1.0  # would raise on a frombuffer view

    def test_numpy_scalars_become_python_scalars(self):
        out = _roundtrip({"x": np.float64(0.25), "n": np.int64(7)})
        self.assertIsInstance(out["x"], float)
        self.assertIsInstance(out["n"], int)

    def test_trajectory_roundtrip(self):
        traj = Trajectory([[0.0, 0.1, 0.2, 0.0], [0.3, 0.4, 0.5, 1.0]], "move to handle")
        out = _roundtrip(traj)
        self.assertIsInstance(out, Trajectory)
        self.assertEqual(out.points, traj.points)
        self.assertEqual(out.desc, traj.desc)

    def test_non_string_dict_keys_are_preserved(self):
        out = _roundtrip({1: "a", 2: "b"})
        self.assertEqual(out, {1: "a", 2: "b"})

    def test_frames_are_single_lines(self):
        # Newline framing is only safe if json escapes embedded newlines.
        frame = dumps(["line1\nline2\r\nline3"])
        self.assertEqual(frame.count(b"\n"), 1)
        self.assertTrue(frame.endswith(b"\n"))
        self.assertEqual(loads(frame.decode("utf-8").rstrip("\n"))[0], "line1\nline2\r\nline3")

    def test_unserializable_type_raises_a_helpful_error(self):
        with self.assertRaises(TypeError) as ctx:
            encode(object())
        self.assertIn("json_ipc", str(ctx.exception))

    def test_encode_is_json_native(self):
        import json
        json.dumps(encode({"a": np.zeros(3), "b": (1, 2)}))

    def test_decode_leaves_plain_dicts_alone(self):
        self.assertEqual(decode({"viewMatrix": [1, 2]}), {"viewMatrix": [1, 2]})


class _Server(threading.Thread):
    """Accepts one client and echoes back whatever handler says."""

    daemon = True

    def __init__(self, server, handler):
        super().__init__()
        self.server = server
        self.handler = handler
        self.error = None

    def run(self):
        try:
            endpoint = self.server.accept(timeout=10)
            self.handler(endpoint)
        except Exception as exc:  # surfaced by the test thread
            self.error = exc


class TestTransport(unittest.TestCase):
    def setUp(self):
        self.server = JsonIpcServer("127.0.0.1", 0)
        self.addCleanup(self.server.close)

    def _start(self, handler):
        thread = _Server(self.server, handler)
        thread.start()
        conn = JsonIpcConnection("127.0.0.1", self.server.port, timeout=10)
        self.addCleanup(conn.close)
        return conn, thread

    def test_handshake_then_command_response(self):
        def handler(ep):
            # Mirrors run_simulation_environment: send the handshake, then serve commands.
            ep.send([[0.1, 0.2, 0.3], "coords", {"door": {"pos": [1.0, 0.0, 0.0]}}, "ready"])
            cmd = ep.recv(timeout=10)
            ep.send({"echo": cmd})

        conn, thread = self._start(handler)
        conn.wait_until_ready(timeout=10)

        eef, coords, state, msg = conn.recv(timeout=10)
        self.assertEqual(eef, [0.1, 0.2, 0.3])
        self.assertEqual(coords, "coords")
        self.assertEqual(state["door"]["pos"], [1.0, 0.0, 0.0])
        self.assertEqual(msg, "ready")

        conn.send([7, np.eye(4)])
        echoed = conn.recv(timeout=10)["echo"]
        self.assertEqual(echoed[0], 7)
        np.testing.assert_array_equal(echoed[1], np.eye(4))

        thread.join(timeout=5)
        self.assertIsNone(thread.error)

    def test_poll_is_non_blocking_and_becomes_true(self):
        ready = threading.Event()

        def handler(ep):
            ready.wait(5)
            ep.send(["late"])
            # Keep the socket open so the client's poll() does not see EOF.
            ep.recv(timeout=10)

        conn, thread = self._start(handler)
        conn.wait_until_ready(timeout=10)
        self.assertFalse(conn.poll())  # nothing sent yet, must not block
        ready.set()
        self.assertTrue(conn.poll(timeout=5))
        self.assertEqual(conn.recv(timeout=5), ["late"])
        conn.send(["bye"])
        thread.join(timeout=5)

    def test_large_message_is_reassembled(self):
        # Far bigger than one TCP segment, to prove the line buffer spans reads.
        big = np.random.RandomState(1).rand(200, 4, 4)

        def handler(ep):
            ep.send([big])

        conn, thread = self._start(handler)
        got = conn.recv(timeout=10)[0]
        np.testing.assert_array_equal(got, big)
        thread.join(timeout=5)

    def test_back_to_back_messages_are_not_merged(self):
        def handler(ep):
            for i in range(5):
                ep.send([i])

        conn, thread = self._start(handler)
        self.assertEqual([conn.recv(timeout=10)[0] for _ in range(5)], [0, 1, 2, 3, 4])
        thread.join(timeout=5)

    def test_recv_timeout_raises(self):
        def handler(ep):
            ep.recv(timeout=5)

        conn, _ = self._start(handler)
        conn.wait_until_ready(timeout=10)
        with self.assertRaises(TimeoutError):
            conn.recv(timeout=0.2)
        conn.send(["bye"])

    def test_peer_close_raises_connection_error(self):
        def handler(ep):
            ep.send(["one"])
            ep.close()

        conn, _ = self._start(handler)
        self.assertEqual(conn.recv(timeout=10), ["one"])
        with self.assertRaises(ConnectionError):
            conn.recv(timeout=5)

    def test_wait_until_ready_reports_a_dead_child(self):
        class _DeadProcess:
            returncode = 3

            def poll(self):
                return 3

        # Port 1 is never listening; the dead-process check must win over the timeout.
        conn = JsonIpcConnection("127.0.0.1", 1, timeout=30)
        with self.assertRaises(ConnectionError) as ctx:
            conn.wait_until_ready(process=_DeadProcess(), timeout=30)
        self.assertIn("exited with code 3", str(ctx.exception))

    def test_poll_before_connect_is_false(self):
        conn = JsonIpcConnection("127.0.0.1", 1, timeout=1)
        self.assertFalse(conn.poll())


if __name__ == "__main__":
    unittest.main()
