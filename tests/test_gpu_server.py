"""Tests for gpu_server (codec, host, TCP/WS transports) and the remote tracker provider.

No model and no GPU: the ``echo`` module and a CoTracker module backed by the fake predictor
of ``tests/test_cotracker_tracker.py`` (registered by dotted path, which also exercises the
dynamic loader). The WS tests skip when fastapi/uvicorn/websockets are missing.

Run:  python -m pytest tests\\test_gpu_server.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpu_server.client import GpuClient, GpuServerError  # noqa: E402
from gpu_server.host import ModuleHost  # noqa: E402
from gpu_server.protocol import ProtocolError, pack, unpack  # noqa: E402
from gpu_server.server import BackgroundServer  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import websockets.sync.client  # noqa: F401
    HAVE_WS = True
except ImportError:  # pragma: no cover
    HAVE_WS = False

if HAVE_TORCH:
    from gpu_server.modules.cotracker import CoTrackerModule

    class FakeCoTrackerModule(CoTrackerModule):
        """CoTracker module whose 'weights' are the unit-test fake predictor."""

        def load(self):
            from tests.test_cotracker_tracker import FakeCoTracker
            self._model, self.device = FakeCoTracker(step=4, dx=1.5), "cpu"

FAKE_CT = "tests.test_gpu_server:FakeCoTrackerModule"
FRAME = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)


class TestProtocol(unittest.TestCase):
    def test_roundtrip_raw_and_png(self):
        arrays = {"frame": FRAME, "pts": np.array([[1.5, 2.25]]), "vis": np.array([True, False])}
        for codec in ("raw", "png"):
            h, out = unpack(pack({"op": "x", "meta": {"a": np.float32(1.5), "n": float("nan")}},
                                 arrays, image_codec=codec))
            self.assertEqual(h["meta"], {"a": 1.5, "n": None})
            for k, v in arrays.items():
                np.testing.assert_array_equal(out[k], v)
                self.assertEqual(out[k].dtype, v.dtype)
            self.assertTrue(out["frame"].flags.writeable)

    def test_jpeg_is_lossy_only_for_rgb_images(self):
        smooth = np.repeat((np.indices((48, 64)).sum(0) * 2).astype(np.uint8)[..., None], 3, axis=2)
        rgba = np.dstack([FRAME, np.full(FRAME.shape[:2], 7, np.uint8)])
        arrays = {"frame": smooth, "rgba": rgba, "pts": np.array([[1.5, 2.25]])}
        _h, out = unpack(pack({"op": "x"}, arrays, image_codec="jpeg"))
        for k, v in arrays.items():
            self.assertEqual((out[k].shape, out[k].dtype), (v.shape, v.dtype))
        self.assertLess(np.abs(out["frame"].astype(int) - smooth).mean(), 3.0)
        np.testing.assert_array_equal(out["rgba"], rgba)      # alpha images fall back to png
        np.testing.assert_array_equal(out["pts"], arrays["pts"])

    def test_garbage_is_a_protocol_error(self):
        with self.assertRaises(ProtocolError):
            unpack(b"not a message")


class TestHost(unittest.TestCase):
    def test_errors_are_replies_not_exceptions(self):
        host = ModuleHost(["echo"])
        for header in ({"op": "nope"}, {"op": "call", "module": "missing", "method": "echo"},
                       {"op": "call", "module": "echo", "method": "fail", "args": {"msg": "x"}},
                       {"op": "call", "session": "dead", "method": "add"}):
            reply, _ = host.handle(dict(header, id=7))
            self.assertFalse(reply["ok"], header)
            self.assertEqual(reply["id"], 7)

    def test_modules_load_lazily(self):
        host = ModuleHost(["echo"])
        self.assertFalse(host.describe()["echo"]["loaded"])
        host.handle({"op": "call", "module": "echo", "method": "echo"})
        self.assertTrue(host.describe()["echo"]["loaded"])

    def test_token_is_enforced(self):
        host = ModuleHost(["echo"], token="s3cret")
        self.assertFalse(host.handle({"op": "ping"})[0]["ok"])
        self.assertTrue(host.handle({"op": "ping", "token": "s3cret"})[0]["ok"])

    def test_unregistered_modules_cannot_be_loaded_over_the_wire(self):
        reply, _ = ModuleHost(["echo"]).handle({"op": "load", "module": "os"})
        self.assertFalse(reply["ok"])

    def test_owner_disconnect_closes_its_sessions_only(self):
        host = ModuleHost(["echo"])
        a = host.handle({"op": "open", "module": "echo"}, owner="A")[0]["session"]
        b = host.handle({"op": "open", "module": "echo"}, owner="B")[0]["session"]
        self.assertEqual(host.close_owner("A"), 1)
        self.assertFalse(host.handle({"op": "call", "session": a, "method": "add"})[0]["ok"])
        self.assertTrue(host.handle({"op": "call", "session": b, "method": "add"})[0]["ok"])


class _TransportCases:
    """Shared by TCP and WS (and inproc); subclasses set ``self.url``."""

    def test_echo_arrays_and_errors(self):
        with GpuClient(self.url) as c:
            self.assertIn("echo", c.ping()["modules"])
            reply, out = c.call("echo", "echo", frame=FRAME, k=3)
            np.testing.assert_array_equal(out["frame"], FRAME)
            self.assertEqual(reply["args"], {"k": 3})
            with self.assertRaises(GpuServerError):
                c.call("echo", "fail", msg="bad")
            self.assertEqual(c.call("echo", "echo")[0]["args"], {})  # connection survives errors

    def test_sessions_are_stateful(self):
        with GpuClient(self.url) as c:
            s1, s2 = c.open("echo"), c.open("echo", start=10)
            s1.call("add", n=2)
            self.assertEqual(s1.call("add")[0]["count"], 3)
            self.assertEqual(s2.call("add")[0]["count"], 11)
            s1.close()
            with self.assertRaises(GpuServerError):
                c.request({"op": "call", "session": "x" * 12, "method": "add"})

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_remote_tracker_matches_the_local_provider_exactly(self):
        from providers.trackers.cotracker_tracker import CoTrackerTracker
        from providers.trackers.factory import get_tracker
        from tests.test_cotracker_tracker import FakeCoTracker

        seeds = np.array([[10.0, 10.0], [30.0, 20.0], [70.0, 5.0]])   # last one off-image
        local = CoTrackerTracker(model=FakeCoTracker(step=4, dx=1.5), device="cpu")
        remote = get_tracker("remote", url=self.url, module="fakect")
        try:
            local.init(FRAME, seeds, obj_id="h")
            remote.init(FRAME, seeds, obj_id="h")
            self.assertEqual(remote.n_points, 3)
            for k in range(1, 11):
                a, b = local.update(FRAME), remote.update(FRAME)
                np.testing.assert_array_equal(a.points, b.points)
                np.testing.assert_array_equal(a.visible, b.visible)
                np.testing.assert_array_equal(a.scores, b.scores)
                for key in ("stale_frames", "fresh", "flushes", "step", "window"):
                    self.assertEqual(a.meta[key], b.meta[key], (k, key))
                self.assertGreaterEqual(b.meta["remote_rtt_ms"], 0.0)
                self.assertEqual(local._flushes, remote._flushes)   # read by session._timed_update
            remote.reset()
            self.assertFalse(remote.initialised)
            remote.init(FRAME, seeds[:2])                 # re-seed after reset
            self.assertEqual(len(remote.update(FRAME).points), 2)
        finally:
            remote.close()


def _host():
    host = ModuleHost(["echo"])
    if HAVE_TORCH:
        host.register("fakect", FAKE_CT)
    return host


class TestTcpTransport(_TransportCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = BackgroundServer(_host(), tcp="127.0.0.1:0")
        cls.url = cls.server.tcp_url

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_disconnect_closes_server_sessions(self):
        c = GpuClient(self.url)
        c.open("echo")
        c.close()
        import time
        for _ in range(50):
            if self.server.host.describe()["echo"]["sessions"] == 0:
                break
            time.sleep(0.02)
        self.assertEqual(self.server.host.describe()["echo"]["sessions"], 0)


@unittest.skipUnless(HAVE_WS, "fastapi/uvicorn/websockets not installed")
class TestWsTransport(_TransportCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = BackgroundServer(_host(), ws="127.0.0.1:0")
        cls.url = cls.server.ws_url

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_rest_endpoints(self):
        import json
        import urllib.request
        base = self.url.replace("ws://", "http://").replace("/v1/ws", "")
        health = json.load(urllib.request.urlopen(base + "/health", timeout=5))
        self.assertTrue(health["ok"])
        req = urllib.request.Request(base + "/v1/call/echo/echo", data=pack({"args": {"q": 1}}, {"a": FRAME}),
                                     headers={"content-type": "application/octet-stream"})
        reply, out = unpack(urllib.request.urlopen(req, timeout=5).read())
        self.assertEqual(reply["args"], {"q": 1})
        np.testing.assert_array_equal(out["a"], FRAME)


class TestInproc(_TransportCases, unittest.TestCase):
    url = "inproc://"

    def test_remote_tracker_matches_the_local_provider_exactly(self):
        self.skipTest("inproc:// cannot register test modules; covered over TCP/WS")


def _wait_for(pred, timeout=5.0, period=0.02):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(period)
    return pred()


class TestServerLifecycle(unittest.TestCase):
    def test_shutdown_is_refused_while_another_client_holds_a_session(self):
        srv = BackgroundServer(_host(), tcp="127.0.0.1:0")
        try:
            with GpuClient(srv.tcp_url) as other, GpuClient(srv.tcp_url) as me:
                other.open("echo")
                with self.assertRaisesRegex(GpuServerError, "busy"):
                    me.request({"op": "shutdown", "wait_s": 0.2})
                self.assertTrue(srv.running)
                me.request({"op": "shutdown", "force": True})
            self.assertTrue(srv.wait(5))
            self.assertIn("shutdown requested", srv.reason)
        finally:
            srv.stop()

    def test_shutdown_waits_for_a_departing_client(self):
        srv = BackgroundServer(_host(), tcp="127.0.0.1:0")
        try:
            other = GpuClient(srv.tcp_url)
            other.open("echo")
            other.close()                    # the server notices asynchronously
            with GpuClient(srv.tcp_url) as me:
                self.assertTrue(me.request({"op": "shutdown", "wait_s": 5})[0]["shutting_down"])
            self.assertTrue(srv.wait(5))
        finally:
            srv.stop()

    def test_idle_timeout_stops_the_server_only_without_requests(self):
        import time
        srv = BackgroundServer(_host(), tcp="127.0.0.1:0", idle_timeout_s=0.6)
        try:
            with GpuClient(srv.tcp_url) as c:
                t_end = time.monotonic() + 1.5
                while time.monotonic() < t_end:          # busy for 2.5x the timeout
                    c.call("echo", "echo")
                    time.sleep(0.1)
                self.assertTrue(srv.running)
                self.assertEqual(c.ping()["idle_timeout_s"], 0.6)
            self.assertTrue(srv.wait(5))
            self.assertTrue(srv.reason.startswith("idle"), srv.reason)
            with self.assertRaises(ConnectionError):
                GpuClient(srv.tcp_url, timeout=1)
        finally:
            srv.stop()

    def test_ports_listen_before_preload_finishes(self):
        host = ModuleHost({"slow": "gpu_server.modules.echo:EchoModule"}, {"slow": {"load_s": 1.0}})
        srv = BackgroundServer(host, tcp="127.0.0.1:0", preload=["slow"])
        try:
            with GpuClient(srv.tcp_url) as c:
                first = c.ping()
                self.assertEqual(first["loading"], ["slow"])
                self.assertEqual(first["loaded"], [])
                s = c.open("slow")                     # queues behind the load
                self.assertEqual(s.call("add")[0]["count"], 1)
                self.assertEqual(c.ping()["loaded"], ["slow"])
        finally:
            srv.stop()


def _free_port_pair():
    """A port p with p-1 also free (the launcher binds ws on p-1 for a tcp:// URL)."""
    import socket
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        try:
            with socket.socket() as s2:
                s2.bind(("127.0.0.1", p - 1))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port pair")


class TestLauncher(unittest.TestCase):
    """Real subprocesses: `python -m gpu_server.server` started detached by the launcher."""

    def setUp(self):
        import tempfile

        import config
        from gpu_server import launcher
        self.launcher = launcher
        self._env = {k: os.environ.get(k) for k in ("GPU_SERVER_PYTHON", "GPU_SERVER_POLICY")}
        self._log_dir = config.gpu_server_log_dir
        self.tmp = tempfile.mkdtemp(prefix="gpu_server_launcher_")
        config.gpu_server_log_dir = self.tmp
        os.environ["GPU_SERVER_PYTHON"] = sys.executable      # not the GPU conda env
        self._policy = launcher._policy
        self.handles = []

    def tearDown(self):
        import config
        for h in self.handles:
            if h.proc is not None and h.proc.poll() is None:
                h.proc.kill()
                h.proc.wait(10)
            self.launcher._servers.pop(h.url, None)
        config.gpu_server_log_dir = self._log_dir
        self.launcher._policy = self._policy
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_launch_attach_and_policy_shutdown(self):
        url = f"tcp://127.0.0.1:{_free_port_pair()}"
        self.launcher.configure("no-shutdown")
        h = self.launcher.ensure_server(url, "echo", timeout=60)
        self.handles.append(h)
        self.assertTrue(h.launched and h.ready)
        reply = self.launcher.ping(url)
        self.assertEqual(reply["pid"], h.pid)
        self.assertEqual(reply["loaded"], ["echo"])
        self.assertEqual(reply["idle_timeout_s"], 1800.0)
        # a second caller (another tracker / process) attaches instead of launching
        self.assertIs(self.launcher.ensure_server(url, "echo"), h)
        # no-shutdown: the exit hook leaves it running
        self.launcher._at_exit()
        self.assertIsNone(h.proc.poll())
        # shutdown: the exit hook stops it
        self.launcher.configure("shutdown")
        self.launcher._at_exit()
        self.assertEqual(h.proc.wait(15), 0)
        self.assertIsNone(self.launcher.ping(url, timeout=0.5))

    def test_launched_server_exits_when_idle(self):
        url = f"tcp://127.0.0.1:{_free_port_pair()}"
        self.launcher.configure("no-shutdown")
        h = self.launcher.ensure_server(url, "echo", timeout=60, idle_timeout_s=1.0)
        self.handles.append(h)
        self.assertEqual(h.proc.wait(20), 0)
        with open(h.log_path, encoding="utf-8") as f:
            self.assertIn("idle for", f.read())

    def test_no_launch_policy_fails_loudly(self):
        url = f"tcp://127.0.0.1:{_free_port_pair()}"
        self.launcher.configure("no-launch")
        with self.assertRaisesRegex(ConnectionError, "no-launch"):
            self.launcher.ensure_server(url, "echo")

    def test_missing_gpu_env_fails_with_setup_hint(self):
        import config
        from unittest import mock
        os.environ.pop("GPU_SERVER_PYTHON", None)
        with mock.patch.object(config, "gpu_server_python", None), \
                mock.patch.object(config, "gpu_server_conda_env", "no_such_env_xyz"):
            with self.assertRaisesRegex(self.launcher.GpuServerEnvError, "3D Tracking"):
                self.launcher.resolve_python("cotracker")
            # prewarm must not swallow it: the run has to stop at startup
            self.launcher.configure("no-shutdown")
            with self.assertRaises(self.launcher.GpuServerEnvError):
                self.launcher.prewarm(f"tcp://127.0.0.1:{_free_port_pair()}", "cotracker")
            self.assertEqual(self.launcher.resolve_python("echo"), sys.executable)
        os.environ["GPU_SERVER_PYTHON"] = os.path.join(self.tmp, "missing", "python.exe")
        with self.assertRaisesRegex(self.launcher.GpuServerEnvError, "does not exist"):
            self.launcher.resolve_python("cotracker")


if __name__ == "__main__":
    unittest.main()
