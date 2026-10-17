"""Synchronous client for :mod:`gpu_server`.

URL schemes:

* ``ws://host:8765/v1/ws`` (or ``wss://``) - FastAPI server; needs ``websockets``.
* ``tcp://host:8766`` - raw TCP transport; stdlib only.
* ``inproc://`` - no server: a :class:`~gpu_server.host.ModuleHost` in this process (when
  the caller's env already has the model's dependencies and the GPU). Arrays are passed by
  reference, nothing is serialised. ``inproc://?modules=cotracker,echo`` limits the modules.

Example::

    from gpu_server.client import GpuClient
    with GpuClient("tcp://127.0.0.1:8766") as c:
        print(c.ping())
        ct = c.open("cotracker")
        ct.call("init", frame=img0, points=pts)
        reply, out = ct.call("update", frame=img1)      # out["points"], out["visible"], ...
        reply, out = c.call("romav2", "match", img_a=a, img_b=b, points_a=p, box=[..])

Array keyword arguments (``numpy.ndarray``) travel as arrays; everything else goes in
``args``. Errors on the server raise :class:`GpuServerError`; transport failures raise
``ConnectionError`` with a hint on how to start the server.
"""

import itertools
import os
import socket
import struct
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from gpu_server.protocol import pack, unpack

_LEN = struct.Struct(">Q")
START_HINT = ("start it with e.g. `python -m gpu_server.server --modules cotracker --preload cotracker` "
              "in the GPU env (see gpu_server/README.md)")


class GpuServerError(RuntimeError):
    """The server answered ``ok: false``."""

    def __init__(self, reply):
        self.reply = reply
        super().__init__(f"{reply.get('error_type', 'Error')}: {reply.get('error')}")


def _split_arrays(kwargs):
    args, arrays = {}, {}
    for k, v in kwargs.items():
        (arrays if isinstance(v, np.ndarray) else args)[k] = v
    return args, arrays


class _TcpTransport:
    def __init__(self, host, port, timeout):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _recv_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("gpu_server closed the TCP connection")
            got += k
        return buf

    def roundtrip(self, payload):
        self.sock.sendall(_LEN.pack(len(payload)) + payload)
        (n,) = _LEN.unpack(self._recv_exact(_LEN.size))
        return bytes(self._recv_exact(n))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class _WsTransport:
    def __init__(self, url, timeout):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("ws:// needs the 'websockets' package (pip install websockets); "
                               "or use tcp://") from exc
        # compression=None: frames are raw pixels, deflate would cost ms for nothing.
        self.ws = connect(url, max_size=None, compression=None, open_timeout=timeout,
                          close_timeout=2)
        self.timeout = timeout

    def roundtrip(self, payload):
        self.ws.send(payload)
        return self.ws.recv(timeout=self.timeout)

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


class _InprocTransport:
    def __init__(self, query):
        from gpu_server.host import ModuleHost
        mods = parse_qs(query).get("modules")
        self.host = ModuleHost(mods[0].split(",") if mods else None)

    def request(self, header, arrays):
        return self.host.handle(header, arrays, owner="inproc")

    def close(self):
        self.host.shutdown()


class GpuClient:
    def __init__(self, url, timeout=60.0, token=None, image_codec="raw"):
        self.url = url
        self.timeout = float(timeout)
        self.token = token if token is not None else os.environ.get("GPU_SERVER_TOKEN")
        self.image_codec = image_codec
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        u = urlparse(url)
        self.scheme = u.scheme
        try:
            if u.scheme == "tcp":
                self._t = _TcpTransport(u.hostname or "127.0.0.1", u.port or 8766, self.timeout)
            elif u.scheme in ("ws", "wss"):
                self._t = _WsTransport(url, self.timeout)
            elif u.scheme == "inproc":
                self._t = _InprocTransport(u.query)
            else:
                raise ValueError(f"unsupported gpu_server URL {url!r} (ws://, tcp://, inproc://)")
        except (OSError, TimeoutError) as exc:
            raise ConnectionError(f"cannot reach gpu_server at {url}: {exc}; {START_HINT}") from exc
        self.last_rtt_ms = None

    # -- core --------------------------------------------------------------
    def request(self, header, arrays=None):
        header = dict(header)
        header["id"] = next(self._ids)
        if self.token:
            header["token"] = self.token
        with self._lock:
            t0 = time.perf_counter()
            if self.scheme == "inproc":
                reply, out = self._t.request(header, dict(arrays or {}))
            else:
                try:
                    raw = self._t.roundtrip(pack(header, arrays, self.image_codec))
                except (OSError, TimeoutError) as exc:
                    raise ConnectionError(f"gpu_server at {self.url} failed mid-request: {exc}") from exc
                reply, out = unpack(raw)
            self.last_rtt_ms = 1e3 * (time.perf_counter() - t0)
        if reply.get("id") != header["id"]:
            raise ConnectionError(f"reply id {reply.get('id')} != request id {header['id']}")
        if not reply.get("ok"):
            raise GpuServerError(reply)
        return reply, out

    def ping(self):
        return self.request({"op": "ping"})[0]

    def modules(self):
        return self.request({"op": "list"})[0]["modules"]

    def load(self, module, **config):
        return self.request({"op": "load", "module": module, "config": config or None})[0]

    def unload(self, module):
        return self.request({"op": "unload", "module": module})[0]

    def call(self, module, method, **kwargs):
        args, arrays = _split_arrays(kwargs)
        return self.request({"op": "call", "module": module, "method": method, "args": args}, arrays)

    def open(self, module, **params):
        sid = self.request({"op": "open", "module": module, "params": params})[0]["session"]
        return RemoteSession(self, module, sid)

    def close(self):
        self._t.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RemoteSession:
    """A stateful module session (e.g. one CoTracker stream)."""

    def __init__(self, client, module, sid):
        self.client = client
        self.module = module
        self.sid = sid

    def call(self, method, **kwargs):
        args, arrays = _split_arrays(kwargs)
        return self.client.request({"op": "call", "session": self.sid, "method": method, "args": args},
                                   arrays)

    def close(self):
        if self.sid is not None:
            try:
                self.client.request({"op": "close", "session": self.sid})
            finally:
                self.sid = None
