"""JSON-lines IPC over localhost TCP, shaped like a ``multiprocessing.Connection``.

The Genesis child runs under a *different* interpreter (see
:mod:`providers.genesis_launcher`) with a different numpy (1.26 here, 2.x there) and a
different torch. ``pickle`` — which is what ``multiprocessing.Pipe`` uses — is not safe
across that boundary: numpy's array pickle format is version-sensitive and a torch tensor
would drag the whole torch import into the payload. So the wire format is plain JSON with
an explicit codec for the few non-JSON types this repo actually sends.

The public API is deliberately the subset of ``multiprocessing.Connection`` that
``env.run_simulation_environment`` and ``agent_runner`` already use::

    conn.send(payload)          # payload is a list or a dict
    conn.recv(timeout=None)     # blocks by default, like a Pipe
    conn.poll(timeout=0)        # non-blocking by default, like a Pipe
    conn.close()

so neither side needs a transport-specific branch.

Framing is one JSON document per ``\\n``-terminated line. ``json.dumps`` escapes newlines
inside strings, so a bare newline can only ever be a frame boundary. Lines stay
human-readable, which matters because this is the seam where two simulators are compared
and it has to be debuggable with nothing more than a text editor.

Set ``LMTG_IPC_DEBUG=1`` to log every frame (direction, command, byte size) to stderr.

Timeouts
--------
``timeout`` configures **connection establishment only**. ``recv()`` blocks forever unless
given an explicit per-call timeout, because that is what the PyBullet ``Pipe`` path does
and because a single ``EXECUTE_TRAJECTORY`` legitimately takes minutes of wall clock.
"""

import base64
import json
import os
import select
import socket
import sys
import time

DEFAULT_CONNECT_TIMEOUT = 30.0
_RECV_CHUNK = 1 << 16

# Codec tags. Prefixed and suffixed with dunders so they cannot collide with a real
# payload key (every dict this repo sends over IPC uses plain identifier-ish keys).
_TAG_NDARRAY = "__ndarray__"
_TAG_TUPLE = "__tuple__"
_TAG_TRAJECTORY = "__trajectory__"
_TAG_BYTES = "__bytes__"
_TAG_DICT = "__dict__"


def _debug_enabled():
    return (os.environ.get("LMTG_IPC_DEBUG") or "").strip() not in ("", "0", "false", "False")


def _debug(direction, payload, nbytes):
    if not _debug_enabled():
        return
    try:
        if isinstance(payload, list) and payload:
            head = payload[0]
        elif isinstance(payload, dict):
            head = "|".join(sorted(payload.keys()))
        else:
            head = type(payload).__name__
        print(f"[ipc] {direction} head={head!s:.60} bytes={nbytes}", file=sys.stderr, flush=True)
    except Exception:
        pass


# --- Codec ---------------------------------------------------------------

def _numpy():
    """Return numpy if importable, else None. Kept optional so the codec never hard-depends."""
    try:
        import numpy as np
        return np
    except Exception:
        return None


def encode(obj):
    """Convert ``obj`` into something ``json.dumps`` accepts, losslessly for our types."""
    np = _numpy()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, bytes):
        return {_TAG_BYTES: base64.b64encode(obj).decode("ascii")}

    if np is not None:
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                # Object arrays have no portable binary form; fall back to nested lists
                # rather than failing, since they only ever appear by accident.
                return [encode(v) for v in obj.tolist()]
            arr = np.ascontiguousarray(obj)
            return {
                _TAG_NDARRAY: base64.b64encode(arr.tobytes()).decode("ascii"),
                "dtype": arr.dtype.str,  # '<f8' — endian-explicit, stable across numpy 1/2
                "shape": list(arr.shape),
            }
        if isinstance(obj, np.generic):
            return obj.item()

    # torch tensors can appear from a Genesis-side state query that forgot to convert.
    # Handle them defensively instead of crashing the whole episode.
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(obj, torch.Tensor):
        return encode(obj.detach().cpu().numpy())

    if isinstance(obj, tuple):
        return {_TAG_TUPLE: [encode(v) for v in obj]}

    if isinstance(obj, list):
        return [encode(v) for v in obj]

    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: encode(v) for k, v in obj.items()}
        # JSON silently stringifies non-str keys; preserve them explicitly instead.
        return {_TAG_DICT: [[encode(k), encode(v)] for k, v in obj.items()]}

    if _is_trajectory(obj):
        return {_TAG_TRAJECTORY: {"points": encode(obj.points), "desc": encode(obj.desc)}}

    raise TypeError(
        f"json_ipc cannot serialize {type(obj).__name__}. Add a codec tag for it in "
        f"providers/json_ipc.py, or convert it to a list/dict before sending."
    )


def _is_trajectory(obj):
    """Duck-type ``common_utils.Trajectory`` so the codec never imports the app layer."""
    return (
        obj.__class__.__name__ == "Trajectory"
        and hasattr(obj, "points")
        and hasattr(obj, "desc")
    )


def decode(obj):
    """Inverse of :func:`encode`."""
    if isinstance(obj, list):
        return [decode(v) for v in obj]

    if not isinstance(obj, dict):
        return obj

    if _TAG_NDARRAY in obj:
        np = _numpy()
        raw = base64.b64decode(obj[_TAG_NDARRAY])
        if np is None:
            raise RuntimeError("Received an ndarray over IPC but numpy is not importable.")
        # frombuffer gives a read-only view onto an immutable bytes object; copy so the
        # receiver can mutate it like any other array.
        return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()

    if _TAG_BYTES in obj:
        return base64.b64decode(obj[_TAG_BYTES])

    if _TAG_TUPLE in obj:
        return tuple(decode(v) for v in obj[_TAG_TUPLE])

    if _TAG_DICT in obj:
        return {decode(k): decode(v) for k, v in obj[_TAG_DICT]}

    if _TAG_TRAJECTORY in obj:
        from common_utils import Trajectory
        body = obj[_TAG_TRAJECTORY]
        return Trajectory(decode(body["points"]), decode(body["desc"]))

    return {k: decode(v) for k, v in obj.items()}


def dumps(payload):
    """Serialize one payload to a newline-terminated frame."""
    return (json.dumps(encode(payload), separators=(",", ":")) + "\n").encode("utf-8")


def loads(line):
    """Deserialize one frame body (no trailing newline required)."""
    return decode(json.loads(line))


# --- Endpoint ------------------------------------------------------------

class JsonIpcEndpoint:
    """A connected socket presenting the ``send``/``recv``/``poll`` Connection contract."""

    def __init__(self, sock, peer_name="peer"):
        self._sock = sock
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._peer_name = peer_name
        self._buf = b""
        self._queue = []
        self._closed = False

    # -- outbound
    def send(self, payload):
        if self._closed:
            raise ConnectionError(f"IPC connection to {self._peer_name} is closed")
        frame = dumps(payload)
        _debug("send", payload, len(frame))
        try:
            self._sock.sendall(frame)
        except OSError as exc:
            self._closed = True
            raise ConnectionError(f"IPC send to {self._peer_name} failed: {exc}") from exc

    # -- inbound
    def _drain_buffer(self):
        while True:
            line, sep, rest = self._buf.partition(b"\n")
            if not sep:
                return
            self._buf = rest
            if line.strip():
                self._queue.append(loads(line.decode("utf-8")))

    def _read_once(self, timeout):
        """Wait up to ``timeout`` for bytes; return True if any arrived."""
        if self._closed:
            raise ConnectionError(f"IPC connection to {self._peer_name} is closed")
        readable, _, _ = select.select([self._sock], [], [], timeout)
        if not readable:
            return False
        try:
            chunk = self._sock.recv(_RECV_CHUNK)
        except OSError as exc:
            self._closed = True
            raise ConnectionError(f"IPC recv from {self._peer_name} failed: {exc}") from exc
        if not chunk:
            self._closed = True
            raise ConnectionError(f"IPC connection to {self._peer_name} closed by peer")
        self._buf += chunk
        self._drain_buffer()
        return True

    def poll(self, timeout=0):
        """True if a whole message is ready. Non-blocking by default, like ``Pipe.poll()``."""
        if self._queue:
            return True
        if self._closed:
            return False
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            # A read can deliver a partial frame, so keep going until the queue fills
            # or the budget is spent.
            self._read_once(remaining if remaining is not None else 0.5)
            if self._queue:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def recv(self, timeout=None):
        """Return the next message. Blocks indefinitely unless ``timeout`` is given."""
        deadline = None
        if timeout is not None and float(timeout) > 0:
            deadline = time.monotonic() + float(timeout)
        while not self._queue:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"Timed out waiting for a message from {self._peer_name}")
            # Cap the select() slice so a blocking recv still reacts to Ctrl-C promptly.
            self._read_once(0.5 if remaining is None else min(0.5, remaining))
        payload = self._queue.pop(0)
        _debug("recv", payload, 0)
        return payload

    def close(self):
        self._closed = True
        try:
            self._sock.close()
        except Exception:
            pass

    @property
    def closed(self):
        return self._closed


# --- Server (Genesis child side) -----------------------------------------

class JsonIpcServer:
    """Listening socket for the simulator child; hands back a :class:`JsonIpcEndpoint`."""

    def __init__(self, host, port, backlog=1):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port)))
        self._sock.listen(backlog)
        self.host, self.port = self._sock.getsockname()

    def accept(self, timeout=None):
        readable, _, _ = select.select([self._sock], [], [], timeout)
        if not readable:
            raise TimeoutError(f"No IPC client connected to {self.host}:{self.port} within {timeout}s")
        conn, addr = self._sock.accept()
        return JsonIpcEndpoint(conn, peer_name=f"{addr[0]}:{addr[1]}")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# --- Client (agent side) -------------------------------------------------

class JsonIpcConnection:
    """Client endpoint used by ``agent_runner``; connects lazily and retries.

    Construction never fails just because the child has not finished booting — Genesis
    takes tens of seconds to ``gs.init`` and ``scene.build()``. Call
    :meth:`wait_until_ready` (or simply ``send``/``recv``) to force the connection.
    """

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = int(port)
        # `timeout` here is the *connect* budget; recv() semantics stay Pipe-like.
        try:
            configured = float(timeout) if timeout is not None else DEFAULT_CONNECT_TIMEOUT
        except (TypeError, ValueError):
            configured = DEFAULT_CONNECT_TIMEOUT
        self.connect_timeout = DEFAULT_CONNECT_TIMEOUT if configured <= 0 else configured
        self._endpoint = None

    # -- connection management
    def _try_connect(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=1.0)
        except OSError:
            return False
        sock.settimeout(None)
        self._endpoint = JsonIpcEndpoint(sock, peer_name=f"{self.host}:{self.port}")
        return True

    def wait_until_ready(self, process=None, timeout=None, poll_interval=0.25):
        """Block until the child accepts a connection.

        ``process`` is the child ``Popen``; if it exits before the port opens we fail
        immediately with its return code instead of burning the whole timeout, because a
        dead child is by far the most common failure and the silent version of it is
        miserable to debug.
        """
        if self._endpoint is not None and not self._endpoint.closed:
            return self
        budget = self.connect_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + budget
        while True:
            if self._try_connect():
                return self
            if process is not None and process.poll() is not None:
                raise ConnectionError(
                    f"Simulator child process exited with code {process.returncode} "
                    f"before opening {self.host}:{self.port}. Check its console output "
                    f"and env_genesis.log."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not connect to the simulator child at {self.host}:{self.port} "
                    f"within {budget:.0f}s."
                )
            time.sleep(poll_interval)

    def _require(self):
        if self._endpoint is None or self._endpoint.closed:
            self.wait_until_ready()
        return self._endpoint

    # -- Connection contract
    def send(self, payload):
        return self._require().send(payload)

    def recv(self, timeout=None):
        return self._require().recv(timeout=timeout)

    def poll(self, timeout=0):
        if self._endpoint is None:
            return False
        return self._endpoint.poll(timeout)

    def close(self):
        if self._endpoint is not None:
            self._endpoint.close()
