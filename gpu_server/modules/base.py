"""Plugin interface for models served by :mod:`gpu_server`.

A *module* wraps one model family (CoTracker, RoMa, a VLM...). The host imports it lazily
by dotted path (``"pkg.mod:Class"``), constructs it with its config dict, and calls
:meth:`GpuModule.load` once, on that module's own worker thread, the first time it is used
(or at startup with ``--preload``). Nothing heavy may happen in ``__init__`` or at import
time of the plugin file beyond cheap imports, so listing modules stays instant.

Two call styles:

* **stateless** - ``call(method, args, arrays)``: one request, one reply (RoMa matching,
  a VLM prompt).
* **stateful** - ``open_session(**params)`` returns a :class:`GpuSession` that keeps state
  between calls (CoTracker's online window). Sessions belong to the connection that opened
  them and are closed when it disconnects, so a crashed client never leaks GPU state.

Every method gets ``args`` (JSON dict) and ``arrays`` ({name: ndarray}) and returns
``(reply_dict, reply_arrays)``. All calls of one module run on a single worker thread, so a
module never has to be thread-safe; different modules run concurrently.
"""


class ModuleError(RuntimeError):
    """A clean, user-facing error raised by a module (bad args, unknown method...)."""


class GpuSession:
    """Per-client state of a stateful module."""

    def call(self, method, args, arrays):
        fn = getattr(self, "m_" + str(method), None)
        if fn is None:
            raise ModuleError(f"{type(self).__name__}: unknown method {method!r}; "
                              f"available: {sorted(m[2:] for m in dir(self) if m.startswith('m_'))}")
        return fn(args, arrays)

    def close(self):
        """Release per-session resources (default: nothing)."""


class GpuModule:
    """Base class. Methods named ``m_<name>`` are callable as ``<name>``."""

    name = "base"
    description = ""
    stateful = False

    def __init__(self, **config):
        self.config = dict(config)

    def load(self):
        """Load weights. Runs once, on the module's worker thread."""

    def unload(self):
        """Free weights/VRAM. The default drops nothing; override for big models."""

    def info(self):
        """JSON dict shown by ``list`` (device, dtype, VRAM...)."""
        return {}

    def call(self, method, args, arrays):
        fn = getattr(self, "m_" + str(method), None)
        if fn is None:
            raise ModuleError(f"module {self.name!r}: unknown method {method!r}; "
                              f"available: {sorted(m[2:] for m in dir(self) if m.startswith('m_'))}")
        return fn(args, arrays)

    def open_session(self, **params):
        raise ModuleError(f"module {self.name!r} is stateless; use 'call' without a session")


def require(arrays, *names):
    """Fetch required input arrays or raise a readable :class:`ModuleError`."""
    missing = [n for n in names if n not in arrays]
    if missing:
        raise ModuleError(f"missing input array(s): {missing}; got {sorted(arrays)}")
    return [arrays[n] for n in names]
