"""Module host: registry, lazy dynamic loading, sessions and request dispatch.

Transport-independent. :meth:`ModuleHost.handle` takes a decoded request
``(header, arrays)`` and returns ``(reply_header, reply_arrays)``; the WebSocket, TCP and
in-process transports are thin wrappers around it.

Ops (``header["op"]``)::

    ping                                   -> {"version", "modules", "loaded", "loading", "pid",
                                               "idle_timeout_s", "idle_s"}
    list                                   -> {"modules": {name: {spec, loaded, stateful, info, stats}}}
    load    {"module", "config"?}          -> {"info"}            (no-op when already loaded)
    unload  {"module"}                     -> {}                  (closes that module's sessions)
    open    {"module", "params"?}          -> {"session"}
    call    {"session" | "module", "method", "args"?} + arrays -> module reply + arrays
    close   {"session"}                    -> {}
    shutdown {"force"?, "wait_s"?}         -> {"shutting_down"}   (network transports only)

``shutdown`` stops the server process. It first waits up to ``wait_s`` (default 10 s) for
sessions of *other* connections to close (a client that just exited may not have been noticed
yet), then refuses with a "busy" error unless ``force`` is true, so one run cannot pull the
server from under another run that shares it.

Every reply has ``ok``, the echoed ``id`` and ``server_ms``. Errors come back as
``{"ok": false, "error", "error_type"}`` instead of killing the connection.

Only *registered* module names can be loaded over the wire. Arbitrary ``pkg.mod:Class``
specs are registered by whoever starts the server (``--module name=pkg.mod:Class``), never
by a client, since importing a dotted path is code execution.
"""

import asyncio
import importlib
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from gpu_server.modules.base import GpuModule, ModuleError
from gpu_server.protocol import PROTOCOL_VERSION

logger = logging.getLogger("gpu_server")

BUILTIN_MODULES = {
    "echo": "gpu_server.modules.echo:EchoModule",
    "cotracker": "gpu_server.modules.cotracker:CoTrackerModule",
    "romav2": "gpu_server.modules.romav2:RoMaV2Module",
}

_MODULE_OPS = ("load", "unload", "open", "call", "close")
SHUTDOWN_WAIT_S = 10.0


def import_spec(spec):
    """``"pkg.mod:Class"`` -> the class."""
    mod_name, _, attr = str(spec).partition(":")
    if not mod_name or not attr:
        raise ValueError(f"module spec must look like 'pkg.mod:Class', got {spec!r}")
    cls = getattr(importlib.import_module(mod_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, GpuModule)):
        raise TypeError(f"{spec!r} is not a gpu_server.modules.base.GpuModule subclass")
    return cls


class _Slot:
    def __init__(self, name, spec, config):
        self.name = name
        self.spec = spec
        self.config = dict(config or {})
        self.instance = None
        self.load_s = None
        self.lock = threading.RLock()
        self.executor = None
        self.calls = 0
        self.busy_s = 0.0

    def get_executor(self):
        # One worker per module: calls of one model are serialised (no thread-safety needed,
        # no two CoTracker windows racing on the GPU); different models overlap.
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-{self.name}")
        return self.executor


class ModuleHost:
    def __init__(self, modules=None, configs=None, token=None):
        """``modules``: list of builtin names or {name: spec}; ``None`` = all builtins.
        ``configs``: {name: {key: value}} passed to the module constructor."""
        configs = dict(configs or {})
        if modules is None:
            modules = dict(BUILTIN_MODULES)
        elif not isinstance(modules, dict):
            unknown = [m for m in modules if m not in BUILTIN_MODULES]
            if unknown:
                raise ValueError(f"unknown builtin module(s) {unknown}; builtins: {sorted(BUILTIN_MODULES)} "
                                 "(register others as name=pkg.mod:Class)")
            modules = {m: BUILTIN_MODULES[m] for m in modules}
        self._slots = {name: _Slot(name, spec, configs.get(name)) for name, spec in modules.items()}
        self._sessions = {}          # sid -> (module name, owner, GpuSession)
        self._lock = threading.Lock()
        self.token = token or None
        # Idle tracking for the server's --idle-timeout-s watchdog.
        self.last_activity = time.monotonic()
        self._inflight = 0
        self.loading = set()         # modules being preloaded in the background
        self.idle_timeout_s = 0.0    # informational (ping); the watchdog lives in server.serve
        # Set by the network server: called with a reason string to stop the process.
        self.on_shutdown = None

    def idle_s(self):
        """Seconds since the last request finished (0 while a request or preload runs)."""
        with self._lock:
            if self._inflight or self.loading:
                return 0.0
            return time.monotonic() - self.last_activity

    def foreign_sessions(self, owner):
        return sum(1 for _, o, _ in list(self._sessions.values()) if o != owner)

    # -- registry ----------------------------------------------------------
    def register(self, name, spec, config=None):
        self._slots[name] = _Slot(name, spec, config)

    def _slot(self, name):
        slot = self._slots.get(name)
        if slot is None:
            raise ModuleError(f"unknown module {name!r}; registered: {sorted(self._slots)}")
        return slot

    def load(self, name, config=None):
        slot = self._slot(name)
        with slot.lock:
            if slot.instance is not None:
                if config and {**slot.config, **config} != slot.config:
                    raise ModuleError(f"module {name!r} is already loaded with config {slot.config}; "
                                      "unload it first to change the config")
                return slot.instance
            if config:
                slot.config.update(config)
            t0 = time.perf_counter()
            logger.info("[host] loading module %s (%s) config=%s", name, slot.spec, slot.config)
            inst = import_spec(slot.spec)(**slot.config)
            inst.load()
            slot.instance = inst
            slot.load_s = time.perf_counter() - t0
            logger.info("[host] module %s ready in %.1f s %s", name, slot.load_s, inst.info())
            return inst

    def unload(self, name):
        slot = self._slot(name)
        with slot.lock:
            for sid in [s for s, (m, _, _) in list(self._sessions.items()) if m == name]:
                self._close_session(sid)
            if slot.instance is not None:
                slot.instance.unload()
                slot.instance = None
                logger.info("[host] module %s unloaded", name)

    async def apreload(self, name):
        """Load ``name`` on its module worker while the transports already serve (pings
        report it under ``loading``; requests for it queue behind the load)."""
        self._slot(name)
        with self._lock:
            self.loading.add(name)
        try:
            await asyncio.get_running_loop().run_in_executor(self._slot(name).get_executor(),
                                                             self.load, name)
        finally:
            with self._lock:
                self.loading.discard(name)
                self.last_activity = time.monotonic()

    def describe(self):
        out = {}
        for name, slot in self._slots.items():
            inst = slot.instance
            out[name] = {"spec": slot.spec, "loaded": inst is not None,
                         "stateful": None if inst is None else bool(inst.stateful),
                         "info": {} if inst is None else inst.info(),
                         "load_s": None if slot.load_s is None else round(slot.load_s, 2),
                         "calls": slot.calls,
                         "mean_ms": round(1e3 * slot.busy_s / slot.calls, 2) if slot.calls else None,
                         "sessions": sum(1 for m, _, _ in self._sessions.values() if m == name)}
        return out

    # -- sessions ----------------------------------------------------------
    def _close_session(self, sid):
        with self._lock:
            entry = self._sessions.pop(sid, None)
        if entry is not None:
            try:
                entry[2].close()
            finally:
                logger.info("[host] session %s (%s) closed", sid, entry[0])
        return entry is not None

    def close_owner(self, owner):
        """Close every session opened by ``owner`` (a connection that went away)."""
        sids = [s for s, (_, o, _) in list(self._sessions.items()) if o == owner]
        for sid in sids:
            self._close_session(sid)
        return len(sids)

    async def aclose_owner(self, owner):
        """:meth:`close_owner` with each close on its module's worker thread, so it cannot
        race an in-flight call on the same session."""
        loop = asyncio.get_running_loop()
        n = 0
        for sid, (name, o, _) in list(self._sessions.items()):
            if o == owner:
                await loop.run_in_executor(self._slot(name).get_executor(), self._close_session, sid)
                n += 1
        return n

    def module_of(self, header):
        """Module name a request touches (``None`` for control ops)."""
        op = header.get("op")
        if op not in _MODULE_OPS:
            return None
        sid = header.get("session")
        if sid:
            entry = self._sessions.get(sid)
            return entry[0] if entry else None
        return header.get("module")

    # -- dispatch ----------------------------------------------------------
    def handle(self, header, arrays=None, owner="local"):
        """Synchronous dispatch. Never raises: errors become ``ok: false`` replies."""
        t0 = time.perf_counter()
        arrays = arrays or {}
        rid = header.get("id")
        with self._lock:
            self._inflight += 1
        try:
            if self.token and header.get("token") != self.token:
                raise PermissionError("bad or missing token")
            reply, out = self._dispatch(header, arrays, owner)
            reply = {"ok": True, **reply}
        except Exception as exc:  # noqa: BLE001 - every failure must reach the client
            if isinstance(exc, (ModuleError, PermissionError, ValueError, KeyError)):
                logger.warning("[host] %s failed: %s: %s", header.get("op"), type(exc).__name__, exc)
            else:
                logger.exception("[host] %s failed", header.get("op"))
            reply, out = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}, {}
        finally:
            with self._lock:
                self._inflight -= 1
                self.last_activity = time.monotonic()
        reply["id"] = rid
        reply["server_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
        return reply, out

    async def ahandle(self, header, arrays=None, owner="local"):
        """Async dispatch for the network transports: module work runs on that module's
        worker thread, so the event loop (and health checks) never blocks on the GPU."""
        if header.get("op") == "shutdown" and not header.get("force"):
            # Give sessions of a client that just exited time to be noticed and closed.
            deadline = time.monotonic() + float(header.get("wait_s", SHUTDOWN_WAIT_S))
            while self.foreign_sessions(owner) and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
        name = self.module_of(header)
        slot = self._slots.get(name) if name else None
        if slot is None:
            return self.handle(header, arrays, owner)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(slot.get_executor(), self.handle, header, arrays, owner)

    def _dispatch(self, header, arrays, owner):
        op = header.get("op")
        if op == "ping":
            with self._lock:
                idle = 0.0 if self._inflight > 1 or self.loading else time.monotonic() - self.last_activity
            return {"version": PROTOCOL_VERSION, "modules": sorted(self._slots),
                    "loaded": sorted(n for n, s in self._slots.items() if s.instance is not None),
                    "loading": sorted(self.loading), "pid": os.getpid(),
                    "idle_timeout_s": self.idle_timeout_s, "idle_s": round(idle, 1)}, {}
        if op == "shutdown":
            others = self.foreign_sessions(owner)
            if others and not header.get("force"):
                raise ModuleError(f"busy: {others} session(s) of other clients are open; "
                                  "send force=true to stop anyway")
            if self.on_shutdown is None:
                raise ModuleError("this host is not served by a stoppable server (inproc?)")
            logger.info("[host] shutdown requested by %s (force=%s, %d foreign session(s))",
                        owner, bool(header.get("force")), others)
            self.on_shutdown(f"shutdown requested by {owner}")
            return {"shutting_down": True}, {}
        if op == "list":
            return {"modules": self.describe()}, {}
        if op == "load":
            inst = self.load(header["module"], header.get("config"))
            return {"info": inst.info(), "stateful": bool(inst.stateful)}, {}
        if op == "unload":
            self.unload(header["module"])
            return {}, {}
        if op == "open":
            name = header["module"]
            inst = self.load(name)
            sess = inst.open_session(**(header.get("params") or {}))
            sid = uuid.uuid4().hex[:12]
            with self._lock:
                self._sessions[sid] = (name, owner, sess)
            logger.info("[host] session %s opened on %s by %s params=%s", sid, name, owner,
                        header.get("params"))
            return {"session": sid}, {}
        if op == "call":
            method = header.get("method")
            args = header.get("args") or {}
            sid = header.get("session")
            if sid:
                entry = self._sessions.get(sid)
                if entry is None:
                    raise ModuleError(f"unknown or closed session {sid!r}")
                name, target = entry[0], entry[2]
            else:
                name = header.get("module")
                target = self.load(name)
            slot = self._slot(name)
            t0 = time.perf_counter()
            reply, out = target.call(method, args, arrays)
            slot.calls += 1
            slot.busy_s += time.perf_counter() - t0
            return dict(reply or {}), dict(out or {})
        if op == "close":
            return {"closed": self._close_session(header.get("session"))}, {}
        raise ModuleError(f"unknown op {op!r}")

    def shutdown(self):
        for sid in list(self._sessions):
            self._close_session(sid)
        for slot in self._slots.values():
            if slot.instance is not None:
                try:
                    slot.instance.unload()
                except Exception:  # noqa: BLE001
                    logger.exception("[host] unload of %s failed", slot.name)
            if slot.executor is not None:
                slot.executor.shutdown(wait=False)
                slot.executor = None
