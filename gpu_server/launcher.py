"""Start gpu_server on demand and decide what happens to it when the run ends.

A tracking run with ``--tracker-provider remote`` needs a server at ``config.tracker_remote_url``.
When that URL is on this machine and nothing answers, :func:`ensure_server` starts
``python -m gpu_server.server`` as a *detached* process (its own process group, output in
``outputs/gpu_server/server_<port>.log``), preloading the requested module, and waits until it
is loaded. The server outlives the run so the next run skips the model load; it exits by
itself after ``config.gpu_server_idle_timeout_s`` (30 min) without requests.

Policy (``--tracking-server``, ``$GPU_SERVER_POLICY``, ``config.tracking_server_policy_default``):

    no-shutdown  launch if needed, leave it running at exit (default)
    shutdown     launch if needed; at exit close our sessions and ask the server to stop
                 (it refuses while sessions of other clients are open)
    no-launch    never start a server; fail loudly if it is not running

Remote URLs (another machine) are never launched or stopped from here, only pinged.

Interpreter: ``$GPU_SERVER_PYTHON`` > ``config.gpu_server_python`` > the conda env named
``config.gpu_server_conda_env`` next to the current env > ``sys.executable``.

Typical use::

    from gpu_server import launcher
    launcher.configure("shutdown")                      # once, from the CLI
    launcher.prewarm(url, "cotracker")                  # non-blocking, at startup
    launcher.ensure_server(url, "cotracker")            # blocking, before the first session
"""

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
import weakref
from urllib.parse import urlparse

logger = logging.getLogger("gpu_server.launcher")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLICIES = ("no-shutdown", "shutdown", "no-launch")
LOOPBACK = ("127.0.0.1", "localhost", "::1")

_lock = threading.RLock()
_policy = None
_servers = {}            # url -> ServerHandle (every server this process launched or used)
_clients = weakref.WeakSet()   # objects with close(), closed before a policy shutdown
_atexit_registered = False


def _cfg(name, default):
    try:
        import config
    except ImportError:  # pragma: no cover - gpu_server used outside the repo
        return default
    return getattr(config, name, default)


class ServerHandle:
    def __init__(self, url, module, proc=None, log_path=None, launched=False):
        self.url = url
        self.module = module
        self.proc = proc
        self.log_path = log_path
        self.launched = launched
        self.pid = proc.pid if proc is not None else None
        self.ready = False

    def __repr__(self):
        return (f"ServerHandle({self.url}, module={self.module}, pid={self.pid}, "
                f"launched={self.launched}, ready={self.ready})")


# -- policy ----------------------------------------------------------------
def configure(policy=None):
    """Set the process-wide policy (``None`` keeps env/config). Exported to
    ``$GPU_SERVER_POLICY`` so child processes (the sim env process) inherit it."""
    global _policy
    if policy is None:
        return get_policy()
    if policy not in POLICIES:
        raise ValueError(f"tracking-server policy must be one of {POLICIES}, got {policy!r}")
    with _lock:
        _policy = policy
        os.environ["GPU_SERVER_POLICY"] = policy
        if policy == "shutdown":
            _register_atexit()
    return policy


def get_policy():
    p = _policy or os.environ.get("GPU_SERVER_POLICY") or _cfg("tracking_server_policy_default",
                                                              "no-shutdown")
    if p not in POLICIES:
        raise ValueError(f"tracking-server policy must be one of {POLICIES}, got {p!r}")
    return p


def register_client(obj):
    """Objects (``RemoteTracker``) to close before a policy shutdown, so their own sessions
    do not make the server refuse."""
    with _lock:
        _clients.add(obj)


# -- probing ----------------------------------------------------------------
def is_local(url):
    u = urlparse(url)
    return u.scheme in ("tcp", "ws") and (u.hostname or "127.0.0.1") in LOOPBACK


def ping(url, timeout=2.0):
    """The server's ping reply, or ``None`` when nothing answers."""
    from gpu_server.client import GpuClient
    try:
        with GpuClient(url, timeout=timeout) as c:
            return c.ping()
    except Exception:  # noqa: BLE001 - refused, timeout, handshake, ...
        return None


def _addresses(url):
    """``--ws``/``--tcp`` for a server reachable at ``url``: the URL's own transport on its
    port, the other on the neighbouring port (ws 8765 <-> tcp 8766, the CLI defaults)."""
    u = urlparse(url)
    h = u.hostname or "127.0.0.1"
    if u.scheme == "tcp":
        p = u.port or 8766
        return f"{h}:{p - 1}", f"{h}:{p}"
    p = u.port or 8765
    return f"{h}:{p}", f"{h}:{p + 1}"


def _find_conda_env(name):
    roots = []
    prefix = os.path.abspath(sys.prefix)
    if os.path.basename(os.path.dirname(prefix)).lower() == "envs":
        roots.append(os.path.dirname(prefix))             # .../envs/<current>
    roots.append(os.path.join(prefix, "envs"))            # base env
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(os.path.join(os.path.dirname(os.path.dirname(conda_exe)), "envs"))
    for root in roots:
        for cand in (os.path.join(root, name, "python.exe"), os.path.join(root, name, "bin", "python")):
            if os.path.isfile(cand):
                return cand
    return None


def module_profile(module):
    """Launch profile of ``module``: ``config.gpu_server_module_profiles[module]`` overlaid
    with ``$GPU_SERVER_<MODULE>_PYTHON`` / ``_PYTHONPATH`` / ``_ARGS`` (e.g. RoMa v2 needs
    torch >= 2.4 and so its own env)."""
    prof = dict((_cfg("gpu_server_module_profiles", {}) or {}).get(module) or {})
    key = f"GPU_SERVER_{str(module).upper()}_"
    if os.environ.get(key + "PYTHON"):
        prof["python"] = os.environ[key + "PYTHON"]
    if os.environ.get(key + "PYTHONPATH"):
        prof.setdefault("env", {})
        prof["env"] = dict(prof["env"], PYTHONPATH=os.environ[key + "PYTHONPATH"])
    if os.environ.get(key + "ARGS"):
        import shlex
        prof["args"] = list(prof.get("args") or []) + shlex.split(os.environ[key + "ARGS"], posix=False)
    return prof


def resolve_python(module=None):
    prof = module_profile(module) if module else {}
    if prof.get("python"):
        return prof["python"]
    if prof.get("conda_env"):
        found = _find_conda_env(prof["conda_env"])
        if found:
            return found
        logger.warning("[launcher] conda env %r of module %s not found; using the default "
                       "gpu_server python", prof["conda_env"], module)
    exe = os.environ.get("GPU_SERVER_PYTHON") or _cfg("gpu_server_python", None)
    if exe:
        return exe
    name = _cfg("gpu_server_conda_env", None)
    if name:
        found = _find_conda_env(name)
        if found:
            return found
    return sys.executable


# -- launching --------------------------------------------------------------
def _spawn(url, module, idle_timeout_s):
    ws, tcp = _addresses(url)
    prof = module_profile(module)
    python = resolve_python(module)
    cmd = [python, "-u", "-m", "gpu_server.server", "--ws", ws, "--tcp", tcp,
           "--preload", module, "--idle-timeout-s", str(float(idle_timeout_s))]
    cmd += [str(a) for a in _cfg("gpu_server_extra_args", ())]
    cmd += [os.path.expandvars(str(a)) for a in (prof.get("args") or ())]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update({k: os.path.expandvars(str(v)) for k, v in (prof.get("env") or {}).items()})
    log_dir = _cfg("gpu_server_log_dir", os.path.join("outputs", "gpu_server"))
    log_dir = log_dir if os.path.isabs(log_dir) else os.path.join(REPO_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"server_{urlparse(url).port or 'default'}.log")
    log = open(log_path, "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launched by pid {os.getpid()}: "
              f"{' '.join(cmd)}\n")
    log.flush()
    kw = dict(cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
              close_fds=True, env=env)
    try:
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            try:   # leave the caller's job object too, so closing its console does not kill it
                proc = subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
            except OSError:
                proc = subprocess.Popen(cmd, creationflags=flags, **kw)
        else:
            proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    finally:
        log.close()
    logger.info("[launcher] started gpu_server pid %d for %s (module %s, python %s, idle %.0f s); log %s",
                proc.pid, url, module, python, idle_timeout_s, log_path)
    return ServerHandle(url, module, proc, log_path, launched=True)


def _log_tail(path, n=25):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _start_or_attach(url, module, policy, idle_timeout_s):
    reply = ping(url)
    handle = _servers.get(url)
    if reply is not None:
        if handle is None:
            handle = _servers[url] = ServerHandle(url, module)
            logger.info("[launcher] gpu_server already running at %s (pid %s, loaded %s, loading %s)",
                        url, reply.get("pid"), reply.get("loaded"), reply.get("loading"))
        handle.pid = handle.pid or reply.get("pid")
        return handle, reply
    if handle is not None and handle.proc is not None and handle.proc.poll() is None:
        return handle, None                       # ours, still binding
    if policy == "no-launch" or not is_local(url):
        from gpu_server.client import START_HINT
        why = "--tracking-server no-launch" if policy == "no-launch" else "the URL is not on this machine"
        raise ConnectionError(f"no gpu_server at {url} and not launching one ({why}); {START_HINT}")
    handle = _servers[url] = _spawn(url, module, idle_timeout_s)
    return handle, None


def prewarm(url=None, module=None):
    """Non-blocking :func:`ensure_server`: start the server (if needed) so the model loads
    while the run does other work. Errors are logged, not raised."""
    try:
        return ensure_server(url, module, wait=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[launcher] prewarm of %s failed: %s", url, exc)
        return None


def ensure_server(url=None, module=None, wait=True, timeout=None, idle_timeout_s=None):
    """Make sure a gpu_server answers at ``url`` with ``module`` loaded (launching one per
    the policy). Returns a :class:`ServerHandle`. ``inproc://`` URLs need no server."""
    url = url or os.environ.get("GPU_SERVER_URL") or _cfg("tracker_remote_url", "ws://127.0.0.1:8765/v1/ws")
    module = module or os.environ.get("GPU_SERVER_MODULE") or _cfg("tracker_remote_module", "cotracker")
    if urlparse(url).scheme == "inproc":
        return ServerHandle(url, module)
    policy = get_policy()
    idle = _cfg("gpu_server_idle_timeout_s", 1800.0) if idle_timeout_s is None else idle_timeout_s
    with _lock:
        _register_atexit()
        handle, reply = _start_or_attach(url, module, policy, idle)
    if not wait:
        return handle
    timeout = float(timeout or _cfg("gpu_server_startup_timeout_s", 300.0))
    t0 = time.monotonic()
    loaded_once = False
    while True:
        if reply is not None:
            if module in (reply.get("loaded") or []):
                if not handle.ready:
                    logger.info("[launcher] gpu_server at %s ready with %s (pid %s) after %.1f s", url,
                                module, reply.get("pid"), time.monotonic() - t0)
                handle.ready = True
                return handle
            if module not in (reply.get("modules") or []):
                raise ConnectionError(f"gpu_server at {url} does not offer module {module!r} "
                                      f"(offers {reply.get('modules')})")
            if module not in (reply.get("loading") or []) and not loaded_once:
                # Running but the module is not loaded/loading (another client's server):
                # opening a session loads it on demand, so ask now.
                loaded_once = True
                from gpu_server.client import GpuClient
                try:
                    with GpuClient(url, timeout=timeout) as c:
                        c.load(module)
                except Exception as exc:  # noqa: BLE001
                    raise ConnectionError(f"gpu_server at {url} could not load {module!r}: {exc}") from exc
                reply = ping(url)
                continue
        proc = handle.proc
        if proc is not None and proc.poll() is not None:
            reply = ping(url)                 # lost a bind race to another launcher?
            if reply is not None:
                handle.proc, handle.launched = None, False
                handle.pid = reply.get("pid")
                continue
            raise ConnectionError(f"gpu_server (pid {proc.pid}) exited with code {proc.returncode} "
                                  f"while starting; last log lines ({handle.log_path}):\n"
                                  f"{_log_tail(handle.log_path)}")
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"gpu_server at {url} not ready with {module!r} after {timeout:.0f} s "
                               f"(last ping {reply}); see {handle.log_path}")
        time.sleep(0.5)
        reply = ping(url)


# -- shutdown ---------------------------------------------------------------
def shutdown_server(url, force=False, wait_s=10.0, timeout=30.0):
    """Ask the server at ``url`` to stop. Returns ``True`` when it stopped."""
    from gpu_server.client import GpuClient, GpuServerError
    handle = _servers.get(url)
    try:
        with GpuClient(url, timeout=timeout) as c:
            c.request({"op": "shutdown", "force": bool(force), "wait_s": float(wait_s)})
    except GpuServerError as exc:
        logger.warning("[launcher] gpu_server at %s kept running: %s", url, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        if ping(url) is None:
            return True
        logger.warning("[launcher] shutdown request to %s failed: %s", url, exc)
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gone = ping(url, timeout=1.0) is None
        if handle is not None and handle.proc is not None:
            gone = gone and handle.proc.poll() is not None
        if gone:
            logger.info("[launcher] gpu_server at %s stopped", url)
            return True
        time.sleep(0.2)
    logger.warning("[launcher] gpu_server at %s still alive %.0f s after shutdown", url, timeout)
    return False


def _close_clients():
    with _lock:
        clients = list(_clients)
        _clients.clear()
    for obj in clients:
        try:
            obj.close()
        except Exception:  # noqa: BLE001
            pass


def _at_exit():
    if get_policy() != "shutdown":
        for h in list(_servers.values()):
            if h.launched:
                logger.info("[launcher] leaving gpu_server pid %s at %s running "
                            "(idle exit after %.0f s; --tracking-server shutdown stops it)",
                            h.pid, h.url, _cfg("gpu_server_idle_timeout_s", 1800.0))
        return
    _close_clients()
    for url in list(_servers):
        if is_local(url):
            shutdown_server(url)


def _register_atexit():
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_at_exit)
        _atexit_registered = True


def at_exit():
    """Apply the policy now (for processes whose ``atexit`` does not run, e.g.
    ``multiprocessing`` children). Idempotent."""
    _at_exit()
    with _lock:
        if get_policy() == "shutdown":
            _servers.clear()
