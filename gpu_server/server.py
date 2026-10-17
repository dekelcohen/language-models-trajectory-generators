"""Generic model server: ``python -m gpu_server.server --help``.

Two transports in one process, same binary protocol (:mod:`gpu_server.protocol`) and the
same :class:`gpu_server.host.ModuleHost`:

* ``--ws HOST:PORT`` - FastAPI/uvicorn. WebSocket ``/v1/ws`` (streaming, stateful sessions)
  plus REST: ``GET /health``, ``GET /v1/modules``, ``POST /v1/modules/{name}/load|unload``,
  ``POST /v1/call/{module}/{method}`` (stateless; JSON body for array-free calls, or a
  packed binary body ``application/octet-stream``). Use this across machines, through
  proxies/tunnels, or from notebooks. Needs ``fastapi uvicorn websockets``.
* ``--tcp HOST:PORT`` - stdlib asyncio TCP, 8-byte length-prefixed messages. No extra
  dependencies; the fast path for a client on the same machine (another conda env).

Examples::

    # GPU env (CoTracker, torch 2.0.1)
    python -m gpu_server.server --modules cotracker echo --preload cotracker
    # RoMa env (torch >= 2.4), other ports
    python -m gpu_server.server --modules romav2 --ws 127.0.0.1:8775 --tcp 127.0.0.1:8776 \
        --set romav2.root=D:/Vision/RoMaV2 --set romav2.setting=turbo
    # third-party / future model (e.g. a local VLM), loaded by dotted path
    python -m gpu_server.server --module qwen=my_pkg.qwen_module:QwenModule --preload qwen

Binding to a non-loopback address requires ``--token`` (or ``$GPU_SERVER_TOKEN``).

Lifecycle: the ports are bound first and ``--preload`` modules load in the background (``ping``
lists them under ``loading``). The server stops on Ctrl+C, on a ``shutdown`` request
(:func:`gpu_server.launcher.shutdown_server`), when a preload fails (exit code 1), or after
``--idle-timeout-s`` seconds without requests. Tracking runs start it on demand through
:mod:`gpu_server.launcher`.
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpu_server.host import BUILTIN_MODULES, ModuleHost  # noqa: E402
from gpu_server.protocol import PROTOCOL_VERSION, ProtocolError, pack, unpack  # noqa: E402

logger = logging.getLogger("gpu_server")

_LEN = struct.Struct(">Q")
MAX_MESSAGE = 512 * 2**20
DEFAULT_WS = "127.0.0.1:8765"
DEFAULT_TCP = "127.0.0.1:8766"


# -- TCP -------------------------------------------------------------------
async def _read_msg(reader):
    (n,) = _LEN.unpack(await reader.readexactly(_LEN.size))
    if n > MAX_MESSAGE:
        raise ProtocolError(f"message of {n} bytes exceeds {MAX_MESSAGE}")
    return await reader.readexactly(n)


def _write_msg(writer, payload):
    writer.write(_LEN.pack(len(payload)) + payload)


async def _tcp_client(host, reader, writer):
    peer = writer.get_extra_info("peername")
    owner = f"tcp:{peer[0]}:{peer[1]}" if peer else f"tcp:{id(writer)}"
    sock = writer.get_extra_info("socket")
    if sock is not None:
        import socket
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    logger.info("[tcp] %s connected", owner)
    try:
        while True:
            try:
                raw = await _read_msg(reader)
            except asyncio.IncompleteReadError:
                break
            try:
                header, arrays = unpack(raw)
            except ProtocolError as exc:
                _write_msg(writer, pack({"ok": False, "error": str(exc), "error_type": "ProtocolError"}))
                await writer.drain()
                continue
            reply, out = await host.ahandle(header, arrays, owner)
            _write_msg(writer, pack(reply, out))
            await writer.drain()
    except (ConnectionError, OSError) as exc:
        logger.info("[tcp] %s dropped: %s", owner, exc)
    finally:
        n = await host.aclose_owner(owner)
        logger.info("[tcp] %s disconnected (%d session(s) closed)", owner, n)
        writer.close()


async def serve_tcp(host, addr):
    h, p = _split(addr)
    server = await asyncio.start_server(lambda r, w: _tcp_client(host, r, w), h, p)
    logger.info("[tcp] listening on tcp://%s:%d", h, p)
    return server


# -- FastAPI ---------------------------------------------------------------
def create_app(host):
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="gpu_server", version=str(PROTOCOL_VERSION),
                  description="Generic model server (CoTracker, RoMa v2, ...). Binary protocol: "
                              "gpu_server/protocol.py")

    def _check(request_token):
        if host.token and request_token != host.token:
            raise HTTPException(status_code=401, detail="bad or missing token")

    def _token(request):
        return request.headers.get("x-gpu-token") or request.query_params.get("token")

    @app.get("/health")
    def health():
        return host.handle({"op": "ping"})[0]

    @app.get("/v1/modules")
    def modules(request: Request):
        _check(_token(request))
        return host.describe()

    @app.post("/v1/modules/{name}/load")
    async def load(name: str, request: Request):
        _check(_token(request))
        body = await request.body()
        cfg = json.loads(body) if body else None
        reply, _ = await host.ahandle({"op": "load", "module": name, "config": cfg,
                                       "token": host.token})
        return JSONResponse(reply, status_code=200 if reply["ok"] else 400)

    @app.post("/v1/modules/{name}/unload")
    async def unload(name: str, request: Request):
        _check(_token(request))
        reply, _ = await host.ahandle({"op": "unload", "module": name, "token": host.token})
        return JSONResponse(reply, status_code=200 if reply["ok"] else 400)

    @app.post("/v1/call/{name}/{method}")
    async def call(name: str, method: str, request: Request):
        _check(_token(request))
        body = await request.body()
        binary = request.headers.get("content-type", "").startswith("application/octet-stream")
        if binary:
            header, arrays = unpack(body)
            args = header.get("args") or {}
        else:
            args, arrays = (json.loads(body) if body else {}), {}
        reply, out = await host.ahandle({"op": "call", "module": name, "method": method,
                                         "args": args, "token": host.token}, arrays, "http")
        if binary or out:
            return Response(pack(reply, out), media_type="application/octet-stream",
                            status_code=200 if reply["ok"] else 400)
        return JSONResponse(reply, status_code=200 if reply["ok"] else 400)

    @app.websocket("/v1/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        c = websocket.client
        owner = f"ws:{c.host}:{c.port}" if c else f"ws:{id(websocket)}"
        logger.info("[ws] %s connected", owner)
        try:
            while True:
                raw = await websocket.receive_bytes()
                try:
                    header, arrays = unpack(raw)
                except ProtocolError as exc:
                    await websocket.send_bytes(pack({"ok": False, "error": str(exc),
                                                     "error_type": "ProtocolError"}))
                    continue
                reply, out = await host.ahandle(header, arrays, owner)
                await websocket.send_bytes(pack(reply, out))
        except WebSocketDisconnect:
            pass
        finally:
            n = await host.aclose_owner(owner)
            logger.info("[ws] %s disconnected (%d session(s) closed)", owner, n)

    return app


def _split(addr):
    h, _, p = str(addr).rpartition(":")
    return (h or "127.0.0.1"), int(p)


def _is_loopback(addr):
    return _split(addr)[0] in ("127.0.0.1", "localhost", "::1")


def _uvicorn_config(host, addr):
    import uvicorn
    h, p = _split(addr)
    return uvicorn.Config(create_app(host), host=h, port=p, log_level="warning",
                          ws="websockets", ws_max_size=MAX_MESSAGE, ws_per_message_deflate=False,
                          ws_ping_interval=None)


async def serve(host, ws_addr=DEFAULT_WS, tcp_addr=DEFAULT_TCP, preload=(), idle_timeout_s=0.0,
                on_ready=None):
    """Serve until Ctrl+C, a ``shutdown`` request, a failed preload or ``idle_timeout_s``
    seconds without requests (0 = never). Returns the stop reason.

    The ports are bound *before* the models load, so a second launcher finds the server
    (``ping`` lists the modules still ``loading``) instead of starting a duplicate.
    ``on_ready(ws_url, tcp_url)`` is called once the transports listen.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    state = {"reason": None}

    def request_stop(reason, delay=0.2):
        # Thread-safe; the delay lets the reply to a 'shutdown' request reach the client.
        def _set():
            state["reason"] = state["reason"] or reason
            stop.set()
        loop.call_soon_threadsafe(loop.call_later, delay, _set)

    host.on_shutdown = request_stop
    host.idle_timeout_s = float(idle_timeout_s or 0.0)
    tcp_server = await serve_tcp(host, tcp_addr) if tcp_addr else None
    uv = uv_task = None
    if ws_addr:
        try:
            cfg = _uvicorn_config(host, ws_addr)
        except ImportError as exc:
            if not tcp_server:
                raise
            logger.warning("[ws] disabled (%s); serving TCP only. pip install fastapi uvicorn websockets", exc)
        else:
            import uvicorn
            uv = uvicorn.Server(cfg)
            uv_task = asyncio.ensure_future(uv.serve())
            while not uv.started and not uv_task.done():
                await asyncio.sleep(0.02)
            if uv_task.done():           # bind failure: uvicorn exits
                if tcp_server is not None:
                    tcp_server.close()
                uv_task.result()
                raise OSError(f"uvicorn could not serve {ws_addr}")
            h, p = _split(ws_addr)
            logger.info("[ws] listening on ws://%s:%d/v1/ws  (REST: http://%s:%d/docs)", h, p, h, p)
    if on_ready is not None:
        on_ready("ws://%s:%d/v1/ws" % _split(ws_addr) if uv else None,
                 "tcp://%s:%d" % _split(tcp_addr) if tcp_server else None)

    def _preload_done(name, task):
        if not task.cancelled() and task.exception() is not None:
            logger.error("[server] preload of %s failed: %s", name, task.exception())
            request_stop(f"preload of {name} failed: {task.exception()}", delay=0.0)
    for name in preload or ():
        t = asyncio.ensure_future(host.apreload(name))
        t.add_done_callback(lambda task, n=name: _preload_done(n, task))

    async def idle_watchdog():
        period = min(30.0, max(0.05, host.idle_timeout_s / 10.0))
        while not stop.is_set():
            await asyncio.sleep(period)
            idle = host.idle_s()
            if idle >= host.idle_timeout_s:
                logger.info("[server] idle for %.0f s (limit %.0f s): shutting down", idle,
                            host.idle_timeout_s)
                request_stop(f"idle for {idle:.0f} s", delay=0.0)
                return
    if host.idle_timeout_s > 0:
        logger.info("[server] will exit after %.0f s without requests", host.idle_timeout_s)
        asyncio.ensure_future(idle_watchdog())

    waiters = [asyncio.ensure_future(stop.wait())] + ([uv_task] if uv_task else [])
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)   # uvicorn returns on Ctrl+C
    finally:
        state["reason"] = state["reason"] or "interrupted"
        logger.info("[server] stopping: %s", state["reason"])
        if uv is not None:
            uv.should_exit = True
            if not uv_task.done():
                try:
                    await asyncio.wait_for(uv_task, 10)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    pass
        if tcp_server is not None:
            tcp_server.close()
        host.shutdown()
    return state["reason"]


class BackgroundServer:
    """Run :func:`serve` on a daemon thread (tests, benchmarks, notebooks)::

        srv = BackgroundServer(ModuleHost(["echo"]), ws="127.0.0.1:0", tcp="127.0.0.1:0")
        GpuClient(srv.tcp_url) ...; srv.stop()

    Port ``0`` picks a free port; the actual URLs are ``srv.ws_url`` / ``srv.tcp_url``.
    ``srv.reason`` is the stop reason once the server has exited (``srv.wait(timeout)``).
    """

    def __init__(self, host, ws=None, tcp=None, timeout=20.0, preload=(), idle_timeout_s=0.0):
        import threading

        self.host = host
        self.ws_url = self.tcp_url = None
        self.reason = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._error = None
        ws = _free_port_addr(ws) if ws else None
        tcp = _free_port_addr(tcp) if tcp else None

        def on_ready(ws_url, tcp_url):
            self.ws_url, self.tcp_url = ws_url, tcp_url
            self._ready.set()

        def run():
            try:
                self.reason = asyncio.run(serve(host, ws, tcp, preload, idle_timeout_s, on_ready))
            except BaseException as exc:  # noqa: BLE001 - includes uvicorn's SystemExit
                self._error = exc
            finally:
                self._ready.set()
                self._done.set()

        self._thread = threading.Thread(target=run, daemon=True, name="gpu-server")
        self._thread.start()
        self._ready.wait(timeout)
        if self._error:
            raise RuntimeError(f"gpu_server failed to start: {self._error!r}") from self._error
        if not (self.ws_url or self.tcp_url):
            raise TimeoutError("gpu_server did not start")

    def wait(self, timeout=None):
        """Block until the server exits; returns ``True`` if it did."""
        return self._done.wait(timeout)

    @property
    def running(self):
        return not self._done.is_set()

    def stop(self):
        if self.running and self.host.on_shutdown is not None:
            self.host.on_shutdown("stopped by BackgroundServer.stop", 0.0)
        self._thread.join(15)


def _free_port_addr(addr):
    import socket
    h, p = _split(addr)
    if p == 0:
        with socket.socket() as s:
            s.bind((h, 0))
            p = s.getsockname()[1]
    return f"{h}:{p}"


def _parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_host(args):
    modules = {m: BUILTIN_MODULES[m] for m in (args.modules or [])}
    for item in args.module or []:
        name, _, spec = item.partition("=")
        if not spec:
            raise SystemExit(f"--module expects name=pkg.mod:Class, got {item!r}")
        modules[name] = spec
    if not modules:
        modules = dict(BUILTIN_MODULES)
    configs = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        mod, _, k = key.partition(".")
        if not k:
            raise SystemExit(f"--set expects module.key=value, got {item!r}")
        configs.setdefault(mod, {})[k] = _parse_value(value)
    return ModuleHost(modules, configs, token=args.token)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", nargs="*", choices=sorted(BUILTIN_MODULES),
                    help="builtin modules to offer (default: all builtins, unless --module is given)")
    ap.add_argument("--module", action="append", metavar="NAME=PKG.MOD:CLASS",
                    help="register an extra module by dotted path (repeatable)")
    ap.add_argument("--set", action="append", metavar="MODULE.KEY=VALUE",
                    help="module config (value parsed as JSON when possible), e.g. cotracker.device=cuda:0")
    ap.add_argument("--preload", nargs="*", default=[], help="load these modules before serving ('all' = every one)")
    ap.add_argument("--ws", default=DEFAULT_WS, help="FastAPI WS/REST address ('' disables)")
    ap.add_argument("--tcp", default=DEFAULT_TCP, help="raw TCP address ('' disables)")
    ap.add_argument("--token", default=os.environ.get("GPU_SERVER_TOKEN"), help="shared secret")
    ap.add_argument("--idle-timeout-s", type=float, default=0.0,
                    help="exit after this many seconds without a request (0 = never; the tracking "
                         "auto-launcher passes config.gpu_server_idle_timeout_s = 1800)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for addr in (args.ws, args.tcp):
        if addr and not _is_loopback(addr) and not args.token:
            raise SystemExit(f"refusing to bind {addr} without --token / $GPU_SERVER_TOKEN")
    host = build_host(args)
    preload = list(host._slots) if args.preload == ["all"] else args.preload
    logger.info("[server] pid %d, modules %s, preload %s", os.getpid(), sorted(host._slots), preload)
    try:
        reason = asyncio.run(serve(host, args.ws, args.tcp, preload, args.idle_timeout_s))
    except KeyboardInterrupt:
        reason = "interrupted"
    logger.info("[server] exited: %s", reason)
    return 1 if reason and reason.startswith("preload") else 0


if __name__ == "__main__":
    sys.exit(main())
