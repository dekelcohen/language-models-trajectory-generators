# gpu_server - generic GPU model server

One process holds the GPU models. CPU-only clients, or clients in another conda env, call
them over TCP or WebSocket. Today it serves CoTracker3 (stateful, online) and RoMa v2
(stateless matching). Local VLMs (Qwen/Molmo) can be added later as further modules.

```
client (vlm_traj, CPU)                     server (env_lapa_tracking, GPU)
RemoteTracker / GpuClient  --tcp|ws-->  ModuleHost --> cotracker | romav2 | echo | <your module>
                                        (one worker thread per module, lazy load)
```

## Run

```powershell
$py = "C:\Users\dekelco\AppData\Local\miniconda3\envs\env_lapa_tracking\python.exe"
$env:TORCH_HOME = "$PWD\cache\torch"
& $py -m gpu_server.server --modules cotracker --preload cotracker      # ws :8765, tcp :8766
& $py -m gpu_server.server --modules romav2 --set romav2.root=<RoMaV2 checkout> --set romav2.setting=turbo --preload romav2 --ws 127.0.0.1:8775 --tcp 127.0.0.1:8776
```

| flag | meaning |
|---|---|
| `--modules a b` | built-in modules to expose (`cotracker`, `romav2`, `echo`) |
| `--module name=pkg.mod:Class` | register any other module class (repeatable) |
| `--set mod.key=value` | module config; the value is parsed as JSON when possible |
| `--preload a b` / `all` | load in the background *after* the ports are bound |
| `--ws host:port` / `--tcp host:port` | listen addresses; `''` disables one |
| `--token` / `$GPU_SERVER_TOKEN` | shared secret; required for non-loopback binds |
| `--idle-timeout-s N` | exit after N s without any request (0 = never; the default) |

WS needs `fastapi` and `uvicorn`. Without them the server serves TCP only and logs a warning.

## URLs and transports

| URL | round trip, 256x256x3 frame | use |
|---|---|---|
| `inproc://?modules=cotracker` | 0.009 ms | same process, same env (no server) |
| `tcp://127.0.0.1:8766` | 1.09 ms | local or LAN, fastest wire |
| `ws://127.0.0.1:8765/v1/ws` | 1.61 ms | default; proxy- and firewall-friendly |

All of these are negligible next to model time (CoTracker flush 0.2-1.2 s, RoMa 0.1-5 s), so
shared memory was not worth adding. `image_codec="png"` only helps on slow links.
Benchmark: `tests/tracking_tools/bench_gpu_server.py`.

## Auto-launch and lifecycle (`gpu_server/launcher.py`)

When tracking uses `--provider remote`, the client starts the server itself:

1. `agent_runner.py` and `tests/tools/run_tracking_eval.py` call `launcher.prewarm()` at
   startup. The model loads while the scene boots.
2. `RemoteTracker` calls `launcher.ensure_server(url, module)` before opening its session.
   - It pings first. A running server is reused, and a missing module is `load`ed into it.
   - Only if the server is down **and** the URL is loopback does it spawn
     `python -m gpu_server.server --preload <module> --idle-timeout-s 1800` as a detached
     process.
   - It then waits until the module is loaded.
   - A remote host is never spawned, only pinged.
3. At exit, the `--tracking-server` policy decides what happens:

| `--tracking-server` | at exit | notes |
|---|---|---|
| `no-shutdown` (default) | leave the server running | the next run attaches instantly with a warm model |
| `shutdown` | stop the server | refused or waited on while *another* client still holds a session |
| `no-launch` | - | never spawn; fail loudly if no server is running |

- **Idle exit:** a launched server exits by itself after `config.gpu_server_idle_timeout_s`
  (1800 s) with no requests. Any request counts as activity, including a ping. In-flight
  requests and loads block the exit; open idle connections do not.
- **Python:** `$GPU_SERVER_PYTHON` > `config.gpu_server_python` > the conda env
  `config.gpu_server_conda_env` (`env_lapa_tracking`) > the current interpreter.
- **Ports:** a `ws://...:p` URL binds ws `p` + tcp `p+1`; a `tcp://...:p` URL binds ws
  `p-1` + tcp `p`. Bind-first plus background preload means a second launcher finds the
  server already listening instead of starting a duplicate.
- **Log:** `outputs/gpu_server/server_<port>.log` (appended).
- **Manual stop:** `python -c "from gpu_server import launcher; launcher.shutdown_server('ws://127.0.0.1:8765/v1/ws')"`,
  or send `{"op": "shutdown", "force": true}`.
- **Other config:** `gpu_server_startup_timeout_s` (300) and `gpu_server_extra_args`.

## Protocol (ops)

The wire format is `GS1` magic + JSON header + raw arrays (`protocol.py`).

| op | does |
|---|---|
| `ping` | `loaded`, `loading`, `pid`, `idle_timeout_s`, `idle_s`, per-module stats |
| `list` | available modules and their info |
| `load` / `unload` | `module`, optional config |
| `call` | `module` + `method` (stateless), or `session` + `method` (stateful); args + arrays |
| `open` / `close` | stateful sessions, owned by the connection and closed when it drops |
| `shutdown` | `force`, `wait_s` (default 10): stops the server; `busy` error while foreign sessions stay open |

Python client:

```python
from gpu_server.client import GpuClient
with GpuClient("tcp://127.0.0.1:8776") as c:
    r, arrs = c.call("romav2", "match", img_a=a, img_b=b, points_a=pts, box=[x0, y0, x1, y1], dense=True)
    s = c.open("cotracker", vis_threshold=0.5)
    s.call("init", frame=f0, points=pts); s.call("update", frame=f1)
```

## Adding a module (e.g. a local VLM)

```python
# gpu_server/modules/qwen_vl.py
from gpu_server.modules.base import GpuModule, ModuleError, require

class QwenVLModule(GpuModule):
    name = "qwen_vl"
    stateful = False                     # True + open_session() for per-client state

    def load(self):                      # runs once on this module's own worker thread
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.config.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"))
        self.model = AutoModelForVision2Seq.from_pretrained(..., device_map="cuda")

    def m_point(self, args, arrays):     # callable as method "point"
        (img,) = require(arrays, "image")
        ...
        return {"points": [[x, y]]}, {}
```

Then add it to `BUILTIN_MODULES` in `host.py`, or pass
`--module qwen_vl=gpu_server.modules.qwen_vl:QwenVLModule`.

- Modules never need to be thread-safe: each one gets a single worker thread.
- Different modules run concurrently.
- Keep heavy imports inside `load()`.

## Security

Bind to loopback by default. A non-loopback bind requires `--token`, and clients pass
`token=` or set `$GPU_SERVER_TOKEN`. The launcher spawns only loopback servers.

## Tests

`tests/test_gpu_server.py` covers:
- the transports (TCP and WS);
- remote-vs-local CoTracker parity;
- the lifecycle: busy/forced shutdown, idle exit, bind-before-preload;
- the launcher with real detached subprocesses: launch, attach, policies, idle exit, no-launch.
