# gpu_server - generic GPU model server

One process holds the GPU models. CPU-only clients, or clients in another conda env, call
them over TCP or WebSocket. Today it serves CoTracker3 (stateful, online), RoMa v2 (stateless
matching) and MolmoPoint (stateless VLM pointing, see [MolmoPoint](#molmopoint-pointing)).

```
client (vlm_traj, CPU)                     server (env_lapa_tracking, GPU)
RemoteTracker / GpuClient  --tcp|ws-->  ModuleHost --> cotracker | romav2 | echo | <your module>
                                        (one worker thread per module, lazy load)
```

## Run

```powershell
$py = "C:\Users\dekelco\AppData\Local\miniconda3\envs\env_lapa_tracking\python.exe"
# Weights cache: $env:TORCH_HOME, default ~/.cache/torch (outside the repo).
& $py -m gpu_server.server --modules cotracker --preload cotracker      # ws :8765, tcp :8766
& $py -m gpu_server.server --modules romav2 --set romav2.root=<RoMaV2 checkout> --set romav2.setting=turbo --preload romav2 --ws 127.0.0.1:8775 --tcp 127.0.0.1:8776
```

| flag | meaning |
|---|---|
| `--modules a b` | built-in modules to expose (`cotracker`, `romav2`, `molmo_point`, `echo`) |
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
shared memory was not worth adding. Benchmark: `tests/tracking_tools/bench_gpu_server.py`.

**Image codec** (client `image_codec=`, or `$GPU_SERVER_CODEC` for `RemoteTracker`). Use it only
over slow links. Door clip, head cam, 15 CoTracker windows through the server:

| codec | bytes / 256x256 frame | encode | CoTracker error median / p95 (px) |
|---|---|---|---|
| `raw` (default) | 192 KB | 0.4 ms | 3.76 / 5.65 |
| `png` (lossless) | 26 KB | 58-93 ms | identical to raw |
| `jpeg` (q=90) | 14.5 KB | 2.6 ms | 3.36 / 6.74 (equivalent) |

`jpeg` applies only to uint8 gray/RGB images. RGBA and other images fall back to `png`, and
non-image arrays are always raw.

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
  `config.gpu_server_conda_env` (`env_lapa_tracking`) > the current interpreter, only if it has
  CUDA torch itself. Otherwise the launcher raises `GpuServerEnvError` pointing at the install
  steps (README "3D Tracking" step 2).
- **Per-module profiles** (`config.gpu_server_module_profiles`): a module that needs a
  different env gets its own server, spawned with that profile's python, args and env.
  This is how CoTracker (py3.10/torch 2.0, port 8765) and RoMa v2 (torch ≥ 2.4, port 8775,
  `config.track_roma_url`) run side by side. Default:
  `{"romav2": {"conda_env": "vlm_roma", "args": ["--set", "romav2.setting=turbo"]}}`.
  The environment overrides it:
  - `GPU_SERVER_<MODULE>_PYTHON`: interpreter;
  - `GPU_SERVER_<MODULE>_PYTHONPATH`: the server's `PYTHONPATH`;
  - `GPU_SERVER_<MODULE>_ARGS`: extra server args, e.g. `--set romav2.root=D:\RoMaV2`.

  `ROMAV2_ROOT` / `TORCH_HOME` are read by the `romav2` module (or use `--set romav2.root=`,
  `--set romav2.torch_home=`).
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

## Kaggle / Colab: expose the server to a CPU-only client (`gpu_server/expose.py`)

Kaggle has no API to start a notebook or to open a port. So the notebook itself runs the
server on loopback, and a **cloudflared quick tunnel** publishes it. The tunnel needs no
account and no inbound port: `wss://<random>.trycloudflare.com/v1/ws`. WebSockets pass
through; raw TCP does not.

1. Open [`kaggle_gpu_server.ipynb`](kaggle_gpu_server.ipynb) on Kaggle and set it up:
   - Accelerator: **GPU T4**;
   - Internet: **On**;
   - Secrets: `GPU_SERVER_TOKEN` (recommended), plus `GITHUB_TOKEN` if the repo is private.

   Then run all cells. To keep it running with the browser closed, use *Save Version → Save &
   Run All*, which runs for up to 12 h. The URL is printed in the log and written to
   `/kaggle/working/gpu_server_url.txt`.
2. On the client:
   ```powershell
   $env:GPU_SERVER_URL   = "wss://<random>.trycloudflare.com/v1/ws"
   $env:GPU_SERVER_TOKEN = "<same token>"
   $env:GPU_SERVER_CODEC = "jpeg"
   python main.py --tracking --tracker-provider remote --tracking-server no-launch ...
   ```
   The launcher never spawns for a non-local URL. Do not use `--tracking-server shutdown`
   against a shared remote server.

The same flow as a CLI, on any Linux or Windows GPU box:
`python -m gpu_server.expose --modules cotracker echo --preload cotracker --hours 11.5`.
`--no-server --port P` tunnels an already-running server instead. The CLI does the following:
- downloads `cloudflared` into `~/.cache/gpu_server` if it is not on PATH;
- checks the public URL (ping, plus a JPEG frame round trip through `echo`);
- prints the client lines;
- restarts a dead tunnel (the URL changes).

**Measured** (tunnel from this laptop through Cloudflare and back, D4 door eval, `jpeg`):
- ping 64-148 ms;
- D4 accuracy is the same as local: median 0.027 m, p95 0.072 m;
- mean step latency 0.87 s, p95 3.1 s on flush steps (two cameras, sequential).

That is fine for lock-step runs and RoMa re-detection. It is too slow for real-time tracking.

Session owners are per connection (`ws:<peer>#<n>`). Behind a tunnel every connection
reports the same peer (`<ip>:0`). Before this fix, the launcher's ping disconnect closed the
tracker's CoTracker sessions (97% lost).

## Linux GPU box on the LAN (Docker, `gpu_server/docker/`)

Runs CoTracker3 and RoMa v2 on a Linux machine with an NVIDIA GPU (tested target: RTX 3070
8 GB). The Windows client connects over the LAN. Both models fit in 8 GB together; RoMa v2 runs
in bf16 on Ampere (`amp=auto`).

**Files:**
- `Dockerfile`: base `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime`; clones RoMaV2 into
  `/opt/RoMaV2` (sources only, no `pip install -e`); copies `config.py`, `providers/`, `gpu_server/`.
- `requirements.txt`: server deps only (fastapi, uvicorn, websockets, opencv-headless, einops, ...;
  plus `transformers==4.57.1`, `accelerate`, `bitsandbytes`, `decord2` for MolmoPoint).
- `docker-compose.yml`: three services from one image, on the client's default ports:
  - `cotracker`: ws `:8765`, tcp `:8766` (default);
  - `romav2`: ws `:8775`, tcp `:8776`. It is opt-in through the `roma` profile, because client RoMa
    re-detection is off by default (`config.track_roma_enable = False`).
  - `molmo`: ws `:8785`, tcp `:8786`, opt-in through the `molmo` profile (MolmoPoint pointing).
- `docker-compose.dev.yml` + `dev-entrypoint.sh`: optional dev overlay (live code, env + caches on
  host dirs), see step 4.
- Root `.dockerignore`: limits the build context to those files.

Weights download on first load into the `model-cache` volume at `/cache`
(`TORCH_HOME=/cache/torch`, `HF_HOME=/cache/huggingface`). Restarts reuse it.

**1. Host prerequisites (once, needs sudo).** Ubuntu 22.04/24.04:
```bash
nvidia-smi                       # any driver >= 525 (e.g. 535 / "CUDA 12.2") is enough, see note below
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER    # log out/in afterwards
# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi   # GPU visible in containers
```
Driver note: `nvidia-smi`'s "CUDA Version" is the newest CUDA the driver supports natively. The
image's torch (cu126) still runs on a 525+ driver through CUDA minor-version compatibility. torch
ships its own CUDA runtime and sm_86 kernels, so no PTX JIT is needed. The image sets
`NVIDIA_DISABLE_REQUIRE=1` so the container runtime does not reject a "12.2" driver. Upgrade the
driver (`sudo ubuntu-drivers install`, 550+) only if `torch.cuda.is_available()` is False in the
container.

**2. Build and run** (from the repo root on the Linux box):
```bash
git clone https://github.com/dekelcohen/language-models-trajectory-generators.git && cd language-models-trajectory-generators
git checkout tactile
echo "GPU_SERVER_TOKEN=$(openssl rand -hex 16)" > gpu_server/docker/.env   # keep this secret
docker compose -f gpu_server/docker/docker-compose.yml up -d --build    # CoTracker only (default)
docker compose -f gpu_server/docker/docker-compose.yml logs -f            # wait for "[host] module cotracker ready"
```
- Add RoMa v2 (for `--track-roma`): `docker compose -f gpu_server/docker/docker-compose.yml --profile roma up -d`.
  Pass `--profile roma` to every later `ps` / `logs` / `down` as well, or set `COMPOSE_PROFILES=roma` in
  `gpu_server/docker/.env`.
- Update: `git pull`, then rerun `up -d --build`.
- Stop: `docker compose -f gpu_server/docker/docker-compose.yml down` (the weights volume stays).
- Firewall: `sudo ufw allow from <client-ip> to any port 8765:8786 proto tcp`.
  Keep these ports off the internet; the token is the only auth.

**3. Client (Windows).** `tcp://` (stdlib) has the least overhead on a LAN. `ws://<ip>:8765/v1/ws`
also works (`websockets` is in `requirements.txt`) and is the only option through tunnels/proxies.
```powershell
$env:GPU_SERVER_URL      = "tcp://<linux-ip>:8766"
$env:GPU_SERVER_ROMA_URL = "tcp://<linux-ip>:8776"                   # only with the roma profile
$env:GPU_SERVER_TOKEN    = "<token from gpu_server/docker/.env>"
python -c "from gpu_server.client import GpuClient; print(GpuClient('tcp://<linux-ip>:8766').ping())"
python main.py --tracking --tracker-provider remote --tracking-server no-launch ...
```
- On a LAN, `raw` frames are fine.
- Over Wi-Fi or slow links, set `$env:GPU_SERVER_CODEC = "jpeg"`.
- Never pass `--tracking-server shutdown` against this shared server.

**4. Dev mode (optional): python env + model caches on host dirs.** Add the dev overlay:
```bash
C="-f gpu_server/docker/docker-compose.yml -f gpu_server/docker/docker-compose.dev.yml"
docker compose $C up -d --build                       # servers run the LIVE repo code (/app bind)
docker compose $C --profile shell up -d shell         # + an idle GPU dev container
docker compose $C exec shell bash                     # pip install ..., python ..., nvidia-smi
docker compose $C restart cotracker                   # pick up code changes
```
- `~/vlm-gpu-dev/env` -> `/workspace/env`: the python env.
  - `DEV_ENV_KIND=venv` (default): a `--system-site-packages` venv over the image env. The image's
    torch is reused; your `pip install`s (e.g. `transformers==4.57.1`) persist in this dir.
  - `DEV_ENV_KIND=conda`: a full `conda create --clone base` (~6 GB), independent of the image.
- `~/vlm-gpu-dev/cache` -> `/cache`: `TORCH_HOME` + `HF_HOME` (transformers / hub models).
- Change the dirs with `DEV_WORKSPACE`, `DEV_CACHE`, `DEV_ENV_KIND` in `gpu_server/docker/.env`.
- The env is seeded on the first start, under a lock (both services start together).
  To reset it, delete the dir. Files are root-owned; `sudo chown -R $USER ~/vlm-gpu-dev` if needed.

**VRAM on an 8 GB card (RTX 3070):**

| Model | Weights | Peak at inference |
|---|---|---|
| CoTracker3 (online) | ~0.1 GB | ~1-2 GB (grid/queries) |
| RoMa v2 (bf16) | ~0.5 GB | ~1.5-2.5 GB |
| MolmoPoint-8B, bf16 | ~8.7 B params = ~17.5 GB | 18.3 GiB (measured, erdi28 card); large GPUs only |
| MolmoPoint-8B, 8-bit | ~10 GB | does not fit |
| MolmoPoint-8B, 4-bit (nf4 LLM, bf16 vision) | 6.6 GiB | 8.7 GiB (measured, erdi28 card); does not fit 8 GB |
| MolmoPoint-Vid-4B, bf16 | 4.86 B params = 9.7 GB | does not fit |
| MolmoPoint-Vid-4B, 8-bit | ~5 GB | ~6.5-7.5 GB (+ video KV cache); alone only |
| MolmoPoint-Vid-4B, 4-bit (nf4) | ~3 GB | ~4-5 GB; fits next to CoTracker |

Video KV cache: Qwen3-4B-like LLM, ~0.15 MB/token in bf16; about 80 tokens/frame, so 32 frames
is ~0.4 GB and 128 frames is ~1.5 GB. Quantized loading of the remote-code model is verified only
by the erdi28 pre-quantized 8B checkpoint (same nf4 recipe as `molmo_point`'s `4bit`); `8bit` and
the 4B in nf4 are unverified. Vid-4B numbers are estimates.

**Without Docker** (conda/venv on the box):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && pip install -r gpu_server/docker/requirements.txt`,
clone RoMaV2, then run the same `python -m gpu_server.server ...` commands as in
`docker-compose.yml` with `--token`. Optional on Linux: `pip install fused-local-corr`
(faster RoMa local correlation).

## MolmoPoint (pointing)

`molmo_point` module (`gpu_server/modules/molmo_point.py`): Ai2's MolmoPoint VLM points at a
described object and returns **pixel** points. Used for wrist-camera pointing
(`api.locate_with_wrist_camera`) instead of a cloud VLM.

- **Method `point`**: input `image` (HxWx3 uint8 RGB); args `description` (prompt becomes
  `"Point to <description>."`) or a verbatim `prompt`, `max_points`, `max_new_tokens`. Reply:
  `points` `[[u, v], ...]` in input pixels (model order), `raw` text, `latency_s`, `model`, `quant`.
- **Method `point_video`** (minimal, unverified): `frames` (T,H,W,3) -> `points` `[[t, u, v], ...]`.
- MolmoPoint emits grounding tokens, not text coordinates; `model.extract_image_points` decodes them
  into source-image pixels. Text formats (Molmo 1 `<point x y>` percent, Molmo 2 `coords="..."` x1000)
  are a fallback parser (`parse_point_text`).
- **Config** (`--set molmo_point.<k>=...`, or env):

| key | env | default |
|---|---|---|
| `model` | `MOLMO_MODEL` | auto by VRAM (below); or any of `allenai/MolmoPoint-8B`, `allenai/MolmoPoint-Vid-4B`, `erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16` |
| `quant` | `MOLMO_QUANT` | auto: 8B -> `bf16` at >= 20 GB else `4bit`; 4B -> `bf16` at >= 16 GB else `4bit`. Also `8bit`, or `none` = as stored (pre-quantized repos) |
| `max_new_tokens` | - | 200 |

`4bit` = bitsandbytes nf4 + bf16 compute + double quant on the language model; vision tower,
connector, patch embedding and pointing head stay bf16 (the erdi28 recipe: quantizing the
vision path hurt pointing). First load downloads the weights (8B: F32 safetensors, ~35 GB; 4B: ~20 GB)
into `HF_HOME`.

**Auto mode** (empty `MOLMO_MODEL` / `MOLMO_QUANT`, the compose default), chosen from the GPU's
total VRAM and logged at load (`[molmo_point] mode: 7.6 GiB VRAM -> auto model ..., quant 4bit`;
also in `list` -> `info.mode`):

| total VRAM | model | quant | peak |
|---|---|---|---|
| < 12 GB (RTX 3070 8 GB) | `MolmoPoint-Vid-4B` | `4bit` (nf4 LLM, bf16 vision) | ~4-5 GB, fits next to CoTracker |
| 12-20 GB | `MolmoPoint-8B` | `4bit` | 8.7 GiB |
| >= 20 GB | `MolmoPoint-8B` | `bf16` | 18.3 GiB; best pointing (PointBench 70.9%) |

- The 8B does not fit 8 GB even in nf4 (8.7 GiB peak), hence the 4B there. `4bit` rather than
  `8bit` for the 4B: 8-bit would be ~6.5-7.5 GB, too close to 8 GB to be a safe default.
- Vid-4B is trained for **video** pointing only; its single-image quality is unknown, so
  benchmark it before relying on it.
- Override with `MOLMO_MODEL` / `MOLMO_QUANT` in `gpu_server/docker/.env`, e.g.
  `MOLMO_MODEL=erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16` + `MOLMO_QUANT=none` (8B nf4, ~7 GB download).

**Run** (Linux box, Docker, see below for setup):
```bash
D="docker compose -f gpu_server/docker/docker-compose.yml"
$D --profile molmo up -d --build molmo
$D --profile molmo logs -f molmo     # "[molmo_point] mode: ..." then "module molmo_point ready"
```
Without Docker: `python -m gpu_server.server --modules molmo_point echo --preload molmo_point
--ws 0.0.0.0:8785 --tcp 0.0.0.0:8786 --token <t>`.

**Client** (Windows): `$env:GPU_SERVER_MOLMO_URL = "tcp://<linux-ip>:8786"` (default
`tcp://127.0.0.1:8786`), `$env:GPU_SERVER_TOKEN = "<token>"`, optional `$env:GPU_SERVER_CODEC = "jpeg"`.
Python: `providers.pointing.molmo_remote.point_with_molmo(image, description, max_points=4, url=None, token=None)`
-> `[[u, v], ...]` pixels (image = path, PIL image or RGB array). No local auto-launch.

Smoke test:
```powershell
python -c "from gpu_server.client import GpuClient; print(GpuClient('tcp://<linux-ip>:8786').ping())"
python -c "from providers.pointing.molmo_remote import point_with_molmo as p; print(p('images/rgb_image_wrist.png', 'the USB port', return_reply=True))"
```

**From `main.py`:** `--wrist-pointing-model gpu-molmo` switches only `locate_with_wrist_camera`
to MolmoPoint (default: `--planner-perception-vlm`). `gpu-molmo:<url>` picks a server explicitly,
e.g. `gpu-molmo:tcp://10.0.0.5:8786`. The rest of the function (depth lift, held-object filter)
is unchanged.

**Benchmark:** `python research/touch/pointing_bench.py --models gpu-molmo ...` (compares it with
the cloud VLMs on the same wrist images).

## Security

Bind to loopback by default. A non-loopback bind requires `--token`, and clients pass
`token=` or set `$GPU_SERVER_TOKEN`. The launcher spawns only loopback servers. A tunnel is
public, so `expose` always sets a token (generated if not given).

## Tests

`tests/test_gpu_server.py` covers:
- the transports (TCP and WS);
- remote-vs-local CoTracker parity;
- the lifecycle: busy/forced shutdown, idle exit, bind-before-preload;
- the launcher with real detached subprocesses: launch, attach, policies, idle exit, no-launch.

`tests/test_molmo_point.py` (CPU, no download): Molmo output parsers, `point` with a fake
model/processor, the client glue.
