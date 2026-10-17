"""Expose a gpu_server on the internet from a box with no inbound ports (Kaggle, Colab, a lab PC).

Kaggle has no API to start a notebook and open a port. The server therefore runs inside the
notebook on loopback, and a **cloudflared quick tunnel** (no account, no config) publishes it
as ``https://<random>.trycloudflare.com``. WebSockets pass through, so clients use
``wss://<random>.trycloudflare.com/v1/ws``. Raw TCP does not pass through.

The tunnel is public, so a token is mandatory. The server checks it on every request
(``--token`` / ``$GPU_SERVER_TOKEN``).

Client side (CPU-only machine)::

    $env:GPU_SERVER_URL   = "wss://<random>.trycloudflare.com/v1/ws"
    $env:GPU_SERVER_TOKEN = "<token>"
    $env:GPU_SERVER_CODEC = "jpeg"      # 14.5 KB/frame instead of 192 KB raw, same accuracy
    python main.py --tracking --tracker-provider remote --tracking-server no-launch ...

The latency is internet RTT plus the model. Calls are synchronous: measured ping is 64-148 ms;
on the D4 door eval the mean step is 0.87 s and p95 3.1 s (flush steps, two cameras). That is
fine for lock-step evals and RoMa re-detection, and too slow for real-time tracking.

CLI (what ``gpu_server/kaggle_gpu_server.ipynb`` runs)::

    python -m gpu_server.expose --modules cotracker echo --preload cotracker --hours 11.5
"""

import argparse
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request

_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
_ASSETS = {
    ("Linux", "x86_64"): "cloudflared-linux-amd64",
    ("Linux", "aarch64"): "cloudflared-linux-arm64",
    ("Windows", "AMD64"): "cloudflared-windows-amd64.exe",
}
_URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def ensure_cloudflared(dest_dir=None):
    """Path to a cloudflared binary: on PATH, already downloaded, or downloaded now."""
    found = shutil.which("cloudflared")
    if found:
        return found
    asset = _ASSETS.get((platform.system(), platform.machine()))
    if asset is None:
        raise RuntimeError(f"no cloudflared asset for {platform.system()}/{platform.machine()}; "
                           "install it and put it on PATH")
    dest_dir = dest_dir or os.path.join(os.path.expanduser("~"), ".cache", "gpu_server")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, asset)
    if not os.path.isfile(path):
        print(f"[expose] downloading {asset} ...", flush=True)
        urllib.request.urlretrieve(_RELEASE + asset, path)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


class Tunnel:
    """A running ``cloudflared tunnel --url`` process and its public URL."""

    def __init__(self, port, binary=None, log_path=None, timeout=90.0):
        self.port = int(port)
        self.binary = binary or ensure_cloudflared()
        self.log_path = log_path
        self.timeout = float(timeout)
        self.proc = None
        self.public_url = None
        self.lines = []

    @property
    def ws_url(self):
        return None if self.public_url is None else \
            self.public_url.replace("https://", "wss://", 1) + "/v1/ws"

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        cmd = [self.binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        found = threading.Event()
        # The pipe must be drained for the tunnel's whole life, or cloudflared blocks on it.
        threading.Thread(target=self._drain, args=(found,), daemon=True).start()
        if not found.wait(self.timeout) or self.public_url is None:
            self.stop()
            raise RuntimeError("cloudflared did not report a trycloudflare.com URL:\n"
                               + "".join(self.lines[-20:]))
        return self

    def _drain(self, found):
        log = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        try:
            for line in self.proc.stdout:
                self.lines = (self.lines + [line])[-200:]
                if log:
                    log.write(line)
                    log.flush()
                m = _URL_RE.search(line)
                if m and self.public_url is None:
                    self.public_url = m.group(0)
                    found.set()
        finally:
            found.set()
            if log:
                log.close()

    def stop(self):
        if self.alive:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_server(modules, preload=(), port=8765, token=None, sets=(), idle_timeout_s=0.0,
                 log_path=None, python=None, ready_timeout=900.0):
    """Start ``python -m gpu_server.server`` on loopback; return the Popen once ``preload`` is loaded."""
    from gpu_server.client import GpuClient

    cmd = [python or sys.executable, "-m", "gpu_server.server", "--modules", *modules,
           "--ws", f"127.0.0.1:{port}", "--tcp", f"127.0.0.1:{port + 1}",
           "--idle-timeout-s", str(float(idle_timeout_s))]
    if preload:
        cmd += ["--preload", *preload]
    for s in sets:
        cmd += ["--set", s]
    env = dict(os.environ)
    if token:
        env["GPU_SERVER_TOKEN"] = token
    out = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    url = f"ws://127.0.0.1:{port}/v1/ws"
    t0 = time.monotonic()
    while time.monotonic() - t0 < ready_timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"gpu_server exited with code {proc.returncode} (log: {log_path})")
        try:
            with GpuClient(url, timeout=5, token=token) as c:
                reply = c.ping()
            if all(m in (reply.get("loaded") or []) for m in preload):
                print(f"[expose] gpu_server ready on {url} after {time.monotonic() - t0:.0f} s "
                      f"(loaded {reply.get('loaded')})", flush=True)
                return proc
        except Exception:  # noqa: BLE001 - still binding / loading
            pass
        time.sleep(2.0)
    proc.terminate()
    raise RuntimeError(f"gpu_server not ready after {ready_timeout:.0f} s (log: {log_path})")


def check_public(ws_url, token, n=5, codec="jpeg", timeout_s=120.0):
    """Ping the server through the tunnel (plus an image round trip via the ``echo`` module
    when offered); returns ``(ping_reply, median_rtt_ms)``.

    A fresh trycloudflare hostname can take up to ~1 min to resolve in DNS (and some
    resolvers cache the early NXDOMAIN), so this retries for ``timeout_s``."""
    import numpy as np

    from gpu_server.client import GpuClient

    last = None
    deadline = time.time() + timeout_s
    attempt = 0
    while attempt == 0 or time.time() < deadline:
        attempt += 1
        try:
            with GpuClient(ws_url, timeout=20, token=token, image_codec=codec) as c:
                reply = c.ping()
                rtts = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    c.ping()
                    rtts.append((time.perf_counter() - t0) * 1e3)
                if "echo" in (reply.get("modules") or []):
                    img = (np.indices((256, 256)).sum(0) % 256).astype(np.uint8)
                    img = np.repeat(img[..., None], 3, axis=2)
                    t0 = time.perf_counter()
                    _r, back = c.call("echo", "echo", frame=img)
                    mae = float(np.abs(back["frame"].astype(int) - img).mean())
                    print(f"[expose] {codec} 256x256 frame round trip {(time.perf_counter() - t0) * 1e3:.0f} ms, "
                          f"pixel MAE {mae:.2f}", flush=True)
            return reply, sorted(rtts)[len(rtts) // 2]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3.0)
    raise RuntimeError(f"tunnel {ws_url} does not answer: {last}")


def client_instructions(ws_url, token, show_token=True):
    tok = token if show_token else "<your GPU_SERVER_TOKEN secret>"
    return "\n".join([
        "# PowerShell (client, CPU-only machine):",
        f'$env:GPU_SERVER_URL   = "{ws_url}"',
        f'$env:GPU_SERVER_TOKEN = "{tok}"',
        '$env:GPU_SERVER_CODEC  = "jpeg"',
        "python main.py --tracking --tracker-provider remote --tracking-server no-launch ...",
        "# bash:",
        f"export GPU_SERVER_URL={ws_url} GPU_SERVER_TOKEN={tok} GPU_SERVER_CODEC=jpeg",
    ])


def keep_alive(server, tunnel, token, hours, every_s=60.0, on_new_url=None):
    """Watch the server and the tunnel until ``hours`` pass. A dead tunnel is restarted (new URL)."""
    from gpu_server.client import GpuClient

    t_end = time.monotonic() + hours * 3600.0
    while time.monotonic() < t_end:
        if server is not None and server.poll() is not None:
            print(f"[expose] gpu_server exited (code {server.returncode}); stopping", flush=True)
            break
        if not tunnel.alive:
            print("[expose] tunnel died; restarting (the URL changes)", flush=True)
            tunnel.public_url = None
            tunnel.start()
            if on_new_url:
                on_new_url(tunnel.ws_url)
        try:
            with GpuClient(f"ws://127.0.0.1:{tunnel.port}/v1/ws", timeout=10, token=token) as c:
                r = c.ping()
            print(f"[expose] {time.strftime('%H:%M:%S')} ok | {tunnel.ws_url} | loaded "
                  f"{r.get('loaded')} | sessions {r.get('sessions')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[expose] local ping failed: {exc}", flush=True)
        time.sleep(every_s)
    tunnel.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modules", nargs="+", default=["cotracker", "echo"])
    ap.add_argument("--preload", nargs="*", default=["cotracker"])
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.KEY=VALUE")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("GPU_SERVER_TOKEN"),
                    help="shared secret (default $GPU_SERVER_TOKEN, else a generated one)")
    ap.add_argument("--no-server", action="store_true", help="only tunnel an already running server")
    ap.add_argument("--hours", type=float, default=11.5, help="keep-alive duration (Kaggle max 12 h)")
    ap.add_argument("--log-dir", default=".")
    args = ap.parse_args(argv)

    token = args.token or secrets.token_urlsafe(18)
    os.makedirs(args.log_dir, exist_ok=True)
    server = None
    if not args.no_server:
        server = start_server(args.modules, args.preload, args.port, token, args.set,
                              log_path=os.path.join(args.log_dir, "gpu_server.log"))
    tunnel = None
    try:
        tunnel = Tunnel(args.port, log_path=os.path.join(args.log_dir, "cloudflared.log")).start()
        try:
            reply, rtt = check_public(tunnel.ws_url, token)
            print(f"[expose] public {tunnel.ws_url} answers (median ping {rtt:.0f} ms, loaded "
                  f"{reply.get('loaded')})", flush=True)
        except RuntimeError as exc:     # DNS of a new quick-tunnel host can lag on this machine only
            print(f"[expose] WARNING public self-check failed ({exc}); the URL usually works "
                  f"from the client a minute later", flush=True)
        print(client_instructions(tunnel.ws_url, token, show_token=not args.token), flush=True)
        keep_alive(server, tunnel, token, args.hours)
    finally:
        if tunnel is not None:
            tunnel.stop()
        if server is not None:
            server.terminate()


if __name__ == "__main__":
    main()
