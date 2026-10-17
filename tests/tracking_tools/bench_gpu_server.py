"""Benchmark gpu_server transports and remote CoTracker.

Part 1 - transport overhead: ``echo`` round trips of one RGB frame (default 256x256) over
every URL given (plus ``inproc://``), raw and PNG.
Part 2 - CoTracker per-frame latency through the server (``--cotracker``): a synthetic
moving-square clip, 12 points, split into buffered (non-flush) and flush frames. With
``--local`` the same clip also runs in-process (needs torch + the model in this env).

    # server (GPU env):  python -m gpu_server.server --modules cotracker echo --preload cotracker
    python tests/tracking_tools/bench_gpu_server.py --url tcp://127.0.0.1:8766 ws://127.0.0.1:8765/v1/ws \
        --cotracker --out outputs/tracking_probes/gpu_server/bench.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gpu_server.client import GpuClient  # noqa: E402


def stats(ms):
    ms = np.asarray(ms, float)
    return {"n": int(len(ms)), "mean_ms": round(float(ms.mean()), 3), "p50_ms": round(float(np.median(ms)), 3),
            "p95_ms": round(float(np.percentile(ms, 95)), 3), "max_ms": round(float(ms.max()), 3)}


def synth_clip(n, size):
    frames = []
    for k in range(n):
        img = np.full((size, size, 3), 40, np.uint8)
        img[::16] = 90                                   # texture
        x = 60 + 2 * k
        img[100:140, x:x + 40] = (200, 120, 40)
        frames.append(img)
    pts = np.array([[60 + 8 * i % 40 + 2, 102 + 12 * (i // 5)] for i in range(12)], float)
    return frames, pts


def bench_echo(url, frame, n, codec):
    with GpuClient(url, image_codec=codec) as c:
        for _ in range(5):
            c.call("echo", "echo", frame=frame)
        rtt, srv = [], []
        for _ in range(n):
            reply, _ = c.call("echo", "echo", frame=frame)
            rtt.append(c.last_rtt_ms)
            srv.append(reply["server_ms"])
    return {"url": url, "codec": codec, **stats(rtt), "server_mean_ms": round(float(np.mean(srv)), 3)}


def run_tracker(tracker, frames, pts):
    tracker.init(frames[0], pts)
    rows = []
    for f in frames[1:]:
        t0 = time.perf_counter()
        r = tracker.update(f)
        rows.append((1e3 * (time.perf_counter() - t0), bool(r.meta.get("fresh")),
                     r.meta.get("remote_server_ms")))
    return rows


def summarise(rows, label):
    fl = [ms for ms, fresh, _ in rows if fresh]
    bf = [ms for ms, fresh, _ in rows if not fresh]
    out = {"label": label, "flush": stats(fl[1:] or fl), "buffered": stats(bf)}   # first flush = warm-up
    srv = [s for _, fresh, s in rows if fresh and s is not None][1:]
    if srv:
        out["flush_server_mean_ms"] = round(float(np.mean(srv)), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", nargs="*", default=["tcp://127.0.0.1:8766", "ws://127.0.0.1:8765/v1/ws"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cotracker", action="store_true")
    ap.add_argument("--local", action="store_true", help="also run CoTracker in-process")
    ap.add_argument("--frames", type=int, default=65)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frame = (np.random.default_rng(0).integers(0, 255, (args.size, args.size, 3))).astype(np.uint8)
    res = {"frame_shape": list(frame.shape), "echo": [], "cotracker": []}
    for url in ["inproc://?modules=echo"] + list(args.url):
        for codec in (("raw",) if url.startswith("inproc") else ("raw", "png")):
            row = bench_echo(url, frame, args.n, codec)
            res["echo"].append(row)
            print(f"echo {url:32s} {codec:4s} mean {row['mean_ms']:7.3f} ms  p95 {row['p95_ms']:7.3f}  "
                  f"(server {row['server_mean_ms']:.3f})")

    if args.cotracker:
        from providers.trackers.remote_tracker import RemoteTracker
        frames, pts = synth_clip(args.frames, args.size)
        for url in args.url:
            with RemoteTracker(url=url, module="cotracker") as tr:
                row = summarise(run_tracker(tr, frames, pts), f"remote {url}")
            res["cotracker"].append(row)
            print(json.dumps(row))
        if args.local:
            from providers.trackers.cotracker_tracker import CoTrackerTracker
            row = summarise(run_tracker(CoTrackerTracker(), frames, pts), "local in-process")
            res["cotracker"].append(row)
            print(json.dumps(row))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
