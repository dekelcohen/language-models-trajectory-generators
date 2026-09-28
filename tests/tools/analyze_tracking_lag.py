"""Decompose tracking error into *lag* (behind the motion) vs *offset* (beside it).

Reads eval/video JSON payloads (``frames[obj][i]`` with ``predicted`` / ``expected`` / ``t``)
and, per frame, projects ``predicted - expected`` onto the ground-truth velocity direction:

* ``along``  < 0 means the estimate trails the object; ``lag_s = -along / speed``
* ``perp``   the part of the error the motion does not explain (bias, drift, depth)

It also fits the single time shift ``d`` that best aligns the predicted trajectory with the
ground truth (``pred(t) ~ gt(t - d)``); ``d`` ≈ the estimate age means the error is just the
publish latency, a larger ``d`` means the tracker itself falls behind (e.g. template matches
snapping back, Kalman smoothing, or points sliding off the object).

    python tests\\tools\\analyze_tracking_lag.py outputs\\tracking_video\\grasp_*_noocc_rt30.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def _series(frames):
    rows = [f for f in frames if f.get("expected") is not None]
    t = np.array([f.get("t") if f.get("t") is not None else f["frame"] for f in rows], float)
    gt = np.array([f["expected"] for f in rows], float)
    pred = np.array([f["predicted"] if f.get("predicted") is not None else [np.nan] * 3
                     for f in rows], float)
    age = np.array([f.get("age_s") or 0.0 for f in rows], float)
    moving = np.array([not f.get("occluded") and not f.get("blind_cams") for f in rows], bool)
    return t, gt, pred, age, moving


def best_shift(t, gt, pred, shifts):
    """Time shift ``d`` minimising median |pred(t) - gt(t - d)| (gt linearly interpolated)."""
    ok = np.all(np.isfinite(pred), axis=1)
    best = (None, np.inf)
    for d in shifts:
        ts = t[ok] - d
        inside = (ts >= t[0]) & (ts <= t[-1])
        if inside.sum() < 5:
            continue
        g = np.stack([np.interp(ts[inside], t, gt[:, k]) for k in range(3)], axis=1)
        err = float(np.median(np.linalg.norm(pred[ok][inside] - g, axis=1)))
        if err < best[1]:
            best = (float(d), err)
    return best


def analyze(path, obj=None):
    payload = json.load(open(path))
    frames = payload["frames"]
    obj = obj or next(iter(frames))
    t, gt, pred, age, moving = _series(frames[obj])
    vel = np.gradient(gt, t, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    err = pred - gt
    ok = np.all(np.isfinite(pred), axis=1) & moving & (speed > 1e-4)
    u = vel[ok] / speed[ok, None]
    along = np.sum(err[ok] * u, axis=1)
    perp = np.linalg.norm(err[ok] - along[:, None] * u, axis=1)
    lag_s = -along / speed[ok]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    d, d_err = best_shift(t, gt, pred, np.arange(-5, 30) * dt)
    cfg = payload.get("config", {})
    return {
        "file": os.path.basename(path),
        "label": cfg.get("label"),
        "n": int(ok.sum()),
        "speed_mps": float(np.median(speed[ok])) if ok.any() else None,
        "err_med_m": float(np.median(np.linalg.norm(err[ok], axis=1))) if ok.any() else None,
        "along_med_m": float(np.median(along)) if ok.any() else None,
        "perp_med_m": float(np.median(perp)) if ok.any() else None,
        "lag_med_s": float(np.median(lag_s)) if ok.any() else None,
        "age_med_s": float(np.median(age[ok])) if ok.any() else None,
        "best_shift_s": d,
        "err_at_shift_m": d_err,
        "bias_m": np.round(np.median(err[ok], axis=0), 4).tolist() if ok.any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="eval JSON payloads (globs allowed)")
    parser.add_argument("--obj", default=None)
    args = parser.parse_args()
    paths = sorted({p for pat in args.paths for p in glob.glob(pat)})
    cols = ["label", "n", "speed_mps", "err_med_m", "along_med_m", "perp_med_m",
            "lag_med_s", "age_med_s", "best_shift_s", "err_at_shift_m", "bias_m"]
    print(" | ".join(cols) + " | file")
    for path in paths:
        r = analyze(path, args.obj)
        cells = [(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols]
        print(" | ".join(cells) + " | " + r["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
