# Tracking probe tools (3D tracking diagnostics)

Offline diagnostics for `tracking/` + `tests/tracking_eval.py` (docs: `docs/plans/3d_tracking.md` §13).
Outputs go to `outputs/tracking_probes/`. `TRK_ROOT` = repo/worktree to import from (default: this repo).

**Never run experiments from the tree you are editing** - the eval spawns subprocesses that import live
files. Use worktrees:

```powershell
git worktree add ..\lmtg_base HEAD          # clean baseline
git worktree add ..\lmtg_exp HEAD           # experiments
.\tests\tracking_tools\sync_exp.ps1         # copy uncommitted changes -> ..\lmtg_exp
```

| Tool | What |
|---|---|
| `run_matrix2.py OUT [ROOT] [-- D4 G5 ...]` | Scenario matrix (D0-D5v door, G0-G6 grasp), parallel (`JOBS`, default 4). env `CFG="key=val ..."` config overrides, `EXTRA="--seeding region --seed-points 12"` extra CLI args |
| `drift_probe.py <tracking_eval args>` | Per-frame trace: phase err, mode, per-cam status / px error. `--every N`, `--trace`, `--detail lo hi` (full lift meta), `--cfg k=v`. env `KFDBG=1`: Kalman state, latency, true velocity |
| `slide_probe.py <args>` | Pre-occlusion slide: tracked vs true seed distance to the handle origin, 2D err direction, depth bias / lateral 3D err, `rotation_observable` |
| `show_seeds.py --scene door\|grasp --seed-points N` | Render `region_seed` points on the seed camera image |
| `dump_d4_bank.py` | D4: re-acquisition banks (NCC vs truth), search candidates, patch crops |
| `dump_door_clip.py` | Door clip -> npz: clean / occluded RGB, depth, handle mask, per-seed GT pixel + visibility (no tracking) |
| `cotracker_alone.py` | GPU env: CoTracker3 online / online+grid / offline on the clip, px error vs GT, on-handle rate |
| `render_clip_tracks.py` | MP4 of a `cotracker_alone` run (GT rings, tracked dots, centroids, x6 zoom) |
| `redetect_probe.py` | Re-detection from a pre-occlusion context crop: NCC (scale x angle), SIFT / ORB / AKAZE; `--detail` = top-5 NCC vs score at the truth + similarity-map videos |
| `export_sim_pairs.py <clip> [--queries ...] [--upload]` | Sim pair PNGs (pre-occlusion frame + crop, post-occlusion frames) + `manifest.json` + paste-able `PAIRS`; `--upload` -> imgbb (`IMGBB_API_KEY` env, never stored) |
| `romav2_redetect_colab.ipynb` | RoMa v2 on Colab T4 / Kaggle 2xT4 (also runs on CPU, torch >= 2.4): any image pair (URL / path / upload), box or crop re-detection, warp/overlap/match visualisations (handle + door-context match lines), settings benchmark |
| `render_romav2_redetect.py manifest --url tcp://127.0.0.1:8776 [--modes fullA_box crop_only]` | RoMa v2 re-detection through a running `gpu_server` (`romav2` module): 5-panel mp4/png per mode into `outputs/tracking_video/` + `results_gpu_server.json` (§13.12 legend, §13.13) |
| `bench_gpu_server.py [--url ...] [--cotracker] [--local]` | `gpu_server` transport round trip (inproc / tcp / ws, raw / png) and remote CoTracker per-frame cost |

`results/`: matrix outputs (`m_base` = HEAD before these fixes, `m_region12_v2`, `ab_*` = A/B 2 reps base vs
region seeding) and seed renders.

## Status / next steps (2026-10-15)

- Done: multi-bank re-acquisition (seed bank + 3 newest, deduped/capped), edge-safe banking, faster search
  (union windows, <= 8 spread points), `region_seed` (`--seeding region`, opt-in).
- A/B: door D0/D1 ~0.031 -> ~0.015 m median, D4 0.096 -> 0.060 (still never recovers post-occlusion);
  grasp G5/G6 p95 slightly worse (0.012-0.019 -> 0.020-0.032) - search stalls the real-time loop.
- Pre-existing failure: `test_tracking_door_pybullet.py::test_optional_shoulder_camera_joins_through_the_pose`
  (0.050 vs 0.04), identical on the old HEAD.
- Next: (1) parent-body (door leaf) tracker -> predict handle via T_door * T_handle(latch angle), tight-window
  verify; (2) fix d: template anchor drift correction (Matthews 2004) + subpixel NCC in
  `providers/trackers/template_tracker.py`; (3) coarse masked context template for re-detection.
  Research: `docs/plans/research_global_context_tracking.md`.

## Status (2026-10-17) - see `docs/plans/3d_tracking.md` §13.11, §13.12

- CoTracker3 alone drifts too on the textureless handle: 0.8 px before the occlusion, 5-6 px per point
  once the lever has turned (centroid ~2 px), but stays on the handle and passes the occlusion (no black).
- NCC / SIFT / ORB / AKAZE re-detection from a 30-140 px context crop fails: truth NCC 0.36-0.54 <
  false positives 0.56-0.70 (wood edges, the twin lever on the other side of the door).
- RoMa v2 cannot run on the A1000 env (torch 2.0.1), but it runs on CPU (env `stmd`, torch 2.12) and,
  via the notebook, on Colab/Kaggle. It **re-detects the handle**: 15/15 post-occlusion frames within 6 px
  (median 2.2 px) with a direct dense-warp lookup, versus NCC top-1 at 18-20 px median (§13.12).
- Next (§13.12): CoTracker3 as the in-pipeline anchor (no reset on occlusion, step 4, latency-compensated
  measurement, optional remote GPU server); RoMa v2 as a remote re-detector; handle ICP only if needed.
