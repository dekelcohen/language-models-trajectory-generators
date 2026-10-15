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
