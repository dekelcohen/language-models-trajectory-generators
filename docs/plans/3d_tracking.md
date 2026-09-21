# 3D Tracking for Robotic Manipulation

Status of this document: **the current pipeline is implemented but only partially tested;
LAPA has been brought up and audited but not yet integrated.** Every claim below is marked
as measured, source-verified, or untested. Nothing here is aspirational unless it sits
under a "TODO" heading.

---

## 0. Status board and how to resume

This section exists so the work can be put down and picked up cold. §7 is the detailed TODO
list; this is the orientation.

### 0.1 Where things stand

| Area | State | Evidence |
|---|---|---|
| OpenGL<->OpenCV convention bridge | **done** | `tracking/camera_convert.py`, 24 tests |
| 3D provider seam + `depth_fusion` + `triangulate` | **done** | `providers/tracker3d/`, 24 tests |
| `rigid_refine` provider | **done** | `providers/tracker3d/rigid_refine.py`, 24 tests (§2.6) |
| Ground-truth eval harness + baseline numbers | **done** | `tests/tracking_eval.py`, 25 tests (§2.5) |
| CoTracker 2D provider | **done**, with a caveat | `providers/trackers/cotracker_tracker.py` (§7.2.1) |
| `tracking/motion.py` | **done** | 62 tests (§6) |
| LAPA env, checkpoint, VRAM/latency | **done** | §3 |
| LAPA constraint audit | **done** | §5 - found two disabling bugs |
| LAPA DLT weight fix | **patch prepared, deliberately not applied** | §5.2.2, §7.4 item 1 |
| LAPA adapter (`lapa_tracker3d.py`) | **not started** | §7.3 |
| Door affordance scene | in flight | §7.1 |
| Motion gates/outputs/monitors | in flight | §7.4 |
| Remote WebSocket server | **not started** | §7.5 |

### 0.2 Environment (the part that is easy to lose)

- **Main env `vlm_traj`** - Python 3.11, torch 2.1.1+**cpu**. Runs the sim, the tests, and
  everything except LAPA/DINOv2.
  `C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe`
- **GPU env `env_lapa_tracking`** - Python 3.10, torch **2.0.1+cu117**, numpy 1.26.4,
  opencv-python-headless 4.10.0.84, transformers 4.44.2, einops, timm.
  `C:\Users\dekelco\AppData\Local\miniconda3\envs\env_lapa_tracking\python.exe`
- **Hard constraint: the GPU driver is 517.89 / CUDA 11.7.** torch **cu118 and cu121 wheels
  cannot initialise CUDA on this machine** ("driver is too old", found 11070). Windows cu118
  needs driver >= 520, and CUDA forward-compat packages are Linux/datacenter-only.
  **torch 2.0.1+cu117 is the newest usable build.** numpy must stay `<2` for it. Do not
  "helpfully" upgrade torch - this cost real time to diagnose.
- **Run interpreters directly. Do not use `conda run`** - it swallows stdout when stdin is
  piped, which silently hides test output.
- LAPA clone: `D:\Vision\3d_Tracking\Look-Around-and-Pay-Attention-LAPA-` (branch `master`;
  the fix lives on `fix/dlt-weight-normalisation`). **External on purpose - never vendored,
  never committed into this repo.**
- Checkpoint: `D:\Vision\3d_Tracking\checkpoints\lapa.pt` (4,494,291 bytes).
  Needs `torch.load(..., weights_only=False)` - it pickles an `argparse.Namespace`.
- CoTracker weights cache to `<repo>/cache/torch` via `TORCH_HOME` (git-ignored).

### 0.3 Artifacts outside the repo

| Path | Purpose |
|---|---|
| `D:\Vision\3d_Tracking\probe_dlt_weights.py` | proves the DLT weight-cancellation bug (§5.2) |
| `D:\Vision\3d_Tracking\verify_dlt_fix.py` | 13 assertions validating the fix |
| `D:\Vision\3d_Tracking\e2e_dlt_fix_checkpoint.py` | the end-to-end negative result (§5.2.2) |
| `D:\Vision\3d_Tracking\bench_lapa.py` | VRAM/latency sweep (§3.2) |
| `D:\Vision\3d_Tracking\patches\*.patch` | the DLT fix, `git apply --check` clean |

### 0.4 Reproducing the numbers

```powershell
$py = "C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe"
# the tracking-related suites (this is the set that must stay green)
& $py -m pytest tests\test_tracking_pybullet.py tests\test_tracking_unit.py `
  tests\test_tracker3d.py tests\test_camera_convert.py tests\test_motion_unit.py `
  tests\test_rigid_refine.py tests\test_tracking_eval.py -q
# regenerate the eval table in section 2.5
& $py tests\tools\run_tracking_eval.py --tracker3d depth_fusion triangulate
```

**Known-good counts:** the suites above pass (178+ passed, 1 skipped, growing as work lands).

**Pre-existing failures that are NOT caused by this work:** the *full* `tests/` suite has
**25 failures + 7 errors** (metaworld server, side-approach reachability). This was verified
by `git stash`-ing the tracking changes and re-running: identical counts before and after.
Do not try to fix them as part of this effort.

---

## 1. Why point tracking

The rollout tracker in `tracking/` + `providers/trackers/` lifts objects to 3D by
**deprojecting a small appearance-matched image patch through the depth buffer** and
weighted-averaging across `head` + `wrist`. Two weaknesses for manipulation:

1. **The 2D stage follows a region, not a point.** `template` (NCC patch, the default) and
   `csrt` (bbox correlation filter) both track a *window*. Under the partial occlusion that
   dominates manipulation - gripper fingers, the arm, the door itself - the window slides
   onto the occluder or the background and the track is silently wrong. A template keeps
   reporting *high* confidence while matching the occluder, so confidence alone never
   catches it.
2. **The 3D stage is depth-sampled.** A point near an object edge samples background depth
   and deprojects metres away. `sample_depth` medians a 3x3 window and `robust_centroid`
   rejects outliers, but the failure is systematic at exactly the moment it matters: while
   the gripper is closing on the object.

There is also a capability ceiling. A bbox/centroid tracker yields **one** point, and from
one point you cannot recover an axis, a plane, or a rotation. Tracking N corresponded
points per object is what makes 6-DoF pose, hinge axes, pull directions and drawer
perpendicularity computable at all (§6).

---

## 2. Current pipeline, as built

### 2.1 Flow

`Robot.step_env_and_record` drives `TrackingSession.on_frame` once per recorded keyframe:

```
1. capture    CameraView per camera: RGB + metric depth + view/projection matrices
2. track 2D   PointTracker.update per camera            -> points_2d, visible, confidence
3. lift 3D    per-camera deproject through depth        -> one world point per camera
4. health     score_camera + decide_reseeds             -> re-seed unhealthy cameras
5. fuse       MultiCamTracker3D.lift                    -> one world point   <- provider seam
6. gripper    FK pose (truth) + optional visual cross-check
7. monitor    ok | warn | record | abort; abort latches so EXECUTE_TRAJECTORY can stop
```

Every step is defensive: exceptions are caught, logged once and downgraded, because a
tracking bug must never break a rollout that would otherwise have succeeded.

### 2.2 Modules

| Module | Role |
|---|---|
| `tracking/session.py` | orchestrator; owns targets, health, re-seeding, the abort latch |
| `tracking/geometry.py` | project/deproject on **metric** depth, `fuse_world_points`, `camera_weight`, `robust_centroid`, `is_visible`, `surface_point` |
| `tracking/camera_convert.py` | **new** - OpenGL <-> OpenCV bridge, AABB normalisation (§4.2) |
| `tracking/health.py` | per-camera health score and the re-seed policy |
| `tracking/monitors.py` | composable `ok/warn/record/abort` predicates |
| `tracking/report.py` | JSONL frame reports + summary |
| `providers/trackers/` | per-camera 2D `PointTracker`: `template`, `csrt`, `remote` (stub) |
| `providers/tracker3d/` | **new** - multi-camera 3D lift `MultiCamTracker3D` (§2.4) |
| `sim_adapter/camera_math.py` | reconciles PyBullet's GL z-buffer with Genesis's linear metric depth |

### 2.3 Honest status table

| Component | Status | Evidence |
|---|---|---|
| `template` 2D tracker | **tested** | `tests/test_tracking_pybullet.py`, `tests/test_tracking_genesis.py` |
| `csrt` 2D tracker | **untested, opt-in** | needs `opencv-contrib-python`, which conflicts with the pinned `opencv_python==4.8.1.78` |
| `remote` 2D tracker | **stub** | documents a WebSocket JSON protocol, raises `NotImplementedError` |
| depth-fusion 3D lift | **tested** | the two suites above |
| 2-camera (`head`+`wrist`) | **tested** | cross-camera bootstrap + occlusion re-seed |
| 3-camera | **untested** | no third camera defined yet |
| Door scene tracking | **untested** | `sim_envs/pybullet/door.py` exposes `door_handle_pos` as free GT, unused so far |
| Ground-truth accuracy numbers | **measured** | `tests/tracking_eval.py` + `tests/test_tracking_eval.py`; grasp-scene baseline in §2.5 |
| Motion semantics (§6) | **not built** | |
| LAPA integration | **not built** | brought up and audited only (§4, §5) |

Test baseline, measured on this branch: the tracking suites give **44 passed, 6 skipped**;
the new `tests/test_camera_convert.py` adds **24 passed** and
`tests/test_tracking_eval.py` a further **25 passed**. The full `tests/` run has
**25 pre-existing failures and 7 pre-existing errors** (metaworld server, side-approach
reachability) that are present identically with and without the tracking changes -
verified by stashing the changes and re-running.

### 2.4 The 3D provider seam (new)

`TrackingSession` step 5 now delegates to a `MultiCamTracker3D`:

```
providers/trackers/     PointTracker        (per camera, 2D)
                          template | csrt | remote(stub) | cotracker
providers/tracker3d/    MultiCamTracker3D   (all views -> one world point)
                          depth_fusion   today's weighted average, the default
                          triangulate    multi-view DLT, no depth buffer
                          rigid_refine   wraps any of the above, adds a rigid-body
                                         constraint (section 2.6)
                          lapa           LAPA's learned view weighting  (TODO)
```

`depth_fusion` reproduces the previous behaviour exactly, so the default path is unchanged
and `tests/golden/**` stay valid. Trust arbitration (health scoring, re-seeding, dropping a
camera that disagrees with the clearly healthiest one, the abort latch) stays in the
session, because it applies to every lift.

**Correspondence is by seed index, not array position.** This is a real trap found while
building the seam: `_reseed` drops seed points its camera cannot image, so `head[2]` and
`wrist[2]` are routinely *different physical points*. Zipping them positionally would
triangulate different points against each other and produce a confident, smooth, entirely
wrong 3D estimate. `_TargetTrackers.point_index[cam]` now records the seed index of each
tracked point (`-1` = centroid fallback, no correspondence), and every triangulator keys on
it.

---

### 2.5 Ground-truth eval harness (new)

`tests/tracking_eval.py` is a library module (not collected by pytest) that scores the
fused 3D estimate against simulator ground truth for **any** (2D provider x 3D provider x
camera set) combination, on identical frames with identical metrics.

* `SceneDriver` - the seam a scene implements: scripted motion, per-frame ground-truth
  pose, an occlusion window, and the seed points detection would have produced.
  `GraspSceneDriver` drives the PyBullet `grasp` scene (translation + a painted head-camera
  occlusion window); `SyntheticSphereScene` is an analytic two-camera sphere that needs no
  simulator, so the harness itself is testable on any machine.
* `TrackingEvalRunner` runs the real `TrackingSession` and records per frame per object:
  fused point, ground truth, `lost`, `disagreement`, `lift_meta`, wall-clock latency.
* Metrics: `median_l2_m`, `p95_l2_m`, `max_l2_m`, `pct_lost`, `err_during_occlusion_m`,
  `frames_to_recover`, `max_jump_m`, `latency_ms_mean`/`_p95`, `peak_vram_mb`
  (`None` on a CPU-only install), `n_frames`, `n_valid_frames`.
* Output is a self-describing JSON dict (provider names, cameras, resolution, scene,
  thresholds, per-frame trace); `write_report` persists it and `render_comparison_table`
  prints configs side by side.

Two decisions make the numbers mean what they say:

1. **Seed-offset correction.** Seeds sit on the object's *visible surface*; sim ground truth
   is the body *origin*. The constant difference (6.3 cm on the grasp cube, ~= the sphere
   radius on the synthetic scene) is captured once at seed time in the **body frame**, so it
   rotates with the object: `expected(t) = position(t) + R(t) @ offset_local`. Without this
   the harness would report a fixed bias as "error" and every comparison would be noise.
   `test_tracking_eval.py` proves a perfect tracker scores ~0 and, as a negative control,
   that scoring against the raw origin returns exactly `|offset|`.
2. **Lost frames are penalised, not dropped.** Every frame is scored; a frame with no usable
   prediction contributes `lost_penalty_m` (1.0 m). A tracker that reports `lost` every
   frame therefore scores 1.0 m median, not an empty-list "perfect" score.
   `median_l2_valid_m` is emitted separately as the optimistic, diagnosis-only view.

Run it:

```powershell
& ...\vlm_traj\python.exe tests\tools\run_tracking_eval.py --tracker3d depth_fusion triangulate
```

#### Baseline, `grasp` scene, 30 frames, head+wrist at 256x256

Occlusion window = frames 12-18 (head camera depth/RGB painted over), recovery threshold
5 cm, lost penalty 1.0 m. Accuracy metrics are reproducible run to run; latency is not.
Payloads: `outputs/tracking_eval/grasp_<provider>_<lift>.json`.

| config | median | p95 | max | %lost | during occl. | frames to recover | max jump | latency mean / p95 | VRAM |
|---|---|---|---|---|---|---|---|---|---|
| `template` + `depth_fusion` (default) | **24.3 mm** | 52.6 mm | 55.1 mm | 0 % | 26.2 mm | 5 | 26.7 mm | 4.1 / 5.7 ms | n/a (CPU) |
| `template` + `triangulate` | 39.4 mm | 134.1 mm | 139.1 mm | 0 % | 131.8 mm | 0 | 145.4 mm | 7.4 / 9.9 ms | n/a (CPU) |

Reading: with both cameras healthy the two lifts are comparable, but `triangulate` degrades
badly *during* the occlusion window (132 mm vs 26 mm) - with the head track rejected it is
effectively single-view, where a DLT has no baseline, while `depth_fusion` still reads the
wrist camera's depth. Latency excludes camera capture (the scene driver renders), so it is
the tracking step itself.

This is the first empirical confirmation of §5.2.1: geometry-only triangulation is *not*
automatically better than depth fusion at 2 cameras. It also tempers an earlier hypothesis
in this document - that our `triangulate` might already dominate LAPA. At 2 cameras it
clearly does not dominate anything.

---

### 2.6 `rigid_refine`: the constraint LAPA lacks (new)

`providers/tracker3d/rigid_refine.py` is a **decorator** provider: it wraps any other
`MultiCamTracker3D` and adds the inter-point constraint that neither LAPA nor plain
triangulation has. Each of the M points is otherwise triangulated *independently*, so
nineteen good points cannot correct one drifting point.

```python
get_tracker3d("rigid_refine", base="triangulate")
```

Per frame it fits a robust Kabsch/Umeyama transform from a template captured at seed time
to the current points (reusing `tracking.motion.fit_rigid_motion`), then reprojects the
fitted pose back onto each point. Points agreeing within `tolerance_m` keep their measured
value; points beyond it are **replaced by the rigid prediction** and flagged in
`Lift3DResult.meta` (`n_corrected`, `corrected_index`, `rigid_rmse_m`, `rigid_inlier_frac`).

**Why this matters more than it first appears.** Per §5.2.1, at 2 cameras the reprojection
residual carries no information, so *every* residual-based defence - LAPA's IRLS, its
`view_weight_head`, our `max_reproj_px` gate - is structurally blind. The rigid-body
constraint uses a completely independent signal (the object's known shape), so it is the
**only mechanism available that works at head+wrist**.

Measured on the designed-to-be-blind case - 10-point box, one point's wrist pixel drifted
47 px:

| | error vs truth | max reprojection residual |
|---|---|---|
| raw `triangulate` | **0.250 m** | 2.5 px (the 25 px gate never fires) |
| `rigid_refine`, noiseless | **< 1 um** | - |
| `rigid_refine`, 0.5 px noise on every track (50 trials) | **4.5 mm** (~55x better) | - |

**The guard that matters most.** If many points disagree, the object has genuinely moved
non-rigidly or the whole track collapsed - and "correcting" every point onto a bogus pose
would manufacture a smooth, confident, undetectably wrong trajectory, which is worse than
the raw error. So refinement requires `min_inlier_frac=0.6` and otherwise falls through to
the raw wrapped output. `max_stale_frames=5` consecutive fall-throughs drops the template
for re-capture; a re-seed leaving fewer than `min_points` shared seed indices invalidates it
immediately. Fewer than 4 points, a collinear spectrum (`s[1] <= 0.05*s[0]`) or a template
extent under 5 mm all disable refinement rather than produce garbage.

**Known limitation - it fixes bad *points*, not a bad *camera*.** Measured on the grasp
scenario (§2.7): `rigid_refine` more than 4x improves the clean scripted-motion case
(2.2 mm vs `depth_fusion`'s 9.8 mm) but during a **full head-camera blackout** it scores
141.4 mm - *identical* to the raw `triangulate` it wraps, i.e. it contributed nothing.

That is the guard behaving exactly as designed, not a bug: when an entire camera drops, every
point loses the same view at once, so every point is wrong together, the inlier fraction
falls below 0.6, and the provider correctly refuses to trust a pose fitted to uniformly bad
data. The constraint is powerful against *one point of N* going bad (its design case,
0.250 m -> 4.5 mm) and powerless against *all points* degrading together.

The fix is not to weaken the guard - it is to give the rigid fit a temporal anchor, so a
full-camera loss coasts on the *predicted* pose rather than falling through to raw. That is
§7.4 item 3 (Kalman filter per rigid body), and this measurement is the concrete argument
for it.

Still to do: score it through the eval harness on the grasp and door scenes (§7.1).

---

### 2.7 Scenario coverage across providers (new)

`tests/tracking_scenario.py` is now parameterised over the 3D provider. Its four behavioural
assertions - scripted-motion accuracy, cross-camera bootstrap, painted-occluder repair and
detach abort - run unchanged against every provider discoverable in
`providers.tracker3d.factory.SUPPORTED`, with unavailable ones skipped (`lapa` currently
skips: module missing). Only the numeric accuracy band varies per provider.

| provider | scripted median | head-blackout error | asserted band (median / occluded) |
|---|---|---|---|
| `depth_fusion` (default) | 9.8 mm | 14.6 mm | 40 mm / < 80 mm |
| `triangulate` | 6.8 mm | 141.4 mm | 40 mm / **[50, 300] mm** |
| `rigid_refine` | **2.2 mm** | 141.4 mm | 40 mm / < 300 mm |

The `triangulate` occlusion band is deliberately **two-sided**: the test fails if the number
silently *improves* as well as if it degrades. A known-bad behaviour that is understood and
explained (§5.2.1) is more useful pinned than papered over - if it ever improves, something
changed that we need to know about.

**The default path is provably unchanged.** The pre-change scenario file
(`git show HEAD:tests/tracking_scenario.py`) was run against the current code: 5/5 pass, and
the new default tests reproduce identical numbers (median 0.0098-0.0099 m, run-to-run
jitter only). When no provider is named, `make_session` forwards no `tracker3d` kwarg at
all, so it is the original code path rather than an equivalent one. No golden files touched.

---

## 3. LAPA bring-up (measured)


- Clone: `D:\Vision\3d_Tracking\Look-Around-and-Pay-Attention-LAPA-` @ `master`, **outside
  the repo**, reached through a bridge module. Never vendored, never committed.
- Env: conda `env_lapa_tracking`, Python 3.10, **torch 2.0.1+cu117**.
  - The local driver is **517.89 / CUDA 11.7**, which cannot run `cu121` or `cu118` wheels
    (`torch.cuda.is_available()` returns False with a "driver is too old" warning). `cu117`
    is the newest build this driver supports. Upgrading the driver would allow newer torch.
- Checkpoint: `bishoygaloaa/LAPA-PointOdyssey-MC/lapa.pt`, 4,494,291 bytes.
  Loads into `LAPA(volume_size=16)` with **no missing and no unexpected keys**.
- `python -m lapa.models.lapa` smoke test passes; 370,696 parameters.

### 3.1 What the checkpoint bakes in

Dumped from `ckpt["args"]` (requires `weights_only=False`; it pickles an `argparse.Namespace`):

| Field | Value | Consequence |
|---|---|---|
| `volume_size` | 16 | grid is built lazily and is *not* in `state_dict`, so it can be changed at load |
| `num_views` | 3 | the trained configuration; view slots 3-4 are zero-padded and out of distribution |
| `max_points` | **64** | **hard cap on points per LAPA call** - our seed sets must respect it |
| `use_gt_tracks` | **False** | good: the released weights were *not* trained on ground-truth 2D |
| `num_frames` | 24 | trained on short clips |
| `epoch` | **12** of 50 | **partially trained** |
| `best_apd` | **100.0** | never improved from its initialisation - the metric looks broken |
| `dataset` | pointodyssey | Y-up, 3-camera ring; PyBullet is Z-up with head+wrist |

`epoch 12/50` with `best_apd` stuck at exactly `100.0` means this is an early, partially
trained checkpoint whose validation metric was probably not functioning. Any evaluation
result must be reported with that caveat; a weak result is at least as likely to indict the
checkpoint as the architecture.

### 3.2 Measured cost on the local 4 GB RTX A1000

64 points, 256x256 renders, median over 17 frames after warm-up:

| Configuration | Peak VRAM | Latency | FPS |
|---|---|---|---|
| LAPA only, 2 views | 22 MB | 16 ms | 61 |
| LAPA only, 3 views | 24 MB | 23 ms | 44 |
| + DINOv2 518px fp32, 3 views | 974 MB | 1916 ms | 0.5 |
| + DINOv2 252px fp32, 3 views | 397 MB | 292 ms | 3.4 |
| + DINOv2 518px fp16, 3 views | 507 MB | 951 ms | 1.1 |
| **+ DINOv2 252px fp16, 3 views** | **221 MB** | **194 ms** | **5.1** |

Conclusions:

1. **VRAM is a non-issue.** Worst case is 974 MB of 4096 MB. The Kaggle/Colab server is
   justified by *the robot PC having no GPU*, not by capacity.
2. **LAPA itself is free** (~24 MB, ~23 ms). **DINOv2 is ~99 % of the cost**, and that cost
   is set by the *input resolution we choose*, not by our render size - DINOv2 resizes
   whatever it is given. Dropping 518px -> 252px (still >= 1 patch per 14 source pixels at
   256x256 renders) plus fp16 is a **~10x speedup** for no loss we can currently detect.
3. Even so, 5 FPS is not real-time on this GPU, and CoTracker3 is not yet in the budget.
   These numbers are for *this laptop GPU*; a T4/desktop GPU is several times faster.

---

## 4. What LAPA actually computes

### 4.1 In plain English

`forward_frame(view_points_2d, view_features, view_K, view_w2c_norm, queries, image_size,
view_valid)` takes per-view **2D pixel tracks** plus **DINOv2-768 descriptors sampled at
those pixels**. **It never sees RGB**, and it never runs a 2D tracker - CoTracker lives only
in LAPA's *offline* cache scripts (`lapa/features/precompute.py`), using the
`cotracker3_offline` entrypoint. Driving a 2D tracker is our job.

The 3D position is produced by `decode_queries`:

1. `view_weight_head(DINOv2 feature, reprojection residual of the previous estimate,
   validity flag)` -> a per-view, per-point weight in `(0, 1)`.
2. `triangulate_dlt_irls(uv, P, view_w, iters=2, sigma_px=5.0, min_views=2)` -> `p_dlt`.
   Direct Linear Transform: each camera turns a pixel into a 3D ray; the point is where the
   rays nearly intersect, solved by SVD. IRLS is *intended* to re-solve twice, shrinking the
   weight of cameras whose rays missed badly. **In practice the weights are cancelled by a
   row-normalisation step and neither the learned weights nor the IRLS affect the result -
   see §5.2.** **This runs under `torch.no_grad()`** - it is a fixed geometric anchor, not
   something the network learned.
3. A learned residual: `points = p_dlt + tanh(delta) * 0.02`, clamped to `[-1.5, 1.5]`.
   Coordinates are normalised so the workspace AABB spans `[-1, 1]`, so the network may
   nudge the point by at most **2 % of the AABB half-extent** - about **+-8 mm** at
   `half = 0.4 m`.

**So the network does not regress 3D position; the geometry does.** What LAPA *intends* to
learn is (a) which camera to trust per point per frame, (b) a visibility logit, (c) an
<=8 mm polish. §5.2 shows (a) never reaches the output, leaving (c) as the only learned
influence on position. Accuracy is therefore bounded by our 2D tracks and our calibration.

Temporal state is a single EMA: `queries = 0.8 * queries + 0.2 * points_3d`. No velocity,
no covariance, no window. `forward()` is a plain loop over `forward_frame()`, so LAPA is
**streaming-capable** - but never call `forward()`, which retains `attn_lists`/`corr_lists`
for every frame.

### 4.2 Convention gotchas

- PyBullet/Genesis return a **flat-16, column-major, OpenGL** view matrix (-Z forward,
  +Y up). LAPA wants **OpenCV** (+Z forward, +Y down):
  `w2c_cv = diag(1,-1,-1,1) @ reshape(vm, (4,4), order='F')`. Getting this wrong makes every
  depth negative, every observation invalid, and the tracker **looks frozen rather than
  erroring**. `camera_convert.assert_opencv_frame_sane` exists specifically to catch it.
- `K` is in **pixels at the render resolution**, and `image_size` must be **our** `(W, H)`.
  The repo hard-codes `(640, 360)` at `lapa/data/mc_dataset.py:194`; `populate_volume`
  divides pixel coordinates by `[W, H]`, so a wrong size mis-scales the attention.
- Extrinsics must be pre-warped by `build_w2c_normalized(w2c, center, half)` into the
  `[-1,1]^3` AABB; the output is normalised and must be de-normalised with the same AABB.
- **Do not reuse `utils.get_intrinsics_extrinsics`** - it pins the principal point at
  `(0,0)` and returns a *camera-to-world* `Rt`. Wrong on both counts.
- 2D points are `(x, y)` pixels, origin top-left, which **already matches** this repo.
- DINOv2 force-resizes internally, so 256x256 renders are fine.
- All three LAPA scripts default to `--device cuda:1`; force `cuda:0`.
- Ignore `inference_lapa.py` and `run_lapa_pipeline.py` - the latter uses randomly
  initialised legacy modules that never load a checkpoint.

---

## 5. Constraint audit: does LAPA implement the consistency checks we need?

The target architecture (below, "reference pipeline") specifies a set of 2D<->3D
consistency constraints intended to survive noise, drift and occlusion. This section checks
each one against the LAPA **source**, not the paper. Verdicts are backed by code locations
and, where decisive, by values read out of the checkpoint.

### 5.1 Summary

| # | Reference-pipeline constraint | In LAPA? | Verdict |
|---|---|---|---|
| 0.1 | Detect + drop N points + lift via depth | **No** | `queries` are a caller-supplied input |
| 0.2 | Cross-project points from cam1 into cam2 to establish correspondence | **No** | LAPA *requires* pre-corresponded `(V, M, 2)`; it cannot build correspondence |
| 0.3 | Visibility + projected-vs-actual depth check | **No** | `view_valid` is an optional caller-supplied mask |
| 0.4 | Cross-camera feature-similarity verification | **No** | features are pooled per view and summed; never compared across views |
| 0.5 | Retry/bootstrap until the second camera verifies | **No** | no re-seeding anywhere |
| 0.6 | 3D template + Kalman init | **No** | no template, no KF |
| 1.2 | **Absolute epipolar validation, hard-reject > 2 px** | **Effectively no** | implemented but inert - see 5.2 |
| 2 | Depth-jump sanity vs prediction | **No** | no gate on the new measurement at all |
| 2b/3 | Cross-camera spatial consistency, early fusion | **Yes in design, broken in code** | fusion happens inside the DLT, but the weights are cancelled - see 5.2 |
| 4 | **Shape/template similarity, 6-DoF via Umeyama/SVD** | **No** | points are triangulated **independently**; no inter-point constraint |
| 5 | Kalman filter + occlusion hierarchy | **No** | only a fixed `0.8/0.2` EMA; `vis_logit` is returned but gates nothing |

**Headline:** of the constraints intended to survive noise, drift and occlusion, LAPA
implements **none of them effectively**. Its one real mechanism - weighted multi-view fusion
with learned per-view trust - is **disabled by a one-line bug** (§5.2). Everything about
initialisation, cross-view verification, temporal filtering, rigid-body structure and
occlusion fallback is absent and remains our responsibility.

### 5.2 The weighted DLT does not use its weights - and this is the big one

`lapa/geom/dlt.py:60-72` scales each row of the DLT matrix by `sqrt(w)` and then normalises
every row to unit length:

```python
sw = w.sqrt().unsqueeze(-1)
row0 = row0 * sw;  row1 = row1 * sw
A = ...stack(row0, row1)...
mag = A.square().sum(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
A = A / mag                      # <-- cancels the weight exactly
```

Row *i* is `sqrt(w_i) * r_i`, so `||row_i|| = sqrt(w_i) * ||r_i||`, and dividing gives back
`r_i / ||r_i||`. **The weights are removed.**

Measured directly against LAPA's own `triangulate_dlt`, 3 cameras, one view drifted 40 px:

| Weights supplied | Resulting mean error |
|---|---|
| uniform | 1.170834 m |
| view 1 crushed to `1e-5` | 1.170834 m |
| view 1 boosted to `1e3` | 1.170834 m |
| random | 1.170833 m |

Changes of **eight orders of magnitude in the weights move the answer by ~1e-8 m.** IRLS
with 0, 2 or 10 rounds is likewise identical (max change `4.8e-7`). Removing the single
normalisation line and re-running with the bad view down-weighted gives **0.00009 m** -
a four-order-of-magnitude improvement.

Consequences:

- `view_weight_head`, the learned per-view trust gate that is the paper's central
  contribution, **has no effect on the output position**.
- The IRLS reweighting **has no effect**.
- The *only* thing the weights still do is `min_views`: `n_valid = (w > 1e-6).sum(0)` is
  computed **before** the normalisation, so a zero-weighted view is still excluded from the
  count (verified - the `min_views=2` fallback still fires correctly).
- Therefore **LAPA's 3D output is plain unweighted DLT plus the <=8 mm learned residual.**

This also explains the odd training telemetry in §3.1 (`best_apd` frozen at `100.0`,
stopped at epoch 12/50): most of the network receives gradient only through a residual
clamped to 2 % of the AABB.

Our `triangulate` provider deliberately **omits** the row normalisation, so its weights do
work. Tuned against a 75-pixel single-view drift with 3 cameras:

| Reweighting | Final error |
|---|---|
| Gaussian `0.05 + 0.95*exp(-(e/5)^2)`, 2 rounds (**LAPA's**) | 0.174 m |
| Huber `5/max(e,5)`, 2 rounds | 0.134 m |
| Huber `2/max(e,2)`, 10 rounds | 0.00024 m |
| **Cauchy `1/(1+(e/2)^2)`, 10 rounds (ours)** | **0.00000 m** |

LAPA's Gaussian is the worst choice even when it *is* applied: once the first solve is
dragged off by the outlier, every view's residual is large, so the Gaussian crushes the
*good* views too and cannot recover. Cauchy decays gently enough to keep them alive.

### 5.2.1 Two cameras cannot detect a bad track at all

Independently of any bug: with exactly 2 views the DLT system is **exactly determined**
(4 equations, 4 homogeneous unknowns). Any pair of pixels therefore triangulates to a point
that reprojects almost perfectly. Measured: drifting one view by 47 px moves the 3D point
by >1 cm while the reprojection residual stays at **0.18 px**.

So at 2 cameras the reprojection residual - the signal both LAPA's IRLS *and* its
`view_weight_head` consume - carries **no information**, and no weighting scheme, learned or
otherwise, can work. Robustness genuinely begins at 3 views. Both facts are pinned by
`tests/test_tracker3d.py::TestTriangulateIrls`.

### 5.2.2 `view_weight_head` is saturated - it emits no signal even if you fix the DLT

Fixing §5.2 (patch prepared, 13/13 verification assertions pass) reveals a second,
independent failure. Driving `forward_frame` with the released checkpoint, the head outputs
a logit of **+6.91 for every view** - `w = 0.9990` across the board - and sweeping the
reprojection residual from 0 to **200 px** moves that logit by **0.01**.

The decisive control: with `iters=0`, the pre-fix and post-fix outputs differ by **exactly
0.000000**. With the IRLS off, the only weights remaining are the head's, and since they are
all identical, a weighted DLT and an unweighted DLT coincide. The head is a constant
function.

Consequently the fix **alone makes accuracy slightly worse** on this checkpoint, because the
entire post-fix delta comes from re-enabling the IRLS, which at the shipped `sigma_px=5.0`
diverges (0.109 -> 0.163 over 10 rounds):

| 2D drift | output PRE-fix | output POST-fix |
|---|---|---|
| 5 px | 0.053377 | 0.053955 |
| 10 px | 0.108126 | **0.124367** |
| 20 px | 0.218605 | 0.225864 |
| 40 px | 0.449090 | 0.448978 |

This is expected rather than surprising: the head was *trained* with its output
disconnected, so it never received a gradient that rewarded discriminating between views -
the same mechanism that left `log_sigma_sfm` at its init value in §5.3.

The headroom is nevertheless real. Feeding **oracle** weights through the fixed solver takes
the 10 px case from 0.109138 to **0.000004** - roughly 30,000x. The geometry can deliver it;
nothing in the released model knows how to ask for it. Hence §7.4 items 1 and 8 are a single
package: **fix and retrain, or do neither.**

### 5.3 The epipolar and SfM masks are inert - with proof

`populate_volume` (`lapa/models/lapa.py:283-400`) builds two geometric masks over the
grid-voxel-to-point attention:

- `epipolar_mask` computes `exp(-d^2 / (2 * sigma_epi^2))` where `d` is a **pixel** distance
  to the epipolar line. The checkpoint's learned `sigma_epi = 0.103` **pixels**, so the mask
  is `exp(-47 d^2)`: already ~1e-22 at a 1-pixel deviation. It is then combined as
  `mask = max(m_epi, depth_gate * 0.5)`, so the constant `0.5` depth gate dominates
  **always**. The epipolar term never binds.
- `sfm_mask` is computed **twice and both results are discarded**; the code replaces them
  with `depth_gate = (depth_a > 0) * (depth_b > 0)`, with in-source comments conceding
  "Self-SfM above is degenerate", "expensive", "Practical: soft-mask attn by
  `max(epi, ones*0.5)`".
  **Decisive evidence:** the checkpoint's `log_sigma_sfm = -0.693147`, which is *exactly*
  `log(0.5)`, its initialisation, unchanged to the last digit after 12 epochs. It received
  **zero gradient**, because nothing it computes ever reaches the loss. (For contrast,
  `log_sigma_epi` moved from `-2.3026` to `-2.2712`, and `log_temperature` from `-2.3026`
  to `-2.3455` - both did receive gradient.)

So the effective mask reduces to *"discard grid voxels that fall behind either camera"*, and
the volumetric attention is plain pixel-distance softmax. Both `sfm_mask` calls are pure
wasted compute.

Note also that the epipolar construction is conceptually redundant here: the "epipolar line"
is built from a **grid voxel whose 3D position is already known**, so the correct constraint
is the point projection - which *is* the distance attention. An epipolar line only adds
information when depth is unknown.

### 5.4 What LAPA does do well

Little, once §5.2 and §5.3 are accounted for. In fairness:

- The **design** is right: fusing inside the DLT rather than averaging two independent 3D
  lifts is better than reference-pipeline step 3, and conditioning a per-view weight on
  appearance + reprojection residual + validity is the correct thing to learn. The idea is
  sound; the implementation discards it.
- `min_views` is honoured, so a point with too few views is at least flagged rather than
  silently invented.
- It is genuinely streaming-capable and extremely cheap (~24 MB, ~23 ms, §3.2).

### 5.5 The structural gaps that matter most for manipulation

1. **No inter-point/rigid-body constraint.** Each of the M points is triangulated
   independently. Nineteen good points cannot correct one drifting point. The reference
   pipeline's Umeyama/SVD step is precisely what supplies this, and it is absent.
2. **`min_views=2` with a silent fallback.** `triangulate_dlt` falls back to
   `fallback=queries` - the previous EMA - when a point has fewer than 2 positive-weight
   views. With only `head`+`wrist`, one occluded view makes the point **coast on its EMA and
   drift, reporting no error**. LAPA has no depth, so it genuinely cannot do the reference
   pipeline's "if one camera lost point i, use the other camera's 3D coordinate". Combined
   with §5.2.1, this is the strongest argument for a third camera.
3. **No outlier gate on the measurement.** A wild triangulation is immediately blended in at
   20 % and then persists.
4. **`vis_logit` gates nothing** inside the model, and on our data its calibration is
   untrusted anyway (trained Y-up, 3-camera ring, PointOdyssey).
5. **`TriangulationMLP` uses `BatchNorm1d` over the point dimension.** In `eval()` mode it
   uses running statistics gathered from batches of 64 PointOdyssey points, which assume
   inputs that fill the normalised `[-1,1]` box. An inflated AABB therefore degrades the
   refinement - which is why `camera_convert.aabb_from_points` uses median+MAD outlier
   rejection rather than LAPA's percentile recipe (percentiles are meaningless at N=6 and a
   single outlier inflated `half` by ~25x in testing).

---

## 6. Higher-level motion semantics (`tracking/motion.py` - built)

Turning tracked **points** into actionable **directions**. Only possible because we track
>= 3 corresponded points per object. Implemented in `tracking/motion.py` (numpy only,
self-contained, importable without the session) with 62 tests in
`tests/test_motion_unit.py`.

`MotionBuffer` holds the last K frames as `(K,N,3)` + a validity mask and **scatters by seed
index, not array position** - the same correspondence contract as §2.4, with `-1` ignored.
`push_lift(result)` accepts a `Lift3DResult` directly.

1. `fit_rigid_motion(P0, P1) -> RigidFit` - Kabsch/Umeyama, robust iterative worst-point
   rejection at `median + 3*1.4826*MAD` (floored at 0.1 mm), with the `det(R) < 0` reflection
   fix. Unpacks as `R, t, rmse`; also carries `inliers`, `n_used`.
2. `screw_decompose(R, t, ref_point) -> ScrewMotion` - rotation angle, axis direction, and
   the **axis location** by least squares on `(I - R) c = t_perp`. `(I - R)` is singular
   along the axis, so this is a min-norm lstsq solution re-footed onto `ref_point` (use the
   tracked centroid). `theta -> 0` returns `axis=None` rather than garbage; `theta = pi`
   takes the `R + I` eigenvector branch. This is the principled extraction of a **hinge
   line** (door, lever) or a **slide direction** (drawer).
3. `classify_motion(...)` -> `static | prismatic | revolute | free`, with hysteresis
   (`MotionClassifier`, dwell `switch_frames=2`) so the label cannot flicker.
4. `fit_plane(points, view_point|view_dir) -> PlaneFit` - SVD normal, sign-disambiguated
   toward the camera.
5. `centroid_direction(buffer, window, dt) -> LineFit` - PCA/total-least-squares line fit on
   the centroid trajectory. More stable than frame-to-frame deltas.
6. `estimate_motion(...) -> MotionEstimate` - 13 fields plus `reason`, with a JSON-safe
   `to_dict()` matching `TrackedObjectState`'s style.

**Measured recovery.** On clean synthetic screw motions: axis-direction error `0.0`,
axis-point `5.5e-16 m`, angle `1.1e-15`, rmse `6.6e-17`. Under noise: sigma = 1 mm gives
median axis error 1.39 deg / axis point 3.4 mm; sigma = 4 mm gives 5.80 deg / 12.3 mm.

**The test that justifies the whole approach.** For a rotating object, screw-based
prediction errs `< 1e-9 m` while naive centroid-velocity errs `> 1e-3 m`, with the naive
tangent off by exactly `theta/2`. That gap is the entire reason for the screw decomposition
rather than differencing positions.

The two requested outputs fall out:

- **Door-handle pull direction** = `omega_hat x (p_handle - p_axis)`, normalised. *Not* the
  raw centroid velocity, which is only tangent to the arc at one instant. Reporting the axis
  too lets a planner predict the whole arc instead of chasing it.
- **Drawer perpendicularity** = `|d_hat . n_hat_surface|` with its angle in degrees. `1.0` is
  a perfectly perpendicular pull; a falling value means the gripper is racking the drawer or
  has slipped.

`confidence == 0` cases set `reason` to one of `insufficient_frames`, `no_correspondence`,
`too_few_points`, `collinear_points`, `near_zero_motion`, `poor_fit`, and **never emit a
direction**. `to_dict()` emits `null` for an absent axis or direction rather than a
plausible-looking garbage vector.

**Still in flight** (§7.4): tightening the robustness gates - the initial confidence model
is too generous (0.911 at 4 mm noise, where axis error is already 5.80 deg), the minimum-
displacement and `disagreement` gates, sign-aligned EMA smoothing, and the monitor wiring
(`moves_along`, `moves_perpendicular_to_surface`, `revolute_axis_stable`) on the existing
`ok | warn | record | abort` contract.

---

## 7. TODO

### 7.1 Foundation

- [x] `tracking/camera_convert.py` + 24 tests - OpenGL<->OpenCV, AABB normalisation.
- [x] `providers/tracker3d/` seam with `depth_fusion` (behaviour-preserving) and
      `triangulate` + 24 tests; seed-index correspondence map in `TrackingSession`.
- [x] **Ground-truth eval harness** (`tests/tracking_eval.py`, §2.5). Provider-agnostic,
      scored against sim GT: median/p95/max 3D L2, % frames lost, error *during* occlusion,
      frames-to-recover, max inter-frame jump, latency, peak VRAM. Seed-offset corrected in
      the body frame; lost frames penalised rather than dropped. 25 tests in
      `tests/test_tracking_eval.py`, including "a perfect tracker scores ~0".
- [x] Record baseline numbers for `template` + `depth_fusion` on the grasp scene (§2.5:
      24.3 mm median, 52.6 mm p95, 0 % lost, 26.2 mm during occlusion, 5 frames to recover).
- [ ] Same baseline on the door scene, once the door affordance scene test below exists.
- [ ] Door affordance scene test: seed from `{"point": [x, y], "label": "door_handle"}`,
      drive the hinge/latch joints, score against `door_handle_pos`.
- [ ] Third `side` camera as a config change, to escape the `min_views=2` cliff.

### 7.2 CoTracker

- [x] `providers/trackers/cotracker_tracker.py` as a **public** `PointTracker`, usable with
      today's depth fusion with **no LAPA clone and no checkpoint**. Registered in
      `factory.SUPPORTED`, exposed as `--tracker-provider cotracker`, 24 fake-model tests
      plus a real-checkpoint test gated on `COTRACKER_REAL_TEST=1`. Weights (97 MB) cache
      into `cache/torch` via `TORCH_HOME`. Verified end to end on `cuda:0` and on CPU.
- [x] Validate `cotracker3_online`. **Done, and it found a problem - see §7.2.1.**

### 7.2.1 The `cotracker3_online` latency granularity problem

This was flagged in the plan as the largest remaining unknown. It is now measured, and the
result is worse than hoped.

`cotracker3_online` runs a sliding window of **16 frames with step 8**. It therefore emits a
new prediction only **once every 8 frames**. Between flushes the provider returns the last
prediction verbatim (never extrapolated) with `meta["stale_frames"]` counting 0..7 and the
score decayed 5 % per stale frame, so the staleness is at least *honest* and visible to the
session. But the underlying fact stands: **for 7 of every 8 frames the 2D point is frozen.**

Measured on the RTX A1000 (cu117, 320x240):

| Quantity | Value |
|---|---|
| First flush (warm-up) | ~6.5 s |
| Steady-state flush | ~1.95 s per 8 frames (~244 ms/frame amortised) |
| Stale-frame cost | ~0 ms |
| Error on a fresh flush | **0.21 px** |
| Error on a stale frame, target moving 5 px/frame | **up to 38 px** |

The accuracy on fresh flushes is superb - 0.21 px, far better than patch matching. The
problem is entirely the staleness: a point moving at 5 px/frame is wrong by up to 38 px just
before the next flush.

**Why this is worse for 3D than it looks in 2D.** Head and wrist cameras flush on the *same*
schedule but see *different* motion, so during a stale run the two views are inconsistent
with each other by different amounts. Triangulating a stale pixel against another stale pixel
does not average out - it produces a confidently wrong 3D point. And per §5.2.1, at 2 cameras
the reprojection residual **cannot detect this**. This is a concrete instance of exactly the
failure the rigid-body constraint (§7.4 item 2) exists to catch.

**Options, to be decided by the eval harness:**
1. Accept 8-frame granularity and run the tracking pipeline at 1/8 the sim rate. Honest,
   simple, and probably fine for slow manipulation motions - but it caps the control rate.
2. Interpolate/extrapolate across the stale window using the 3D velocity estimate rather
   than freezing 2D. Better, but it invents data, so it must be flagged in `lift_meta`.
3. Use the score decay as-is and let `TrackingSession` health down-weight coasting cameras.
   Cheapest; already implemented, just not consumed yet (see the wiring note below).
4. Reduce the step size. Worth testing whether `step < 8` is configurable at inference on
   the online model; it trades latency for compute.

**Not yet wired:** `TrackingSession` does not consume `result.meta["stale_frames"]`. It
should fold something like `1/(1+stale)` into the health score, and must not mistake a stale
frame for temporal discontinuity when deciding to re-seed.

**Revised expectation.** The plan assumed "CoTracker alone may be most of the win". Its
*accuracy* claim survives (0.21 px), but its *latency* profile means CoTracker is not a
drop-in replacement for the 244 ms/frame patch tracker at full rate. The eval must score
CoTracker on wall-clock-comparable terms, not just per-fresh-flush error.

### 7.3 LAPA adapter

- [ ] `providers/tracker3d/lapa_tracker3d.py` + bridge module to the external clone.
- [ ] Respect **`max_points = 64`** per call; chunk or subsample larger seed sets.
- [ ] Drive `forward_frame` per sim step; `del out` each frame; never call `forward()`.
- [ ] Sample DINOv2 at tracked pixels; default to **252 px fp16** (~10x faster than the
      518 px fp32 default, §3.2), with resolution and precision configurable.
- [ ] One-off bring-up assertion using sim-GT 2D tracks to prove conventions and wiring.
      **This is a wiring check, never an evaluation result** - `--use_gt_tracks` is what
      makes the LAPA repo's own numbers look good, and the released checkpoint was
      explicitly trained with it `False`.
- [ ] Threshold `vis_logit` on our own data rather than trusting `> 0`.

### 7.4 Improving LAPA (from the §5 audit)

Ordered by expected value. Item 1 is a one-line change that plausibly unlocks the method's
entire intended benefit; items 2-4 address the gaps that most directly cause the
manipulation failures this work exists to fix.

1. **Fix the weighted-DLT row normalisation (§5.2) — necessary, but NOT sufficient.**
   Apply `sqrt(w)` *after* normalising the unweighted rows, in `lapa/geom/dlt.py`. Patch is
   prepared at `D:\Vision\3d_Tracking\patches\0001-fix-dlt-apply-view-weights-AFTER-row-normalisation-s.patch`
   (`git apply --check` clean against `master`), verified by 13/13 assertions in
   `verify_dlt_fix.py`: weight sensitivity goes from 5.96e-07 m to 1.310762 m, crushing the
   bad view drops error 1.170834 m -> 0.000080 m (14550x), `d(out)/d(w)` goes 0 -> 0.19,
   uniform weights are an exact no-op (0.0 delta, so it is backward-compatible), `min_views`
   semantics are unchanged, and clean-data accuracy against an independent reference is
   3.3e-16 with conditioning only mildly worse (cond 8.0 -> 14.3).

   **But end-to-end with the released checkpoint the fix does not help — it slightly hurts.**
   Measured through `forward_frame`, 5 seeds x 64 points, error in normalised units:

   | 2D drift | output PRE-fix | output POST-fix |
   |---|---|---|
   | 5 px | 0.053377 | 0.053955 |
   | 10 px | 0.108126 | **0.124367** |
   | 20 px | 0.218605 | 0.225864 |
   | 40 px | 0.449090 | 0.448978 |

   **Root cause: `view_weight_head` is saturated and emits no signal.** It outputs a logit of
   **+6.91 for every view** (`w = 0.9990`), and moving the reprojection residual from 0 to
   **200 px** changes that logit by **0.01**. Decisive control: with `iters=0` the PRE/POST
   difference is **exactly 0.000000** — with the IRLS disabled, the fix changes nothing,
   because the only weights left are the head's, and they are all identical. Every bit of the
   POST-fix delta therefore comes from re-activating the IRLS, which at the shipped
   `sigma_px=5.0` **diverges** on this data (0.109 -> 0.163 by 10 iterations).

   The headroom is real, though: feeding *oracle* weights through the fixed solver takes
   0.109138 -> **0.000004** (~30,000x). The geometry can deliver; the head simply was never
   trained to drive it — unsurprising, since it was trained with its output disconnected.

   **Therefore: fix + fine-tune (item 8) is a package, not two independent items.** Applying
   the fix alone to the released checkpoint is a regression. If we ship the fix before
   retraining, also set `iters=0` or retune to `sigma_px≈2.0`. And note this means **every
   published LAPA number was produced with the weighting disabled.**
2. **Add the rigid-body constraint LAPA lacks (§5.5.1).** Run Umeyama/Kabsch over the M
   corresponded points against a template captured at t=0, then *reproject the fitted rigid
   pose back onto individual points* to correct or replace drifting ones. This needs no
   retraining - it wraps LAPA rather than modifying it - and it is the only mechanism that
   can fix a bad point at **2 cameras**, where the reprojection residual is provably blind
   (§5.2.1). It also directly produces the 6-DoF pose §6 needs.
3. **Add a measurement outlier gate and a real temporal filter (§5.5.3).** Replace the fixed
   `0.8/0.2` EMA with a constant-velocity Kalman filter per rigid body (better than per
   point - it shares statistical strength across points). Reject measurements exceeding a
   Mahalanobis threshold instead of blending them in at 20 %. Feed the KF prediction in as
   the `fallback` argument of `triangulate_dlt` so a point with `< min_views` coasts on a
   *predicted* trajectory rather than a frozen EMA.
4. **Close the `min_views=2` hole (§5.5.2, §5.2.1).** Two options, not exclusive: (a) add a
   third camera - which §5.2.1 shows is required for *any* residual-based robustness to
   function at all; (b) for single-view points, intersect the camera ray with the rigid-body
   pose predicted by item 2 - a ray plus a known object pose *is* a determined point, which
   recovers the reference pipeline's "use the other camera's coordinate" without depth.
5. **Make the epipolar check actually bind (§5.3).** Either apply it as a hard pre-filter on
   the 2D tracks at a sane threshold (~2 px, as the reference pipeline specifies) *before*
   triangulation, or re-parameterise `sigma_epi` in normalised-diagonal units and re-tune.
   As shipped it is inert. Simplest effective version: a standalone epipolar gate in our
   adapter, independent of the model. Note this is one of the few checks that still works
   at 2 cameras, which makes it more valuable than its current status suggests.
6. **Delete the dead `sfm_mask` computations (§5.3).** Two full `cdist` passes over
   `(volume_size^3, M)` per view per frame whose results are discarded. Free speedup, and it
   removes a misleading code path. If a depth-consistency check *is* wanted, we have real
   metric depth and can implement reference-pipeline step 0.3 properly in the adapter
   (project the 3D point into each view, compare predicted vs sampled depth, reject on a
   > 5 cm mismatch) - far more useful than the grid-voxel version.
7. **Add cross-camera feature verification (§5.1 row 0.4).** We already hold DINOv2
   descriptors per view; comparing point *i*'s descriptor across views is nearly free and
   directly detects the "template latched onto the occluder" failure. LAPA sums per-view
   features and never compares them. Like item 5, this works at 2 cameras.
8. **Fine-tune the checkpoint — now a prerequisite, not an option (§3.1, item 1).** It is
   `epoch 12/50` with `best_apd` stuck at `100.0`, and it was trained with its core
   mechanism disconnected, which is why `view_weight_head` is saturated at a constant
   logit of +6.91. Retraining with the fixed DLT is what converts item 1 from a regression
   into the measured ~30,000x oracle headroom. Cheap: 371k parameters. A weak evaluation
   result indicts this checkpoint at least as much as the architecture.
9. **Consider dropping LAPA's learned part entirely — now the leading candidate.** Per §5.2
   LAPA's 3D output is *unweighted DLT plus a <=8 mm nudge*, and per item 1 its learned
   weighting head is saturated at a constant value, so it contributes nothing even once the
   solver is fixed. Our `triangulate` provider already computes a properly weighted DLT with
   a better robust loss (Cauchy: 0.00000 m vs LAPA's Gaussian 0.174 m on the same outlier).
   A hand-written weight from (visibility, reprojection residual, cross-view feature
   similarity) plausibly beats the learned head **with no DINOv2 at all**, removing **99 % of
   the compute** (§3.2) and the GPU requirement along with it — which would also make the
   §7.5 remote server unnecessary. **The eval harness must settle this**, and it is why
   `triangulate` ships as a first-class eval config rather than a debugging aid.

### 7.5 Deployment

- [ ] Extend the WebSocket JSON protocol documented in `providers/trackers/remote_tracker.py`
      to carry 3D results; implement both ends. The robot PC may have no GPU at all - that,
      not VRAM (§3.2), is the reason for a remote server.
- [ ] Kaggle/Colab T4 + ngrok demo server.
- [ ] Written integration recommendation.

### 7.6 Docs

- [ ] Update `document.md` §13 (Rollout tracking) once the seam and eval land.
- [ ] Changelog entry.

---

## 8. Risks and open questions

- **The headline risk is no longer "does LAPA beat triangulation".** Per §4.1 and §5, LAPA
  *is* unweighted triangulation plus an <=8 mm residual, because its weighting is disabled
  by the §5.2 bug. The question the eval must answer is: **does CoTracker 2D + multi-view
  triangulation beat patch-matching 2D + depth sampling, against sim ground truth, under
  occlusion?** LAPA-vs-`triangulate` is then a second, narrower question about whether the
  learned weighting adds anything once fixed.
- **CoTracker alone may be most of the win**, with no LAPA, no clone, no checkpoint and no
  third camera. Sequenced first for exactly that reason.
- **Train/test distribution shift**: Y-up PointOdyssey with a 3-camera ring vs Z-up PyBullet
  with head+wrist. Mostly benign because everything is AABB-normalised, but it threatens
  `vis_logit` calibration and the frozen `BatchNorm1d` statistics.
- **Local GPU driver ceiling** (§3): CUDA 11.7 pins us to torch 2.0.1. Some newer CoTracker
  or DINOv2 code may require a newer torch, which would force either a driver upgrade or the
  remote server.
- **First run needs internet** (HF `facebook/dinov2-base` ~330 MB, torch.hub CoTracker3).
  Pin `TORCH_HOME`/`HF_HOME`, then set `HF_HUB_OFFLINE=1`.
- **Determinism**: tracking stays opt-in and the capture path is untouched when inactive, so
  `tests/golden/**` remain valid. GPU tests must `skipUnless` CUDA + clone + checkpoint.
