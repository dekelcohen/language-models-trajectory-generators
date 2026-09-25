# 3D Tracking for Robotic Manipulation

Status of this document: **the `pose_tracker` pipeline (§9) is built and measured on the
grasp and door scenes, in 2- and 3-camera configurations and under camera-visibility
scenarios (§11). LAPA was brought up, audited and dropped (§10, D1).** Every claim below is
marked as measured, source-verified, or untested. Nothing here is aspirational unless it
sits under a "TODO" heading. §3-§5 and §7.3/§7.4 are kept as the record of the LAPA
evaluation; they are no longer on the critical path.

---

## 0. Status board and how to resume

This section exists so the work can be put down and picked up cold. §7 is the detailed TODO
list, §9 the pipeline as built, §10 the decision log; this is the orientation.

### 0.1 Where things stand

| Area | State | Evidence |
|---|---|---|
| OpenGL<->OpenCV convention bridge | **done** | `tracking/camera_convert.py`, 24 tests |
| 3D provider seam + `depth_fusion` + `triangulate` | **done** | `providers/tracker3d/`, 24 tests |
| `rigid_refine` provider | **done** | `providers/tracker3d/rigid_refine.py`, 24 tests (§2.6) |
| **`pose_tracker` provider** (epipolar + jump gates, Kabsch, Kalman, green/yellow/red/black) | **done** | `providers/tracker3d/pose_tracker.py`, `tests/test_pose_tracker.py` (§9) |
| Rigid-body Kalman filter | **done** | `tracking/kalman.py` (§9.6) |
| Mask-grid seeding (affordance points first) | **done in the eval harness**; not yet wired into real rollouts | `tracking/seeding.py`, `tests/test_seeding.py` (§9.1) |
| Robot-mounted 3rd `shoulder` camera | **done, optional** (default is 2 cameras; opt in with `--track-cameras head wrist shoulder`) | `tests/test_shoulder_camera_pybullet.py` (§10, D6, D19) |
| Ground-truth eval harness + per-point / pose GT | **done** | `tests/tracking_eval.py` (§2.5, §11) |
| Door scene tracking + baseline | **done** | `tests/test_tracking_door_pybullet.py`, `DoorSceneDriver` (§11.3) |
| Camera-visibility scenarios (head-only, wrist-only, hand-over) | **done** | `tests/test_tracking_scenarios_pybullet.py` (§11.2) |
| One-row-per-seed 2D tracker contract | **done** (bug fix) | `providers/trackers/base.py` (§10, D17) |
| CoTracker 2D provider | **done, optional**; fp16 autocast on CUDA; 8-frame window is configurable (step 4 feasible). Not real-time on the A1000 with 2 cameras at 30 fps (§12) | `providers/trackers/cotracker_tracker.py` (§7.2.1, §12) |
| **Real-time model** (sim-time camera clock + virtual latency, drop/queue policy) | **done** - rollouts (`--track-camera-fps 30`, `--track-latency measured` by default) and the eval CLI | `tracking/realtime.py`, `tests/test_realtime.py` (§12) |
| KLT (Lucas-Kanade) 2D provider | **done, optional** (`--tracker-provider klt`); lost to `template` at 30 fps (§12) | `providers/trackers/klt_tracker.py`, `tests/test_klt_tracker.py` |
| Kalman filter in sim-time units | **done** (bug fix found by the real-time runs, D22) | `tracking/kalman.py`, `track_kf_ref_hz` |
| `tracking/motion.py` | **done** | 62 tests (§6) |
| LAPA | **dropped** (kept selectable as `--tracker3d lapa`, not maintained) | §3-§5, §10 D1 |
| Tracking video renderer | **done** | `tests/tools/render_tracking_video.py` (§0.4) |
| Door recovery after a long occlusion | **open** | §11.3 |
| Remote WebSocket server | **not started**, and no longer required (CoTracker runs locally) | §7.5 |

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
  tests\test_rigid_refine.py tests\test_tracking_eval.py tests\test_tracking_door_pybullet.py `
  tests\test_shoulder_camera_pybullet.py tests\test_pose_tracker.py tests\test_seeding.py `
  tests\test_transforms.py tests\test_cotracker_tracker.py `
  tests\test_tracking_scenarios_pybullet.py tests\test_realtime.py tests\test_klt_tracker.py -q
# eval tables (§2.5, §11, §12). Every 2D x 3D combination given is run; JSON -> outputs\tracking_eval\
# The CLI default is REAL TIME (30 fps camera, measured latency, motion 10 Hz; §12). Add
# --camera-fps 0 for the historical lock-step numbers of §11.
& $py tests\tools\run_tracking_eval.py --tracker3d depth_fusion pose_tracker --seeding grid
& $py tests\tools\run_tracking_eval.py --scene door --seeding grid --tracker3d pose_tracker
& $py tests\tools\run_tracking_eval.py --scenario handoff --seeding grid --tracker3d pose_tracker
& $py tests\tools\run_tracking_eval.py --cameras head wrist shoulder --seeding grid --tracker3d pose_tracker
& $py tests\tools\run_tracking_eval.py --provider template klt --seeding grid --tracker3d pose_tracker --camera-fps 0
# annotated MP4 (per-camera tiles, tracked points, estimate vs GT) - legend in §0.5
& $py tests\tools\render_tracking_video.py --scene door --seeding grid --tracker3d pose_tracker
```

Both tools share one CLI (`tracking_eval.add_eval_args`), grouped as *scene*, *trackers*,
*seeding*, *visibility* (`--occlusion`, `--scenario`, `--blind cam:start:end`), *timing*
(`--camera-fps`, `--latency`, `--motion-rate`) and *output*. Occlusion/blind windows are in
motion frames (10 per sim second by default) in both modes.
CoTracker runs need the GPU env and `$env:TORCH_HOME="$PWD\cache\torch"`.

**Known-good counts:** the suites above give **315 passed, 2 skipped** (measured on this
branch after §10 D21-D24).

**Pre-existing failures that are NOT caused by this work:** the *full* `tests/` suite has
**25 failures + 7 errors** (metaworld server, side-approach reachability). This was verified
by `git stash`-ing the tracking changes and re-running: identical counts before and after.
Do not try to fix them as part of this effort.

**Test-order pitfall:** the door env rewrites the robot pose (`config.base_start_*_franka`,
`config.joint_start_positions_franka`) and the head camera (`config.camera_*`,
`config.head_camera_*`) process-wide. The door test class restores them in
`tearDownClass`; any new door-booting test class must do the same, or a grasp-scene test that
runs later in the same pytest process boots with the door robot/camera (the wrist then sees
nothing: `0/20 predicted points depth-verified`).

### 0.5 Visualize / debug

**Render a video** (same harness and CLI as `run_tracking_eval.py`, so the video and the JSON
next to it come from one execution):

```powershell
& $py tests\tools\render_tracking_video.py --scenario handoff --seeding grid --tracker3d pose_tracker
& $py tests\tools\render_tracking_video.py --scene door --cameras head wrist shoulder --seeding grid --tracker3d pose_tracker --tag 3cam
# options: --fps N (default: the camera fps in real-time runs = real speed; 4 in lock-step)
#          --scale 2 (pixel upscale)  --png (also per-frame PNGs)  --tag X  --out DIR
#          --camera-fps 0 (lock-step)  --latency a1000_fp16  (see §12)
```

Output: `outputs/tracking_video/{scene}_{2D}_{3D}[_{tag}].mp4` + `.json` (+ `_png/`).
Drawing code: `tracking/visualize.py` (`draw_camera_overlay`, `compose_frame`).

**Legend** - one tile per camera, side by side, under a text banner:

| Mark | Meaning |
|---|---|
| small **filled green dot** | a tracked 2D point (a seed) the 2D tracker reports **visible** this frame |
| small **hollow orange ring** | a tracked point reported **not visible** (occluded, off-image, or a dead seed slot, D17) |
| **cyan cross** (thick) | the **estimate**: the fused 3D object point (`state.world_point`) reprojected into this camera |
| **red ring** (larger, thin) | **ground truth**: body pose + the seed centroid's body-frame offset, reprojected |
| **magenta line** cross -> ring | the 3D error as seen in this view; its pixel length is the reprojected error |
| **large dark disc** (not an overlay) | the synthetic occluder painted into the head camera's RGB (value 30) *and* depth (25 cm in front of the object), frames `--occlusion` [12, 19). It stands in for the arm crossing the line of sight (`tracking_scenario.paint_occluder`) |
| **amber tile border + "OCCLUDED"** | occlusion window, on a camera whose track status is occluded/lost/rejected/low_confidence |
| **fully black tile** | camera blinded by the `--scenario` / `--blind` schedule (black RGB, invalid depth) |

Text on each tile: `<cam>: <status> conf=.. health=.. vis=<n visible>` and, when it happened,
`reseed: <reason>`. Status values: `ok`, `unseeded` (never seeded - waiting to join via the
pose, §9 step 0.5), `rejected` (its lift failed a gate this frame), `lost`, `occluded`,
`low_confidence`.

Banner: `frame NNN  tracking|LOST  err=<mm>  disagree=<mm between cameras>  OCCLUSION`, then
`lift: ...` (the first 6 keys of the 3D provider's meta, e.g. `provider`, `dt`, `jump_gate_m`,
`rejected_jump`, `epipolar_worst_px`, `n_fused`, `mode`), then `<scene> | 2D=.. 3D=..`, and in
real-time runs `t=<sim s>  estimate from frame <k>  age=<ms>`: the markers are the latest
*published* estimate (computed on camera frame k) drawn over the *current* frame, so any lag
is visible as the cross trailing the red ring.
The banner turns red when the object is lost.

**How to read it:**

* short or no magenta line = good. Green dots on the object while the cross slides away from
  the red ring = the 3D stage is **coasting** on the Kalman prediction (red/black mode), not
  measuring - check `mode` in the banner.
* green dots sliding off the object while still green = 2D drift the gates did not catch
  (the door handle, §11.3).
* orange rings clustering = the tracker itself gave up on those points; if the whole camera
  goes `rejected`, the depth-jump or epipolar gate threw its lift out.

**Per-frame numbers without video** - every eval/video JSON has `frames[obj][i]` with
`error_m` (centroid), `point_err_m`, `pose_err_m`, `cam_status`, `blind_cams`, `occluded`
and `lift` (the full provider meta: `mode`, `pose_reseed` per camera with its reason such as
`skip: 5/20 predicted points depth-verified`, `rejected_epipolar`, `rejected_jump`,
`unmeasured_frames`). Quick dump:

```powershell
@'
import json; d = json.load(open(r"outputs\tracking_eval\door_template_pose_tracker_scn5_3cam.json"))
for i, f in enumerate(next(iter(d["frames"].values()))):
    print(i, f["error_m"], f["pose_err_m"], f["lift"].get("mode"), f["cam_status"], f["lift"].get("pose_reseed"))
'@ | & $py -
```

**Logs:** rollouts write `track.jsonl` + `summary.json` under `--track-log-dir`; add
`--track-save-depth` to dump the depth array behind each decision. Eval logs use the
`[eval]` / `[video]` prefixes; `pose_tracker` decisions are logged at DEBUG.

**Debugging pitfalls seen so far:**

* harness runs used to differ run-to-run (arm sag); fixed by D20 - if numbers drift between
  identical runs again, check `restore_robot()` is still called in `reset()`;
* the centroid error alone can look fine with wrong correspondences - always check
  `point_err_m` / `pose_err_m` too (D15, D17);
* formatting floats with `!s:.6` truncates scientific notation (`4.8393e-05` prints as `4.8393`).

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
                          weighted_triangulate  DLT with per-view confidence weights
                          rigid_refine   wraps any of the above, adds a rigid-body
                                         constraint (section 2.6)
                          pose_tracker   per-point lift + gates + Kabsch 6-DoF pose +
                                         Kalman filter + occlusion hierarchy (section 9)
                          lapa           evaluated and dropped (section 10, D1)
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

**The 2D trackers must not drop rows either** (found later, §10 D17). `template`, `csrt`
and `cotracker` used to silently skip seeds they could not track (patch crossing the image
border, featureless patch), so the tracker returned fewer rows than `point_index` had
entries and `_indices_for` fell back to positional order - the same wrong-correspondence
failure one layer down. The `PointTracker` contract is now **one output row per seed, in
seed order**; an untrackable seed is a never-visible slot, and `_install_seed` refuses a
tracker that breaks the contract.

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

### 5.2.1 Two cameras cannot detect a bad track *from the reprojection residual*

> **Correction (later).** The original title of this section said "at all", which is wrong.
> The DLT reprojection residual is blind at 2 views, but the **epipolar constraint** is not:
> a drift *across* the epipolar line is directly measurable (1 dof per point) and is gated
> in `pose_tracker` (§9.3). Only drift *along* the line is invisible - and the 47 px example
> below was an along-line drift. The rigid-body fit (§9.5) and the depth jump gate (§9.4)
> are the other two checks that work at 2 cameras.

Independently of any bug: with exactly 2 views the DLT system is **exactly determined**
(4 equations, 4 homogeneous unknowns). Any pair of pixels therefore triangulates to a point
that reprojects almost perfectly. Measured: drifting one view by 47 px moves the 3D point
by >1 cm while the reprojection residual stays at **0.18 px**.

So at 2 cameras the reprojection residual - the signal both LAPA's IRLS *and* its
`view_weight_head` consume - carries **no information**, and no *residual-based* weighting
scheme, learned or otherwise, can work. Residual-based robustness begins at 3 views. Both
facts are pinned by `tests/test_tracker3d.py::TestTriangulateIrls`.

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
- [x] Same baseline on the door scene (§11.3).
- [x] Door affordance scene test: `tests/test_tracking_door_pybullet.py` seeds from
      `{"point": [x, y], "label": "door_handle"}`, drives the hinge/latch joints and scores
      against the latch link pose; `DoorSceneDriver` runs it through the eval harness.
- [x] Third camera: the robot-base-mounted `shoulder` camera (`--track-cameras head wrist
      shoulder`, §10 D6).
- [x] `pose_tracker` pipeline (§9), rigid-body Kalman filter, mask-grid seeding.
- [x] Camera-visibility scenarios + per-point / pose ground truth (§11).
- [x] One-row-per-seed tracker contract (§10 D17).
- [ ] **Door recovery after a long occlusion** (§11.3): stays red, drifts ~4 mm/frame.
- [ ] **Grid seeding in real rollouts**: `track_objects` still seeds from detection world
      points; needs the segmentation mask plumbed through (and a Genesis segmentation
      decoder - PyBullet encodes `uid + ((link + 1) << 24)`, Genesis is not handled).
- [ ] Back-fill a CoTracker flush's whole window into the Kalman filter (today only the
      newest frame of a flush is a measurement).
- [ ] Template tracker scale drift at close range (wrist closing on the object, §11.2):
      rescale templates by the depth ratio, or prefer CoTracker for the wrist.

### 7.2 CoTracker

- [x] `providers/trackers/cotracker_tracker.py` as a **public** `PointTracker`, usable with
      today's depth fusion with **no LAPA clone and no checkpoint**. Registered in
      `factory.SUPPORTED`, exposed as `--tracker-provider cotracker`, 24 fake-model tests
      plus a real-checkpoint test gated on `COTRACKER_REAL_TEST=1`. Weights (97 MB) cache
      into `cache/torch` via `TORCH_HOME`. Verified end to end on `cuda:0` and on CPU.
- [x] Validate `cotracker3_online`. **Done, and it found a problem - see §7.2.1.**

### 7.2.1 The `cotracker3_online` latency granularity problem

> **Update (later measurements, supersedes the latency table below).** The 1.95 s/flush
> figure was the first, unoptimised fp32 run. Measured since on the same RTX A1000 (4 GB,
> torch 2.0.1+cu117): **0.223 s per flush in fp16** at the default window 16 / step 8, and
> step 4 (window 8, a new estimate every 4 frames) was measured to be feasible. The provider
> now runs fp16 autocast on CUDA (`config.tracker_cotracker_fp16`). The window
> is a constructor argument (`step=`), not baked into the checkpoint. **Real-time behaviour
> (2 cameras, one flush per camera per 8 frames) is measured in §12**: it does not keep up at
> 30 fps on the A1000.
> `stale_frames` **is now consumed**: `pose_tracker` enters `coast` mode on stale frames
> (Kalman prediction, no new measurement, no false "jump") - option 3 below, done. The
> sawtooth error profile described below is real and was observed in the rendered videos.
> Measured CoTracker accuracy in our scenes is in §11 - on these textureless sim objects it
> is *worse* than the template tracker, so it stays optional (§10, D2).

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

> **Dropped (§10 D1).** Kept for the record; nothing below is planned.

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

- [x] `document.md` §13 (Rollout tracking): `pose_tracker`, `--tracker3d`,
      `--track-cameras`, seeding, the shoulder camera, eval/video tools.
- [x] This plan: §9 pipeline as built, §10 decision log, §11 scenarios and results.
- [ ] Changelog entry.

---

## 8. Risks and open questions

**Current (after dropping LAPA, §10):**

- **Door tracking** (§11.3): the handle's points slide ~1.7 mm/frame while still green, then
  re-acquire after the occlusion fails (2 cameras) or locks in a wrong rotation (3 cameras).
- **Template tracker scale drift at close range** (§11.2): the wrist is rejected from ~frame
  24 in `handoff`; the KF coasts at ~1 cm. Multi-scale templates or CoTracker would help.
- **2-camera blind spot** (§5.2.1, §9.3-9.5): drift *along* the epipolar line that stays
  consistent in depth and rigid is not detected. The optional 3rd camera (D6) closes it.
- **Rollouts still seed from detection points**; grid seeding is harness-only (§7.1).
- **CoTracker numbers are pre-D17** and need re-measuring on the GPU env.

**Historical (LAPA era, kept for context):**

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
- **Frame-counted session counters (open, §12.5).** The Kalman filter and pose modes are in
  sim time (D22), but `track_pose_reseed_cooldown`, the session's re-seed/lost patience and
  monitor patience still count *frames*, so at 30 fps they fire 3× sooner in seconds than at
  the 10 Hz they were tuned at. Convert them to seconds if real-time runs show early re-seeds.
- **Door handle drift (open).** Every local 2D tracker (template, KLT, CoTracker) slides on
  the thin, textureless, rotating handle; real time does not change this (§11, §12.3).

---

## 9. `pose_tracker`: the point-tracking pipeline as built

`--tracker3d pose_tracker` (`providers/tracker3d/pose_tracker.py`) implements the
"CoTracker3-style joint-point tracking + SVD pose + Kalman" design step by step. The 2D stage
is any `PointTracker` (`template` by default, `cotracker` optional); everything below is 3D.
All thresholds live in `config.py` (`track_pose_*`, `track_kf_*`, `track_seed*`).

| Design step | Implemented as | Status |
|---|---|---|
| 0.1 detect + N points + lift (PC_1) | `tracking/seeding.mask_grid_seed`: farthest-point grid over the eroded instance mask, affordance points first, lifted through the seeding camera's depth | **done in the eval harness**; rollouts still seed from detection points (§7.1) |
| 0.2 project PC_1 into cam 2 | `TrackingSession._reseed_from_pose` projects the template through the current pose | **done** |
| 0.3 visibility + depth check | `geometry.is_visible` per point (in frustum, depth agrees) | **done** |
| 0.4 cross-camera appearance check | - | **not done** (the depth check has sufficed in sim; revisit for real cameras) |
| 0.5 "repeat until cam 2 verifies" | `health.decide_reseeds` retries every frame for an `unseeded` camera, with a cooldown; it joins once >= `join_min_frac` (25 %) of the projected template is depth-verified (D19) | **done**, tested by the `handoff` scenario (§11.2) |
| 0.6 template + KF init | template = the seed-time world points of **one** camera (no PC_1/PC_2 averaging; see D9); KF initialised on the first fit | **done** |
| 1.1 2D tracking with visibility | `PointTracker.update` -> points, visible, scores; one row per seed (D17) | **done** |
| 1.2 epipolar gate | `epipolar_distance`, reject > `track_pose_epipolar_px` (3 px) | **done** (catches across-line drift only, §5.2.1) |
| 2 per-point lift + depth-jump gate | per-camera deprojection; reject a point > `jump_m` (5 cm) + KF uncertainty (cap 25 cm) from its predicted position | **done** |
| 3 point fusion by seed index | weighted `confidence / z^2`; on disagreement > `fuse_tol_m` keep the reading nearer the prediction; single-camera points pass through | **done** |
| 4 SVD pose (Umeyama) | Kabsch **without scale** (`motion.fit_rigid_motion`) vs the template, iterative threshold trimming (`inlier_m`, `inlier_median_k`) | **done** |
| 5 green | fit RMSE < `green_rmse_m` (1.5 cm), inliers >= 50 % -> KF update, measured pose | **done** |
| 5 yellow | per-camera fits; the good camera's pose updates the KF and the other cameras are re-seeded from it (`meta['correct_cams']`) | **done** |
| 5 red | KF prediction (`predicted=True`); re-acquire = re-seed every camera where >= 50 % of the predicted points are depth-verified | **done** |
| 5 black | red for > `black_after` (8) frames; if the object was grasped the pose follows gripper FK (`source="fk"`) | **done** |

### 9.1 Seeding

`--seeding affordance` (historical) puts 5 points at one pixel cross; the pose is then
translation-only because rotation is unobservable from a 4 px cluster. `--seeding grid`
spreads `track_seed_points` (20) over the eroded instance mask by farthest-point sampling,
starting from the affordance points so they are always tracked. A rotation is fitted only
when the points span >= `min_rot_extent_m` (2 cm) and are not collinear
(`meta['rotation_observable']`).

### 9.2 Pose convention

The pose is **relative to seed time**: `current_i = R @ seed_i + t`. There is no object model,
so "identity" means "as it was when detection ran". Absolute orientation needs an object frame
from detection/CAD - out of scope.

### 9.3-9.5 Gates and fit

See the table; the reasoning for each threshold is in `config.py` next to it. The three
checks that work at **2 cameras** are the epipolar gate (across-line drift), the depth jump
gate (occluder in front of one point) and the rigid fit (a point that does not move with the
others). None of them sees drift *along* an epipolar line that is also consistent in depth
and rigidly plausible - that is the residual risk at 2 cameras.

### 9.6 Kalman filter

`tracking/kalman.py`, one filter per object: constant velocity in position and rotation,
12-D error state `[p, v, theta, omega]` with the rotation folded into a reference matrix after
every update (no Euler wrap). Mahalanobis gate at chi^2(6) 99.9 %, but `reset_after` (2)
consecutive agreeing rejections re-initialise it, so a real abrupt change (a dropped object)
is followed instead of rejected forever. `max_coast` (30) predictions without an update and it
gives up.

### 9.7 CoTracker staleness

A CoTracker frame without a fresh flush (`meta['stale_frames'] > 0`) puts the provider in
`coast` mode: KF prediction, no measurement, no jump/re-seed decision on frozen pixels.

### 9.8 What reaches the monitors

Monitors ignore any pose that is `predicted` (red/black/coast) or `source="fk"`. A coasting
or FK-following pose must never be read as evidence that the object is in the gripper.

---

## 10. Decision log

| # | Decision | Why (evidence) |
|---|---|---|
| D1 | **Drop LAPA.** Kept selectable, not maintained. | Its 3D output is unweighted DLT + <=8 mm (§4-§5); the learned view weighting is saturated (+6.91 logit for every view); fixing the DLT alone makes it worse; DINOv2 is 99 % of its compute and needs a GPU. No measured gain over our own triangulation/pose pipeline. |
| D2 | **CoTracker3 stays an optional 2D provider**, not the default. | Runs on the A1000 4 GB (0.223 s/flush fp16), 3070 8 GB and a Kaggle T4. On our textureless sim objects it is *worse* than the template tracker (grasp 0.048 vs 0.003 m median with `pose_tracker`); expected to matter more on real, textured scenes. Window step is configurable (8 default, 4 measured feasible). |
| D3 | **Real cameras are RGB-D**, so the 3D stage lifts through depth per point; triangulation is a cross-check, not the primary lift. | Depth gives a 3D point per camera per point, which makes single-camera tracking and the depth jump gate possible. |
| D4 | **Seeding = mask grid, affordance points first** (§9.1). | 5 points at one pixel make rotation unobservable; 20 spread points make the rigid fit meaningful and survive partial occlusion. |
| D5 | **Kabsch without scale, not ICP.** | Correspondence is known by seed index, so the closed-form fit is exact and deterministic; the object does not change size, so a free scale would only absorb error. |
| D6 | **A robot-mounted `shoulder` camera as an *optional* 3rd view.** The default stays 2 cameras (`config.tracking_cameras = ("head", "wrist")`); opt in with `--track-cameras head wrist shoulder` (rollouts) or `--cameras head wrist shoulder` (eval). Nothing in the pipeline assumes 3 cameras - `pose_tracker` runs on any subset down to one (the `head_only`/`wrist_only` scenarios). | A 3rd view makes residual-based checks possible (§5.2.1) and gives a view the arm rarely blocks. Measured after D17/D19 (template): grasp 0.004 / 0.011 -> 0.003 / 0.004 m (2 cameras already enough). Door centroid 0.047 / 0.137 -> 0.029 / 0.122 m, **but the door pose error gets worse** (0.049 -> 0.083 m): the shoulder re-acquires the position after the occlusion with a wrong rotation (§11.3). With CoTracker (pre-D17) mixed. |
| D7 | **Rotation only with extent >= 2 cm and non-collinear points.** | Below that the rotation is fitted noise and corrupts the pose. |
| D8 | **Never trust a fit on fewer than `min_points` (3).** | A 1-point fit is trivially perfect (RMSE 0) and was reported green. |
| D9 | **Template from one camera; no PC_1/PC_2 averaging at t=0.** | The second camera usually cannot verify all points at t=0; waiting would delay tracking. A camera that joins later is seeded *from the template through the pose*, so its points are the same physical points by construction. |
| D10 | **Threshold trimming inside the fit** (`fit_rigid_motion(trim_m, trim_median_k)`). | One slid point otherwise drags the whole pose; trimming at max(`inlier_m`, k x median residual) removes it. |
| D11 | **"Degraded yellow" tried and reverted.** | Accepting a weak single-camera fit as yellow made the door worse (2.4 -> 5.6 cm median). |
| D12 | **Coast mode for stale CoTracker frames** (§9.7). | Frozen pixels are not a measurement; treating them as one caused false jumps and sawtooth error. |
| D13 | **Monitors ignore predicted and FK poses** (§9.8). | Otherwise a coasting estimate would "prove" the object is still attached after a drop. |
| D14 | **FK coasting in black when grasped.** | During a grasp the arm occludes everything; the object moves with the gripper, so FK is the best predictor - flagged, never trusted as a measurement. |
| D15 | **Ground truth = body pose + the body-frame offset of each seed point.** Adds per-point and pose error next to the centroid error (§11.1). | The centroid metric alone rewarded wrong correspondences by luck (D17). |
| D16 | **Visibility scenarios are blackout schedules** (`tracking_eval.blind_schedule`). | Head-only, wrist-only and hand-over need no new scenes - a blind camera renders black RGB and invalid depth. |
| D17 | **2D trackers keep one output row per seed** (`PointTracker` contract); stateful 3D providers re-seed through the pose (`_reseed_from_pose`), not through surface snapping. | Found by the hand-over scenario: the wrist was seeded with 18 points, the template tracker silently kept 13, and the lift matched them to template ids 0-12. The rigid fit then sat at 2-5 cm RMSE (red/black). Snapped re-seed points are also not the template's physical points (point error 7 cm). After the fix: hand-over p95 0.10 -> 0.014 m, wrist-only 0.35 -> 0.046 m. |
| D18 | **Configs centralised in `config.py`; rotation helpers in `sim_adapter/transforms.py`; one tracking CLI** (`tracking.session.add_tracking_args`, used by `main.py` and the Genesis env) with the new `--tracker3d` and `--track-cameras`. | Duplicated defaults had already drifted (the `pose_tracker` read keys that did not exist in `config.py`). |
| D19 | **A never-seeded camera joins with a lower bar** (`track_pose_join_min_frac` 0.25) than re-acquire (`track_pose_reseed_min_frac` 0.5); both floored at `track_pose_min_points`. Every installed point is still individually depth-verified. | After D17 the shoulder camera never joined on the door (it verifies 5-8 of 20 handle points; 50 % needs 10), so 3-camera door = 2-camera door exactly. With 0.25: 3-camera door centroid 0.043 / 0.133 -> 0.029 / 0.122 m (pose error 0.058 -> 0.083 m, see §11.3); grasp scenarios unchanged. Re-acquire keeps 0.5 because the occluder may still be in front. |
| D20 | **The eval harness restores the robot joints on `reset()`** (`SimSceneDriver.restore_robot`). | The arm sagged ~0.1 mm per run, so repeated runs in one process gave 3-7 mm medians for the same config; now bit-identical. All §11 numbers are from after this fix. |
| D21 | **Real-time model: a sim-time camera clock plus virtual latency** (`tracking/realtime.py`), used by rollouts and the eval CLI. A result is published at `frame_time + latency`; per-frame trackers *drop* frames while busy, CoTracker *queues* them. Latency = measured tracking wall time (rendering excluded), `zero`, or a device profile. | The sim used to pause the world while the tracker computed, and rollouts only tracked on motion-gated VLM keyframes (≤5 fps, nothing while the arm is still) - so a ball dropped while holding still was never seen, and any tracker looked instant. Alternatives rejected: a threaded tracker racing a wall-clock sim (non-deterministic, depends on host load) and a fixed track period (hides the device's real cost). |
| D22 | **The Kalman filter steps in sim time**: `dt = elapsed × track_kf_ref_hz` (10 Hz, the rate its noise was tuned at); coast budget, reset patience and `black_after` are in the same units. | Found by D21: stepping `dt = 1` per frame at 30 fps tripled the modelled acceleration per second; good measurements fell outside the gate and the pose coasted off (grasp wrist-only p95 0.09-0.39 m, hand-over 10 % lost). With time units: 0.009 / 0.020 m, 0 % lost. Lock-step (no clock) is bit-identical. |
| D23 | **`template` stays the default 2D tracker; `klt` and `cotracker` are options.** CoTracker runs fp16 on CUDA. | At 30 fps real time (§12.3) template beats KLT on every grasp scenario and the 2-camera door (KLT drifts more with 3× more updates; it only wins the 3-camera door). CoTracker on the A1000 cannot keep up with 2 cameras at 30 fps (results age to 1.5-2 s); at 10 fps it keeps up but is less accurate than template here. Its advantages (learned features, joint tracking, re-finding points after occlusion, robustness on real textures) are real but do not show on these flat-shaded sim objects at this GPU budget. |
| D24 | **The eval library default stays lock-step; the eval CLI default is real time.** | Every existing test and §11 number keeps its meaning (`run_eval` with no timing kwargs = the historical loop, and camera fps = motion rate with zero latency reproduces it exactly, `test_realtime.py`), while ad-hoc runs show what a robot would see. |

---

## 11. Evaluation scenarios, ground truth and results

### 11.1 What is scored

`tests/tracking_eval.py` scores three errors per frame against simulator ground truth:

* **centroid** (`median_l2_m`, `p95_l2_m`) - the fused world point vs the body pose plus
  the seed centroid's body-frame offset (constant in the *body* frame, so rotation does
  not fake error);
* **per point** (`median_point_err_m`) - every tracked point's own depth lift vs its own
  seed point `GT_i = R_gt @ local_i + p_gt`. This is the one that exposes wrong
  correspondence;
* **pose** (`median_pose_err_m`) - the mean distance between the template moved by the
  estimated pose and the true points.

A lost frame is penalised (1 m), not dropped.

### 11.2 Camera-visibility scenarios (grasp scene, 30 frames, disc occluder on the head for frames 12-19)

`template` 2D, grid seeding (20 points), head + wrist (2 cameras). Measured after D17/D19/D20:

| Scenario | `depth_fusion` median / p95 / lost | `pose_tracker` median / p95 / lost |
|---|---|---|
| `occlusion` (both cameras) | 0.033 / 0.057 / 0 % | 0.004 / 0.011 / 0 % |
| `head_only` | 0.042 / 0.266 / 0 % | **0.003 / 0.006** / 0 % |
| `wrist_only` | 0.057 / 1.0 / 10 % | **0.011 / 0.044** / 0 % |
| `handoff` (wrist seeded from the head at frame 10, head blind from 15) | 0.038 / 1.0 / 10 % | **0.009 / 0.013** / 0 % |

Pinned by `tests/test_tracking_scenarios_pybullet.py`. Known limit: at close range the
wrist's fixed-size NCC templates drift under the scale change (from ~frame 24 in `handoff`),
the jump gate rejects the wrist, and the pose coasts on the Kalman filter - still ~1 cm, not
lost. Fix candidates in §7.1.

### 11.3 Grasp and door, provider comparison (median / p95, m)

`grid` seeding, disc occluder on. Rows marked **current** are measured after D17, D19 and D20
(the code as it is now); the other rows are historical (before D17) and kept for comparison.

| Config | Grasp | Door |
|---|---|---|
| template + depth_fusion, **current** | 0.033 / 0.057 | 0.037 / 0.065 |
| template + pose_tracker, **current** (2 cameras, the default) | 0.004 / 0.011 | 0.047 / 0.137 |
| template + pose_tracker, 3 cameras, **current** | 0.003 / 0.004 | 0.029 / 0.122 |
| template + pose_tracker (before D17) | 0.003 / 0.050 | 0.024 / 0.088 |
| template + pose_tracker, 3 cameras (before D17) | 0.004 / 0.005 | 0.033 / 0.051 |
| cotracker + depth_fusion (before D17) | - | 0.052 / 0.180 |
| cotracker + pose_tracker (before D17) | 0.048 / 0.124 | 0.047 / 0.176 |
| cotracker + pose_tracker, 3 cameras (before D17) | 0.030 / 0.104 | 0.091 / 0.137 |

CoTracker rows have not been re-measured after D17 (needs the GPU env); do that before
drawing CoTracker conclusions.

**2 vs 3 cameras.** The 3rd camera is optional (D6). On the grasp 2 cameras are already at
~4 mm median; the 3rd view mainly tightens p95. On the door the 3rd view improves the
*centroid* (median 0.047 -> 0.029 m) but not the *pose* (0.049 -> 0.083 m): after the
occlusion it re-acquires with the right position and a wrong rotation (frame 21: centroid
0.018 m, pose 0.088 m, then growing). More cameras do not fix the door.

**The door already drifts before the occlusion.** In both rigs the error grows ~1.7 mm/frame
from frame 0 while the mode is still green (0.016 m at frame 9): the fit tolerances
(`green_rmse_m` 1.5 cm) absorb a steady slide of the template-tracked points on the rotating,
foreshortening handle. That, not only the re-acquire bar, is the root of the door problem.

**Door after D17.** The centroid median moved 0.024 -> 0.043 m, while the pose error improved
(0.082 -> 0.058 m): the old code had the right position with the wrong correspondences and a
wrong rotation. Both versions fail the same way after the occlusion (frame 19 onwards): the
head's points slid during the occlusion, fits sit at 2-5 cm RMSE (red), and re-acquire needs
>= 50 % of the predicted points depth-verified, which the thin handle never reaches - so the
pose coasts and drifts ~4 mm/frame. This is the main open item (§7.1).

---

## 12. Real-time model (D21-D24)

### 12.1 Why

Before D21 the simulator stopped while the tracker ran, and rollouts tracked only on the
motion-gated VLM keyframes (at most ~5 fps, none while the arm holds still). Every tracker
therefore looked instant, and an object that fell while the arm was still was never observed.
A real robot's camera keeps running and the world keeps moving while the tracker computes.

### 12.2 Design (`tracking/realtime.py`)

- **Camera clock in sim time.** A frame is captured every `1 / track_camera_fps` sim seconds
  (default 30), starting at the first poll. Rollouts poll every physics step
  (`robot._tick_tracking`); the harness polls at the camera rate while the scene moves at
  `--motion-rate` (10 Hz, so occlusion/blind windows keep their meaning).
- **Virtual latency.** A result computed on the frame at time `t` is *published* at
  `t + latency`; until then the robot, monitors and scores see the previous estimate.
  `track_latency_mode`: `measured` (the tracker's own wall time on this machine, render time
  excluded; CoTracker's first model warm-up flush is not charged), `zero`, or a profile name
  from `track_latency_profiles` (e.g. `a1000_fp16` = 0.223 s per CoTracker flush, `cpu`).
- **Busy policy.** Per-frame trackers (template, KLT) *drop* frames that arrive while a
  result is still pending. Window trackers (`track_buffering_providers`, i.e. CoTracker)
  *queue* them, because every frame is part of the next window.
- **Deferred commit.** `TrackingSession.tick()` computes immediately but commits (updates
  `state`, runs the monitor, may abort) at publish time; `drain()` flushes at the end.
  Reports carry `sim_time`, `camera_frame`, `available_at`, `latency_s` and timing.
- **Metrics.** The harness scores the latest published estimate against the ground truth at
  the *current* time: `age_s` (median/p95/max) and `s_to_recover` are new columns. Frames
  before the first publish are warm-up and are not scored.
- **Render cost** (not charged as latency, but it is real on a robot's host): ~43 ms per
  camera frame in PyBullet, so a 30 fps, 2-camera run takes ~2.6 s of wall time per sim second
  (3 cameras ~3.9 s).

### 12.3 Results

pose_tracker, grid seeding, 2 cameras (door 3-cam where noted), occlusion frames 12-19,
median / p95 pose error in metres. Real time = 30 fps camera, measured latency.

| Scenario | template lock-step | template 30 fps | KLT lock-step | KLT 30 fps |
|---|---|---|---|---|
| grasp occlusion | 0.0042 / 0.0114 | 0.0056 / 0.0088 | 0.0048 / 0.0100 | 0.0152 / 0.0226 |
| grasp wrist-only | 0.0114 / 0.0436 | 0.0045 / 0.0090 | 0.0008 / 0.0116 | 0.0052 / 0.0186 |
| grasp hand-over | 0.0087 / 0.0128 | 0.0077 / 0.0200 | 0.0051 / 0.0110 | 0.0155 / 0.0210 |
| door | 0.047 / 0.137 | 0.0371 / 0.0702 | 0.041 / 0.084 | 0.0620 / 0.1581 |
| door, 3 cameras | - | 0.0575 / 0.1496 | - | 0.0495 / 0.0894 |

Template/KLT latency is 7-19 ms per frame on the CPU: no frames dropped and the estimate age
is one camera period (33 ms). KLT (door kwargs: window 11, forward-backward 0.5 px) wins in
lock-step but drifts more at 30 fps, where it takes 3× more incremental steps.

CoTracker3 on the RTX A1000 (fp16, measured latency, same scenes; door without occlusion):

| Camera fps | grasp | door | estimate age |
|---|---|---|---|
| 30 | 0.082 / 0.139 | 0.107 / 0.20 | up to 1.5-2.1 s (queue grows 3-4 s) |
| 10 | 0.030 / 0.110 | 0.064 / 0.092 | max 0.5-0.6 s |
| lock-step (no latency) | - | 0.0215 / 0.041 | - |

fp32 was worse (grasp 0.126 at 30 fps). Two cameras cost two flushes per 8 frames
(~0.45 s of GPU per 0.27 s of video), so the queue never drains at 30 fps.

### 12.4 Findings

- **The Kalman filter was tuned in frames, not seconds** (fixed, D22). At 30 fps the first
  runs were *worse* than lock-step (hand-over 9 % lost, KLT wrist-only p95 0.37 m) because
  the motion model tripled its acceleration per second and rejected good measurements. With
  sim-time units, 10 fps real time is bit-identical to lock-step and 30 fps is as good as or
  better.
- **CPU is enough in sim.** Template at 30 fps on the CPU has ~33 ms age, so "the ball is
  away from the gripper" is visible within a couple of frames, far inside the 0.5 s budget.
- **Why CoTracker3 at all.** Template and KLT are local appearance matchers: they drift under
  scale, rotation and foreshortening and cannot re-find a point after it was hidden.
  CoTracker3 tracks all points jointly with learned features over a temporal window, predicts
  visibility and re-finds points after occlusion, and is far more robust on real textured
  scenes. Here the objects are flat-shaded and the A1000 is too small for 2 cameras at
  30 fps, so the advantage does not show. It stays an option for the real robot, with one of:
  a stronger GPU, batching both cameras in one forward pass, window 8 / step 4 (needs a
  state-dict adjustment, not implemented), or a <= 10 fps feed.

### 12.5 Open items

- Session re-seed cooldown / lost patience / monitor patience are still in frames (§8).
- CoTracker: batch the cameras in one forward pass; window 8 / step 4.
- Grid seeding is used by the harness but not wired into rollouts.
- Door handle drift (§8, §11).

### 12.6 Videos

`tests\tools\render_tracking_video.py` plays real-time runs at real speed (playback fps =
camera fps) with a `t / estimate from frame / age` line (§0.5). Rendered with tags
`rt30_*` (template, KLT) and `rt10_gpu` / `rt30_gpu` (CoTracker) in `outputs\tracking_video\`.
