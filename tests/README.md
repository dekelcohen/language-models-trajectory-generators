# Tests

## Interpreters

| Suite | Conda env | Notes |
|---|---|---|
| PyBullet suites (everything below today) | `vlm_traj` | `C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe` |
| Genesis suites (`test_genesis_*.py`, planned) | `vlm_genesis` | Genesis needs `Pillow>11`, `numpy>=2`, `mujoco`, `numba`, ... which collide with this repo's pinned `requirements.txt` (`Pillow==10.1.0`, `numpy==1.26.2`); it therefore lives in its own env and runs as a child process. |

`tests/test_genesis_launcher.py` runs under `vlm_traj` and does **not** require Genesis.

### Selecting the Genesis interpreter

`providers/genesis_launcher.py` is the single place that decides which interpreter runs the
Genesis child, mirroring how `METAWORLD_PYTHON` works — but stricter (see below).

| Env var | Default | Meaning |
|---|---|---|
| `GENESIS_PYTHON` | *(unset)* | Absolute path to the interpreter. Wins over everything. |
| `GENESIS_CONDA_ENV` | `vlm_genesis` | Name of the conda env to auto-discover. |
| `GENESIS_HOST` | `127.0.0.1` | Host the child serves IPC on. |
| `GENESIS_PORT` | `8770` | Port the child serves IPC on. |

Resolution order: `GENESIS_PYTHON` → conda env named `GENESIS_CONDA_ENV` → the current
interpreter, but *only* if it can actually `import genesis`. Otherwise it raises
`GenesisInterpreterNotFound` with the list of directories searched and the exact commands
to fix it.

Unlike `METAWORLD_PYTHON` — which falls back to `sys.executable` — an explicitly configured
but missing `GENESIS_PYTHON` is a hard error. Falling back would launch Genesis under the
main app's env, where `import genesis` fails inside the child with a traceback that hides
the real cause.

Check what it resolves to:

```powershell
& ...\vlm_traj\python.exe -m providers.genesis_launcher
```

`--sim genesis` is now an explicit branch in `agent_runner.init_agent`; previously any
unrecognised `--sim` value silently fell through to the obsolete Metaworld path.

Always run from the repository root — the sim-envs load URDFs via relative paths.

```powershell
cd D:\NLP\Robotics\VLM_Robotics\lmtg_genesis
& C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe -m pytest tests\test_pybullet_regression.py -q
```

## `test_pybullet_regression.py` — the refactor safety net

Pins the numeric behaviour of `env.py` / `robot.py` / the sim-env profiles **before** the
sim-adapter refactor, so the refactor has to be behaviour-preserving.

* Boots `grasp` and `door` headlessly through the production classes (`env.Environment` +
  `robot.Robot`) and settles for a fixed `SETTLE_STEPS`.
* Runs a scripted sequence (camera captures, pinhole round-trip, gripper close/open, a
  3-point trajectory) with `debug/trace_utils.py` tracing enabled.
* Compares the trace against `tests/golden/pybullet/<task>.jsonl` at **rtol=0, atol=1e-9**.

What the golden covers: head + wrist `viewMatrix` / `projectionMatrix` (values **and**
shapes), camera position/orientation, `znear`/`zfar`, the full 3D→2D→3D round trip through
the production `utils` functions, EEF pose/euler/quaternion, joint positions/velocities/
applied torque, `simenv.get_state()`, the wrist-camera params, and every call/return of
`Robot.move`, `Robot.get_camera_image` and `Robot.step_env_and_record`
(~700 records per task).

Regenerate after an *intentional* behaviour change:

```powershell
$env:LMTG_UPDATE_GOLDEN = "1"
& ...\vlm_traj\python.exe -m pytest tests\test_pybullet_regression.py -q
$env:LMTG_UPDATE_GOLDEN = "0"
```

### Determinism requirements

The test is in-process on purpose. `run_simulation_environment` calls `env.update()` in a
free-running loop, so how far the physics advances between two IPC messages depends on
wall-clock timing — a subprocess trace can never be bit-reproducible.

Two sources of non-determinism are neutralised:

1. **`config.RANDOM_TARGET_GRASP_OBJ_POSE`** (`config.py:39`) randomises the grasp object's
   spawn pose at *import* time (±0.2 m in x, 0.4–0.8 m in y, ±π yaw). The test pins it to
   config's own documented fixed values. Production is untouched.
2. **Timestamps** are omitted from traces by default (`debug/trace_utils.py`); set
   `LMTG_TRACE_TIME=1` only for interactive profiling.

Verified reproducible across three consecutive runs at `atol=1e-9`.

## Tracing

`debug/trace_utils.py` is off unless `LMTG_TRACE=<path>` is set, so there is no permanent
hot-loop logging cost. It emits JSON lines (`kind`, `name`, `seq`, `ctx`, `data`) with floats
rounded to 12 decimals and numpy arrays encoded with their dtype and shape.

```powershell
$env:LMTG_TRACE = "D:\tmp\run.jsonl"
& ...\vlm_traj\python.exe main.py --task door ...
```

## Genesis camera semantics — measured, not assumed

`tests/test_genesis_camera_semantics.py` (runs under `vlm_genesis`, skips elsewhere) pins
the four facts the Genesis camera adapter rests on. Each would otherwise fail as *subtly
wrong 3D coordinates* rather than as a crash. Measured against Genesis **1.3.3**:

| Question | Answer | Consequence |
|---|---|---|
| `camera.projection_matrix` vs `p.computeProjectionMatrixFOV` | **Identical, transposed** (diff `0.0`) | Same dims (4x4) and same meaning. `flatten(order='F')` gives PyBullet's exact 16-vector, so `utils.get_intrinsics_extrinsics` needs no Genesis branch. |
| `inv(camera.transform)` vs `p.computeViewMatrix` | **Identical, transposed** (diff `6.9e-08`) | Same. The residual is float32 precision — cross-sim matrix tolerance must be `~1e-5`, not `1e-9`. |
| Depth encoding | **Linear metric `z_eye`** (metres along the optical axis) | *Not* euclidean range, *not* OpenGL non-linear `[0,1]`. Verified by rendering a plane perpendicular to the optical axis: all 65 536 pixels read one value (`unique_count == 1`), 1.9999847 for a 2.0 m camera height. Euclidean range would have spread the corners by ~0.56 m. `utils.get_world_point_world_frame` assumes the OpenGL form, hence the `depth_encoding="linear_metric"` branch. |
| Background / no-hit pixels | **`~far`** (99.9786 for `far=100`), never `0`, `NaN` or `inf` | No-hit pixels are detectable by thresholding near `far`; unprojecting them would emit bogus world points. |

Two further facts, also asserted:

* **Debug markers *do* render into offscreen captures** when the camera is created with
  `debug=True` (`vis/rasterizer.py`: `skip_markers = not camera.debug`). 3 460 of 65 536
  pixels changed after one `draw_debug_sphere`, and `clear_debug_object` restored the frame
  exactly. This is why the PyBullet massless-MultiBody marker hack is **not** needed on
  Genesis.
* **`camera.extrinsics` is a `@cached_property` that goes stale**: after a `set_pose` that
  moved `camera.transform` by 1.707, `camera.extrinsics` changed by **0.0**. The adapter
  must recompute `inv(camera.transform)` on every capture and never read `extrinsics`. The
  test is a canary — it fails if Genesis ever fixes this.

## Cross-simulator parity — `test_genesis_vs_pybullet.py`

Answers "same dims/meaning?" numerically instead of by inspection. Run under `vlm_genesis`
against the `pybullet_{door,grasp}.json` goldens in `tests/golden/cross_sim/`, produced by
`tests/tools/dump_sim_state.py` (see its module docstring for the exact regen commands —
one invocation per interpreter, since PyBullet and Genesis can't share a process).

| Kind | Tolerance | Why |
|---|---|---|
| Camera matrices (view/projection) | `atol=1e-5` | Pure math; float32 round-trip through Genesis' renderer. |
| Static scene positions (door frame/handle/hinge, pole) | `atol=1e-3` | Both sims place these from the same URDF; a real divergence here is a bug. |
| Settled arm (EE pos/joints, 180 steps) | `atol=2e-2` / `5e-2` | Two different solvers/integrators/contact models; physical agreement, not bit-exactness, is the honest bar. |

Index tables (link/joint numbering) are deliberately **not** compared — PyBullet and
Genesis genuinely disagree on numbering, which is exactly why the app layer resolves
everything by name (`JointInfo`, `get_link_index_by_name`, ...).

## IPC contract — `test_ipc_contract.py`

The *shape* counterpart to the numeric tests above: every live IPC command
(`EXECUTE_TRAJECTORY`, `CAPTURE_IMAGES`, `OPEN_GRIPPER`, `CLOSE_GRIPPER`, `GET_STATE`,
`GET_ROBOT_STATE`, `RESET_EEF`, `VISUALIZE_GRASP_POSE`, `VISUALIZE_BOUNDING_BOX`, ...) is
sent over the **real transport** both sims use in production, and the reply's keys/types/
shapes are asserted — never exact floats, since `run_simulation_environment` free-runs
`env.update()` and no float is reproducible across runs (that's what
`test_pybullet_regression.py` and `test_genesis_vs_pybullet.py` are for).

* `TestPyBulletIpcContract` — subprocess of the current interpreter, `multiprocessing.Pipe`.
* `TestGenesisIpcContract` — child in the `vlm_genesis` interpreter, JSON-lines TCP
  (`providers/json_ipc.py` + `providers/genesis_launcher.py`); skipped if that interpreter
  can't be resolved, so the suite stays green on a PyBullet-only machine.

Both subclasses share one `IpcContractMixin`, so a new command gets its contract checked on
both sims by adding one test method in one place.

## Baseline (recorded before the sim-adapter refactor)

`vlm_traj`, repo root, one file per invocation.

| File | Result | Time |
|---|---|---|
| `test_skill_registry.py` | 23 passed | 4 s |
| `test_llm_cache.py` | **10 failed**, 7 passed | 3 s |
| `test_env_direct.py` | 1 passed | 1 s |
| `test_2d_pixel_coords_to_3d_world_coords.py` | 1 passed, 8 subtests passed | 11 s |
| `test_pybullet_regression.py` | 1 passed, 2 subtests passed | 10 s |
| `test_genesis_launcher.py` | 26 passed | 1 s |
| `test_genesis_camera_semantics.py` | 7 passed *(under `vlm_genesis`)* / 7 skipped *(under `vlm_traj`)* | 10 s |
| `test_franka_kitchen_ipc.py` | 6 passed | 19 s |
| `test_franka_kitchen_head_camera.py` | 21 passed | 13 s |
| `test_adroit_door_ipc.py` | **skipped** (hung before; see below) | — |
| `test_metaworld_server.py` | **7 errors** (collection) | 2 s |

Pre-existing failures — **not** caused by, and not in scope for, the refactor, but they must
not get *worse*:

* `test_llm_cache.py` — 10 failures unrelated to the simulation layer.
* `test_adroit_door_ipc.py` — sends `config.SET_DOOR_STATE` (18) and
  `config.CAPTURE_TRAJECTORY_FRAME` (19). Both constants still exist in `config.py:124-125`
  but the handlers were removed from `run_simulation_environment`, so the test blocked
  forever in `parent_conn.recv()`. Dead test against a dead command; now marked
  `@pytest.mark.skip` so it no longer wedges a full-suite run. Un-skip only after restoring
  both handlers.
* `test_metaworld_server.py` — metaworld is obsolete and is not being maintained.

### Missing dependencies found while baselining

`requirements.txt` lists them, but they were absent from `vlm_traj` and made `utils.py` and
`skill_registry.py` unimportable:

```powershell
& ...\vlm_traj\python.exe -m pip install "scikit-learn==1.9.0" "PyYAML==6.0.3"
```
