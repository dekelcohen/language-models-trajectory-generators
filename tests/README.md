# Tests

## Interpreters

| Suite | Conda env | Notes |
|---|---|---|
| PyBullet suites (everything below today) | `vlm_traj` | `C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe` |
| Genesis suites (`test_genesis_*.py`, planned) | `vlm_genesis` | Genesis needs `Pillow>11`, `mujoco`, `numba`, ... which collide with `vlm_traj`; it therefore lives in its own env and runs as a child process. |

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

## Baseline (recorded before the sim-adapter refactor)

`vlm_traj`, repo root, one file per invocation.

| File | Result | Time |
|---|---|---|
| `test_skill_registry.py` | 23 passed | 4 s |
| `test_llm_cache.py` | **10 failed**, 7 passed | 3 s |
| `test_env_direct.py` | 1 passed | 1 s |
| `test_2d_pixel_coords_to_3d_world_coords.py` | 1 passed, 8 subtests passed | 11 s |
| `test_pybullet_regression.py` | 1 passed, 2 subtests passed | 10 s |
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
