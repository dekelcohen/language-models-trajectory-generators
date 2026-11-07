# Plan: Holonomic mobile base under the Franka (mobility primitives)

Status: proposal (2026-11-08). Companion plan: [semantic_navigation.md](semantic_navigation.md).
Research background: [../models/mobile_manip.md](../models/mobile_manip.md) (base types, LeKiwi,
TidyBot++, OWMM-Agent, OK-Robot ...), [../models/navigation.md](../models/navigation.md).

---

## 1. Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | A round, **holonomic** wheeled base (kiwi drive: 3 omni wheels at 120°, Ø ≈ 0.5 m) with the Franka on a mount plate, in **PyBullet and Genesis** behind `SimAdapter` | the `franka_mobile` robot loads in both sims; base + arm move |
| G2 | LLM-callable primitives: turn left/right, forward/backward, strafe left/right, any planar twist, stop, odometry pose | sim test: forward 1.0 m → error ≤ 3 cm; turn 90° → error ≤ 2°; strafe 0.5 m → error ≤ 3 cm (no added noise) |
| G3 | Every base motion is **guarded**: it stops before an obstacle seen in depth, or on a contact force | test: drive at a wall → stops ≥ `base_guard_dist` − 5 cm from it, `stopped_by="depth"` |
| G4 | Arm and perception keep working after the base moved: camera extrinsics and arm targets follow the **odometry** pose | test: drive 1 m, then `detect_object` + pick succeeds |
| G5 | **Off by default.** Fixed-base tasks, prompts and golden traces are byte-identical | `tests/test_pybullet_regression.py` passes unchanged |
| G6 | Same tool API on the real robot (LeKiwi-class base, or a Franka on a kiwi/caster base) | interface stub + ROS2 `cmd_vel` mapping documented |

## 2. Non-goals

- Semantic navigation, mapping, path planning (→ [semantic_navigation.md](semantic_navigation.md)).
- Moving the arm and base at the same time (whole-body MPC). v1 is decoupled: the base moves,
  then the arm moves.
- Learned locomotion; stairs, ramps, rough terrain; outdoor driving.
- Physical omni-wheel roller simulation in v1 (it is an option in milestone M4).
- A full real-robot driver. Only the interface and a LeKiwi/ROS2 stub are in scope.

## 3. High-level approach

```
 LLM code (exec)                 API process (api.py)              Sim process (env.py)
 ───────────────                 ───────────────────               ───────────────────────────────
 move_forward(0.5)  ──►  API.move_base(dx=.5,dy=0,dyaw=0) ─pipe─►  MOVE_BASE handler
 turn_left(90)                    result dict ◄────────────────┐     BaseController.move_relative()
 get_base_pose()                                              │       ├─ velocity profile (trapezoid)
                                                              │       ├─ kiwi IK  -> wheel speeds (logged)
                                                              │       ├─ wheel FK -> odometry (+noise)
                                                              │       ├─ depth / force guard
                                                              │       └─ sim: base_x/base_y/base_yaw velocity
                                                              └──── {ok, reason, achieved, odom_pose, stopped_by}
```

Key decisions:
1. **Sim model = kinematic planar base.** The URDF gets three virtual joints between a
   world-anchored root and the chassis: `base_x` (prismatic), `base_y` (prismatic),
   `base_yaw` (revolute, continuous). They are velocity-controlled with force limits.
   This is how TidyBot++, ManiSkill and Habitat model holonomic bases. It is fast and stable,
   and it stays physically honest: the joints have force limits, so a wall stops the base.
   There is no teleport.
2. **Wheel kinematics are still computed.** Twist → 3 wheel speeds → encoder angles →
   odometry. So the code path and logs match the real robot, and odometry drift can be added
   as noise (sim2real).
3. **"World" seen by the agent = odometry frame** (origin at the start pose). Camera
   extrinsics and arm targets are computed from the odometry pose. They are never computed
   from the sim ground-truth base pose (privileged). The ground truth is logged only, to
   measure the drift.
4. **Decoupled arm/base.** Arm IK solves only the 7 arm joints, with the base joints frozen.
   `execute_trajectory` refuses to run while the base is moving.

## 4. Pipeline (inputs → preprocessing → models/algorithms → outputs)

### Step 1 — Inputs
| Input | Type / units / frame | Rate | Source |
|---|---|---|---|
| Motion request | `(dx, dy, dyaw)` in m, m, deg, **base frame** (x forward, y left, yaw CCW+), or a twist `(vx, vy, ω)` in m/s, rad/s plus `duration` s | per tool call | LLM code → `API.move_base` |
| Wheel encoders | 3 wheel angles (rad) | every sim step (240 Hz PyBullet / Genesis dt) | sim: wheel FK of the virtual-joint velocities; real: motor encoders |
| Head RGB-D | H×W×3 uint8 + H×W float32 depth (m), intrinsics K, extrinsics `T_odom_cam` | guard check at ≥ 10 Hz while moving | head/mast camera on the chassis |
| Contact force | external force estimate on the chassis (N) | every step | sim: base joint reaction / applied-force readback; real: motor current / bumper |
| Limits | `max_lin_vel`, `max_ang_vel`, `max_lin_acc`, `max_ang_acc`, tolerances | static | `robot_profiles.BaseProfile`, `config.BASE_*`, CLI |

### Step 2 — Preprocessing
1. **Validate + clamp**: reject NaN; cap |dx|, |dy| ≤ `BASE_MAX_STEP_M` (default 3 m) and
   |dyaw| ≤ 360° per call. Larger moves must be split. That keeps the LLM in short,
   verifiable steps.
2. **Motion profile**: compute a time-optimal trapezoidal (or Ruckig-like jerk-limited)
   profile for each of x, y and yaw. The three are synchronized so they finish together
   (straight line in the base frame while rotating). The output is a twist setpoint
   `(vx, vy, ω)(t)` in the base frame.
3. **Guard field of view**: from the depth image, build an obstacle point cloud in the base
   frame. Keep points between floor + 3 cm and robot height. Keep only points inside a
   corridor swept along the motion direction (width = base diameter + `BASE_GUARD_MARGIN`).
   Take the minimum distance along the motion direction. When rotating in place, use the
   full circle around the base, radius = base radius + margin.

### Step 3 — Models / algorithms
1. **Kiwi inverse kinematics** (wheel i mounted at angle θi ∈ {90°, 210°, 330°}, base radius R,
   wheel radius r):
   `ω_i = (−sin θi · vx + cos θi · vy + R · ω) / r`. Wheel speeds are clamped to motor max.
   If any wheel saturates, the whole twist is scaled down uniformly, so the direction is kept.
   - LeRobot's LeKiwi (`lekiwi.py`, `_body_to_wheel_raw`) uses the same model form. Each row
     is `[cos a_i, sin a_i, R]`, with `a = [240, 0, 120] − 90` deg, `r = 0.05 m`,
     `R = 0.125 m`, and the same proportional scaling.
   - Ours corresponds to `a_i = θ_i + 90°`. The difference is only in where the wheels are
     mounted and which direction counts as a positive wheel turn.
   - So `BaseProfile` stores `wheel_angles_deg` and a per-wheel `wheel_sign`. The LeKiwi bridge
     (M5) then only swaps profile values.
   - A unit test checks both conventions against LeKiwi's own function.
   - See [mobile_manip.md §6](../models/mobile_manip.md).
2. **Forward kinematics / odometry**: twist = pseudo-inverse of the 3×3 matrix × wheel
   speeds. Integrate in the odometry frame with a mid-point rule. Optional noise model
   (`--odom-noise σ`): multiplicative wheel-slip noise σ on each wheel delta, plus a yaw bias.
   This is how drift appears in sim.
3. **Execution in sim**: command the virtual joints with the world-frame velocity
   (`R(yaw) · [vx, vy]`, ω), with force limits `BASE_MAX_FORCE`. The kinematic model uses the
   *commanded* twist for the joints and the *noisy wheel FK* for odometry.
4. **Closed loop on odometry**: a P-controller on the remaining `(dx, dy, dyaw)` delta, in
   odometry, corrects the profile near the end. It exits when the error ≤ tolerance
   (2 cm / 1°), or on stall (progress < 5 mm over 1 s) or timeout.
5. **Guard**: stop (decelerate at `max_lin_acc`) when the corridor distance <
   `base_guard_dist`, or the contact force > `BASE_MAX_CONTACT_N`. Result:
   `stopped_by ∈ {None, "depth", "contact", "stall", "timeout"}`.
6. **Arm/base frames**: `T_odom_armbase = T_odom_base · T_base_mount` (static mount transform
   from the URDF). `detect_object` / `convert_2d_point_to_3d_world` use
   `T_odom_cam = T_odom_base · T_base_cam`. `Robot.move` converts world (= odometry) targets
   into the arm-base frame before IK.

### Step 4 — Outputs
- To the LLM, returned by every base tool:
  ```python
  {"ok": bool, "reason": str, "requested": [dx, dy, dyaw_deg], "achieved": [dx, dy, dyaw_deg],
   "odom_pose": [x, y, yaw_deg], "stopped_by": None|"depth"|"contact"|"stall"|"timeout",
   "min_obstacle_m": float|None}
  ```
- To the sim: virtual-joint velocity targets every step. Recorded video frames continue as in
  `Robot.step_env_and_record`.
- To logs: one PROGRESS line per call (request, profile duration, achieved, stop reason,
  odometry vs ground-truth error: **log only**). A temporary per-step debug log of wheel
  speeds is behind `--base-debug-log`.
- To the reviewer: before/after head images and the odometry delta (see §7).

## 5. Code-level changes and integration points

| File | New/Mod | Change |
|---|---|---|
| `assets/mobile_base/kiwi_base.urdf` (+ `meshes/`) | new | `world_root` → `base_x` (prismatic, ±50 m) → `base_y` (prismatic) → `base_yaw` (revolute, continuous) → `chassis` (cylinder Ø0.5 × 0.25 m, mass ≥ 40 kg, low CoM, to resist tipping. See §10) → 3 wheel links (visual omni wheels, fixed in the kinematic model) → `mount_plate` (fixed) → `mast` + `head_camera_link` (≈1.3 m high, pitched down ~35°) |
| `assets/mobile_base/franka_mobile.urdf` | new | `kiwi_base` + Panda (`panda_link0` fixed to `mount_plate`, 0.30 m above the floor, forward offset 0.1 m). Generate it with a small xacro-free Python script `assets/mobile_base/build_urdf.py`, so the Panda URDF is not copied by hand |
| `robot_profiles.py` | mod | `@dataclass BaseProfile: joint_names=("base_x","base_y","base_yaw"); wheel_radius=0.05; base_radius=0.20; wheel_angles_deg=(90,210,330); wheel_sign=(1,1,1); max_lin_vel=0.5; max_ang_vel=1.5; max_lin_acc=0.5; max_ang_acc=2.0; max_force=500.0; head_camera_link="head_camera_link"; mount_link="mount_plate"`. Add `base: Optional[BaseProfile] = None` to `RobotProfile`. New profile `franka_mobile` (Franka fields + `base=BaseProfile()` + new `urdf`) |
| `mobile/__init__.py`, `mobile/base_kinematics.py` | new | pure functions (numpy only): `twist_to_wheels(vx, vy, w, prof) -> np.ndarray(3)`, `wheels_to_twist(wd, prof) -> (vx, vy, w)`, `integrate_odom(pose, twist, dt) -> pose`, `trapezoid_profile(delta, vmax, amax) -> (T, f(t))`, `scale_to_limits(...)` |
| `mobile/base_controller.py` | new | `class BaseController: __init__(sim, body, profile, logger, odom_noise=0.0, seed=None)`, `move_relative(dx, dy, dyaw_rad, max_speed=None, guard=True, step_fn=None) -> dict`, `set_twist(vx, vy, w)`, `stop()`, `odom_pose() -> (x, y, yaw)`, `true_pose()` (logging/eval only), `is_moving`. `step_fn` is `robot.step_env_and_record`, so video and tracking frames keep flowing |
| `mobile/obstacle_guard.py` | new | `corridor_min_distance(depth, K, T_base_cam, direction_xy, width, z_band) -> float`; `ring_min_distance(...)` for rotation. Uses existing depth decoding (`--depth-format`) |
| `mobile/real_lekiwi.py` | new (stub) | `class LeKiwiBase` with the same `move_relative / set_twist / stop / odom_pose`, over the LeRobot LeKiwi client (ZMQ). Notes for ROS2: publish `geometry_msgs/Twist` on `/cmd_vel` and read `/odom` |
| `sim_adapter/base.py` + `pybullet_adapter.py` + `genesis_adapter.py` | mod | `inverse_kinematics(..., dofs=None)`: solve only these joints. PyBullet: null-space IK with `lowerLimits = upperLimits = current` for the base joints, then an FK residual check. Genesis: `dofs_idx_local=arm_dofs`. Add `get_joint_reaction_force(body, joint)` if missing (for the contact guard); implement it with existing calls |
| `robot.py` | mod | `Robot.__init__`: new branch `elif args.robot == "franka_mobile"`. Load with `fixed_base=True` (the root is anchored; the virtual joints move the base). Split the movable joints into `self.base_joint_indices` and `self.joint_indices` (arm + fingers), so `joint_start_positions` still map to the arm. Base joints get velocity-mode gains (Genesis `configure_motor_gains` kv only). Create `self.base = BaseController(...)`. `Robot.move`: if `self.base`, transform target pose odom → arm base, and pass `dofs=self.joint_indices[:arm_count]` to IK. `get_camera_image("head")`: when the profile has `head_camera_link`, take the extrinsics from `T_odom_base · T_base_cam` (odometry) instead of `config.head_camera_position` |
| `config.py` | mod | IPC constants `MOVE_BASE = 30`, `GET_BASE_POSE = 31`, `STOP_BASE = 32`; constants `BASE_GOAL_TOL_M = 0.02`, `BASE_GOAL_TOL_DEG = 1.0`, `BASE_MAX_STEP_M = 3.0`, `BASE_GUARD_MARGIN = 0.05`, `BASE_MAX_CONTACT_N = 40.0`, `BASE_STALL_S = 1.0`, `BASE_TIMEOUT_S = 30.0`, `base_start_pose_franka_mobile = [0, 0, 0]` (x, y, yaw), `joint_start_positions_franka_mobile` |
| `env.py` | mod | import the new constants; in `run_simulation_environment`'s command loop (next to `EXECUTE_TRAJECTORY`, ~l.950) add the `MOVE_BASE` / `GET_BASE_POSE` / `STOP_BASE` handlers. `MOVE_BASE` calls `robot.base.move_relative(..., step_fn=lambda: robot.step_env_and_record(env))`. `EXECUTE_TRAJECTORY` refuses (`reason="base moving"`) while `robot.base.is_moving`. `GET_ROBOT_STATE` adds `base_pose` (odometry) when mobile |
| `api.py` | mod | `API.move_base(dx=0, dy=0, dyaw_deg=0, max_speed=None, guard=True)`, plus thin wrappers `turn_left(deg)`, `turn_right(deg)`, `move_forward(m)`, `move_backward(m)`, `strafe_left(m)`, `strafe_right(m)`, `get_base_pose()`, `stop_base()`. Each calls `_budget_tick("move_base", ...)` (repeated-no-progress protection), logs, and records into `self.task.base_moves` for the reviewer. `get_robot_state()` gets a `base_pose` key |
| `helpers/main_utils.py:get_exec_locals` | mod | add the base tools **only if** `api.args.mobile_base` (keeps prompts and goldens unchanged) |
| `task_state.py` | mod | `base_moves: list` per subtask (request, result, step index) |
| `prompts/main_prompt.py` | mod | a ~6-line paragraph appended only when mobile: frames (world = odometry, x forward at start), the tool list, "move the base before reaching if the target is > 0.75 m from the arm base". Details live in the skill |
| `prompts/planner_prompt.py` | mod | when mobile: the planner may emit "reposition base" subtasks (e.g. "drive to face the table, 0.5 m away") |
| `prompts/review_prompt.py` / `api.run_vlm_review` | mod | add the base-move evidence text: `"base moved +0.48 m fwd, -0.01 m left, +0.3°; stopped_by=None"` and before/after head images |
| `prompts/skills/mobile-base/SKILL.md` | new | when to use (target out of reach, view occluded), frames and sign conventions, tool signatures, examples (approach a table, re-centre the target in the head image, back off after a grasp), pitfalls (odometry drift → re-detect after every move; never call `execute_trajectory` with targets from before a move) |
| `sim_envs/pybullet/mobile.py` | new | `SimEnvMobile(SimEnvBase)`: room 6×6 m, walls, a table 2 m in front with YCB objects; tasks `mobile:drive_to_table`, `mobile:pick_far`, `mobile:turn_and_pick` |
| `sim_envs/genesis/mobile.py` | new | Genesis twin (same layout) |
| `sim_envs/registry.py` | mod | suite `"mobile": _mobile_suite` (lazy import), as for `franka_kitchen`. `_apply_required_robot` forces `franka_mobile` for `mobile:*` tasks |
| `helpers/rollout_report.py` | mod | `summary.md`: base path length, odometry vs ground-truth drift (eval only), guard stops |
| `document.md` | mod | new section "Mobile base"; argument reference; changelog |

### New CLI arguments (`main.py`, "sim" group)
| Flag | Type | Default | Help |
|---|---|---|---|
| `-r/--robot` | choice | `sawyer` | add `franka_mobile` |
| `--mobile-base / --no-mobile-base` | BooleanOptional | `None` → True iff robot is `franka_mobile` | expose base tools and base prompt text |
| `--base-model` | `{kinematic, wheels}` | `kinematic` | sim base model (`wheels` = physical rollers, M4) |
| `--base-max-speed` | float m/s | `0.3` | default linear speed cap for base moves |
| `--base-guard-dist` | float m | `0.25` | stop distance to the nearest obstacle in the motion corridor; `<= 0` disables (tests only) |
| `--odom-noise` | float | `0.0` | wheel-slip noise σ (fraction of each wheel delta), for drift experiments |
| `--base-start` | `"x,y,yaw_deg"` | from task | override the start pose (scene setup only; not shown to the agent as truth) |
| `--base-debug-log` | flag | off | temporary per-step wheel/twist log |

## 6. Sim2Real rules for this feature

- The agent only sees the **odometry pose** (wheel encoders + optional IMU / visual odometry
  on the real robot). `true_pose()` is used only in logs, `summary.md` drift metrics and tests.
- The obstacle guard uses only depth plus force/current. It never uses engine contact lists.
- Head camera extrinsics = odometry pose ∘ calibrated mount transform. The hand-eye /
  mount calibration error (~1 cm) adds to drift, so the skill says to **re-detect after
  every base move**.
- Real mapping: LeKiwi (LeRobot) for a cheap prototype; it has the same kiwi geometry, so the
  same kinematics. A Franka on a holonomic base (TidyBot++-style casters or Ridgeback) maps
  to `/cmd_vel` + `/odom`.

## 7. Logging / observability

- `[base] move_relative req=(dx,dy,dyaw) T=..s` → `[base] done achieved=(...) stopped_by=... odom=(...) drift_gt=(...)cm` (PROGRESS level, one per call).
- `[base] guard: min_obstacle=0.31 m dir=(+x)` when the guard trips or comes within 2× the guard distance.
- Per-step wheel speeds only with `--base-debug-log`.
- Video: base moves are recorded through `step_fn`. `rollout_report` adds a top-down odometry
  trace image (matplotlib) to the subtask folder.

## 8. Tests and evaluation

| Test | Kind | What it checks |
|---|---|---|
| `tests/test_base_kinematics.py` | unit | IK/FK round trip; with θ = (90°, 210°, 330°): pure forward → wheels ∝ `(−1, +½, +½)`, pure left strafe → `(0, −0.866, +0.866)`; pure rotation → equal wheel speeds; saturation keeps the direction; trapezoid duration |
| `tests/test_base_controller_pybullet.py` | sim (PyBullet DIRECT) | forward 1 m, backward 0.5 m, strafe ±0.5 m, turn ±90°, composite move: errors within G2 tolerances; guard stop at a wall; `--odom-noise 0.02` → odometry error grows but the result is still `ok` |
| `tests/test_base_genesis.py` | sim (Genesis, skipped if not installed) | same as the PyBullet test (shortened) |
| `tests/test_mobile_ipc.py` | IPC contract | `MOVE_BASE` / `GET_BASE_POSE` payloads (as in `test_ipc_contract.py`) |
| `tests/test_exec_locals_persistence.py` | mod | base tools absent unless `--mobile-base` |
| `tests/test_pybullet_regression.py` | regression | **unchanged** goldens for fixed-base tasks |
| e2e rollouts | eval | `mobile:drive_to_table`, `mobile:pick_far` with the LLM. Packaged under `outputs/<date>/<run>/` + HF upload (repo convention). Metrics: success, base path length, # base calls, drift |

## 9. Milestones

| M | Deliverable | Demo |
|---|---|---|
| M1 | `base_kinematics` + `BaseController` + URDF + `franka_mobile` profile + IPC + API tools (PyBullet) | scripted square path 1 m × 1 m returns within 3 cm |
| M2 | Arm on base: frame transforms, IK with frozen base, head camera on mast, `mobile:pick_far` | LLM drives to the table and picks a YCB object |
| M3 | Genesis twin + skill + planner/review prompt integration | same task in Genesis; reviewer uses the base evidence |
| M4 | Optional physical omni-wheel model (`--base-model wheels`, rollers as passive joints) | slip/drift realism comparison |
| M5 | Real bridge stub (LeKiwi / ROS2 `cmd_vel`) | dry-run with recorded odometry |

## 10. Risks and open questions

- **PyBullet IK with frozen base joints**: null-space limits are soft. Mitigation: FK residual
  check + retry. Fallback: a second arm-only IK body in a DIRECT client.
- **Head camera height/pitch**: one camera must serve both nav (far view) and manipulation
  (table view). Possibly add a pan-tilt joint (TidyBot++, Stretch have one). Decide in M2.
- **Arm reach vs base placement**: v1 uses an LLM heuristic. Later: a learned or sampled
  base-placement scorer (see N2M2 / reachability maps in `mobile_manip.md`).
- **Tipping / centre of mass**: the Franka arm is ~18 kg, and it reaches 0.85 m out from a
  base about 0.5 m wide. A LeKiwi-size base (R = 0.125 m, a few kg) would tip over.
  - In sim: a chassis of at least 40 kg with a low centre of mass, and acceleration limits.
  - Stow the arm before driving (`mobile-base` skill).
  - Log a static-stability margin, computed as the CoM projection distance to the support
    triangle edge. Use the URDF masses for this, and use it for logs and tests only.
  - Real hardware for a Franka-class arm: a TidyBot++-style powered-caster base or a
    Ridgeback. LeKiwi serves only as the reference for kinematics and sensors (SO-101 arm).
- **Genesis velocity control of prismatic virtual joints** with large mass: tune kv / force
  range in the profile, as for the arm.
