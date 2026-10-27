# Plan: touch-guided grasp-and-pull (doors, drawers, chairs) — no LLM in the loop

Status: implemented in Genesis (2025-10; see §8). Branch `tactile`. Builds on `touch/` (USB insertion) and
`helpers/pull_proprio.py`. Sim2Real rules apply: inputs are EE pose (FK), estimated external
wrench (joint torques), gripper width, camera tracking. No hinge axis, door angle or contact list.

## 1. Why the door is unstable today (from the bisect, `outputs/door_detailed/`)

- The **same geometry** (grasp 15–20 cm along the lever, rot -0.7…-1.05, press 2–2.5 cm, swing +y)
  sometimes opens the door and sometimes does not, so the outcome is stochastic and open-loop.
- The pull is one fixed straight line or a guessed arc written by the LLM. Once the door turns, the hand
  pulls against the hinge constraint. The jaws are levered off the bar, or the arm stalls.
- The unlatch press is a fixed 2–3 cm whatever the lever does. Too deep levers the jaws off.
- Outcomes are judged after the fact from video and `pull_proprio`, with false negatives and false
  positives. An attempt that did open was reported `part_did_not_move`, so the LLM "fixes" a working plan.

## 2. What DoorBot does (arXiv 2504.09358, ICRA 2025, 90 % on 20 unseen doors)

- A state machine of primitives: approach → grasp → unlock (lever/knob) → open → traverse. Every error
  edge goes back to grasp or approach. There is **no LLM in the low-level loop**.
- Haptics come from cheap **joint motor current** plus gripper resistance:
  - `GRASP_MISS`: low gripper resistance, so the grasp failed.
  - Unlock: rotate until the elbow current crosses a threshold (stop at the handle's limit). If the
    current is large early, reverse the direction.
  - Open: a small backward probe. If the current rises the door is push-type; if it falls it is
    pull-type.
  - `*_MISS`: contact lost mid-motion. `*_COLLISION`: a current spike. Both transition back to grasp.
- Lesson for us: the LLM, or vision, picks *where and roughly which way*. A deterministic, force-aware
  primitive does the motion and returns a **discrete outcome**.

## 3. Generic primitives (new module `touch/pull.py`, same pattern as `touch/insertion.py`)

Each one is a pure controller: `step(TouchState) -> PullCommand` (EE target + stiffness). It has a
`run_reactive` loop, a sim runner, an IPC command, and an `api.py` function. The functions return
`{"outcome", "travel_m", "dir_final", "events", ...}`, all from proprioception only.

| API function (LLM-callable) | Does | Outcomes |
|---|---|---|
| `grasp_with_touch(close_force=40)` | force-limited close (Franka `grasp(width, speed, force)`), then a 5 mm tug | `held` / `closed_on_air` (width ≈ 0) / `slipped_on_tug` / `fixed` (tug felt > 15 N with 0 travel) |
| `press_with_touch(dir=[0,0,-1], f_stop=15, max_d=0.04)` | guarded press (lever unlatch, button). Stops on a force threshold or when travel stops | `pressed` (moved, then the end stop was felt) / `blocked` (force with no travel) / `lost` (width collapsed) |
| `probe_directions(cands, d=0.01)` | DoorBot push/pull probe: a 1 cm compliant move per candidate. Ranks the candidates by felt resistance | ranked list, `none_free` |
| `pull_with_touch(dir0, dist, f_max=30, mode="auto")` | **core**: compliant pull that follows the mechanism (below) | `opened` / `slipped` / `blocked` / `end_stop` / `collision` |
| `regrasp_on_handle(track_name)` | re-acquire the moved handle and resume (section 4) | `held` / `lost_track` |

### 3.1 `pull_with_touch` — follow the constraint, do not fight it
Controller: Cartesian impedance in Genesis torque mode, like `run_touch_insert` (Genesis only, like the
USB touch; a real Franka gives the same via its torque interface). It is stiff along the current pull direction `d` and soft across
it. It moves the setpoint along `d` at 2–5 cm/s. Each tick (50 Hz):

1. **Direction update (constraint following).** The mechanism pushes back across `d`, and that lateral
   force tells the hand where the path bends: `d ← normalize(d + η · F_lat / |F|)`, limited to
   ≤ 3°/tick. A door arc, a drawer slide and a chair drag are handled alike (Niemeyer & Slotine
   adaptive door opening; Vi-TacMan contact regulation).
2. **Online hinge fit.** After 4–5 cm of travel, fit a circle to the EE path (least squares, in the
   horizontal plane). A radius under 1.5 m means hinged: feed the tangent forward and keep the force
   update as a correction. A larger radius means a slide: keep a straight line.
3. **Slip detection.** Three cues:
   - the width drops by more than 3 mm, or reaches ≈ 0;
   - the felt pull force collapses to free-space (< 3 N) while still commanding travel;
   - lateral force > 0.8 · µ · grip_force (the friction cone is about to break).
   On the last cue, first *reduce* the pull speed and rotate `d` toward the lateral force (pre-slip).
   On the first two, stop with `slipped`.
4. **Blocked / end stop.** Felt force > `f_max` with travel < 2 mm over 0.5 s. Before 3 cm of travel
   this means `blocked` (latched, or wrong direction). After 3 cm it means `end_stop` (fully open), so
   stop.
5. **Collision.** A force spike in a direction with no travel component (the hand hits the door
   face), so stop with `collision`.

### 3.2 Deterministic recovery policy (`open_with_touch(handle, dir0)`, no LLM)
```
grasp_with_touch -> closed_on_air: re-detect the handle, offset 1 cm, retry (max 2) -> else fail
press_with_touch (lever only)          -> blocked: skip the press, try the pull anyway
probe_directions([dir0, -dir0, ±lateral])   -> pick the least resistance (DoorBot push/pull)
loop (max 3 regrasps):
    pull_with_touch(d)
      opened / end_stop -> release, back off -> done
      slipped -> regrasp_on_handle(): handle pose = last EE pose at slip (proprio)
                 + tracking report (CoTracker/ROMA, handle moved with the door)
                 -> re-approach along the current fitted tangent, resume from the new pose
      blocked (<3 cm) -> the next direction from probe_directions / pull_proprio.alternative_directions
      collision -> back off 2 cm along -F, re-plan the approach
```
The LLM writes one call (`open_with_touch(...)`) instead of a guessed arc. Its job shrinks to
grasp pose, handle type and a rough initial direction, which are perception tasks it does well.

## 4. Re-grasp on the moved handle (tracking)

- Primary source: the EE pose at the moment of slip, because the fingers were on the handle. Tracking
  CoTracker or ROMA world points on the handle gives its new pose if the door swung back or kept
  moving (`get_tracking_report()`). If both are available, take the tracking estimate when it agrees
  within 3 cm.
- The approach direction is the current fitted tangent's normal (the door face normal at the new
  angle). Use the arc fit (centre, radius) to rotate the grasp yaw by the door angle swept so far.

## 5. Reuse beyond doors

- Drawer: straight-line slide; the hinge fit returns R = ∞. The end stop is the drawer stop.
- Sliding door: `probe_directions(±lateral)` picks the free side.
- Chair or box drag: a pull with no constraint. `blocked` means stuck or too heavy; `slipped` means
  re-grasp.
- Lid or hatch: a vertical arc, the same circle fit in the vertical plane.
- Not covered: knobs that need wrist twist + pull (would need a torque-axis variant), or locked doors.

## 6. Prerequisites in the sims

- **Genesis only** (no PyBullet torque work). The door task already loads in Genesis through the
  adapter (`sim_envs/genesis/genesis_env.py --task door --direct`: scene built, Franka PD set), and
  the adapter has the torque API used by `run_touch_insert`. `pull_with_touch` reuses that runner;
  PyBullet/MetaWorld return `unsupported` and the skill is listed only with `-s genesis --touch`.
- To check first: door hinge damping/friction and handle contact in Genesis (an open-loop `main.py
  --task door -s genesis` baseline), and that the 87/12 Nm actuator caps can pull the latched door.
- Calibrate the wrench noise on the door: free-space pull vs latched vs hinge-bound, as in
  `research/touch/genesis_touch_probe.py`.

## 7. Milestones & validation

1. `touch/pull.py` controllers with unit tests on synthetic `TouchState` streams: arc following
   converges, slip/blocked/end-stop are classified (like `tests/test_touch_*`).
2. Genesis torque-mode pull runner (`touch/sim_runner`) and `PULL_WITH_TOUCH` IPC. Scripted harness `research/touch/door_pull_e2e.py`
   (like `usb_pipeline_e2e.py`): perturb the grasp offset (±1.5 cm) and the initial direction
   (±30°) over 10 seeds. Compare the open-loop baseline with touch. Use the door angle for
   **eval only**.
3. Recovery policy, then tracking-based re-grasp.
4. API + skill: replace the section 2/3/4 motion text with `open_with_touch` (2–3 lines). Then run
   `main.py` door ×3 with gemini-3.8-flash.
5. Drawer (franka_kitchen) to check that the primitives generalise.

Targets: door success ≥ 8/10 scripted (today ~stochastic), no false `part_did_not_move` (the outcome
comes from the felt pull, not from video).

## 8. Implementation status (Genesis)

Code: `touch/pull.py` (PullController), `touch/sim_runner.py::run_touch_pull` (impedance loop),
IPC `TOUCH_PULL` -> `api.pull_with_touch`, `touch/recovery.py::open_with_touch` (pull, slip ->
re-grasp, resume), skill `touch-guided-pull`. `--no-touch` returns `disabled` (A/B). Harness:
`research/touch/door_pull_e2e.py [--touch|--no-touch] [--dir normal|skew|tangent] [--err dx,dy,dz]
[--grip N] [--max-regrasps K] [--debug]` (Genesis python).

Root causes found on the way (each one fixed):
1. The door lever had no collision geometry -> fingers closed on nothing (URDF: Side Cap 1 now collides).
2. Genesis IK re-sampled random seeds -> branch jumps that swung the arm through the door
   (continuity solve first; a re-sampled branch only if the joint jump < 0.8 rad or continuity is > 5 cm off).
3. Franka finger force ~2 N -> raised to kp 2000 / force 70 (real hand: 70 N continuous).
4. A top-down grasp is infeasible (the lever is ~10 cm off the face) -> side approach, jaws vertical.
5. Soft-contact creep: the lever slid out at ~7 N although friction capacity was ~70 N ->
   `noslip_iterations=5` for door scenes (same as USB).
6. A stiff wrist yaw fought the hinge through the rigid grasp (end stop at 42°) -> compliant yaw (`k_yaw` 3 Nm/rad).
7. The swept angle was measured from the initial pull direction (wrong when skewed) -> angle about the fitted hinge.
8. Re-grasp turning to the door's new normal is out of reach once the door swings -> keep the hand
   orientation, re-approach a few cm, resume on the earlier hinge fit (short arcs fit badly).
9. The lever escapes the pads past ~70° -> default target 65°.

A/B (door angle at the end, eval only):

| case | touch | open-loop |
|---|---|---|
| normal | 61.3° | 36.2° |
| skew (30° off) | 62.1° | 9.7° |
| perception error 1 cm | 61.3° | (37.8° earlier) |
| tangent start | 61.2° | — |
| weak grip 4 N (forced slip at 51°) + 1 re-grasp | 64.5° | — |

Open: a re-grasp guided by detection (not only proprio + tracking); drawers and chairs untested in sim.
