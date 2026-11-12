# USB insert: speed, pointing, door, skill simplification (plan, 2026-11-12)

The status and the step-by-step guide are in `docs/status/usb_insert.md`.

**Progress (2026-11-12, resumed):**
- Done: 1 (sim 2x faster), 2 (`--perception-reasoning-effort`, default low; reviewer keeps `--reasoning-effort`),
  3-4 (bench: Gemini Flash 4.4 mm vs terra 24 mm median; now the usb wrist-pointing default), 5 (Molmo module,
  untested on GPU), 6 (door analysis + 4 fixes, see `open_door.md`; still stalls at ~35 deg: forearm `panda_link5`
  hits the panel), 8 (docs).
- 7 (skill ablation): done, 0/4 seated, keep the full skill. It found a new blocker: re-picking a dropped plug fails. See the usb_insert.md "Skill ablation" section.
- Still open: touch search (after LLM/VLM) in 1-3 min wall; door forearm collision. Door rerun: Genesis
  `door:hidden` SUCCESS (71 deg) on 2026-11-12.

## Decisions (user)
- **Pointing comparison:** offline benchmark on saved wrist views first, then 1 live run with the best model.
  - Save the wrist RGB, depth and camera pose for each view.
  - Truth is the sim port pixel, used for eval only and never sent to the agent.
- **Effort:** add a separate `--perception-reasoning-effort` flag, default `low`. Measure low vs medium terra latency and accuracy.
  - Today `--reasoning-effort` applies to the LLM, perception and the reviewer together.
- **Molmo:**
  - The gpu_server module should support both `allenai/MolmoPoint-Vid-4B` and `allenai/MolmoPoint-8B`.
  - Two modes: large VRAM (bf16) and 8 GB, for the RTX 3070 Linux box. The 8 GB mode uses bitsandbytes nf4 or int8; there is a community quant, `erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16`.
  - No SSH access: prepare the code, a Docker compose profile and instructions, and the user runs them.
- **Gemini pointing:** `or-google/gemini-3.8-flash` (OpenRouter, $0.83/M input), up to 30 calls. `OPENROUTER_API_KEY` is in `.env`.
- **Skill ablation:** 4 rollouts (e.g. 2 variants x 2 runs).
- **door:hidden (Genesis):** analyse the logs, then 1 rerun to confirm. Do NOT switch to a side grasp: the top-down lever grasp has always worked, so find out why it failed.
- **Experiment runs** may last 30 min.
- **Open item:** keep the search code (after the LLM/VLM calls) limited to 1-3 min wall time.

## Findings so far
- **Terra wrist locate:** calls start 15-35 s apart (an image is saved at the start of each call).
  - s2 made 16 calls over 8 views (the skill says 3-4), so locating took ~6 min.
  - The perception effort was `medium` in the 2026-11-12 runs (inherited from `--reasoning-effort`).
- **The insert state machine is pure sim time and slow:**
  - o1 ran 48 s of sim in ~6 min wall (RTF 0.13); s2 ran 8 s of sim in ~1 min.
  - Cause (to confirm by profiling): each 4.17 ms step in `touch/sim_runner.py` `kin()` makes ~17 separate Python->Genesis calls (link pose, Jacobian, 7 position reads, 7 velocity reads, plus torques). `read_state()` calls `kin()` again on every tick.
- **The search is inefficient by design:**
  - Raster at 5 mm/s over ±search_half (up to 80 mm) x 11 rows (0.4 mm pitch) = up to ~350 s of sim.
  - Every partial catch costs a probe (0.4 s), a settle (0.2 s) and a back-off.
- **Ideas:**
  - batched joint reads (`get_dofs_position` / `get_dofs_velocity`);
  - a faster first pass (10-15 mm/s);
  - rows ordered centre-out;
  - fewer rows when the z estimate is confident;
  - one VLM call per view (port + edge together).
- **Model check:** `allenai/MolmoPoint-8B`, `allenai/MolmoPoint-Vid-4B` and `google/gemini-3.8-flash` all exist. The gpu_server already has a module pattern, Docker for the 3070, and a VRAM table (`gpu_server/README.md` ~L288).

## Todo
1. Profile `run_touch_insert` on Genesis and add batched joint reads with a per-joint fallback, without changing the physics. The aborted profiler read the code but measured nothing:
   - `sim_adapter/genesis_adapter.py` `get_joint_state` makes 3 Genesis calls per joint (position, velocity, `get_dofs_control_force`), so ~42 calls per step in `kin()`.
   - Fix: batched `get_joint_positions` / `get_joint_velocities` via `entity.get_dofs_position/velocity(dofs)`.
2. Add `--perception-reasoning-effort` (`main.py`, `api.py:596` wrist locate, planner perception). Decide whether the reviewer (`api.py` ~1893) gets it too.
3. Pointing dataset:
   - a sidecar (`*.npz`: depth + wrist cam pose/intrinsics) for each `locate_with_wrist_camera` image;
   - a harness that captures ~15 views around the port face.
4. Benchmark script: terra low/medium vs gemini-3.8-flash (≤30 calls) vs Molmo (later, on Linux). Metrics: pixel error, along-face 3D error, hit rate.
5. gpu_server Molmo module plus a compose profile `molmo` (bf16 / 8-bit / nf4 via env).
6. door:hidden Genesis: analyse the 2026-11-10 logs (HF `rollouts/2026-11-10/door_hidden_genesis_touchpull_fail`) against the PyBullet successes, then 1 rerun.
7. Skill ablation: list the complex SKILL.md parts (e.g. tip-direction rules, face-order rules, the long step-3 prose), remove candidates, run 4 rollouts with a 30-min cap.
8. Update `docs/status/usb_insert.md` (open item: search ≤1-3 min) and `open_door.md`, then commit/push.
