# Status: kettle -> mug pour with particle water (`franka_kitchen:kettle_mug`)

Living status log; updated whenever there are conclusions or new data, so work can resume
after an interruption.

## Goal (user request)
- Add a mug (cylinder) to the kettle env/task; prompt: grasp the kettle and pour into the mug.
- Model water with small particles; test the task end-to-end; commit + push to `tracking`.
- Keep `prompts/skills/pour-from-held-container/SKILL.md` short and simpler.

## Decisions
- **Genesis rejected for this task** (measured; superseded below by "Genesis native water", which
  accepts slower-than-real-time): Genesis 1.3.3 is in the `vlm_genesis` conda
  env with CPU-only torch (RTX A1000, 4 GB). A minimal SPH scene (512 particles, no
  kitchen/robot) ran 26 ms/step = 6.4x slower than real time. The kitchen is PyBullet-only
  (`franka_kitchen/simenv.py` calls `p.*` directly). Not worth porting.
- **PyBullet water = rigid spheres** (common practice, e.g. GitHub
  AswinRetnakumar/Pouring_Project uses `sphere_1cm.urdf`); PyBullet has no fluid solver.
- The kettle collision body is a solid box, so water is a **reservoir released at the spout
  tip**: when the tip is >= 1 cm below the kettle body centre (tilt ~20 deg toward the spout),
  one sphere (r 8 mm, 2 g, blue) spawns every 8 steps (30/s) with the kettle velocity +
  0.15 m/s along the spout. 40 particles. Particle-kettle collisions disabled.
- Mug: hollow compound body (bottom disc + 16 wall boxes), outer r 0.05, height 0.09, 0.3 kg,
  red, on the free counter spot left of the hob at (0.22, 0.48, z 1.20) (found by ray-casting;
  the spout faces +y, toward it).
- Success: >= 50 % of particles inside the mug's inner cylinder (`goal_pour.min_fraction`).

## Code (uncommitted so far)
- `sim_envs/pybullet/franka_kitchen/tasks.py`: `KitchenTask.goal_pour`; task `kettle_mug`.
- `sim_envs/pybullet/franka_kitchen/simenv.py`: `MUG_*`, `WATER_*`, `KETTLE_SPOUT_*`;
  `_load_mug_and_water`, `_update_water` (from `step_hook`), `water_in_mug`; `get_state` adds
  `mug_pos` and `water {total, in_kettle, in_mug, outside_mug}`; `task_error` = fraction of
  water not in the mug; `_threshold` for pour tasks.
- Probe (session files, not committed): `mug_probe.py` - grasp, lift, carry to
  (tx, ty, carry_z), tilt with `pour_pose(..., rot + pi/2)`, hold, count water.

## Probe data
| hand target (x, y, z) | reached z | result | note |
|---|---|---|---|
| 0.22, 0.40, 1.60 | 1.54 | 0/40 in mug | tilted tip ~4 cm past and 0.155 m below the hand at first, then kettle rotated in the fingers to ~57 deg and the stream landed short (y 0.34) |
| 0.22, 0.48, 1.53 | 1.47 | 0/40, mug knocked over | arm sags ~6 cm under payload; kettle bottom skimmed the mug rim (1.29) |
| 0.22, 0.51, 1.62 | 1.56 | **36/40 in mug, success** | tilt 0.9, hold 2.5 s; kettle bottom at 1.385, clears rim |
| 0.22, 0.50, 1.66 | 1.60 | **36/40 in mug, success** | same; placement tolerant to +-1 cm y / +-4 cm z |

**Placement rule (verified):** hand over the mug (~3 cm past its centre in the pour
direction), hand z ~0.33 above the mug rim (kettle hangs ~0.175 below the hand and the arm
sags ~6 cm), tilt `pour_pose` 0.9 toward rot + pi/2, hold ~2 s.

Conclusions so far:
- Spout goes down toward +y as intended; kettle stays held but rotates further in the fingers
  (tilt 0.9 -> ~1.0 rad) and swings back ~13 cm in -y while holding, so the stream lands
  behind the commanded point.
- Mug wall had a visible slit between segments: segment overlap raised 1.15 -> 1.3.

## Full LLM runs (gemini-3.8-flash for LLM, reviewer, perception)
| run | result |
|---|---|
| 1 | Planner: 2 subtasks (pick up + carry/tilt over mug). Both passed on attempt 1, `plan_completed`. Pour subtask used the new short skill: hand ended (0.221, 0.511, 1.524), kettle (0.226, 0.484, 1.349). Reviewer: success ("poured water particles directly into the mug"). Water count not logged (env prints not captured) -> added `water` to the `[world-state]` log line. |
| 2 | Same 2-subtask plan, both passed on attempt 1, `plan_completed`. **Measured: 36/40 particles in the mug (90 %), 4 outside, success.** Hand ended (0.22, 0.466, 1.541). |

**Conclusion: task works end-to-end** with the shortened skill (3.3 KB, was ~9 KB). Tests:
`tests/test_kettle_mug_water.py` + kitchen IPC + side-approach: 35 passed.

## Next steps
1. Optional: more runs for a success rate; the planner's 2nd subtask prompt sometimes omits
   "tilt back upright" (run 2 ended still tilted - harmless here, the water was all poured).

## Old next steps (done)
1. Finish probe sweep -> placement rule; confirm >= 20/40 in mug.
2. Rewrite the pour SKILL.md short/simple.
3. Full LLM run: `python main.py --task franka_kitchen:kettle_mug -r franka --sim pybullet
   -lm or-google/gemini-3.8-flash --review-provider vlm:or-google/gemini-3.8-flash
   --planner-perception-vlm or-google/gemini-3.8-flash --attempts 2 --max-planner-iter 4
   -c "Pick up the kettle and pour water into the mug"`.
4. Tests (mug/water spawn, state, success), `document.md` kitchen section, commit + push.

## Related parallel work (background agents)
- Tracker uncommitted changes (CoTracker backfill): `docs/status/tracker_backfill.md`.
- Slide-cabinet reviewer false positive: `docs/status/slide_cabinet_review.md`.

## Genesis native water (`--sim genesis --task kettle_mug`)
Separate Genesis task with real SPH liquid (the PyBullet version above is unchanged).

Run:
```
python main.py -lm copilot-claude-opus-5.5 --sim genesis --task kettle_mug -r franka \
  --planner-perception-vlm or-google/gemini-3.8-flash --review-provider vlm:or-google/gemini-3.8-flash \
  --attempts 2 --max-planner-iter 4 --reasoning-effort medium -c "Pick up the kettle and pour water into the mug"
# scripted (no LLM) grasp -> carry -> pour_pose tilt -> hold, prints water counts per phase:
C:\Users\dekelco\.conda\envs\vlm_genesis\python.exe research\pour\genesis_kettle_mug_e2e.py --stage all --out outputs\pour\genesis_e2e --hold 240
# tests (count_water anywhere; scene test needs genesis, ~90 s):
python -m pytest tests\test_kettle_mug_genesis.py
conda run -n vlm_genesis python -m unittest tests.test_kettle_mug_genesis
```

Decisions:
- **Code**:
  - `sim_envs/genesis/kettle_mug.py` (scene, URDF generation, water counting, `get_state`).
  - Additive changes: `sim_adapter/genesis_adapter.py` (`sph` / `pbd` scene options pass-through) and `sim_envs/registry.py` (`GENESIS_TASKS += kettle_mug`, scene options, simenv).
- **SPH, not PBD**:
  - Material: DFSPH, 8 mm particles, `mu=0.002`, 4 substeps (1 ms).
  - PBD liquid's rigid coupling blew the kettle away.
  - SPH `mu` 0.005 is unstable at a 1 ms substep.
- **Water**: 624 particles, filling the kettle bottom 3 cm (~0.3 kg).
- **Kettle (hollow, physically honest)**:
  - Box-compound URDF: bottom, 16 wall boxes, lid, an open spout duct (38°, 6.5 cm) through a hole in the wall, and a bar handle on two posts.
  - Smooth trimesh visual meshes.
  - Nothing is spawned or teleported; water leaves only through the spout.
- **Mug**: compound (disc + 20 wall boxes), r 0.05, h 0.09.
- **Grip** (no weld; contact + friction only). Genesis contacts are points, so a squeezed handle spins freely about the finger axis (= the pour axis). Fixes:
  - `RigidOptions.enable_torsional_friction` with `kettle.set_friction_torsional(0.012)` (pad patch).
  - `noslip_iterations=5`.
  - Finger kp 6000, which saturates the profile's 70 N cap.
  - Lighter kettle (0.3 kg) with a lower handle (bar 0.20 above the bottom), to cut the tilt torque.
- **Layout**:
  - Kettle at (0, 0.25), yawed so the spout points +y. Mug at (0, 0.52) on the spout side.
  - Handle bar along y, so the natural top-down grasp is **rotation 0**. Reason: the subtask prompt always says "rotation value at 0" (hard-coded in `prompts/main_prompt.py`), so with a held object the LLM always assumes rot 0.
  - The first layout (spout +x, grasp rot π/2) made the pour subtask tilt about the wrong axis and drop the kettle (run 3).
  - Mug at y 0.72 is too far: the tilted pose is out of reach.
- **Head camera**: spherical, from -x (yaw 270, pitch -20, distance 1.0, target z 0.35), 512² captures.
  - Framed up to z ~0.85 so a kettle lifted by the planner's default 50 cm pick-up stays visible.
  - The default fixed head camera lost it (run 1).
- **Rendering**: `vis_mode="particle"`, translucent blue. Recon surface (`LMTG_WATER_VIS=recon`, pysplashsurf) looks smoother but was ~25x slower (358 s vs 14 s per move), so it is off.
- **Skill**: `pour-from-held-container` hold changed from 30 → 240 points (~1 s). SPH needs time to drain; 30 points (0.125 s) poured only ~45 %.
- **No GPU**: `vlm_genesis` torch is CPU-only, so `gs.gpu` is not available; the CPU backend is used.

Step time (CPU, robot + 624 particles + 2 cameras):
- ~0.10-0.11 s per control step (1/240 s), i.e. ~25x slower than real time.
- ~0.17 s/step during the active pour, with recording.
- Env start-up: 60-100 s.
- One full LLM run: ~9 min.

Scripted e2e (`genesis_kettle_mug_e2e.py`, tilt 0.8, hand 3 cm past the mug, mug top + 0.30):

| run | setup | result (in mug / 624) |
|---|---|---|
| 1-6 | first kettle (0.5 kg, handle 0.25 m) | kettle slipped or spun in the grip; no pour |
| 7-8 | lighter kettle, lower handle; hold 30 | 44-48 % |
| 9, 11 | hold 240, spout +x layout | **576 (92 %)**, 0 outside |
| 12 | same, pour height mug + 0.50 | 462 (74 %) |
| 13 | spout +y, mug y 0.72 | out of reach: tilt only 20°, 0 % |
| 14 | mug y 0.58 | 420 (67 %) |
| 15 | mug y 0.52 (final) | 410 (66 %), 153 outside (early gush during the fast tilt overshoots) |

LLM runs (`outputs/rollouts_20261102/kettle_mug_gn/`):

| run | LM | result |
|---|---|---|
| 1 | gpt-6.1-terra | Grasp + lift OK, but the lifted kettle left the fixed head camera's view. The LLM looped on "cannot verify". Fixed with the camera above. |
| 2 | gpt-6.1-terra | Over-cautious: never descended (handle-height detection 0.177 judged unreliable). Stopped. |
| 3 | claude-opus-5.5 | Old layout: grasp at rot π/2, then the pour subtask (told "rotation 0") tilted about the wrong axis. Attempt 2 dropped the kettle. Fixed by the layout change. |
| 4 | claude-opus-5.5 | **Success**: 2 subtasks (pick up: attempt 2 passed; pour: attempt 1 passed), `plan_completed`. **352/624 = 56 % in the mug**, 15 outside, 257 still in the kettle (pour height mug + 0.28). Video: `run4_head_full.mp4`. |

Open issues:
- The subtask prompt hard-codes "rotation value at 0". The real grasp yaw (from proprioception) should be reported; the layout above works around it.
- The early gush during the 30-point tilt overshoots the mug. A slower tilt would spill less.
- The water inside the mug is barely visible from the side camera.
- gpt-6.1-terra stalls on perception uncertainty in this scene.

## 2026-11-02: gentle pour

The fast tilt made an early gush that overshot the mug. Fix:
- `common_utils.limit_speed` + `execute_trajectory(..., max_speed, max_rot_speed)`: re-samples a trajectory so no segment exceeds the linear / angular speed. Exact end poses are kept; mixed 4/6-value poses interpolate position only.
- Skill `pour-from-held-container`: gentle-motion rule (`max_speed=0.1` m/s, `max_rot_speed=0.5` rad/s on the approach, tilt and tilt-back).

| run | setup | in mug / 624 | outside |
|---|---|---|---|
| baseline (run 15) | fast tilt | 410 | 153 |
| gentle2 (scripted) | 0.1 m/s, 0.5 rad/s | **594** | 30 |
| kettle_tt1 (LLM, claude-opus-5.5, all perception/review = gpt-6.1-terra) | skill's gentle rule | **598** | **10** (16 left in kettle) |

- kettle_tt1: picked up, then poured on attempt 1. The reviewer (terra) accepted it: "spout tilted over the opening and water entered the mug". `plan_completed`.
- Rollouts: `outputs/2026-11-02/kettle_pour_gentle_scripted_SUCCESS`, `outputs/2026-11-02/kettle_pour_LLM_ALLterra_gentle`.
- Harness: `research\pour\genesis_kettle_mug_e2e.py --stage all --hold 240 --max-speed 0.1 --max-rot-speed 0.5`.
