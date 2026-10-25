# Status: kettle -> mug pour with particle water (`franka_kitchen:kettle_mug`)

Living status log; updated whenever there are conclusions or new data, so work can resume
after an interruption.

## Goal (user request)
- Add a mug (cylinder) to the kettle env/task; prompt: grasp the kettle and pour into the mug.
- Model water with small particles; test the task end-to-end; commit + push to `tracking`.
- Keep `prompts/skills/pour-from-held-container/SKILL.md` short and simpler.

## Decisions
- **Genesis rejected for this task** (measured): Genesis 1.3.3 is in the `vlm_genesis` conda
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
