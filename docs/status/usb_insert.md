# USB insert (Genesis): status

Task: `--task usb --sim genesis -c "Pick up the USB plug and insert it into the laptop's USB port"`.
Skill: `prompts/skills/touch-guided-insertion/SKILL.md`, primitive `insert_with_touch(axis, tool_length, max_s, grip_force, tool_offset=None, search_radius=None, search_line=None, search_half=0.02)`; touch calibration via `move_until_contact()` / `get_grasp_info()`.

**Status: OPEN.** Pick-up and touch calibration work in LLM runs; the remaining blocker is finding the port (which face, where along it). The scripted harness seats the plug end to end with touch calibration and the line search.

## Runs (2026-11-02, rollouts under `outputs/2026-11-02/`)

| run | models (lm / perc / review) | result | root cause | folder |
|---|---|---|---|---|
| usb_gn1 | claude / terra / terra | FAIL, 12+ `search_exhausted` | Skill told the LLM to aim at "opening z + 0.06" (the pick-up rule); after the grasp the plug sits at about the EE height | `usb_terra_FAIL_inhand_offset` |
| usb_tt2 | claude / terra / terra | FAIL: 2× timeout, `no_contact`, plug dropped, then a ~40 min loop with no `task_completed()` | Wrist cam could not see the held tip; fallback offset (0.025, 0, 0) is off by ~7 mm in z; the LLM refused to end the attempt | `usb_ALLterra_FAIL_vertical_offset_then_drop_idle_loop` |
| usb_tt3 | claude / terra / terra | FAIL: 5× timeout, 4× `no_contact`, 10+ `search_exhausted` in ONE attempt (66 min, never reviewed) | Wrist locate of the held tip gave impossible offsets (tip 13 cm above the hand), which the LLM rightly rejected. The height scan (0, +5, −5, +10 mm) plus an alternate port did not land within the 3 mm spiral | `usb_ALLterra_FAIL_hidden_inhand_offset_endless_attempt` |

## Fixes made
- Skill step 3: measure the in-hand offset with the wrist camera from 2 views (`offset = tip - get_robot_state()["eef_pos"]`), aim with `target = opening - offset - axis*0.004`, and pass `tool_offset=offset`.
- Re-measure if the first contact z is off by more than 5 mm.
- If the tip is not visible, scan heights 0, +5, −5, +10 mm.
- `insert_with_touch(..., tool_offset=...)`: a world vector from the EE to the tip; it overrides `tool_length`.
- `get_robot_state()` / `gripper_holding()` are exposed in the exec scope.
- Agent loop: an idle guard (`max_idle_agent_turns=10`) and a per-attempt cap (`max_agent_turns_per_attempt=25`), so endless retries get reviewed.

## 2026-11-03: touch calibration, carry speed cap, line search

Scripted harness (`research/touch/usb_pipeline_e2e.py --grasp-touch --regrasp --calib touch`) seats the plug end to end:
- in-hand offset from touch (top touch before the grasp + face touch beside the port), not vision;
- `--sweep 0.025` (new): seated with an 18 mm and a −20 mm y aim error (+1 mm z), found at 14.6 / 22 mm along the face.

LLM runs (claude / terra / terra, Genesis; folders under `outputs/2026-11-03/`):

| run | result | root cause | fix | folder |
|---|---|---|---|---|
| usb_g2 | FAIL | grasp settled 4.4 mm low; LLM then invented an edge scan with the held tip | skill: re-command the grasp until eef within 1 mm; no pressing / edge scans with the held tip | `usb_ALLterra_FAIL_grasp_settled_low` |
| usb_g3 | FAIL, plug dropped | body height taken from the detection box (incl. the stand) → grasped 22 mm low; unbounded settle-correction loop | body height = housing only (USB 8 mm); `get_grasp_info()["grasp_depth_below_top"]` check; settle correction ≤3×, <1 cm | `usb_ALLterra_FAIL_grasp_on_stand_drop` |
| usb_g4 | FAIL, plug flew out | unthrottled transits between wrist-view poses (~4 m/s commanded) | API caps hand speed at 0.3 m/s while the gripper is closed (`config.CLOSED_GRIPPER_MAX_SPEED`); check `gripper_holding()` before calibrating | `usb_ALLterra_FAIL_fast_transit_drop` |
| usb_g5 | FAIL (timeouts) | plug held, offset correct (off_x 25.5 mm vs ~26.6), but terra's wrist pointing scatters ±2 cm along the face (several painted slot marks, real port ~3 px); spiral covers 3 mm | touch **line search**: `insert_with_touch(..., search_line=[0,1,0], search_half=0.025)` | `usb_ALLterra_FAIL_port_y_scatter` |
| usb_g6 | FAIL | the laptop texture has port marks on BOTH side faces; terra + LLM chose the +x face (no opening). Sweep correctly returned `search_exhausted` 3×, but the skill said "height wrong, retry" → same face again | skill: an exhausted full sweep at a verified height = no opening on that face → next candidate face (nearest the arm base first) | `usb_ALLterra_FAIL_wrong_face_sweep` |
| usb_g7 | FAIL | correct −x face; but the LLM aimed at the single outlier wrist candidate z 0.051 (3 others ~0.069-0.071, truth 0.069) and guessed deck z 0.058 for the face touch → 17 mm low, sweep exhausted | skill: opening z = median of all candidates (drop > 5 mm outliers); face touch 7 mm below the opening z (no separate deck estimate) | `usb_ALLterra_FAIL_outlier_port_z` |
| usb_g8 | FAIL | aim now right (median z 0.0686 vs 0.069). The pick held the plug 8.5 mm below its top (8 mm housing): the LLM's settle correction added the error to the MEASURED `pose` (= the old target) → same sag; the depth-check regrasp repeated the same command. The plug pivoted on the 4 N face touches (off_x 9 mm) and fell | prompt + skill: correct settle errors on the previous COMMAND; shift the regrasp by the depth error | `usb_ALLterra_FAIL_low_grasp_settle_bug` |
| usb_g9 | FAIL | pick OK (grasp depth 5.7 mm), offsets good (off_x 28.3, off_z 1.7 mm). Only 3 wrist candidates, all near the deck edge (z 0.073–0.077) → aim 4.6 mm high, y 2.1 cm off. Sweep rows (0.4 mm × 5) cover only ±0.8 mm in z → `search_exhausted` | harness test: does a z-band sweep recover a few-mm z error? (below) | `usb_ALLterra_FAIL_port_z_4mm_high` |
| harness hz1 | FAIL | Genesis harness, z err +2.5 mm, y −15 mm, default rows: 4 false drops (0.4–0.8 mm), never seated, 537 s | z error > ~1 mm defeats the line sweep → need a z band or better z localisation | `outputs\touch\e2e_z25` |
| harness hz2 | FAIL | same error, coarse rows (1.5 mm pitch × 7 = ±4.5 mm z): one partial entry 2.7 mm then jammed (edge), 706 s | z bands don't help → fix z localisation: skill rejects candidates within 3 mm of the wrist-located top edge, and re-views head-on if < 4 remain | `outputs\touch\e2e_z25b` |
| usb_g10 | FAIL | edge-reject rule worked: z from 10 candidates = 0.0676 (truth 0.069, 1.4 mm low vs g9's 4.6 mm high); y 0.497 (truth 0.4959). Pick + offsets OK. Two line searches exhausted (rows cover ±0.8 mm), then the agent moved to other faces | new `insert_with_touch(search_cross=)` widens the height band (0.002 = 11 rows) | `usb_ALLterra_FAIL_port_z_1p4mm_low` |
| harness hz3 | FAIL | z −1.4 mm, 11 rows: the sweep found the opening twice (entries 2.4 / 3.5 mm) but both jammed off-height, 797 s | open: sub-mm z, or z compliance during the insert push (let the tip settle into the slot) | `outputs\touch\e2e_zm14` |
| harness jam1 | FAIL | aim err y −2, z −1.4 mm (tip actually −2.1 / −2.2 mm after touch calib), 11 rows: `search_exhausted`. The "false drops" (0.5–1.8 mm) were partial catches: the tip centre on a side edge of the opening, corner dipping in. Catches at −4.6 and +8.1 mm along the line, midpoint +1.8 vs true +2.1 mm | edge finder (below) | `outputs\touch\e2e_jam1` |
| harness edge2 | **INSERTED** 21 s, depth 11 mm | same as jam1. Two side-edge catches → centre → column raster seated it | `InsertConfig.edge_*`, `col_*` | `outputs\touch\e2e_edge2` |
| harness edge3 | **INSERTED** 29 s | y −15, z +2.5 mm (hz1's failing error), 5 rows, `--calib touch`: seated straight from the raster | – | `outputs\touch\e2e_edge3` |
| harness edge4 | FAIL, **false "seated"** | y +12, z −2.5 mm, 5 rows: the tip caught the lower edge, the plug pivoted / slid back in the grasp, and the hand advanced 15 mm like a real insertion (eval depth −1.3 mm) | seat check: push the hand 2 mm sideways; a seated plug moves ≤ clearance | `outputs\touch\e2e_edge4` |
| harness ver2 | **INSERTED** 15 s | jam1 error again, with the seat check: moved 0.27 mm → seated | – | `outputs\touch\e2e_ver2` |
| harness ver4 | FAIL, correctly reported `slipped` | edge4 error: the seat check moved 2.4 mm → `slipped` (the plug had been lost from the grasp) instead of a false seat | open: push gentler on partial catches near a z-edge so the plug doesn't pivot out | `outputs\touch\e2e_ver4` |

**Edge finder** (`touch/insertion.py`, line sweep only):
- A blocked push that still sank ≥ 0.4 mm is a partial catch.
- Two catches 9–16 mm apart along the line (≈ one USB-A opening width) put the centre at their midpoint.
- A column raster there (±0.9 mm along, 0.3 mm pitch, ±4.5 mm across) finds the height.
- Up to 2 cycles, then the raster resumes.
- This finds the port height by touch: a flat face touch can't, because the plug is as big as the opening.
- Unit test: `test_edge_catches_locate_an_off_row_opening`.
- **Seat check** (`InsertConfig.verify_*`): before reporting `seated`, the hand is pushed 2 mm sideways at 3 N axial. Moving more than 0.8 mm → `slipped` (not in the opening). Real-robot feasible (proprioception only). Unit test: `test_grasp_slip_that_looks_seated_is_caught_by_the_sideways_check`.

**Videos of killed runs:** clips are written only when an attempt ends, so killed runs had no insertion video. `helpers/rollout_report.py` now encodes `videos/*_unfinished.mp4` from the trajectory frames (backfilled for every `outputs/2026-11-03/usb_*` folder).

Line search (`touch/insertion.py`, `InsertConfig.sweep_*`): raster passes ±`sweep_half` along `sweep_dir` at 5 mm/s, rows 0.4 mm apart across it (5 rows), same drop detection / probe / insert as the spiral. `api.insert_with_touch` extends `max_s` to fit.
Geometry (sim truth, for eval only): port (0.092, 0.4959, 0.069), axis +x, `TIP_X` = 0.0245.

## 2026-11-04: why no LLM run ever pushed the tip into the slot
Eval-only truth logging: `env.py` `_log_insert_truth` prints `[eval] insert start|end` (tip−port, true plug axis/tilt, true eef→tip vs `tool_offset`). Log only; never sent to the agent.

- **Root cause (every LLM run, g9/g10/r2):** the insert subtask prompt hard-coded "rotation value at 0" (`prompts/main_prompt.py`). The pick closes across the plug at rotation ±1.571, so the insert poses yawed the hand 90° and the plug pointed along y (r2 eval: tilt 90°, tip−port [−36, 5, −12] mm). The tip could never enter.
- **Fix:** the measured top-down yaw is now in `get_robot_state()["rotation"]`, the world-state log and the main prompt (`[INSERT EE ROTATION]`, plus a "held part turns with the hand, keep this rotation" hint). Skill step 1 has the same rule. Tests: `test_prompt_states_measured_rotation_while_holding`, `test_measured_topdown_rotation`.
- **Wrist views at carry rotation:** the palm and plug hide the laptop face (r3: 0 port candidates in 11 views). The skill now says to take the views at carry rotation ±1.571, then turn back before touch.
- **Face order:** finish a face before viewing the next; skip faces > 0.8 m from the base.

| run | result | package (`outputs/2026-11-04/`) |
|---|---|---|
| usb_r1 | perception said "upright"; grasped 21 mm low on the pedestal → plug ejected | `usb_terra_repro_eval_1` |
| usb_r2 | root cause found: plug yawed 90° at insert | `usb_terra_repro_eval_2` |
| usb_r3 | rotation fix works (tilt 0°), but no wrist port candidates → insert at a guess 58 mm off → jammed | `usb_terra_rotation_fix_3` |
| usb_r4 | rotation kept; −x face candidates found but abandoned; insert on the far face (no port, reach limit) → `search_exhausted`, plug lost; `holding` false positive (fingers stalled on the laptop) | `usb_terra_wristview_fix_4` |

Open: wrist port detection is sparse (terra); `holding` uses finger width only, so fingers jammed on a surface look like a grasp.

## 2026-11-05: insert-subtask root causes (Genesis, `outputs/2026-11-05/usb_ALLterra_rootcause_1`)
- **Port found:** pick OK (width 15.8 mm). At wrist rotation −π the wrist view found the TRUE port: (0.0907, 0.4967, 0.0708) vs truth (0.092, 0.4959, 0.069).
- **Root cause 1: plug flung by a wrist spin.**
  - "Turn back to carry rotation" went −3.14 → 1.571, a 270° spin in 6 poses with no rotation cap.
  - It produced 405 N of real inertial load and a jam trip; the plug was lost.
  - Fix: while holding, `execute_trajectory` caps rotation at `CLOSED_GRIPPER_MAX_ROT_SPEED` = 1 rad/s, shortest way round (−90° here). Test: `test_closed_gripper_caps_unthrottled_moves`.
  - The jam rule now also needs a lag behind the command, so a pure wrist turn is no jam.
- **Root cause 2: regrasps closed on air (5×).**
  - The plug fell to the floor next to its slab.
  - Wrist locate put it at z 0.036–0.042, but the top touch made contact at z 0.050 (the slab edge), 8–14 mm ABOVE the located point.
  - The LLM grasped "5–9 mm below the top" = beside the slab → the width guard tripped "closed on air" on every lift.
  - The guard was correct each time (no false positives).
  - Fix: touch skill sanity rule: a contact > 5 mm above the located part means a support or neighbour was hit; re-locate from above.
- **u2 after both fixes:** SUCCESS, the first true LLM insertion (`outputs/2026-11-05/usb_ALLterra_rotcap_SEATED_2`).
  - Wrist views at both rotations; port candidate (0.0905, 0.489, 0.069).
  - Face touch, then line-search insert → `seated`.
  - Eval: tip 8.2 mm inside the port, tilt 0°.
  - The reviewer accepted.
  - Pick grasp depth −0.4 mm (shallow) still held.
- Still open:
  - `test_touch_sim_runner::test_finds_offset_hole_and_seats...` gives `over_force` (pre-existing);
  - one success out of 2 runs, so not yet stable.

## 2026-11-08: u3 root causes (Genesis, claude-opus-5.5 / terra / terra, code before the fixes below)
Run: `images_usb_u3`, packaged to `outputs/2026-11-08/usb_u3_dropped_output_FAIL`.
- **Pick (subtask 1):** success on attempt 1 (00:20, ~4 min).
- **Root cause 1: the LLM never saw its own port estimate.**
  - The view block printed > 2000 chars (all candidates of 4 views), and `execute_python_blocks`
    dropped any block output >= 2000 chars WHOLE, so the LLM got "executed successfully with no output".
  - It fell back to the first single candidate (0.096, 0.424, 0.055): 1.4 cm low, 7 cm off along the face.
  - The 9 on-face candidates it had collected were good: z 0.066-0.072 (median 0.070, truth 0.069).
  - Fix: long output is truncated to head + tail (6000 chars) with an "N chars omitted" marker, never
    dropped (`agent_runner.PRINT_OUTPUT_MAX_CHARS`, test `test_long_print_output_is_truncated_not_dropped`).
- **Root cause 2: hand-rolled candidate aggregation.**
  - Every LLM wrote its own median / edge-reject code; edge hits were often BELOW the port hits
    (bad edge pointing), which the skill's "drop within 3 mm of the edge" rule turns into dropping real hits.
  - Fix: deterministic `estimate_opening(port_cands, edge_cands, axis)` (`touch/opening_estimate.py`):
    face-plane filter, median z with 4 mm rejection, edge hits used only when the edge is ABOVE the port,
    returns `search_line` / `search_half` / `search_cross` for `insert_with_touch`. On u3's real
    candidates: z 0.0703 (1.3 mm), y 1.2 cm off, inside the +-4 cm line search
    (`tests/test_opening_estimate.py`). The skill now calls it instead of describing the math.
- **Insert attempt 1:** line search at the wrong height (z 2.3 cm low), only along y -> hooked the
  base corner (933 N), plug dropped -> `slipped`.
- **Attempts 2-3:** the plug was on the floor; the LLM chased the fixed stand at (-0.125, 0.443) as if
  it were the plug, grasps closed on air at ~27 N. (Not fixed yet: the re-pick needs the same
  "contact > 5 mm above the located part = a support" rule applied to the stand.)

## 2026-11-08: u4, false "slipped" from the seat check (fixed)
Run: `images_usb_u4`, packaged to `outputs/2026-11-08/usb_u4_false_slipped_seatcheck_FAIL` (stopped manually after attempt 1).
- **The u3 fixes worked.** `estimate_opening` put the port at (0.091, 0.504, 0.071) / (0.091, 0.488, 0.070)
  vs truth (0.092, 0.496, 0.069), i.e. < 1 cm, and the LLM used it. Pick OK on attempt 1.
- **Insert:** the edge finder found two side-edge catches, the column raster seated the plug
  (advance 7.2 mm), then the 2 mm sideways seat check moved 1.51 mm > `verify_tol` 0.8 mm -> `slipped`.
  The hand ended at the port; the next motion pulled the hand off the plug (held by the port) and it fell.
- **Harness repro** (`usb_pipeline_e2e.py --grasp-touch --regrasp --calib touch --err 0.0075,0.0006 --sweep 0.032`,
  `outputs/touch/e2e_u4err.txt`): truth at the check was **depth 11.5 mm, lateral 0.2 mm (seated)**,
  but the check moved 0.87 mm -> `slipped`. A false negative.
- Seat-check data: seated 0.27 (ver2) / 0.87 (e2e_u4err) / 1.51 mm (u4); lost plug 2.4 mm (ver4).
  **Fix:** `InsertConfig.verify_tol` 0.8 -> 1.8 mm (the slip unit test, a full 2 mm move, still trips).
- **Eval logging:** `env._touch_insert` now logs `[eval] insert phase verify|done` (truth at the
  seat check), via the new `run_touch_insert(on_phase=...)` hook. The old end-of-call truth came after
  the back-off, when a seated plug could already have fallen out. Log only, never sent to the agent.

## 2026-11-08: u5 search window missed the port (fixed); u6 SUCCESS
- **u5** (`outputs/2026-11-08/usb_u5_search_window_miss_FAIL`): `estimate_opening` kept hits at y 0.462 /
  0.4713 / 0.498 (truth 0.496) -> median 0.4713; the old `search_half` = spread/2 + 5 mm = 23 mm stopped 2 mm
  short of the port -> `search_exhausted` twice; the 2.5 cm back-off then tore the plug from the fingers
  (237 -> 3.6 N, Genesis flung it ~1 m).
  **Fix** (`touch/opening_estimate.py`): `search_half = clamp(max|hit - median| + 6 mm, 15, 40 mm)`, i.e. the
  search always reaches the farthest kept hit (`test_u5_search_reaches_the_farthest_kept_hit`).
- **u6** (`outputs/2026-11-08/usb_u6_search_fix_SUCCESS`, opus-5.5 + terra perception, 12 min):
  estimate y 0.4773 +- 0.021 (covers the port), seated on attempt 1: plug tip (0.100, 0.496, 0.069) vs port
  mouth (0.092, 0.496, 0.069) = 8 mm in, released and retreated; the reviewer accepted.
- Also fixed in `agent_runner`: a long print output was dropped silently (now head + tail), and LLM code
  calling `raise SystemExit` killed the whole run (now caught, see `document.md` exec table).


## 2026-11-08 (night): model variants (same pipeline, `--attempts 3 --max-planner-iter 5`)
| run | LLM / perception | result | notes |
|---|---|---|---|
| u6 | opus-5.5 / terra (defaults) | SUCCESS, 12 min | seated attempt 1 (see above) |
| u7 | gpt-6-astra / gpt-6-astra | FAIL | `outputs/2026-11-08/usb_u7_astra_llm+perception_FAIL` |
| u8 | gpt-6-sol / terra | FAIL | held the plug, but the planner spent 5 iterations on camera-view subtasks |
| u9 | opus-5.5 / gpt-6-sol | FAIL (90 min budget) | plug slipped; insert searched y 0.58-0.66 (port at 0.496), then a line search with an empty hand |
- Only the defaults (opus LLM + terra perception) inserted. Swapping either model hurt: sol/astra as LLM
  wander in the planner; sol as perception put the port box ~8 cm off. Keep the defaults.
