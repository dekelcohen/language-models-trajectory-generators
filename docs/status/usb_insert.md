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


## 2026-11-09: generalized to plug insertion (not USB-specific)
- Scene `sim_envs/genesis/usb.py` (task `usb`) is parametric, defaults = the original USB-A scene:
  `PLUG_TYPE=usb_a|usb_c|hdmi|peg`, `PORT_FACE=left|right|back`, `PORT_DIR_DEG` (laptop turned so the face
  normal points there; port opening kept at the same reachable spot), `PORT_U`, `PLUG_YAW_DEG`,
  `PORT_DISTRACTOR=<type>` (a second keyed port: a foreign plug stops at 3 mm), `PLUG_SCENE_SEED` (random,
  headings limited to faces the fixed head camera sees >= 15 deg). Retention/success in the port frame.
  Smoke-tested (`usb_pipeline_e2e.py --stage scene`, env `vlm_genesis`): default, hdmi/back/10 deg + usb_a
  distractor, seed 3, peg/right/200 deg, usb_c/seed 7 all build with the plug resting on its pedestal.
- Perception: `estimate_opening` works for any face heading; `estimate_face_normal(point)` (head depth) and
  `face_normal_from_touches` give the insertion axis. NEW perception dependency: the head camera must see
  the port face's depth (else two touches on the face give the axis).
- Control: `insert_with_touch(insert_force=, opening_width=)`; the skill adapts them per retry.
- Not yet run end to end with the LLM on the new layouts (next step).

## 2026-11-10: regression after removing the overfitted skill content (commit 76177a6)
Default USB-A scene, opus-5.5 LLM / terra perception+reviewer, `--touch --attempts 3`. Each run hit a
different general gap that the deleted scene-specific text had been papering over:
| run | result | root cause | fix (general, no scene numbers) |
|---|---|---|---|
| p1 | FAIL at pickup | the box merged the plug and its pedestal; the LLM assumed the plug lay on the floor | skill step 1: a box top is a guess when the object sits on a support; grasp from the measured top |
| p2 | FAIL | circular `body_h = 2 * grasp_depth` gave a 0 tool z offset | skill: housing-height priors per connector family + ban on deriving it from the grasp depth |
| p3 | FAIL | pointing scatter along the face read as two openings; point insert at the wrong one, blind back-off flung the plug | skill: points a few cm apart at one height = ONE opening unless an image shows two; line-search with `estimate_opening` |
| p4 | FAIL | "right-side edge" (head-image words) sent the LLM to the far face; plug jammed into the screen | skill: resolve side words by measured positions (face nearest the pickup); carry around/above the device |
| p5 | FAIL | wrist pointing scattered 16 cm along the face; the hard-coded +/-4 cm search cap stopped 1 mm short | `touch/opening_estimate.py`: `MAX_SEARCH_HALF=0.08`, note asking for more head-on views when the along-face spread > 5 cm; test `test_wide_scatter_widens_the_search` |
| p6 | FAIL | height hits bimodal (0.045-0.054 vs 0.070-0.072); the median fell between the clusters and rejected all | `estimate_opening`: height = the densest z cluster; test `test_bimodal_heights_keep_the_densest_cluster` |
| p7 | FAIL | the low cluster was the HELD PLUG: in straight-on wrist views it covers the port and terra points at it (same pixel ~(127,92) in every view) | `locate_with_wrist_camera`: prompt "never point at an object held between the fingers" + while holding, drop hits within 10% of the image of the projected fingertip centre (proprioception + wrist calibration) |
| p8 | FAIL | estimate right (z 0.069 = truth), but `search_half` = farthest hit + half a width: the sweep ended on the far edge, the edge finder never got the near one | `estimate_opening(opening_width=)`: reach + a full width; test `test_line_search_reaches_both_edges_of_the_far_opening` |
| p9 | FAIL | line search found the opening (5.4 mm in) but stalled: tool offset 16.7 mm, the finger pads (~9.3 mm half-width) hit the face first; plug lost after | skill: grasp the middle of the HOUSING, `off_a` >= tip length + 1.2 cm else regrasp; `insert_with_touch` returns/prints `finger_clearance_m` (`config.FINGER_PAD_HALF_WIDTH`) |
| p10 | FAIL | held-plug filter worked, estimate right, tool offset 24 mm, BUT the plug was carried back-to-front (tilt 180 deg): the wrist "metal tip" hit was on the stand (z 0.028, 3 cm below the plug top) and only 4 mm from the body hit, so tip - body had the wrong sign; the body met the face, the probe "advanced" past the plug length and the plug was flung | skill step 1: trust the tip direction only if both hits are ON the part (z near its top) and >= ~1 cm apart; else re-view along its length |
| p11 | FAIL | tip direction right (tilt 0.2 deg), start error 4 mm, but the grasp was 17.7 mm from the tip; the LLM's range check was `0.015 <= off_a` (prose lower bound ignored): stalled at 2.6 mm, plug pivoted 28 deg | skill: grasp gate as CODE in the step 3 block (`tip_len + 0.012 <= off_a <= expected + 0.005` else regrasp) + tip/housing length priors in step 1 |
| p12 | FAIL | gate fired (off_a 16.5 mm) but the planner's subtask text said "do NOT re-grasp", so the LLM inserted anyway and the plug pivoted 66 deg | skill: the gate regrasp is required even if the task text says not to re-grasp (that only means do not drop it) |
| p13 | FAIL (closest) | gate fired, regrasp shifted 12 mm (true eef->tip 25.3 mm), line search found the opening (final tip error 1.4 / -1.7 mm), but under the ~18 N push the plug slid 9 mm back in the fingers; next attempt flung it | open: grip friction vs insert force (controller/physics, see scripted A/B) |
| v1 HDMI + USB-A distractor | FAIL | pickup OK, gate + regrasp OK (true eef->tip 25.1 mm), plug never lost; but the opening estimate was 48 mm along the face past the HDMI port (beyond the USB-A distractor) -> +-15 mm search exhausted; the re-estimate landed at z 0.018 (port at 0.0625) | open: wrist pointing for the target port among keyed distractors (perception) |
- Pickup has worked in every run since fix 1; tool offset is correct (+1.8 mm) since fix 2.
- Scripted A/B (`research/touch/usb_pipeline_e2e.py --stage all --calib touch --grasp-touch --sweep 0.025`):
  the pre-rewrite scene (worktree at 76177a6~1) seated 1/3, the new parametric scene 0/4. Collision
  geometry, physics constants and plug URDF are identical (only inertia differs; restoring the old inertia
  did not help, reverted). Traces match until contact and then diverge chaotically, so the insert
  controller was already marginal; the scene rewrite is not the systematic regression - the skill text is.
- Open: after a slip, the plug is sometimes flung metres away (contact blow-up); not fixed.
- Harness check of the finger limit (`outputs/touch/e2e_wiggle1`, err y -13 / z +0.8 mm, sweep 32 mm):
  found and seated the opening, but the harness grasp (eef->tip 19.3 mm) stops at 10.0 mm = exactly the
  success threshold -> "not inserted". A wiggle-on-stall and a final seat push were tried in
  `touch/insertion.py` and reverted (the stall is the fingers, not alignment).
- No new perception dependency added.
- Rollouts: `outputs/2026-11-10/plug_usba_*` (HF `rollouts/2026-11-10/`).

## 2026-11-11 - lone-catch centre guess, `config` shadowing crash

- **Root cause of the far-error misses (p13, harness):** the line search is a one-directional raster pass, so it only feels the edge it is LEAVING.
  - With a -29 mm along-face aim error the harness (`--err=-0.029,-0.0008 --sweep 0.042 --rows 5`) logged partial catches only at the far edge (34.5 / 36.8 / 34.6 mm; centre ~29 mm). The two-catch bracket never formed, and the plug was lost. Grip 20 N vs 40 N made no difference.
- **Fix** (`touch/insertion.py`): on a lone catch, guess the centre half an opening width BACK against the sweep direction (`sweep_sign`), then raster a short column there.
  - The column is +-`guess_col_half` (2 mm) along x +-`cross_half` across, for at most `max_guess_cycles=2` guesses, counted apart from the edge cycles.
  - The guess comes from the measured catch + the vision opening width only; no scene numbers.
  - Harness: the guess landed at 28.0 mm and the plug went in 10.7 mm. Then it pivoted 89 deg in the grasp, with a tip ~1-3 mm off in height after many probes, so the run is still "not inserted". The far-error case remains marginal.
- **p14 (LLM, USB-A): the plug SEATED** (11.9 mm deep, 0.2 mm lateral, tilt 2.5 deg) through the centre guess (`outputs/2026-11-11/plug_usba_genesis_guess_seated_then_config_crash_FAIL`).
  - Then `insert_with_touch` raised `AttributeError: 'dict' object has no attribute 'FINGER_PAD_HALF_WIDTH'`. Its local tuning dict was named `config`, which hid the `config` module.
  - The LLM re-ran the whole block and dragged the seated plug out with closed fingers.
  - Fixed: the dict is renamed `cfg`, and `max_s` now also budgets the guess columns. Regression test `test_insert_with_touch_with_tuning_args_does_not_shadow_the_config_module`.
- No new perception dependency added.
- **p15 (LLM, USB-A): FAIL.** The wrist-pointing estimate was 45 mm off along the face. At the first catch the plug pivoted 81 deg in the grasp, so the attempt was lost (HF `rollouts/2026-11-11/plug_usba_genesis_45mm_err_pivot_FAIL`).
  - The harness and p15 also showed a "lone catch" on the very first probe, before any sweep motion. That is not an edge being left, so the guess is now gated on the sweep having moved (`sweep_s > col_pitch`). Test `test_lone_catch_guesses_the_centre_half_a_width_back_but_not_before_the_sweep_moved`.
- Open: (1) the plug pivots in the grasp on repeated partial catches against the face (far aim errors); (2) the along-face wrist-pointing error sometimes exceeds 3 cm.
- **p16 (LLM, USB-A): FAIL (stopped at the time limit).** The insert started at -31 mm along and -3.9 mm in height. The touch-measured in-hand z offset was +1.3 mm vs a true -3.6 mm, about 5 mm off for a 4.5 mm-tall slot. The line search kept catching the far edge (along 37 mm) and could not seat (HF `rollouts/2026-11-11/plug_usba_genesis_inhand_z_err_5mm_FAIL`).
  - Open (3): the in-hand height offset from touch can be ~5 mm off. Re-check it, or widen `search_cross`, when catches repeat at one along position without seating.
