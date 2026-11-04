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
