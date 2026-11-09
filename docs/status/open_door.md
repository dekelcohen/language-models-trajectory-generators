# Status: open door, no touch (`door`, `door:hidden`)

Living status log so work can resume after an interruption. Branch `tactile`.
Last update: 2026-11-03 (stall feedback, pose frame, loop guard, reviewer evidence).

## Request
"Open the door" (no touch) never succeeded, even without the grey-cylinder occlusion; both
versions worked in September. Find why the robot can't grasp, slips at the start of the
pull, or pulls the wrong way, then fix it. Keep the lever collision (realism) in PyBullet
and Genesis, then test `door:hidden`. Budget was 15 `main.py` runs (all used).

Reference command (no `--sim`, so PyBullet; add `--sim genesis --no-touch` for Genesis):
```
python main.py -lm copilot-claude-opus-5.5 --seg-provider moondream --task door -r franka \
  --planner-perception-vlm or-google/gemini-3.8-flash --review-provider vlm:or-google/gemini-3.8-flash \
  --attempts 2 --max-planner-iter 4 --reasoning-effort medium --delete-images -c "open the door"
```

## Root causes and fixes
No code regression was found. The task failed for several independent reasons:

| # | Cause | Fix | Where |
|---|---|---|---|
| 1 | Old URDF lever had no collision: the jaws closed on air. | Lever collision added (earlier, `bda2b3e`). | `my_assets/adroit_door/adroit_door.urdf` |
| 2 | PyBullet latch was held by a 200 N·m position motor, so every press stalled and pinned the grasp. | Soft 3 N·m return spring, like a real lever. | `sim_envs/pybullet/door.py` (~l.105) |
| 3 | A fixed wrist yaw twists the lever out of the jaws after ~20° of arc. | Skill: turn the wrist with the door (top-down `rotation += a`, side `approach_yaw += a`). | door skill, section 2 |
| 4 | Moondream labels the frame post beside the handle (latch-side jamb, ~(0.005, 0.11)) as "door hinge"; the true hinge is about (-0.40, -0.36). The LLM then pulled the wrong way. | Skill: a "hinge" within ~0.3 m of the handle is the latch jamb; the hinge is the far edge the lever tip points to. Fallback: 0.4-0.8 m along the tip direction (too far is safer than too near). | door skill, sections 1-2 |
| 5 | The handle-detection centre is biased 4-7 cm toward the rose; jaws hit the rose disc. The LLM sometimes shifted the grasp the wrong way. | Skill: grasp mid-bar, shift ~5 cm toward the tip; the tip points away from the latch jamb (check the sign). | door skill, section 2 |
| 6 | Detection with the arm in front of the door (e.g. after the pole subtask) gives junk positions, and retries reused them. | Skill: clear the arm, then detect; after `closed_on_nothing`, detect again. | door skill, section 2 |
| 7 | Genesis IK kept inaccurate continuity solutions (3-20 cm off) or jumped branches, flinging the arm. | Local seeded solves around the current qpos; pick the accurate one (<1 cm, <0.05 rad) with the smallest joint jump. | `sim_adapter/genesis_adapter.py` `inverse_kinematics` |
| 8 | The reviewer was told "the door did not start to open" when the fingers slipped, but slipped fingers moving along an arc still push the door. | `slip_may_push()`: no veto, no "did not move" retry hint, neutral wording. A test was added. | `helpers/pull_proprio.py`, `tests/test_tracking_review_evidence.py` |
| 9 | Skill line "lever parallel to the door face → grasp from the side" (added for Genesis touch) pushed every PyBullet run to a side grasp; jaws landed near the rose and the press stalled. | Lever stays top-down (step-4 offset into the gap); side only on 2b trigger 3. Tip direction computed from the panel's far edge, printed. | door skill, section 2 (`4548ee1`) |
| 10 | Toppled pole rolled forever (no rolling friction): x 3 → 100 m, then fell off the plane. | `rollingFriction=0.05`, linear/angular damping 0.5; it now stops ~0.8 m away. | `sim_envs/pybullet/door.py` `_load_pole` (`4548ee1`) |
| 11 | Pole-clearing subtask: VLM said success, but the pull-proprio veto overruled it (the prompt says "door"; the hand slipped off the pole = "handle never travelled"). The planner then re-ran the clearing subtask twice against a pole already on the floor. | `is_articulated_task`: a relocate subtask (move / out of the way / place / topple...) is articulated only with an explicit open/close intent on a part ("open the door"). Checked on all 27 subtask prompts of the 2026-10-29 rollouts; test added. | `helpers/pull_proprio.py`, `tests/test_tracking_review_evidence.py` |

## Results (door angle from sim state; for evaluation only)
| Run | Sim / task | Outcome |
|---|---|---|
| pb2 | PyBullet `door` | Success on attempt 1 (~32°). |
| pb7 | PyBullet `door` | Success; grasp at lever mid, handle moved ~37 cm. |
| gn6 | Genesis `door --no-touch` | Success; handle moved ~37 cm. |
| pb8, pb9 | PyBullet `door:hidden` | Pole cleared and the door opened 60-100°, yet the reviewer (gemini-3.8-flash) rejected every attempt. |
| gn1-gn5, pb3-pb6 | Before fixes 4-8 | Failed: `closed_on_nothing`, wrong hinge, IK fling. |
| door_pb (10-29) | PyBullet `door`, before fix 9 | Side grasp near the rose, press blocked every attempt; killed. |
| door_pb2 (10-29) | PyBullet `door`, after fixes 9-10 | Success on attempt 1, top-down; handle moved ~27 cm. |
| door_hidden_pb2 (10-29) | PyBullet `door:hidden`, after fixes 9-10 | Pole toppled and stopped on the floor; door opened (handle ~29 cm), top-down. Pole subtask wrongly vetoed → fix 11. |

Videos + logs: `outputs/rollouts_20261029_final/<run>/`.

Harness numbers (`research/touch/door_pull_e2e.py`, scripted, no LLM):
- PyBullet press + arc + yaw-follow: 44° (side approach), 40° (top-down).
- Hinge-error sensitivity: hinge 5 cm too far gives 52°; 5 cm too near gives 30°.
- Genesis touch pull: still 61° after the IK change.

## Open issues / next steps
1. **Reviewer false negatives on `door:hidden`.** gemini-flash says "door remained closed"
   even when it is open 60-100°. When opened that far the panel faces the camera, and on a
   retry the start image is already open. Options:
   - a stronger review model;
   - feed tracking-based door-angle evidence to the reviewer;
   - compare against the first (pre-task) image instead of the attempt start.
2. **Grasp still misses sometimes (`closed_on_nothing`), mainly after the pole subtask.** Bad
   moondream detections with the arm or pole in view. A better detection retry is planned.
3. **Re-verify Genesis `door:hidden`.** Not run after fixes 4-8.
4. **Pull proprio covers only labelled pull segments.** Arc segments printed without a
   matching label are not counted (pb7 reported `part_did_not_move` while the door opened).
5. **Touch-guided pull is Genesis only** (`open_with_touch()`, `docs/plans/touch_pull.md`).
   It is unaffected by these fixes.

## 2026-11-02: terra everywhere, reviewer / agent-loop fixes

Models:
- `-lm copilot-claude-opus-5.5` (default) or `-lm copilot-gpt-6-sol`.
- Perception (`--planner-perception-vlm`) and reviewer (`--review-provider vlm:...`) now default to `copilot-gpt-6.1-terra`, which is cheaper than gemini-flash.
- Copilot models accept images only, so the reviewer gets key frames instead of video (see "Key frames" below).

Runs (PyBullet; rollout folders under `outputs/2026-11-02/`):

| run | models (lm / perc / review) | result | folder |
|---|---|---|---|
| door_pb1 | claude / gemini / gemini | success | `door_gemini_baseline` |
| door_pb2 | claude / gemini / terra | success after one false negative | `door_terra_falseneg_then_success` |
| doorhid_pb1 | claude / gemini / terra | FAIL: pole loop | `door_hidden_terra_baseline_FAIL` |
| doorhid_gm1 | claude / gemini / gemini | FAIL: the unreviewed final attempt was accepted | `door_hidden_gemini_FAIL_unreviewed_final` |
| doorhid_pb3 / pb4 | claude / gemini / terra | **success** | `door_hidden_terra_newcode_SUCCESS`, `door_hidden_terra_finalreview_runbaseline` |
| doorhid_tt1 | claude / terra / terra | **success** in 8 min, door 32.6° | `door_hidden_ALLterra_perc_review` |
| door_g6 | gpt-6-sol / terra / terra | FAIL: door 27-31° rejected, then an idle loop | `door_gpt6sol_FAIL_reviewer_process_cond_and_idle_loop` |
| door_g6b | gpt-6-sol / terra / terra | **success**, attempt 1, 16.4° | `door_gpt6sol_ALLterra_after_fixes` |

Root causes and fixes:
- **Pole re-removal loop (door:hidden).** The reviewer saw no change after the pole was already gone, and did not credit the pole's displacement.
  - Displacement evidence: tracked object start→end shift, with run-level baselines, so a pole moved in an earlier attempt still counts.
  - The planner's final attempt is now reviewed instead of accepted unreviewed.
- **Reviewer demanded process proof (gpt-6-sol plans).** The plan said "Success = door open AND lever actuated", and terra wanted proof of the lever.
  - `REVIEW_OVERALL_COMMAND_SECTION`: a reached goal state implies its process conditions.
  - Planner-chosen amounts (e.g. "lift 50 cm") are approximate: within ~25 %, or as far as the arm can reach, counts as met.
- **LLM refused to call `task_completed()` after a self-judged failure** (74+ turns).
  - The idle guard ends the attempt for review after `config.max_idle_agent_turns` (10) turns without robot motion.
  - `config.max_agent_turns_per_attempt` (25) ends an attempt that keeps moving forever (usb_tt3: 66 min in one attempt).
  - The prompts say `task_completed()` ends the attempt for review and is not a success claim.
- **Eval truth.** The handle moved only 2.5 cm in a "success". Door `get_state()` now has eval-only `door_angle_deg` (hinge joint), which is logged in `[world-state]`. Never put it in prompts.

Key frames for image-only reviewers (`helpers/review_frames.py`):
- Old: a fixed stride (head every 5th, wrist every 7th) from the attempt start, truncated from the end. The final wrist frames were cut first, and one USB review got only 4 frames.
- New: a change-aware selector over the frames on disk.
  - It always keeps the first frame, the final frame and one ~3 frames before the end.
  - The remaining budget follows cumulative visual change (48×48 grey thumbnails, with a floor so static stretches still get frames).
  - Budgets: `config.review_key_frames_head=24`, `review_key_frames_wrist=14`, under `max_allowed_vlm_images`.
  - Frames are labelled "head cam step N" / "wrist cam step N", and the last one is marked "<- FINAL STATE".
  - Over the cap, wrist frames are thinned from the middle; the end is never cut.

Terra vs gemini:
- Terra worked as perception and reviewer on door, door:hidden, grasp, slide_cabinet and kettle.
- Terra is more literal: it rejected achieved goals over planner wording until the review-prompt rules above were added.
- No terra-specific misperception was seen. The slide_cabinet hinge confusion also happens with gemini.

## 2026-11-03: stall feedback, pose frame, loop guard, reviewer evidence

All runs: `-lm copilot-claude-opus-5.5`, perception + reviewer `copilot-gpt-6.1-terra`, PyBullet.
Folders under `outputs/2026-11-03/`.

| run | result | folder |
|---|---|---|
| door_a1 | success, attempt 1 | `door_ALLterra_after_reviewer_fixes_1` |
| door_a2 | success, attempt 2 (attempt 1 a correct reject) | `door_ALLterra_after_reviewer_fixes_2` |
| door_a3 | success, attempt 1 | `door_ALLterra_stall_feedback_3` |
| door_a4 | success, attempt 3 at 54.6° (attempts 1-2 were correct rejects at 14.8° / 6.5°) | `door_ALLterra_pose_frame_fix_4` |
| doorhid mask / disocclusion / 3 | success; the pole was removed once (no re-removal loop) | `door_hidden_ALLterra_robotmask`, `..._disocclusion`, `..._stall_speedcap_3` |
| doorhid_4 | FAIL: an endless push loop (door actually ~open) | `door_hidden_ALLterra_FAIL_endless_push_loop` |
| doorhid_5 | door opened 18.6° (reviewer TP), but `plan_completed()` came in the reply after the iteration cap and was never run | `door_hidden_ALLterra_loop_guard_5` |
| doorhid_6 | **success**: pole cleared once, door 73°, both subtasks on attempt 1, planner completed | `door_hidden_ALLterra_planner_cap_fix_6` |
| grasp, slide_cabinet | regression OK | `grasp_ALLterra_*`, `slide_cabinet_ALLterra_regression_1103` |
| slide_r3 (after loop guard / planner cap / review rule) | SUCCESS on attempt 3 (door slid 0.063 → -0.224 m); no regression. An earlier r2 used task id `slide_cabinet` (no `franka_kitchen:` prefix = a different scene) and was discarded | `slide_cabinet_ALLterra_regression_final` |
| door_a5 | SUCCESS on attempt 1 (`part_moved_with_hand`) | `door_ALLterra_stability_5` |
| doorhid_7 | SUCCESS: pole cleared once (attempt 1), door opened on attempt 1. An arm-clear prep subtask was rejected twice (stalled rises left the arm across the door frame); the planner went on correctly | `door_hidden_ALLterra_stability_7` |

Root causes and fixes:
- **Unreached press target chained as the next start.** The hand stalled on the lever, the code continued from the planned pose, and the grip was lost.
  - `execute_trajectory` returns the measured outcome (`reached`, `stalled`, `pose`, `holding`) and prints `STALLED` (shortfall > 2 cm and < 5 mm settle motion).
  - The skill re-plans from the measured pose.
- **Frame bug (pose vs eef).** `eef_pos` is the fingertip centre; trajectory poses are 6 cm up the approach axis. The skill said to hold `eef_pos` z, so every swing pressed 6 cm into the lever.
  - The motion feedback now has `pose` (measured, in the trajectory frame). The prompt and the door skill chain from it.
- **Finger joint-limit violation / fling.** Clamped the finger width; closed-gripper moves are capped at `config.CLOSED_GRIPPER_MAX_SPEED` = 0.3 m/s.
- **Endless loop (doorhid_4).** `while phi < 108` never ended: the hand rested 2.6 mm short (not a stall), so ~1900 identical moves ran in 40 min.
  - `execute_trajectory` raises after 15 consecutive calls with the same target and no hand motion (`API.REPEAT_NO_PROGRESS_CALLS`).
  - The prompt asks for tolerant loop exits.
- **Planner decision lost at the iteration cap (doorhid_5).** The reply produced by the last allowed planner turn was never executed.
  - `run_plan` now executes it for its terminal decision only; `execute_subtasks` is refused there (`PlannerAPI.dispatch_closed`).

Reviewer FP / FN (terra), and evidence changes:

| # | Case | Type | Fix |
|---|---|---|---|
| A | pull-proprio said "door did not start" when slipped fingers still pushed it | FN | neutral wording (`slip_may_push`) |
| B | process conditions ("lever actuated") demanded as proof | FN | review rules: a reached goal implies them |
| C | the arm hovering in the end snapshot read as an occluder by the displacement check | FN | robot mask (robot's own pixels excluded) |
| D | planner amounts ("30°") read as exact | FN | within ~25 % counts |
| E/F | invented extras (rest place of the pole); door + pole one detection blob | FN/FP | rules + `same_region_as` caution |
| G | the pull-proprio gate skipped "Open the tall upright wooden door…" (regex allowed 2 adjectives) | missing evidence | `OPEN_INTENT_RE` allows 4 words |
| H | 8 labels of one blob → 8 near-identical displacement lines (prompt bloat) | noise | merged into one line per region |
| I | doorhid_4 pole: cleared, but the parked arm overlapping the handle → "no clear view" | FN | review rule: ignore the parked arm |
| J | slide_r3 attempt 2: door slid 11 cm, rejected. The planner called the SLIDING door hinged ("~45°"), and the reviewer judged against that | borderline FN (planner text) | planner rule 7: when unsure whether it hinges or slides, state a motion-neutral success ("visibly open, interior exposed") when the hinge/slide type is unknown |
| K | slide_r3 attempt 1: handle `left_place 0.99` while nothing moved (ncc 0.64; the arm hovered at the handle) | displacement FP (reviewer correctly ignored it) | open: the handle template is small and the arm covers it; trust the door-panel line |

Correct rejects (true negatives) seen: door_a2 attempt 1; door_a4 attempts 1-2; doorhid_3 door attempt 1 (4.6°); slide attempt 1.
Disocclusion evidence ("the surface behind got N cm farther = the object in front left") worked on the pole.

## 2026-11-04: execution limits (`helpers/run_budget.py`, values in `config.py`)
| Level | Limit |
|---|---|
| planner | 4 LLM iterations (`--max-planner-iter`); subtask `max_attempts` from the planner capped at 4 (`max_subtask_attempts`) |
| attempt (LLM turns) | 25 turns, 10 without motion → ended for review |
| code block (no LLM) | 15 identical no-progress moves; 300 robot actions; churn: 40 actions inside a 1 cm box with no rotation |
| attempt (time) | 2 × p90 of past normally-ended attempts of the same `sim:task` (clamped 4–45 min; 20 min until 5 recorded) → ended for review. History: `outputs/run_budget_history.json` |
| command (time) | 90 min → planner stops dispatching |

Block / time limits raise `BudgetExceeded` (a BaseException, so LLM `except Exception` cannot swallow it; it stays tripped for the rest of the block). Smoke: `door_ALLterra_budget_smoke_6` SUCCESS, attempt 158 s.

## 2026-11-04: why door:hidden needs many retries after the pole is removed (analysis, no code change)
Truth (eval only): hinge (-0.399, -0.361); lever bar from hub (-0.212, 0.085) (t=0) to free end (-0.343, -0.066) (t=1). t = grasp position along the bar.

| | plain `door` | `door:hidden`, open-door subtask |
|---|---|---|
| start pose | reset (arm high, out of view) | wherever pole removal ended: parked in front of / right of the lever, at handle height |
| affordance point #1 | (-0.291, 0.013), t=0.53 mid-lever, identical every run (same first image → LLM cache) | often the hub, e.g. (-0.157, 0.095), t=0.00, 6–7 cm off the bar (= plain point #4, ranked first) |
| `detect_object("door handle")` box | 0.24 × 0.12 m, t=0.36, long axis 0.85 rad (correct) | 0.12 × 0.13 m (near-square), t=0.03, axis -0.60 rad (83° wrong): the arm hides the free half → hub only. doorhid_7: the arm itself detected as the handle |
| hinge prior in planner prompt | "edge farther from the handle" (correct) | "door's LEFT edge (low confidence)" (wrong side) → pull arc / `tip_dir` reversed |
| extra subtasks | none | "move arm clear" prep subtasks, which stall or are rejected |
| grasps | all t 0.18–0.64; failures came from execution or review | every grasp at t ≤ 0.05 (hub) slipped; successes at t 0.10–0.81 |

The reviewer was not the problem: failed attempts opened 0.4–1.6°, successes 18.6–73°.

**Caption leak (all tasks).** `robot.desc` (the trajectory caption) is burned into saved camera images (`robot.py` `draw_text_overlay_image`). One-point moves and touch primitives leave it set, so perception and review images carry text such as "retreat up and away from the door". **Fixed 2026-11-04:** captions are drawn only on trajectory (video) frames.

**Recommended fixes (not implemented):**
- Park the arm high/back out of view after removing an occluder, then re-perceive.
- Hinge prior: the lever's free end points toward the hinge (not an image-side guess).
- Grasp at the middle of the affordance-point spread; reject points at the hub.
- A near-square handle box means the lever is occluded → re-view.

## 2026-11-04: hinge hint, occluder perception, arm retreat (implemented)
- **Hinge hint origin:** the perception VLM guessed "hinge near left edge (low confidence)" from frame appearance; the planner copied it. Fix: perception item 4 + planner rule 7 say only "hinge on the edge FARTHEST from the handle"; never copy a left/right guess.
- **Occluder rule** (perception item 2): a shape covering the target, crossing its frame, reaching the floor in front, or shaded like a curved surface is a SEPARATE occluder, not part of the target. Before it, terra merged the pole into the door ("tall gray door").
- **Arm retreat:** clearing already existed (planner rule 8). The new "retreat up and toward the base" wording was harmful: z=1.1 m is out of reach, and moving near the base axis folded the arm low (h3: every later lift stalled). Now: a clearing subtask ends by moving the hand back to the pose the subtask started from (known reachable, clear of the target).
- **Affordance images:** `affordance_points_<obj>_<step>.png` saved per step; the rollout package copies them and the head scene images into `perception/`.

| run | result | package (`outputs/2026-11-04/`) |
|---|---|---|
| door_p1 (plain door) | SUCCESS, attempt 1 | `door_terra_hingefix_regress_1` |
| doorhid_h1 | stopped: pole merged into door, removal skipped | `door_hidden_terra_hingefix_1` |
| doorhid_h2 | success, but the door swing knocked the pole over | `door_hidden_terra_hingefix_2` |
| doorhid_h3 | success after 3 planner iterations (folded arm from "toward base" retreat) | `door_hidden_terra_occluderfix_3` |
| doorhid_h4 | **clean SUCCESS**: pole removed attempt 1, hand back near home, door opened attempt 1, ~10 min | `door_hidden_terra_retreatfix_4` |

## 2026-11-05: grasp-slip guard (wrong-hinge arc caught in-flight)
- **Why earlier slips weren't caught:** tracking is opt-in (`--tracking`) and was off in h1–h4. The LLM never called `track_objects` or `pull_with_touch`. `pull_proprio` runs only after the attempt, and the arc was split into many 1-point trajectories.
- **Fix:** `touch/slip_guard.py`, always on, sampled per waypoint in `env.py`. It uses width, force (Genesis) and tracking; details are in document.md §7. A trip stops the trajectory and fails the attempt with feedback ("it resisted this way: wrong hinge side / use the other arc / pull_with_touch").
- **Harness** (`--arc-wrong` flips the arc):

| sim | correct arc | wrong arc |
|---|---|---|
| PyBullet | door 45.7°, no trip | width trip after 0.6 cm |
| Genesis | door 50°, end-stop jam at 44 cm (not a failure) | jam trip (396 N) after 1.1 cm |

- **Remote CoTracker** (`gpu_server`, `--tracker-provider remote`): ping OK. Synthetic shift test: ≤0.3 px error, ~20–30 ms RTT per frame, 135–225 ms per 8-frame flush. That is fast enough for the tracking check at waypoint rate.

## How to debug
- `main.py` logs are UTF-16. Grep with `Select-String -Encoding unicode`. Markers:
  - `[pull_proprio] outcome=`
  - `Review: success=`
  - `Planner: overall command completed`
  - `[world-state] ... door_handle_pos` (sim truth, for eval only)
- The Genesis subprocess log is `<images-root>/env_genesis.log` (IK messages: `IK: continuity solve off by`).
- Run `main.py` with the default `python` (vlm_traj). Genesis spawns its own `vlm_genesis` env.
- Harness:
  ```
  python research/touch/door_pull_e2e.py --sim pybullet --press --arc 50 --arc-yaw [--hinge-err 0.05]
  ```
  For Genesis, use the vlm_genesis python and `--sim genesis`.
- Geometry (sim truth, for eval only):
  - lever mid (-0.277, 0.009, 0.672); hinge about (-0.399, -0.361), radius ~0.39;
  - door normal yaw about -41° to -47°; robot base (-0.3, 0.5, 0).
- Tests:
  ```
  python -m pytest tests/test_tracking_door_pybullet.py tests/test_touch_pull.py tests/test_tracking_review_evidence.py -q
  ```
  39 pass. `test_optional_shoulder_camera_joins_through_the_pose` is a known flake.

## 2026-11-05: guarded pull, jam rules, door:hidden rollouts (code `958b5f5`, tracking off)
- **Opt-in guard** (`guard_motion`, `guarded_pull`, §7 of `document.md`). It needs a settle of ~0.5 s before re-acquiring, because tracker batches lag 135–225 ms.
- **Genesis harness** (`door_pull_e2e.py --sim genesis --guarded`):
  - A false jam at 6 cm came from a decaying press preload (1075 → 282 → 58 N) while the hand moved freely.
  - Fix: a jam now needs a stall (< 1.5 cm over the streak) and a lag behind the command (≥ 1 cm).
  - Press 0.03 + wrong arc: jam trip at 1.1 cm (396 N).
  - Correct arc: door 55° open.
  - Without a press, both arcs open ~25° and end blocked at 17–18 cm (likely reach / IK; open).
- **Rollouts** (`outputs/2026-11-05/door_hidden_ALLterra_slipguard_{1,2,3}`): 3/3 SUCCESS.

| run | result | root cause / note |
|---|---|---|
| 1 | door 88° | Pole attempt 1 pushed it over; it was never grasped. Door subtask 2: width slip at 2.6 cm, door 9°, reviewer accepted. Subtask 3 opened it to 88°. |
| 2 | door 15° | Pole toppled. Width trip at 8.5 mm finger drop. |
| 3 | door 9° | Pole fell during descent. Reviewer FN at 11.7° ("swung back closed"); attempt 2 accepted at 9.4°. Many width trips at 8–9 mm drop with fingers at 21–27 mm. |

Open:
- The pole is toppled, never grasped.
- Door openings are mostly small (9–15°).
- Width trips at an 8–9 mm drop on the round lever may be FPs (unproven; threshold unchanged).

## 2026-11-08: success = door >= 60°, back-face push in `guarded_pull`, SystemExit crash
- **Goal change**: the door must open ≥ 60° (a future mobile robot passes through); clearing the pole any
  way (toppled is fine) counts as success.
- **Why the pull stops at 18-36°**: the panel swings into the forearm (`panda_link5`); it pries the jaws open
  (width guard trip, fingers forced 30 → 54 mm) or stalls. No hinge spring (velocity motor, force 2); the
  forearm closes it a further 0-14° on release.
- **Rollouts**: r1 (opus) SUCCESS 64° in 38 min with an LLM-written back-face push; r2 (opus) reached 45°
  then the LLM code `raise SystemExit` killed the process (fixed: `execute_python_blocks` catches it, keeps
  the output, tells the LLM to use task_completed()/task_failed()).
- **Deterministic push** (`push_open(route="around")`, now default via `push_after_deg=15`), found with the
  harness (`DOOR_DIAG=1 door_pull_e2e.py --guarded right --gp-kw ...`), failures in order:

| harness runs | change | door (final) | failure |
|---|---|---|---|
| around4 | r_out = r_grip+0.12 | 0° | arcing round at the free-edge radius swept a 37° door shut |
| around5 | r_out +0.18 | 55° | wrist met the back face coming in; aborted as "blocked" |
| around6-10 | push from where it stopped; retreat variants | 28-57° (peak 66°) | retreat dragged the door back (~2 kN on the high-friction latch) |
| around11-13 | push at r_grip-0.10 | 89, 16, 90° | one-shot in-move: the IK path flung the door open or shut |
| around14-21 | r_out +0.30, stepped in-move, below 0.15 | 0-21° | arm column blocked by the latch spindle (sticks 10 cm out of the back face) |
| around25-27 | push from the spindle stop, drop-then-slide retreat | 54, 46, 70° | hand rode up over the spindle (below 0.05) |
| around28-30 | below 0.15 | (pull blocked), 61, 60° (peaks 79, 77) | unloaded drop still lost ~17° |
| around31-33 | unload 2 cm along the arc first; stall < 2° | (pull 21° -> push_blocked), 65, 74° | - |

- Open: short pulls (16-21°, ~1 in 3) leave no wedge -> `push_blocked`, the LLM must re-grasp and pull again.


## 2026-11-08 (later): hinge side from depth (`touch/door_plane.py`), door angle check
- **r3 (opus, old code) FAIL**: pole cleared, but the LLM's door model was wrong: pull_dir ~35° off the face
  normal, hinge on the WRONG side (perception's "door" box was 5 cm wide, the "hinge" box was the handle).
  Pull stalled at 25°, the push landed on the panel, later grasps closed on air; stopped after ~60 min.
- **guarded_pull can't fix a wrong hinge side**: on PyBullet a wrong-side arc rarely trips early (the hand
  slides 3-4 cm along the lever), and hinge re-fits from the hand path drift (refit1-4: 29, 14° ...); the
  push needs the hinge within a few cm (10 cm off -> "pushed 83°" while the door was at 23°). Removed the
  re-fit; an LLM hint >20 cm behind / >5 cm in front of the held lever (along the pull) is now clamped onto
  the door face band; a failed re-grasp at the last-held point falls back to the start grasp.
- **New `api.estimate_door_plane(rose_xyz)`** (pixels + depth only): RANSAC vertical plane behind the
  handle -> `pull_dir`, panel extent grown until a density dip (the frame gap, even with a coplanar jamb),
  hinge = far panel end. Anchor at the rose (lever-door joint): side ratio 2.4-7 vs ~1.4 at the lever
  middle. Needs the occluder cleared (returns None while the pole stands). Normal within ~1°, hinge ~9 cm
  inside the true axis (the panel edge seen at handle height).
- **New `measure_door_angle()`** + automatic check in `api.guarded_pull` after `estimate_door_plane`:
  `vision_deg`, and "opened" with depth < half the swept angle -> outcome `swung_shut`. Harness
  vis7/vis8: 61.6 / 60.7° measured vs 61.1 / 60.4° true.
- **Harness `--vision` (normal + hinge from depth only)**: vis3-8 -> 0.9 (swung shut), 67.4, 2.6 (grip
  slipped at 26°, door slammed shut by link5), 68.0, 61.1, 60.4° => 4/6 >= 60°, both failures now
  detectable as `swung_shut`. Push still stalls at ~63-64° hand angle (marginal vs 60°).
- Root of the remaining failures: the forearm (`panda_link5`) presses the panel (800-1500 N) during the
  pull; when the grip slips, it pushes the door shut.

## 2026-11-08 (night): agent rollouts with the depth door model (opus, `--attempts 2 --max-planner-iter 4`)
| run | result | door (truth) | time | what happened |
|---|---|---|---|---|
| r4 | SUCCESS | 73.9° | 21 min (r1: 38) | plane + hinge from depth, pull + push; `swung_shut` caught live once (73° swept vs 6° depth), re-pulled |
| r5 | FAIL | - | stopped | LLM anchored `estimate_door_plane` at the lever TIP: side ratio 1.09 -> wrong hinge side; `measure_door_angle` used 8-16 pts (garbage) |
| r6 | FAIL | 71.7° | stopped | door opened, then the LLM re-ran `estimate_door_plane` on the OPEN door (overwrote the closed reference) -> angle None/0 -> reviewer failed it ("angle could not be measured") |
| r7 | (see below) | | | correct side; several grasps `closed_on_nothing`; re-pull reached 25.8° |

Fixes after r5/r6:
- **Hinge side from the rose bump**: points 1-6 cm proud of the face within 0.3 m of the anchor = the rose/
  lever; hinge = the panel end FARTHER from the rose. Any anchor along the handle now works (offline: 19/21
  anchors correct, 2 return None, 0 wrong).
- `measure_door_angle` requires >= 25 points in the best bin (else None).
- **Closed-face reference is kept**: a re-estimate while the door is > 10° open returns the original plane
  plus `door_open_deg`.
- **Reviewer evidence**: `_door_angle_review_evidence` adds "DOOR ANGLE (head depth camera): X°" to the
  review prompt, so the reviewer doesn't have to judge the angle from an image; `partly_open` outcome
  (< 55° by depth).
- Remaining: the grasp itself is LLM-placed (z / offset along the lever) and misses often (r7). Next: a
  deterministic grasp point from the rose + plane (mid-lever, ~0.10 m proud of the face, rose z).

## 2026-11-08 (morning): deterministic `open_door` (helpers/door_open.py)
- r8 (opus, SUCCESS 67.6°, 18 min): a re-estimate with the arm in view saw a narrower panel and moved the
  hinge 19 cm -> push_blocked. **Fix:** with a closed door, a re-estimate narrower than the earlier one
  (same normal) keeps the earlier plane (arm occlusion only shrinks the panel). Fired 3x live in q1.
- q1 (qwen3.8-27b, before the helpers, SUCCESS 66.3°, 42 min): ignored `lever_grasp`, ~8 grasps
  closed on nothing, wrong rotations.
- **`lever_grasp`** in `estimate_door_plane`: bar points 7-16 cm proud of the face on the hinge side
  of the rose; middle of the seen span, z = median - 1.5 cm; rotation across the bar. Offline 1.5 cm
  from the true lever middle; same point for anchors 8 cm off.
- **`open_door(handle)`** (one call; also `grasp_door_lever`, `push_door`): plane -> mid-lever grasp
  (+-2 cm retries, holding + width check, 3 cm press) -> `guarded_pull` -> release + lift -> depth angle;
  until >= 65° (60 + depth error): door >= 30° -> push from behind without a grasp (`push_door`), else
  re-grasp at the lever swung about the hinge. Re-grasping a swinging lever at ~21° missed 3x and
  knocked the door shut (od2), hence the push path.

| harness (PyBullet, `--vision --vision-grasp --open-door`) | anchor err | door (truth) |
|---|---|---|
| vg1 / vg3 (grasp only + guarded_pull) | 0 | 60.9 / 61.2 |
| vg2 | -8,+2,+1 cm | 1.6 (push_blocked) |
| od1 / od3 | 0 / 3 cm | 60.3 / 59.4 (one pull) |
| od2 (old loop) | -8 cm | 52.8 (re-grasp at 21° missed, door knocked shut) |
| od4 / od5 (`--start-deg` 40 / 50, push_door) | 0 | 87.7 / 90.0 |
| od6 / od7 (new loop) | -8 cm / 0 | 63.7 / 63.1 (one pull) |

## 2026-11-08 (late morning): why the door fell back, and the fixes
- q2 (qwen3.8-27b, calls `open_door`; SUCCESS truth 90°, 29 min, but messy): first pull jammed at
  45° ("held part forced the fingers 30 mm open"), push out of reach, and after release the door read
  29°, then 5.8° after a "push_stalled hand 67°". The LLM then hand-wrote more attempts; reviewer
  accepted at 102° by depth.
- **od4/od5 above were invalid**: `--start-deg` reset the panel into the hand and it was flung to
  ~88°. The harness now lifts the hand first; a 20° start settles at ~13°.
- **Root causes (DOOR_DIAG, eval-only truth per move):**
  1. Vision hinge ~9 cm short of the true axis: in this URDF the axis sits past a 6 cm frame gap, at
     the jamb. The pull arc is then too small and pries the lever out of the jaws at 25-45°.
  2. The released panel can fall back on the lift (od12/od13: 27 -> 9°, it leaned on the forearm).
     `push_open(around)` then came in at the old hand angle, i.e. IN FRONT of the panel, and pushed it
     shut ("push_stalled hand 71°" -> door 2°).
  3. A diagonal approach from high up to the hover point crossed the swung panel (od2, q2).
- **Fixes** (commit d497400):
  1. `estimate_door_plane`: hinge at the first dense bin after the hinge-side gap (`hinge_gap_m`);
     err 9.3 -> 6.1 cm (the rest: axis 4.5 cm behind the face). Real doors: gap of a few mm, no change.
  2. `push_open(around)`: after the release lift, `measure_door_angle` (head depth); < 19° ->
     `door_fell_back` (open_door re-grasps a near-closed door, which works); else aim 20° behind
     the measured panel, not the hand.
  3. `door_open._approach`: up over the panel top (`top_z`, depth), over to 15 cm in front of the
     swung face, down, in.

| harness (closed start, new code) | anchor err | door (truth) |
|---|---|---|
| od15 (diag, before fix 2) | 0 | 72.1 (pull 36° -> push to 76°) |
| od21 / od23 | 0 / -2,+2 cm | 73.7 / 73.1 (one pull + push) |
| od22 | +2,-3 cm | 73.1 (push_blocked -> door 0 -> re-grasp, pull + push) |

### Agent rollouts with the fixes (d497400), door:hidden
| run | LLM | door (truth) | time | path |
|---|---|---|---|---|
| r9 | opus | 76.3° | 6 min | pull -> 61.7°, measured 68° after release, push -> 70.4° |
| q3 | qwen3.8-27b | 77.6° | 5 min | pull -> 56°, measured 70°, push -> 74.9° |
| q4 | qwen3.8-27b | 76.0° | 10 min | pull -> `door_fell_back` 0.6° (push skipped), re-grasp, pull -> 72.7° |

Before the fixes: r4-r8 16-21 min (2 of 5 FAIL), q1 42 min, q2 29 min. All three now open in one
`open_door` call with 2 moves; the small model only has to detect the handle and call it.


## 2026-11-09: hinge by touch + depth (no frame-gap search, no VLM)
- `estimate_door_plane` gives only a hinge HYPOTHESIS (the visible panel end, `hinge_source="panel_end"`);
  the old 10 cm jamb search (sized to this door's 6 cm frame gap) is gone.
- `open_door` -> `probe_hinge`: after the lever grasp, a short guarded straight pull (18 cm request, stops
  at the width/hold guard, usually 10-16 cm, door 9-20 deg), then `find_hinge_from_depth` fits the swung
  panel in head depth and intersects it with the closed face (`hinge_from_swung_panel`, also detects a
  wrong-side hypothesis: `flipped`). `guarded_pull(start_deg=...)` continues from there; a guard trip after
  a measured hinge is a slip (re-grasp), not a wrong-side flip.
- Hinge error ~ u * dtheta / theta: 6.6 cm at 8 deg, ~4 cm at 13-20 deg (a re-measure at 25+ deg would cut it).
- Scripted harness (`research/touch/door_pull_e2e.py --sim pybullet --vision --vision-grasp --approach top
  --guarded right --open-door`, run ONE at a time - parallel runs share the head image/depth files):
  - original door: 89.9 deg OPEN.
  - `DOOR_POSE=0.05,-0.04,10` (shifted, turned 10 deg): 89.6 deg OPEN (probe hinge at 9 deg).
  - `DOOR_MIRROR=1` (hinge on the other side): hinge measured [0.184, 0.376] vs true [0.202, 0.335] at 20 deg;
    first run flipped it wrongly on a guard trip (fixed); see the latest result below.
- Fixes during the mirrored runs: the panel line is refined from points > 8 cm from the hinge (frame/jamb
  faces flattened it: 17.9 vs true 21.5 deg -> now 22.4 vs 24.4, mirrored 20.1 vs 20.9, 34.4 vs 35.2);
  `probe_hinge` continues along the arc about the first fit and re-fits when the first fit is < 15 deg;
  `push_open(around)` re-fits the hinge from depth when the released door is >= 25 deg; `open_door` re-pulls
  a partly open door with `start_deg` = the measured angle. The depth hinge is where the panel's front face
  meets the closed face: a constant 4-6 cm from the joint axis on this door (the motion-derived arc is what
  matters for the pull, and it works).
- Results after the fixes: original door 82.3 deg OPEN (hinge_orig6); DOOR_POSE 89.6 deg OPEN.
- OPEN ISSUE, mirrored door (hinge_mirror3..7): the hinge and side are found correctly every time, the pull
  reaches 35-51 deg, but after the release the push-around route (out past the free edge, round, behind)
  sweeps the panel shut: on this placement the free edge swings toward the robot base and the arm cannot
  reach behind the panel (descent lands on the panel / blocked 28-30 cm short). Needs a different push
  route (e.g. push from the inner face near the free edge, or regrasp-and-pull in steps); not a hinge issue.
