# Status: slide-cabinet reviewer false positive + retry direction diversity

Owner: slide-cabinet agent. Branch `tracking`. Updated on every new conclusion/data point.

## Goal
1. Reproduce `franka_kitchen:slide_cabinet` with "open the right cabinet": robot pulls the
   handle toward itself, door never moves, reviewer says SUCCESS (false positive).
2. Fix the reviewer with measured, non-privileged evidence (proprioception / tracking), never
   sim joint values / task_error.
3. Retry must diversify the opening direction (pull failed with no motion -> try lateral).
4. Verify with real runs (reviewer rejects no-motion attempt; later attempt slides; sim success).
5. Unit tests + document.md note. Commit only own files.

## Decisions / Conclusions
- Sim truth (probe, pybullet DIRECT, `resetJointState(slide_cabinet, 0.3)`): the slide panel
  moves along world **-y** (handle `slidelink_1` AABB centre y 0.106 -> -0.194). Head camera
  image-right == -y, so the panel slides to the RIGHT in the image. Handle is a vertical bar
  (0.245 m tall) on the panel's left (+y) edge. Success threshold 0.3 (unscaled) => ~5.3 cm
  of slide (scaled) is enough.
- Ground truth during runs is read from the `[world-state]` log line (`target_link_pos` =
  handle AABB centre) - logging only, never shown to the LLM/reviewer.
- Root cause: (1) the perception VLM labels the panel "revolute, hinged on the right", so the
  planner/subtask pull straight out (-x) or along a hinge arc; the vertical bar slips out of
  the fingers and nothing moves. (2) the reviewer only gets a short video + end-of-attempt
  gripper state (gripper is already re-opened and retreated), so arm motion "looks like" an
  opening; no measured signal says the held part never travelled. (3) retries only see the
  reviewer's free-text, which blames grasp depth/alignment -> same pull repeated.
- Fix: per-trajectory proprioception (commanded vs reached EE, finger width, holding flag,
  last EE pose while still holding) recorded in env.py, summarized by
  `helpers/pull_proprio.py` -> reviewer evidence block + deterministic veto (articulated task
  and held pull carried < 3 cm and no later open-hand push) + retry direction hint that lists
  tried directions and recommends untried ones (sideways along the face first, then other
  side, lift, push-in). Tracking recommendation also switches to lateral after a straight,
  aligned pull that moved nothing. No privileged sim state used.
- Pre-existing, not mine: `tests/test_ipc_contract.py::test_get_state` fails on missing
  `pole_pos`/`pole_dims` (door-scene state contract).

## Data
- run1 (baseline, OLD code, no tracking): `python main.py --task franka_kitchen:slide_cabinet -r franka --sim pybullet
  -lm or-google/gemini-3.8-flash --review-provider vlm:or-google/gemini-3.8-flash
  --planner-perception-vlm or-google/gemini-3.8-flash --images-root ./images_slide_run1
  --attempts 4 --max-planner-iter 4 -c "open the right cabinet"`.
  Planner subtask 1: "pull it open, rotating the door outward about its right hinge".
  att1 rejected (closed on air), att2 rejected (closed on air), att3 accepted WITHOUT review
  (final attempt); sim handle y 0.106 -> 0.066 (4 cm, below the ~5.3 cm needed). Planner
  iter 2 again "pull open the door outward around its right-side hinge": att1, att2 rejected.
- run2 (NEW code, same command, `--images-root ./images_slide_run2`), planner iter 1 =
  "pull it open, rotating the door outward about its right hinge" (max 3 attempts):
  - att1 hinge-arc pull: `[pull_proprio] part_did_not_move` (held, slipped, 1.6 cm carried);
    reviewer False; retry hint "did NOT move when pulled along [-0.96,-0.25,-0.09] ... slide
    sideways along the face [+0.01,-1.00,0]" injected.
  - att2 (followed the hint) slid -y: `part_moved_with_hand` carried 15+14 cm. Reviewer said
    FALSE ("door did not slide, gripper slipped off") -> **false negative** (256-px clip; frames
    show the gap opening; DIRECT replay of the same waypoints gives joint 0.33 = success). The
    old reviewer improvement steps then pushed "replace sliding with a hinge arc".
  - att3 (final, unreviewed) went back to the arc, re-grasped at the STALE handle coords
    (y=0.106) and dragged the panel closed again; measured veto overruled the (implicit)
    success: "held, never travelled". World-state end: handle y=0.106 (closed) - veto correct.
  - planner iter 2 (saw the measured veto + hint): "slide it sideways to the right" ->
    `part_moved_with_hand` 24 cm; reviewer True; world-state handle y=-0.11 (21.6 cm slide,
    task_error ~0.15 < 0.3 => **sim success**); plan_completed.
- NOTE: the `Env state:` log line in agent_runner is a stale start-of-run snapshot (always
  joint 0.0) - only `[world-state]` lines are live truth. Added a `[world-state] attempt N end
  (failed)` log in `handle_task_failure` (logging only).
- run1 (old code) iter 2 final unreviewed attempt left handle y=-0.02 (sim success by luck of
  an arc moving -y), yet the planner kept re-dispatching "swing the door about its hinge".
- Follow-up fixes after run2: (a) `part_moved_with_hand` evidence now says "it DID move; do not
  claim it stayed shut; a panel that slid open IS open even if the task text expected a hinge";
  (b) retry hint for moved-but-rejected: keep the direction, handle is now at grasp point +
  measured hand travel (old detected coords are STALE), open + back straight away before
  moving sideways; (c) reason text for the unreviewed final attempt.
- run1 final (old code): 4 planner iterations, every subtask "swing the door about its hinge";
  stopped "after 4 iterations without a terminal decision". Sim ended with handle y=-0.132 (open
  by accident of arc paths drifting -y) but the agent never knew it.
- run3 (NEW code incl. follow-ups, `--images-root ./images_slide_run3`, no tracking):
  - att1 "arc pull 1": `[pull_proprio] part_did_not_move` (slipped, 1.5 cm); reviewer False;
    truth `[world-state] attempt 1 end (failed)` handle y=0.096 (1 cm) -> correct rejection;
    hint "did NOT move when pulled along [-0.97,-0.23,-0.09] ... slide sideways" injected.
  - att2 "T3: Slide right cabinet door ...": `part_moved_with_hand` 33 cm; reviewer True;
    truth handle y=-0.195 (30 cm slide, task_error ~0.07 => **sim success**); plan_completed.
  - Found `handle_now=None`: `traj_start` read `trajectory[0]` but `Trajectory` is not
    indexable -> fixed to `trajectory.points[0]`.
- run4 (NEW code + `--tracking`, template tracker; the LLM never called `track_objects` on the
  handle, so no tracking evidence was produced -> no with/without-tracking comparison possible):
  - iter1 att1 arc pull: `part_did_not_move`; reviewer False; truth y=0.082 (correct).
  - iter1 att2 slide with fingers fully closed: `closed_on_nothing`, yet truth y=-0.218 (closed
    fingers PUSHED the bar, panel open). Reviewer False = **false negative**, misled by the
    evidence text "never held anything, so nothing was pulled" -> FIXED: text now says nothing
    was held but closed fingers can still push the part; judge movement from the clips.
  - Later iterations started with the panel already near its open limit (y~-0.22): sideways
    pushes were `blocked` (panel at end stop) and read as `part_did_not_move`. Limitation: a
    part already at its limit looks identical to a part that does not open that way.
    Veto fired once (iter1 att3 final), truth handle y=0.109 (closed) -> veto correct.
  - Planner stopped after 4 iterations without a terminal decision; final truth y=-0.222 (open).
- Tests: `pytest tests/test_tracking_review_evidence.py tests/test_side_approach_e2e.py` 23 passed
  2 skipped (8 new TestPullProprio); `pytest tests/test_ipc_contract.py -k execute_trajectory`
  4 passed; `tests/test_exec_locals_persistence.py test_skill_registry.py
  test_trajectory_normalization.py` 59 passed; `tests/test_franka_kitchen_ipc.py` 10 passed.

## Code changed (committed on `tracking`)
- new `helpers/pull_proprio.py`; `env.py` (per-trajectory proprio record in
  EXECUTE_TRAJECTORY reply[2]); `robot.py` (`last_ee_target`); `api.py` (motion_log, evidence
  block, veto, hint); `task_state.py`; `agent_runner.py` (hint in retry prompt); `config.py`
  (`review_pull_proprio`); `prompts/task_failure_prompt.py`; `tracking/review_evidence.py`;
  tests.

## Next steps
- commit, push, cleanup (document.md section added: §7 'Measured pull proprioception' + changelog).
