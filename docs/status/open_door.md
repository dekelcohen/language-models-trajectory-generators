# Status: open door, no touch (`door`, `door:hidden`)

Living status log so work can resume after an interruption. Branch `tactile`.
Last update: 2026-10-28, commit `a5a345b`.

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

## Results (door angle from sim state; for evaluation only)
| Run | Sim / task | Outcome |
|---|---|---|
| pb2 | PyBullet `door` | Success on attempt 1 (~32°). |
| pb7 | PyBullet `door` | Success; grasp at lever mid, handle moved ~37 cm. |
| gn6 | Genesis `door --no-touch` | Success; handle moved ~37 cm. |
| pb8, pb9 | PyBullet `door:hidden` | Pole cleared and the door opened 60-100°, yet the reviewer (gemini-3.8-flash) rejected every attempt. |
| gn1-gn5, pb3-pb6 | Before fixes 4-8 | Failed: `closed_on_nothing`, wrong hinge, IK fling. |

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
