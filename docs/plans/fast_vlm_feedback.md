# Plan: fast-VLM pointing and mid-subtask visual checks (not implemented)

Status: **plan only**. Builds on the contact guard (`touch/slip_guard.py`, `guard_motion`,
`guarded_pull`, see document.md §7). Sim2Real rules apply: pixels + proprioception only.

## Why
- The guard already catches slips/jams from width + force + tracking with no LLM call.
- After a trip, the agent needs **where the part is now**. Today the order is:
  1. tracking re-acquire after a ~0.5 s settle (`motion["guard"]["reacquired"]`, CoTracker ~135–225 ms per batch);
  2. proprioception: `last_held_pos` (where the hand last held it);
  3. full re-plan: VLM pointing + LLM trajectory (slow, ~30–90 s, costly).
- Gap between 2 and 3: a **fast pointer** (~0.2–0.5 s) that answers "where is the handle in this image" mid-subtask.
- Tracking fails when the part was occluded at seeding, left the view, or the track drifted on a slip.

## 1. Multiple GPU servers, one role each
- Today: one `GPU_SERVER_URL` (CoTracker box). Add role-based routing:
  - `GPU_SERVER_URL_TRACK` (CoTracker, existing small GPU), `GPU_SERVER_URL_POINT` (Molmo, stronger GPU);
    fallback to `GPU_SERVER_URL` when a role var is unset. Tokens: `GPU_SERVER_TOKEN_<ROLE>` (env only).
  - `gpu_server/client.py`: `client_for(role)` cache; modules stay as they are (`modules/<name>.py`).
- New module `gpu_server/modules/molmo_point.py`:
  - model: Molmo-4B (video/multi-frame variant when available), bf16, preloaded.
  - `point(images: [H,W,3] x N, query: str, max_points=3) -> [{"cam", "uv", "score"}]`; N = head + wrist (+ 1–2 recent frames).
  - target latency < 0.5 s for 2 images at 640x480; JPEG codec on the wire.
- Agent side: `api.fast_point(name, cameras=("head","wrist"))` -> world points via depth + calibrated extrinsics (same lift as `detect_object`), with a `visible` flag per camera.

## 2. Re-acquire ladder after a guard trip
1. Settle `settle_s` (done) -> tracking `reacquired` if `trusted`.
2. Else `fast_point` on head + wrist; accept when the point lifts to within ~5 cm of `last_held_pos`
   (or the fitted hinge arc) - proprioception gates the VLM.
3. Else occlusion search (below).
4. Else full re-plan (today's path).

## 3. Occlusion search heuristic (arm hides the handle)
- Start: `last_held_pos`, hinge fit, tracked points (even untrusted), touch contacts.
- Step A: one image pair -> `fast_point` + a yes/no "is the <part> visible / occluded by the robot?" query.
  The robot self-mask (already in perception) gives the occluding pixel region.
- Step B: if occluded, move the arm to clear the line of sight, choosing among a few candidates:
  - lift 10–15 cm along -approach (cheapest; keeps the hand near the part);
  - slide sideways away from the head-camera ray through `last_held_pos`;
  - park high / out of view (the door:hidden retreat) when the above fail.
  Score each candidate by the projected self-mask overlap with the part's expected pixel box (pure geometry, no VLM).
- Step C: alternatively aim the **wrist camera** at `last_held_pos` from a stand-off (look-at pose), then point.
- Stop after 3 views; fall back to full re-plan with the images gathered.

## 4. Fast-VLM as a mid-subtask checker
- Uses: spill visible? glass broken / tipped? handle still between the fingers before the pull? lid seated?
- When: a few fixed moments, not every frame:
  - after a grasp, before moving an articulated part (one check; two if they disagree with width/force);
  - at the end of a pour / carry;
  - on a guard `note` (e.g. "load fell, grip unchanged") to disambiguate unlatch vs slip.
- Expensive and can hallucinate, so:
  - never the sole reason to abort: it can **confirm** a proprio/tracking flag, or **flag** for the reviewer;
  - binary questions with a "can't see" option; two frames, majority vote;
  - logged with images into the rollout package so FP/FN can be audited.
- LLM opt-in like `guard_motion`: e.g. `visual_check("is the handle between the fingers?", when="after_grasp")`.

## Phases
| # | deliverable | acceptance |
|---|---|---|
| P1 | role routing + `molmo_point` module + `fast_point` | door handle point error < 3 cm on 20 saved frames; < 0.5 s |
| P2 | re-acquire ladder in `guarded_pull` | door/door:hidden: a late slip recovers without re-plan in ≥ 2/3 runs |
| P3 | occlusion search | door:hidden post-pole: handle visible within ≤ 2 arm moves |
| P4 | `visual_check` opt-in + prompt examples | no extra false aborts on kettle/pick/slide goldens |

## Risks
- Pointing from the head cam at a parked-arm-occluded handle: the main reason for P3.
- Hand-eye calibration error (~0.5–1 cm real): gate VLM points with proprioception, never the reverse.
- Two remote servers add network failure modes: every call has a timeout and falls through the ladder.
