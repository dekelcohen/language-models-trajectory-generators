# Status: CoTracker window-row backfill (tracker agent)

Last updated: 2026-10-25 01:55 (+02:00). Branch `tracking`.

## Goal

Investigate the uncommitted tracker changes (`providers/trackers/cotracker_tracker.py`,
`tracking/session.py`, `tracking/types.py`), check that they are complete, find out whether
they caused the `tests/test_motion_gates.py` failures, add tests and docs, then commit and push.

## Decisions / Conclusions

- **What the change does:** online CoTracker flushes only every `step` frames. A flush
  predicts every frame since the previous flush, but only the last one was used. Those rows
  are now returned on a fresh result as `meta["window_rows"]` (`points/visible/scores` oldest
  first, `frames_back`; the last row is the current frame). `TrackingSession` keeps the last
  `BACKFILL_VIEWS` (32) views per camera and converts each row to 3D with the depth of
  **its own** frame into `CamTrack.backfill` (`[{"frame_idx", "points_world"}]`, indexed by
  seed point, `nan` = unseen).
- This is a backfill *inside the CoTracker window*. It is **not** the "template tracker runs
  live, CoTracker re-runs over the recorded clip" design.
- Backfill is a record only: the live pose and monitors do not use it, and
  `CamTrack.to_dict` does not write it out.
- The diff was complete and coherent: no debug prints, no half-finished code, imports OK.
  No code fixes were needed.
- **The `test_motion_gates.py` failures were already there before this change:** the same 5
  failures happen on an unmodified copy of HEAD in a temporary git worktree. The test imports
  only `tracking.motion`, which this change does not touch. I left them alone because they are
  outside this task.
- Memory cost: 32 views per camera, about 140 MB at 640x480 with 2 cameras. Views are kept
  even for non-windowed providers. This is acceptable for now; see Next steps.

## Data (runs / tests)

Python: `C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_traj\python.exe`

- Before (uncommitted diffs):
  `python -m pytest tests\test_motion_gates.py tests\test_cotracker_tracker.py tests\test_tracking_unit.py tests\test_tracker3d.py tests\test_tracking_eval.py tests\test_tracking_review_evidence.py tests\test_klt_tracker.py tests\test_pose_tracker.py -q -p no:warnings`
  -> 5 failed, 205 passed, 2 skipped. All 5 failures are in `test_motion_gates.py`:
  `TestDoorHandlePullDirection` (x2), `TestDrawerPerpendicularity::test_dropped_points_are_carried_forward_into_the_plane_fit`,
  `TestDisagreementGate::test_a_disagreeing_frame_cannot_drag_the_axis`,
  `TestConfidenceCalibration::test_high_confidence_implies_low_angular_error`.
- HEAD baseline (temporary worktree `git worktree add --detach .wt_head HEAD`, since removed):
  `pytest tests\test_motion_gates.py` -> the same 5 failed, 44 passed.
- After (same set + `tests\test_realtime.py`): 5 failed (the same motion gates), 230 passed,
  2 skipped.
- New tests: `test_cotracker_tracker.py::TestWindowStaleness::test_flush_exposes_fresh_rows_for_the_frames_since_the_previous_flush`,
  `test_realtime.py::SessionBackfillTest::test_rows_are_lifted_with_the_view_of_their_frame`
  (both pass).
  - The session test can tell a wrong-frame view apart: with the correct view the 3D distance
    to the sphere centre is 0.0799 m (= the radius). With another frame's view it is off by
    11 mm or more. The tolerance is 3 mm.

## Code changed

Committed and pushed as `2f6d5e0` ("Tracking: CoTracker window rows backfill past frames with
their own depth"):
- `providers/trackers/cotracker_tracker.py`, `tracking/session.py`, `tracking/types.py`
- `tests/test_cotracker_tracker.py`, `tests/test_realtime.py` (+1 test each)
- `document.md`: the CoTracker paragraph in the tracking section, plus test counts
  (cotracker 26 free + 1 gated, realtime 24).

Not committed (and must not be): `outputs/**`, plus the other agents' files
(`sim_envs/pybullet/franka_kitchen/*`, etc.).

## Next steps

- Optional: use `CamTrack.backfill` downstream (e.g. motion estimation / review evidence)
  instead of treating the in-between frames as stale.
- Optional: keep `_recent_views` only when a windowed provider is active, to save memory.
- Separate task: investigate the existing `test_motion_gates.py` failures in `tracking/motion.py`.
