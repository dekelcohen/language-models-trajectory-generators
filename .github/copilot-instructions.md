# Copilot instructions

VLM-driven robot manipulation (planner -> LLM trajectory code -> VLM reviewer) in PyBullet /
Genesis / MetaWorld sims. Full design notes: `document.md`. Living status logs: `docs/status/`.

## Rollout outputs

Package every rollout under `outputs/<run-date>/<descriptive-run-name>/` (e.g.
`outputs/2026-11-08/door_hidden_opus_guardedpull_1/`, via
`python -m helpers.rollout_report <images_root> --name <name> --out outputs/<run-date>`):
- `videos/`: the rollout videos.
- the logs (`vlm_traj.log`, and `env_genesis.log` when present).
- `summary.md`: the task, models, outcome per subtask, wall time, eval metrics and root causes.
- important images, one folder per subtask attempt (`s<subtask>_a<attempt>/`, e.g. `s1_a2/`) with the
  pointing, segmentation, head and wrist images of that attempt.

After long runs on a dev branch, update `docs/status/`, docs and plans, then commit and push.
Do not commit videos or other very large files.

### Upload to Hugging Face

Also upload every packaged rollout to the HF dataset
[RoboTrick/vlm_traj_rollouts](https://huggingface.co/datasets/RoboTrick/vlm_traj_rollouts) (private).
The token is `HF_TOKEN` in `.env`; the CLI is `hf`, from `huggingface_hub`.
- Upload to the same path as the local folder: `rollouts/<run-date>/<run-name>/`, e.g.
  `hf upload RoboTrick/vlm_traj_rollouts outputs/<run-date>/<run-name> rollouts/<run-date>/<run-name> --repo-type dataset`.
- Keep the HF "videofolder" layout, so the Dataset Viewer plays the videos if the dataset becomes
  public or the org gets a paid plan. Today it is private on a free org, so the built-in viewer is
  off; browse with the static Space `RoboTrick/vlm_traj_rollouts_viewer` (`research/hf_rollout_viewer/`).
  The layout:
  - Each run folder has a `metadata.jsonl` next to `videos/`, one line per video.
  - `file_name` holds the relative video path (`videos/<x>.mp4`).
  - Always use the same columns: `run_date, run_name, task, sim, llm, perception, reviewer, camera,
    outcome, success, wall_min, metric, notes` (strings / bool / float; use `""` when unknown).
  - The repo `README.md` dataset card has
    `configs: [{config_name: default, data_files: ["rollouts/**/videos/*.mp4", "rollouts/**/metadata.jsonl"]}]`,
    so images, logs and `summary.md` sit beside the videos without breaking the viewer.
  - Videos must be browser-playable H.264 MP4 (convert with ffmpeg if needed).

### Local disk retention

When writing a new rollout, keep only today's and yesterday's `outputs/<date>/` rollout folders
locally.
- Delete older date folders, but first verify each one is fully on HF (`hf` / `HfApi().list_repo_files`).
- Leave behind `outputs/<date>/HF_LINK.md` with the URL
  `https://huggingface.co/datasets/RoboTrick/vlm_traj_rollouts/tree/main/rollouts/<date>`.
- Committed summaries remain in git history. Commit the deletion together with `HF_LINK.md`.
- Never delete a folder whose upload you could not verify.

## Sim2Real

Goal: the sim pipeline must run unchanged on a real Franka Panda / FR3. The robot operates
from **pixels + proprioception only**.

**Allowed inputs to the planner / LLM / reviewer / retry prompts** (all available on the real robot):
- Camera images (RGB, depth) and calibrated camera intrinsics/extrinsics.
- Proprioception: EE pose (forward kinematics), joint positions, gripper width / holding,
  joint torques / estimated external force, the arm base mount pose.
- Anything derived from those (detections, tracking, point clouds), and the task text.

**Never leak privileged sim info** into any prompt or decision path: `simenv.get_state()` /
`sim_state` object poses, articulation joint values (door angle, hinge axis/arc), success
criteria / `task_error`, particle counts, URDF geometry, engine contact lists. These are only
for logs, evaluation and tests. Tests must not hand the agent such info either (e.g. no true
hinge as a starting assumption).

**Keep the physics honest**: no weld / attach-constraint grasps or other cheats; objects are
held by contact + friction only.

## Generalize, do not overfit the scene

Skills, prompts, helpers and control code must work for the same object category (door, drawer,
plug/socket, kettle...) in a different setup: moved, rotated, mirrored, resized, stiffer/looser,
different looks, lighting or camera view. Do not write code that only works in the current scene.
- **Never hard-code** absolute world coordinates, scene-specific offsets/gaps, or a force, size or
  angle tuned on one asset. Two examples: "the hinge is 10 cm past the panel", or `hinge=[0.42, -0.30]`
  in a prompt example. Measure such values from pixels / depth / proprioception, or derive them from
  measured geometry. Make them parameters that adapt across retries.
- **Category priors are fine** when they hold for most instances, and should be stated as priors. For
  example: "on a lever door the hinge is on the panel edge far from the handle". So is "a plug enters
  along the socket's normal". Prefer verifying them by touch / motion (e.g. pull a little and see how
  the panel swings) over trusting them blindly.
- Prompt examples and skills: use symbolic values (`p["hinge"]`), not numbers or names from the test
  scene. Keep rollout anecdotes in `docs/status/`, not as rules in `SKILL.md`.
- Validate on perturbed scenes (`docs/plans/perturb_scenes.md`), not only on the scene you tuned on.
- **New perception / sensing assumptions or dependencies must be flagged.** Examples: a VLM that
  locates the hinge, a new detector, a new sensor, or a model of the frame. Call them out explicitly
  in the summary to the user. Preferably ask first (plan mode) before adding them.

**Does the real robot know its gripper 3D position?** Yes. libfranka reports the EE pose
(`O_T_EE`) at 1 kHz from the joint encoders + forward kinematics, and the Franka Hand reports
the finger width.
- Repeatability (returning to a taught pose) is < ±0.1 mm (FR3 datasheet).
- Absolute accuracy is not specified by Franka. Uncalibrated it is typically a few mm (kinematic
  model errors, link/gripper deflection under payload).
- The dominant real error is usually the camera-to-robot (hand-eye) calibration, ~0.5–1 cm.
  That error applies to every object position computed from pixels.

So reporting the EE pose in prompts is realistic. Sim gives it exactly; do not rely on
sub-cm agreement between the measured EE and camera-derived object positions.
