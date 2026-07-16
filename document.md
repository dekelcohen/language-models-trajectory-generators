# VLM Robotics Trajectory Generators — Living Design Document

> **Living document.** Update this file in the SAME change whenever you add or edit a
> feature (new CLI arg, prompt, tool, planner/subtask behavior, perception/trajectory
> change). Keep the high-level flow readable; put deep code-level detail and historical
> notes inside the foldable `<details>` sections. Add every notable change to the
> [Changelog](#changelog).

---

## 1. What this is

An LLM/VLM-driven robot-arm agent. A user gives a natural-language command (e.g.
"open the door"). A **planner** LLM observes a head-camera image, decomposes the command
into ordered **subtasks**, and dispatches them. Each subtask is executed by a **low-level
code-gen agent** that writes Python (`detect_object`, `generate_linear_trajectory`,
`execute_trajectory`, gripper ops…), runs it against a simulator (PyBullet or MetaWorld),
then a **VLM reviewer** judges success from trajectory frames and drives retries.

```
user command
   │
   ▼
run_plan ──(--no-plan)──► execute_task (single subtask)
   │
   ▼  (planner enabled)
PLANNER LLM  ── observes head image + command (+ SCENE ANALYSIS from perception VLM)
   │  emits ```python: execute_subtasks([{prompt, max_attempts}, ...])
   ▼
execute_subtasks  ── runs ONLY the NEXT subtask, then returns
   │        the subtask ▼
   │   execute_task ── low-level code-gen agent (attempts loop)
   │        │  detect_object → 2D seg → 3D perception
   │        │  generate_linear_trajectory → execute_trajectory → gripper
   │        │  task_completed() → VLM review → pass/fail
   │        ▼  TaskResult(success, attempts, reviewer_reason, improvement_steps, …)
   ▼   printed subtask summary flows back to PLANNER
run_scene_perception ── perception VLM RE-RUNS on the updated scene after every subtask
   ▼   UPDATED SCENE ANALYSIS prepended to next planner turn
PLANNER reevaluates (insert prep/recovery subtask, e.g. move arm clear), dispatches
   next subtask, then plan_completed() or plan_failed()
```

Design split: **`API`** holds long-lived handles (sim connection, models, camera
calibration, continuous `trajectory_step`); **`TaskState`** (`api.task`) holds everything
fresh-per-subtask (conversation, attempts, review outcome). See
[§8 State model](#8-state-model-api-vs-taskstate).

---

## 2. Command line usage

Entry point: `python main.py [options]`. Interactive loop prompts `Enter a command:` and
calls `run_plan(ctx, command)` per line until an empty line / Ctrl+C.

### Common
```bash
# Default agentic run (planner on, PyBullet, moondream seg, VLM review)
python main.py

# Bypass planner: run the raw command as ONE subtask
python main.py --no-plan

# Pick model + limits
python main.py -lm azure-gpt-5 --max-tokens 60000 --reasoning-effort high
python main.py -lm or-google/gemini-2.5-flash --no-lm-images

# MetaWorld backend on a specific task
python main.py -s metaworld --task sawyer_door_v3 --transport ws
```

### Argument reference
| Arg | Default | Purpose |
|-----|---------|---------|
| `-lm, --language_model` | `azure-gpt-5` | Main LLM (planner + subtask + default reviewer). e.g. `gpt-4o`, `or-google/gemini-2.5-flash`. |
| `--lm-images / --no-lm-images` | `True` | Attach images to LLM prompts. |
| `--max-tokens` | `60000` | Max completion tokens. |
| `--reasoning-effort` | `None` | `xhigh…none` for reasoning models. |
| `--llm-cache / --no-llm-cache` | `True` | Disk cache of LLM responses (`config.llm_cache_dir`). |
| `-r, --robot` | `sawyer` | `sawyer` \| `franka`. |
| `-m, --mode` | `default` | `default` \| `debug`. |
| `-s, --sim` | `pybullet` | `pybullet` \| `metaworld`. |
| `--transport` | `auto` | `auto`(pipe for pybullet, ws for metaworld) \| `pipe` \| `ws`. |
| `--task` | `sawyer_door_v3` | Env name (MetaWorld only). |
| `--seg-provider` | `moondream` | 2D segmentation: `langsam` \| `sam3` \| `moondream`. |
| `--depth-format` | `norm_1m` | Depth reconstruction: `norm_1m` \| `norm_zfar` \| `raw`. |
| `--timeout` | `15.0` | Timeout secs; `<=0` disables. |
| `--delete-images` | off | Wipe image folders before recreating. |
| `--review-provider` | `vlm` | Success check: `vlm`, `vlm:<model>` (e.g. `vlm:or-openai/gpt-5.5`), or `xmem`. |
| `--perception-vlm` | `or-google/gemini-3.5-flash` | VLM run on the head image before every planner call; its scene analysis is injected into the planner prompt. |
| `--attempts` | `2` | Global default per-task attempts (1 first + retries with review between). |
| `--no-plan` | off | Skip planner; run command as a single `execute_task`. |
| `--prepend-prompt PATH` | None | Text prepended to the first `MAIN_PROMPT` only. |

<details><summary>Diagnostics / visualization / override args</summary>

| Arg | Purpose |
|-----|---------|
| `--ovr-bbox "x1,y1,x2,y2"` | Force a segmentation bbox (pixels). |
| `--ovr-obj REGEX` | Apply `--ovr-bbox` only to predictions whose label matches regex (else all). |
| `--viz-point / --vis-point JSON` | Add permanent 3D world marker(s): `"[x,y,z]"` or `"[[..],[..]]"`. |
| `--vis-traj` | Draw trajectory preview points (sphere markers) in sim. |
| `--vis-grasp` | Draw grasp-pose candidates (axes + fingers) in sim. |
| `--vis-box REGEX` | Draw 3D bbox (cylinders) for objects whose label matches regex. |
| `--save-grasp-inputs` | Save `masks[0]` `.npy` + projection/view matrices per `detect_object`. |

</details>

<details><summary>Replay & learning (LLM-less / offline) args</summary>

| Arg | Purpose |
|-----|---------|
| `--replay-log PATH` | Parse a conversation log, extract ```python blocks, execute them (no LLM). `execute_blocks_from_log`. |
| `--replay-vlm-review` | During replay, also run VLM-review blocks between attempts (default: skip). |
| `--learn-from-trajs PATH` | LLM generates an improved in-context example from past trajectories, then exits (`learn_from_past_trajs`). |

Replay/learn paths run **before** the interactive loop and return early. In replay mode
`task_completed()` ignores `max_attempts` so all blocks execute; review is skipped unless
`--replay-vlm-review`.

</details>

---

## 3. Modules map

| File | Responsibility |
|------|----------------|
| `main.py` | Arg parsing, logging, wiring: `init_agent → (replay/learn) → run_plan loop → teardown_agent`. |
| `agent_runner.py` | Core orchestration: `init_agent`/`teardown_agent`, `execute_task` (subtask agent), `run_plan` + planner loop, prompt builders, `TaskResult`, sim/handshake/context helpers. |
| `planner_api.py` | `PlannerAPI` — planner-level tools (`execute_subtasks`, `plan_completed`, `plan_failed`, `detect_object`) injected into the planner's exec env. |
| `prompts/planner_prompt.py` | Merged `PLANNER_PROMPT` + static `RECOVERY_FROM_FAILURE`. |
| `prompts/scene_perception_prompt.py` | `SCENE_PERCEPTION_PROMPT` run by the perception VLM before each planner call. |
| `prompts/main_prompt.py` | Subtask `MAIN_PROMPT` + shared vars (`COLLISION_AVOIDANCE`, `INITIAL_PLANNING_1/2`, detect-object tool variants), `IN_CONTEXT_EXAMPLE`. |
| `api.py` | `API`: `detect_object`, `get_grasp_poses`, trajectory gen/exec, gripper, `task_completed`/`task_failed`, `run_vlm_review`. |
| `task_state.py` | `TaskState`: per-subtask mutable state contract. |
| `env.py` | Simulator process: message loop (`CAPTURE_IMAGES`, `EXECUTE_TRAJECTORY`, grippers, `GET_STATE`…), sim envs, camera capture, marker drawing. |
| `helpers/main_utils.py` | `get_exec_locals` (subtask tool injection), `execute_blocks_from_log`, `learn_from_past_trajs`. |
| `segmentation_adapter.py` | Provider-agnostic 2D segmentation dispatch. |
| `utils.py` | Point-cloud → bounding cube, 3D↔2D projection, intrinsics/extrinsics. |

---

## 4. Planner

The planner is a **single continuous agentic conversation** (Arch 1). It does **not**
generate motion code — only decomposition + dispatch.

- **Scene perception (pre-step)**: before **every** planner LLM call, `run_scene_perception`
  runs the `--perception-vlm` model (default `or-google/gemini-3.5-flash`) on the current
  head image with `SCENE_PERCEPTION_PROMPT` (`[INSERT USER COMMAND TASK]` ← command). Its
  free-text answer (objects, target-affordance visibility/occluders, collision risks) is
  injected into the planner prompt's `[INSERT SCENE ANALYSIS]` section on the first call,
  and prepended as "UPDATED SCENE ANALYSIS" on subsequent iterations (subtasks may have
  changed the scene). `DECOMPOSITION RULES` reference this section instead of raw pixels.
  Best-effort: perception failure yields a fallback string and the planner continues.
- `run_plan(ctx, command, max_iterations=8)`:
  - `--no-plan` → `execute_task(ctx, command, max_attempts=args.attempts)` and return.
  - else: build `PLANNER_PROMPT` (with the initial `run_scene_perception` SCENE
    ANALYSIS), seed the LLM with the head image, then loop: execute the ```python
    blocks in the latest message → **re-run `run_scene_perception` on the updated
    scene** → prepend the UPDATED SCENE ANALYSIS + captured stdout/errors as the
    follow-up user prompt → call LLM again — until `plan_completed_flag` /
    `plan_failed_flag` or `max_iterations` (default 16).
  - Returns the `PlannerAPI` instance (`.plan_completed_flag`, `.plan_failed_flag`,
    `.subtask_results`).
- **Tools** (`planner_api.get_planner_exec_locals`): `execute_subtasks`, `plan_completed`,
  `plan_failed`, `detect_object`, plus `planner`, `math`, `np`, `logger`.
- **`execute_subtasks(subtasks)`**: runs **only the NEXT (first) subtask** in the list,
  then **stops and returns** so perception + the planner re-run on the updated world
  state. Returns:
  ```python
  {"executed": prompt | None,
   "success": bool,
   "result": <TaskResult.as_summary_dict()> | None,
   "remaining": [subtask, ...]}
  ```
  The concise printed form flows back as the planner's next user turn. A single dict is
  accepted and wrapped as a one-element list.
- **Per-subtask perceive→replan**: after every subtask the perception VLM re-runs and the
  planner is re-invoked with an UPDATED SCENE ANALYSIS. This lets it react to state
  changes between subtasks — e.g. after clearing an occluder, insert a "move the arm
  clear of the target" subtask before the next manipulation (fixes: arm self-occluding
  the door handle right after removing the gray cylinder). Cost: one perception + planner
  LLM call per subtask (intentional trade-off for correctness).
- **Subtask prompt contract**: each `prompt` must be self-contained and END with explicit
  REVIEW/VERIFICATION instructions (what the VLM reviewer must observe to accept success).

<details><summary>Prompt placeholders & recovery details</summary>

`_build_planner_prompt` fills: `[INSERT INITIAL PLANNING 1/2]`, `[INSERT COLLISION
AVOIDANCE]`, `[INSERT RECOVERY_FROM_FAILURE]`, `[INSERT SCENE ANALYSIS]`, `[INSERT 3D
COORDINATES PROMPT SECTION]` (from `ctx.coords_section`), `[INSERT EE POSITION]`,
`[INSERT TASK]`.

`RECOVERY_FROM_FAILURE` is **static guidance** (not a per-failure dump): after every
subtask (success or failure) it tells the planner to read the UPDATED SCENE ANALYSIS +
the last subtask's printed `result`, on failure read `result.reviewer_reason`/
`improvement_steps` and retry/insert-a-blocker-removal, on success watch for NEW
occlusions/blockers (incl. the arm itself) and insert a prep subtask, then dispatch the
next subtask or `plan_completed()`/`plan_failed()`.

`_run_planner_code_blocks` splits on ```` ```python ````, `exec`s each block with
`globals().copy()` updated by `planner_locals`, captures stdout via `redirect_stdout`.
On exception → `ERROR_CORRECTION_PROMPT`; on stdout → `PRINT_OUTPUT_PROMPT`; on no tool
call → a nudge to emit a proper block.

</details>

<details><summary>Planner in-context example (per-subtask replanning)</summary>

Command "Open the door" with a gray cylinder occluding the handle → step 1: clear the
cylinder (only this runs); step 2: UPDATED SCENE ANALYSIS shows the arm now occludes the
handle → insert "move the arm clear" subtask; step 3: open the door; final: `plan_completed()`.
Each `execute_subtasks([...])` call runs exactly one subtask. Full text lives in
`prompts/planner_prompt.py`.

</details>

---

## 5. Subtask (low-level code-gen agent)

`execute_task(ctx, prompt, max_attempts=None, in_context_example=True) -> TaskResult`.
The subtask agent writes and runs Python; each attempt ends in a VLM review that decides
retry vs. done.

- Fresh `api.task = TaskState(command=prompt, max_attempts, start_trajectory_step=api.trajectory_step)`.
- **First attempt** prompt = `MAIN_PROMPT` filled with the `detect_object` tool + optional
  in-context example + head image (`config.rgb_image_head_path`), sent as `role="system"`.
  Optional `--prepend-prompt` text is prepended to the first command only.
- **Loop** until `task.completed_task`: parse ```` ```python ```` blocks from the latest
  message, `exec` each with injected tools (`get_exec_locals`), capture stdout.
  - Exception → `ERROR_CORRECTION_PROMPT` (block number + traceback), `error=True`.
  - stdout (`< 2000` chars) → `PRINT_OUTPUT_PROMPT` fed back.
  - `task_completed()` triggers review (see §7); on review failure `failed_task=True`.
- **Retry path** (`failed_task`): generate a `TASK_SUMMARY_PROMPT` summary (appended to
  `attempt_summaries`), reset `start_attempt_trajectory_step`, rebuild the prompt with the
  **no-detect-object** variant + latest EE pose + combined `TASK_FAILURE_PROMPT` summary,
  reset `messages`, continue.
- Returns `TaskResult(success=review_succeeded, attempts, summaries, messages,
  reviewer_reason, improvement_steps, accepted_without_review)`.

### Available subtask tools (`get_exec_locals`)
`detect_object`, `get_grasp_poses`, `visualize_grasp_pose`, `execute_trajectory`,
`open_gripper`, `close_gripper`, `task_completed`, `generate_linear_trajectory`, plus
`api`, `math`, `np`, `logger`.

<details><summary>Why the retry drops detect_object + in-context example</summary>

On retry the arm usually occludes the target, so re-detecting gives worse coordinates than
reusing the first attempt's printed object coords. `_build_main_prompt` is called with
`NO_DETECT_OBJECT_TOOL` / `NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING` and an empty in-context
example; the prompt instructs the model to infer positions from conversation history.
`ENABLE_EEF_POS_IMAGE` optionally appends the live EE pose snippet + trajectory frames.

</details>

<details><summary>MAIN_PROMPT placeholders</summary>

`_build_main_prompt` fills: `[INSERT DETECT_OBJECT_TOOL]`,
`[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]`, `[INSERT COLLISION AVOIDANCE]`,
`[INSERT INITIAL PLANNING 1]`, `[INSERT INITIAL PLANNING 2]`, `[INSERT EE POSITION]`,
`[INSERT TASK]`, `[INSERT 3D COORDINATES PROMPT SECTION]`, `[INSERT IN CONTEXT EXAMPLE]`.
The shared planning vars live in `prompts/main_prompt.py` and are reused by the planner
prompt. `INITIAL_PLANNING_1` deliberately excludes the detect-object line (that line is
tool/attempt-specific and only belongs in the subtask prompt).

</details>

---

## 6. 2D segmentation + 3D perception — `detect_object` flow

`API.detect_object(segmentation_text)` — segment in 2D, transform to 3D world coords, add
bounding cubes to sim, and **print** each object's position/dimensions/orientation for the
LLM. Returns nothing (prints only).

High-level flow:
1. `CAPTURE_IMAGES` → env captures head + wrist RGB/depth; returns camera position +
   orientation (and, for MetaWorld WS, intrinsics `cam_info` = `{K_head, K_wrist}`).
2. Load head RGB + depth (prefer raw `.npy`; else 8-bit `/255`; `--depth-format` controls
   handling).
3. 2D segmentation via `get_segmentation_output` (`--seg-provider`), save an overlay image.
4. Masks (`config.segmentation_threshold`) → `utils.get_bounding_cube_from_point_cloud`
   → 3D world-frame bounding cubes + orientations.
5. `ADD_BOUNDING_CUBES` to sim; per object print `Position`, `Width/Length/Height`, and
   orientation (short vs long side); also project 3D pos back to 2D pixel for logging.
6. Record `task.segmentation_texts`, increment `task.segmentation_count`.

<details><summary>Code-level camera math (from detect_object docstring)</summary>

- `robot.get_camera_image("head")`: for the door head camera with spherical view,
  `_view_and_pos_from_spherical(target, distance, yaw, pitch)` →
  `p.computeViewMatrixFromYawPitchRoll(...)` and a camera position.
- `utils.get_bounding_cube_from_point_cloud(head_pos, head_orient_q, K_override=None)`:
  - contour pixel points of the segmented object;
  - `get_world_point_world_frame(...)` per pixel:
    - `K, Rt = get_intrinsics_extrinsics(image_height, cam_pos, cam_orient_q, K_override)`;
    - head camera pixel remap: `pixel_point = [-py, -px, pz]`;
    - `world_point_camera_frame = (inv(K) @ pixel_point) * depth`;
    - `world_point_world_frame = Rt @ [cam_frame; 1]`.
- `cube[4]` = object center; `Z` offset by `config.bounding_cube_depth_offset`.
- Payload parsing is robust to Pipe (PyBullet) vs WS (MetaWorld): handles list len 6/5/1.
- `--save-grasp-inputs` dumps `masks[0]` `.npy` + view/projection matrices.

</details>

---

## 7. Trajectory creation, execution, review & correction

### Creation & execution (`api.py` ↔ `env.py`)
- `generate_linear_trajectory(desc, start_pose, end_pose, num_points=20)` → `Trajectory`
  (linear interp of `[x,y,z,theta]`); validates lengths, logs the 2D projection of the end
  pose; **no** side effects.
- `execute_trajectory(trajectory)`:
  - optional `--vis-traj` preview (downsampled to ≤5 points) via `ADD_TRAJECTORY_POINTS`;
  - `EXECUTE_TRAJECTORY` → env moves the EE through each point (`robot.move`), records
    frames, returns `[msg, trajectory_step]`; `api.trajectory_step` updated (continuous
    across subtasks); `task.trajectory_length += len(points)`.
- `open_gripper()` / `close_gripper()` → `OPEN_GRIPPER` / `CLOSE_GRIPPER`.

### Reviewer & correction (`task_completed` → `run_vlm_review`)
- `task_completed()`:
  - render trajectory videos; `attempt_number += 1`;
  - **final attempt** (`attempt_number >= max_attempts`, non-replay) → accept without
    review: `completed_task=True`, `accepted_without_review=True`;
  - else `TASK_COMPLETED` then dispatch review by `--review-provider`.
- `run_vlm_review()` (default `vlm`): subsample head RGB frames (stride 5) from
  `start_attempt_trajectory_step` + wrist frames (stride 7), cap at
  `config.max_allowed_vlm_images`; ask the review model (`vlm` = main model, `vlm:<model>`
  = override) for strict JSON `{success, reasoning, improvement_steps}`.
  - success → `completed_task=True`, `review_succeeded=True`;
  - failure → `failed_task=True` (drives the retry path in `execute_task`).
  - captures `review_reason` / `review_improvement_steps` on `task`.
- `task_failed()`: sets `failed_task=True` without resetting env/counters (mimics
  real-world retries from current state).

### attempt_summaries
On each failed attempt, `execute_task` asks the LLM for a `TASK_SUMMARY_PROMPT` summary and
appends it to `attempt_summaries`; a combined summary is injected into the next attempt via
`TASK_FAILURE_PROMPT`. Summaries are returned on `TaskResult.summaries` and surfaced to the
planner in `execute_subtasks`' printed `result` report.

<details><summary>XMem review path (legacy)</summary>

With `--review-provider xmem`, `task_completed` runs `get_xmem_output`, reconstructs
per-object trajectory positions/orientations from masks across frames, and feeds them into
a `SUCCESS_DETECTION_PROMPT`; the model emits ```python calling `task_completed`/
`task_failed`. Preserved for compatibility; the default path is `vlm`.

</details>

---

## 8. State model: `API` vs `TaskState`

- **`API` (long-lived, per process)**: `main_connection`, models (`langsam`/`xmem`),
  `device`, camera calibration (`head_camera_position/orientation`, `cam_info`, image
  sizes), `llm_cache`, `coords_section`, `ee_pos_for_prompt`, and the **continuous**
  `trajectory_step` (image/video numbering + review frame sampling must not reset between
  subtasks).
- **`TaskState` (`api.task`, fresh per subtask)**: `command`, `max_attempts`,
  `conversation_messages`, `attempt_number`, `start_attempt_trajectory_step`,
  `completed_task`, `failed_task`, `review_succeeded`, `review_reason`,
  `review_improvement_steps`, `accepted_without_review`, `segmentation_texts`,
  `segmentation_count`, `trajectory_length`.
- `execute_task` assigns a fresh `api.task`; a local `task` alias makes
  `api.task_completed()` mutations visible. `teardown_agent` resets `api.task`.

---

## 9. Init / teardown

- `init_agent(args, logger) -> AgentContext`: OpenAI client (optional), seg/review models,
  device, simulator connection (PyBullet `Pipe` + `EnvProcess`, or MetaWorld WS), env
  handshake (`coords_section`, `ee_pos_for_prompt`, `sim_state`), `API`, `LLMCache`,
  `get_exec_locals`.
- `teardown_agent(ctx)`: close connection, terminate spawned processes (best-effort), reset
  `api.task`.
- `AgentContext` fields: `args, logger, client, api, main_connection, env_process,
  server_proc, llm_cache, exec_locals, coords_section, sim_state, ee_pos_for_prompt`.

---

## 10. End-to-end example

```
$ python main.py -lm azure-gpt-5
Enter a command: open the door

[planner] observes head image; gray cylinder occludes the door handle
[planner] execute_subtasks([
    {prompt: "Pick up the gray cylinder in front of the door and place it ~30cm aside... Success = handle unobstructed", max_attempts: 2},
])
  subtask → execute_task: detect_object("gray cylinder") → grasp → lift/place → task_completed() → VLM review PASS
execute_subtasks: executed=... success=True remaining_subtasks=0
[perception] re-runs → UPDATED SCENE ANALYSIS: robot arm now hovers over / occludes the door handle
[planner] inserts a prep subtask:
[planner] execute_subtasks([
    {prompt: "Move the arm up and clear of the door handle... Success = handle fully visible, unobstructed by the arm", max_attempts: 2},
])
  subtask → execute_task → task_completed() → VLM review PASS
[perception] re-runs → handle now visible and clear
[planner] execute_subtasks([
    {prompt: "Grasp the door lever and open it... Success = hinge angle clearly increased", max_attempts: 3},
])
  subtask → execute_task: detect_object("door lever") → rotate about hinge → task_completed() → VLM review PASS
[planner] plan_completed()
```

If a subtask fails, the printed `result.reviewer_reason` /`improvement_steps` flow back;
the planner retries it with a refined prompt or inserts a recovery subtask, then
re-dispatches — or calls `plan_failed()` if unreachable.

---

## Changelog

- **Per-subtask perceive→replan**: `execute_subtasks` now runs **only the next subtask**
  then returns; `run_plan` re-runs `run_scene_perception` and re-invokes the planner after
  **every** subtask (not just at batch end / on failure). The planner reevaluates the
  UPDATED SCENE ANALYSIS and can insert prep subtasks (e.g. "move the arm clear of the
  target") between subtasks. Return shape changed to
  `{executed, success, result, remaining}`; `max_iterations` default 8 → 16. Fixes: after
  removing an occluding gray cylinder, the open-door subtask started while the robot arm
  itself occluded the door handle. Prompt/example in `prompts/planner_prompt.py` updated.
- **Scene perception pre-step**: new `--perception-vlm` (default
  `or-google/gemini-3.5-flash`) + `prompts/scene_perception_prompt.py`. Runs on the head
  image before every planner LLM call; its text is injected into the planner prompt's
  `[INSERT SCENE ANALYSIS]` section; `DECOMPOSITION RULES` now reference that section
  instead of the raw image.
- **Merged planner + TaskState split**: single `PLANNER_PROMPT` (+ static
  `RECOVERY_FROM_FAILURE`); `execute_subtasks` (now one-subtask-per-call) replaces
  single `execute_subtask`; unified single agentic `run_plan` loop (removed the two-phase
  JSON parse/recovery-agent path); per-subtask state extracted from `API` into
  `TaskState`; `prompts/initial_plan.py` → `prompts/planner_prompt.py`.
- **Prompt var extraction**: `COLLISION_AVOIDANCE`, `INITIAL_PLANNING_1/2` shared between
  the subtask `MAIN_PROMPT` and the planner prompt.
- **Refactor of `main.py`**: monolithic `__main__` split into
  `init_agent`/`teardown_agent`/`execute_task`/`run_plan` in `agent_runner.py`; slim
  `main.py` entrypoint.
- **Enriched `TaskResult`**: `success`, `attempts`, `summaries`, `messages`,
  `reviewer_reason`, `improvement_steps`, `accepted_without_review` + `as_summary_dict()`.

> When you change behavior, add a dated/summary bullet here and update the relevant section.
