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
   │        │  (reuses the planner's SCENE ANALYSIS; no per-subtask perception re-run)
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
calls `run_plan(ctx, command)` per line until an empty line / Ctrl+C. Prompt supports
bash-like command history: arrow-up/down recalls previous commands (see `helpers/command_utils.py`).
History merges a repo-committed seed file `config/vlm_traj_user_commands.txt` with a
per-user file `~/vlm_traj_user_commands.txt` (duplicates removed); newly entered commands
are appended to the home file only. Uses stdlib `readline` on Unix; on Windows uses
`pyreadline3`'s line editor directly (built-in `input()` bypasses it there).

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
| `--planner-perception-vlm` | `or-google/gemini-3.5-flash` | VLM run on the head image before every planner call; its scene analysis is injected into the planner prompt. |
| `--affordance-points` / `--no-affordance-points` | on | Ask the perception VLM for ranked 2D grasp-affordance points on the target object, convert them to 3D world coords and inject them into the scene analysis. Disable to drop the pointing block from the perception prompt entirely. |
| `--attempts` | `2` | Global default per-task attempts (1 first + retries with review between). |
| `--no-plan` | off | Skip planner; run command as a single `execute_task`. |
| `--reset-eef` | off | Re-home the **arm only** (`RESET_EEF`) at the start of every subtask, before capturing the EE start pose. Does **not** reset object/world state and does **not** reset `trajectory_step` (prior subtask frames preserved). Default off = real-world behavior: the arm starts wherever the previous subtask left it. |
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
| `agent_runner.py` | Core orchestration: `init_agent`/`teardown_agent`, `execute_task` (subtask agent), `run_plan` + planner loop, prompt builders, `TaskResult`, sim/handshake/context helpers. (Scene perception lives in `helpers/perception_scene_analysis.py`.) |
| `planner_api.py` | `PlannerAPI` — planner-level tools (`execute_subtasks`, `plan_completed`, `plan_failed`, `detect_object`) injected into the planner's exec env. |
| `prompts/planner_prompt.py` | Merged `PLANNER_PROMPT` + static `RECOVERY_FROM_FAILURE`. |
| `prompts/scene_perception_prompt.py` | `SCENE_PERCEPTION_PROMPT` run by the perception VLM before each planner call. |
| `prompts/main_prompt.py` | Subtask `MAIN_PROMPT` (now with a `SCENE ANALYSIS` section) + shared vars (`COLLISION_AVOIDANCE`, `INITIAL_PLANNING_1/2`, detect-object tool variants), `IN_CONTEXT_EXAMPLE`. |
| `api.py` | `API`: `detect_object`, `get_grasp_poses`, trajectory gen/exec, gripper, `task_completed`/`task_failed`, `run_vlm_review`. |
| `task_state.py` | `TaskState`: per-subtask mutable state contract. |
| `env.py` | Simulator process: message loop (`CAPTURE_IMAGES`, `EXECUTE_TRAJECTORY`, grippers, `GET_STATE`…), sim envs, camera capture, marker drawing. |
| `helpers/main_utils.py` | `get_exec_locals` (subtask tool injection), `execute_blocks_from_log`, `learn_from_past_trajs`. |
| `helpers/perception_scene_analysis.py` | `run_scene_perception` (fresh head capture + perception VLM call + prompt assembly) and affordance pointing: `_capture_fresh_head_image`, `_parse_affordance_points_block`, `_process_affordance_points` (2D→3D + text splice). |
| `helpers/video_utils.py` | Per-attempt review clip encoding + incremental `ffmpeg -c copy` growth of the per-camera full session video. |
| `providers/llms/message_media.py` | Provider-agnostic multimodal message building: `encode_media`, `append_images`, `append_videos`, `append_to_messages` (canonical `image_url` / `video_url` parts each provider converts). |
| `segmentation_adapter.py` | Provider-agnostic 2D segmentation dispatch. |
| `utils.py` | Point-cloud → bounding cube, 3D↔2D projection, intrinsics/extrinsics. |

---

## 4. Planner

The planner is a **single continuous agentic conversation** (Arch 1). It does **not**
generate motion code — only decomposition + dispatch.

- **Scene perception (pre-step)**: before **every** planner LLM call, `run_scene_perception`
  **re-captures the head camera** and  runs the `--planner-perception-vlm` model (default `or-google/gemini-3.5-flash`) on 
  head image with `SCENE_PERCEPTION_PROMPT` (`[INSERT USER COMMAND TASK]` ← command). Its
  free-text answer (objects, target-affordance visibility/occluders, collision risks) is
  injected into the planner prompt's `[INSERT SCENE ANALYSIS]` section on the first call,
  and prepended as "UPDATED SCENE ANALYSIS" on subsequent iterations (subtasks may have
  changed the scene). The same per-turn analysis is also **forwarded to the subtask agent**
  (`execute_subtasks` → `execute_task(scene_analysis=...)`), filling `MAIN_PROMPT`'s
  `[INSERT SCENE ANALYSIS]` section — perception is run once per planner turn, not re-run
  per subtask. `DECOMPOSITION RULES` reference this section instead of raw pixels.
  Best-effort: perception failure yields a fallback string and the planner continues.
- **Target-object affordance pointing**: `SCENE_PERCEPTION_PROMPT` also asks the perception
  VLM for a ranked (best-first) JSON block of **4 grasp-affordance points** on the target
  object, delimited by an `AFFORDANCE_POINTS:` marker + fenced ```json. `run_scene_perception`
  parses this block out, converts the 2D points to **3D world coordinates**
  (`API.convert_2d_point_to_3d_world`), and **replaces the raw 2D points with the 3D coords**
  in the scene-analysis text (so the planner never reasons over confusing 2D pixels). The
  coordinate format is VLM-dependent: if `--planner-perception-vlm` contains `gemini`, points
  are `[y, x]` normalized 0-1000 and denormalized via
  `features_markers.bbox_providers.gemini_bbox_provider.denormalize_yx_point_to_xy_pixels`;
  otherwise they are `[x, y]` pixels used directly. The 2D→3D mapping is stored on
  `ctx.affordance_points` for debugging, and a green→yellow (best→worst) overlay of the 2D
  points is saved to `images/affordance_points_{object}.png`.

<details><summary>Affordance-pointing code-level detail</summary>

- Prompt: `prompts/scene_perception_prompt.py` splits the pointing block into
  `AFFORDANCE_POINTING_SECTION`, appended into `SCENE_PERCEPTION_PROMPT`'s
  `[INSERT AFFORDANCE POINTING SECTION]` slot only when `--affordance-points` is on
  (with `--no-affordance-points` the section is empty, so the VLM never generates points
  and no 2D→3D conversion runs). The section carries a `COORDINATES_FORMAT_PLACEHOLDER`
  token that `run_scene_perception` replaces with `config.affordance_coords_format_gemini`
  ("The points are in [y, x] format normalized to 0-1000") or
  `config.affordance_coords_format_pixels` ("The points are in [x, y] pixel coordinates").
- `helpers.perception_scene_analysis._parse_affordance_points_block(text)` splits the response at `AFFORDANCE_POINTS:`
  and extracts the JSON array (fenced or first `[...]`).
- `helpers.perception_scene_analysis._process_affordance_points(ctx, command, analysis, points)` builds the
  `[[x,y],...]` list (denormalizing when gemini), calls
  `ctx.api.convert_2d_point_to_3d_world(points_xy, object_name)`, appends an
  "AFFORDANCE POINTS (…3D world coords, best-first)" section, and records
  `ctx.affordance_points`.
- `api.API.convert_2d_point_to_3d_world(points_xy, object_name)`: reuses
  `_capture_head_image_and_depth()` (extracted shared helper also used by `detect_object`),
  reads `depth_array[y, x]` per point (bounds/NaN-guarded), calls
  `utils.get_world_point_world_frame(head_pos, head_orient_q, "head", head_image_size,
  [x, y, z], cam_info)`, prints `Affordance-pointing of {object_name}: <xyz>`, and saves the
  overlay via `_overlay_affordance_points` (cv2 circles colored green→yellow by rank).
- `prompts/task_failure_prompt.py` now advises retrying the different candidate positions of
  the same object (ranked affordance points + the segmentation-bbox position) instead of
  repeating a failed grasp position.

</details>
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

`execute_task(ctx, prompt, max_attempts=None, in_context_example=True, scene_analysis="") -> TaskResult`.
The subtask agent writes and runs Python; each attempt ends in a VLM review that decides
retry vs. done.

- Fresh `api.task = TaskState(command=prompt, max_attempts, start_trajectory_step=api.trajectory_step)`.
- **Optional arm re-home** (`--reset-eef`): before capturing the EE start pose, call
  `api.reset_eef()` → sends `RESET_EEF`, which re-homes the **arm only** to
  `ee_start_position` (gripper open) and does **not** touch object/world state or
  `trajectory_step`. Default off; when on it makes a later subtask start from the same
  canonical arm pose as subtask #1 (addresses "open-door as subtask #2 starts far from the
  door" — the carried-over EE pose otherwise seeds a bad first-attempt trajectory).
- **Scene analysis (reused, not re-run)**: the planner passes its already-computed
  `scene_analysis` (from `run_scene_perception`) down through `execute_subtasks`; it fills
  `MAIN_PROMPT`'s `[INSERT SCENE ANALYSIS]` section and is reused across this task's
  attempts. `execute_task` does **not** run the perception VLM itself (one perception call
  per planner turn, not per subtask/attempt). Under `--no-plan`, `run_plan` runs perception
  once and passes it in.
- **First attempt** prompt = `MAIN_PROMPT` filled with the `detect_object` tool + optional
  in-context example + SCENE ANALYSIS + head image (`config.rgb_image_head_path`), sent as
  `role="system"`. Optional `--prepend-prompt` text is prepended to the first command only.
- **Loop** until `task.completed_task` — `run_task_agent_loop`, split into focused funcs
  (all in `agent_runner.py`):

  | function | intent | states / cases handled |
  |---|---|---|
  | `execute_python_blocks(ctx, task, assistant_content) -> feedback` | the **only** place LLM-generated code runs; must never let the LLM believe an action ran when it did not | 1) no ```` ```python ```` block → `NO_TOOL_CALL_PROMPT`; 2) block raises → `ERROR_CORRECTION_PROMPT` (block number + traceback), remaining blocks **aborted**; 3) block prints (`< 2000` chars) → collected into `PRINT_OUTPUT_PROMPT`, execution **continues**; 4) `task_completed()` / `task_failed()` fires mid-response → remaining blocks skipped on purpose, no notice; 5) blocks skipped by case 2 → `BLOCKS_NOT_EXECUTED_PROMPT` naming exactly which blocks never ran |
  | `handle_task_failure(ctx, task, prompt, scene_analysis, attempt_summaries, feedback) -> messages` | close a FAILED attempt and open the next one (state: `task.failed_task` set by reviewer / `task_failed()` / no-motion guard) | summarize attempt (`TASK_SUMMARY_PROMPT`, images stripped) → append to `attempt_summaries`; re-baseline `start_attempt_trajectory_step`; fresh conversation (`messages=[]`) with retry `MAIN_PROMPT` (no `detect_object`, no in-context example, latest EE pose) + combined `TASK_FAILURE_PROMPT`; clear `failed_task` |
  | `continue_task_turn(ctx, task, feedback) -> messages` | continue the CURRENT attempt (not completed, not failed): feed execution results back and get the next response | attaches latest head/wrist frames + `EEF_POS_SNIPPET` per `config.ENABLE_EEF_POS_IMAGE` / `--lm-images`; never sends an empty user turn (providers such as Bedrock reject an assistant-final conversation) → falls back to `CONTINUE_TASK_PROMPT` |
  | `run_task_agent_loop(ctx, task, prompt, scene_analysis, attempt_summaries) -> messages` | drive one (sub)task to termination: execute → feedback → next LLM turn | invariant: conversation always ends on an assistant response with pending blocks. Per iteration exactly one of: `completed_task` → exit; `failed_task` → `handle_task_failure`; else → `continue_task_turn`. Assumes the first assistant response was produced by `execute_task` |

  - All blocks of one response share a single namespace (a variable/import from an earlier
    block is visible to later ones); it is rebuilt per response.
  - **A `print()` must not abort the response.** Earlier versions reused one `error` flag for
    "exception" and "has print output", so a first block that printed silently dropped every
    later block (typically all `execute_trajectory` calls) while the LLM was told only
    "Print statement output: …" and then claimed the task was done.
  - `task_completed()` triggers review (see §7); on review failure `failed_task=True`.
- **Retry path** (`failed_task`): generate a `TASK_SUMMARY_PROMPT` summary (appended to
  `attempt_summaries`), reset `start_attempt_trajectory_step`, rebuild the prompt with the
  **no-detect-object** variant + latest EE pose + the same reused SCENE ANALYSIS +
  combined `TASK_FAILURE_PROMPT` summary, reset `messages`, continue. (The retry turn is
  text-only — no head image — so the reused scene analysis is its main visual context.)
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

### 6.1 How grasp coordinates are computed (image w/ a box → grasp pose)

Given a head image + a text label (e.g. `"box"`), the grasp point and orientation are
derived from the object's fitted 3D bounding cube. Chain:
`detect_object` → `utils.get_bounding_cube_from_point_cloud` → `get_world_point_world_frame`.

Steps:
1. **Segment** the box in 2D (LangSAM / SAM3), take the largest contour (`get_max_contour`).
2. **Deproject** every in-contour pixel `(col, row, depth)` to world XYZ
   (`get_world_point_world_frame`) → raw object point cloud.
3. **Clean the cloud**: DBSCAN (eps=2.5cm) clusters, keep the cluster *closest to the
   camera* (drops mask-bleed/background); then a top-surface Z filter drops table bleed.
4. **Fit footprint**: `minimum_rotated_rectangle` (shapely) over the XY points → rotated
   rectangle; if footprint <1.5cm (thin ridge) refit using all cluster points.
5. **Build cube**: 4 top corners @ `max_z`, 4 bottom @ `min_z`, and a centroid appended to
   each → 10-element `bounding_cube_world_coordinates`.
6. **Grasp point** = `cube[4]` (top-surface centroid), Z reduced by
   `config.bounding_cube_depth_offset` → printed as `Position` (`obj_position`).
7. **Grasp orientation** = `arctan2` of the rectangle edges (width & length axes); robot
   aligns the gripper to the **shorter** side. Dimensions (W/L/H) come from corner norms.

The printed `Position` + dimensions + orientation are what the code-gen LLM consumes to
generate the grasp trajectory. (Alternate source: `api.get_grasp_poses(object_name)` loads
precomputed GraspGen candidates from `outputs/graspgen/grasp_poses_{object}.npz`.)

<details><summary>Code-level detail — cube indices & deprojection paths</summary>

- **Cube layout** (`bounding_cubes.append(box_top + box_btm)`), indices:
  - `0..3` top corners (@ `max_z`), `4` = **top centroid** ← grasp point,
  - `5..8` bottom corners (@ `min_z`), `9` = bottom centroid.
  - Dims: `width=|cube[1]-cube[0]|`, `length=|cube[2]-cube[1]|`,
    `height=|cube[5]-cube[0]|`.
  - Orientation: `orient_width=arctan2(box[1]-box[0])`,
    `orient_length=arctan2(box[2]-box[1])`.
- **Two deprojection paths** in `get_world_point_world_frame`:
  - **New** (`cam_info['new_3d_proj']`): pixel+depth → NDC → `inv(Projection@View)` →
    perspective divide → world XYZ.
  - **Legacy PyBullet**: recenter pixel `[u-W/2, H/2-v, 1]`, head remap `[-py,-px,pz]`,
    `world_cam = inv(K)·pixel · depth`, `world = Rt @ [world_cam;1]`.
- **DBSCAN** `eps=0.025, min_samples=5`; target cluster = `min` mean-distance to camera.
- **Top-surface filter**: keep pts with `z > max_z - config.point_cloud_top_surface_filter`
  (falls back to all target points if <3 survive).
- **Footprint refit** when `min(width,length) < 0.015`.
- `project_3d_world_pos_to_2d_pixel` reprojects `obj_position` back to a pixel for logging.

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
  - build this attempt's trajectory videos (`create_trajectory_videos(logger, start_attempt_trajectory_step)`
    → `task.review_clips`, see *Trajectory videos* below); `attempt_number += 1`;
  - **no-motion guard** (non-replay): if `trajectory_step <= start_attempt_trajectory_step`
    the robot never moved in this attempt (generated blocks did not reach the env), so the
    claim is rejected — `failed_task=True` (retry), or on the last attempt
    `completed_task=True` with `review_succeeded=False` (never `accepted_without_review`);
  - **final attempt** (`attempt_number >= max_attempts`, non-replay) → accept without
    review: `completed_task=True`, `accepted_without_review=True`;
  - else `TASK_COMPLETED` then dispatch review by `--review-provider`.

#### Trajectory videos (`helpers/video_utils.py`)
Per review, only the **new** attempt's frames are encoded; the full session video is grown
by appending that clip — no O(n²) re-encode from frame 0 (the old code re-encoded every
frame twice on every attempt).

- `build_attempt_clip(base, start_idx)` → `config.video_folder/<base>_attempt_<start>_<end>.mp4`
  (cv2, `config.trajectory_video_fps`), where `<end>` is the index of the last frame actually
  written — `create_video_from_images` encodes to a temp file and resolves the `{end}` token
  in `output_filename` on rename. This is the reviewer's input.
- `update_full_video(base, clip)` → appends the clip to `config.video_folder/<base>_full.mp4` with
  `ffmpeg -f concat -c copy` (stream copy, no re-encode). First call copies the clip.
  Fallback when ffmpeg is missing / concat fails: re-encode the whole sequence from 0.
- `build_review_clips(logger, start_idx)` does both for head + wrist and returns the clip
  paths (`[]` when the attempt produced no frames → drives the zero-frame guard).

#### Review media: video (preferred) vs key frames (fallback)
- `models.model_supports_video(model)`: `gemini-*` ✅, `or-*gemini*` ✅ (OpenRouter
  `video_url`), `aws-*` ❌ (Claude on Bedrock has no video modality; video blocks are
  Converse-API/Nova-only), `azure-*` ❌ (images only), other OpenAI-compatible ❌.
- **Video mode**: scene image first, then the 2 attempt clips (head, wrist); key frames are
  dropped entirely. Prompt uses `REVIEW_MEDIA_VIDEO_SECTION` (lists the clips + fps and
  asks for `cam @ MM:SS` references).
- **Frames mode** (auto-fallback): the legacy stride-5 head + stride-7 wrist key frames,
  capped by `config.max_allowed_vlm_images`; prompt uses `REVIEW_MEDIA_FRAMES_SECTION`.
- The `REVIEW_SCENE_ANALYSIS_SECTION` sentence about "the remaining attachments" switches
  between `REMAINING_MEDIA_SENTENCE_VIDEO` / `..._FRAMES` accordingly.
- `run_vlm_review()` (default `vlm`): attaches the attempt clips (video mode) or the
  subsampled head/wrist key frames (fallback, capped at `config.max_allowed_vlm_images`);
  asks the review model (`vlm` = main model, `vlm:<model>` = override) for strict JSON
  `{success, reasoning, improvement_steps}`.
  - **Zero-frame guard**: if neither clips nor frames exist from `start_attempt_trajectory_step`,
    the VLM is **not** called — `success=False` with reason "no trajectory frames captured /
    robot did not move". Without it the reviewer judged from conversation text alone and
    hallucinated key frames ("the door is clearly swung open") that were never rendered.
  - **Start-of-attempt scene**: the head image the perception VLM analyzed (saved by
    `run_scene_perception` to `config.scene_analysis_image_path`) is attached **first**,
    before the clips/key frames, and `REVIEW_SCENE_ANALYSIS_SECTION` (from `task.scene_analysis` /
    `task.scene_analysis_image_path`) tells the reviewer that image is *not* execution
    evidence and what the remaining attachments are — an unambiguous "before" reference.
    One image slot is reserved for it in the `max_allowed_vlm_images` budget (frames mode).
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

### XMem review path (legacy, `--review-provider xmem`)

Geometric alternative to the VLM reviewer: instead of *looking* at frames, it **tracks the
segmented objects through the whole trajectory** and lets the LLM judge success from the
resulting numeric pose sequences.

- **Seeding (during `detect_object`)** — every LangSAM mask is baked into a single label
  image `./images/xmem_input.png` (`utils.save_xmem_image`); object *k* = pixel value *k*.
  The file is reset to zeros at the start of `detect_object`.
- **Tracking (at `task_completed`)** — `models.get_xmem_output` runs XMem
  (`InferenceCore`) over head-camera frames `trajectory/rgb_image_{step}.png` for
  `step = 0 .. task.trajectory_length`, propagating the seed masks frame-by-frame. Returns
  one integer label mask per sampled frame; overlays saved to `images/xmem_output_{step}.png`.
- **Pose reconstruction** — for each tracked object and each frame, the mask + depth image
  go through `utils.get_bounding_cube_from_point_cloud` → world-frame **position**
  (cube centre) and **z-rotation**; orientations are unwrapped to the closest of the 4
  symmetric candidates relative to the previous frame (avoids ±90° flips).
- **Judgement** — positions/orientations (every `xmem_lm_input_every`-th sample) are
  appended to `SUCCESS_DETECTION_PROMPT` with the task command; the LLM replies with a
  python code block calling `task_completed()` or `task_failed()`, which is `exec`'d.
- **Requirements** — `XMem` submodule + weights `XMem/saves/XMem.pth`; model loaded eagerly
  in `agent_runner` only when `--review-provider xmem` (hard error if missing). CUDA
  strongly recommended (`torch.cuda.amp.autocast`).
- **Limits** — head-camera only, depth-based, needs LangSAM masks to exist; no textual
  reasoning/`improvement_steps`, so `review_reason` stays empty and retries get less
  guidance than the `vlm` path. Default remains `vlm`.

<details><summary>Code-level details</summary>

**Config** (`config.py`)
- `xmem_config`: `top_k=30`, `mem_every=5`, `deep_update_every=-1`, long-term memory on
  (`num_prototypes=128`, `min/max_mid_term_frames=5/10`, `max_long_term_elements=10000`).
- `xmem_output_every=1` (track stride), `xmem_visualise_every=1` (overlay save stride),
  `xmem_lm_input_every=20` (pose subsampling fed to the LLM).
- Paths: `xmem_input_path = ./images/xmem_input.png`,
  `xmem_output_path = ./images/xmem_output_{step}.png`.

**Seeding** (`utils.save_xmem_image`, called at end of `API.detect_object`)
```python
xmem_array = np.array(Image.open(config.xmem_input_path).convert("L"))
xmem_array = np.unique(xmem_array, return_inverse=True)[1].reshape(xmem_array.shape)  # → 0..N labels
for mask in masks:
    xmem_array[mask.astype(bool)] = np.max(xmem_array) + 1   # each mask gets a new label id
Image.fromarray((xmem_array / max_val * 255).astype(np.uint8)).save(config.xmem_input_path)
```
Note the file is stored **normalised to 0..255** and re-quantised via `np.unique(...)` on
load, so label ids round-trip. `api.detect_object` first writes an all-zeros image the size
of the depth map, so labels accumulate only within one detection pass.

**Model load** (`agent_runner.py`)
```python
sys.path.append("./XMem/")
from XMem.model.network import XMem
xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
```
Passed into `API(..., xmem_model, device)`; `None` for other providers.

**Tracking loop** (`models.get_xmem_output`) — XMem imports are lazy so the dependency is
optional:
```python
processor = InferenceCore(model, config.xmem_config)
processor.set_all_labels(range(1, num_objects + 1))
for i in range(0, trajectory_length + 1, config.xmem_output_every):
    frame_torch, _ = image_to_torch(np.array(Image.open(rgb_path(i))), device)
    prediction = processor.step(frame_torch, mask_torch[1:]) if i == 0 else processor.step(frame_torch)
    masks.append(torch_prob_to_numpy_mask(prediction))
    if i % config.xmem_visualise_every == 0:
        Image.fromarray(overlay_davis(frame, prediction)).save(xmem_output_path(i))
```
`num_objects = len(np.unique(seed_mask)) - 1` (minus background).

**Pose extraction** (`api.task_completed`, per object `1..num_objects`, per frame `i`)
- binarise: `object_mask = (mask == object)`;
- depth from `trajectory/depth_image_{i}.png` scaled `/255.`;
- `utils.get_bounding_cube_from_point_cloud(rgb, [object_mask], depth, head_camera_position,
  head_camera_orientation_q, object-1, cam_info)` → `bounding_cubes`, `orientations`;
- `position = bounding_cube[4]` (centre); orientation wrapped to `[-pi, pi]`;
- empty result → frame dropped and `idx_offset += 1` so the previous-orientation lookup
  stays aligned;
- symmetry disambiguation:
```python
possible = [wrap(orientation + k*pi/2) for k in range(4)]
orientation = possible[argmin(circular_distance(possible, previous_orientation))]
```

**Prompt** (`prompts/success_detection_prompt.py`) states the axis convention (x right,
y depth, z up, metres; z-rotation −pi..pi), injects the command, then appends per object:
```
<segmentation_text> trajectory positions and orientations:
Positions:
[[...]]
Orientations:
[...]
```
Reply is parsed by splitting on the python fence marker and `exec`-ing blocks with local
`task_completed` / `task_failed` bound to the API methods — i.e. the LLM's chosen call
directly sets `completed_task` / `failed_task`. `task_completed()` re-entry is guarded by
the attempt counter (final attempt accepts without review).

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

- **Attempt clips named with the real end frame**: `<base>_attempt_<start>_inf.mp4` →
  `<base>_attempt_<start>_<last_frame_idx>.mp4` (e.g. `rgb_image_attempt_261_410.mp4`).
  `debug/dbg_utils.create_video_from_images` now supports an `{end}` token in
  `output_filename` (and uses it in the default name when `end_idx` is infinite): frames are
  encoded to a temp file and renamed once the last written index is known. Also fixed a
  crash in the no-frames path (`find_available_frame` returns `(None, None)` → the error
  message did `None + 1`), which had been silently swallowed by `build_review_clips`.

- **Fix: stale scene-analysis image (perception saw the world BEFORE the last subtask)**:
  `run_scene_perception` analysed `./images/rgb_image_head.png` as left by the previous
  subtask's `detect_object` — the only capture happened *after* the VLM call, inside
  `convert_2d_point_to_3d_world`. Symptom (log 06/08 18:12): `scene_analysis_head_410.png`
  still showed the grey cylinder in front of the door although `trajectory/rgb_image_410.png`
  showed it already cleared, so the analysis (and the affordance points + the planner's
  attached image) kept describing a removed occluder.
  Fix: `helpers/perception_scene_analysis._capture_fresh_head_image(ctx)` re-captures the
  head camera at the start of every perception run (best-effort; falls back to the last
  image on failure). `API._capture_head_image_and_depth(capture=True)` /
  `API.convert_2d_point_to_3d_world(..., capture=True)` gained the flag, and perception
  passes `capture=False` so the 2D→3D conversion reuses that exact frame instead of
  re-rendering (also guarantees the points match the analysed pixels). The load half was
  split out into `API._load_head_image_and_depth()`.

- **All videos now written to `./images/videos`**: new `config.video_folder` const; every
  clip / full video (`helpers/video_utils.py`) and the generic
  `debug/dbg_utils.create_video_from_images` default output dir point there (previously the
  parent of the frames folder, i.e. `./images`). `common_utils.ensure_image_dirs_exist`
  creates (and with `--delete-images` clears) it alongside `images/trajectory` and
  `images/overlay`; the IPC/metaworld tests assert the new location.

- **VLM reviewer takes VIDEO instead of dozens of key frames**:
  - New `providers/llms/message_media.py` — one canonical multimodal message builder shared by
    every provider (`encode_media`, `append_images`, `append_videos`, `append_to_messages`),
    moved out of `azure_openai.py` (which now re-exports it). Video parts use the OpenRouter
    `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,…"}}` shape; Gemini converts
    to `inline_data`, Bedrock raises a clear error (Claude has no video modality — video blocks
    are Converse-API/Nova-only), Azure OpenAI is images-only.
  - `models.model_supports_video(model)` gates it; `run_vlm_review` sends scene image + the 2
    attempt clips when supported, else auto-falls back to the old key-frame path.
    `models.call_llm_cached/_call_llm_provider_wrapper` gained `video_paths`; the
    strip/has-media helpers and prompt logging now cover video parts too.
  - `prompts/review_prompt.py`: `[INSERT MEDIA SECTION]` with `REVIEW_MEDIA_VIDEO_SECTION` /
    `REVIEW_MEDIA_FRAMES_SECTION` variants (+ matching "remaining attachments" sentence).
  - **Perf**: `helpers/video_utils.py` encodes ONLY the current attempt's frames
    (`<base>_attempt_<start>_inf.mp4`) and appends them to `<base>_full.mp4` via
    `ffmpeg -f concat -c copy`. Previously every `task_completed` re-encoded all frames from
    step 0 for both cameras (589 frames × 2, per attempt). The full video is still updated
    before each review so long runs stay inspectable mid-execution.
  - Verified live end-to-end against `or-google/gemini-3.6-flash` (OpenRouter) and
    `gemini-2.5-flash` (direct REST): both correctly describe motion direction and final frames.

- **Phantom-success fix (multi-block execution + review guards)** — a `door` run removed the
  occluding cylinder, never touched the handle, yet reported success. Three chained defects,
  all fixed:
  - `agent_runner.py` reused one `error` flag for "exception" and "block printed something", so
    a first block that called `print()` **silently skipped the remaining 6 blocks** (all
    `execute_trajectory` calls). The LLM saw only the print output and asserted every step had
    run. Now prints never abort the response, and any genuinely skipped blocks are reported via
    the new `BLOCKS_NOT_EXECUTED_PROMPT`.
  - `run_vlm_review` was called with **0 key frames** (nothing moved → no frames from
    `start_attempt_trajectory_step`), and the reviewer hallucinated a final frame showing the
    door open. It now short-circuits to `success=False` without calling the VLM.
  - `task_completed()` gained a **no-motion guard** (`trajectory_step <= start_attempt_trajectory_step`)
    so a completion claim without any executed trajectory can never be accepted (the planner
    then sees failure instead of a phantom success).
  - New prompt constants in `prompts/print_output_prompt.py`: `NO_TOOL_CALL_PROMPT`,
    `BLOCKS_NOT_EXECUTED_PROMPT`.
  - `execute_task`'s inline `while` loop refactored into `run_task_agent_loop` +
    `execute_python_blocks` / `handle_task_failure` / `continue_task_turn` (see §5 table).

- **Scene analysis + its image given to the reviewer VLM**: `run_scene_perception` now snapshots
  the exact head image it analyzed to `config.scene_analysis_image_path`
  (`./images/scene_analysis_head_{step}.png`, kept out of the trajectory frames) and records it on
  `ctx.scene_analysis_image_path`. `execute_task` threads both the analysis text and that path into
  `TaskState.scene_analysis` / `.scene_analysis_image_path`, and `run_vlm_review` prepends a
  `REVIEW_SCENE_ANALYSIS_SECTION` ("START-OF-ATTEMPT SCENE") to `REVIEW_PROMPT` and attaches the
  scene image **first**, ahead of the key frames. The section states explicitly that the first
  image is *not* a trajectory frame and that all remaining images are the listed key frames, so the
  reviewer has an unambiguous "before" reference. The image budget reserves a slot for it
  (`max_allowed_vlm_images - 1 - 1`).

- **Bedrock opus-5 `'text'` KeyError fix**: `providers/llms/aws_bedrock.py` no longer indexes
  `model_response["content"][0]["text"]`. Newer models (e.g. `aws-eu.anthropic.claude-opus-5`)
  can emit `thinking`/`redacted_thinking` blocks first, so `content[0]` isn't a text block and
  every attempt failed with `unexpected error: 'text'`. New `_extract_text_from_bedrock_content`
  joins **all** `type == "text"` blocks (falling back to any block with a `text` field) and
  raises a descriptive error (with `stop_reason` + block types) when there is genuinely no text.

- **Scene perception extracted to `helpers/perception_scene_analysis.py`**: `run_scene_perception`,
  `_parse_affordance_points_block` and `_process_affordance_points` moved out of `agent_runner.py`
  (which now imports `run_scene_perception` from the new module).

- **Target-object affordance pointing in scene analysis**: perception VLM now also returns a
  ranked JSON block of 4 grasp-affordance points (`AFFORDANCE_POINTS:` marker in
  `SCENE_PERCEPTION_PROMPT`). `run_scene_perception` parses them, converts 2D→3D via new
  `API.convert_2d_point_to_3d_world` (reusing `_capture_head_image_and_depth` extracted from
  `detect_object`), and **replaces the 2D points with 3D world coords** in the scene text.
  Coordinate format branches on `gemini` in `--planner-perception-vlm` ([y,x] 0-1000 +
  denormalize vs [x,y] pixels). Denormalization extracted to importable
  `denormalize_yx_point_to_xy_pixels` in robotic_perception `GeminiBBoxProvider`. 2D→3D map
  stored on `ctx.affordance_points`; green→yellow overlay saved to
  `images/affordance_points_{object}.png`. `task_failure_prompt` advises trying the different
  candidate positions (affordance points + segmentation-bbox) on retry. Toggle with
  `--affordance-points` / `--no-affordance-points` (default on); when off the
  `AFFORDANCE_POINTING_SECTION` is omitted from the perception prompt and no conversion runs.

- **Scene analysis shared with subtask agent**: `MAIN_PROMPT` gains a `SCENE ANALYSIS`
  section (`[INSERT SCENE ANALYSIS]`). The planner's already-computed `run_scene_perception`
  output is forwarded to the subtask agent via `PlannerAPI.scene_analysis` →
  `execute_subtasks` → `execute_task(scene_analysis=...)` and reused across the task's
  attempts (including the otherwise image-less retry turn). Perception is **not** re-run
  inside `execute_task` (one call per planner turn); `--no-plan` runs it once and passes it
  in. Also replaced `getattr(args, ...)` with direct `args.` access in `agent_runner.py`
  (parser guarantees the fields).
- **§6.1 grasp-coord doc**: documented how a box's grasp point/orientation is derived from
  the fitted 3D bounding cube (`cube[4]` top centroid − `bounding_cube_depth_offset`;
  `arctan2` edge orientation; DBSCAN + top-surface cleaning; shapely rotated rect).
- **`--max-planner-iter` (default 4)**: CLI cap on planner LLM turns per command; `run_plan` defaults to `ctx.args.max_planner_iter` (was hardcoded 16).
- **Shared `CODE_BLOCK_CONVENTIONS`**: dedup ```python-block/logger/print/import rules into one `main_prompt` constant reused by `MAIN_PROMPT` + `PLANNER_PROMPT` via `[INSERT CODE BLOCK CONVENTIONS]`; clarifies `logger` is an injected object (not a module) to stop `import logger` errors.
- **Shared exec env per response**: ```python blocks in one assistant/planner response now share a namespace (rebuilt each turn), so later blocks see earlier vars/imports; reset between responses.
- **Empty user-turn fix (Bedrock)**: in `execute_task`'s continuation branch, a code
  block that ran successfully but printed nothing (or printed ≥2000 chars) left both
  `new_prompt` and `_imgs_paths` empty when `ENABLE_EEF_POS_IMAGE`/images are off, so no
  user message was appended and the conversation ended on the assistant turn. AWS Bedrock
  (e.g. Claude Opus 4.8) rejects this with *"does not support assistant message prefill;
  conversation must end with a user message."* Now falls back to `CONTINUE_TASK_PROMPT`
  so the turn always carries user content.
- **Per-subtask perceive→replan**: `execute_subtasks` now runs **only the next subtask**
  then returns; `run_plan` re-runs `run_scene_perception` and re-invokes the planner after
  **every** subtask (not just at batch end / on failure). The planner reevaluates the
  UPDATED SCENE ANALYSIS and can insert prep subtasks (e.g. "move the arm clear of the
  target") between subtasks. Return shape changed to
  `{executed, success, result, remaining}`; `max_iterations` default 8 → 16. Fixes: after
  removing an occluding gray cylinder, the open-door subtask started while the robot arm
  itself occluded the door handle. Prompt/example in `prompts/planner_prompt.py` updated.
- **Scene perception pre-step**: new `--planner-perception-vlm` (default
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
