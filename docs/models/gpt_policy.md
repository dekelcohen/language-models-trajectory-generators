# GPT-Policy - In-Context Robot Learning with VLM Agents

Reference notes for the sibling repo `D:\NLP\Robotics\VLM_Robotics\GPT-Policy`
(paper arXiv:2609.19138). Every code reference below names a real file and symbol in that
repo, so the document stays greppable.

---

## Summary (copy-paste block)

```
* GPT-Policy - In-Context Robot Learning with VLM Agents
  ) Goal: TLDR - human demo video / robot rollout with actions / goal image / self-history are
    compiled offline into a compact annotated keyframe story --> a fixed frontier VLM (GPT-6 Astra)
    sees that story plus a small JSON observation and returns exactly one structured robot-tool call
    --> a constrained Cartesian adapter resolves the TCP target, interpolates and SLERPs the path,
    solves per-sample IK, retimes with Ruckig and executes --> measured result and rejection reason
    are fed back as the next observation. No gradient updates, no action head, no fine-tuning.
  ) Problem: no finite demonstration corpus covers every deployment situation, and current robot
    policies cannot learn at test time. The paper asks how much robotic in-context learning is
    already reachable with a general VLM that was never trained as a robot policy.
  ) Tags: Zeroshot, OneShot, VLM, TextPrompt, ImagePrompt, GoalImage, HumanVideos, KeyFrames,
    Planning, PrimitivePolicies, Closed-Loop, Long Horizon, MotionPlanner (Cartesian interpolation
    plus IK only, no collision checking), Imitation Learning (in-context, not trained)
  ) Project: https://cheng-haha.github.io/GPT-Policy/
  ) Paper: https://arxiv.org/pdf/2609.19138
  ) Video: https://cheng-haha.github.io/GPT-Policy/#results - full experiment gallery, three trials
    per condition, human demos at real time and robot runs accelerated as labeled
  ) Method:
    ) Data Collection: no training data. Reference episodes only - first or third person RGB human
      videos; teleoperated robot rollouts recorded as timestamped images plus measured joint and
      end-effector states plus motion and gripper commands; goal images as overhead screenshots or
      operator photographs. One demonstration per task.
    ) Data Augmentation: none. The pipeline is deterministic sampling plus VLM annotation.
    ) Preprocessing: ffprobe reads real presentation timestamps, never nominal FPS; one ffmpeg pass
      decodes all requested frames via a select=eq(n,i)+... expression; 2 fps candidate sampling,
      at most 24 candidates per 30 s window, 768 px candidates and 1280 px keyframes at JPEG q=3;
      extra candidates injected at +/-0.3 s around gripper transitions and motion stops; multi-camera
      views paired by shared capture clock within 0.1 s; numeric samples compressed to endpoints plus
      about 1 Hz plus gripper-command transitions plus channel loss and recovery boundaries.
    ) Training: none. Model parameters stay fixed for every context condition.
      ) Regularization: not applicable.
    ) Inference: per decision the runner snapshots cameras, reads robot state, builds a compact JSON
      observation, asks the VLM for one tool call, executes it, and returns execution or rejection
      feedback for the next turn.
      ) Test time techniques: two-pass keyframe selection (per-window select, then a global review
        that merges windows into 12 to 16 frames and must keep the first and last); strict JSON
        schemas with host-side validation and retry; guard-rail sentences such as gripper closure
        alone does not prove a grasp and similar start and end poses do not imply no motion;
        check_path as a look-before-you-move probe; IK rejection returned as recoverable feedback so
        the model replans instead of crashing; a sliding live-image window that drops stale live
        frames and replays a text-only historical execution record; content-addressed caching so
        the expensive vision passes are paid once per video, instruction and config.
      ) Tools: move_to, move_eef_chunk, set_gripper, check_path, locate_point, done, give_up -
        defined as data in configs/tools.json, with a mandatory short note field and separate
        parameter sets for single-arm and bimanual modes.
      ) Close-loop: yes. Every step re-observes cameras and proprioception; the previous tool result
        is embedded as previous_result; terminal tools trigger a return-home; a decision budget ends
        the episode; model-reported done is explicitly not treated as verified physical success.
    ) Usage of lower level models: a pixel localizer converts a clicked pixel into an undistorted
      camera ray and, when two wrist views agree, into a metric 3D point with parallax, condition
      and residual gates; no detector, no segmentation, no grasp model, no video prediction.
  ) Results:
    ) Real robots, GPT-6 Astra, three trials per condition. Pick Red Towel 0/3 without context and
      2/3 with a human video; Pick Up Notebook 0/3 and 2/3; Unscrew Bottle Cap 0/3 none, 2/3 with
      robot video, 3/3 with video plus action; Remove and Reinsert Plug 0/3, 0/3, 2/3.
    ) Goal-image tasks, self-interaction history tasks and human-robot interaction tasks all 3/3.
    ) Context also cuts effort - Pick Red Towel falls from about 96 decisions to about 77, and
      Unscrew Bottle Cap to about 55 decisions with video plus action.
    ) Baselines are matched model comparisons and context ablations rather than other frameworks.
  ) Pro: no robot data and no training - any capable VLM can be dropped in; the demonstration digest
    is portable text plus a few JPEGs and can be replayed with zero model calls.
  ) Pro: the controller is deterministic and auditable - the model only picks targets, while
    interpolation, IK residual checks and time scaling are host side and bounded.
  ) Pro: the prompt explicitly separates reference data from commands, which suppresses blind replay
    of historical coordinates into a different scene.
  ) Con: execution precision is the bottleneck - contact-rich tasks fail on centimetre and degree
    errors rather than on reasoning.
  ) Con: no collision checking anywhere; inter-arm collisions were observed and the authors state
    existing safeguards are insufficient.
  ) Con: high per-decision latency and API cost; episodes run tens of decisions and many minutes.
  ) Con: success labels are assigned by a human - model-reported completion is not verification.
  ) Dataset: no training set. Appendix C documents the reference corpus - human demonstrations use
    one view and yield 6 to 8 keyframes per task with listed timestamps; robot demonstrations add
    wrist views so one keyframe time can carry several images; video and video plus action share the
    same selected images and differ only by the presence of measured states and action segments.
  ) Assets: configs/tools.json tool catalog, configs/default.json and configs/examples/*.json
    machine profiles, prompts embedded in source, prices.json for cost accounting.
  ) Code: https://github.com/cheng-haha/GPT-Policy - 214 stars, 3 forks, Python, license
    NOASSERTION, last push 2026-09-17. Install with pip install -e .[arx] or
    python scripts/install_drivers.py; gpt-policy --check runs preflight. README plus per-rig docs
    are thorough; error strings and required motion notes are partly Chinese.
  ) Models: no released weights. Policy is a fixed commercial VLM (GPT-6 Astra by default) driven
    through the Codex app server; the keyframe selector and reviewer use the same provider config
    with high thinking effort; gpt-5.6-luna is used only to name runs and reuse saved requests;
    providers for Claude Code and Gemini CLI are also implemented.
  ) Hardware: ARX X5, I2RT YAM and Morphi Kino arms, single and bimanual; RealSense cameras with a
    fixed top view plus two wrist views; ARX5 SDK and i2rt drivers are optional extras.
  ) Simulator: none for the reported results - all trials are real robots. A simulated machine
    profile exists at configs/examples/simulated.json for dry runs.
  ) References: Ruckig for jerk-limited time parameterization; ARX5 SDK; I2RT; RoboCurve
    inspect-robots; commercial VLMs compared in the ablation - GPT-6 Astra, Claude, Gemini.
```

---

## 1. TL;DR

GPT-Policy runs a commercial VLM as a robot policy without any fine-tuning. An offline *context
compiler* turns demonstration videos, recorded trajectories and goal images into a short annotated
keyframe story, and at run time the model sees that story plus a fresh JSON observation and must
answer with exactly one structured robot-tool call. A constrained Cartesian adapter then resolves
the requested TCP pose, densifies and SLERP-interpolates the path, solves per-sample inverse
kinematics, retimes it with Ruckig, executes it, and feeds the measured result back as the next
observation - so all of the intelligence lives in prompt construction and tool contracts rather
than in learned weights.

## 2. Paper at a glance

The policy is written as `a_t ~ pi_theta( . | T, c_t, o_t, f_{t-1})` with a state update
`(o_{t+1}, f_t) = E(a_t, o_t)`. `T` is the task instruction, `c_t` the compiled context, `o_t` the
observation and `f_{t-1}` the previous execution feedback (absent at the first decision).
`theta` never changes.

Five context families are studied:

| Family | What it supplies | Notable result |
|---|---|---|
| Human video | Procedural order without robot actions | Pick Red Towel 0/3 -> 2/3 |
| Robot video / video + action | Embodied motion, optionally with measured states | Unscrew Bottle Cap 0/3 -> 2/3 -> 3/3 |
| Target image | Desired end state without a sequence | 3/3 |
| Self-interaction history | Closed-loop memory and failure recovery | 3/3 |
| Human interaction | Pointing, turn history, live feedback | 3/3 |

Stated findings: context reliably improves completion, including from human video with no action
labels; aligned action references add most on contact-sensitive tasks; the remaining failures are
execution failures (pose precision, contact, verification, safety), not language or planning
failures. Stated limitations: no collision checking, observed inter-arm collisions, high latency
and cost, and the fact that a model saying `done` is not evidence of physical success.

## 3. Architecture map

| Package | Responsibility |
|---|---|
| `input/` | The context compiler: resolve inputs, extract frames, select keyframes, compress actions, emit the demonstration block, cache everything |
| `harness/` | Provider sessions (Codex / Claude Code / Gemini CLI), the observation and instruction protocol, prompt text, input serialization, usage and cost |
| `tools/` | The model-visible tool catalog and the executor that binds tool names to host callbacks |
| `motion/` | Cartesian path sampling, continuous IK, Ruckig retiming, bimanual synchronization |
| `hardware/` | ARX / YAM / Morphi adapters, cameras, calibration |
| `vision/` | Pixel to ray and two-view triangulation |
| `recording/` | Append-only run trace on disk |
| `runtime/` | The decision loop itself |
| `settings.py`, `configs/` | Layered JSON configuration with `extends`, machine profiles, agent profiles |

## 4. Multi-modal prompts - the context compiler

This is the part most worth stealing. The policy model never receives a video. It receives text
plus a handful of JPEGs.

### 4.1 Input types

| Input | How it enters | What it becomes |
|---|---|---|
| Human video | An `.mp4` path, quoted inside the instruction or passed as a demonstration argument (`input/references.py`) | A `TextPart` keyframe story plus N `ImagePart` keyframes |
| Robot rollout directory | A directory holding `top.mp4`, `config.json`, `status.json`, `events.jsonl`, `video-frames.jsonl`, `states.jsonl`, optional wrist videos (`input/recorded_demo.py`) | Same, with multi-view keyframes and optional numeric state |
| MCAP episode | `input/mcap_demo.py` | Same |
| Goal / reference image | Quoted path in the instruction or an image argument, mime validated by `references.image_path` | An `ImagePart` passed through untouched, order preserved |
| Recorded numeric trajectory | `mode = "video+action"` on a rollout directory | Compressed `sample_rows` attached to each keyframe |
| Mixed JSON manifest | `input.json` with a `content` array of strings, `{image}` and `{video, mode}` entries (`input/manifest.py`) | The canonical form; CLI arguments are normalized into it by `input/request.py` |
| Pre-digested bundle | A directory containing `demo.json` | Replayed directly with **zero** model calls |

### 4.2 Stage table

| # | Stage | Code |
|---|---|---|
| 0 | Resolve CLI text, quoted media paths and manifests into typed content parts | `input/references.py`, `input/request.py`, `input/manifest.py` |
| 1 | Probe and decode bounded candidate frames | `input/video.py` (`FfmpegVideoExtractor`) |
| 2 | Per-window VLM keyframe selection (paper prompt P0a) | `harness/video_selector.py` (`CodexVideoSelector.select`) |
| 3 | Global review and merge across windows (paper prompt P0b) | `harness/video_selector.py` (`.review`) |
| 4 | Attach and compress the numeric trajectory | `input/recorded_demo.py`, `input/action_sampling.py` |
| 5 | Compile the `HISTORICAL DEMONSTRATION` block (paper prompt P2) | `input/demonstration.py` (`prepare_demonstration`) |
| 6 | Splice the result back into the run content, in place | `input/preparation.py` (`prepare_input_videos`) |
| 7 | Serialize to provider wire items and guard the size | `harness/input_content.py` |
| 8 | Content-addressed caching of every expensive stage | `input/video_cache.py` |
| 9 | Orchestration and `--prepare-only` | `main.py` (`_prepare_inputs`) |

### 4.3 Candidate extraction

`VideoProcessingConfig` in `input/video.py`:

```
target_fps        = 2.0     max_candidates   = 24      max_keyframes = 8
window_s          = 30.0    candidate_width  = 768     keyframe_width = 1280
jpeg_quality      = 3       max_duration_s   = 1800    max_file_bytes = 2 GiB
command_timeout_s = 120.0
```

Design points:

- Frame times come from `ffprobe` (real presentation timestamps), never from nominal FPS, so a
  variable-frame-rate phone recording still maps correctly onto the timeline.
- All requested frames are decoded in a **single** ffmpeg pass using a `select=eq(n\,i)+...`
  expression, which is far cheaper than one seek per frame.
- `event_times` injects additional candidates at `event +/- 0.3 s` around gripper transitions and
  motion stops, so the compiler does not rely on uniform sampling to catch the decisive instants.
- `_with_views` pairs frames from several cameras by shared capture clock within `0.1 s`, producing
  one keyframe *time* that may carry a top view and two wrist views.

### 4.4 Two-pass keyframe selection

`harness/video_selector.py` runs a short-lived, **tool-less** vision thread per window.

- Windows are cut with a shared boundary frame (`range(0, n - 1, size - 1)`), so nothing falls
  between two windows, and local indices are remapped back to global indices.
- The reply must match the strict schema `select_video_frames`:
  `selected[{index, reason, stage, left, right, result}]` plus a `summary`.
- `_validate_selection` enforces index bounds, rejects duplicates, requires a non-empty `reason`
  and sorts chronologically.
- `.review()` then merges all windows into one global story, normally 12 to 16 frames, and must
  retain the first and last frame of the whole sequence. The limit is computed as
  `min(24, 48 // views_per_frame)` so the total image count stays at or below 48.
- `cache_identity()` includes a sha256 of the prompt text itself, so editing the prompt invalidates
  the cache.

The prompts carry deliberate guard-rails that are worth copying verbatim in spirit: *gripper
closure alone does not prove a grasp*, *similar start and end poses do not imply no motion*, *do
not mislabel empty-gripper pressing as regrasping*, and *select only supplied candidate indices*.

### 4.5 Numeric trajectory digestion

`input/action_sampling.py`:

- `compress_samples()` keeps segment endpoints, roughly 1 Hz samples, every gripper-command
  transition, and every channel loss or recovery boundary.
- `gripper_events` detects band crossings (normalized `<= 0.1` and `>= 0.9`) plus a `0.08`
  magnitude change threshold.
- `event_times` adds moving-to-still detection (`ptp <= 0.002` over a `0.3 s` window).
- The result is documented in-source as *a sparse historical example, not an error-bounded control
  trajectory*. Full-precision samples are written separately to `actions.jsonl`.

### 4.6 The token-saving encoding

`_frame_context()` in `input/demonstration.py` is the actual compression trick:

- All samples share one flattened column header (`action_sample_encoding.columns`) produced by
  `_sample_fields`, and each sample becomes a positional row instead of a dict.
- A value identical to the previous row's value in the same motion column is replaced by `"="`.
- Motion columns are rounded to 3 decimals, everything else to 6 (`_context_numbers`), with an
  explicit comment that this is display precision only and never becomes a robot command.
- Measured geometry already visible at the keyframe is dropped from the rows.
- `null` means *absent*, never zero - and the prompt says so.

### 4.7 The `HISTORICAL DEMONSTRATION` preamble

`HISTORICAL` (shared), `HISTORICAL_VIDEO` and `HISTORICAL_ACTION` (mode specific) are prepended,
and an `END HISTORICAL DEMONSTRATION` sentinel closes the block. Their content, condensed:

- Images and actions describe a previous episode, not the current scene and not pending commands.
- The current user goal takes precedence; annotations are reference data, not new instructions.
- A requested action or a gripper closure alone does not prove execution, grasp or success;
  completing the demonstrated stage sequence does not prove today's task succeeded.
- For imitation, preserve contact side, object-to-gripper orientation, push or pull direction, arm
  roles and stage order; explain deviations with live evidence.
- After an IK rejection, first compare approach positions, heights and permitted axial rotations.
  **Do not tilt the grasp or replace the manipulation merely to make IK pass.**
- Historical coordinates require a verified frame mapping - the constraints describe relationships,
  not absolute coordinates and not blind trajectory replay.
- Video mode: only images and annotations exist; do not invent recorded numeric poses.
- Video plus action mode: numeric fields are planning references only, after checking source robot,
  base frame, TCP site and quaternion order; commanded poses are not measured poses; missing fields
  are unknown, not zero; do not stream old absolute joint commands.

### 4.8 Ordering, validation, publication

- `input/preparation.py` replaces each video part **in place**, so the surrounding text and image
  ordering the user wrote is preserved.
- `prepare_demonstration` validates 1 to 24 keyframes, at most 48 unique images, strictly increasing
  `t_s`, and sha256 of every image plus of the source video **before and after** processing.
- Output is published atomically by `os.rename` from a temporary directory and consists of
  `demo.json`, `input.json`, `provenance.json` and `actions.jsonl`.

### 4.9 Wire format and budget

`harness/input_content.py` converts parts into interleaved text and image items with `data:` URLs
and a per-image `detail` level. `validate_input_size` enforces `MAX_INPUT_CHARS = 1048576` with
`FIRST_TURN_RESERVE_CHARS = 65536` held back, and **fails loudly** with an actionable message
rather than silently truncating the demonstration.

### 4.10 Caching

`input/video_cache.py` keeps three layers - decoded media, per-window selection, and the global
review. The key is a sha256 over `{cache_format, video_sha256, instruction, label, extractor config
and events and end time, selector identity}`. Duplicate work is serialized with `fcntl.flock`,
decoded frames are shared by hard link instead of being re-decoded, and entries are published
atomically. Note for us: `fcntl` is Linux only.

## 5. The closed loop

`runtime/runner.py` `run_loop()`, per step:

1. Snapshot all cameras.
2. Read robot state, with `_state_with_recovery` handling stale or unhealthy feedback.
3. Build the observation via `harness/protocol.observation()`.
4. Record the step into the run trace.
5. Ask the agent for exactly one decision (`agent.decide(AgentTurn(...))`); the compiled run input
   is attached only at step 0.
6. If the tool is terminal (`done` / `give_up`), stop and `_return_home`.
7. Otherwise execute through `tools/runtime.ToolExecutor` and pass
   `compact_execution_result` back as the next `previous_result`.

Robustness details:

- `MODEL_RETRY_DELAYS_S = tuple(min(2.0 * 2**attempt, 8.0) for attempt in range(20))` with jitter,
  and `MODEL_RECOVERY_TIMEOUT_S = 300.0`. A robot action is never replayed as part of a retry.
- `TrajectoryIKError` becomes a `motion_not_executed` feedback item; nothing is sent on the bus and
  the loop continues so the model can replan.
- The decision budget ends the episode with `budget_exhausted`.
- `_trajectory_summary` keeps planned geometry and timing but strips internal `_trace` data.
- Motor temperature and feedback health are monitored; `_return_home` runs on every exit path.
- The code is explicit that a model-reported `done` is not verified physical success.

## 6. Observation and instruction protocol

`harness/protocol.py`:

- `instructions()` assembles the system prompt from calibration notes, `scene.safety_notes` (facts
  no camera can show), the rendered tool catalog, and the orientation, reachability, grasp
  sequencing, release verification and persistence policies. This corresponds to the paper's P1
  (controller contract), P3 (motion, grasp, release) and P4 (recovery and termination).
- `observation()` emits compact JSON: `{instruction, images, state, extra{env_step, ...},
  previous_result}`. `_model_numbers` rounds to 1e-6 and telemetry the model cannot act on
  (temperatures, settled bookkeeping) is stripped.

## 7. Tool interface

`configs/tools.json` is data, version 1, with `$schema` and `$template` resolution and
`parameters_by_mode` for single versus bimanual rigs. `tools/catalog.py` renders both the
human-readable catalog text injected into the system prompt and the strict JSON schema:
`_strict_parameters` makes optional fields nullable and sets `additionalProperties: false`, because
the provider requires every property to be listed as required. `tools/runtime.py` binds handler ids
(`robot.*`, `vision.locate_point`, `terminal.*`) to host callbacks through a fixed whitelist.

| Tool | Purpose |
|---|---|
| `move_to` | Absolute TCP pose target for one arm |
| `move_eef_chunk` | A short sequence of TCP waypoints |
| `set_gripper` | Normalized aperture command |
| `check_path` | Plan-only feasibility probe, nothing is executed |
| `locate_point` | Pixel to ray, and to a metric point when two views agree |
| `done` | Terminal - the model believes the task is complete |
| `give_up` | Terminal - the model abandons the task |

Every call carries a mandatory short `note` (the repo requires one or two brief Chinese sentences),
which doubles as a rationale log.

## 8. Motion, IK and timing

`motion/planner.py` `EefTrajectoryPlanner.plan()`:

- Model-supplied TCP targets are **preserved exactly**. The planner only densifies and retimes -
  its `path_policy` string is literally
  `model_tcp_targets_preserved_with_bounded_ik_residual_no_rejection_by_derivative_limits`.
- Each segment is sampled, solved and retimed, then `endpoint_hold_s` hold waypoints are appended,
  and the report includes `endpoint_fk_translation_error_m` and `endpoint_fk_rotation_error_rad`.

`motion/trajectory.py`:

- `sample_pose_segment` does linear interpolation in position and SLERP in orientation, with both
  endpoints pinned exactly (the paper's pose interpolation equation).
- `sample_count_for_segment` derives the sample count from segment geometry and nominal duration,
  not from how many waypoints the model happened to send.
- `retime_path_segment` applies Ruckig to the **scalar path coordinate `s`**, not per joint, so
  samples are never moved, removed or reordered. An analytic safety factor follows:
  `alpha0 = max(1, r_v, sqrt(r_a), cbrt(r_j))`, and when `alpha0 > 1` the whole segment is stretched
  by `alpha = 1.001 * alpha0`.

`motion/ik.py` `ContinuousIK`:

- Seeds with `solver.multi_trial_ik(target, seed, 5)`, then refines with damped least squares on a
  finite-difference Jacobian, characteristic length `0.1`, damping ramped near singularities and the
  step clipped to `0.05 rad`.
- A solution is accepted only inside the execution tolerances of `0.002 m` and about `1 degree`
  (numerical stopping tolerances are `1e-4 m` and `5e-4 rad`); otherwise `TrajectoryIKError` is
  raised and surfaced to the model as feedback.

Bimanual motions are synchronized by slowing the faster arm. There is **no collision checking
anywhere** - not between arms, not with the table, not with the scene.

## 9. Vision

`vision/perception.py` `PixelLocalizer.locate_point`:

- Undistorts the pixel with the stored intrinsics and returns a camera ray; with fixed top-camera
  extrinsics the ray is expressed in both arm base frames.
- If two wrist views see the same point, it triangulates, gated by
  `min_triangulation_angle_deg = 5`, `max_triangulation_condition = 500`,
  `max_triangulation_residual_m = 0.01` and positive depths.
- `metric_position_available` is the only flag the model should trust for a 3D position; otherwise
  it gets a ray and must reason about depth another way.

## 10. Agent harness, context management and cost

- `CodexAppServer` talks JSON-RPC over stdio to the provider app server, starts a thread, asks for a
  decision under an output schema, and consumes token-usage notifications.
- **Sliding live-image window** (`harness/providers/codex.py`): once the configured
  `live_image_window` of image groups is exceeded, or after a recovery, the thread is torn down and
  restarted, and the history is replayed as a text-only `HISTORICAL EXECUTION RECORD` containing the
  past observations, host feedback and decisions, closed by `END HISTORICAL EXECUTION RECORD`. Old
  live images are dropped; demonstration images are retained. This is the runtime twin of the
  offline demonstration digest.
- `harness/factory.py` and `harness/prompts.py` support Codex, Claude Code and Gemini CLI providers.
- `harness/usage.py` plus `prices.json` account cost per phase (selection, review, control).
- `harness/task_name.py` uses a cheap text-only model (`DEFAULT_TASK_NAME_MODEL = "gpt-5.6-luna"`)
  to name the run and to reuse an equivalent previously saved `request_json`.
- `harness/task_prompts.py` intentionally ships **no** private task profiles in the public repo -
  `control_prompt_profile()` returns `None`, and applications are expected to add their own policy
  layer. There is no skills system.

## 11. Recording and reproducibility

`recording/trace.py` writes an append-only run directory: `events.jsonl`, `frames/`, `config.json`,
`protocol.json`, `transcript.json`, `status.json`, `recording.json`. Because that layout is exactly
what `input/recorded_demo.py` consumes, **a finished run can become the next demonstration**.
`--prepare-only` produces the portable digested input without touching a robot.

## 12. Configuration

Layered JSON with `extends`, selected by `--machine` or `--config`; agent profiles under
`configs/agents/`; `tool_catalog` may be a path or an inline object; `scene.safety_notes` injects
facts no camera can show; calibration matrices live in the machine profile; `gpt-policy --check`
runs a preflight over environment, config, provider and hardware.

## 13. What GPT-Policy does not do

No fine-tuning and no action head. No video embeddings, no optical flow, no learned keyframe
detector - keyframe choice is itself a VLM call. No dense trajectory replay. No collision checking
and no sampling-based motion planner; only straight-line plus SLERP interpolation with IK residual
gating. No separate reviewer or critic model at run time - verification is done by the same policy
VLM from fresh observations. No automatic success labelling.

## 14. Reusable takeaways for this repo

- **Never send raw video to the policy model.** Digest to text plus a few labelled keyframes. Our
  `providers/llms/message_media.py` can already carry both, but the digest is what keeps it cheap.
- **Two-pass select then review** is the cheap way to get a globally coherent story out of a long
  video without a long-context model.
- **Force a strict JSON schema and validate host side.** Bounds, duplicates, empty reasons and
  chronological order are all checked before anything reaches the next stage.
- **Write the guard-rails into the preamble**, especially *reference data, not commands* and
  *historical coordinates need a verified frame mapping* - our sim frame is not the demo frame.
- **Content-address the cache** on media hash plus instruction plus config plus prompt hash, so
  editing a prompt invalidates it. Use a Windows-portable lock instead of `fcntl`.
- **Fail loudly on prompt size** instead of truncating a demonstration in the middle.
- **Keep the model's targets and let the host densify and retime.** That maps well onto our
  trajectory generation, where the sub-task agent already emits sparse waypoints.
- **Feed rejection reasons back as normal feedback** so the agent replans instead of the run dying.
- The runtime `HISTORICAL EXECUTION RECORD` trick is directly applicable to our long planner and
  sub-task conversations.

## 15. References

Ruckig (jerk-limited online trajectory generation); ARX5 SDK; I2RT / YAM; RoboCurve
inspect-robots; the commercial VLMs compared in the ablations (GPT-6 Astra, Claude, Gemini).

- Project page: <https://cheng-haha.github.io/GPT-Policy/>
- Paper: <https://arxiv.org/abs/2609.19138>
- Code: <https://github.com/cheng-haha/GPT-Policy>
- Local checkout used for this document: `D:\NLP\Robotics\VLM_Robotics\GPT-Policy`
