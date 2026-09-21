# Show-Harness - Just a VLM Agent Can Play Robots

Reference notes for `D:\NLP\Robotics\VLM_Robotics\Show-Harness`
(Show Lab @ NUS, paper arXiv:2609.10522). Every code reference names a real file and symbol in
that clone, so the document stays greppable.

---

## Summary (copy-paste block)

```
* Show-Harness - Just a VLM Agent Can Play Robots
  ) Goal: TLDR - one or two live RGB views (front plus eye-in-hand wrist) plus a compact text
    prompt --> a VLM picks ONE discrete incremental action unit per step from a fixed vocabulary
    (MV_LEFT, MV_RIGHT, MV_FWD, MV_BACK, MV_UP, MV_DOWN, ROTATE_CW/CCW, GRASP, RELEASE, DONE)
    --> an embodiment-specific interpreter deterministically grounds that unit into a Cartesian
    setpoint increment (2 cm, 0.15 rad by default) --> robot moves, cameras are re-read, loop.
    Two modes share one interface: a frontier VLM zero-shot, or a LoRA fine-tuned small VLM
    emitting a single bare action token per step.
  ) Problem: VLAs need huge robot datasets, do not transfer across embodiments and cannot adapt at
    test time, while direct end-effector pose regression from a VLM is brittle. Put the physical
    grounding in a deterministic interpreter and leave only semantics to the model.
  ) Tags: Zeroshot, Fewshot, Finetune-VLM, VLM, TextPrompt, ImagePrompt, KeyPoints, HumanVideos,
    Planning, PrimitivePolicies, Closed-Loop, Long Horizon, Imitation Learning, Sim2Real,
    Real2Sim-Generate (real rollouts are re-tokenized into sim datasets), Tracking (not used)
  ) Project: https://showlab.github.io/Show-Harness/
  ) Paper: https://arxiv.org/pdf/2609.10522
  ) Video: demo reel embedded in the GitHub README and the project page gallery
  ) Method:
    ) Data Collection: GUMI - a browser teleoperation UI that maps every action unit to a key or
      button, so a human (or a GUI-driving agent) demonstrates by playing the robot; every step is
      recorded directly as a training-ready observation/action pair, with no post-processing and no
      teleoperation hardware. The same key bindings power live human takeover during autonomous
      rollouts. scripts/trajectory/real2sim/ additionally records sim demos and tokenizes
      continuous trajectories into the same discrete units.
    ) Data Augmentation: not a visual augmentation pipeline. The released corpus mixes real
      Franka/Piper rollouts with RoboLab and ManiSkill sim rollouts, and one released adapter
      (qwen3_5_2b_sim) is trained purely in simulation and evaluated on the real robot.
    ) Preprocessing: observations are resized and pad-to-square (core/franka/camera_utils.py
      resize_with_pad); the controller sees camera_resolution 256 px images, while the planner and
      the affordance tools get hd_res 640 px frames. Images are embedded as OpenAI-style data URLs
      by core/record/images.py image_to_data_url.
    ) Training: LoRA via LLaMA-Factory on rollouts converted to an alpaca-style chat dataset
      (train/data_preparation/rollouts_to_alpaca.py); under a few H200 GPU-hours. Released adapters
      cover Qwen3.5 at 0.8B/2B/4B/9B and Gemma 4 E4B, plus one sim policy.
      ) Regularization: LoRA itself; no other scheme is documented in the repo.
    ) Inference: stateless per step. Each decision is a NEW single user message containing the
      current image(s) plus the rendered controller prompt - there is no accumulated chat history
      and no past images in the conversation. All memory is verbalized into the prompt text.
      ) Test time techniques: subgoal planning into visually checkable stages; a draw-and-verify
        affordance loop that renders a coloured dot on the image and asks a verifier VLM whether
        the dot sits exactly on the named part; a WRIST: YES/NO self-report that switches the
        guiding view and the step size; action chunking (plan up to 3 moves open loop while the
        target is far); anti-oscillation rules from a 3 to 5 move history; empty-grasp detection
        with reopen and plan rollback; guided JSON or guided_choice decoding, with a malformed
        answer recovered by regex, then a strict token-only retry, then an MV_DOWN fallback.
      ) Tools: there is no tool-calling API - the action vocabulary IS the interface. The
        nine paper plugins are host-side capabilities: Multi-View Guidance, Proprioception,
        Subtask Planning, Situated Planning, Action Chunking, Adaptive Step, Visual Prompt,
        Action History, Failure Recovery; plus unlisted rotation, smooth, dagger, video_ref,
        ego, wrist_frame, coords, mcq and action_ablation.
      ) Close-loop: yes, and very tight - one model call per 2 cm of motion. Every step re-reads
        the cameras, re-verbalizes proprioception, and re-decides. Failure recovery intercepts
        before and after each decision; auto_release reopens any grasp measuring below 1 mm.
    ) Usage of lower level models: none. No detector, no segmentation, no grasp predictor, no
      depth or point cloud, no motion planner and no collision checking. Grounding is a lookup
      from a token to a unit vector in configs/primitives_<embodiment>.yaml.
  ) Results:
    ) Both modes beat all three baseline families - VLA (pi0.5, GR00T), VLA-centric agents (H-VLA,
      G-VLA) and code-as-policy (CaP-X, RATS); best baseline was 57% cross-task and 52% on the
      cross-environment/cross-embodiment split.
    ) Zero-shot clears every environment shift (background, lighting, viewpoint, distractors) at
      20/20, including tasks held out from fine-tuning.
    ) The fine-tuned model trained ONLY in simulation transfers to the real robot at 13/20, where
      both trainable VLA baselines score 0/20.
    ) Adaptation is an interpreter change, not a retrain - a 1 cm step lifts zero-shot 60 to 80
      percent and fine-tuned 40 to 65 percent with no retraining; rotation units extrapolate to an
      unseen 90 degrees (10/10 vs 20% for pi0.5); joint dual-arm prediction lifts the average from
      50 to 80 percent and removes every collision.
    ) One demonstration video takes demonstrated-order following from 20% to 20/20.
    ) Backbone scaling is flat from 2B to 9B (88-92%), but on a rolling tennis ball 2B beats 9B
      (80 vs 60 percent) because 39 ms per decision tracks a moving target and 79 ms does not.
    ) Plugin ablations from a 96% configuration: Multi-View Guidance -38, Subtask Planning -36,
      Proprioception -28, Failure Recovery -24, Action History -20 (and it cuts 23% of steps);
      Action Chunking holds 96% with 19% fewer VLM calls; Adaptive Step reaches 96% in 30 steps.
    ) Across five frontier VLMs, planning is near saturated (instruction following at or above 98
      percent, subtask plans at or above 97 percent) while empty grasps climb from 6 to 26 percent
      down the ranking - the gap is fine-grained spatial grounding, not language.
  ) Pro: stable and easy to reason about - the model never emits coordinates, so a wrong answer is
    a 2 cm error, not an unreachable pose or a singularity.
  ) Pro: embodiment agnostic - switching robots changes the interpreter and one YAML of unit
    vectors; the model-facing vocabulary and prompts are unchanged.
  ) Pro: cheap adaptation - physical changes (finer step, new rotation unit, joint dual-arm
    prediction) are interpreter edits; a small open model reaches frontier-level success after a
    few GPU-hours of LoRA.
  ) Pro: honest ablations - a disabled plugin is byte-identical to no plugin.
  ) Con: not flexible - a discrete axis-aligned vocabulary cannot express a diagonal, a curved
    approach or a continuous 6-DoF pose; tasks needing fine orientation rely on a single yaw unit.
  ) Con: one VLM call per 2 cm makes episodes 30 to 60 steps long; latency dominates on moving
    targets and cost dominates with frontier backbones.
  ) Con: fine-grained spatial grounding is the failure mode - empty grasps, centimetre errors.
  ) Con: no collision checking, no motion planning, no XY workspace bounds - only a per-command
    delta clamp and a Z floor.
  ) Dataset: https://huggingface.co/datasets/showlab/Show-Harness-Data - real Franka and Piper
    rollouts plus RoboLab and ManiSkill sim rollouts. A sample is one step: the agentview image,
    the optional wrist image, the rendered prompt context and the single ground-truth action
    token, recorded by GUMI as the human plays the robot through the same vocabulary the policy
    will later emit.
  ) Assets: configs/primitives_franka.yaml and primitives_piper.yaml (unit vectors and step
    magnitudes), prompts/controller.txt and common_context.txt, per-plugin <name>.txt prompt
    fragments, models/chat_templates/ jinja templates for serve-time rendering.
  ) Code: https://github.com/showlab/Show-Harness - 418 stars, 24 forks, Python, Apache-2.0,
    topics embodied-agent/foundation-models/harness/robotics, released 2026/09. Install with
    bash scripts/setup.sh base (add --real for the Franka/Piper hardware layer, serve for vLLM);
    preflight with python scripts/check_setup.py; run with python scripts/run_real.py. Docs are
    good - per-rig runbooks in docs/, plus plugins/README.md, configs/README.md, train/README.md,
    gumi/README.md, interpreters/README.md and a Chinese README.
  ) Models: five real-corpus LoRA adapters at https://huggingface.co/showlab/Show-Harness-VLMs -
    qwen3_5_0_8b, qwen3_5_2b, qwen3_5_4b, qwen3_5_9b, gemma4_e4b - plus qwen3_5_2b_sim covering
    both simulators. Real-robot configs default to the qwen3_5_2b_showharness_ft adapter served by
    vLLM. Zero-shot backbones evaluated: Gemini-3.1 Pro (the shipped default), GPT-5.6-sol,
    Opus 5, GPT-5.6-luna, Gemini-3.6-flash; InternVL3.5 2B/8B also fine-tuned.
  ) Hardware: Franka (Polymetis impedance control), AgileX Piper single and dual arm (joint
    streaming or endpose), RealSense cameras giving a front/agentview plus eye-in-hand wrist
    views. Teleoperation needs NO hardware - GUMI is a browser UI driven with WASD/arrow keys.
  ) Simulator: ManiSkill and Isaac Lab (plus RoboLab), and a synthetic tabletop --sim world inside
    GUMI for trying the interface before any hardware exists.
  ) References: LLaMA-Factory (training), vLLM (serving), Polymetis (Franka control), ManiSkill,
    Isaac Lab, AgileX Piper SDK, Qwen3.5 / Gemma 4 / InternVL3.5 backbones; baselines pi0.5,
    GR00T, H-VLA, G-VLA, CaP-X, RATS; companion survey Awesome-Multimodal-Embodied-Agent.
```

---

## 1. TL;DR

Show-Harness makes the action space semantic instead of numeric. The VLM never outputs a pose - it
outputs one word such as `MV_LEFT` or `GRASP`, and a deterministic per-embodiment interpreter turns
that word into a bounded Cartesian increment. Because the interface is a vocabulary rather than a
learned action head, the same prompts drive a frontier VLM zero-shot and a 2B model fine-tuned for a
few GPU-hours, across Franka, AgileX Piper, ManiSkill and Isaac Lab, and adapting to a new physical
demand means editing the interpreter rather than retraining the model.

## 2. The action vocabulary

`core/action_units.py` is the single source of truth, and the module docstring states the intent:
these tokens are the entire contract between the VLM and the robot, and every other module imports
them from here so the vocabulary cannot drift between prompts, runners and interpreters.

| Group | Tokens |
|---|---|
| `MOVE_ATOMS` | `MV_FWD`, `MV_BACK`, `MV_LEFT`, `MV_RIGHT`, `MV_UP`, `MV_DOWN` |
| `ROTATE_ATOMS` | `ROTATE_CW`, `ROTATE_CCW` (only when the rotation plugin is on) |
| Idle | `STOP` (sim controller holds the setpoint for one env step) |
| Gripper | `GRASP`, `RELEASE` |
| Terminal | `DONE` |
| Dual-arm | `STILL` (hold one arm while the other acts; also the human-override hold) |

Directions are defined **relative to the current reference view**; the metric realization lives only
in `configs/primitives_<embodiment>.yaml`. For Franka:

```yaml
step_m: 0.02            # 2 cm per MV_* unit
yaw_step_rad: 0.15
osc_translation_output_max_m: 0.05
osc_rotation_output_max_rad: 0.5
atomic_primitives:
  MV_FWD:  [ 1.0, 0.0, 0.0]
  MV_BACK: [-1.0, 0.0, 0.0]
  MV_LEFT: [ 0.0,-1.0, 0.0]   # the robot's LEFT is base -Y
  MV_RIGHT:[ 0.0, 1.0, 0.0]
  MV_UP:   [ 0.0, 0.0, 1.0]
  MV_DOWN: [ 0.0, 0.0,-1.0]
  ROTATE_CW: 1.0              # sign calibrated to the WRIST view, not the base frame
  ROTATE_CCW: -1.0
```

The comments in that file are themselves instructive: both the left/right convention and the
rotation sign were corrected against real rollouts, because the model judges direction from the
image and the eye-in-hand camera mirrors the base-frame right-hand rule.

## 3. The control loop

### 3.1 Zero-shot single arm - `core/runners/real.py` (`RealEpisodeRunner.run`)

Startup: `controller.step("RELEASE")`, then one observation, then - if the subgoal plugin is on -
a single planner call that receives the HD views (`agentview_hd`, `wrist_hd`) and returns an ordered
list of `Subgoal`s. Without the planner, `_single_task_subgoal()` fabricates one whole-task stage.

Then per step, up to `max_steps` (200 in `configs/robot_franka.yaml`):

1. Enforce the per-subgoal step cap.
2. `session.get_observation()` - fresh frames.
3. `_images()` extracts `agentview` and, when `use_wrist_image`, `wrist`.
4. `_premark_affordance()` may draw the affordance dot onto both images.
5. Build a `SkillContext` (task, current subgoal, indices, observation, images, proprioception).
6. Optionally resolve a `deepplan` pivot.
7. Pre-decision recovery hook.
8. Choose the action source, in priority order: **recovery override -> queued DAgger human intent
   -> queued action-chunk token -> a new VLM decision** (`controls.controller.decide(...)`).
9. Parse `response.token`; pull `target_in_wrist` and `chunk_plan` out of `response.payload`.
10. Execute: `STILL` does nothing, otherwise
    `controller.step(token, target_in_wrist=..., continuous=...)`.
11. Post-step recovery, possible forced `RELEASE`.
12. Update recovery notes, subgoal completion and move history; record; display.
13. Stop on `DONE`, on all subgoals finishing, on a stage cap, or at `max_steps`.

**Statelessness is the key design decision.** Every request is a brand-new single user message with
the current image(s) and the rendered prompt. There is no chat history and no past images in the
conversation - all memory is verbalized as text (`recent moves`, `previous_direction`,
proprioception, recovery notes). `RECENT_MOVES_MAX` is 3 in `real.py` and 5 in `mvtoken.py`.

Images per call: one (agentview) or two (agentview + wrist) for single arm; three for dual
(front, left wrist, right wrist). Controller frames are `camera_resolution` 256 px, square-padded by
`core/franka/camera_utils.py:resize_with_pad`; the planner and affordance roles use `hd_res` 640 px.

### 3.2 Fine-tuned single arm - `core/runners/mvtoken.py` (`MvTokenRunner.run`)

Same skeleton, far fewer moving parts: no planner, no subgoals, no stage control. Per step it
captures, calls `MvTokenController.decide()`, receives **one bare action token**, executes it unless
it is `DONE`/`STILL`, optionally auto-releases an empty grasp, records `t_obs_ms` / `t_decide_ms` /
`t_exec_ms`, and sleeps out the remainder of `loop_period_s`. Gripper state comes from the
*commanded* value rather than the measured one, matching how the training data was recorded.

### 3.3 Dual arm

`core/runners/dual.py` and `dual_mvtoken.py`. `core/vlm/dual_roles.py:DualControllerAgent` asks for
**one structured response containing both arms' decisions** (`_dual_decision_schema`), which is what
the paper reports as joint dual-arm prediction - it lifted the average from 50 to 80 percent and
eliminated the collisions that per-arm control produced.

## 4. The decision contract

Defined in `core/vlm/roles.py` (`_default_output_contract`, `_decision_schema`, `ControllerAgent.decide`)
and injected through the `{output_contract}` placeholder. The default answer is JSON:

```json
{"decision": "MV_LEFT", "reasoning": "one visual sentence"}
```

with `decision` constrained by a JSON-schema `enum` over the allowed tokens. Degradation ladder when
the model misbehaves:

1. `complete_json` with `guided_json` (vLLM) or `response_format: json_object` (hosted).
2. `_recover_decision_from_malformed_json()` - regex an allowed token out of the raw text.
3. `_token_retry_or_fallback()` - a strict token-only retry via `complete_token`, which on vLLM uses
   `guided_choice` and on a second failure re-asks at `temperature 0.0` with thinking disabled.
4. Commit a fallback: `previous_direction` if it was a direction token, else `MV_DOWN`.

In chain-of-thought mode (`reasoning_cot: true` in the backend profile) `_complete_cot_decision()`
uses `complete_text` and recovers the token from prose - deliberately *not* guided JSON.

The fine-tuned path is stricter and simpler: `complete_action_token()` returns one bare token parsed
by `_parse_single_token()`, no JSON, no reasoning, thinking forced off, and a `RuntimeError` bubbles
up to the runner's explicit `FALLBACK_TOKEN = "MV_DOWN"`.

## 5. The VLM client

`core/vlm/vlm_client.py:VLMClient` speaks one OpenAI-compatible dialect with three variants -
`vllm`, `openai`, `gemini`. `_finalize_payload()` strips vLLM-only fields (`chat_template_kwargs`,
`guided_choice`, `logprobs`) for hosted providers and translates `guided_json` into
`response_format: {"type": "json_object"}`; OpenAI wants `max_completion_tokens`, Gemini wants
`max_tokens`.

Message layout is always **images first, then the prompt text** - image A agentview, image B wrist,
then text - and images are inlined as data URLs by `core/record/images.py:image_to_data_url`.

Methods: `complete_json` (schema-guided), `complete_text` (free prose, CoT), `complete_token`
(`guided_choice`, budget capped at `max(8, min(max_tokens, 24))`), `complete_action_token` (the
fine-tuned single token), plus `complete_action_token_pair` / `complete_action_token_chain` for dual
and chunked policies.

Retries: `max_retries` 5 by default (10 in the shipped Gemini and ChatGPT profiles), exponential
backoff from 2.0 s to 30.0 s with 0.8-1.2 jitter, honouring `Retry-After` hints up to 90 s, on
408/409/425/429, any 5xx, network errors, malformed bodies and empty choices. A 401/403 triggers one
API-key refresh and a single retry.

**There is no client-side response cache.** `complete_action_token_chain` merely keeps a common
prompt prefix so the *server's* prefix cache can help.

## 6. Prompts and prompt assembly

`core/prompting/prompt_loader.py:load_prompt_dir` requires exactly two files - `common_context.txt`
and `controller.txt` - loads optional per-backend `controller_<backend>.txt` variants, and
recursively loads an optional `prompts/skills/` tree. Notably, plugin-owned prompts are *not* loaded
here: `subgoal_planner.txt`, `affordance_*.txt` and `deepplan_*.txt` belong to their plugins.

`prompts/controller.txt` is a small, dense template:

```
TASK: {task}
STAGE: {stage}
TARGET: {target}
AFFORD: {affordance}
Stage goal: {description}
DONE WHEN: {completion}
Gripper now: {gripper_state}
{mem_text}
{recovery}
{proprio}

DIRECTION:
Is STAGE for grasping AND TARGET inside the wrist view?
A) YES -> wrist is the primary guide. ...
B) NO  -> AgentView is the primary guide. ...
C) MV_UP when: ...
...
Think one visual sentence, then commit.
{output_contract}
```

Three mechanics are worth copying:

- **Substitution is `str.replace`, never `str.format`.** Prompt fragments contain literal JSON
  braces such as `{"decision":"LETTER",...}`, which `str.format` would try to evaluate.
- **Plugin prompt text lives beside the plugin** in `<name>.txt`, split into `[section]` blocks
  parsed by `plugins/prompt_text.py:fragment()` (`_HEADER = ^\[([A-Za-z0-9_.-]+)\]\s*$`, cached with
  `lru_cache`), so an operator can edit the exact sentences without reading Python.
- **Build-time transforms rewrite the template in memory** before the episode, in the fixed order
  `coords -> wrist_frame -> ego -> action_ablation`. The source files are never modified.

Placeholder ownership: `{mem_text}` and `{mem_text_rules}` from `mem_text`, `{recovery}` from
`recovery`, `{proprio}` from `proprioception`, `{rotation}` from `rotation`, `{variable_step}` from
`variable_step`, `{action_chunk}` from `action_chunk`, `{affordance}` rewritten by `affordance`,
`{stage}/{target}/{description}/{completion}` from the current subgoal, `{output_contract}` from the
active answer-protocol plugin.

## 7. The plugin system

### 7.1 Contract

Three hook surfaces (`plugins/README.md`, `plugins/__init__.py`):

1. **Build-time prompt transforms** - `apply(prompt_template) -> str`.
2. **Per-step context providers** - render text for named placeholders; answer-protocol plugins may
   own the whole contract via `answer_tokens` / `output_contract()` / `fallback_answer()` /
   `map_response()`.
3. **Execution interceptors** - `recovery.before_decision` / `after_step`, `deepplan.is_pivot` /
   `resolve`, DAgger preemption, affordance annotation, and interpreter-side hooks from
   `variable_step`, `smooth` and `rotation`.

**Disabled means byte-identical.** A disabled plugin still constructs, but every hook returns its
inert value - `""`, `[]`, `None`, or the identity transform - so the loop with the plugin off is
exactly the loop without it. That is what makes single-plugin ablations meaningful.
`PluginsConfig.enabled(name, default=True)` resolves a missing key to the caller's default, a bool
directly, and a mapping to enabled-unless-`enabled: false`.

Plugins that call the model take a **duck-typed `client`** exposing `complete_json` /
`complete_text` / `complete_token` - never a concrete class. `plugins/assembly.py` centralizes only
the mode-independent constructors (`build_dagger_plugin`, `build_video_ref_plugin`,
`build_auto_release_plugin`); each runner still chooses its own set. There is no registry.

### 7.2 The nine paper plugins

| Paper name | Code | Ablation delta |
|---|---|---|
| Multi-View Guidance | view-role prompt scaffolding + `core/prompting/wrist_marker.py`, `plugins/view_select` on the dual rig | -38 |
| Subtask Planning | `plugins/subgoal` | -36 |
| Proprioception | `plugins/proprioception` | -28 |
| Failure Recovery | `plugins/recovery` (+ `plugins/auto_release`) | -24 |
| Action History | `plugins/mem_text` | -20 (but +23% steps) |
| Action Chunking | `plugins/action_chunk` | 96% at 19% fewer calls |
| Adaptive Step | `plugins/variable_step` | 96% in 30 steps |
| Visual Prompt | `plugins/affordance` | off by default; 35% -> 85% on its case |
| Situated Planning | `plugins/deepplan` | off by default; 35% -> 85% on its case |

**`subgoal`** - one planning VLM call turns task plus image into an ordered plan. Schema:
`{subgoals: [{id, target, affordance, motion, description, completion}]}`. The discipline is that
every `completion` must be judgeable from raw 2D images - a movement stage ends in a *visible*
spatial relation, never an invisible gripper event. Segmentation rules merge
approach/align/lower/close into one `GRASP`, split the lift clear into `LIFT`, and require a
`RETREAT` after every `RELEASE`.

**`affordance`** - the draw-and-verify loop. `AffordancePointerAgent.locate` asks for
`point = [y, x]` on a **0-1000 grid** with rules such as "use a rim for containers", "choose an end
rather than the middle of long objects", and permission to return an empty list when the part is not
visible. The candidate is then **drawn back onto the image as a coloured dot** (red for front/left
arm, blue for right) and a verifier is asked to "judge STRICTLY whether the dot's center sits exactly
on {part}", returning `{present, on_target, why, point}`; a wrong dot yields a corrected point for
the next round (`verify_rounds` default 1). The plugin then rewrites the controller's `AFFORD` line
into e.g. `handle = RED dot in front view` - so the model steers toward a dot it can see rather than
toward a noun. A second reference image can carry the already-committed dot for cross-view
disambiguation.

**`deepplan`** - conditional tasks. The planner must emit REVEAL stages, exactly one
`"motion": "REASON"` pivot whose `description` is a complete set of visible `IF ... THEN ...` cases,
and a `PLACEHOLDER` goal. `is_pivot()` fires on the `REASON` stage, and `resolve()` runs a resolver
call with `{task, branch_rule, observe_condition, remaining_goal}` plus live images, splicing
concrete subgoals into the plan. A "look further" branch must re-emit the same `REASON` so the pivot
is not lost.

**`mem_text`** - the whole plugin is 39 lines and injects
`Recent moves, newest first: {moves}` plus three rules: do not re-`GRASP` in place after an empty
close, **never choose the opposite of the newest move** (`MV_LEFT/MV_RIGHT`, `MV_FWD/MV_BACK`), and
when opposite directions already appear, prefer `MV_DOWN`/`MV_UP`. `max_recent` default 3.

**`proprioception`** - converts numbers to sentences in centimetres: gripper height above the table,
the fine and coarse step magnitudes, a `If height > {high_cm} cm, MV_DOWN first` hint, gripper width,
and a contact signal derived from a stalled descent -
`Last MV_DOWN lowered {moved_cm} of {commanded_cm} cm -> already in contact, do NOT MV_DOWN again`.

**`variable_step`** - keyed off the shared marker in `core/prompting/wrist_marker.txt`:
`WRIST CHECK: begin your reasoning with 'WRIST: YES' if the TARGET is visible in the wrist view,
else 'WRIST: NO'`. `parse_wrist_marker` takes the **last** marker in the response and stores
`payload["target_in_wrist"]`. `WRIST: NO` (or a lift) selects `coarse_step_m` 0.04; `WRIST: YES`
returns to `fine_step_m` 0.02.

**`action_chunk`** - `step_num` default 3. Only when `target_in_wrist is False` may the model answer
`PLAN: M1, M2, ...`; the first token executes and the rest queue up and run open-loop with
`continuous=True`, without further VLM calls. When `WRIST: YES` it must decide a single move.

**`recovery` + `auto_release`** - an empty grasp is detected from measured gripper width.
`AutoReleasePlugin.should_release(width_m, gripper_closed)` is simply
`enabled and gripper_closed and width_m < empty_width_m` (1 mm), checked after *every* step so a
later slip also reopens. `recovery` adds prompt notes ("Empty close; do not retry on an edge/corner.
Recenter body and confirm depth.") and, crucially, `before_decision` can **roll the plan back to the
`GRASP` stage** rather than merely releasing.

### 7.3 The unlisted plugins

`rotation` (adds `ROTATE_CW/CCW`, default 30 deg per command, and rotates wrist-judged `MV_*` by the
accumulated yaw - force-disabled on Piper); `smooth` (min-jerk setpoint ramp, `settle_steps` 4,
`settle_dt_s` 0.05); `dagger` (live human override - `core/runners/preemption.py:InterruptibleDecider`
runs `decide()` in a worker thread, polls at 0.05 s, and **abandons an in-flight VLM answer** when
human input arrives); `video_ref` (see below); `ego` and `wrist_frame` (per-rig direction-frame
rewrites that anchor on exact controller sentences); `coords` (replaces camera-edge wording with
right-handed base axes, `MV_FWD -> +x` and so on); `mcq` (letter-choice answer protocol);
`action_ablation` (modes `off|bare|letters|letters_blind`, rewriting other plugins' sentences by
regex - in blind mode the six directions become opaque `ACT_A..ACT_F` with the mapping withheld, the
pre-action frame is attached on the next step, and the model must emit
`NOTE[{symbol}]: <what it did>` so the harness can learn the mapping).

## 8. Multi-modal prompt mechanisms

This is the section most relevant to our own work. Two lists.

### 8.1 Non-text modalities that reach a model

1. Live front/agentview RGB - controller, planner, affordance pointer, deepplan resolver.
2. Live wrist RGB (eye-in-hand) - controller, affordance wrist tracking, the WRIST judgment.
3. **Drawn overlays** - the affordance dot rendered onto the image and sent to a verifier.
4. **Cross-view reference pair** - image A clean plus image B carrying the committed dot.
5. Dual-arm multi-view sets - front, left wrist, right wrist, explicitly labelled
   `VIEW: Front View | LEFT Wrist | RIGHT Wrist`.
6. **Before/after frame pairs** - blind action ablation attaches the pre-action frame next to the
   current one so the model can infer what an opaque symbol did.
7. **Demonstration video frames** - the `video_ref` plugin.

### 8.2 `video_ref` - in-context learning from one demo video

`plugins/video_ref/plugin.py`. This is the cheapest useful design in either repo and the direct
counterpart to GPT-Policy's context compiler.

- `sample_frames()` takes **8 frames** (`DEFAULT_NUM_FRAMES`) uniformly across the clip, first and
  last always included, in one sequential decode pass with no random seeks, falling back to a
  bounded 1-of-N stride (about 4 fps, capped at 2400 buffered frames) when the container reports no
  frame count. Frames are thumbnailed to **512 px** (`DEFAULT_MAX_SIDE`).
- `extract_brief()` makes **exactly one** VLM call, guided by `_brief_schema`, returning
  `{task, operations: [{arm, action, object, grasp, destination}]}`; it retries once with free JSON,
  then raises rather than silently degrading into ordinary planning.
- The analyst prompt (`video_ref_single.txt`) is explicit about what matters for replication: name
  the object so it cannot be confused, report **the exact part grasped** (stem, rim, edge, handle),
  and the destination **with placement nuance**; report only what the frames show.
- `render_prompt()` emits a `### REFERENCE DEMO` block into the planner prompt's `{video_ref}` slot
  with numbered operations, and the guard-rail sentence that matters:
  *"Object positions may differ from the demo - plan from where things are NOW."* In dual mode the
  numbering is the demo's time order **across both arms**, and a later operation must wait for the
  visible completion of an earlier one on the other arm.
- Deliberate scope, quoted from the module docstring: **the brief is TEXT and the demo frames are
  not re-sent with the planner call**; replication is plan-level (order, arm, grasp part,
  destination), not trajectory-level; extraction runs **once per run** and then feeds every replan.
- The split-role rationale is worth stealing verbatim: two calls, not one, so the brief is a
  loggable, operator-checkable artifact and the planner never divides attention between N demo
  frames and the live scene.

### 8.3 Non-text signals converted to text

Subgoal plan fields; the affordance point becoming `handle = RED dot in front view`; the wrist
judgment becoming `WRIST: YES/NO`; proprioception in centimetres; move history; recovery notes; the
deepplan branch rule; the selected view; robot-axis wording from `coords`; and the learned
`NOTE[ACT_X]` mapping in blind ablation. Human teleoperation input is the one signal that never
reaches the model - it preempts the decision at execution time instead.

## 9. Interpreters - grounding

`interpreters/real_atomic_controller.py:RealAtomicController` keeps a commanded target pose
(`_target_pos`, `_target_euler`), initialized by `sync_from_robot()` from `robot.get_ee_pose()`.
`intended_motion()` maps a token to `delta_position = move_vector * step_m` and
`delta_yaw = yaw_sign * yaw_step_rad`; a non-motion token raises `ValueError`.

`_apply_motion()` clamps each command to `max_position_delta_m` / `max_rotation_delta_rad` and
enforces the **Z floor** - a target below `z_floor_m` is snapped back to the floor and the descent is
blocked. `resolve_z_floor()` supports `--no-z-floor`, `--z-floor-m X`, `--z-floor` (capture the
current height), or the config value; `configs/robot_franka.yaml` sets `enable_z_floor: true`, and
Piper refuses to capture a floor automatically on real hardware.

Gripper handling: `GRASP` closes, waits for settling (`gripper_settle_s` 1.2 s on real runs,
`gripper_poll_dt_s` 0.05), reads the measured width and updates state; if `grasp_min_width_m` is
configured and the measured width is at or below it, the close is flagged `grasp_empty` and the
gripper reopens. `RELEASE` opens to `grasp_open_width_m` 0.06.

Embodiments: Franka goes through Polymetis impedance (`start_impedance: true`, with
`ensure_controller` able to restart impedance once after a recoverable setpoint failure); Piper uses
`motion_backend: joint_stream` (or `endpose`) with on-board IK. ManiSkill, Isaac Lab and RoboLab have
their own atomic controllers.

**There is no collision checking and no geometric motion planning** - only clamped Cartesian setpoint
increments, an optional min-jerk ramp, and the Z floor.

## 10. GUMI - data collection

`gumi/collect_rollouts_web.py` (and `_dual`) serve a browser UI on port 8600. Every action unit is
bound to a key or button, so a human drives the robot **through the same vocabulary the policy will
later emit** - which is why each step is already a training-ready `(observation, action)` pair with
no post-processing and no retargeting. `--sim` gives a synthetic tabletop world for trying the
interface with no hardware. `gumi/gpt_operator/` lets a GUI-driving agent play the robot instead of a
human, and the same bindings provide live human takeover during autonomous rollouts (the `dagger`
path). `scripts/trajectory/real2sim/` records sim demos and tokenizes continuous oracle trajectories
into the same units (`atomic_tokenizer.py`, `follow_tokenize.py`, `make_dataset.py`).

## 11. Training

`train/` is self-contained and builds its own venvs against upstream LLaMA-Factory.
`train/data_preparation/rollouts_to_alpaca.py` converts rollouts, `register_dataset.py` registers
them, and `train/configs/*.yaml` hold the LoRA recipes (`qwen3_5_2b_lora.yaml`,
`internvl3_5_2b_lora.yaml`, `gemma4_e4b_lora.yaml`, `qwen3_5_2b_sim.yaml`).
`generate_affordance.py` and `generate_subgoals.py` synthesize the plugin-side supervision.

One sharp operational warning from `models/README.md`: LLaMA-Factory renders the conversation itself
during training and never reads a jinja chat template, so `models/chat_templates/` exists only to
make vLLM reproduce that rendering at serve time. A base model's own template does not match, and
**the mismatch fails silently**.

## 12. Configuration

Precedence (`core/launch.py:build_config`, starting from `deep_merge(DEFAULTS, robot_cfg)`):
code `DEFAULTS` -> `defaults:` files merged *under* the robot config -> the robot config ->
`overlays:` merged *over* it -> CLI overrides -> the resolved VLM backend profile. Site identity
(robot IP, camera serials) is expected in `configs/site/franka.yaml`, which is gitignored.

Shipped `configs/robot_franka.yaml` highlights: `max_steps: 200`, `high_above_table_m: 0.08`,
`coarse_step_m: 0.04`, `fine_step_m: 0.02`, `up_step_m: 0.04`, `rotate_step_deg: 40`,
`action_chunk_step_num: 3`, `mem_text_len: 5`, `hd_res: 640`, `video_ref_frames: 8`,
`planner_max_tokens: 6144`, `empty_width_m: 0.001`, `open_width_m: 0.07`, `enable_z_floor: true`.
Plugin defaults: `subgoal`, `proprioception`, `recovery`, `variable_step`, `action_chunk`,
`mem_text`, `smooth`, `dagger` **on**; `coords`, `mcq`, `deepplan`, `rotation`, `video_ref`,
`affordance` **off**.

`execution_token_swap` deserves a mention: `core/launch.py:install_execution_token_swap` exchanges
tokens pairwise **after** the model answers and **before** the interpreter executes, to absorb
checkpoints whose learned convention swaps e.g. `MV_FWD` and `MV_BACK` between rigs. The prompt
vocabulary and the primitive vectors are untouched.

## 13. Timing and cost

`MvTokenRunner` records `t_obs_ms`, `t_decide_ms`, `t_exec_ms` per step; the zero-shot runner tracks
`_last_vlm_ms`. `EpisodeLogger.log_step` writes `images/agentview/`, `images/wrist/`, full records in
`steps.json` and reasoning-truncated records in `steps.jsonl`. Both loops sleep out `loop_period_s`
after execution - it is a pacing target, not a hard deadline, and VLM latency counts against it. The
only real execution overlap is action-chunk streaming; DAgger preemption abandons a pending answer
but does not overlap robot motion with model thinking.

## 14. Show-Harness vs GPT-Policy

| Dimension | Show-Harness | GPT-Policy |
|---|---|---|
| Action space | Discrete semantic units (`MV_LEFT`, `GRASP`) | Continuous TCP poses via 7 JSON tools |
| Granularity | ~2 cm per model call | A whole reach/segment per call |
| Grounding | YAML unit-vector lookup + clamps + Z floor | SLERP densification, DLS IK, Ruckig retiming |
| Training | Zero-shot **or** LoRA fine-tuned 2B | Zero-shot only, parameters always fixed |
| Conversation | Stateless; memory verbalized per step | Persistent thread with a sliding live-image window |
| Demo digestion | 8 uniform frames -> one analyst call -> text brief, once per run | ffmpeg candidates -> two-pass VLM select+review -> keyframe images + text, cached |
| Demo images at control time | No - text brief only | Yes - keyframes stay in context |
| Planner | `subgoal` plugin produces visually checkable stages | No planner; the policy plans implicitly |
| Verification | Verifier VLM on a drawn dot; measured gripper width | The same policy VLM re-observing |
| Collision checking | None | None |
| Extensibility | Byte-identical plugin ablations | JSON tool catalog + machine profiles |

They are complementary: Show-Harness has the better *interface* and the cheaper demo digest,
GPT-Policy has the better *controller* and the more rigorous media pipeline.

## 15. Reusable takeaways for this repo

- **A text brief beats re-sending frames.** `video_ref` gets the headline in-context result
  (20% -> 20/20) from 8 uniform frames, one VLM call and a text block. That is a one-day feature,
  not a pipeline.
- **Two calls, not one.** Keeping the analyst separate from the planner makes the digest a loggable
  artifact and stops the planner dividing attention between demo frames and the live scene.
- **Say "positions may differ from the demo - plan from where things are NOW."** That single
  sentence is the whole frame-mapping guard-rail.
- **Verbalize state instead of growing the conversation.** Recent moves, proprioception in
  centimetres and recovery notes keep every call stateless and cheap.
- **Draw-and-verify.** Rendering a candidate point back onto the image and asking a second call
  "is the dot exactly on the handle?" is a direct upgrade for our `--affordance-points`.
- **Label the views.** Telling the model what each camera is for was worth 38 points.
- **Disabled means byte-identical.** Any `[INSERT DEMONSTRATION]` feature we add should leave the
  existing prompts unchanged when the flag is off - otherwise our ablations are meaningless.
- **`str.replace`, not `str.format`** - which our repo already does, and now we know why it matters.
- **Anti-oscillation rules** are a cheap fix for a failure mode our retry loop also shows.

## 16. References

LLaMA-Factory; vLLM; Polymetis; ManiSkill; Isaac Lab; AgileX Piper SDK; Qwen3.5, Gemma 4 and
InternVL3.5 backbones; baselines pi0.5, GR00T, H-VLA, G-VLA, CaP-X, RATS; the companion survey
Awesome-Multimodal-Embodied-Agent.

- Project page: <https://showlab.github.io/Show-Harness/>
- Paper: <https://arxiv.org/abs/2609.10522>
- Code: <https://github.com/showlab/Show-Harness>
- Models: <https://huggingface.co/showlab/Show-Harness-VLMs>
- Data: <https://huggingface.co/datasets/showlab/Show-Harness-Data>
- Local checkout used for this document: `D:\NLP\Robotics\VLM_Robotics\Show-Harness`
