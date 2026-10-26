# Ideas from other embodied agents

Features that helped three external LLM/VLM robot agents perform well, and that this repo does **not**
implement yet (or implements only partially). Ideas are ordered by measured impact, then by fit with our
pipeline (planner → code-gen subtask agent → VLM reviewer).

**Sources**

- **PyRUA-Lean.** arXiv:2610.01939; code https://github.com/DAGroup-PKU/PyRUA-Lean. Summarized in
  `docs/models/PyRUA-Lean.md`.
- **SenseNova-RoboRSI.** https://github.com/OpenSenseNova/SenseNova-RoboRSI (commit `b513a1c`) and the report
  *Has GPT-6-Astra Achieved Embodied Intelligence?* (sensenova.ai, 66 pp).
- **GPT-as-Policy.** *Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies*,
  arXiv:2609.38537; code https://github.com/anonymous-report-421/GPT-as-Policy.

**Sim2Real check.** Every idea below uses only pixels, calibration, proprioception and the task text. Where an
external agent relied on a privileged signal, that is flagged and the idea is reshaped to avoid it.

## Summary table

| # | Idea | Evidence of impact | Our status | Effort |
|---|---|---|---|---|
| 1 | Multi-primitive code cells with structured results (no exception-driven control flow) | PyRUA: +8.6 pp SR, −49% LLM calls, −65% tokens over tool calling | Partial: code-gen exists, but primitives raise and the namespace resets each attempt | M |
| 2 | Episode-persistent Python namespace | PyRUA: the namespace survives exceptions, so the next cell repairs and continues | Per attempt only (`task_state.py` `exec_env`) | S |
| 3 | Selective feedback (`show(cam)`, `display(img)`) instead of auto-attached images | Part of PyRUA's token savings; Coder learned to drop images from 19.3 to 0.8 per episode | Not implemented (head and wrist images every turn) | S–M |
| 4 | Feedback/verification sub-agent plus a structured per-step assessment | RoboRSI: largest single component, +9.2 pp SR (15.6 → 24.7). GPT-as-Policy gate schema | Partial: reviewer JSON `{success, reasoning, improvement_steps}` per attempt | M |
| 5 | History compression (last N turns plus a rolling summary) | RoboRSI: +5.6 pp SR **and** −66% tokens | Not implemented | S–M |
| 6 | Richer perception primitives exposed to code: multi-camera `locate`, `back_project`, `region_center`, `grasp_yaw` | PyRUA RoboDojo first place with SAM3 as the only learned perception; precision 4.0 → 37.3 vs direct control | Partial: `detect_object` uses the head camera only; back-projection and triangulation are internal | M |
| 7 | Orientation-aware motion primitives (`move_pose` pitch/yaw, `rotate_wrist`, `path="line"`, ≤0.3 m per call) | PyRUA: needed for recessed targets, singularities and side grasps | Not implemented (`[x, y, z, theta]` only) | M |
| 8 | Task-agnostic, robot-specific operating-knowledge doc | PyRUA guides: +6 pp SR on LIBERO-PRO for code (88 → 94) | Partial: task-scoped SKILL.md files | S |
| 9 | Self-judged success without a success flag (re-locate everything before `finish`, verify only after release) | PyRUA RoboDojo rules; RoboRSI's #1 limitation is weak success verification | Partial: VLM reviewer plus measured-evidence veto | S |
| 10 | Introspection-generated API reference | PyRUA: the prompt "cannot drift from the code" | Hand-written `prompts/main_prompt.py` | S |
| 11 | Episode budgets, call ledger and token/cost accounting plus a batch benchmark harness | Needed to measure 1–10 like the papers do | Partial: attempt counts, `motion_log` | M |
| 12 | Optional VLA as a primitive, or closed-loop VLA review (accept/edit/reject chunks) | RoboRSI: closed loop 23.6 vs VLA alone 6.9; with ICL 54.2 SR. GPT-as-Policy Hybrid 48% vs Direct 26% | Not implemented | L |
| 13 | Knowledge packages / experience notes refined between trials (ICL) | RoboRSI: SIMPLE 70.8 → 90.8; RoboDojo 25.8 → 54.2 with closed-loop VLA | Partial: offline `--learn-from-trajs` | M–L |
| 14 | Larger interaction budget plus explicit recovery | RoboRSI: 1× → 10× budget, RoboDojo 54 → 82, Bench2Dex 26 → 62 | Partial: 1–3 attempts per subtask | S |
| 15 | Active vision: the agent moves a camera or free arm to see occluded targets | RoboRSI real robot: the flower insertion fails on one arm and succeeds once the free arm repositions its camera | Not implemented (fixed shoulder camera) | M |
| 16 | Adaptive action chunk length chosen by the model | RoboRSI: adaptive 20.0 SR vs fixed L=4 11.4 vs baseline 15.6 | Not implemented | S–M |
| 17 | Code sandbox with privileged-access audit | PyRUA `sandbox_flags` make "reached for sim internals" visible | Plain `exec` | S |
| 18 | GRPO fine-tuning of a small coder on binary success | PyRUA Coder 4B: 18.8 → 44.1 unseen; beats the 27B base model | Not implemented | L |

## 1. Multi-primitive code cells with structured results

**What they do.** In PyRUA-Lean, one LLM call is one Python cell. A cell can run many primitives and check
their return values before going on. Each primitive returns a frozen dataclass such as `Move.reached`,
`Pick.success`, `Segment.found/score` or `Gripper.opening`. Only misuse raises an exception. Ending the
episode raises `EpisodeFinished(BaseException)`, so a bare `except Exception` cannot swallow it.

**Why it helps.** On the same planner, primitives and VLAs it scored 63.1% vs 71.7% and needed 17.0 vs 8.7
calls per solved episode. Of the 96 instances that only the code agent solved, the tool-calling baseline ran
out of call budget in 52 (arXiv §3.5, §3.8). The ablation that kept on-demand images but stayed with tool
calling *lost* success (83.0 → 68.5), so the gain comes from composition.

**Gap here.** Our subtask agent already writes code, but `api.py` failure paths raise
(`_handle_tracking_abort` → `RuntimeError`). The prompt also does not push "act → check result → branch"
inside one cell.

**Proposal.**

- Make `execute_trajectory`, `close_gripper`, `detect_object` and similar calls return result objects with
  `reached`, `final_pose`, `gripper_width` and `holding` fields, computed from proprioception only.
- Keep exceptions for misuse.
- Add prompt guidance and an example cell with in-code verification.

## 2. Episode-persistent namespace

**What they do.** Variables and helper functions survive across cells and across exceptions for the whole
episode. Only cross-episode state is reset.

**Gap here.** `task_state.py:29-41` documents a namespace lifetime of **one attempt**, and
`agent_runner.py:707-752` clears it.

**Proposal.**

- Add a flag such as `--namespace-scope {attempt,subtask,episode}`.
- Under `episode`, keep measured quantities (detections, grasp yaw, container centers) and helper functions.
- Log which names persisted, so stale state is easy to debug.

## 3. Selective feedback

**What they do.** PyRUA-Lean supports three image policies: `on-motion`, `on-demand` (only cameras requested
with `robo.show(cam)`) and `none`. A cell can also call `display(img)` to attach a crop or overlay. Text
feedback is stdout plus the value of a trailing expression, trimmed to 8k chars (head and tail kept). In
`rich` mode it also includes the last 40 ledger records.

GPT-as-Policy does something similar. It attaches a downsampled preview but keeps the full-resolution path,
so the model can crop and zoom itself.

**Gap here.** `agent_runner.py:903` `build_llm_context_images_and_pose` attaches head and wrist images every
turn.

**Proposal.**

- Add `show(camera)` and `display(img)` to `get_exec_locals`.
- Make an image policy flag; default to `on-motion` to keep current behaviour close.
- Expose a `crop(camera, box)` helper for fine inspection.

## 4. Feedback sub-agent and structured step assessment

**Evidence.**

- RoboRSI harness ablation on 42 RoboDojo tasks (Table 13): the feedback sub-agent gave 24.72 SR vs a 15.56
  baseline, the largest single gain, for only +7% tokens. An execution sub-agent by itself *hurt* (15.28).
- GPT-as-Policy forces a JSON `assessment` at every chunk:
  - `execution_status ∈ {not_started, progressing, failed, uncertain, recovered}`
  - `intent_status ∈ {aligned, misaligned, uncertain}`
  - `task_progress{verified_completed, currently_attempting, remaining}`
  - evidence fields
- Under that gate, a takeover is allowed **only** on `failed` or `misaligned`, never on uncertainty alone.
  The system prompt says: "A gripper command does not prove grasp, and an IK target does not prove arrival."

**Gap here.** Our reviewer gives one verdict per attempt.

**Proposal.**

- Extend `prompts/review_prompt.py` output with `execution_status`, `task_progress` and `evidence` fields.
- Let the planner keep a `verified_completed / remaining` ledger across subtasks. This addresses RoboRSI's
  observed failure of "losing track of other changes it has caused (fridge left open)".

## 5. History compression

RoboRSI keeps the last 4 turns plus a rolling summary of at most 2,000 tokens. That gave 21.1 SR vs 15.6 and
**−66% tokens** (2.56M → 0.86M). PyRUA's Coder evaluation also lost 19–31% of base-model episodes to context
overflow.

**Gap here.** `task_state.py` conversation lists grow without bound.

**Proposal.**

- Summarize older turns into a structured progress note and keep the last N raw turns.
- Log what was compressed.

## 6. Richer perception primitives exposed to code

**What they have.**

- **`locate(text, camera)`** (PyRUA RoboDojo). SAM3 segments the object in one camera, then finds it in the
  others. It returns `center_xyz`, `top_z`, `height_m`, `length_m×width_m`, `long_axis_deg` and `grasp_yaw`.
  It also reports how many cameras agreed: "one camera cannot measure a height".
- **`back_project(row, col)`.** Lets the LLM pick a pixel and get its 3D point. Both RoboRSI and its real
  PiPER setup expose depth *only* through such pixel queries.
- **`region_center(rows, cols, z_min, z_max)`.** Finds the center of a container cavity.

**Gap here.**

- `detect_object` uses one head camera.
- Multi-view triangulation (`providers/tracker3d/weighted_triangulate.py`) exists but only inside tracking.
- `convert_2d_point_to_3d_world` and `tracking/geometry.py` `deproject_pixel_to_world` exist but are not in
  `get_exec_locals`.

**Proposal.** Expose `back_project`, `region_center` and a multi-camera `locate` (head plus wrist) that
returns a camera-agreement count and a grasp yaw. This is cheap, since most of the code already exists.

## 7. Orientation-aware motion primitives

PyRUA-Lean's LIBERO API has:

- `move_to(xyz, gripper, tol, step_clip, target_yaw)`, with a 0.30 m cap per call and a `ValueError` beyond
  it.
- `move_pose(xyz, target_pitch, target_yaw)`, which co-varies position and orientation to get around
  singularities and recessed targets.
- `rotate_wrist` / `rotate_pitch`.

RoboDojo adds `path="line"` vs a joint-space swing, plus pitch/yaw/roll grasp semantics for side grasps of
upright objects. The knowledge guide tunes `step_clip` per object class (0.012–0.025).

**Gap here.** Only `[x, y, z, theta]` waypoints plus the `side_grasp_pose` / `pour_pose` presets.

**Proposal.** Add pitch- and roll-capable targets, a straight-line Cartesian path option, and a per-call
travel cap that returns `reached=False` instead of silently going partway.

## 8. Task-agnostic operating-knowledge doc

PyRUA injects one `knowledge/<robot>.md` per robot: frames, units, gripper semantics, step-size tuning,
"carry with `gripper=+1` in every move or it drops", and "no reset — recover in place". It contains no task
recipes. With code the guides were worth +6 pp on LIBERO-PRO. For tool calling they sometimes hurt (RoboTwin
without VLA: 42.8 → 60.4 *without* guides), so measure before adopting.

**Gap here.** Our `SKILL.md` files are task-family specific (door/drawer, pour).

**Proposal.** Write `prompts/knowledge/franka_panda.md` with robot-wide pitfalls and always inject it, with a
`--no-guides` ablation flag.

## 9. Judging success without a success flag

PyRUA reads `robo.done` (privileged under our rules) on LIBERO, RoboTwin and RoboCasa. On **RoboDojo** it
has no flag and still ranks first, using these code-level rules:

- A grasp counts only if the target moves with the arm.
- A placement counts only after the gripper opens and the arm withdraws ("on a rim ≠ placed").
- Before finishing, re-`locate` every named object and compare it to the instruction.
- Measure stated amounts with a 2–3 cm margin.
- Never repeat the same failed recovery twice.
- Say so when you cannot tell.

RoboRSI also decouples the agent's `finish(success)` from the environment's `solved()` in logs.

**Proposal.** Add these as a reviewer and planner checklist. Have the planner do a final "re-perceive all
task objects" pass before `plan_completed`. Log the agent's claim against `task_error` (evaluation only) to
measure the false-success rate.

## 10. Introspection-generated API reference

PyRUA's `prompt.py` builds the API section from `inspect.signature` and `getdoc` on the live robot class.
Ours is hand-written in `prompts/main_prompt.py:4-129` and can drift. Generate it from the functions in
`get_exec_locals`, with docstrings as the single source of truth.

## 11. Budgets, ledger, accounting and harness

PyRUA records the following per episode:

- `result.json`: success, status/reason, the agent's `finish` claim, calls, env steps, token usage,
  compactions, sandbox flags and timings.
- `calls.jsonl`: every primitive with args, outcome, elapsed time and sim steps.

It also applies `--max-turns` and `--budget-s` budgets. `pyrualean summarize` aggregates SR, status, calls and
tokens.

**Gap here.**

- No unified LLM-call ledger.
- No token or cost accounting.
- No wall-clock cap.
- No batch evaluation over tasks × seeds. The `ablations/` folder holds prompt texts only.

**Proposal.**

- Add `calls.jsonl` and `result.json` to every run.
- Add a `scripts/eval_suite.py` that runs `franka_kitchen:*` × N seeds and prints an SR, calls and tokens
  table. Ideas 1–9 can't be validated without it.

## 12. VLA as an optional primitive or a closed-loop review

**Evidence.**

- **RoboRSI** collaboration modes on RoboDojo:

  | Mode | SR |
  |---|---|
  | π0.5 alone | 6.9 |
  | Open loop (LLM subgoals) | 5.8 |
  | **Closed loop** (LLM reviews, accepts, modifies or rejects VLA chunks via FK previews) | 23.6 |
  | Closed loop with a review horizon of L=4 | 25.8, with −63% tokens |
  | Closed loop + ICL | 54.2 |

- **GPT-as-Policy Hybrid.** 48% vs 26% for Direct on RoboDojo, with only 14.4% of executed steps
  LLM-corrected. On dexterous tasks: 61.6 vs π0.5 44.2 vs Direct 16.6.
- **PyRUA.** VLA calls are ordinary primitives (`pi0_pick` with a heuristic `Pick.success`). The guide warns
  to verify a grasp independently.

**Caveats.**

- RoboRSI found Hybrid can *destroy* successes the policy achieves alone.
- On unseen compositions, Direct beat Hybrid (56% vs 36%), and the McNemar test was not significant.
- PyRUA left VLAs out of RoboDojo because of run-to-run variance.

**Proposal (long term).**

- Add a `vla_act(prompt, max_chunks)` primitive behind a provider interface, ideally a π0.5 / OpenVLA server.
- Add an "FK-preview review" mode in which the LLM accepts N steps or edits ≤5 cm / ≤0.35 rad.

## 13. Knowledge packages and refined notes between trials

**What they do.**

- **RoboRSI** builds a frozen package per task family *before* testing:
  - guidance `.md`
  - motion references `.npz`
  - visual references `.png`
  - executable helpers `.py`
  - checks and recovery
  - robot metadata

  No test-episode information leaks into it. Results: SIMPLE at 1× budget went from 70.8 to 90.8, and with
  closed-loop VLA it reached the best RoboDojo score.
- **GPT-as-Policy** refines written notes between trials while the weights stay fixed. It warns that
  retention is unreliable: one procedure passed 3/3, then 2/3 on retest.
- **RoboRSI caveat.** Benefit "weakens as tasks move away from the data used to build the package".

**Gap here.** `--learn-from-trajs` makes one offline LLM call and the result is pasted in by hand.

**Proposal.**

- Automate a `knowledge/<task_family>/` package (notes, helper code, reference crops) refined from logged
  runs.
- Gate it behind a flag and keep train and test seeds separate.

## 14. Larger interaction budget with explicit recovery

RoboRSI's 10× budget lifts RoboDojo from 54 to 82 and Bench2Dex from 26 to 62. Recovery ability ("detect
failed actions, revise strategy, recover") is what turns extra budget into success. PyRUA's guide says
"There is no reset. Recover in place."

The caveat on real hardware: irreversible failures (a dropped object), protective stops and wall-clock limits
cap this.

**Proposal.** Make the budget a wall-clock or LLM-call budget instead of a fixed `max_attempts` of 1–3. Log
recovery events so the gain can be measured.

## 15. Active vision

On the real dual-arm Piper, GPT-6-Astra **moved its free arm's camera** until the vase mouth was visible, and
then succeeded. The single-arm Franka failed because the wrist camera, attached to the held stem, never saw
the target ("what it can do depends on what it can see").

**Proposal.**

- Add a `look_at(xyz)` / `move_camera_view(pose)` primitive that moves the wrist to a viewing pose and
  returns images.
- Encourage it in the reviewer when the target is occluded.

## 16. Adaptive action chunking

RoboRSI's action-representation ablation:

| Variant | SR |
|---|---|
| Baseline | 15.6 |
| Model picks chunk length L ∈ {1..4} | 20.0, tokens 1.15M |
| Fixed L=4 | 11.4 |
| Directional motion commands | 18.1 |
| Joint-angle control | 9.7, a regression |

This argues for keeping a Cartesian or directional action space. When a trajectory runs to completion
without intermediate checks, let the code (or the LLM) choose how many waypoints to execute before
re-observing.

## 17. Sandbox with audit

PyRUA's sandbox:

- import allow-list
- guarded `open`
- blocked `__globals__` / `__subclasses__`
- `audit_source()` flags for references to backend, ledger or sim internals

The flags are stored per cell so that "an episode that reached for the simulator's internals is visible …
instead of silently counting as a success".

This directly supports our Sim2Real rule of not leaking privileged state. Add an audit pass that flags
generated code touching `simenv`, `get_state` or `p.getBasePositionAndOrientation`, even if it is not
blocked.

## 18. GRPO fine-tuning of a small coder

PyRUA Coder v0.1-4B:

- Qwen3.5-4B with full fine-tuning and **binary success reward only**.
- 45 instances, 30 iterations × 128 episodes, about 12 h on 8 GPUs.
- Unseen SR went from 18.8 to 44.1 (27B base: 29.2).
- The gain was mostly interface discipline: one cell per call, few image requests, no context overflow.

RoboRSI's 27B-Preview shows the same pattern (RoboCasa365 26.0 → 46.5 with −73% output tokens).

Prerequisite: idea 11 (an automated harness with a programmatic success signal used **only as the reward**,
never in prompts). This is a long-term direction, not a near-term change.

PyRUA training details (arXiv 2610.01939v2, section 5 and appendix J):

- Each iteration samples 16 instances with 8 rollouts each.
- Advantage is normalized within each group of 8. Clip range is 0.2 / 0.28. No KL or entropy term.
- Adam with a constant LR of 1e-6, full fine-tuning, no SFT stage and no demos.
- Rollouts are capped at 24 cells, 25 min and 4,096 tokens per call.
- Runs on the miles framework with SGLang rollouts, on one node with 8 GPUs, about 12 h.
- The cost is dominated by **sim rollouts**: 3,840 episodes. The gradient steps are cheap (60 steps).

### 18b. Alternative: on-policy self-distillation from a privileged same-model teacher (OPSD) + LoRA

> **Update (2026-10, see `docs/train/self_distillation_variants.md`).**
> For multi-turn tool or code agents, pure OPSD with privileged info in the KL loss is **not recommended**:
> - In PSP (arXiv 2609.29051) it fell below the untrained base model.
> - A 132-run reality check (arXiv 2610.19789) found no self-distillation method that consistently beats GRPO.
>
> Use privileged info only to *collect data* or *write hints* (PSP), not as a token target.
> For 1-2 GPUs, follow 18c.

Idea:

- The student (the local model, pixels + proprio only) generates its own episode on-policy.
- The **same model** with privileged sim info in context acts as the teacher. Privileged info here means
  object poses, the hinge axis, or a hindsight summary of a successful episode.
- The teacher re-scores the student's tokens in a single prefill pass.
- Loss: per-token reverse KL(student || teacher). This can be limited to tokens where the two diverge
  strongly or the student is uncertain (high entropy).
- Train a LoRA adapter only. The teacher is the frozen base with the adapter off, which keeps it stable.

Why it should be cheaper than GRPO:

- **Dense signal.** Every token gets a target, while GRPO gives one bit per episode. One rollout per
  instance is enough instead of a group of 8, and episodes that fail everywhere still teach something.
  GRPO gets zero gradient from groups that are all-0 or all-1.
- The teacher costs one prefill per episode, with no generation.
- LoRA fits one or two GPUs. A 4B model in bf16 plus a LoRA adapter can train and serve rollouts with
  SGLang/vLLM on a single 48-80 GB card.
- Expected order of magnitude: about 4-8x fewer episodes than GRPO. Sim time still dominates, so the
  saving is in rollouts, not in gradient compute.

It fits our Sim2Real rule:

- Privileged info goes **only into the teacher context during training**.
- The deployed student never sees it.
- This is the standard 'privileged teacher' pattern, as in RL sim2real.

Risks and mitigations:

- **Unobservability gap.** The teacher can be confident about things the student cannot infer from
  pixels, such as the exact hinge axis. Imitating that teaches the student to hallucinate confident
  numbers.
  - Mitigation: give the teacher *observable-level* hints, e.g. 'the door opens to the left', or a
    hindsight log of what worked. Avoid raw ground-truth floats.
  - Mitigation: mask the loss on numeric tokens that come straight from privileged values.
- **Teacher not much better than student.** Self-distillation only helps if privileged context really
  improves the model's code.
  - Check this first: run the local model with privileged hints and measure SR. If it doesn't rise, OPSD
    has nothing to transfer.
- **No outcome grounding.** The student can match a teacher that is itself wrong.
  - Mitigation: weight the per-episode loss by the binary success reward, or mix in a small GRPO / RFT
    term later.
- **Cheapest first step:** rejection-sampling SFT (RFT) on successful privileged-teacher episodes with
  LoRA. It is offline and simple, and gives a baseline that OPSD must beat.

Prerequisites, same as idea 18:

- idea 11: the automated harness and a success check used only as reward;
- a train/test seed split;
- a local OpenAI-compatible server (SGLang/vLLM) that can serve the LoRA adapter for rollouts.

### 18c. Recommended low-compute path (1-2 H100): SFT/RFT + LoRA first, GRPO last

Why not GRPO first:

- VRAM is not the problem. A 4B model with LoRA fits one H100; CaP-RL ran GRPO for a 7B model on one H100.
- The cost is **sim rollouts**. GRPO needs a fresh group of 8 multi-turn episodes per task at every step,
  and throws each one away after one update. In 12 h, that caps the dataset at a few thousand episodes
  over about 45 tasks.
- SFT-style data is collected once, reused for many epochs, and can be scaled to many tasks and seeds.

Stages (each one is a baseline the next must beat):

1. **Teacher-data SFT (no GPU for data).**
   - A frontier model (GPT-6 class) runs our normal pipeline on many tasks and seeds.
   - Keep only sim-verified successes. Train a LoRA on them (multi-turn chat format, loss on assistant
     turns only).
   - Evidence: Robo-Harness K1 trained a useful 9B student from 107 teacher episodes (arXiv 2609.29389).
2. **Privileged hints for collection only.**
   - Where the teacher fails every attempt, re-run it with a short qualitative hint written from sim state
     (PSP-style). Train on the successful run *with the hint removed*.
   - PSP reports that SFT on such trajectories already captures most of its gain.
   - Risk: the 'privilege gap' — the student learns confident actions it cannot justify from pixels.
     Keep only runs whose code detects and measures before acting. Reject runs that hard-code
     ground-truth numbers.
3. **Iterated RFT (rejection-sampling fine-tuning / expert iteration).**
   - The student samples k attempts per task and is fine-tuned on its *own* successes. Repeat 2-4 rounds.
   - On-policy data, so there is no privilege gap. The cost is sampling only, with no group-relative
     training loop.
   - RAFT++ (arXiv 2504.11343) found this competitive with GRPO early in training, falling behind later.
4. **Optional DPO/KTO** on success/fail pairs from the same task and seed. It is offline and cheap, and
   uses the failures that RFT discards.
5. **Optional short GRPO with LoRA**, only on tasks with mixed success (0 < SR < 1), where GRPO has signal.

Monitor:

- Unseen-task SR on a held-out seed split.
- Wrong-claim rate: code that uses values it cannot observe.
- Number of re-detect / check calls, to catch suppressed verification.
- Cells and tokens per episode (PyRUA's main gain was interface discipline).

Known weakness: offline SFT generalizes worse than on-policy training. Experience Distillation
(arXiv 2607.21051) found SFT kept only 3.8% of in-context gains, versus at least 64.8% for on-policy
distillation. Stage 3 exists to close part of this gap.

## Explicitly *not* to copy

- **Reading the benchmark's success flag** (`robo.done`, `terminated`) in the agent loop. It is privileged.
  PyRUA's RoboDojo setting shows strong results without it.
- **Joint-angle direct control by the LLM** (RoboRSI −5.8 pts vs EEF) and **dense whole-body references**
  (GPT-as-Policy locomotion: 0/5 attempts reached the goal).
- **"More steps" as a substitute for verification.** On real robots, RoboRSI saw protective stops,
  irreversible drops, and about 90% of runtime spent deliberating on G1.

## Suggested order

1. Idea 11 (harness and ledger), so everything else can be measured.
2. Ideas 2, 3, 10 and 17: small, mechanical changes.
3. Ideas 1, 6, 7 and 9: the interface and perception upgrades with the strongest evidence.
4. Ideas 4, 5 and 8: reviewer, context and knowledge.
5. Ideas 12–16 and 18: larger research items.

## Related-work agents

The 12 related agent papers mined from the related-work sections are summarized in `docs/models/related_agent_papers.txt`. The ideas most relevant to us: Thea evaluator exit codes (hidden per-tool post-conditions, 3-state verdict), Robo-Harness K1 perception-as-tools (back-projection, cross-view depth check, anchor co-motion), Zetta action-rate code critics, ReTrace cross-stage failure attribution, and CaP-X visual-differencing text feedback.
