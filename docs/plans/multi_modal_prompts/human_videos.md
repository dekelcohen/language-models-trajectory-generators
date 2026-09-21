# Plan: multi-modal prompts - conditioning on human demonstration videos

Status: **proposal / not implemented.** This document is the design for adding demonstration
conditioning (human demo video, goal image, robot rollout video, numeric trajectory log) to
`language-models-trajectory-generators`.

Companion reference docs:

- [`../../models/gpt_policy.md`](../../models/gpt_policy.md) - GPT-Policy's offline context
  compiler: ffmpeg candidates -> two-pass VLM keyframe select + review -> compact text + keyframe
  images, content-addressed cache.
- [`../../models/show_harness.md`](../../models/show_harness.md) - Show-Harness's `video_ref`
  plugin: 8 uniform frames -> one analyst call -> a **text-only** brief spliced into the planner
  prompt. Cheapest known design, and the source of the headline result *demonstrated-order
  following 20% -> 20/20 from a single demo video*.

---

## 1. TL;DR

Today a run is conditioned on a text command plus live camera images. We want to also accept
`--demo-video human_video1.mp4`, `--goal-image goal.png` and/or a trajectory log, digest them
**once, offline, with a cheap VLM pass**, and splice the result into the planner and sub-task
prompts through a new `[INSERT DEMONSTRATION]` slot - reusing the exact pattern that
`[INSERT SCENE ANALYSIS]` already uses today.

The guiding principle from both reference systems: **the demonstration is digested into text (plus
at most a handful of labelled keyframes) before the policy model ever sees it.** Raw video never
enters the control loop.

Milestone 0 is deliberately tiny: sample 8 frames, make one VLM call, inject a text block. That
alone reproduces the Show-Harness result.

---

## 2. Why this is worth doing

| Evidence | Source |
|---|---|
| One demo video takes demonstrated-order following from 20% to 20/20 | Show-Harness paper |
| Pick Red Towel 0/3 -> 2/3; Pick Up Notebook 0/3 -> 2/3; Unscrew Bottle Cap 0/3 -> 2/3 (video) -> 3/3 (video+action); Remove and Reinsert Plug 0/3 -> 2/3 | GPT-Policy paper |
| Goal-image, self-history and human-interaction tasks all 3/3 | GPT-Policy paper |

In both systems the same fixed model fails the task without the demonstration and succeeds with it.
No gradient updates are involved - this is purely a prompt-construction win, which is exactly the
kind of change our repo can absorb.

---

## 3. Current state of this repo (verified)

The infrastructure is already here; what is missing is the digest step and one prompt slot.

| Capability | Where |
|---|---|
| Pre-pass VLM writes into a prompt slot | `helpers/perception_scene_analysis.py` -> `[INSERT SCENE ANALYSIS]` |
| Slot filling | `agent_runner.py:543` (sub-task) and `:944` (planner), both `.replace("[INSERT SCENE ANALYSIS]", scene_analysis)` |
| Slot declarations | `prompts/main_prompt.py:132`, `prompts/planner_prompt.py:64` |
| Skills injected the same way | `build_skill_tools_section(...)`, `build_skills_section(...)` at `agent_runner.py:550-551, 948-949` |
| Self-history already works | `helpers/main_utils.py:141` `[INSERT PAST TRAJS]`, `prompts/learn_in_context_examples_from_past_attempts.py`, CLI `--learn-from-trajs` |
| Video already reaches a VLM | review pipeline `api.py:808-838` (`[INSERT VIDEO LIST]`, `[INSERT FRAME PATHS]`, `[INSERT SCENE ANALYSIS]`), fed by `helpers/video_utils.py:build_review_clips` |
| ffmpeg wrappers | `helpers/video_utils.py`: `ffmpeg_available`, `concat_videos`, `build_attempt_clip`, `update_full_video`, `build_review_clips` |
| Multimodal message assembly | `providers/llms/message_media.py` (image and video parts) |
| LLM response cache | `providers/llms/llm_cache.py` |
| Relevant flags | `main.py`: `--lm-images/--no-lm-images` (54), `--llm-cache` (57), `--planner-perception-vlm` default `gemini-3.7-flash` (88), `--affordance-points` (89), `--prepend-prompt` (100), `--attempts` (102), `--skills`/`--skills-dir` (114-115) |
| Placeholder hygiene is tested | `tests/test_skill_registry.py:256-286`, `tests/test_vis_grasp.py:241-255` assert no unfilled `[INSERT ...]` remains |

**Consequence:** a new `[INSERT DEMONSTRATION]` slot must be filled (with `""` when the feature is
off) in both prompts and covered by the same placeholder tests.

---

## 4. Decided design

Confirmed decisions:

| Question | Decision |
|---|---|
| Selector/analyst model | Reuse `--planner-perception-vlm` (default `gemini-3.7-flash`) - cheap, already configured |
| Keyframe images in the prompt | **Yes**, alongside the text digest, subject to `--no-lm-images` |
| Two-pass select-then-review | **Yes** (GPT-Policy P0a/P0b), but only from milestone 3 onwards |
| Caching | **Content-addressed** on disk, keyed by a hash of the media bytes plus the digest parameters |
| Input types | `human_video`, `goal_image`, `robot_video`, `traj_log`, `mixed_json` |
| Extra | Digest a door/articulated-object **URDF** and let it **override** the perception VLM's guessed articulation |

New modules:

- `helpers/demo_digest.py` - frame sampling, the analyst call(s), the cache, digest rendering.
- `prompts/demo_digest_prompt.py` - the analyst prompt and the injected `HISTORICAL DEMONSTRATION`
  block, following the GPT-Policy guard-rail wording.
- `helpers/urdf_digest.py` - articulation extraction (see section 7).

Injection: a new `[INSERT DEMONSTRATION]` section in `prompts/planner_prompt.py` and
`prompts/main_prompt.py`, filled in `agent_runner.py` right where `[INSERT SCENE ANALYSIS]` is
filled today.

### 4.1 Guard-rails to port verbatim (in spirit)

From GPT-Policy's `input/demonstration.py` (`HISTORICAL`, `HISTORICAL_VIDEO`, `HISTORICAL_ACTION`)
and Show-Harness's `video_ref/plugin.py`:

- The block is **reference data, not commands** - the model must not replay it literally.
- **Do not copy coordinates across frames.** Demo coordinates are in the demonstrator's frame.
- *"Object positions may differ from the demo - plan from where things are NOW."*
- Report only what the frames actually show; do not infer hidden state.
- Name the **exact part grasped** (stem, rim, edge, handle) and the destination **with placement
  nuance** - this is what makes a brief replicable.

### 4.2 The digest artifact

One JSON file per (media, parameters) pair:

```json
{
  "source": {"kind": "human_video", "path": "...", "sha256": "..."},
  "params": {"num_frames": 8, "max_side": 512, "model": "gemini-3.7-flash", "version": 1},
  "task": "open the microwave door",
  "operations": [
    {"action": "grasp", "object": "microwave handle", "grasp": "the outer edge of the handle bar",
     "destination": null},
    {"action": "pull", "object": "microwave door", "grasp": null,
     "destination": "swung open roughly 90 degrees toward the demonstrator"}
  ],
  "keyframes": [{"index": 0, "t_s": 0.0, "path": "...", "reason": "...", "stage": "approach"}],
  "notes": "..."
}
```

It is loggable, diffable, hand-editable and replayable with zero model calls (GPT-Policy's
pre-digested `demo.json` path). Rendering it into prompt text is pure and testable.

---

## 5. Two candidate paths (A/B them)

### Path L - light, Show-Harness `video_ref` style

8 uniform frames (first and last always included), 512 px, **one** analyst call with a strict JSON
schema, a text-only block injected into the planner prompt. No offline keyframe search, no review
pass, no images at control time.

Cost: one cheap VLM call per run. Complexity: ~150 lines. This is milestone 0 and the permanent
A/B baseline.

> **Caveat from SeeDo (arXiv 2410.08792), added after the literature pass.** Uniform sampling is
> the weak link: SeeDo-Unif. - *the same prompts* with 16 uniform frames - scores **0.0 / 0.0 / 0.0**
> task success where motion-cued keyframes score **60.5 / 26.7 / 21.6**, because "the context limit
> is often exceeded when inputting all sampled frames at once." Show-Harness survives uniform
> sampling only because it uses 8 small frames in a *separate* analyst call. So: keep Path L as the
> baseline, but plan **Path L+** (motion-cued keyframe selection, below) as the first upgrade - it
> is ~60 lines of numpy and does not need the full Path H machinery.

### Path H - heavy, GPT-Policy context-compiler style

ffmpeg candidate extraction at 2 fps (<= 24 candidates per 30 s window, 768 px) -> **P0a select**
(strict per-frame schema: index / reason / stage / left / right / result, plus a summary) ->
**P0b review** (target 12-16 frames, always keep first and last, limit
`min(24, 48 // views_per_frame)`) -> compact text plus <= 16 labelled keyframe images at 1280 px ->
content-addressed cache.

Cost: two to three VLM calls plus ffmpeg per unique video, amortized by the cache. Complexity: the
bulk of the work. Justified only if Path L measurably under-performs.

**Decision rule:** ship Path L, measure, and only add Path H stages that a measured failure demands.

---

## 6. Technique catalog from the wider literature

Collected so the design is not over-fitted to two systems. Every arXiv ID below was verified
against the arXiv API and the code-release status was checked by fetching the repo. Grouped by
*how the extra modality reaches the model*.

### 6.1 Human video -> a frozen VLM's prompt (the closest analogues)

- **SeeDo** (arXiv 2410.08792, IROS 2025, <https://github.com/ai4ce/SeeDo>, **code released**) -
  long-horizon human pick-and-place video -> GPT-4o -> a natural-language step plan -> executed via
  Code-as-Policies on a UR10e. This is the single most useful reference for us.
  - **Keyframes are motion-cued, not uniform.** MediaPipe hand landmarker -> hand-centroid speed per
    frame -> `np.interp` over NaN gaps -> `gaussian_filter(sigma=5)` ->
    `find_peaks(-curve, prominence=0.8)`. Valleys (the hand slows down) are the keyframes; valleys
    within 15 frames of the previous one are dropped. The count is **variable**, roughly
    `#picks + #places`.
  - **A chain of single-image calls.** Exactly **one image per API call**, ~4 chained GPT-4o calls
    per pick-place pair, `resize=768`, `temperature=0`, `max_tokens=400`: name the objects in frame
    0 -> use that text as the open-vocab GroundingDINO query -> SAM2 track -> then per keyframe ask
    (a) is the hand actually manipulating (a second, VLM-side keyframe filter), (b) which object is
    picked, (c) which is the reference object, (d) the spatial relation from a closed set of 6.
  - **Marks, not coordinates.** Mask **contours** (not filled masks, "to avoid obstructing
    appearance") plus integer track IDs are drawn on each keyframe. The paper claims centroid
    coordinates are appended as text; in the shipped `vlm.py` that injection is commented out.
  - Numbers: SeeDo **60.5 / 26.7 / 21.6** TSR vs **SeeDo-Unif. 0.0 / 0.0 / 0.0** (same prompts,
    16 uniform frames), GPT-4o init+final only 39.5 / 13.3 / 10.3, native Gemini video input
    39.5 / 16.7 / 0. Removing contours+IDs on the block task: TSR 21.6 -> 0.0, spatial error 87.5%.
- **Video-to-BT** (arXiv 2509.16611, **no code**) - human assembly video + **ASR text of the spoken
  narration** + a symbolic domain spec -> GPT-4o -> reactive Behavior Tree XML. Uses Set-of-Mark
  numbering alongside **mesh-rendered CAD reference images** to match mask-ID to part name, and
  carries execution state forward as text between subtree prompts. The narration channel is unique
  in this survey and nearly free if our demo videos have audio.
- **Vid2Robot** (arXiv 2403.12943, RSS 2024, **no code**) and **Gen2Act** (arXiv 2409.16283,
  **no code**) - the *trained* contrast class. Both take **16 frames, first and last always
  included**, and compress them with a 2-layer Perceiver Resampler to **64 latents**; Vid2Robot's
  own arithmetic is 16+8 frames at ViT-B/16 = 4704 tokens, ~22M attention entries, vs ~0.3M after
  resampling. Vid2Robot with human prompt videos 52.8% vs BC-Z 30.6%. Gen2Act *generates* the human
  video from the fixed text template `"A person {task}, static camera"` and gets 60 avg vs 26 for
  its own last frame used as a goal image.
- **Track2Act** (arXiv 2405.01527, ECCV 2024, partial code) - no VLM and no text at all: initial
  image + goal image -> predicted future 2D point tracks -> rigid transforms -> an open-loop EE
  plan. Its anti-video-generation argument is worth keeping: "predicting an RGB video followed by
  tracking suffers due to issues of implausible generation because video generation is a much more
  complex task than predicting the tracks of a set of points."

### 6.2 Human video -> affordance / trajectory transfer (heavier, perception-model based)

- **RAM** - extracts 2D affordance (contact point plus post-contact trajectory) from hand-object
  interaction videos into a large memory, then retrieves hierarchically (semantic -> geometric ->
  ranking) with vision-foundation features and lifts the match into a 3D trajectory. Also supports
  one-shot conditioning on a single reference image.
- **SKIL-H** - CoTracker keypoint flows through human video, lifted to 3D, feeding a trajectory
  prediction module. Human video as a dense motion prior rather than a text brief.
- **ZeroMimic** - image-goal-conditioned skills distilled from EpicKitchens, deployable with no
  robot data.
- **VL-MP** - `DemoTraj`: keypoint poses in **object-centric local frames**, which is what makes a
  demo transferable when the object has moved.
- **ATM** (arXiv 2401.00025) and **KAT** (arXiv 2403.19578, RSS 2024) - ATM pre-trains point-track
  prediction from action-free video; KAT is the one result where a **frozen text-pretrained GPT-4
  does in-context imitation directly from tokenized 3D keypoints**, matching diffusion policies in
  the low-data regime. KAT is the closest precedent for feeding numeric trajectory rows as text.

Relevance to us: all of these agree that the useful content of a human video is *contact point plus
post-contact motion in an object-centric frame*, not pixels. Our analyst prompt should ask for
exactly that - `grasp` part and `destination` nuance are the text-only version of it.

### 6.3 Visual prompting - marks drawn on the image

- **Set-of-Mark** (arXiv 2310.11441, <https://github.com/microsoft/SoM>, **code released**) - the
  root mechanism that MOKA, CoPa, OmniManip and Video-to-BT all build on. Segment -> overlay
  speakable alphanumeric marks -> ask the model to ground to **mark IDs**. Mark placement is the
  non-obvious part: sort masks by ascending area, subtract the union of already-placed masks, run a
  **distance transform** on the residual and place the label at the point maximizing the minimum
  distance to a boundary. Headline number for this whole document: RefCOCOg
  **25.7 (ask for coordinates) -> 86.4 (ask for mark IDs)**, same model, **+60.7**. Mask quality is
  the bottleneck, not the VLM (75.6 mIoU with proposed masks vs 90.1 with ground truth), and the
  ability is "emergent only in GPT-4V-class models" - LLaVA-1.5 and MiniGPT-v2 "can hardly interpret
  the marks."
- **MOKA** (arXiv 2403.03174, RSS 2024, **code + prompt files released**) - two mark types on one
  image: **6 candidate keypoints per object** (farthest-point sampling on the mask contour plus the
  geometric center; red `P[i]` on the grasped object, blue `Q[i]` on the target) and a **5x5 chess
  grid** `a-e` by `1-5` for waypoints. Free-space depth is *dodged, not solved* - the VLM only picks
  a discrete height in `{same, above}`. Grasping is not left to the VLM either: 30 antipodal grasps
  are sampled and the one nearest the chosen keypoint is executed. Rationale, verbatim: "VLMs are
  better at multiple-choice problems than directly producing continuous-valued locations."
  **Ablation that matters most: removing the hierarchy** (decompose into subtasks *before* marking,
  and mark only what the current subtask needs) drops reasoning success from ~0.8 to ~0.1 - a bigger
  effect than marks, CoT or legend text combined.
- **CoPa** (arXiv 2403.08248, **code released**) - coarse-to-fine SoM: object-level marks -> the VLM
  picks an object -> **crop it** -> a second SoM pass at *part* granularity. Robot-arm masks are
  filtered out by rendering the URDF because "numeric marks on the robotic arm may affect VLMs'
  selection". The VLM never outputs coordinates - it emits **symbolic spatial constraints** ("Vector
  A and Vector B are on the same line, opposite direction") that are solved numerically. 63% full vs
  46% without coarse-to-fine vs **37% when the VLM emits numeric poses directly**, and it needs 3
  in-context examples where VoxPoser needs 85.
- **PIVOT** (arXiv 2402.07872, ICML 2024, **no code**) - continuous control as iterative VQA: sample
  candidate actions, **draw them as numbered arrows**, ask the VLM to pick, refit the sampling
  distribution, repeat. No perception pre-pass at all. Depth is encoded in the *annotation style*
  (red = away, blue = toward; larger circle = closer). Defaults: **10 samples, 3 iterations, 3
  parallel calls**; prompt ordering **preamble -> image -> task** works best. 25% -> 75-100% on
  navigation, grasp 0% -> 67%. More samples is not better: "the region of the image around the
  correct answer gets crowded and causes significant issues with occlusions."
- **Code-as-Monitor** - paints constraint masks and danger zones onto the frame for visual failure
  detection.
- **OWMM-Agent** - multi-view scene-image graph + egocentric frame + text memory, outputs boxes.
- **HAMSTER** (arXiv 2502.05485, ICLR 2025, partial code) and **RT-Trajectory** (arXiv 2311.01977) -
  a **drawn 2D gripper path** as the interface: Ramer-Douglas-Peucker simplification to 2-5 points,
  drawn with a **color gradient for temporal progression** and **circles at gripper changes (green
  = closing, blue = opening)**. Deliberately fine-tuned, because "pre-trained VLMs struggle with
  predicting such a path in a zero-shot manner." Useful to us as an *output verification* rendering
  rather than as an input.
- **RoboPoint** (arXiv 2406.10721, CoRL 2024, **code + weights released**) - the anti-visual-prompting
  position: instruction-tune a 13B model to emit normalized `(x,y)` points directly, nothing drawn at
  deployment. Two cautions verified against the paper's own tables: the claimed "+21.8% affordance"
  is really +17 to +35 depending on benchmark (~+23 mean), and **GPT-4o with 14 in-context examples
  scores worse than plain GPT-4o on all three benchmarks** (14.5 vs 29.1 on Where2Place), with 4-6x
  the variance. In-context exemplars actively hurt coordinate regression.

### 6.4 Point and coordinate prompting

- **Afford2Act** - text-conditioned functional keypoints from DINO/SAM features, tracked with
  CoTracker.
- **ManipLLM** - RGB + depth + text -> chain-of-thought -> a 2D contact pixel and a **discretized**
  SO(3) approach direction. The same instinct as Show-Harness's action units, applied to orientation.

Relevance: asking a VLM for a **pixel plus a discrete direction** is consistently more reliable than
asking for a pose.

### 6.5 Feedback loops, self-history and goal images

- **Wonderful Team** (arXiv 2407.19094, TMLR 2025, **code released, 85 KB of prompts in
  `llm_prompt.yaml`**) - **benchmarks directly against this repo's upstream paper**, so its numbers
  are the most actionable comparison available:

  | VIMABench task | Trajectory Generators (2310.11604) | Wonderful Team |
  |---|---|---|
  | Visual Manipulation | 60 | 100 |
  | Scene Understanding | 40 | 100 |
  | Novel Adjective | 10 | 70 |
  | Novel Noun | 0 | 100 |
  | Without Exceeding | 10 | 90 |
  | Fetch Box long-horizon (1 attempt / with replan) | 0 / 5 | 50 / 80 |

  Six GPT-4o agents, **no detector at all**. Mechanisms worth copying: **zoom-in crops with drawn
  pixel-tick axes**; a **before/after montage** ("the top image is the original bounding box, the
  bottom is the proposed revision"); 5 numbered semi-transparent candidate points on the crop; and a
  **rendered-trajectory self-check** - after the supervisor writes trajectory Python, `draw_func_traj`
  renders the path **in red on the scene image** and sends it back ("if the visualization shows the
  correct geometric shape... otherwise request a new function") inside a 5-retry loop. History is
  per-agent, **text-only, FIFO-capped at 10 exchanges, and images are never retained**. The
  statistics that justify all of it: raw GPT-4o coordinates are "close to the target" 90% of the time
  but only **33% directly actionable**, while GPT-4o classifies a *drawn bounding box* correctly
  **97%** of the time.
- **SAIL** (arXiv 2603.08269, **no code**) - trajectory-level MCTS over in-context imitation, and the
  best self-history design found. The generator sees **numbers** (discretized 3D keypoints); a
  separate scorer sees **50 uniformly sampled frames** of a candidate rollout and grades each against
  the current subtask, producing an **annotated trajectory whose waypoints carry completion scores**;
  the next expansion prompt is told to "preserve high-scoring segments while modifying low-scoring
  segments." Its feedback-modality ablation is directly on point for us:

  | Feedback placed in the prompt | Avg success |
  |---|---|
  | **Step-level scored waypoints** | **65%** |
  | Sparse final score only | 49% |
  | Trajectory text only | 48% |
  | Images + trajectory, no scores | 46% |
  | Rollout images only | 45% |

  Retrieval ablation: similarity-based **K=1 gives 65%**, a fixed demo K=3 gives 49%, random K=3
  gives 53% - **relevance beats quantity**, and K=1 keeps the token budget flat.
- **OmniManip** (arXiv 2501.03841, **code 404s**) - the **RRC loop (Resample, Render, Check)**: for
  each candidate constraint, render a synthetic pre-visualization of the resulting interaction and
  show *that image* back to the VLM for success / failure / refinement. Closed loop **68.3%** vs open
  loop 51.7%. Also handles invisible points (the center of a teapot opening) by switching to an
  orthogonal view, and grounds directions by **captioning each candidate axis in words and scoring
  the captions**, never by reading numeric axes.
- **ReKep** (arXiv 2409.01652, CoRL 2024, **code released**) - **exactly one annotated image plus the
  instruction** in a single message, `temperature=0.0`, `max_tokens=2048`, no image-text in-context
  examples. Keypoints from DINOv2 features -> SAM -> PCA -> **k-means k=5** -> numeral overlay; the
  VLM writes Python constraint functions over a `(K,3)` keypoint array. Auto keypoint proposal costs
  **24 points** (44.3% vs 68.6% with human-annotated keypoints), and it is viewpoint-fragile (0/10
  frontal vs 7/10 top-down).
- **ViLA** (arXiv 2311.17842) - the canonical frozen-GPT-4V long-horizon planner with **multimodal
  goal specification** and visual feedback; the reference point for goal images in an agent rather
  than in a trained net.
- **Goal images** are also used by ZeroMimic and RAM (one-shot reference image). But the measured
  verdict is consistent: a goal image is a **supplement, not a substitute**. Gen2Act's own goal-image
  variant scores 26 vs 60 for the full video; SeeDo's init+final-frame baseline scores 39.5 vs 60.5.

### 6.6 Digestion strategies, compared

| Strategy | Systems | What enters the prompt | Images/call | Measured effect |
|---|---|---|---|---|
| Detections as ASCII floats, no images | **this repo today**, VoxPoser | `Position of X: [x,y,z]`, dims, yaw | 0 | 57.3% vs CaP 22.0%, but 48.3% of failures are gripper-pose prediction |
| Raw uniform frames dumped in | SeeDo-Unif. | 16 uniform frames | 16 | **0.0% - total collapse** |
| Native video input | Gemini video baseline | the whole mp4 | n/a | 39.5 / 16.7 / 0 - beaten by keyframes |
| **Motion-cued keyframes + tracked-ID overlay, one image per call** | **SeeDo** | 1 annotated keyframe per call, chained | **1** | **60.5 / 26.7 / 21.6** |
| Init + final frame (goal image) | Gen2Act RT1-GC, SeeDo I+F | 2 frames | 2 | 26 vs 60; 39.5 vs 60.5 |
| Numbered segmentation marks | SoM, MOKA, CoPa, OmniManip | 1 marked image + a mark legend | 1 | RefCOCOg 25.7 -> 86.4; CoPa 63% vs 37% |
| Numbered candidate points | MOKA (6/obj + 5x5 grid), ReKep (k=5), Wonderful Team (5) | 1 | 1 | MOKA w/o hierarchy 0.8 -> 0.1; ReKep auto 44.3 vs 68.6 |
| Candidate action arrows, iteratively refit | PIVOT | 10 arrows, 3 iters x 3 parallel | 1 (x9 calls) | nav 25% -> 75-100%; grasp 0% -> 67% |
| Zoom crops + pixel ticks + before/after montage | Wonderful Team | several | several | 0 -> 100% on several tasks |
| **Render the proposal back and ask for approval** | OmniManip RRC, Wonderful Team | 1 synthetic render | 1 | +15 pts closed vs open loop; ~+80% from self-correction |
| Drawn 2D path, color gradient + gripper circles | HAMSTER, RT-Trajectory | overlay | n/a | overlay 0.83 vs 6-channel concat 1.00 |
| Numeric trajectory rows as text | this repo (`xmem_lm_input_every = 20`), KAT, SAIL | a text table | 0 | trajectory-text-only 48% vs 65% with scores |
| **Step-scored waypoints** | **SAIL** | annotated trajectory text | 0 | **65% vs 45-49% for any single modality alone** |
| Learned latent tokens | Vid2Robot, Gen2Act, ContextFlow | 16 frames -> 64 latents | n/a | requires training - out of scope |

### 6.7 Ranked recommendations for this repo

1. **One image per VLM call, chained - never dump N frames into one call.** Best ratio in the
   survey (60.5 vs 0.0 with the same model and prompts).
2. **Motion-cued keyframe selection** (~60 lines of numpy). The motion proxy does not have to be
   MediaPipe hands: for robot rollouts, TCP speed and gripper-state transitions are already logged.
3. **Numbered mask-contour overlays plus a text legend; never ask for raw coordinates.** SoM's
   +60.7 and Wonderful Team's 33%-actionable-coords vs 97%-box-classification are the same finding
   from two directions. We already have segmentation and XMem masks, so this is ~40 lines of OpenCV
   using SoM's distance-transform label placement.
4. **Decompose into subtasks first, then mark only what the current subtask needs** (MOKA's
   hierarchy, ~0.8 -> ~0.1 if removed). This is a prompt-flow change, not new code.
5. **Step-scored waypoint feedback for the self-history path** (SAIL, 65% vs 45-49%). Pure string
   formatting over arrays we already have; slots into
   `prompts/learn_in_context_examples_from_past_attempts.py`.
6. **Render-and-check**: draw the proposed trajectory back onto the scene image and ask the VLM to
   approve it, with a retry cap. We already have the camera projection in `sim_adapter/camera_math.py`.
7. **Relevance-based retrieval of past successful attempts, K=1** (SAIL: K=1 similarity 65% vs
   fixed K=3 49%).
8. **Images never persist in conversation history** (Wonderful Team). The main defense against
   token blowup once we start attaching keyframes.
9. **Discrete multiple choice wherever depth is involved** - MOKA's `{same, above}`, CoPa's symbolic
   constraints (63% vs 37%), ManipLLM's discretized SO(3).
10. **Goal image as a supplementary channel only** (2 lines of work, but do not expect it to replace
    a demo).
11. **ASR of demo narration** (Video-to-BT) - nearly free if the videos have audio, and orthogonal
    to every other channel.

### 6.8 Pitfalls, with attributions

- **Depth and 3D**: "none of the VLMs we tested are capable of reliably choosing actions based on
  depth" (PIVOT, despite encoding depth in arrow color *and* size). "The VLMs currently in use... lack
  a genuine grounding in the 3D physical world" (CoPa). Viewpoint fragility is severe - ReKep goes
  0/10 to 7/10 on the same task purely by camera angle.
- **Token budget**: SeeDo, "the context limit is often exceeded when inputting all sampled frames at
  once"; Vid2Robot's 4704-token / 22M-attention-entry arithmetic; OmniManip, "multiple VLM calls
  present computational challenges, even with parallel processing."
- **Hallucination**, PIVOT's mechanism verbatim: "it stochastically connects the thought process to
  the incorrect arrow... once the number is decoded, the VLM must justify it, even if incorrect, and
  thus hallucinates an otherwise reasonable thought process." Implication: **commit the mark ID
  first, reason second**, or use parallel calls and vote.
- **Demo-to-robot identity mismatch**: SeeDo, "tracking associates an object with a text description
  that the VLM refers to as another" - fixed by having the VLM name objects first and feeding those
  names as the detector query. Vid2Robot applies the right motion to the wrong object under
  distractors. Track2Act executes "trajectories that don't conform to the specified goal image."
- **Annotation legibility**: SoM marks get attributed to the wrong overlapping object; Wonderful
  Team notes SoM annotations "become unreadable when objects overlap or are small"; CoPa filters
  robot-arm masks via URDF rendering because marks on the arm bias selection - **if our sim renders
  include the arm, filter it**; PIVOT degrades past ~10 candidate arrows from crowding.
- **In-context examples can hurt**: RoboPoint measures GPT-4o *worse* with 14 exemplars than with
  none. Prefer **structure** (hierarchy, marks, step-scored feedback) over **exemplars**.

### 6.9 Code-release reality check (verified)

Usable today: SeeDo, MOKA (including its prompt `.txt` files), Set-of-Mark, CoPa (including its
few-shot image+text exemplars), RoboPoint, ReKep, VoxPoser, Wonderful Team (85 KB of prompts),
Track2Act (partial), HAMSTER (inference only).
**Not released**: PIVOT, Vid2Robot, Gen2Act, Video-to-BT, OmniManip (repo 404s), SAIL.

---

## 7. URDF articulation digestion (override)

Sim-only, independent of video, and cheap.

Parse `my_assets/adroit_door/adroit_door.urdf` and
`my_assets/franka_kitchen/item_assets/{hingecabinet,slidecabinet,microwave,oven}.urdf` for:

- joint type (`revolute` / `prismatic` / `continuous`), axis, and limits;
- the hinge and handle link frames and their relative transform;
- the derived motion arc: for a revolute joint, the handle sweeps a circle of radius
  `|handle_origin - hinge_axis|` about the axis, between the joint limits.

Render as a short, unambiguous text block, e.g.

```
ARTICULATION (ground truth from URDF):
- microwave door: revolute hinge, axis +z, at (x, y, z) in object frame, range 0 to 1.57 rad
- handle: at (x, y, z), 0.28 m from the hinge axis
- opening motion: pull the handle along an arc of radius 0.28 m about the hinge, NOT straight back
```

**Override semantics:** when a URDF is available for the manipulated object, this block replaces the
perception VLM's guessed articulation inside `[INSERT SCENE ANALYSIS]` rather than sitting beside
it, and says so explicitly so the model does not try to reconcile two stories. When absent, nothing
changes.

---

## 8. Phased roadmap

No time estimates - strictly ordered by dependency and by measured need.

| Phase | Content | Exit criterion |
|---|---|---|
| 0 | `helpers/demo_digest.py` frame sampling (8 uniform, first+last, 512 px) + one analyst call + `[INSERT DEMONSTRATION]` in both prompts, behind `--demo-video`. Digest JSON written to the run dir. | A demo video changes the generated plan in the expected way on one task |
| 0.5 | **Motion-cued keyframes** (SeeDo): speed curve -> `gaussian_filter(sigma=5)` -> `find_peaks(-curve, prominence=0.8)` -> min-gap 15 frames, with a VLM validity filter per keyframe. A/B against phase 0's uniform sampling. | Keyframe count adapts to the video and the digest names the right number of operations |
| 1 | Content-addressed disk cache; `--demo-digest <path.json>` to replay a digest with zero model calls; placeholder tests extended; **images never retained in conversation history** | Re-running a task makes zero extra VLM calls |
| 2 | Goal image support (`--goal-image`); keyframe images attached to the prompt under `--lm-images`, **one image per analyst call, chained** | Goal-image-only conditioning works end to end |
| 3 | Two-pass select + review for long videos; ffmpeg candidate extraction; per-window keyframe budget | Long (>60 s) videos digest without blowing the token budget |
| 4 | Robot rollout video and numeric trajectory logs (column-encoded rows, gripper-transition sampling); **step-scored waypoints** in the past-attempts prompt | A previous successful attempt can be replayed as a demonstration |
| 5 | `mixed_json` manifest combining several media for one task; **K=1 relevance retrieval** over past successes | One manifest drives video + goal image + trajectory |
| 6 | URDF articulation override | Door/microwave tasks use exact hinge geometry |
| 7 (independent) | Draw-and-verify upgrade to `--affordance-points`; **numbered mask-contour overlays with a text legend**; **render-and-check** of the proposed trajectory on the scene image | Verified marks measurably reduce bad grasp points and bad trajectory shapes |

---

## 9. Logging and Q/A plan

Permanent, one line per stage (cheap, outside any loop):

- `demo_digest: source=<kind> path=<name> sha256=<12 chars> bytes=<n>`
- `demo_digest: sampled <n> frames in <ms> ms (fps_source=<x>, max_side=<n>)`
- `demo_digest: cache <hit|miss> key=<12 chars>`
- `demo_digest: analyst model=<name> latency=<ms> ops=<n> chars=<n>`
- `demo_digest: injected into <planner|subtask> prompt, <n> chars, <n> images attached`
- one WARNING when the digest is dropped because of the token budget or `--no-lm-images`

Temporary (removed after validation): the full rendered block, per-frame selection reasons, and the
exact prompt sent.

Artifacts written next to the run's images root: `demo_digest.json`, `demo_frames/*.jpg`, and
`demo_block.txt`. A `--prepare-only` dry run should build the digest and print the block **without
starting the simulator**, which is the fastest Q/A loop available.

Regression guard: the digest renderer is a pure function, so a golden-file test on
`digest JSON -> prompt block` is enough to keep the wording stable.

---

## 10. Open questions (TBD)

1. First validation task and video - a door/microwave open is the best candidate because the URDF
   override applies to the same task.
2. Digest scope - one digest per command, or a fresh one per sub-task? Show-Harness extracts once
   per run and reuses it for every replan; start there.
3. Frame mapping for human demos - human hands are not a parallel gripper. The guard-rail sentence
   may be enough; if not, ask the analyst to describe the grasp in gripper-compatible terms.
4. Token budget and overflow behaviour - GPT-Policy fails loudly (`MAX_INPUT_CHARS = 1048576`,
   `FIRST_TURN_RESERVE_CHARS = 65536`) rather than truncating. Do we fail loudly or drop keyframes?
5. `--no-lm-images` fallback - text-only digest, or refuse the feature?
6. Reuse of existing review clips (`helpers/video_utils.py:build_review_clips`) as robot-video
   demonstration input - probably free.
7. **Windows-portable locking.** GPT-Policy's `input/video_cache.py` uses `fcntl.flock`, which is
   Linux-only. We need `msvcrt.locking` or an atomic `os.replace` + temp-file protocol instead.
8. Whether to commit a reusable digested artifact per task under `docs/assets/` so CI and demos need
   no VLM calls.
9. Interaction with `--affordance-points` - does the demo supply the contact point, or only bias it?
10. Whether to adopt GPT-Policy's sliding live-image window plus a text-only execution record, which
    is what lets demonstration images survive a long episode.
11. Whether a Show-Harness-style discrete action vocabulary is worth offering as an alternative
    low-level interface - orthogonal to this feature, but the same repo answers both questions.
12. Motion proxy for keyframe selection - MediaPipe hands adds a 7.8 MB CPU dependency; for robot
    rollouts, TCP speed and gripper transitions are already logged and need nothing new.
13. Whether to adopt one-image-per-call chaining globally (it is the strongest single finding in the
    survey) or only inside the demo digest, given the extra round trips.
14. Whether demo-video audio exists for our clips at all, which decides whether the ASR narration
    channel is worth a line of code.

---

## 11. Non-goals

- No training, no fine-tuning, no gradient updates.
- No trajectory-level mimicry of the demonstration - plan-level replication only (order, grasp part,
  destination), exactly as `video_ref` scopes it.
- No *new* perception models in the phases above. The mark-overlay work reuses the segmentation and
  XMem masks the repo already produces; CoTracker, DINO features, depth-based affordance memories
  and video generation (sections 6.2 and 6.3) are explicitly deferred.
- No change to the existing prompts when the feature is off - the byte-identical rule.

---

## 12. Sources

Reference docs in this repo: [`../../models/gpt_policy.md`](../../models/gpt_policy.md),
[`../../models/show_harness.md`](../../models/show_harness.md).

Papers cited above, by arXiv ID (all verified against the arXiv API):
SeeDo 2410.08792 - Video-to-BT 2509.16611 - Vid2Robot 2403.12943 - Gen2Act 2409.16283 -
Track2Act 2405.01527 - Set-of-Mark 2310.11441 - PIVOT 2402.07872 - MOKA 2403.03174 -
CoPa 2403.08248 - RoboPoint 2406.10721 - HAMSTER 2502.05485 - RT-Trajectory 2311.01977 -
Wonderful Team 2407.19094 - OmniManip 2501.03841 - SAIL 2603.08269 - ViLA 2311.17842 -
VoxPoser 2307.05973 - ReKep 2409.01652 - Code as Policies 2209.07753 - KAT 2403.19578 -
ICRT 2408.15980 - Instant Policy 2411.12633 - ATM 2401.00025 - ContextFlow 2609.06852 -
SynthICL 2606.08154 - GPT-Policy 2609.19138 - Show-Harness 2609.10522 - and this repo's upstream
paper, Language Models as Zero-Shot Trajectory Generators, 2310.11604.
RAM, SKIL-H, ZeroMimic, VL-MP, Code-as-Monitor, OWMM-Agent, Afford2Act and ManipLLM are cited by
name only.
