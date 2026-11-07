# Navigation - VLN, semantic / object-goal navigation, navigation foundation models

Research notes for adding semantic navigation to the VLM robot ("go to the kitchen", indoor and
outdoor, with or without a map). Implementation plan: [../plans/semantic_navigation.md](../plans/semantic_navigation.md).
Related: [mobile_manip.md](mobile_manip.md) (base + arm), [slam_mapping.md](slam_mapping.md)
(concurrent mapping and localization).

Compiled 2026-11-08 from arXiv (titles and abstracts checked with `docs/research_tools/paper_research.js arxiv`),
the @AIPaperSlop YouTube channel (`yt-search` / `yt-watch`), GitHub (stars on 2026-11-08), project
pages and READMEs.

How sure each claim is:
- No tag: checked against the abstract, README or project page.
- **[unverified]**: from memory. Check it before you rely on it.
- `n/a`: not found.

The Embodied Arena navigation leaderboard (https://www.embodied-arena.com/#/leaderboard/navigation/task)
is a JS-only page that we could not scrape. Its ranking ("InternVLA-N1 1st, Gemini 2.5 Pro 12th")
comes from the user's notes.

---

## 0. TL;DR for our stack

1. **Zero-shot, agentic, map-optional**. This fits planner → LLM code → reviewer. It is the
   same idea as **SuperNav** (2026): a general MLLM decides, and navigation *tools* execute,
   with **no nav fine-tuning**.
   - Tools: frontier exploration with a VLM value map (**VLFM**), commonsense room–object
     priors (**ESC / L3MVN / SG-Nav**), and a visual-point interface ("go to this pixel",
     as in **PIVOT / VLMnav / CoNVOI / SuperNav**).
   - The VLM reviewer verifies arrival.
2. **With a map from a prior tour**. A **Mobility VLA**-style topological graph of tour
   keyframes, where a long-context VLM picks the goal frame. Or a **HOV-SG / ConceptGraphs**
   room/object scene graph ("kitchen" = room node). Classical A* / graph search does the motion.
3. **Specialist models as optional tools / ablations**, served from `gpu_server`:
   - **InternVLA-N1** (InternNav, MIT, wheeled embodiment, >30 Hz local system)
   - **StreamVLN / NaVILA / Uni-NaVid** (open weights)
   - **NoMaD / ViNT** (MIT) for mapless exploration or image goals
   - **CityWalker** (Apache-2.0) for outdoor point goals
   - Newer 2026 generalists: **Qwen-RobotNav** (agentic task modes) and **LightNav-0**
     (compact, pointing-based).
4. **Embodiment gap**. Almost every model was trained on a Go2 / Jackal / Stretch / Spot.
   Always wrap model outputs (pixel goals, waypoints) with our own depth-guarded base
   controller ([../plans/mobile_base.md](../plans/mobile_base.md)).

---

## 1. Concepts

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| VLN vs ObjectNav vs instruction nav | VLN: follow a route description. ObjectNav: find an object category. Instruction / demand nav: free-form goals ("I'm hungry") | "Go to the kitchen" is room-goal / semantic nav. It is closer to ObjectNav than to VLN | VLN-CE, VLFM, InstructNav |
| VLN-CE (continuous) | VLN in continuous 3D with low-level actions (forward 0.25 m, turn 15°), no nav-graph, no oracle localization | The benchmark setting closest to a real base with RGB-D + odometry | VLN-CE |
| Frontier | Boundary between known-free and unknown cells in an occupancy map | Sensor-only (depth + odometry) "where to go next" when there is no map | Yamauchi '97, CoW, VLFM, ApexNav |
| Language-grounded value map | Top-down grid. Each cell stores a VLM image-text score for the goal (BLIP-2 / CLIP). The highest-value frontier wins | A scalar "kitchen-ness" the planner can query; no training needed | VLFM, InstructNav, WMNav |
| Commonsense room–object priors | An LLM states where things usually are (fridge → kitchen, kitchen near the dining room) | The heart of "go to the kitchen" without a map | ESC, L3MVN, SG-Nav, UniGoal |
| Topological map | Graph of keyframes/poses with traversability edges. Localize and plan by image similarity | "With a map" mode that tolerates odometry drift; cheap storage | ViNT, LM-Nav, Mobility VLA |
| Long-context tour reasoning | Feed a whole tour video plus the query to a long-context VLM; it returns the goal frame | One call answers "go to the kitchen / where do I return this" | Mobility VLA |
| Open-vocabulary 3D scene graph | Floor > room > object nodes, each with CLIP features and captions | Room goals become graph queries. Fits a prior tour | ConceptGraphs, HOV-SG, SG-Nav, Hydra |
| Open-vocabulary map features | CLIP/LSeg pixel features fused into a 2D/3D grid and queried by text | "Go between the sofa and the TV"; the LLM writes code over a map API | VLMaps, NLMap, OpenScene |
| Video-LLM nav agent | Streaming RGB + instruction → video VLM → next action(s) | Pixel-only, no map. Fits sim2real | NaVid, Uni-NaVid, NaVILA, StreamVLN |
| Dual-system (slow System 2 + fast System 1) | A slow VLM predicts a pixel goal or latent plan; a fast diffusion/RL policy outputs a trajectory at >30 Hz | Same split as our LLM planner + base controller | InternVLA-N1, NaVILA |
| Language mid-level actions | The VLM outputs text such as "move forward 75 cm"; a separate controller executes it | Maps 1:1 onto our base primitives | NaVILA |
| Agent harness with nav tools | An MLLM with skills, tools, task-progress / context management and a visual-point interface. No nav fine-tuning | **Our architecture.** SuperNav shows it beats fine-tuned baselines on instance, multi-object and demand tasks | SuperNav, Qwen-RobotNav (agentic task modes) |
| Visual prompting for action | Overlay candidate points, arrows or numbers on the image; the VLM picks one | Uses our hosted VLM directly; no map needed | PIVOT, VLMnav, CoNVOI, TopV-Nav |
| Goal-conditioned nav foundation model | Policy trained on many robots to reach image goals from RGB history | Pixel-only local policy and explorer | GNM, ViNT, NoMaD |
| Goal masking | One diffusion policy. With the goal masked it explores; with a goal it reaches it | Exploration before the target is seen | NoMaD |
| Lifelong instance memory | Per-instance multi-view features kept across goals | Repeated errands get faster | GOAT, DynaMem |
| Success / SPL | SR = fraction that stops within d of the goal. SPL = SR × shortest / max(actual, shortest) | Report both SR and SPL | all |

## 2. Techniques

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Frontier selection by VLM value | Choose the frontier with the highest value-map score; replan every few steps | The simplest proven zero-shot baseline (real Spot robot) | VLFM |
| LLM scoring of frontiers | Describe the objects near each frontier in text; the LLM picks one | Easy to plug into the planner, but loses spatial detail | L3MVN, ESC |
| Scene-graph CoT + re-perception | The LLM walks room → group → object nodes. Detected goals are re-checked up close | Explainable; cuts false positives | SG-Nav, UniGoal |
| Target-centric semantic fusion | Long-term memory of the target and its look-alikes; fuse confidences before stopping | Robust stop decision | ApexNav |
| Semantic ↔ geometric exploration switch | Fall back to pure geometric frontiers when semantic cues are weak | Avoids aimless wandering in corridors and outdoors | ApexNav |
| Top-view map as visual prompt | Render the semantic map as an image; the MLLM picks a target point; dynamic zoom | Our VLM can reason over a map image | TopV-Nav |
| World-model / curiosity map | The VLM predicts an action's outcome and compares it with the real observation | Reduces hallucinated directions | WMNav |
| Multi-source value maps + dynamic chain-of-navigation | The LLM plans stages; action / semantic / history / trajectory value maps are fused | One system for ObjectNav, VLN and demand nav | InstructNav |
| Goal-frame retrieval + graph search | The VLM finds the goal frame in the tour; Dijkstra on the graph; local waypoint policy | Reference for prior-tour mode | Mobility VLA |
| LLM landmarks + CLIP grounding + graph search | The LLM parses landmarks; CLIP scores graph nodes; a planner picks the path | Zero-shot map-based route following | LM-Nav |
| LLM writes code over a map API | `pos = map.find("sofa"); go_between(...)` | Same as our "LLM writes trajectory code" | VLMaps, NLMap |
| Numbered free-space marking | Number free-space regions on the RGB; the VLM picks numbers → path | Zero-shot local nav with a hosted VLM, indoor and outdoor | CoNVOI, VLMnav |
| Pixel-goal → 3D waypoint | The VLM outputs a pixel; depth lifts it to a 3D point; the base controller drives there | Our existing `convert_2d_point_to_3d_world` already does the lift | InternVLA-N1, SuperNav, LightNav-0 |
| Dual-channel pointing + action tokenizer | The VLM points (embodiment-agnostic intent); an RVQ action tokenizer maps it to trajectories | Clean split between intent and embodiment | LightNav-0 |
| Parameterised observation interface | Task mode + token budget + per-camera weights switchable at inference | An upper planner can switch nav behaviour per subtask | Qwen-RobotNav |
| SlowFast context / token pruning | Sliding-window KV cache plus a pruned long-term memory | Bounded latency on long episodes | StreamVLN, NaVid, Uni-NaVid |
| Diffusion waypoint head with goal masking | DDPM over waypoint sequences | Smooth, multimodal, collision-aware local paths | NoMaD, InternVLA-N1 System 1 |
| Web / tour video relabeling | VLM/VO-labelled walking or touring videos → training data | Data scaling without robots | NaVILA, CityWalker, LeLaN, MBRA |
| Imperative learning with geometric cost fields | Fine-tune a VLM against a differentiable depth cost instead of labelled trajectories | Depth-aware collision-free paths, <1 % of parameters (LoRA) | Dear LLaVA, Please Drive |
| Model-based look-ahead + model-free RL | Imagine future observations; combine with an RL policy | Historical root of planned-ahead VLN | Look Before You Leap |

## 3. Edge cases

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Goal absent / unreachable | No kitchen on this floor, or the door is locked | Needs a budget- or coverage-based "not found" and a planner replan | VLFM, UniGoal (blacklist), ApexNav |
| False-positive detection | A look-alike triggers an early stop (sofa vs bed, kitchenette vs kitchen) | The reviewer VLM must verify at close range | SG-Nav, ApexNav, UniGoal |
| Closed doors | Occupancy marks the door as a wall, so the room is unreachable | **Gap in the literature.** We can hand off to the arm (our door-opening skill) | n/a (our engineering) |
| Oscillation / loops | The agent flips between two frontiers | Frontier commitment, history map, blacklist | VLFM, InstructNav, UniGoal |
| Odometry drift | The metric map warps over long runs | Topological maps tolerate it; SLAM + loop closure | Mobility VLA, VGGT-SLAM, [slam_mapping.md](slam_mapping.md) |
| Stale tour map | Furniture moved, people around | Re-verify on arrival; lifelong update | GOAT, DynaMem, Mobility VLA |
| Weak semantic cues | Corridors and outdoors give a flat value map | Geometric fallback | ApexNav |
| Hallucinated / unreachable direction | The VLM points at a wall | Restrict choices to navigable candidates | VLMnav, PIVOT, WMNav, CoNVOI |
| Instance ambiguity | "The kitchen" vs "a kitchen"; two sinks | Instance memory; ask the user for clarification | GOAT, HOV-SG, VL-LN Bench |
| Multi-floor | Stairs / elevator | A wheeled base can't climb. Floor nodes only; out of scope | HOV-SG |
| Depth failure | Glass, sunlight (IR), black surfaces | Outdoors needs stereo or monocular depth; pixel-only fallbacks | VL-Explore, Dear LLaVA, CityWalker |
| VLM latency | 3–10 s per decision while the robot moves | Async: the VLM runs at keyframes, the local controller stays reactive | InternVLA-N1, StreamVLN, Mobility VLA |
| Dynamic obstacles / people | Static-map assumptions break | Fast reactive layer under the slow VLM | InternVLA-N1, CityWalker |
| Embodiment mismatch | Camera height, FoV and kinematics differ from the training robot | Wrap waypoints with our own controller; fine-tune or prompt-adapt | GNM, ViNT, NavFoM |
| Spatial / metric instructions | "Stop 1 m left of the chair" | VLMs are weak at this. Use the depth-based metric tools | NavSpace, TVRBench |
| Active viewpoint reproduction | Move until the view matches a target image | Even frontier models score ≤12 % zero-shot. Body translation is much harder than rotation | Where to Look / TVRBench |

## 4. Benchmarks

| Name | Task | Scenes / sim | Metrics | Link | Notes |
|---|---|---|---|---|---|
| R2R-CE / RxR-CE (VLN-CE) | Continuous instruction following | MP3D in Habitat | SR, SPL, NE, OSR, nDTW, SDTW | https://github.com/jacobkrantz/VLN-CE | Main VLN leaderboard. Old habitat-lab 0.1.7 |
| R2R / RxR / REVERIE / SOON (discrete) | Graph VLN / remote object grounding | MP3D | SR, SPL, RGS | https://github.com/peteanderson80/Matterport3DSimulator | LLM-agent VLN (NavGPT, MapGPT) |
| ObjectNav (Habitat challenge) | Find an object category | HM3D-Semantics v1/v2, MP3D, Gibson | SR, SPL, SoftSPL, DTG | https://aihabitat.org/challenge/ | Main zero-shot table (VLFM, SG-Nav, ApexNav) |
| HM3D-OVON | Open-vocabulary ObjectNav (379 categories) | HM3D | SR, SPL | arXiv 2409.14296 | Closest to free-form "find X" |
| GOAT-Bench | Multimodal lifelong goals (category, image, text) | HM3D | SR, SPL | arXiv 2404.06609 | Multi-goal sessions |
| RoboTHOR ObjectNav / Pasture | ObjectNav in a sim with real replicas / uncommon objects | AI2-THOR | SR, SPL | arXiv 2203.10421 | Sim2real flavour |
| DDN (demand-driven nav) | Goals expressed as human needs | AI2-THOR / HM3D | SR, SPL | InstructNav | "I'm hungry" → kitchen |
| VLN-PE | Physically realistic VLN for humanoid, quadruped and **wheeled** robots | Isaac Sim | SR, SPL | InternNav README, arXiv 2507.13019 | Includes a wheeled embodiment |
| NavSpace | Spatial-intelligence instructions | Habitat | SR | arXiv 2510.08173 | 1,228 pairs, 6 categories |
| TVRBench | Reproduce a target viewpoint by active motion | Indoor sim | Success | arXiv 2606.01247, https://github.com/aim-uofa/TVRBench | SFT raises a 9B model from ~8 % to 50.8 % |
| BEHAVIOR-Spatial | Spatial QA + grounding + action (14 manipulation + 10 nav tasks) | BEHAVIOR-1K | Success, QA accuracy | arXiv 2610.34051 | Spatial-language grounding |
| ESI-Bench | Embodied spatial intelligence via perception–action loop | OmniGibson | Accuracy | arXiv 2605.18746 | Active evidence gathering |
| Embodied Arena (navigation) | Aggregated leaderboard | various | various | https://www.embodied-arena.com/#/leaderboard/navigation/task | InternVLA-N1 #1, Gemini 2.5 Pro #12 per user notes |
| Real-world protocols | Mobility VLA 836 m² office; GOAT 9 homes / 675 goals; CityWalker urban | real | SR | papers | The only real proof of sim2real |

### 4.1 Live leaderboards and current state of the art (checked 2026-11-08)

Searched: HF Spaces/papers, EvalAI, Embodied Arena, EmbodiedBench, aihabitat.org.
- **Readable:** EmbodiedBench challenge (CVPR'26).
- **Not readable:**
  - Embodied Arena (JS-only page).
  - Habitat/VLN-CE EvalAI boards (API returned 406).
  - `EmbodiedCity/EmbodiedCity-Leaderboard` HF Space (asleep).
- No public ObjectNav / OVON / VLN-CE board is maintained any more. The SOTA below is
  **self-reported in 2026 papers** (abstract-level). arXiv IDs are verified.

| Leaderboard / benchmark | Top entries (score) | Kind | Source |
|---|---|---|---|
| EmbodiedBench challenge, held-out (EB-ALFRED / EB-Nav, <10B open VLMs, no APIs) | NJU-LAMDA-SZ 67.3 / 37.0 (avg 52.2); Ideal-Embody 75.3 / 22.0; Team233 44.7 / 28.0 | team VLM agents (methods not published) | https://embodiedbench.github.io/challenge.html, arXiv 2502.09560 |
| R2R-CE / RxR-CE | **Robostral Navigate**: SR 77.4 / 75.1 (monocular RGB, 8B, RL) | trained | arXiv 2607.20785 |
| R2R-CE / RxR-CE | **MemVLN** (streaming episodic memory, claims SOTA on most metrics); **MOON** (plug-in "forget on deviation") | trained | 2610.35066, 2610.33546 |
| GOAT-Bench / IR2R-CE | **NavHarness**: GOAT 83.7 s-SR / 36.9 e-SR; IR2R-CE 85.9 s-SR | **training-free agent harness** | 2609.34276, https://github.com/billzhao1030/NavHarness |
| GOAT-Bench | **STEGNav**: 66.3 SR / 39.7 SPL (also HM3D v1 64.0, v2 69.4) | training-free VLM + event graph | 2608.28279 |
| HM3D-OVON | **SoftNav**: SR 74.2 / 68.3 / 66.7 (val seen / synonyms / unseen); GOAT 67.2 | frozen VLM + 3D scene tokens (~17M trainable) | 2607.14586 |
| Zero-shot ObjectNav HM3D v1 / MP3D | **Hierarchical Fast-Slow ReAct**: 68.75 / 47.29 SR | zero-shot (VLM over a value-map controller) | 2608.09816 |
| Urban / POI + indoor | **ABot-N1**: POI arrival 77.3 %; 95.4 % indoor / 92.9 % outdoor SR (new benchmarks) | trained, slow-fast (pixel goal → waypoints) | 2607.10383 |
| ImageNav (zero-shot) | **MG-Nav**: sparse spatial memory graph + navigation foundation policy | zero-shot | 2511.22609 |

**Takeaways for our plan:**
1. Training-free harnesses (NavHarness, STEGNav, Fast-Slow ReAct, SuperNav) lead the
   multi-goal and lifelong boards. This validates the agentic design in
   [semantic_navigation.md](../plans/semantic_navigation.md).
2. Fast-Slow ReAct is the closest analogue to our N2 design: a VLM deliberates over a value-map
   frontier controller.
3. Pixel-goal slow-fast models (ABot-N1, Robostral, InternVLA-N1) are the best candidates for the
   specialist `--nav-policy`.
4. Event or episodic graph memory (STEGNav, MemVLN) is the next step after our TopoGraph (N4+).

Not yet verified:
- **Benchmarks:** EvoNav-Bench (2609.08292; lifelong nav in evolving environments) and VLNVerse
  (2512.19021).
- **Candidate methods:** OVSegDT (2508.11479), LagMemo (2510.24118), S2E (2507.22028) and
  OpenFMNav (2402.10670).

## 5. Datasets

| Name | Content | Why it matters | Link / source |
|---|---|---|---|
| HM3D / HM3D-Semantics | 1,000 scanned buildings (+ semantic labels) | Main ObjectNav scenes; house-scale assets | arXiv 2109.08238, https://aihabitat.org/datasets/hm3d |
| MP3D | 90 buildings | VLN / ObjectNav | https://niessner.github.io/Matterport |
| HSSD-200 | Synthetic, curated homes (ObjectNav) | Clean, editable houses; HomeRobot OVMM | arXiv 2306.11290 |
| ReplicaCAD | Interactive apartment | Habitat 3.0 / ManiSkill-HAB | Habitat |
| ProcTHOR-10k | 10k procedural houses | Training at scale | arXiv 2206.06994 |
| Gibson | Scanned homes | Legacy ObjectNav | StanfordVL |
| R2R / RxR / EnvDrop / ScaleVLN | Instruction–path pairs (ScaleVLN ~150k episodes on HM3D) | VLN fine-tuning | StreamVLN, NaVILA |
| InternData-N1 | >50M egocentric images, >3,000 scenes, 4,839 km | Largest open nav dataset | https://huggingface.co/datasets/InternRobotics/InternData-N1 |
| Uni-NaVid mix | 3.6M samples (VLN, ObjectNav, EQA, following) | Multi-task nav | arXiv 2412.06224 |
| GNM / ViNT mixture | RECON, SCAND, GoStanford, SACSoN, TartanDrive… hundreds of hours | Local policy data | visualnav-transformer |
| LeLaN data | >130 h indoor/outdoor robot + YouTube video, VLM-labelled | Language object-nav | arXiv 2410.03603 |
| CityWalker videos | Thousands of hours of city walking | Outdoor | arXiv 2411.17820 |
| Qwen-RobotNav corpus | 15.6M samples, multi-task + VL co-training | Scaling evidence | arXiv 2606.18112 |
| LightNav-0 corpus | 2K+ scenes, 4K+ h of nav data | Compact generalist | arXiv 2608.30935 |
| Self-collected tours | Phone or robot walkthrough + labels | Needed for prior-tour mode | Mobility VLA, OK-Robot |

Simulator choice, metrics definitions and the full sim comparison:
[../plans/semantic_navigation.md §3 (decision 6, Simulator)](../plans/semantic_navigation.md#3-high-level-approach) and
[mobile_manip.md §Simulators](mobile_manip.md#simulators).

---

## 6. Papers

### 6.1 Agentic / zero-shot VLM navigation (closest to our architecture)

* **SuperNav: An Agentic Navigation System for Any Task in Any Scene** (arXiv 2610.12126, Oct 2026)
  ) Goal: TLDR - request + egocentric images --> pretrained MLLM in an agent harness (Navigation
    Skills, agent-oriented Tools, task-progress / context management) --> **visual-point interface**
    (destination picked directly in the image) --> navigation tools execute, feedback revises the decision.
  ) Problem: MLLMs fine-tuned to predict nav actions are tied to the coverage of their nav data.
  ) Tags: Agentic, Zeroshot (no nav fine-tuning), MLLM, Skills, Tools, Visual pointing, Real robot (quadruped)
  ) Project: https://zju3dv.github.io/SuperNav/ / Paper: https://arxiv.org/abs/2610.12126 /
    Code: https://github.com/zju3dv/SuperNav (233★, created 2026-10) / Video: AIPaperSlop `nHi9ewdtsmQ`
  ) Results: beats 4 baselines on instance-level, multi-object and demand-driven tasks; category-level eval on HM3D.
  ) Relevance: **Blueprint for us.** It mirrors our skills (`prompts/skills`) + tools (`get_exec_locals`)
    + reviewer. Port its tool set and visual-point interface first.

* **Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs** (arXiv 2407.07775, 2024, Google DeepMind)
  ) Goal: TLDR - demo tour video (places verbally highlighted) + multimodal instruction (text, image,
    sketch, audio) --> Gemini 1.5 Pro long context finds the goal frame --> topological graph from
    tour frames --> waypoint policy.
  ) Problem: MINT - navigation from a prior tour with multimodal instructions ("take me somewhere to draw things").
  ) Tags: Prior-tour map, Topological graph, Long-context VLM, Real robot
  ) Paper: https://arxiv.org/abs/2407.07775 / Video: https://www.youtube.com/watch?v=ACXxxo3awz8 / Code: n/a
  ) Results: high end-to-end success in an 836 m² office; ~86–90 % [unverified].
  ) Relevance: reference for **N4 prior-tour mode**. Reproducible with any long-context VLM API.

* **VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation** (ICRA 2024)
  ) Goal: TLDR - RGB-D + odometry + target text --> depth occupancy + frontiers; BLIP-2 ITM value map
    --> best frontier --> PointNav policy; detector (GroundingDINO / YOLOv7 + MobileSAM) triggers the final approach.
  ) Tags: ObjectNav, Zeroshot, Frontier, Value map, Real robot (Spot)
  ) Paper: https://arxiv.org/abs/2312.03275 / Code: https://github.com/bdaiinstitute/vlfm (≈834★, MIT)
  ) Results: SOTA zero-shot SPL on Gibson / HM3D / MP3D at release; HM3D ≈ 52.5 SR / 30.4 SPL [unverified].
  ) Relevance: **N2 baseline**. It uses only RGB-D + odometry, which matches our Sim2Real rule.

* **VL-Explore: Zero-shot Vision-Language Exploration and Target Discovery by Mobile Robots** (arXiv 2502.08791, 2025)
  ) Goal: TLDR - monocular RGB --> CLIP similarity drives exploration + obstacle avoidance --> real-time target discovery, no map.
  ) Tags: Zeroshot, CLIP, Monocular, UGV / Code: n/a
  ) Results: beats map-traversal baselines (qualitative); real Open Rover UGV.
  ) Relevance: minimal-sensor outdoor option.

* **InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment** (CoRL 2024)
  ) Goal: TLDR - RGB-D + free-form instruction --> Dynamic Chain-of-Navigation (GPT-4V) + multi-source
    value maps (action / semantic / history / trajectory / intuition) --> waypoints.
  ) Paper: https://arxiv.org/abs/2406.04882 / Code: https://github.com/LYX0501/InstructNav (~216★, no license)
  ) Results: first zero-shot R2R-CE; +10.48 % over zero-shot ObjectNav SOTA; +86.34 % on DDN.
  ) Relevance: one system for ObjectNav + VLN + demand nav. Closest to planner → LLM → tools.

* **SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot ObjectNav** (NeurIPS 2024)
  ) Goal: TLDR - RGB-D --> online hierarchical scene graph (object / group / room) --> LLM hierarchical CoT picks a frontier; re-perception verifies goals.
  ) Paper: https://arxiv.org/abs/2410.08189 / Code: https://github.com/bagh2178/SG-Nav (~358★, MIT)
  ) Results: >10 % SR over prior zero-shot on MP3D / HM3D / RoboTHOR; first zero-shot to beat supervised on MP3D.
  ) Relevance: room nodes make "kitchen" a direct query. Re-perception handles false positives.

* **UniGoal: Towards Universal Zero-shot Goal-oriented Navigation** (CVPR 2025)
  ) Goal: TLDR - goal graph (category / image / text) + online scene graph --> graph matching + LLM --> exploration goal, with blacklist + verification.
  ) Paper: https://arxiv.org/abs/2503.10630 / Code: https://github.com/bagh2178/UniGoal (~373★, MIT)
  ) Relevance: text goals like "the kitchen with the red fridge". The blacklist breaks loops.

* **ESC: Exploration with Soft Commonsense Constraints for Zero-shot ObjectNav** (ICML 2023)
  ) Goal: TLDR - GLIP detections + LLM room/object commonsense as soft logic --> frontier scoring.
  ) Paper: https://arxiv.org/abs/2301.13166 / Code: n/a
  ) Results: +288 % relative SR over CoW on MP3D.
  ) Relevance: source of the "kitchen has a fridge / stove" prior.

* **L3MVN: Leveraging LLMs for Visual Target Navigation** (IROS 2023)
  ) Goal: TLDR - semantic map → frontiers described in text → LLM picks one.
  ) Paper: https://arxiv.org/abs/2304.05501 / Code: https://github.com/ybgdgh/L3MVN (~171★)
  ) Relevance: the easiest LLM integration, but text loses geometry.

* **CoWs on Pasture (CLIP on Wheels)** (CVPR 2023)
  ) Goal: TLDR - frontier exploration + CLIP / OWL-ViT open-vocab localization; Pasture benchmark.
  ) Paper: https://arxiv.org/abs/2203.10421 / Code: https://github.com/real-stanford/cow
  ) Relevance: the original "frontier + open-vocab detector" template.

* **ApexNav: Adaptive Exploration with Target-centric Semantic Fusion** (RA-L 2025)
  ) Goal: TLDR - RGB-D --> switch between semantic and geometric frontiers by cue strength + target / look-alike memory fusion.
  ) Paper: https://arxiv.org/abs/2504.14478 / Code: https://github.com/Robotics-STAR-Lab/ApexNav (~480★, **GPL-3.0**)
  ) Results: SOTA SR + SPL on HM3D v1/v2 and MP3D; real-world validation.
  ) Relevance: copy its fallback + fusion ideas. The GPL license means don't vendor the code.

* **WMNav: Integrating VLMs into World Models for ObjectNav** (IROS 2025)
  ) Goal: TLDR - the VLM predicts outcomes; a curiosity value map + two-stage proposer (explore → localize).
  ) Paper: https://arxiv.org/abs/2503.02247 / Code: https://github.com/B0B8K1ng/WMNavigation (~184★)
  ) Results: +3.2 SR / +3.2 SPL on HM3D; +13.5 SR on MP3D over zero-shot SOTA.
  ) Relevance: predict-then-verify loop, the same pattern as our reviewer.

* **TopV-Nav: Top-View Spatial Reasoning of MLLM for Zero-shot ObjectNav** (CVPR 2025)
  ) Goal: TLDR - the semantic top-view map is rendered as an image; the MLLM picks a target with dynamic zoom.
  ) Paper: https://arxiv.org/abs/2411.16425 / Code: https://github.com/LinqingZhong/TopV-Nav
  ) Relevance: show the occupancy / semantic map image to our VLM.

* **VLMnav: End-to-End Navigation with VLMs: Spatial Reasoning as QA** (arXiv 2411.05755, 2024)
  ) Goal: TLDR - RGB-D → navigable candidate directions drawn and numbered on the image → the VLM picks a number → move.
  ) Code: n/a / Relevance: minimal zero-shot local policy (PIVOT-style) for our hosted VLM.

* **PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs** (ICML 2024)
  ) Goal: TLDR - sample candidate actions → draw them on the image → the VLM selects → refit → iterate.
  ) Paper: https://arxiv.org/abs/2402.07872 / Relevance: a generic "VLM picks a heading / point" template.

* **CoNVOI: Context-aware Navigation using VLMs in Outdoor and Indoor Environments** (IROS 2024)
  ) Goal: TLDR - RGB + local occupancy → numbered free-space marks → VLM (GPT-4V) picks per context
    rule ("stay on the pavement") → reference path → extrapolate to cut VLM calls.
  ) Paper: https://arxiv.org/abs/2403.15637 / Code: n/a
  ) Relevance: prompt-only local nav indoors **and outdoors** with our hosted VLM.

* **NavGPT** (AAAI 2024) / **NavGPT-2** (ECCV 2024) / **MapGPT** (ACL 2024)
  ) Goal: TLDR - LLM / VLM agents for discrete R2R. NavGPT puts text descriptions of the views
    into GPT. NavGPT-2 fine-tunes a VLM + topological policy. MapGPT serializes a text topological
    map into the prompt with adaptive planning.
  ) Papers: arXiv 2305.16986, 2407.12366, 2401.07314 / Code: GengzeZhou/NavGPT (~356★), GengzeZhou/NavGPT-2 (~258★, MIT), chen-judge/MapGPT (~138★)
  ) Relevance: how to serialize a map / history for the LLM. The discrete graph is not directly usable.

* **LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action** (CoRL 2022)
  ) Goal: TLDR - instruction → GPT-3 landmarks → CLIP grounding on a topological graph of prior images → ViNG graph search → GNM policy.
  ) Paper: https://arxiv.org/abs/2207.04429 / Code: n/a
  ) Relevance: zero-shot map-based route following. The original "LLM + VLM + nav model" stack.

* **"Dear LLaVA, Please Drive": A Depth-Aware VL Agent for Closed-Loop Robotic Control** (arXiv 2609.22925, Sep 2026)
  ) Goal: TLDR - language + stereo depth → LLaVA with task LoRAs (semantic target selection + path planning) → collision-free paths.
  ) Method: imperative learning against differentiable geometric cost fields (no labelled trajectories), <1 % of parameters tuned.
  ) Results: competitive SPL in unseen environments; sim2real without real fine-tuning. Code: n/a.
  ) Relevance: a depth-only way to make a VLM output safe paths. Very new.

* **Look Before You Leap: Bridging Model-Free and Model-Based RL for Planned-Ahead VLN** (ECCV 2018)
  ) Goal: TLDR - instruction + panorama → a look-ahead environment model imagines future
    observations, combined with model-free RL → discrete R2R actions.
  ) Paper: https://arxiv.org/abs/1803.07729 / Code: n/a
  ) Relevance: historical (VLM + RL / planning lineage). The look-ahead idea returns in WMNav
    (world-model prediction) and world-action models.

### 6.2 Navigation foundation models / VLN VLAs (candidate specialist tools)

* **InternVLA-N1 / DualVLN: "Ground Slow, Move Fast" dual-system VLN foundation model** (arXiv 2512.08186, Dec 2025)
  ) Goal: TLDR - instruction + RGB(-D) → System 2 VLM predicts a pixel goal + latent plan tokens → System 1 diffusion-transformer policy → trajectories at >30 Hz.
  ) Problem: end-to-end VLM-to-discrete-action pipelines are slow and jerky, and weak against dynamic obstacles.
  ) Tags: VLN-CE, VLN-PE, Dual-system, Diffusion policy, Wheeled / Quadruped / Humanoid, Zero-shot real
  ) Project: https://internrobotics.github.io/internvla-n1.github.io / Code: https://github.com/InternRobotics/InternNav (1,168★, MIT) /
    Models: https://huggingface.co/InternRobotics/InternVLA-N1 / Data: InternData-N1
  ) Method: train System 2 (pixel goals) and System 1 separately, then freeze System 2 and fine-tune latent-plan tokens with System 1 asynchronously.
  ) Results: "SOTA on 6 benchmarks, +3 % to 28 %"; RGB-only R2R val-unseen SR 55.4 / SPL 52.1. Embodied Arena #1 (user notes).
  ) Hardware: Go2 deployment code; wheeled supported.
  ) Relevance: **the best open specialist** for our `nav_policy=internvla_n1` ablation. Its System 1 can consume pixel goal + depth.

* **Qwen-RobotNav: A Scalable Navigation Model Designed for an Agentic Navigation System** (arXiv 2606.18112, Jun 2026)
  ) Goal: TLDR - visual stream + task mode + observation parameters (token budget, per-camera
    weights) → Qwen-based nav model (2B–8B) → actions. An upper-level planner switches task mode mid-episode.
  ) Data: 15.6M samples. Co-training with vision-language data prevents collapse into a reactive mapper.
  ) Results: new SOTA across major nav benchmarks; zero-shot on real robots. Code / weights: n/a [check].
  ) Relevance: designed to sit *under* an agentic planner, the same role as our nav tool.

* **LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation** (arXiv 2608.30935, Aug 2026)
  ) Goal: TLDR - compact VLM + dual-channel pointing (spatial intent) + residual-VQ action tokenizer
    (embodiment-specific trajectories) → instruction following, open-vocabulary ObjectNav and tracking in one model.
  ) Training: ER mid-training → SFT → RL. 2K+ scenes, 4K+ hours.
  ) Results: SOTA monocular SR on all 10 public nav sim settings; zero-shot across embodiments.
  ) Relevance: shows a small VLM with *pointing* is enough. Pointing matches our 2D → 3D lift.

* **StreamVLN: Streaming VLN via SlowFast Context Modeling** (arXiv 2507.05240, 2025)
  ) Goal: TLDR - streaming RGB + instruction → LLaVA-Video (Qwen2) multi-turn with KV-cache reuse → actions.
  ) Code: https://github.com/InternRobotics/StreamVLN (~642★, license unclear) / Models: on HF
  ) Results: R2R-CE val-unseen SR 56.4 / SPL 50.2; RxR SR 54.4 / SPL 45.4 (README). Go2 deployment.
  ) Relevance: strong open VLN-CE model; check the license.

* **NaVILA: Legged Robot VLA for Navigation** (RSS 2025)
  ) Goal: TLDR - instruction + frames → NVILA (SigLIP + Llama3-8B) → text mid-level actions ("move forward 75 cm") → RL locomotion.
  ) Paper: https://arxiv.org/abs/2412.04453 / Code: https://github.com/AnjieCheng/NaVILA (~733★, Apache-2.0) / Models: HF
  ) Data: R2R, RxR, EnvDrop, ScanQA, YouTube touring videos. Results: R2R-CE SR ≈ 54 [unverified].
  ) Relevance: its text action interface = our `move_forward / turn_left` tools.

* **NaVid** (RSS 2024) / **Uni-NaVid** (RSS 2025)
  ) Goal: TLDR - RGB-only video VLM → next action. Uni-NaVid unifies VLN, ObjectNav, EQA and human following (3.6M samples, online token merging).
  ) Papers: arXiv 2402.15852, 2412.06224 / Code: jzhzhang/NaVid-VLN-CE (~460★, MIT), jzhzhang/Uni-NaVid (~364★, MIT)
  ) Relevance: RGB-only (no depth, no map) proves pixel-only navigation works. Uni-NaVid is a generalist "go / find / follow".

* **NavFoM: Embodied Navigation Foundation Model** (arXiv 2509.12129, 2025)
  ) Goal: TLDR - multi-camera video + instruction → one model for VLN, search, tracking and driving;
    camera / time identifier tokens; 8M samples across quadruped, drone, **wheeled** and vehicle.
  ) Code: n/a [check]. Relevance: cross-embodiment including wheeled.

* **OmniVLA: An Omni-Modal VLA for Robot Navigation** (arXiv 2509.19480, 2025)
  ) Goal: TLDR - goals as 2D pose, egocentric image or language → one VLA → actions.
  ) Code: https://github.com/NHirose/OmniVLA (~365★, license unclear)
  ) Relevance: one model for "go to (x, y)", "go to this picture" and "go to the kitchen".

* **GNM** (ICRA 2023) / **ViNT** (CoRL 2023) / **NoMaD** (ICRA 2024)
  ) Goal: TLDR - RGB history + (optional) goal image → EfficientNet + Transformer (NoMaD: + diffusion
    decoder with goal masking) → waypoints + temporal distance. ViNT adds diffusion subgoal proposals
    and prompt-tuning to GPS / route goals; kilometre-scale with a topological graph.
  ) Papers: arXiv 2210.03370, 2306.14846, 2310.07896 / Project: https://general-navigation-models.github.io /
    Code: https://github.com/robodhruv/visualnav-transformer (1,341★, MIT) / Models: https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg
  ) Size: ViNT ≈ 31M, NoMaD ≈ 19M parameters [unverified]. Jetson-class onboard compute.
  ) Relevance: **light local policy / explorer** for mapless and outdoor modes. Image goals only, so add language with LeLaN / OmniVLA.

* **LeLaN: Learning a Language-Conditioned Navigation Policy from In-the-Wild Videos** (CoRL 2024)
  ) Goal: TLDR - image + object-referring language → policy → velocities to reach the object; data relabelled by VLM + detector + CLIP (>130 h).
  ) Paper: https://arxiv.org/abs/2410.03603 / Relevance: short-range "go to the X object".

* **CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos** (CVPR 2025)
  ) Goal: TLDR - egocentric video + target position → transformer → urban waypoints; thousands of hours of walking video with automatic action labels.
  ) Code: https://github.com/ai4ce/CityWalker (~240★, Apache-2.0) / Relevance: **outdoor point-goal local policy** (N7).

* **ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints** (RSS 2022)
  ) Goal: TLDR - goal GPS + RGB + schematic / satellite map → traversability + heuristic search.
  ) Paper: https://arxiv.org/abs/2202.11271 / Relevance: outdoor map-assisted reference. It tolerates wrong maps.

* **MBRA / LogoNav: Learning to Drive Anywhere with Model-Based Reannotation** (arXiv 2505.05592, 2025)
  ) Goal: TLDR - relabel crowd-sourced teleop + YouTube data with a model-based expert → long-horizon GPS-goal policy.
  ) Code: NHirose/learning-to-drive-anywhere-with-MBRA (MIT) / Relevance: an outdoor data-scaling recipe.

* **VLN-CE: Beyond the Nav-Graph** (ECCV 2020)
  ) Goal: TLDR - defines continuous VLN in Habitat. Seq2Seq / CMA baselines; actions forward 0.25 m / turn 15° / stop.
  ) Paper: https://arxiv.org/abs/2004.02857 / Code: https://github.com/jacobkrantz/VLN-CE (~900★, MIT)
  ) Relevance: the standard action interface and metrics.

### 6.3 Maps and memory for navigation (details in [slam_mapping.md](slam_mapping.md))

* **VLMaps** (ICRA 2023): LSeg features fused into a top-down grid from an RGB-D tour; the LLM writes
  code over map queries ("go between the sofa and the TV"). arXiv 2210.05714, https://github.com/vlmaps/vlmaps (~732★, MIT).
  **Our closest analogue: the LLM writes code over a map API.**
* **NLMap** (ICRA 2023): a CLIP / ViLD queryable object map + SayCan planner. arXiv 2209.09874. Code: n/a.
* **ConceptGraphs** (ICRA 2024): RGB-D + poses → SAM masks + CLIP / LLaVA captions → object graph with
  LLM-inferred relations. arXiv 2309.16650, https://github.com/concept-graphs/concept-graphs (~986★, MIT).
* **HOV-SG** (RSS 2024): floor / room / object hierarchy from an RGB-D tour, multi-floor. arXiv 2403.17846,
  https://github.com/hovsg/HOV-SG (~565★, MIT). **Best fit for room goals from a prior tour.**
* **GOAT: GO to Any Thing** (RSS 2024): instance-aware lifelong memory + frontier exploration for
  category / image / text goals; 83 % SR over 9 homes / 675 goals; 60 % → 90 % with experience.
  arXiv 2311.06430. Code: n/a.
* **OK-Robot** nav part: an iPhone-scan tour → CLIP voxel map → A* on 2D occupancy. See [mobile_manip.md](mobile_manip.md).
* **VGGT-SLAM** (MIT SPARK, NeurIPS 2025; the paper behind MIT News "Teaching robots to map large
  environments", 2025-11-05): uncalibrated RGB → VGGT submaps aligned on SL(4) + loop closure.
  arXiv 2505.12549. See [slam_mapping.md](slam_mapping.md).
* **Spatial-TTT** (arXiv 2603.12255): test-time-trained fast weights keep spatial evidence from
  unbounded video. A possible long-horizon memory for tour reasoning.
* **NapMem / Active Memory Navigation** (arXiv 2607.05794): memory as a structured *action space*
  (memory tools) rather than passive retrieval. A design idea for exposing the map to the LLM as tools.
* **CoV: Chain-of-View prompting** (arXiv 2601.05172): training-free; the VLM actively picks or
  adjusts views over a 3D scene before answering. Useful for `look_around()` + EQA-style verification.

### 6.4 Surveys / lists
* awesome-vision-language-navigation: https://github.com/eric-ai-lab/awesome-vision-language-navigation
  (~611★, MIT). Last pushed 2024-05, so it is stale for 2025–26. Use the InternNav README and arXiv for newer work.
