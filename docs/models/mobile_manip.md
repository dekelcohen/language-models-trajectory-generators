# Mobile manipulation - base + arm, nav→manip hand-off, simulators

Research notes for putting the Franka on a round holonomic (kiwi) base and doing navigation +
manipulation from pixels + proprioception. Implementation plans:
[../plans/mobile_base.md](../plans/mobile_base.md),
[../plans/semantic_navigation.md](../plans/semantic_navigation.md).
Related: [navigation.md](navigation.md), [slam_mapping.md](slam_mapping.md).

Compiled 2026-11-08. Sources:
- arXiv titles and abstracts were checked with `docs/research_tools/paper_research.js arxiv`.
- New 2026 papers come from the @AIPaperSlop channel.
- GitHub star counts are as of 2026-11-08.
- The LeKiwi kinematics were checked against LeRobot's `lekiwi.py`.

Tags: **[unverified]** = from memory, so check it before relying on it. `n/a` = not found.

---

## 0. TL;DR for our stack

1. **Decouple navigation from manipulation**: navigate, stop, re-perceive, then use the arm.
   OK-Robot, HomeRobot and OWMM-Agent all do this. Whole-body control (MoManipVLA, BRS, Mobile ALOHA) comes later.
2. **VLM agent → JSON / function-call primitives** (`nav`, `pick`, `place`), each executed by a
   classical planner. OWMM-Agent and BUMBLE use this pattern, and it is the same as our
   LLM-writes-code-against-tools design.
3. **"Last-mile" base placement matters.** Success depends on *where the base stops* for the arm, not just distance to the target (MoMa-Kitchen).
   - Score standoff poses with the Franka reach map before driving.
4. **Re-detect the target after every base move.** Odometry drifts, and OK-Robot found that
   navigation and perception failures dominate the errors.
5. **Holonomic base.** Kiwi drive (LeKiwi) gives cheap kinematics. Powered casters (TidyBot++) give a
   real-payload holonomic base for a Franka-class arm.
   - A Franka (~18 kg arm + controller) **cannot ride a LeKiwi-size base** without tipping.
   - So use a larger round base in sim, and keep LeKiwi only as the reference for kinematics and sensors.

---

## 1. Concepts

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| OVMM (open-vocabulary mobile manipulation) | Pick any object in an unseen home and place it at a commanded receptacle | The task family we target ("bring the mug to the kitchen") | HomeRobot, OK-Robot, OWMM-Agent |
| Decoupled nav → manip | The base moves, stops, then the arm plans with the base fixed | Simplest. Fits our primitives and existing arm stack | OK-Robot, HomeRobot, OWMM-Agent |
| Whole-body control | The base and arm move together (an extra 3 DoF in IK / policy) | Larger reach and smoother motion. Needed for doors and wiping | MoManipVLA, BRS, Mobile ALOHA, N2M2 |
| Last-mile navigation / base placement | Choosing the final base pose so the target lies in the arm's dexterous workspace | Hand-off point between nav and arm. A wrong stop means a failed grasp | MoMa-Kitchen, N2M2 |
| Action primitives as function calls | The VLM emits `nav(region)`, `pick(obj)`, `place(recep)` as JSON | Maps 1:1 onto our tool calls | OWMM-Agent, SayCan, BUMBLE |
| Posed multi-view scene memory | Pre-mapped keyframes with poses fed to the VLM as global context | Pixel-only global scene context without a metric semantic map | OWMM-Agent, Mobility VLA |
| Dynamic spatio-semantic memory | Point-cloud memory with add/remove updates as the scene changes | Handles moved objects and stale maps | DynaMem |
| Affordance grounding | A pixel/point output by the VLM is lifted to 3D with depth | We already do this (`convert_2d_point_to_3d_world`) | OWMM-Agent, π0.5, LightNav-0 |
| Holonomic base | Independent vx, vy, ω (strafe while rotating) | Simplifies docking and approach angles. The user's chosen base | TidyBot++, LeKiwi |
| Base–arm co-registration | Calibrated arm-mount transform relative to base odometry | Every world-frame arm target goes through it | TidyBot++, Mobile ALOHA |
| Hierarchical VLA | A high-level semantic subtask prediction drives low-level actions | Analogous to our planner → LLM → reviewer | π0.5, ABot-M0.5 |
| Co-training static + mobile data | Mix fixed-arm demos into mobile-manipulation imitation learning | Cheap data scaling | Mobile ALOHA, π0.5 |
| World Action Model (WAM) | A video world model + inverse dynamics produces actions | Emerging alternative to reactive VLAs for long horizons | ABot-M0.5 |
| Step-wise verification + state tracking | The agent summarises progress and verifies each skill | Our VLM reviewer and task_state | BUMBLE, OWMM-Agent |
| Real-to-sim scene replicas | A phone or RGB-D scan is turned into an editable sim scene | Evaluate in a twin of the real home | LiteReality, Lucida, GS-Playground |

## 2. Techniques

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Kiwi IK / FK | `v_i = [cos a_i, sin a_i, R]·[vx, vy, ω]` and `ω_wheel = v_i / r`. FK uses the inverse of the 3×3 matrix | Closed-form base control and wheel odometry (see §6) | LeRobot `lekiwi.py` |
| Proportional wheel-speed saturation | If any wheel exceeds its max speed, scale all wheels by the same factor | Keeps the motion direction under saturation | LeRobot `lekiwi.py` |
| Powered casters | Each caster is steered and driven, so the base is holonomic | Payload-capable holonomic design | TidyBot++ |
| Voxel/CLIP map + A* | Open-vocabulary voxel map from a tour, A* on a 2D obstacle map | Navigation baseline that works in real homes | OK-Robot |
| AnyGrasp + heuristic place | A learned grasp model plus geometric placing | Robust pick in clutter | OK-Robot |
| Agentic data synthesis | PDDL plan → sim execution → keyframes → template CoT + LLM paraphrase → VLM SFT | Recipe to fine-tune our VLM on our own tools later | OWMM-Agent |
| Reach-map base pose scoring | Score floor cells by whether IK reaches the target (or a learned affordance-on-floor map) | Picks the standoff pose before driving | MoMa-Kitchen, N2M2 [unverified] |
| VLA transfer to mobile | A fixed-base VLA predicts EE waypoints; bi-level optimisation finds base + arm motion | Reuses fixed-base skills | MoManipVLA |
| Whole-body teleop | Teleop that drives the base and arms together (backdriving the base) | Collects data for later learning | Mobile ALOHA, BRS (JoyLo), AhaRobot (RoboPilot), TidyBot++ (phone) |
| Skill library + VLM selection | The VLM selects gross/fine skills with dual-layer memory | Building-scale long horizons | BUMBLE |
| LLM + affordance value | LLM proposals × learned feasibility | Rejects infeasible primitives | SayCan |
| Personalised rules via LLM summarisation | Generalise "where things go" from a few examples | Placement preferences | TidyBot |
| Mixture-of-Transformers action split | Separate subspaces for base and arm actions | Avoids nav/manip action conflicts | ABot-M0.5 |

## 3. Edge cases

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Odometry drift + omni-roller slip | Commanded and achieved base motion diverge, especially on carpet or under load | Re-detect the target and re-localise before grasping. Add slip noise in sim | OK-Robot, TidyBot++ |
| Wheel-speed saturation | Combined translation and rotation exceeds the motor limit | Scale proportionally, as LeRobot does | LeKiwi |
| Tipping / centre of mass | A tall, heavy arm on a small base, worse while the arm is extended or braking | Size the base, limit acceleration, use a stow pose while driving | TidyBot++ (design discussion), our analysis |
| Base stopped at a bad pose | Close to the object but outside the arm's workspace or facing wrongly | Last-mile placement; retry with a strafe | MoMa-Kitchen |
| Arm occludes camera / camera moved | The arm blocks the view; after driving, the camera extrinsics change | Stow before driving; update the camera extrinsics from odometry | HomeRobot, OK-Robot |
| Stale map / moved objects | The tour map is out of date | Dynamic memory; re-verify | DynaMem |
| Door / drawer / articulated objects | The base must move during the interaction | Whole-body control or sequenced moves | BEHAVIOR-1K, RoboCasa, Mobile ALOHA |
| Hallucinated / infeasible calls | The VLM calls pick on an unreachable or nonexistent object | Validate against the skill registry and state; reviewer | OWMM-Agent, SayCan |
| Multi-floor | Stairs or elevators | Out of scope for a wheeled base. BUMBLE uses elevators | BUMBLE |
| Navigation-induced failures dominate | Most real failures come from navigation or perception, not grasping | Invest in navigation verification | OK-Robot |

## 4. Benchmarks

| Name | Task | Scenes / sim | Metrics | Link |
|---|---|---|---|---|
| HomeRobot OVMM | Open-vocabulary pick and place in homes, sim + real | HSSD in Habitat | Stage-wise partial success, SR | https://github.com/facebookresearch/home-robot |
| ManiSkill-HAB (MS-HAB) | GPU home rearrangement: TidyHouse, PrepareGroceries, SetTable | ReplicaCAD in ManiSkill3 | SR | https://github.com/arth-shukla/mshab |
| Habitat 3.0 | Social navigation, social rearrangement, human-in-the-loop | HSSD / ReplicaCAD | SR, efficiency | https://github.com/facebookresearch/habitat-lab |
| BEHAVIOR-1K / BEHAVIOR challenge | 1,000 activities in 50 scenes with 9k+ objects | OmniGibson (Isaac) | Success + partial credit | https://behavior.stanford.edu |
| RoboCasa | Kitchen tasks with generative assets | robosuite / MuJoCo | SR | arXiv 2406.02523 |
| MoMa-Kitchen | Last-mile placement, 100k+ samples | Kitchens | Placement success | arXiv 2503.11081 |
| OWMM-Bench (sim + real) | VLM mobile manipulation agent | Habitat / HSSD | SR | OWMM-Agent project page |
| BEHAVIOR-Spatial | Spatial QA + point grounding + 14 manipulation / 10 navigation action tasks | BEHAVIOR-1K | Accuracy, SR | arXiv 2610.34051 |
| ESI-Bench | Active spatial intelligence (perception–action loop), 10 categories | OmniGibson | Accuracy | arXiv 2605.18746 |
| Real-home protocols | OK-Robot in 10 homes; π0.5 in unseen homes; BUMBLE building-wide | real | SR | papers |

### 4.1 Live leaderboards (checked 2026-11-08)

- **BEHAVIOR-1K 2026 Challenge** (CoRL'26).
  - Board: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard
  - Task: 100 full-length household tasks in 7 scenes, from RGB-D + proprioception.
  - Metric: Q score (partial credit) and full-task SR.
  - Training data: 20k teleop demos (1,950 h). Baselines: π0.5 and GR00T N1.7.
  - Submissions closed 16 Oct 2026; winners announced 4 Nov 2026.
  - **All top entries are trained VLA / world-action-model (WAM) policies.**

| Rank | Team | Method | Q | Full SR |
|---|---|---|---|---|
| 1 | The North Star (Huawei Noah EAI) | Hierarchical Physical RSI | 0.683 | 48.2 % |
| 2 | IF1001 (Intellifusion) | VLA WAM | 0.615 | 41.2 % |
| 3 | Cyber (Xiaomi Robotics) | WAM | 0.596 | 39.6 % |
| 4 | Anyverse | A²MEAS | 0.498 | 32.0 % |
| 5 | Autolab | A²WLA | 0.495 | 30.9 % |

- **BEHAVIOR 2025** (NeurIPS'25, 50 tasks). Both top solutions are π0.5 fine-tunes:
  - 1st: arXiv 2512.06951, correlated-noise flow matching + System-2 stage tracking.
    Code: https://github.com/IliaLarchenko/behavior-1k-solution
  - Close 2nd: **Openpi Comet**, arXiv 2512.10071. Code: https://github.com/mli0603/openpi-comet
- **EmbodiedBench challenge** (CVPR'26, EB-ALFRED + EB-Navigation, open VLMs < 10B).
  - Held-out winner: NJU-LAMDA-SZ, avg 52.2 % (ALFRED 67.3, Nav 37.0).
  - The same team scored 82.5 % on the open split, a big held-out generalisation gap.
  - Board: https://embodiedbench.github.io/challenge.html
- **Could not read:** the Embodied Arena mobile-manipulation tabs (JS-only), the HomeRobot OVMM
  board (aihabitat.org redirects to the 2023 page), and the ManiSkill-HAB / RoboCasa / RoboArena
  / ALFRED boards. None had a public board we could find.
- **Takeaway:** long-horizon household success is still below 50 % even for the best trained
  policies, and agent-style systems are not on these boards. Our hierarchical VLM-agent plus
  classical navigation design remains reasonable. The arm skill under it, however, is where
  trained VLAs win (see [gpt_policy.md](gpt_policy.md) / VLA docs).

## 5. Datasets

| Name | Content | Why it matters | Link |
|---|---|---|---|
| OWMM-Agent-data | Synthetic agentic multimodal mobile-manipulation episodes (CoT + JSON actions) | Ready SFT data for a mobile-manipulation VLM | https://huggingface.co/datasets/hhyrhy/OWMM-Agent-data |
| HSSD-200 | Curated synthetic homes | Scenes for OVMM / OWMM | arXiv 2306.11290 |
| ReplicaCAD | Interactive apartment | MS-HAB / Habitat 3.0 | Habitat |
| ProcTHOR-10k | Procedural houses | Large-scale scenes | arXiv 2206.06994 |
| BEHAVIOR-1K assets + demos | 9k+ objects, 50 scenes; BRS teleop demos | Whole-body data | behavior.stanford.edu |
| RoboCasa | 2,500+ kitchen assets, 100 tasks with generated demos [unverified counts] | Kitchen manipulation | arXiv 2406.02523 |
| MS-HAB demos | Large filtered RL demonstrations | Low-level home rearrangement | arXiv 2412.13211 |
| Mobile ALOHA demos | ~50 demos per task + static ALOHA | Co-training recipe | arXiv 2401.02117 |
| LeRobot hub (LeKiwi / SO-101) | Community teleop datasets | Cheap real data | https://huggingface.co/lerobot |

---

## 6. Base hardware and kinematics

| Base type | Strafe? | Pros | Cons | Example |
|---|---|---|---|---|
| Differential drive | No | Simple, robust, good odometry | Needs to turn to face the target; awkward docking | Stretch, Mobile ALOHA (AgileX Tracer), Jackal |
| **Kiwi (3 omni wheels at 120°)** | **Yes** | Cheap, 3 motors, holonomic | Roller slip, vibration on thresholds, low payload, odometry drift | **LeKiwi** |
| Mecanum (4 wheels) | Yes | More payload than kiwi | Heavy slip, 4 motors, lower efficiency | KUKA youBot, many AGVs |
| Powered casters | Yes | True holonomy with payload; supports any arm | More complex (steer + drive per caster, calibration) | **TidyBot++** |
| Legged + arm | Yes | Terrain and stairs | Complex; not our scope | Spot + arm, Go2 |

**Kiwi IK / FK** (verified from LeRobot `src/lerobot/robots/lekiwi/lekiwi.py`, `_body_to_wheel_raw` / `_wheel_raw_to_body`):
```
a_i = radians([240, 0, 120] - 90)        # left, back, right wheel mount angles
M   = [[cos a_i, sin a_i, R] for i]      # 3x3
v_wheel = M @ [vx, vy, omega]             # linear rim speeds (m/s)
w_wheel = v_wheel / r                     # rad/s ; LeKiwi: r = 0.05 m, R = 0.125 m
# saturation: if max|w| > w_max: w *= w_max / max|w|   (keeps direction)
[vx, vy, omega] = inv(M) @ (w_wheel * r)  # forward kinematics -> odometry
```
LeKiwi hardware:
- https://github.com/SIGRobotics-UIUC/LeKiwi (1,510★)
- SO-101 arm, Feetech STS3215 servos (3 for the wheels), Raspberry Pi 5.
- Two RGB cameras (front + wrist), 12 V battery.
- The repo provides a URDF.
- Docs: https://huggingface.co/docs/lerobot/lekiwi.

**For a Franka-class arm**:
- Use a ~0.5–0.6 m diameter round base, ≥40 kg with a low centre of mass, with the same kiwi kinematics (`R` ≈ 0.25 m).
- The real-hardware analogue is TidyBot++, an open-source holonomic base with powered casters that supports arbitrary arms (https://github.com/jimmyyhwu/tidybot2, 622★).

## 7. Simulators

| Sim | OS | Manipulation physics | Scene datasets | Mobile base | ROS2 | License | Use for us |
|---|---|---|---|---|---|---|---|
| **Genesis** (current) | Win / Linux / macOS | Good (rigid, MPM, SPH) | Imports URDF / MJCF / GLB; no HM3D / HSSD loader | URDF wheels or planar joints | No | Apache-2.0 | **M1–M3 base + arm, Windows dev**. Benchmark house-scale GLB rendering first |
| **PyBullet** (current) | Win / Linux / macOS | Moderate | URDF / OBJ only | URDF | Community | zlib | M1 kinematics + regression |
| **Habitat-Sim / Lab 3.0** | Linux / macOS (no Windows) | Weak contact (magic grasp in Rearrange) | HM3D, MP3D, Gibson, HSSD, ReplicaCAD | Fetch, Spot, Stretch | Bridges | MIT | **House-scale semantic navigation evaluation** (ObjectNav, VLN-CE, OVMM) on the Linux GPU box |
| **ManiSkill3 / MS-HAB** | Linux (Windows partial) | Good, GPU-parallel (SAPIEN / PhysX) | ReplicaCAD, HSSD-based | Fetch and others | No | Apache-2.0 | **Navigation + manipulation** with real low-level grasping |
| Isaac Sim / IsaacLab | Linux / Windows, RTX | Strong (PhysX 5) | USD | Carter, Jackal, mobile manipulators; Nav2 | First-class | NVIDIA / BSD-3 | High-fidelity, ROS2 sim2real (heavy) |
| OmniGibson (BEHAVIOR) | Linux, RTX | Strong + object states | BEHAVIOR-1K | Tiago, Fetch, R1 | Partial | MIT | Long-horizon household evaluation |
| AI2-THOR / ProcTHOR | Win / Linux | Abstract grasp | ProcTHOR-10k | LoCoBot (discrete) | No | Apache-2.0 | Not recommended (no real manipulation) |
| MuJoCo / MJX | all | Excellent contact | Manual | MJCF | No | Apache-2.0 | RoboCasa only |
| GS-Playground | Linux [unverified] | Parallel physics + batched 3DGS rendering (10⁴ FPS at 640×480) | Real-to-sim Gaussian scenes | yes | n/a | n/a | Photoreal vision RL, future |
| CARLA | Win / Linux | n/a (driving) | Towns | Vehicles | ros-bridge | MIT | Outdoor (deferred) |

GitHub stars (2026-11-08): habitat-lab 3,187★; Genesis and ManiSkill are not re-checked.

---

## 8. Papers

* **OWMM-Agent: Open World Mobile Manipulation With Multi-modal Agentic Data Synthesis** (NeurIPS 2025)
  ) Goal: TLDR:
    - Inputs: multi-view scene images + egocentric RGB-D.
    - Model: a fine-tuned InternVL-2.5 VLM with state tracking and CoT.
    - Output: JSON primitives (retrieve, nav, pick, place) for path and motion planners.
  ) Problem: connect high-level reasoning to low-level control, and stop VLM hallucination under robotic domain shift.
  ) Tags: VLM agent, Function calling, Mobile manipulation, Synthetic data, SFT
  ) Project: https://hhyhrhy.github.io/owmm-agent-project / Paper: https://arxiv.org/abs/2506.04217 /
    Code: https://github.com/HHYHRHY/OWMM-Agent (34★) / Models: https://huggingface.co/hhyrhy/OWMM-Agent-Model /
    Data: https://huggingface.co/datasets/hhyrhy/OWMM-Agent-data
  ) Method:
    - Data: a 4-stage pipeline in Habitat + HSSD with YCB / GSO objects. PDDL plans → executed in sim →
      keyframes where objects are visible and reachable → template CoT, paraphrased by GPT-4o-mini. 143 scenes.
    - Training: InternVL-2.5 at 8B and 38B, 1 epoch on A100s. The InternViT is frozen; the MLP projector and the LLM are trained.
    - Inference: multi-turn, with posed frames + a history summary → JSON. State tracking avoids dead loops.
      The bbox centre is used as the affordance point. A human confirms each step on the real robot.
  ) Results: SOTA over GPT-4o-class agents in sim; strong zero-shot real-world generalization (project page).
  ) Relevance: **closest analogue to our design**. Copy the function-call schema and the history-summary state. Its data recipe is our path to fine-tuning.

* **OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics** (2024)
  ) Goal: TLDR:
    - Phone-scan tour → CLIP / OWL-ViT voxel map → A* on a 2D obstacle map → navigate to the object.
    - Then AnyGrasp pick → heuristic drop.
  ) Paper: https://arxiv.org/abs/2401.12202 / Code: https://github.com/ok-robot/ok-robot (~617★, MIT) / Hardware: Hello Robot Stretch
  ) Results: 58.5 % pick-and-drop in 10 real homes; 82 % in cleaner scenes.
    - Most failures come from nuanced integration: navigation, retrieval and grasping.
  ) Relevance: the baseline modular stack. Copy its re-approach and margins.

* **HomeRobot: Open-Vocabulary Mobile Manipulation** (CoRL 2023)
  ) Goal: TLDR: defines OVMM (pick any object in an unseen home, place it at a receptacle). Provides a sim (Habitat + HSSD) + real (Stretch) stack and baselines.
  ) Paper: https://arxiv.org/abs/2306.11565 / Code: https://github.com/facebookresearch/home-robot
  ) Relevance: task definition, metrics, and a real/sim shared API like our SimAdapter.

* **TidyBot: Personalized Robot Assistance with LLMs** (IROS 2023 / AuRo)
  ) Goal: TLDR: a few user examples → the LLM summarises placement rules → a mobile manipulator tidies rooms.
  ) Paper: https://arxiv.org/abs/2305.05658 / Relevance: "where does this go" preferences via LLM summarisation.

* **TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning** (CoRL 2024)
  ) Goal: TLDR: a cheap, robust holonomic base with **powered casters** that supports arbitrary arms. Phone teleop for imitation learning.
  ) Paper: https://arxiv.org/abs/2412.10447 / Project: https://tidybot2.github.io / Code: https://github.com/jimmyyhwu/tidybot2 (622★)
  ) Relevance: **the real-hardware option for a Franka-class holonomic base**. Its control stack and odometry are a reference.

* **LeKiwi (+ LeRobot)** (2025, open hardware)
  ) Goal: TLDR: a low-cost holonomic kiwi base + SO-101 arm with LeRobot teleop and policies (ACT, π0, SmolVLA).
  ) Code: https://github.com/SIGRobotics-UIUC/LeKiwi (1,510★), https://github.com/huggingface/lerobot (28.7k★)
  ) Relevance: **the kinematics and real-bridge reference** for `mobile/real_lekiwi.py`.

* **Mobile ALOHA: Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation** (CoRL 2024)
  ) Goal: TLDR: whole-body teleop on a wheeled base (AgileX Tracer, differential drive) → behaviour cloning; co-training with static ALOHA data.
  ) Paper: https://arxiv.org/abs/2401.02117 / Results: co-training improves success by up to ~90 % relative with 50 demos per task (abstract).
  ) Relevance: data recipe. Its differential base cannot strafe.

* **BUMBLE: VLMs for Building-wide Mobile Manipulation** (ICRA 2025)
  ) Goal: TLDR: a VLM + open-world RGB-D perception + gross-to-fine skills + dual-layer memory → building-scale tasks (rooms, floors, elevators).
  ) Paper: https://arxiv.org/abs/2410.06237 / Results: 90+ h of evaluation; tasks of up to 12 ground-truth skills; beats baselines.
  ) Relevance: skill selection + verification + memory, the same shape as planner / reviewer / task_state.

* **MoMa-Kitchen: 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation** (2025)
  ) Goal: TLDR: learn the optimal final base position (affordance-grounded floor labels) for a smooth hand-off to manipulation.
  ) Paper: https://arxiv.org/abs/2503.11081 / Relevance: `choose_standoff_pose()` in our plans.

* **MoManipVLA: Transferring VLA Models for General Mobile Manipulation** (CVPR 2025)
  ) Goal: TLDR: a fixed-base VLA predicts EE waypoints → bi-level trajectory optimisation finds base + arm motions for reachability and smoothness.
  ) Paper: https://arxiv.org/abs/2503.13446 / Relevance: a path to whole-body control that reuses fixed-base skills.

* **N2M2: Learning Navigation for Arbitrary Mobile Manipulation Motions** (RA-L 2022) [unverified]
  ) Goal: TLDR: RL base policy that keeps the arm's end-effector motion kinematically feasible during base motion. arXiv ID n/a.
  ) Relevance: whole-body base feasibility. Background only.

* **SayCan: Do As I Can, Not As I Say** (CoRL 2022)
  ) Goal: TLDR: LLM skill likelihood × learned affordance value → feasible skill sequence on a mobile manipulator.
  ) Paper: https://arxiv.org/abs/2204.01691 / Relevance: feasibility filtering of LLM-proposed primitives.

* **DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation** (2024)
  ) Goal: TLDR: dynamic 3D point-cloud memory with add/remove updates; open-vocabulary localization via VLM features or multimodal LLM queries.
  ) Paper: https://arxiv.org/abs/2411.04999 / Relevance: stale-map handling for semantic navigation (N3–N6).

* **π0.5: a VLA with Open-World Generalization** (Physical Intelligence, 2025)
  ) Goal: TLDR:
    - Co-training on multi-robot data, high-level semantic subtask prediction and web data.
    - It is hierarchical: it first predicts a subtask, then actions.
    - It cleans kitchens and bedrooms in unseen homes with a mobile manipulator.
  ) Paper: https://arxiv.org/abs/2504.16054 / Code: openpi (weights for π0 / π0.5 released [unverified for mobile])
  ) Relevance: evidence that hierarchical semantic + action works end to end. The data is not reproducible.

* **ABot-M0.5: World Action Model for Mobile Manipulation** (arXiv 2607.00678, Jul 2026)
  ) Goal: TLDR:
    - A video World Action Model with intermediate latent actions.
    - A dual-level Mixture-of-Transformers that separates **base movement and arm actions**.
    - "Dream-forcing" trains the inverse dynamics on predicted videos.
  ) Results: SOTA long-horizon success and fine-grained control on mobile-manipulation benchmarks.
  ) Relevance: newest learned alternative. Its base/arm action split supports our decoupled tool set.

* **AhaRobot: Low-Cost Open-Source Bimanual Mobile Manipulator** (2025)
  ) Goal: TLDR: a SCARA-like dual-arm on a mobile base + backlash / friction compensation + RoboPilot teleop.
  ) Paper: https://arxiv.org/abs/2503.10070 / Relevance: low-cost hardware reference.

* **BEHAVIOR Robot Suite (BRS)** (2025)
  ) Goal: TLDR: a bimanual wheeled robot with a 4-DoF torso + JoyLo whole-body teleop + WB-VIMA policy for household tasks.
  ) Paper: https://arxiv.org/abs/2503.05652 / Relevance: the three key capabilities (bimanual, stable navigation, reachability).

* **BEHAVIOR-1K** (2024)
  ) Goal: TLDR: 1,000 everyday activities, 50 scenes, 9k+ objects, OmniGibson physics including deformables and liquids.
  ) Paper: https://arxiv.org/abs/2403.09227 / Relevance: long-horizon evaluation; door and articulated-object edge cases.

* **ManiSkill-HAB (MS-HAB)** (ICLR 2025)
  ) Goal: TLDR: a GPU-accelerated Home Assistant Benchmark with realistic low-level control, >3× faster than magic-grasp versions; RL/IL baselines + filtered demos.
  ) Paper: https://arxiv.org/abs/2412.13211 / Code: https://github.com/arth-shukla/mshab
  ) Relevance: **first choice for nav + manipulation evaluation with real grasping** (R4 recommendation).

* **ManiSkill3** (2024)
  ) Goal: TLDR: GPU-parallel simulation + rendering (SAPIEN), 10–1000× faster, heterogeneous scenes.
  ) Paper: https://arxiv.org/abs/2410.00425 / Relevance: the platform under MS-HAB.

* **Habitat 3.0: Humans, Avatars and Robots** (ICLR 2024)
  ) Goal: TLDR: humanoid simulation + human-in-the-loop + social navigation and social rearrangement tasks.
  ) Paper: https://arxiv.org/abs/2310.13724 / Relevance: house-scale navigation evaluation stack (Linux).

* **RoboCasa: Large-Scale Simulation of Everyday Tasks** (RSS 2024)
  ) Goal: TLDR: kitchen scenes with thousands of assets across 150+ categories, generative textures and MimicGen demos.
  ) Paper: https://arxiv.org/abs/2406.02523 / Relevance: kitchen task set for N6.

* **ProcTHOR** (NeurIPS 2022)
  ) Goal: TLDR: procedural houses (10k); RGB-only agents trained on them transfer zero-shot.
  ) Paper: https://arxiv.org/abs/2206.06994 / Relevance: scalable house generation.

* **HSSD: Habitat Synthetic Scenes Dataset** (CVPR 2024)
  ) Goal: TLDR: 211 curated synthetic homes. Shows that scene quality matters more than quantity for ObjectNav.
  ) Paper: https://arxiv.org/abs/2306.11290 / Relevance: house assets for our nav suite.

* **LiteReality** (2025) and **Lucida** (arXiv 2608.30821, 2026)
  ) Goal: TLDR:
    - LiteReality: an RGB-D scan becomes an interactive, articulated, PBR scene graph via asset retrieval.
    - Lucida: a cluttered capture goes through parse → generate → place, giving composable object assets.
  ) Papers: https://arxiv.org/abs/2507.02861, https://arxiv.org/abs/2608.30821
  ) Relevance: **real-to-sim twin** of the user's home for navigation + manipulation evaluation.

* **GS-Playground** (arXiv 2604.25459, 2026)
  ) Goal: TLDR: parallel physics + batched 3D Gaussian Splatting rendering (10⁴ FPS at 640×480) + an automated real-to-sim asset workflow.
  ) Relevance: photoreal vision RL for navigation policies, in the future.

* **BEHAVIOR-Spatial** (arXiv 2610.34051) / **ESI-Bench** (arXiv 2605.18746) (2026)
  ) Goal: TLDR:
    - BEHAVIOR-Spatial: spatial QA, point grounding and action (14 manipulation + 10 navigation tasks) on BEHAVIOR-1K.
    - ESI-Bench: active embodied spatial intelligence (the agent acts to see occluded or containment facts).
  ) Relevance: probes whether our VLM's spatial reasoning drives the right action.

---

## 9. Recommendations (feed into the plans)

1. Build the kiwi base with the LeKiwi IK, sized for a Franka (`R` ≈ 0.25 m). Use proportional saturation.
2. Keep the primitives decoupled: turn, forward, backward, strafe, then stop → re-perceive → arm. Whole-body control is a later milestone.
3. Add `choose_standoff_pose(target_xyz)`, which scores floor cells by Franka IK reachability. This is the MoMa-Kitchen / N2M2 idea.
4. Stow the arm before driving. After each move, update the camera extrinsics from odometry and **re-detect the target**.
5. Add odometry noise and slip in sim so drift bugs show up before real hardware.
6. Use the OWMM-style JSON / function primitives plus a history summary. The reviewer verifies each skill (BUMBLE).
7. Evaluation path: Genesis / PyBullet for the base, then MS-HAB or HomeRobot OVMM for navigation + manipulation, then a real-to-sim twin (LiteReality / Lucida).
