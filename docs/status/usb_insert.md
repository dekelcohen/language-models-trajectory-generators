# Plug / USB insert (Genesis): status and guide

Task: `python main.py -r franka --sim genesis --task usb --touch --attempts 3 --max-planner-iter 5 -c "Pick up the USB plug and insert it into the laptop's USB port"`
(run in conda env `vlm_traj`; it spawns the `vlm_genesis` server. `-r franka` is required, the default robot is sawyer).
Skill: `prompts/skills/touch-guided-insertion/SKILL.md`. Controller: `touch/insertion.py`.
Door status: see [`open_door.md`](open_door.md) (door:hidden: PyBullet SUCCESS 2/2 on 2026-11-11;
Genesis FAIL, top-down lever grasp misses).

## Status (2026-11-12): OPEN, about 1 seat per 3-4 LLM runs

- **Pickup works** in every run since 2026-11-10 (except a Sonnet wrong-rotation pick, s1).
- **Seated** (planner + reviewer success): u2, u6, p14 (Opus; p14 then crashed), s2 (Sonnet).
  s2's eval depth was 8.8 mm against a 10 mm seat threshold (lateral 0.4 mm), so the sim counts it as only partly seated.
- **The main failure is now the plug pivoting or sliding in the grasp** during repeated partial catches on the
  face, when the along-face aim error is large (p13, p15, o1). The second failure is the in-hand height offset
  being ~3-5 mm wrong (p16, s2).
- **Speed:** one rollout takes 17-30+ min, and the 20-min cap kills runs in the middle of the insert (s3). See "Where the time goes".
- **Default LLM for `--task usb`** is now `copilot-claude-sonnet-5.5` at `--reasoning-effort medium`
  (`main.py` `TASK_DEFAULT_LM`). An explicit `-lm` / `--reasoning-effort` wins. Perception VLM calls (scene analysis,
  wrist pointing) use `--perception-reasoning-effort` (default `low`; `inherit` = the main effort); the reviewer keeps
  `--reasoning-effort`. Wrist pointing for usb defaults to `or-google/gemini-3.8-flash` (benchmark below).

### Sonnet 5.5 vs Opus 5.5 (2026-11-12, medium effort, terra perception and reviewer)

The comparison looks at the LLM's parts only (pick choices, gate handling, code bugs). The insert physics is shared.

| run | LLM | pick | gate / regrasp | LLM mistakes | result |
|---|---|---|---|---|---|
| s1 | Sonnet | rot 0: closed along the plug length (34 mm wide); the reviewer accepted it | off_a 38.6 mm, so it regrasped; lift tripped the slip guard and lost the plug | wrong rotation; settle check compared command z with eef z (+0.06 offset), so it never corrected | FAIL (20-min cap) |
| s2 | Sonnet | rot 1.571, 5.7 mm deep | 16.8 mm, then 2 regrasps, then 23.8 mm (printed `gate ok: False`, inserted anyway) | ignored the gate (it was the gate's bug, see below) | **SEATED** 17 min (eval 8.8 mm) |
| s3 | Sonnet | rot 1.571, 4.7 mm | 15.0 mm, then 1 regrasp, then 22.9 mm ("gate miss by 1 mm, proceed") | none | capped (20 min) inside `insert_with_touch` |
| o1 | Opus | rot 1.571, 5.7 mm, fast (3 min) | 1 regrasp to 8.1 mm depth (off_z 4.1 mm) | forbidden z scan touching the face with the held tip (11 touches) | FAIL: plug pivoted 56 deg on far-edge catches, `slipped` |
| p14-p16 (11-11) | Opus (effort unset) | OK | p15 rejected 23.6 mm by 0.4 mm | gate bug | 1 seated then crashed (since fixed), 2 FAIL |

- **Verdict: no LLM-side regression.** Each model made about one skill violation per run. Sonnet's s1 rotation error is the
  most dangerous one: it is not caught by the reviewer. Sonnet is cheaper and faster, so it is now the usb default.
- **Bug found, fixed in this pass (skill):** the grasp gate's lower bound `tip_len + 12 mm` (24 mm for USB-A) is ABOVE
  `expected = tip_len + housing_len/2` (23 mm with the LLM's 22 mm housing prior). So a perfect grasp failed the gate
  (p15, s2, s3) and caused regrasp loops. The bound is now `min(tip_len + 12 mm, expected - 3 mm)`.

### Skill ablation (2026-11-12): simplifying SKILL.md did not help, so keep the full skill

Variants built by `research/touch/skill_ablation/make_variants.py`, loaded with `--skills-dir outputs/_ablation/skills_<v>`:
- `A_no_situational` (19.5k → 16.8k chars): no tip-direction, multi-face, multiple-openings or wrong-opening rules.
- `B_lean` (14.4k chars): A, minus the sanity-check, settle-correction, off-centre prose and force adaptation.

All runs used Sonnet medium and a 30-min cap. A1/B1 used terra wrist pointing; A2/B2 used Gemini, which became the default during the batch.

| run | variant | pointing | what happened | result |
|---|---|---|---|---|
| A1 | A | terra | off_a 8.6 mm, so it regrasped; dropped on lift (34 N); then re-picks closed on air | CAPPED, no insert |
| B1 | B | terra | reached insert: `slipped` after 117 s of partial catches (plug slid 18.5 mm in grasp); plug lost; 9 re-picks failed | CAPPED |
| A2 | A | Gemini | **wrong face**: parked at the laptop's right face (x 0.50, port is at 0.09). Gemini returned [] on the first view, then 2 false hits; `search_exhausted` x2, kept retrying that face | CAPPED |
| B2 | B | Gemini | right face found; then 15+ grasps: lifts resist 30-40 N and the fingers close on air | CAPPED, no insert |

- 0/4 seated, against 1/4 with the full skill (s1-s3, o1). Removing the wrong-face rule directly caused A2's loop.
  **Keep the full skill**; no removed part was shown to cause bugs.
- **New dominant failure: re-picking a put-down or dropped plug** (A1, B1, B2; also s1). The touched top is
  5.0-5.2 cm, against 5.78 cm on its stand, so the plug has left the stand. Every lift then resists 30-40 N and slips.
  Next: inspect the scene/contacts after a put-down (plug wedged against the stand?). Make put-down regrasps
  place the plug back ON the stand, or regrasp in place (open, shift, close) without lifting.
- Gemini pointing found the correct port in B2 but gave 2 false hits on the wrong face in A2. Pointing does not replace face selection.
- HF: `rollouts/2026-11-12/plug_usba_abl{A,B}_*`.

## Step-by-step: what runs, and how each step has failed

The planner splits the task into 2 subtasks: (1) pick up the plug, (2) insert it. Each subtask is LLM code in the agent
loop (`agent_runner.py`), up to 3 attempts, with a VLM reviewer verdict after each attempt.

### Step 1: pick (skill step 1)
- Head-camera perception boxes the plug. The LLM moves above it, `move_until_contact([0,0,-1])` touches the TOP
  (`touched_top_z`), then grasps at housing mid-height, `rotation` across the narrow width.
- `get_grasp_info()` records `grasp_eef_pos`, `touched_top_z` and `grasp_depth_below_top` (used later for off_z).
- Tip direction: two wrist hits (metal tip vs body) count only if both are on the part and >= 1 cm apart.

| failure | runs | fix |
|---|---|---|
| box merged plug + pedestal: grasped low, or assumed the plug was on the floor | g3, r1, p1 | grasp from the TOUCHED top; housing-height priors (USB-A 8 mm) |
| settle error added to the measured pose instead of the command | g8, s1 | prompt: correct settle errors on the previous COMMAND |
| `body_h = 2 * grasp_depth` (circular) gave off_z = 0 | p2 | skill bans it; use priors |
| plug carried back to front (tip hit on the stand) | p10 | tip-direction trust rule |
| wrong rotation: closed along the length (34 mm) | s1 (Sonnet) | open: no API check; the reviewer missed it. Idea: gate on gripper width vs the housing width prior |
| fast transit / wrist spin flung the plug | g4, u-root1 | API caps 0.3 m/s and 1 rad/s while the gripper is closed |

### Step 2: locate the opening (skill step 2, `api.locate_with_wrist_camera`, `touch/opening_estimate.py`)
- Head image gives the candidate face. The hand parks ~10 cm in front at carry rotation and takes 3-4 wrist views
  (port + top-edge pointing per view, terra VLM).
- `estimate_opening(port_cands, edge_cands, axis, opening_width)`: face-plane filter, densest z cluster, held-plug
  filter, returns `position`, `search_line`, `search_half` (reaches the farthest kept hit + one width), `search_cross`, `confident`.
- `estimate_face_normal` (head depth) or two face touches give the insertion axis.

| failure | runs | fix |
|---|---|---|
| wrong face (texture marks on both sides; side words) | g6, p4, r4 | sweep exhausted = no opening on that face; resolve side words by measured position |
| outlier z / bimodal z / held plug pointed at | g7, p6, p7 | `estimate_opening` densest cluster; drop hits at the projected fingertip |
| output > 2000 chars dropped, so the LLM never saw its estimate | u3 | head + tail truncation |
| along-face scatter 2-16 cm | g5, p5, p15 (45 mm), o1 (38 mm, spread 13 cm) | line search up to +-80 mm. Open: large errors lead to far-edge catches, which lead to pivots |

### Step 3: in-hand offset by touch, then the grasp gate (skill step 3)
- Two `move_until_contact(axis)` face touches 3 cm either side of the opening, 7 mm below it, give
  `off_a = (port - mean(hits)) . a` (eef to tip end along the axis). `off_z = touched_top_z - body_h/2 - grasp_eef_z`.
- Gate: `lo <= off_a <= expected + 5 mm`, else put the plug down, shift by `off_a - expected`, regrasp, re-touch.
  (`lo` was 24 mm until 2026-11-12; see the bug above.)
- Align: command `opening - offset - 4 mm * axis`, re-command until within 1 mm.

| failure | runs | fix |
|---|---|---|
| wrist-camera offset (tip hidden by the fingers) | tt2, tt3 | superseded: touch calibration |
| grasp too near the tip: the finger pads hit the face, stall, pivot | p9, p11, p12 | gate as code; regrasp required even if the task says "do not re-grasp" |
| gate lower bound > expected | p15, s2, s3 | fixed 2026-11-12 |
| off_z wrong by 3-5 mm (true eef to tip z vs measured) | p16 (+1.3 vs -3.6), s2 (-2.5 vs +0.9), o1 (8.1 mm deep regrasp, off_z 4.1) | **open**: the plug shifts or tilts in the fingers between the top touch and the insert; `search_cross` 2 mm (11 rows) sometimes covers it |
| LLM touch-scans the held tip against the face (forbidden) | g2, o1 | skill rule; LLMs still do it |
| lift after regrasp tripped the slip guard, plug lost | s1 | - |

### Step 4: `insert_with_touch` (api.py `insert_with_touch` -> env `_touch_insert` -> `touch/sim_runner.run_touch_insert`)
- `api.insert_with_touch(axis, tool_offset, search_line, search_half, search_cross, insert_force, opening_width)` builds
  an `InsertConfig` (`cfg`), sets rows from `search_cross` (0.4 mm pitch), and stretches `max_s` to fit the raster.
- It runs in the Genesis process: a 240 Hz torque impedance loop with external force from joint torques
  (real-robot feasible: proprioception only). Eval-only truth is logged as `[eval] insert start|phase|end`.
- It returns `outcome`, `tip_pos`, `events`, `finger_clearance_m`. The events are printed into `vlm_traj.log` as `[touch]   (t, phase, msg)`.

#### Controller state machine (`SpiralInsertController.step`)
```
APPROACH --F>3N--> PROBE --advance>2mm--> INSERT --depth>=7mm, slow--> VERIFY --moved<=1.8mm--> DONE seated
   |                 |  blocked                 |  stopped early              |  moved more
   no_contact        v                          v                             v
   (20 mm)        _catch() --> SEARCH <-- BACKOFF 1mm/0.3s              slipped
                     ^         |  drop>0.3mm x2
                     +--PROBE--+     exhausted -> search_exhausted;  |F|>35N -> over_force;  4 inserts -> jammed
```
| phase | what it does | key params (`InsertConfig`) |
|---|---|---|
| APPROACH | moves along the axis at 10 mm/s until axial F > 3 N | `f_contact` 3 N |
| PROBE | pushes `f_insert` (18 N) for 0.4 s. Advance > 2 mm: INSERT. Advance >= 0.4 mm: partial catch (`_catch`) | `probe_s`, `f_insert` |
| SEARCH | holds 3 N (settles 0.2 s), slides along the spiral (r <= 3 mm) or the line raster (+-`sweep_half` x rows), 5 mm/s. The axial reference creeps (tau 1 s) so in-grip creep is ignored. A drop > 0.3 mm on 2 ticks: PROBE | `spiral_speed` 5 mm/s, `creep_tau` |
| edge finder | two catches 9-16 mm apart (0.75-1.35 x opening width) give the centre; column raster +-0.9 mm along x +-cross, 0.3 mm pitch; max 2 cycles | `edge_*`, `col_*` |
| lone-catch guess | one catch after the sweep moved (`sweep_s > col_pitch`): guess the centre half a width BACK against `sweep_sign`, column +-2 mm; max 2 | `guess_*` |
| INSERT | pushes; advance > 18 mm: `slipped` (slid in the grasp). Stall at >= 7 mm: VERIFY; else "stopped early", catch + BACKOFF | `seat_depth` 7 mm, `seat_depth_max` 18 mm, `max_inserts` 4 |
| VERIFY | 2 mm sideways at 3 N axial; moved <= 1.8 mm: seated | `verify_shift`, `verify_tol` 1.8 mm (was 0.8: false `slipped` in u4) |

| failure | runs | fix / status |
|---|---|---|
| spiral (3 mm) far too small for vision error | g5 | line raster |
| z error > 1 mm defeats the 5-row raster | g9, g10, hz1-3 | `search_cross` rows; edge finder |
| partial catches (tip on a side edge) read as false drops | jam1 | edge finder (edge2/3, ver2 seated) |
| false "seated" (plug slid back in the grasp) | edge4 | VERIFY sideways check |
| false `slipped` (tolerance too tight) | u4 | `verify_tol` 1.8 mm |
| one-directional raster feels only the trailing edge | p13, harness | lone-catch guess (p14 and s2 seated through it) |
| guess on the first probe, before any sweep | p15 | guard `sweep_s > col_pitch` |
| `config` dict shadowed the module: crash after a seat | p14 | renamed `cfg` + regression test |
| **plug pivots / slides in the grasp on repeated catches at 18 N** | p13 (9 mm slide), p15 (81 deg), o1 (56 deg) | **open**: lower the force on catches, or stop after N catches without a seat |
| plug flung metres after a slip (contact blow-up) | u5, p13 | open (sim artefact) |

### Step 5: outcome handling (skill steps 4-5)
- `seated`: open the gripper and retreat along -axis. Others: the skill adapts force, width and search per retry; on
  `slipped`, check `gripper_holding()` and redo step 3.
- The reviewer (terra) judges from images. It accepted s1's bad pick (false positive).

## Where the time goes (why a rollout takes 17-30 min)

| phase | wall time | why |
|---|---|---|
| pick + review | 3-8 min | ~5-10 LLM turns (20-60 s each) + review |
| wrist locate | **5-13 min** | each terra `locate_with_wrist_camera` takes 15-30 s, x2 (port + edge) x 3-8 views, plus view moves; cache misses because the pose differs. **2026-11-12:** usb wrist pointing now defaults to gemini-3.8-flash (6.5 s/call, more accurate, see below) and perception calls default to `--perception-reasoning-effort low` |
| touch calibration + regrasp | 3-8 min | each regrasp re-runs put-down, grasp, lift, 2 touches + LLM turns |
| `insert_with_touch` | 4-15 min | sim time = search_half x (1 + 2 rows) / 5 mm/s, e.g. 52 mm x 11 rows: ~240 s sim. Pure sim time, no LLM inside (o1: 48 s sim took ~6 min wall). |

**Sim speed-up (2026-11-12, physics unchanged)**, measured with `research/touch/profile_touch_loop.py`
(5 s guarded move, Genesis CPU):

| ms per 4.17 ms sim step | before | after |
|---|---|---|
| `get_joint_state` (14-18 calls/step, 3 Genesis reads each) | 10.3 | 0.8 (batched `get_joint_positions_velocities`, cached dof indices) |
| `time.sleep(control_dt)` in `Env.update()` after every step | 4.5 | 0 (now sleeps only what is left of dt: real-time pacing, never slower) |
| `get_link_pose` (failed `link_ref_frame` import + log line on every call) | 2.2 | 1.5 (import resolved once, warning logged once) |
| physics `step` | 5.6 | 6.0 |
| total | 27.3 (0.15x real time) | **14.0 (0.30x)** |

The `Env.update()` sleep fix also speeds every Genesis trajectory (pick, view moves), not only the touch loop.
Physics (6 ms) is now ~45% of the step; the rest is Python (Jacobian, gravity, recording frames).

### Wrist pointing benchmark (2026-11-12, offline)
`research/touch/capture_wrist_views.py` (vlm_genesis) saves 16 wrist views around the port face (8 open hand, 8 closed
fingers; the plug pick in that harness missed, so no held-plug views), each with depth and camera pose. 
`research/touch/pointing_bench.py` (vlm_traj) sends the exact `locate_with_wrist_camera` prompt and scores the best
point against the projected port (eval only): pixel error, 3D error of the depth-lifted point, latency. 14 views have the port visible.

| model:effort | hit < 5 mm | hit < 10 mm | px err (median) | 3D err mm (median) | latency s (median) |
|---|---|---|---|---|---|
| copilot-gpt-6.1-terra:low | 0.07 | 0.29 | 13.0 | 23.9 | 13.5 |
| copilot-gpt-6.1-terra:medium | 0.21 | 0.36 | 13.0 | 24.0 | 16.5 |
| **or-google/gemini-3.8-flash:low** | 0.57 | 0.64 | 2.2 | 4.4 | **6.5** |
| or-google/gemini-3.8-flash:medium | **0.79** | **0.86** | **1.4** | **3.3** | 18.8 |

- Terra is ~5x less accurate than Gemini Flash; its 13 px median miss is the "along-face scatter" of runs p15/o1.
- `--task usb` now defaults `--wrist-pointing-model or-google/gemini-3.8-flash` (`main.py` `TASK_DEFAULT_WRIST_POINTING`),
  at the perception effort (low). Medium is more accurate but as slow as terra. Planner perception stays terra.
- 29 Gemini prompts used. MolmoPoint (`--wrist-pointing-model gpu-molmo`, `gpu_server` module `molmo_point`) is
  ready to benchmark with the same script once a GPU server runs it.
- Each live `locate_with_wrist_camera` now also saves `<image>_cam.npz` (depth + wrist pose), so real rollout views can be added to the bench.

Remaining ideas: one VLM call for port + edge; fewer rows when z is confident; a coarse first raster pass at
10-15 mm/s, centre-out rows. A real robot runs the search in real time (~4 min worst case for the same raster).

## Open issues (priority)
0. Re-picking a put-down / dropped plug fails: the lift resists 30-40 N, then the fingers close on air (ablation A1, B1, B2, s1). This blocks every regrasp path.
1. Plug pivots in the grasp under repeated 18 N partial catches (p13, p15, o1).
2. In-hand z offset 3-5 mm off (p16, s2, o1).
3. Wrist pointing along-face error up to 4.5 cm. That means long searches and more catches (feeds 1).
   2026-11-12: Gemini Flash pointing (default now) cuts the median 3D error from 24 to 4.4 mm offline; to confirm live.
4. Rollout time: runs may take up to 30 min during experiments. **Target: the touch search (after the
   LLM/VLM calls) must finish in 1-3 min wall.** Sim loop is 2x faster now (0.30x real time); still needed: coarse
   first pass / centre-out rows / fewer rows when z is confident.
5. Sonnet pick rotation is not verified (s1); add a width check against the housing-width prior.
6. Eval: s2 reported seated at 8.8 mm, under the 10 mm success threshold (VERIFY uses 7 mm advance).

## Runs index
- 2026-11-12: `outputs/2026-11-12/plug_usba_{sonnet_s1,sonnet_s2,sonnet_s3,opus_o1}_*` (HF `rollouts/2026-11-12/`).
- 2026-11-10/11: HF `rollouts/2026-11-10/`, `rollouts/2026-11-11/`.
- 2026-11-02 to 05: local `outputs/2026-11-0{2..5}/` (not on HF yet). Harness: `outputs/touch/e2e_*`.

<details><summary>History (compacted): 2026-11-02 to 2026-11-11</summary>

**2026-11-02 (tt1-3, gn1):** the in-hand offset came from the wrist camera (tip hidden), so impossible offsets and endless
attempts. Added `tool_offset`, `get_robot_state()`, an idle guard and a per-attempt turn cap. The wrist-camera offset was
replaced by touch on 11-03.

**2026-11-03 (g2-g10, harness):** touch calibration; carry speed cap; line search; edge finder; seat check (tol 0.8 mm,
later 1.8). Harness seated edge2/edge3/ver2. g-runs failed on grasp height, settle bug, wrong face, outlier z, z error.

**2026-11-04 (r1-r4):** every LLM insert used rotation 0 (prompt hard-coded): plug at 90 deg. Fixed: measured rotation
in the state and prompt. Wrist views at carry rotation.

**2026-11-05:** a 270 deg wrist spin flung the plug (rotation cap 1 rad/s); regrasps on air next to the slab (touch
sanity rule). **u2: first LLM SEAT** (8.2 mm, tilt 0).

**2026-11-08 (u3-u9):** dropped long output (fixed); `estimate_opening` added; false `slipped` (tol 1.8 mm); search window
too short (reach the farthest hit). **u6 SEATED** (Opus/terra, 12 min). Model variants: only Opus + terra worked then
(astra / sol as LLM wandered; sol as perception was 8 cm off).

**2026-11-09:** parametric scene (`PLUG_TYPE`, `PORT_FACE`, `PORT_DIR_DEG`, distractor, seed); `insert_force`,
`opening_width` args; face normal from depth or touches.

**2026-11-10 (p1-p13, v1):** removing the overfitted skill text exposed general gaps (see the step tables). Scripted A/B:
the insert controller is marginal (chaotic after contact) in both the old and the new scene.

**2026-11-11 (p14-p16):** lone-catch centre guess; p14 SEATED 11.9 mm then the `config` shadow crash; p15 45 mm error
and pivot; p16 z offset 5 mm off.
</details>
