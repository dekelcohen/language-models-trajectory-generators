# Plan: Scene perturbations (anti-overfitting) + 2-week review of door / USB changes

Status: proposal (2026-11-09). Scope: review commits `97c122a..b9ad73a` (2026-10-26 → 2026-11-08,
~12.7k lines in skills, control code, sim code and assets), then define a perturbation suite.
The suite checks that the door, USB and new plug skills generalise beyond the single scene each was tuned on.

Related: [touch.md](touch.md), [touch_pull.md](touch_pull.md), [insert_key.md](insert_key.md),
[../status/open_door.md](../status/open_door.md).

---

## 0. Review findings (what the last 2 weeks overfit)

Each `door:hidden` and USB scene had exactly **one** fixed configuration. Several fixes were tuned
on single rollouts (`r3`..`r9`, `q2`..`q4`, `od12/od13`, `usb_u4`, `usb_u5`), and the comments say so.
Most of the *mechanisms* are general: depth door-plane fit, vision door angle, guarded pull with hinge
mirroring, the touch insertion state machine and the slip guard. Most of the *constants and
assumptions* are not.

### 0.1 Door: overfit items

| # | Where | What is specific to the current scene | Breaks when |
|---|---|---|---|
| D1 | `helpers/door_open.py:58` `_approach` (commit d497400) | Reaches the lever **over the panel top** (`top_z + 0.06`). Works only because the Adroit panel is ~0.9 m tall and the base is close. | Any real-height door (≥ 1.8 m) or a farther base. |
| D2 | `touch/door_plane.py:125` `jamb(max_m=0.10)` | Searches for the hinge axis past a frame gap. The comment says "PyBullet door: a 6 cm gap, axis 8.7 cm beyond the panel's visible end". | Doors without that gap, i.e. real doors (gap of a few mm), or a different frame. |
| D3 | `touch/door_plane.py:10-11,135-165` | Takes the hinge as the panel end **farther from the rose**. Rose window `rose_off_m=(0.01,0.06)`, `rose_radius_m=0.3`. Lever grasp assumes a bar "7-16 cm proud of the face". | Knob handles, bar pulls, a handle centred on the panel, a recessed handle, or a different lever length. |
| D4 | `helpers/door_open.py:140`, `api.py:1106,1138` | `goal_deg=65` (60 + depth margin) and a hard `deg < 55.0` → `partly_open`. | Tasks that need a different opening (cabinet 90°, "ajar"), or a door whose end stop is < 65°. |
| D5 | `helpers/guarded_pull.py:65` | `behind_deg=20, ahead_deg=17, min_open_deg=11, hover=0.12, below=0.15`. The "panel meets the forearm at 20-35 deg" logic (prompt §6c) was tuned on the current base pose and door. | A moved or rotated door, a different base pose, a wider panel. |
| D6 | `helpers/guarded_pull.py` (~323-330) | Hinge-hint clamp 0–20 cm behind the lever along the pull normal. | Thick doors, or handles mounted far from the free edge. |
| D7 | `prompts/main_prompt.py:134` | The guarded_pull example hard-codes `direction=[0, -1, 0], hinge=[0.42, -0.30]` (world coordinates). | The LLM copies the numbers in any other scene. |
| D8 | `prompts/planner_prompt.py:87-92` | The planner few-shot example is **the test scene itself**: "gray cylinder in front of the door ... ~30 cm to the side". | Any other occluder; it also inflates measured success. |
| D9 | `prompts/skills/open-close-door-cabinet-drawer/SKILL.md:38-93` | Rollout anecdotes (`door:hidden r3/r7`, "harness: 60-61 deg") plus fixed offsets: shift 0.05 m toward the tip, press 2-3 cm, hinge 0.4-0.8 m away, arc to 75-80°. | Other doors; these turn one door's numbers into rules. |
| D10 | `sim_envs/pybullet/door.py:96-110` | Hinge rest force 2.0 and latch spring 3.0 (down from 200 so "press to unlatch" works). Handle friction 2.0, spinning 1.0. | Stiffer or looser real hardware. These are plausible, but the code is only tested at one value. |
| D11 | `my_assets/adroit_door/adroit_door.urdf` (lever collision diff) | Lever collision was added on one side only. The far-side lever has **no collision** because it "would jam the door at ~1 deg". | Mirroring the hinge or handle side gives an asymmetric, partly ghost handle. |
| D12 | `sim_envs/pybullet/door.py:71-78,128-136` | Door pose `[-0.11,0.04,0.25]` yaw 4 rad; base `[-0.3,0.5]` yaw −60° with joint 1 = −1.5; pole at `[-0.15,0.30]` r 0.15 m h 1 m. All constants. The only knob is the env var `DOOR_BASE`. | Not a bug by itself, but every success so far is from this one pose. |

### 0.2 USB: overfit items

| # | Where | What is specific | Breaks when |
|---|---|---|---|
| U1 | `sim_envs/genesis/usb.py:39-51` | Port is always on the laptop's **−x face** with the laptop axis-aligned. The plug spawns with its **tip already pointing +x**, i.e. pre-aligned to the insertion axis. | Port on another face, a rotated laptop, or a plug lying at an arbitrary yaw. |
| U2 | `touch/opening_estimate.py:35-38` | Face plane = the **dominant world axis** of `axis`, and `search_line` = a world axis. This assumes an axis-aligned face. | A laptop yawed 20°, or a socket on a slanted face. |
| U3 | `touch/insertion.py:30-95` `InsertConfig` | Forces tuned to the Genesis USB retention (`f_insert=18`, "above ~12 N, below grip slip ~24 N"). Edge finder `edge_min_sep/max_sep = 9-16 mm` ("defaults: USB-A, 12 mm wide"). Spiral 3 mm, `verify_tol=1.8 mm` from `usb_u4`, `seat_depth_max=0.018` ("12 mm tip"). | A 2-pin power plug (pins 19 mm apart, 4.8 mm round), HDMI, or any part with different retention. |
| U4 | `prompts/skills/touch-guided-insertion/SKILL.md:37,53,65-81,103,131-154` | Words and numbers specific to USB-A and laptops ("USB-A plug: 8 mm", "laptop side", `axis=[1, 0, 0]`, `tool_length=0.025`, "both sides of a laptop"). | Any other connector. The skill should be part-agnostic, with USB as one example. |
| U5 | `sim_envs/genesis/usb.py:157-169` `step_hook` | Retention = friction switched on when the **sim tip pose** is inside a box around the port. This is valid env physics (it models the spring contacts) and not an agent leak. It is still the only reason the plug "stays seated", and its 12 N is what `f_insert` was tuned against. | Must become a parameter (U3/U5 co-tuned). |
| U6 | `sim_envs/genesis/usb.py:115-146` | Plug armature 0.5 with gravity compensation, pad torsion 0.05 Nm, `GRIP_KP=2500`. These are solver conditioning for a 12 g part; this is physics plumbing, not cheating. | Other part masses; re-derive per part (formula, not constant). |
| U7 | `sim_envs/genesis/usb.py:95-101` | Head camera hand-placed at yaw 300 / pitch −30 to face the −x port face. | Any other port face. |

### 0.3 Already general (keep)

- Door plane, hinge and angle **from head depth** (`touch/door_plane.py`, `api.measure_door_angle`), not from the hinge joint.
- Hinge mirroring on an early slip or jam (`guarded_pull`) handles a wrong hinge-side hypothesis.
- Slip guard uses width, felt force and camera tracking (`touch/slip_guard.py`), with no engine contact lists.
- `insert_with_touch(axis, tool_offset, search_line, ...)` already takes the axis as an argument; only the defaults and the skill text are USB-shaped.
- `sim_envs/pybullet/insert_key.py:142-147` already has seeded pose randomisation (`randomize`, `rng`, `XY_JITTER`, yaw range). Reuse this pattern.

---

## 0.4 Sim2Real review (2 weeks)

No change found that puts privileged sim info **into a prompt or a decision**. Door angle, hinge side,
lever grasp, port location and seat checks are all derived from depth/RGB, wrist-camera pointing, or
(simulated) EE force and finger width. Findings, most important first:

| # | Severity | Finding | Fix |
|---|---|---|---|
| S1 | **Medium (latent leak)** | The env handshake sends `simenv.get_state()` to the agent process (`env.py:880,897`), stored as `api.sim_state` / `ctx.sim_state` (`agent_runner.py:590,605`). `GET_STATE` is also callable from the agent side (`env.py:1181-1192`). Since 11-02 (`dbc0a2a`), door `get_state` adds `door_angle_deg` and `hinge_pos`; USB adds `port_pos` and `insertion_depth`. Today these are only **logged** (`agent_runner.py:181-205`, `[world-state]` lines), but they sit one attribute away from LLM-written code that runs with `api` in scope. | Strip `sim_state` from the handshake. Serve it on an eval-only channel, or tag the response and drop it in `api`. Add a test: `api` has no `sim_state` attribute, and no prompt string contains `door_angle_deg`/`port_pos`/`hinge_pos`. |
| S2 | Medium (indirect) | `vlm_traj.log` now contains GT `door_angle_deg` and object `*_pos` (`[world-state] ... objects=`). `--learn-from-trajs` (`helpers/main_utils.py:163-185`) feeds a past-trajectories file to the LLM. If that file is cut from a log, GT leaks into in-context examples. | Prefix GT lines with `[EVAL-GT]` and filter them in `learn_from_past_trajs`, `rollout_report` (and any summary fed back to an LLM). |
| S3 | Low (realism) | Touch "force" in the Genesis runner is the **impedance-controller error** plus Gaussian noise (`touch/sim_runner.py:103-107`), not measured joint torques. Prompts call it "felt force" or "joint torque sensing" (`helpers/touch_evidence.py:55-58`). For a quasi-static Cartesian impedance controller this equals the real Franka `O_F_ext_hat_K` to first order, so it is acceptable. However, it has no model error, payload error or friction bias (the real one has a 2-5 N bias). | Add a configurable bias/drift and scale error (`touch_force_bias_n`, `touch_force_scale_err`). Optionally cross-check against `J^T`-projected measured joint torques in Genesis. Re-validate the thresholds `f_contact=3 N` / `f_search=3 N` against a ~3 N bias. |
| S4 | Low | USB retention switch keys on the sim tip pose (`usb.py:157-169`). This is env physics, not agent input, so it is OK. Keep it out of any IPC reply. | None (document). |
| S5 | Low (physics honesty) | Far-side door lever has no collision (D11). This is a non-physical asymmetry, not a weld. No weld/attach constraints, teleports or kinematic grasps were found in the window. Grasps are contact + friction. Plug armature with gravity compensation is solver conditioning with correct weight. | Fix by giving both levers collision with a frame clearance (generator, §5.1). |
| S6 | Info | Camera intrinsics/extrinsics given to the agent are **exact**. On the real robot hand-eye error is 0.5-1 cm. Depth is noiseless; the door-plane RANSAC and the 3° angle bins were tuned on clean depth. | Perturbation P8 (§4) adds calibration and depth noise. |

---

## 0.5 Progress (2026-11-09)

| Item | Status |
|---|---|
| D2 | **Done, replaced by touch + depth.** `estimate_door_plane` no longer searches a frame gap: the hinge is the visible panel end, `hinge_source="panel_end"` (a hypothesis). `helpers/door_open.probe_hinge` makes a short guarded straight pull on the held lever, then `touch.door_plane.hinge_from_swung_panel` intersects the swung panel line with the closed face (head depth) -> `hinge_source="swung_panel"`, including the flipped (wrong side) case. `guarded_pull(start_deg=...)` continues from the probed angle and does not flip a measured hinge. No VLM hinge pointing. |
| D7 | Done: `main_prompt` example uses `p = estimate_door_plane(handle_xyz); guarded_pull(direction=p["pull_dir"], hinge=p["hinge"])`; documents `probe_door_hinge`. |
| D8 | Done: the planner few-shot example is now "apple into a drawer, cereal box blocking it". |
| D9 | Done: rollout anecdotes and absolute numbers removed from the door skill; the hinge is described as a hypothesis measured by the probe. |
| P1/P2 knobs | `DOOR_POSE="dx,dy,dyaw_deg"` and `DOOR_MIRROR=1` (hinge on the other side, `my_assets/adroit_door/adroit_door_mirrored.urdf`, generated). Harness: original door 89.9 deg; `DOOR_POSE=0.05,-0.04,10` 89.6 deg; mirrored: hinge measured within ~4.5 cm at 20 deg (see docs/status/open_door.md). |
| U1, U7 | Done: `sim_envs/genesis/usb.py` is a parametric plug scene: `PLUG_TYPE` (usb_a, usb_c, hdmi, peg), `PORT_FACE` (left, right, back), `PORT_DIR_DEG` (laptop turned to match), `PORT_U`, `PLUG_YAW_DEG` (plug not pre-aligned), `PORT_DISTRACTOR` (a second, keyed port: a foreign plug stops at 3 mm; P6b), `PLUG_SCENE_SEED`. Random headings face sideways and toward the fixed head camera (>= 15 deg). Defaults = the original scene. |
| U2 | Done: `estimate_opening` works in the face's own frame for any heading; new `estimate_face_normal(point)` (head depth) and `face_normal_from_touches(p1, p2, hint)` give the insertion axis. |
| U3 | Done: `insert_with_touch(insert_force=, opening_width=)`; the skill says how to adapt them per retry (jammed with the tip in -> x1.3, slipped -> x0.7 + more grip, over_force -> re-aim). |
| U4 | Done: the insertion skill is part-agnostic (USB-A/C, HDMI, peg, mains plug), measures the axis, turns the hand by the plug's measured heading, and handles a wrong (look-alike) opening. |
| U5, U6 | Open (sim-only conditioning; documented, not a decision input). |

## 1. Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | Every scene parameter that a skill could overfit is a **sampled, seeded parameter** of the scene, not a buried constant. | `--scene-variant <name>` / `--scene-seed N` reproduce a scene exactly; the variant dict is logged and written to `summary.md` + HF `metadata.jsonl` |
| G2 | Door task succeeds across door pose, hinge side, handle type/side/height, door size, hinge/latch forces, occluder variants | success rate per axis on a fixed seed set ≥ 70% of the nominal-scene rate |
| G3 | USB insertion succeeds with the port on any laptop face, any laptop yaw, any plug yaw | same metric |
| G4 | A new **electric-plug** task (2-pin / 3-pin plugs into wall plate / power strip) passes with the **same** skill, API and controller code as USB; only scene + part description differ | `grep -i usb touch/ prompts/skills/touch-guided-insertion` returns only examples; plug success ≥ 50% at M3 |
| G5 | Robust to camera viewpoint, lighting/shadows, distractors, calibration and depth noise | per-axis success; perception-failure counts in the logs |
| G5b | Two-slot scenes (USB + HDMI/USB-C look-alike): the robot recovers from a wrong slot by touch and seats in the right one | success ≥ 70% with the look-alike first; `wrong_slot_tries` ≤ 2; no press > `f_max` |
| G5c | Distractors (P10) in **every** task, not only USB | per-task success with `distractors/lookalike` ≥ 70% of nominal; `picked_wrong_object` = 0 |
| G6 | Default (nominal) scenes stay byte-identical | `tests/golden/pybullet/door.jsonl`, `tests/test_pybullet_regression.py` unchanged |

## 2. Non-goals

- Full domain randomisation for RL training (this is an **evaluation** suite; training is out of scope).
- Photorealistic rendering (Isaac/RTX). We use PyBullet/Genesis rasterisers, with shadows where available.
- Changing the agent architecture. Fixes are limited to removing the scene-specific constants found in §0.
- Real-height (2 m) doors with a fixed-base Franka. These are out of reach. Use the mobile base plan
  ([mobile_base.md](mobile_base.md)), or cap height at what the arm can span (see §4 P3).

## 3. High-level approach

```
 CLI: --task door:hidden --scene-variant door/hinge_right --scene-seed 7
          │
          ▼
 sim_envs/perturb.py        sample(variant, seed) -> SceneParams dataclass (frozen, JSON-able)
          │                    (ranges per axis; "nominal" = today's constants exactly)
          ▼
 sim_envs/*/<task>.py       SimEnvDoor(params) / SimEnvInsertion(params)
          │                    builds URDF from a generator (door, socket face), places occluder,
          │                    cameras, lights, physics params
          ├── visibility check (eval-side, scene build only): target parts project inside the head
          │   image and are not fully occluded unless the variant intends it -> resample otherwise
          ▼
 env.py                     logs params (EVAL-GT), evaluation get_state()/check_success use params
          │
          ▼
 agent (unchanged)          sees only pixels + proprio, as today
          │
          ▼
 tools/perturb_bench.py     runs N seeds x M variants, aggregates success / time / failure reason
```

Rules:
- `nominal` reproduces today's scene **exactly** (G6).
- Each perturbation axis is independent, so a failure can be attributed to one axis.
- Combined variants (`door/mix`) come only after each axis passes alone.

## 4. Perturbation axes

### P1. Door pose (move the door)

| Param | Nominal | Range | Notes |
|---|---|---|---|
| `door.xy` | `[-0.11, 0.04]` | ±0.10 m each | keep the handle within arm reach (IK-check the lever pose; resample otherwise) |
| `door.yaw` | 4.0 rad | ±25° | not axis-aligned: exercises `pull_dir` from depth |
| `door.z` | 0.25 | 0.20–0.30 | handle height changes with it |
| `robot.base_yaw` | −60° | ±20° | replaces the `DOOR_BASE` env var (`door.py:128-133`) |

### P2. Hinge side and handle side

- `door.hinge_side ∈ {left, right}`. The URDF generator mirrors the frame, hinge origin and latch origin, and moves the handle to the other edge.
- `door.handle_side ∈ {front, back, both}`. The current URDF has both levers (far one without collision, D11). The generator gives both collision and adds a ≥ 2 cm frame clearance.
- `door.open_direction ∈ {pull, push}`: hinge axis sign. A push door is a new behaviour; it is a stretch goal at M4.

Expected code impact: D2/D3/D6. Hinge-side inference must cope with a handle near either edge, and
the mirroring fallback in `guarded_pull` gets exercised.

### P3. Door and handle size / type

| Param | Nominal | Range |
|---|---|---|
| `door.width` | 0.40 | 0.30–0.60 m |
| `door.height` | 0.90 | 0.60–1.30 m (above ~1.1 m the "over the top" approach (D1) must be replaced by a "round the free edge / in front at handle height" approach) |
| `door.thickness` | 0.10 (Adroit) | 0.03–0.10 m |
| `door.frame_gap` | 0.06 | 0.003–0.06 m (D2: real doors ≈ 3-5 mm) |
| `handle.type` | lever | lever, knob (no latch: pull only), bar pull (cabinet), recessed/flush pull |
| `handle.length` | 0.20 lever | 0.10–0.25 m |
| `handle.height` | from URDF | ±0.10 m |
| `handle.offset_from_edge` | ~0.06 m | 0.04–0.15 m |
| `handle.color` | foil/silver | silver, black, wood-matched (low contrast) |

Implementation: `my_assets/adroit_door/build_door.py`, an xacro-like Python generator modelled on
`my_assets/insert_key/tools/build_assets.py`. It writes `tmp/door_<hash>.urdf` from `SceneParams`. Reuse the
Adroit meshes for the lever; use primitives for knob and bar. The nominal params must reproduce today's URDF
(test: element-by-element equal).

### P4. Hinge, latch and handle forces

| Param | Nominal | Range | Real-world reference |
|---|---|---|---|
| `hinge.rest_force` (velocity motor) | 2.0 | 0.5–8.0 | light interior door to sticky cabinet |
| `hinge.damping` / `friction` (URDF) | 1.0 / 2.0 | 0.2–3.0 / 0.5–5.0 | |
| `hinge.self_close` (spring to 0) | none | 0–2 N·m | soft-close doors, which fall back shut (exercises `swung_shut`) |
| `latch.spring` | 3.0 | 0.5–6.0 N·m | real levers 1-3 N·m |
| `latch.required_deg` (unlatch threshold) | none (no latch lock) | 0 / 20° / 40° | add a real latch: the hinge is locked until the lever is pressed past the threshold |
| `handle.friction` | 2.0 lateral | 0.6–2.0 | slip-prone metal handle |
| `door.mass` | URDF | ×0.5–×3 | |

The current skill says "press to unlatch". With no lock in the sim, that step is never tested. A
real latch (`latch.required_deg`) makes it testable, and force variation tests the slip/jam thresholds (D5).

### P5. Occluder (gray cylinder)

| Param | Nominal | Range |
|---|---|---|
| `occluder.shape` | cylinder | cylinder, box, thin board leaning on the door (`_load_board` exists), chair-like compound, 2 objects |
| `occluder.size` | r 0.15, h 1.0 | r 0.05–0.20, h 0.4–1.2 |
| `occluder.xy` | `[-0.15, 0.30]` | anywhere on the robot→handle line ±0.15 m lateral, not overlapping the door swing at spawn |
| `occluder.color` | gray `[0.5,0.5,0.55]` | random hue; also wood-like (matches the door: hard segmentation) |
| `occluder.mass` | 1.0 | 0.3–5.0 (heavy = not topple-able → must pick/slide or go around) |
| `occluder.partial` | full | partial occlusion (handle half visible), none |

This needs D8 fixed first: the planner few-shot example must not be the test scene. Replace it with a
generic "an object occludes the target" example that uses a different object (e.g. a box in front of a drawer).

### P6. USB port location / laptop pose

| Param | Nominal | Range |
|---|---|---|
| `laptop.xy` | `[0.30, 0.55]` | ±0.08 m |
| `laptop.yaw` | 0 | ±30°, plus 90/180/270° (port face toward any side) |
| `port.face` | −x (left side) | left, right, back |
| `port.offset_along_face` | −0.0541 | anywhere on the face with ≥ 2 cm margin |
| `port.height` | 0.019 above riser | 0.010–0.020 |
| `port.count` | 1 | 1–3 (multiple ports, any free one is success); dummy marks are already supported in the skill text |
| `plug.spawn_yaw` | tip +x (pre-aligned) | uniform 0–360° (U1) |
| `plug.spawn` | pedestal | pedestal, flat on the table (needs a different grasp), in a tray |
| `port.retention_n` | 12 | 6–20 N (co-tuned with U3) |
| `port.clearance` | 0.25/0.30 mm | 0.15–0.5 mm |

#### P6b. Two slots on one face: the right one and a look-alike (try wrong → recover → right)

A human facing a USB port next to an HDMI port often tries the wrong one, feels it does not go
in, and moves to the other. The robot must do the same with **the same code**. This tests candidate
ranking, failure detection by touch, memory of tried slots, and re-targeting.

| Param | Values | Notes |
|---|---|---|
| `slots` | `[usb_a, hdmi]` (default), `[usb_a, usb_c]`, `[usb_a, rj45]`, `[usb_a, usb_a_blocked]` | one accepts the plug; the other is a look-alike |
| `slot_order` | target first / look-alike first / random | look-alike first forces the recovery path |
| `slot_gap` | 8–25 mm centre-to-centre along the face | closer = harder to point at separately |
| `lookalike_geometry` | see below | decides which failure signal the robot gets |

Failure signals per look-alike (all from proprioception, none privileged):
- **USB-C** (8.4 × 2.6 mm): the USB-A tip cannot enter. Contact, then no advance, then `search_exhausted` or a jam. This is the clean signal; start here.
- **HDMI** (13.9 × 4.45 mm trapezoid): the USB-A tip (12 × 4.5 mm) can **partly** enter the wide top. In the sim, model the trapezoid's narrow bottom (10.4 mm) as collision so the plug wedges at ~2-4 mm. The insertion state machine already separates a wedge from a seat (`in_slot_depth` / `seat_depth`, `touch/insertion.py:58-62`), and the sideways verify check (`verify_shift`) catches a false seat.
- **RJ45**: big opening, wrong depth/shape. The plug enters but rattles: lateral `verify_shift` moves far more than `verify_tol`, so the outcome is `slipped`/not seated.
- **Blocked USB-A** (a cap or a dust plug): identical look; only touch can tell.

Required behaviour and code changes (generic, not USB-specific):
1. **Candidates.** `locate_with_wrist_camera("<part> opening")` must return **all** plausible openings with scores, not only the best one. `estimate_opening` clusters the hits into **separate openings** instead of rejecting the second cluster as outliers (today it median-fuses them; two slots 15 mm apart would fuse to a point between them). Return `openings: [{position, score, size_est}]`.
2. **Ranking.** The VLM ranks by the description in the task text ("the USB port"). Measured opening size from wrist depth, compared with the part width (`PartHint`), breaks ties. Any candidate is allowed to be wrong.
3. **Try loop with memory.** New helper `insert_into_candidates(openings, axis, ...)`: for each opening in rank order, call `insert_with_touch(...)`. If the result is not `seated` (or `seated` fails verify), back off, log `wrong_slot`, mark the opening as tried and move on. Cap the tries (default 3). Never press a wrong slot above `f_max` (35 N): a real connector is damaged by force, not by a wrong touch.
4. **Feedback to the LLM/reviewer.** Report the tried openings and their outcomes (`"opening 1 (left): jam at 2 mm, not seated; opening 2 (right): seated"`). The reviewer's success check is about the correct slot from the wrist image and the seated verify, not "any slot".
5. **Skill text.** Add the rule: "a port that will not take the part after one probe + search cycle is probably the wrong port; try the next candidate before re-grasping".

Evaluation (sim truth, eval only): `seated_in_target`, `wrong_slot_tries`, `max_force_on_wrong_slot`. Success = seated in the target slot. A rollout that seats on the first try with the look-alike first is fine (it means perception chose correctly). The suite must contain both orders, so the recovery path is exercised.

The same two-slot variant applies to P7 plugs: a Euro socket next to a USB-charger slot, or a power strip with one socket occupied or child-locked.

Generator: `scene_urdf()` is already parametric internally. Lift its constants into `SceneParams`, and
compute the pocket boxes in the **port frame** (face normal, along-face, up), then transform them to world.
Evaluation (`get_state`/`check_success`, `step_hook`) uses `tip` in the port frame instead of `tip[0] - FACE_X`.

### P7. New task: electric plug into a socket (same code as USB)

The goal is to prove that the insertion stack is part-agnostic. A new scene, `sim_envs/genesis/insertion.py`
(`SimEnvInsertion`), replaces the USB-specific module. `usb` becomes one `PartSpec`:

```python
@dataclass(frozen=True)
class PartSpec:                     # scene-side truth (evaluation + geometry), never sent to the agent
    name: str                       # "usb_a", "euro_2pin", "us_2pin", "uk_3pin", "dc_barrel"
    body: tuple                     # housing box (m) the hand grasps
    pins: list                      # [(shape: "box"|"cyl", size, offset_in_part_frame)], USB = one box tip
    holes: list                     # matching socket holes (+ clearance) in the face frame
    keyed: bool                     # needs roll alignment about the insertion axis (US/UK plugs, USB-A is 2-fold)
    retention_n: float              # friction-switch force while seated
    seat_depth: float
@dataclass(frozen=True)
class SocketSpec:
    mount: str                      # "laptop_side", "wall_plate" (vertical face), "power_strip" (floor, axis -z)
    face_pose: tuple                # world pose of the socket face (normal = insertion axis)
```

| Plug | Pins | Axis | What it adds over USB |
|---|---|---|---|
| Euro 2-pin (Type C) | 2 round Ø4 mm, 19 mm apart | horizontal into a wall plate | two-pin alignment; roll symmetric (2-fold) |
| US 2-pin (Type A) | 2 flat 6.3x1.5 mm, 12.7 mm apart | horizontal | roll matters (flat blades) |
| UK 3-pin | 3 rectangular | horizontal | keyed: 1-fold roll |
| Power strip | any of the above | **vertical (−z)** | gravity-aligned insertion; the head camera sees the face from above |

Code changes needed so that **the same code** passes USB and plugs:
- `touch/opening_estimate.py:35-38`: use the face frame from the **measured face normal**
  (fit a plane to wrist-depth points around the candidates), not the dominant world axis (U2).
- `touch/insertion.py` `InsertConfig`: derive the edge finder (`edge_min_sep/max_sep`), spiral radius,
  `seat_depth_max` and `verify_tol` from a **perceived or stated opening size** (a `PartHint(width_m, height_m,
  pin_len_m)` passed from the LLM, which reads it from the task text or wrist-camera measurement) instead of USB
  defaults (U3). Derive `f_insert` from a measured "free push" baseline plus a margin, capped below the
  measured grip-slip force. This removes the hard-coded "above 12 N" assumption.
- Roll alignment: new `align_part_roll(part_desc, opening_desc)`. It estimates the pin/slot angle in the wrist image
  (pointing two points on each), then rotates the hand about the axis. It is a no-op for 2-fold symmetric parts within tolerance.
- Skill text (U4): rename examples to "part"/"opening"; keep USB, Euro plug and power strip as three short examples;
  remove `axis=[1, 0, 0]` literals and use `axis=est["axis"]` from the estimate.
- Test (G4): one parametrised e2e (`tests/test_touch_insertion.py` style, sim_runner) over `PartSpec ∈ {usb_a,
  euro_2pin, us_2pin}` × `face ∈ {side, wall, strip}` with the **same** controller calls.

### P8. Camera viewpoint, calibration and depth realism

| Param | Nominal | Range |
|---|---|---|
| `head.yaw / pitch / distance / target` | per task (`door.py:25-28`, `usb.py:95-101`) | yaw ±35°, pitch −20…−60°, distance ×0.8–×1.3, target ±5 cm |
| `head.fov` | 60 | 45–75 |
| `shoulder` offsets | per task | ±10 cm, ±15° |
| `calib.extrinsic_err` (given to the agent, render uses truth) | 0 | 0.5–1 cm, 0.5–1° (hand-eye error) |
| `depth.noise` | none | σ = 1 mm·z² (stereo-like), 2% dropout holes, edge flying-pixels |
| `image.resolution` | 256 | 224–512 |

Visibility constraint (sampling-time, evaluation side only): project the sim-truth bounding boxes of the
**target parts** (handle/lever, port face) into the head image. Require ≥ 60% of the projected area
to be inside the frame. Unless the variant is an occlusion variant, also require no full occlusion by static scene geometry
(ray-cast test). Resample the camera on failure (max 50 tries), and log the attempts.
Calibration error is applied only to the matrices sent over `CAPTURE_IMAGES`, which models a real hand-eye error.

### P9. Lighting, shadows and visual distractors

| Param | PyBullet | Genesis |
|---|---|---|
| shadows on | `getCameraImage(..., shadow=1, lightDirection=d, lightColor, lightDistance)` | `gs.options.VisOptions(shadow=True)`, `scene.add_light(...)` |
| light direction / intensity / colour | `lightDirection`, `lightAmbientCoeff`, `lightDiffuseCoeff`, `lightSpecularCoeff` | light pos/dir/intensity |
| cast shadow over the handle / port | add an overhead "lamp shade" or shelf body outside the arm workspace | same |
| textures | swap door wood / floor / laptop textures from a small bank (`my_assets/textures/`) | same |
| distractors | see P10 (all tasks) | same |
| image post-processing | in `env.py` `CAPTURE_IMAGES`: exposure ±30%, white balance, Gaussian noise, motion blur on the wrist cam, JPEG q=60 | same |

### P10. Distractor objects (every task, not only USB)

Today every scene holds only the target objects plus at most one occluder, so detection,
pointing and segmentation have nothing to confuse them. Distractors are a generic layer added to
**every** task (door, USB, plug, grasp, kettle pour, insert_key, Franka Kitchen), with the same
sampler, seeds and logging.

| Kind | Purpose | Examples per task |
|---|---|---|
| **Semantic look-alikes** | wrong-target detection; the task text must name the target uniquely | door: a second (cabinet) door or drawer handle, a decoy bar handle on the wall. USB: a second USB stick of another colour, a lighter or eraser of USB size, a sticker or port-shaped mark (dummy faces already in the skill). Plug: other plugs and cables. Pour: 2–3 mugs (target given by colour), a second kettle-like jug. Grasp: same-shape cubes of other colours. insert_key: a second key / lock. |
| **Clutter** | segmentation masks and bounding cubes merge with neighbours | YCB-like objects (bowl, can, box, banana) near the target, touching it, or on top of the laptop |
| **Path obstacles** | motion and the guards: hits must be seen and avoided, not pushed through | an object in the door swing arc (the door stops: `blocked`), a mug next to the pour target, a cable lying across the plug approach |
| **Background / visual** | colour/texture confusion | a gray box matching the occluder colour, a wood-textured panel matching the door, a dark rectangle matching the port opening |
| **Moving / late** (stretch) | re-observation, tracking | a distractor dropped into the scene mid-task (the hand must re-detect) |

Implementation: `sim_envs/distractors.py`.
- An asset bank of primitives, plus a few low-poly meshes under `my_assets/distractors/`. Each has a name, size, colour range, mass and an optional `lookalike_of` tag.
- Each task declares keep-out zones in its params. Examples: the target grasp region, the door swing disc, the plug approach corridor, and the camera visibility cone of the target (P8 visibility check).
- Each task also declares the **look-alike set** that makes sense for it.
- `place(sim, params.distractors, keep_out, rng)` drops objects onto the table or floor with rejection sampling, then lets them settle.
- Params: `count` (0–6), `kinds` (subset of the table), `lookalike_count`, `near_target_m` (minimum distance to the target, down to touching).
- Distractor names and poses are `[EVAL-GT]` only.
- Evaluation adds `touched_distractor`, `moved_distractor_m` (sim truth) and `picked_wrong_object`. Moving a distractor is not a failure by itself; toppling it or using it as the target is.
- Task text rule: when a look-alike is present, the task must name the target with features a human could see ("the red mug", "the USB port, not the HDMI one"). The sampler refuses variants where only privileged info could separate the two.

Decoys matter most. They test the pointing/segmentation step and the "dummy marks on several faces"
logic, not only the controller.

## 5. Code-level changes and integration points

### 5.1 New / changed files

| File | Change |
|---|---|
| `sim_envs/perturb.py` (new) | `SceneParams` dataclasses per task, `VARIANTS = {"door/nominal", "door/pose", "door/hinge_right", "door/size", "door/forces", "door/occluder", "door/camera", "door/visual", "door/mix", "usb/...", "plug/..."}`, `sample(variant, seed)`, `to_json()` |
| `my_assets/adroit_door/build_door.py` (new) | Door URDF generator (P2/P3/P4); nominal == current URDF |
| `sim_envs/pybullet/door.py` | `SimEnvDoor(sim, hide_door_with_object, params=None)`; every constant at lines 71-136 and 25-28 is read from `params`; occluder factory (P5); `get_state` adds `variant` |
| `sim_envs/genesis/insertion.py` (new) | `SimEnvInsertion(params: PartSpec, SocketSpec)`; `usb.py` becomes `SimEnvInsertion(USB_A, LAPTOP_SIDE)` for back-compat |
| `sim_envs/registry.py` | tasks `plug:euro_wall`, `plug:us_strip`; pass params through |
| `sim_adapter/base.py`, `pybullet_adapter`, `genesis_adapter` | `set_lighting(...)`, `enable_shadows(bool)`, `add_light(...)`; texture swap helper |
| `env.py` `CAPTURE_IMAGES` (~905-920) | Optional image/depth post-processing + calibration error from params. Strip `sim_state` from the handshake (S1). |
| `agent_runner.py:181-205` | Prefix GT world-state lines with `[EVAL-GT]` (S2) |
| `helpers/main_utils.py:163` | Drop `[EVAL-GT]` lines in `learn_from_past_trajs` (S2) |
| `touch/sim_runner.py:103-107` | Force bias/scale error params (S3) |
| `tools/perturb_bench.py` (new) | Runs the matrix via `main.py` subprocesses, then aggregates `summary.md` outcomes into a CSV/markdown table |
| `sim_envs/distractors.py` (new) + `my_assets/distractors/` | P10 sampler, asset bank, keep-out zones; called from every SimEnv's `load_assets` when `params.distractors` is set |
| `touch/opening_estimate.py` | Cluster candidates into separate openings (`openings: [...]`) instead of fusing them (P6b) |
| `helpers/insert_candidates.py` (new) + `api.insert_into_candidates` | Try-loop over ranked openings with a memory of tried slots and per-slot outcomes (P6b) |

Control-code de-overfitting (needed before the suite can pass):

| Finding | Change |
|---|---|
| D1 | `_approach`: choose between "over the top" (if `top_z` is reachable + margin) and "in front at handle height, around the free edge"; the decision is made from measured `top_z` and IK reachability |
| D2 | `jamb(max_m)` from measured panel thickness / depth resolution; default ≤ 0.02 m; keep 0.10 only as a fallback with a lower confidence |
| D3 | Rose / lever window from measured geometry; add handle-type branch (lever, knob, bar) chosen from the VLM's handle description; hinge-side confidence includes "handle centred → ambiguous" |
| D4 | `goal_deg` comes from the task (planner sets it: "wide enough to pass" vs "ajar"); `partly_open` = `deg < goal_deg − margin` (no literal 55) |
| D5/D6 | Express `behind_deg/ahead_deg/hover/below` and the hinge-hint clamp in terms of measured panel width and thickness |
| D7 | Prompt example uses symbolic values: `guarded_pull(direction=p["pull_dir"], hinge=p["hinge"])` |
| D8 | Replace the planner few-shot example with a different scene/object |
| D9, U4 | Move rollout anecdotes from SKILL.md to `docs/status/`; keep the rules, drop the numbers or make them relative ("~1/4 of the lever length") |
| U2, U3 | §P7 bullets |

### 5.2 CLI args

| Flag | Default | Meaning |
|---|---|---|
| `--scene-variant` | `nominal` | variant name from `sim_envs/perturb.py` |
| `--scene-seed` | `0` | sampling seed (scene only; does not touch agent sampling) |
| `--scene-params` | none | JSON file/string overriding individual params (debug a failing seed) |
| `--visual-noise` | off | enable P9 image post-processing independent of the variant |

## 6. Sim2Real rules

- `SceneParams` are **evaluation truth**. They are logged with `[EVAL-GT]` and never sent over IPC to the agent, so they never reach a prompt.
- Visibility resampling uses sim truth at scene build only. The agent never learns why a camera pose was chosen.
- The plug's `PartSpec` sizes are not given to the LLM. If a task needs part knowledge, it goes in the
  **task text** (as a human would say "a Euro plug"), or the agent measures it from the wrist camera.
- Calibration and depth noise model the real hand-eye error (0.5-1 cm) and stereo depth; no thresholds may assume sub-cm agreement.
- Physics stays honest: generated levers and knobs have collision on both sides; latch and retention are joint/friction models, never welds.

## 7. Logging / observability

- One line at scene build: `[EVAL-GT] scene variant=<v> seed=<s> params=<json>`.
- Visibility check: `[EVAL-GT] visibility tries=<n> handle_in_frame=<frac>`.
- `rollout_report` writes `variant`, `seed` and the main params into `summary.md`. HF `metadata.jsonl` `notes` gets `variant=...;seed=...` (keep the fixed column set).
- `perturb_bench` table: per variant, success / N, median wall time, top-3 failure reasons
  (from reviewer verdicts and helper outcomes such as `grasp_failed`, `no_door_face`, `search_exhausted`).

## 8. Tests and evaluation

Unit (fast, no LLM):
- `tests/test_perturb_params.py`: `sample()` is deterministic per seed; nominal == today's constants; the door URDF generator nominal == checked-in URDF.
- `tests/test_door_plane.py`: add synthetic point clouds for a mirrored hinge, a small frame gap, a knob, a centred handle and a yawed door (D2/D3).
- `tests/test_opening_estimate.py`: yawed face (U2); two hit clusters 15 mm apart give two openings, not a fused midpoint (P6b).
- `tests/test_touch_insertion.py`: wrong-slot sim_runner case. A USB-C look-alike first gives `wrong_slot` on opening 1, then `seated` on opening 2, with peak force ≤ `f_max`.
- `tests/test_distractors.py`: placement is deterministic per seed, respects keep-out zones, and the objects settle without overlap.
- `tests/test_touch_insertion.py`: parametrise over `PartSpec` (P7) with the sim_runner.
- `tests/test_sim2real_boundary.py`: `api` has no `sim_state`, and the main/planner/review prompt builders produce no GT keys (S1/S2).

Integration (LLM, sim): the matrix below, ≥ 5 seeds per cell at first, 10 for reporting.

| Task | Variants (one axis each) |
|---|---|
| `door:hidden` | nominal, pose, hinge_right, size, forces, latch, occluder, camera, visual |
| `usb` | nominal, port_face, laptop_yaw, plug_yaw, two_slot_usbc (look-alike first), two_slot_hdmi (look-alike first), camera, visual |
| `plug` | euro_wall, us_wall, euro_strip, euro_next_to_usb_charger |
| all tasks (incl. grasp, pour, insert_key) | `distractors/clutter`, `distractors/lookalike`, `distractors/path` |

Report the success-rate ratio vs nominal (G2/G3). Package each run under `outputs/<date>/<task>_<variant>_s<seed>/`, as usual.

## 9. Milestones

| M | Content | Exit |
|---|---|---|
| M0 | Sim2Real hygiene: S1, S2 (handshake strip, `[EVAL-GT]` filter), D7, D8 prompt fixes | boundary test passes; nominal door still passes |
| M1 | `perturb.py`, door generator (nominal-identical), P1 pose, P2 hinge side, P5 occluder, P8 camera | bench runs; baseline numbers recorded (expect failures; that is the point) |
| M2 | Control de-overfitting D1-D6, P3 size/handle type, P4 forces + real latch | door G2 met on pose / hinge / occluder / camera |
| M3 | `SimEnvInsertion` + USB variants P6, U2/U3 fixes, P6b two-slot (USB-C first, then HDMI) with `insert_into_candidates`, euro/us plug P7 | G3, G4, G5b |
| M4 | P10 distractors for all tasks, P9 lighting / shadows / image noise, depth noise, calibration error, `door/mix` | G5, G5c; combined-variant numbers in `docs/status/` |

## 10. Risks and open questions

- **Reach.** Larger or moved doors leave the fixed-base Franka workspace quickly. Sampling must IK-check the handle and the swing path, otherwise "failures" are infeasible scenes. Real-size doors depend on the mobile base.
- **Genesis vs PyBullet parity.** Shadows and lights differ. Keep the perturbation visual params per sim and report results per sim.
- **Mirrored Adroit meshes.** Negative-scale meshes flip normals in some renderers. Prefer re-posing the hinge/handle joints over mirroring meshes.
- **Cost.** Each LLM door run is 5-10 min. The full matrix (~25 cells × 10 seeds) is ~30 h of runs; start with 5 seeds and the cheaper models.
- **Latch realism.** Adding a real lock (P4) may break the nominal success, so it is a separate variant, not the new nominal.
- **Open:** should `goal_deg` come from the planner (task semantics) or stay a tool default? Proposal: the planner states it; the tool default is 65 only as a fallback.
