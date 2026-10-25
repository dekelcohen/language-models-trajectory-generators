# Touch sensing — status & findings (branch `tactile`)

Goal: touch feedback for contact-rich tasks — **USB peg insertion** (primary), gripper **slip**, and
**joint-direction following** (doors/drawers/kettle). Constraints: no privileged sim info to the robot
(pixels + proprioception + buyable sensors only, ~$1–2k), no physics cheats, gemini-3.8-flash for all models.

## 1. Does the simulator support touch?

| | PyBullet | Genesis 1.3.3 (source, `vlm_genesis` env) |
|---|---|---|
| Joint torque / motor current | `getJointState` reaction + applied torque (position ctrl) | `JointTorque` sensor (noise configurable) + `get_dofs_actuator_force` |
| Contact force | `getContactPoints` (normal + lateral friction) | `ContactForce` sensor, `get_links_net_contact_force`, `get_contacts` |
| Tactile array | no — needs **TACTO** / **Tactile Gym** / **Taxim** | built-in `KinematicTaxel`, `ElastomerTaxel` (GelSight-like), `ProximityTaxel`, `ContactDepthProbe` |

Extensions (to evaluate): TACTO (facebookresearch, PyBullet, DIGIT/OmniTact render), Tactile Gym
(dexterousrobot, PyBullet, TacTip), Taxim (GelSight example-based), TacSL (Isaac Gym), DiffTactile
(Genesis-Embodied-AI, Taichi FEM), ManiSkill-ViTac (SAPIEN+IPC, has peg-insertion), MuJoCo touch grid.
**Decision: Genesis** — sensors built in; no extra package needed.

## 2. Does motor current help? — Yes (measured, `research/touch/genesis_touch_probe.py`)

- **Wrench from joint torques** `W = -pinv(J^T)(tau - tau0)` (tau0 = free-space reading at the pose) tracks the
  true end-effector Fz within ~2–10 % at 0.1 Nm torque noise; contact detected at the same 0.5 mm step as
  ground truth (threshold 3 N). This is what a real Panda gives (`O_F_ext_hat_K`, ~1–2 N resolution).
- In the USB pushes the felt force cleanly separates outcomes: free motion ≈ 1–4 N, retention friction
  ≈ 12–16 N axial, jam ≈ 20–23 N axial **plus a lateral component** (8.8 N at 15°, 15.6 N at 30° push error).
- Low-cost arms (SO-101, STS3215 servos): present-load/current is coarse (high gear ratio) — usable for
  contact/jam/stall detection, not for fine force control. Add a wrist F/T or tactile pad for insertion.
- Literature: **DoorBot** (arXiv 2504.09358) — elbow motor current for stop thresholds, push-vs-pull
  classification (current rising/falling during a small probe), GRASP_MISS, COLLISION. **Vi-TacMan**
  (arXiv 2510.06339) — vision gives grasp + coarse direction distribution (vMF), a 50 Hz GelSight
  contact-regulation controller keeps contact → follows articulated joints without a kinematic model.

## 3. USB insertion test bench (Genesis)

Assets: `sim_envs/assets/usb_laptop/` (made by `research/touch/make_usb_assets.py` from the user's
`my_assets/79-obj.rar`, low-poly HP notebook, extracted to `research/touch/assets/laptop_src/`).
The laptop has a USB port drawn on its left side texture; it was located by a straight-on render and stored
as `usb_port_center_local` in `laptop_meta.json`. Collision = riser + 5 boxes forming a real-size USB-A
pocket (12.5 × 5.1 × 14 mm, i.e. 0.25 mm / 0.3 mm clearance per side for the 12.0 × 4.5 mm tip). Plug =
`usb_plug.urdf` (housing 25×16×8 mm, tip 12×12×4.5 mm, 12 g). The `my_assets/10-usb.zip` model is Cinema 4D
only (not loadable) → the plug is built from primitives.

### Physics fixes needed for realism (none leak info to the robot)
| Problem | Fix |
|---|---|
| Arm sags 2.8 mm under position control | `gs.materials.Rigid(gravity_compensation=1.0)` on the Franka (= real controller's g(q)) |
| Plug creeps out of a 20 N grasp at ~4 N (soft friction) | `RigidOptions(noslip_iterations=5)` → grip now slips at **24 N = 2·μ(0.6)·20 N**, exact Coulomb |
| 12 g plug makes every constraint soft vs a 2 kg arm (misaligned plug squeezes through walls, dof frictionloss degrades to ~0.3 N viscous drag) | plug dof **armature** 0.5 kg (inertia only, not weight) + stiff joint `sol_params` for the frictionloss row |
| IK only accurate to 0.5 mm (default `pos_tol`) | `pos_tol=1e-5` in the probe |
| Isotropic soft impedance lets the hand drift sideways 0.7 mm in a light push | anisotropic Cartesian impedance: K along the push, 2000 N/m lateral (like a real insertion controller) |

### USB retention model
Real USB-A: insertion ≤ 35 N, extraction ≥ 10 N, typically 10–20 N (spring tabs). Modelled as Coulomb
dry friction `PORT_FRICTION_N = 12 N` (+0.05 Nm rotational) on the plug's free-joint dofs, switched on while
the tip is ≥ 0.5 mm inside the port (env physics only; never exposed to the robot).

### Push sweep (Cartesian impedance, perfect start alignment, `--exp push`)
| case | felt Fx | depth | result |
|---|---|---|---|
| light, K 250 N/m (≤ 6 N) | 5.9 N | 0.8 mm | **fail** (held by retention) |
| medium, K 600 (≤ 11 N) | 11.4 N | 4.2 mm | **fail** (just below 12 N) |
| firm, K 1500 aligned | 15.9 N | 12.2 mm | **IN** |
| push dir 15° off-axis | 21.7 N, Fy 8.8 | 0.4 mm | **jam** |
| push dir 30° off-axis | 22.6 N, Fy 15.6 | 0.4 mm | **jam** |
| plug axis yawed 4° | 14.5 N | 10.7 mm | IN (compliant wrist + in-grip rotation self-aligns — realistic) |
| axis 4° + push along it | 14.3 N | 6.3 mm | partial |
| plug axis yawed 8° | 16.3 N | 4.2 mm | **fail** |
| lateral offset 1.0 / 1.5 mm | 22.4 N | 0.5 mm | **jam on the face** |

When jammed the plug slides back 7–11 mm in the grasp → a measurable **slip** cue (finger taxels / width).
Signature summary: *light* = low force + no progress; *misaligned/off-axis* = high axial force + lateral
force/torque + no progress + in-grip slip; *success* = force plateau at retention level then depth stops at
pocket bottom (force rises with no motion).

### Reactive spiral insertion (`touch/`, `--exp search`)
Sim-agnostic package: `touch/state.py` (`TouchState`: EE pose, external wrench est., width, holding) and
`touch/insertion.py` (`SpiralInsertController` approach -> probe -> spiral search -> insert -> backoff,
`run_reactive(...)` loop). Inputs are proprioception + the estimated wrench only; sim truth is used by the
bench just for scenario setup and scoring. Unit tests with quasi-static mock ports: `tests/test_touch_insertion.py`.

Findings on the Genesis bench (start error <= 2 mm lateral, +-2 deg yaw, 0.5 N force noise):
- **Compliance centre must be the TCP (plug tip)**, not the flange: with Kr=40 Nm/rad a push 10 cm below
  the flange tilts the hand ~3 deg and fakes "advance". Bench now uses `get_jacobian(local_point=tcp)`.
- **Lateral stiffness**: >= 5000 N/m while spiralling (face friction makes a soft hand lag the spiral);
  800 N/m once in the hole (RCC-like) lets the chamfer centre the plug.
- **Pinch grasp pitches** about the finger-normal axis under tip friction (tip rises ~4 mm, ~9 deg) -> the
  spiral loses its reference -> most remaining `search_exhausted`. Observable only with in-hand
  sensing (GelSight marker rotation / wrist cam) -> strongest argument for tactile pads.
- **Grasp creep** at 17-18 N: ~2-3.5 mm/s both when seated and when wedged (free insert 20-40 mm/s).
  Stall/creep detectors confuse the two; rule used: judge the depth the *first* time motion turns slow
  (< 6 mm/s over 0.2 s): advance >= 9 mm -> seated, else jam -> backoff -> search.
- Grip 30/40 N is worse (soft-contact penetration); keep 20 N. Lower search force (1.5 N) is worse.

| method (3 seeds x 5 trials) | truly inserted | notes |
|---|---|---|
| naive firm push | 2-3/15 | |
| touch spiral (default) | 10/15 | 1 false "seated" (9.5 mm), 2 seated-but-reported-exhausted, 1 jam |
| touch, `--f_search 1.5` | 8/15 | more `search_exhausted` |

## 4. Hardware shortlist (≤ $1–2k)
Panda built-in external wrench estimate (free) · DIGIT (~$350/pad) · GelSight Mini (~$500) · AnySkin (DIY,
cheap, magnetic) · low-cost wrist F/T (e.g. Bota/OnRobot are above budget; DIY strain-gauge or SO-101
servo load as a coarse fallback).

## 5. Open items / next steps
- [ ] Grasp-pose realism: plug can tilt 1–2° / sit ~1.7 mm low in the grasp after transport — the robot must
      correct from vision + touch (spiral/tilt search), not from sim truth.
- [x] Touch API: `touch/` (`TouchState`, `SpiralInsertController`, `run_reactive`) with exits (no contact,
      over-force, jam, slipped, search exhausted, timeout). TODO: finger taxels / slip in `TouchState`,
      `review_every_s` VLM hook.
- [ ] Insertion robustness: detect in-grasp pitch (taxels) and re-reference the spiral; fix missed seats
      after search (2/15).
- [ ] USB env through the adapter API, visible vs half-occluded port; eval = insertion depth.
- [ ] Touch skill (`prompts/skills/touch-guided-contact/SKILL.md`), LLM runs (gemini-3.8-flash).
- [ ] Slip + joint-direction follow-ups (kettle grasp, `helpers/pull_proprio.py`).

## Reproduce
```
$env:PYTHONIOENCODING="utf-8"
C:\Users\dekelco\AppData\Local\miniconda3\envs\vlm_genesis\python.exe research\touch\genesis_touch_probe.py --exp push   # ~150 s CPU
...  --exp grip | press | slot | slip | creep | render
...  --exp search --trials 5 --seed 0 --methods naive touch   # ~5-10 min per seed
python -m pytest tests\\test_touch_insertion.py -q
```
