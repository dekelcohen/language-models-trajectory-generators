# insert_key task (RoboDojo port, single-arm Panda)

Pick key from stand -> insert into lock keyway -> turn >= 60 deg. Adapted from
RoboDojo `insert_key` (bimanual) to our single Franka Panda. Key mesh/texture from
RoboDojo (Apache-2.0); table, stand and lock are procedural (`my_assets/insert_key/tools/build_assets.py`).

## Status

| Sim | Pick | Insert | Turn (success) |
|---|---|---|---|
| PyBullet | OK | OK (44-50 mm, xy 1-2 mm) | OK 3/4 scripted runs (plug 85-93 deg); 1 fail: plug twisted -39 deg during insertion |
| Genesis | OK | OK (48 mm, xy 2 mm) | OK, thin margin: plug 64-66 deg (3/3 scripted runs), needs >= 60 |

Tests: `tests/test_insert_key.py` (3, PyBullet) pass. Full suite: no new failures
(same 15 pre-existing/flaky failures with and without this branch's changes).

## Run

```powershell
python main.py --task insert_key --sim pybullet -r franka -c "Pick up the key, insert it into the keyhole, then turn it"
python env.py --task insert_key            # viewer
```

Assets: `my_assets/insert_key/tools/extract_robodojo_usdz.py` (stage 1, needs `usd-core`,
output to gitignored `cache/robodojo/`), then `build_assets.py` (stage 2) regenerates
URDF/OBJ/textures.

## Genesis turn - findings so far

- Genesis gives URDF joints armature 0.1 kg m^2 (plug I ~4e-5) -> plug nearly immovable.
  Fixed: new adapter `set_joint_armature` (PyBullet no-op), plug armature 1e-4.
- Genesis `control_dofs_velocity` damps implicitly with `kv` regardless of force range ->
  the "zero-velocity motor = joint friction" PyBullet idiom makes the joint near-rigid
  (test: 0.08 Nm for 0.5 s -> 5 deg with velocity hold vs 42 deg with `frictionloss`).
  Fixed: new adapter `set_joint_friction` (PyBullet: velocity motor; Genesis: `frictionloss`).
  Note: other profiles using `set_joint_velocity(0, force=...)` as friction have the same issue in Genesis.
- Key (30 g) held by the stiff arm penetrated the keyway walls ~2 mm (soft contact) and slipped
  past them: mesh blade is chunky (12.5 x 10.3 mm), half-diagonal 8.25 vs slot half 8.0 x 6.6 mm.
  Fixed: collision-only flat blade (3 mm, `KEY_BLADE_COLL_HY`) in a 4.4 mm keyway
  (`KEYWAY_COLL_HY`); visuals unchanged. A rigid key at 20/45/70 deg now shoves the plug to ~angle-12.
- Remaining Genesis loss: key slips in the soft finger grasp under plug friction.
  Fixed: per-sim plug friction `PLUG_RESIST_TORQUE = {"pybullet": 0.05, "genesis": 0.01}`
  (different friction models anyway).
- Tried and rejected: heavier key (0.1 kg) -> tilts in grasp, insert fails; shared 0.02 N*m -> both sims flaky.
- Genesis textures rendered white: trimesh ignores `map_Kd ../textures/x.png`. Fixed: textures live in `meshes/`.
- Scripted demo (scratch, not in repo) needed: approach waypoint high enough for the 46 mm blade,
  closed-loop tip-over-keyway xy correction before descending.

## Next steps

1. PyBullet: plug can be twisted by funnel contact during insertion. Option: pin-tumbler-like
   detent (plug blocked until key fully seated) or gentler funnel.
2. Genesis margin is thin (65 vs 60 deg): reduce key slip in grasp (finger friction/force) or
   turn past 90 deg in prompts/examples.
3. Fix Genesis textures: insert_key meshes render white (MTL `map_Kd` not picked up).
3. Scripted demo with fixed seed for deterministic comparison; add a Genesis test.
4. Update `document.md` with an insert_key section.
