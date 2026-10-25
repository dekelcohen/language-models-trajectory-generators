# insert_key task (RoboDojo port, single-arm Panda)

Pick key from stand -> insert into lock keyway -> turn >= 60 deg. Adapted from
RoboDojo `insert_key` (bimanual) to our single Franka Panda. Key mesh/texture from
RoboDojo (Apache-2.0); table, stand and lock are procedural (`my_assets/insert_key/tools/build_assets.py`).

## Status

| Sim | Pick | Insert | Turn (success) |
|---|---|---|---|
| PyBullet | OK | OK (44 mm, xy 3 mm) | OK (plug 94 deg, holds after release) |
| Genesis | OK | OK (43-50 mm) | **INCOMPLETE** - plug reaches 8-32 deg, needs >= 60 |

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
- Remaining: key (30 g, izz 2.5e-6) held by the stiff arm penetrates the keyway walls ~2 mm
  (soft contact for light bodies) and slips past them. Blade is chunky (12.5 x 10.3 mm):
  half-diagonal 8.25 mm vs slot half 8.0 x 6.6 mm, so ~1.6 mm penetration is enough to slip.
- Tried: heavier key (0.1 kg) -> key tilts in grasp, insert fails (reverted).
  Per-geom `sol_params` already at defaults (timeconst 2*dt) -> no stiffening headroom at dt=1/240.
- Results vary run-to-run (randomized layout, seed=None): 8 deg vs 32 deg.

## Next steps

1. Add a hidden collision "ward" fin to the key blade (thin, wide) + matching keyway notch,
   so slipping needs > 3 mm penetration; or tighten slot clearance.
2. Alternatively smaller sim dt / substeps for this task (stiffer contacts) if the adapter allows per-task.
3. Fix Genesis textures: insert_key meshes render white (MTL `map_Kd` not picked up).
4. Scripted demo with fixed seed for deterministic comparison; add a Genesis test.
5. Update `document.md` with an insert_key section.
