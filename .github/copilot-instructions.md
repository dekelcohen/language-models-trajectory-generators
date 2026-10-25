# Copilot instructions

VLM-driven robot manipulation (planner -> LLM trajectory code -> VLM reviewer) in PyBullet /
Genesis / MetaWorld sims. Full design notes: `document.md`. Living status logs: `docs/status/`.

## Sim2Real

Goal: the sim pipeline must run unchanged on a real Franka Panda / FR3. The robot operates
from **pixels + proprioception only**.

**Allowed inputs to the planner / LLM / reviewer / retry prompts** (all available on the real robot):
- Camera images (RGB, depth) and calibrated camera intrinsics/extrinsics.
- Proprioception: EE pose (forward kinematics), joint positions, gripper width / holding,
  joint torques / estimated external force, the arm base mount pose.
- Anything derived from those (detections, tracking, point clouds), and the task text.

**Never leak privileged sim info** into any prompt or decision path: `simenv.get_state()` /
`sim_state` object poses, articulation joint values (door angle, hinge axis/arc), success
criteria / `task_error`, particle counts, URDF geometry, engine contact lists. These are only
for logs, evaluation and tests. Tests must not hand the agent such info either (e.g. no true
hinge as a starting assumption).

**Keep the physics honest**: no weld / attach-constraint grasps or other cheats; objects are
held by contact + friction only.

**Does the real robot know its gripper 3D position?** Yes. libfranka reports the EE pose
(`O_T_EE`) at 1 kHz from the joint encoders + forward kinematics, and the Franka Hand reports
the finger width.
- Repeatability (returning to a taught pose) is < ±0.1 mm (FR3 datasheet).
- Absolute accuracy is not specified by Franka. Uncalibrated it is typically a few mm (kinematic
  model errors, link/gripper deflection under payload).
- The dominant real error is usually the camera-to-robot (hand-eye) calibration, ~0.5–1 cm.
  That error applies to every object position computed from pixels.

So reporting the EE pose in prompts is realistic. Sim gives it exactly; do not rely on
sub-cm agreement between the measured EE and camera-derived object positions.
