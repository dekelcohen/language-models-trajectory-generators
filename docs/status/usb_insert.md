# USB insert (Genesis): status

Task: `--task usb --sim genesis -c "Pick up the USB plug and insert it into the laptop's USB port"`.
Skill: `prompts/skills/touch-guided-insertion/SKILL.md`, primitive `insert_with_touch(axis, tool_length, max_s, grip_force, tool_offset=None)`.

**Status: OPEN.** The pick-up works every time. Insertion has not succeeded in an LLM run, but the scripted harness `research/touch/usb_pipeline_e2e.py` succeeds when given the true in-hand offset.

## Runs (2026-11-02, rollouts under `outputs/2026-11-02/`)

| run | models (lm / perc / review) | result | root cause | folder |
|---|---|---|---|---|
| usb_gn1 | claude / terra / terra | FAIL, 12+ `search_exhausted` | Skill told the LLM to aim at "opening z + 0.06" (the pick-up rule); after the grasp the plug sits at about the EE height | `usb_terra_FAIL_inhand_offset` |
| usb_tt2 | claude / terra / terra | FAIL: 2× timeout, `no_contact`, plug dropped, then a ~40 min loop with no `task_completed()` | Wrist cam could not see the held tip; fallback offset (0.025, 0, 0) is off by ~7 mm in z; the LLM refused to end the attempt | `usb_ALLterra_FAIL_vertical_offset_then_drop_idle_loop` |
| usb_tt3 | claude / terra / terra | FAIL: 5× timeout, 4× `no_contact`, 10+ `search_exhausted` in ONE attempt (66 min, never reviewed) | Wrist locate of the held tip gave impossible offsets (tip 13 cm above the hand), which the LLM rightly rejected. The height scan (0, +5, −5, +10 mm) plus an alternate port did not land within the 3 mm spiral | `usb_ALLterra_FAIL_hidden_inhand_offset_endless_attempt` |

## Fixes made
- Skill step 3: measure the in-hand offset with the wrist camera from 2 views (`offset = tip - get_robot_state()["eef_pos"]`), aim with `target = opening - offset - axis*0.004`, and pass `tool_offset=offset`.
- Re-measure if the first contact z is off by more than 5 mm.
- If the tip is not visible, scan heights 0, +5, −5, +10 mm.
- `insert_with_touch(..., tool_offset=...)`: a world vector from the EE to the tip; it overrides `tool_length`.
- `get_robot_state()` / `gripper_holding()` are exposed in the exec scope.
- Agent loop: an idle guard (`max_idle_agent_turns=10`) and a per-attempt cap (`max_agent_turns_per_attempt=25`), so endless retries get reviewed.

## Root cause still open
The plug's pose in the hand (true ≈ eef + (−9, 0, +7) mm) is **unobservable by vision while held**:
- the fingers and the hand occlude the tip in the wrist camera;
- the head camera is too coarse (~1 cm).

The `insert_with_touch` spiral covers only r ≤ 3 mm (`InsertConfig.spiral_r_max`), so offset errors above ~3 mm never seat.

Next steps (Sim2Real-legal):
1. **Touch-based offset calibration.** Add a guarded move: lower the held plug onto a flat surface (laptop deck or floor, height from depth) until the force rises. Contact z minus EE z gives the vertical offset. A side touch would give x.
2. Expose a search radius (`spec["config"]={"spiral_r_max": r}`) for a coarse 6-8 mm search. It is slow (~100 s at 8 mm), so use it only as a fallback.
3. Grasp the plug at a known point (centred, fixed depth), so the offset is predictable from the grasp pose.

Geometry (sim truth, for eval only): port (0.092, 0.4959, 0.069), axis +x, `TIP_X` = 0.0245.
