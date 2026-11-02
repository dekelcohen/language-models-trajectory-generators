---
name: touch-guided-insertion
scope: subtask
description: >
  Pick up a small peg/plug (USB stick, connector, pin) and insert it into a tight hole or port
  using force feedback. Load it for the PICK-UP subtask too (grasp depth rule). Vision alone is
  ~1-2 mm off; this finds the opening by touch. Gives locate_with_wrist_camera(), insert_with_touch().
license: MIT
metadata:
  category: manipulation
  tags: [insert, usb, plug, peg-in-hole, touch, force]
  requires: [touch]
---

# Touch-guided insertion

1. Grasp the part across its body (not the tip), fingertip centre near its middle; lift; check
   `gripper_holding()` (True = closed fingers stalled on an object, measured finger width; trust it
   over a wrist/head image of the fingers, which the hand itself occludes).
   Depth: the fingertips end ~0.06 m below the commanded pose z, so command z = part centre z + 0.06
   (z = part top crushes it; lower drives the palm into the part and flings it). Get the part top
   from the HIGHEST affordance point z (a depth hit on the top surface, world frame) - NOT from
   detect_object's Position z, which is already lowered by 0.06. A plug on a stand/pedestal: only
   its top ~1 cm is the body, the rest is the stand - centre z = top z - 0.005, so e.g. top 0.058
   -> command z = 0.058 - 0.005 + 0.06 = 0.113. Print top z and the commanded z.
   Close across the narrow width (rotation 0 closes along world x).
   Small parts: take x,y from the affordance point, not the detection box centre (the box may merge the stand).
2. Locate the opening with the WRIST camera (the head camera cannot resolve a port; detect_object on
   it returns the whole face/edge, cm off). Holding the part, park the fingertips ~0.12-0.15 m in
   front of the face (opposite the axis), ~0.10 m above the opening's guessed height, offset ~0.1 m
   sideways from the guessed opening so the gripper does not hide it, then:
   ```python
   cands = locate_with_wrist_camera("USB-A port opening on the laptop side")  # best first
   ```
   Each entry: `{"label", "position": [x, y, z], "pixel"}` (~1 mm accurate). A device can show
   several similar openings (only one may fit): keep all candidates. None found or the gripper
   hides the face: shift the hand 0.1 m sideways / higher and call again.
   **Confirm before aiming**: move the hand ~5 cm sideways (still not hiding the face) and call
   again. Accept a position only if both views agree within ~3 mm (use their mean). If they
   disagree, one view picked a wrong spot (often the edge next to the hand that hides the real
   opening): take a third view from the other side and keep the two that agree.
3. Align: the part's tip must point along the insertion `axis` (world unit vector, into the hole).
   **Measure the in-hand offset** - do NOT reuse the pick rule (+0.06): closing and lifting
   pushes the part up into the fingers, so the held part sits near the commanded pose height.
   Lift, view the held tip with the wrist camera from 2 viewpoints (as in step 2):
   `offset = tip_position - get_robot_state()["eef_pos"]` (mean of the two views; print it).
   Commanded pose target = opening - offset - axis * 0.004 (tip **3-5 mm in front of the
   opening**, at the opening's height).    `tool_length` = offset projected on `axis`, or pass the whole measured vector:
   `insert_with_touch(axis=[1, 0, 0], tool_offset=offset)` (then `res["tip_pos"]` is the real tip).
   Approach the last ~3 cm slowly along the axis. Never push into the face yourself.
   The arm settles a few mm off target: read `get_robot_state()["eef_pos"]` and re-command
   `target + error` until < 1 mm.
   First touch contact far from the opening height (`res["tip_pos"]` z vs the opening z off by
   > 5 mm): the offset is wrong - re-measure it, do not just retry.
   Tip not visible in the wrist views (the fingers hide it): the vertical offset is unknown by a few
   mm - assume offset = axis * 0.025 and, on `search_exhausted`/`timeout`, retry at heights
   +5 mm, then -5 mm, then +10 mm (the 3 mm spiral covers each band) before re-locating the opening.
4. Call (gripper stays closed; do not open it before this returns):
   ```python
   res = insert_with_touch(axis=[1, 0, 0], tool_length=0.025)  # tool_length: fingertip centre -> tip, m
   ```
   It probes, spirals (3 mm radius) on the face to find the opening, and pushes the part in (~20 N max).
5. `res["outcome"]`: `seated` = in: open the gripper, back off along -axis. Anything else = NOT in:
   NEVER open the gripper. `search_exhausted` / `jammed` / `timeout`: back off 1 cm along -axis and try the NEXT
   candidate from step 2 (then re-locate). Do not repeat the same aim more than twice: the opening is
   more than the 3 mm search radius away, so re-locate with the wrist camera from a different viewpoint.
   `no_contact` / `slipped` (advanced past the part length =
   nothing was hit): the opening is not where you aimed - back off, re-locate with the wrist camera,
   check `gripper_holding()`. `unsupported`/`disabled`: report failure.
