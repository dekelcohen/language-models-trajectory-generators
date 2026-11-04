---
name: touch-guided-insertion
scope: subtask
description: >
  Pick up a small peg/plug (USB stick, connector, pin) and insert it into a tight hole or port
  using force feedback. Load it for the PICK-UP subtask too (grasp depth rule). Vision alone is
  ~1-2 mm off; this finds the opening by touch. Gives locate_with_wrist_camera(), move_until_contact(), get_grasp_info(), insert_with_touch().
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
   **Measure the top by touch first** (depth on a small part is a few mm off; a grasp below the body's
   mid-height lets it pivot or slide in the fingers later): `close_gripper()`, move the fingertips
   3 cm above the guessed top at the part's x,y, then
   `r = move_until_contact([0, 0, -1], max_travel=0.05, force=3.0)`;
   `top_z = r["finger_bottom_z"]`. Rise 3 cm, `open_gripper()`, grasp at centre z = top_z - body
   height / 2 (command z = centre z + 0.06). body height = the HOUSING only (USB-A plug: 8 mm, so
   centre = top_z - 0.004), never the detection box / depth height (it merges a stand or cable);
   no stand/floor heuristics. The arm settles a few mm LOW under load: before
   `close_gripper()` read `get_robot_state()["eef_pos"]` and re-command `target + error` until
   within 1 mm (a grasp 4 mm below the centre lets the part pivot during insertion).
   Settle correction (here and below): new command = the PREVIOUS COMMANDED pose + error (NOT the
   measured `pose` + error - that equals the old target and sags the same again); at most 3
   re-commands, only for errors < 1 cm; if the error
   does not shrink the arm is blocked (contact) - stop correcting, back off, never chase it.
   Free-space transit / camera-view poses need NO correction.
   `close_gripper()` records the grasp; a later subtask
   reads it with `get_grasp_info()`. Right after closing, check
   `get_grasp_info()["grasp_depth_below_top"]` ≈ body height / 2 (USB: 0.004); off by > 3 mm =
   bad grasp: open, rise, redo the grasp with the command shifted by the depth error (repeating the
   same command gives the same depth); do not lift a part held below its housing.
2. Locate the opening with the WRIST camera (the head camera cannot resolve a port; detect_object on
   it returns the whole face/edge, cm off). Which face holds the port is a guess too: list the device's
   side faces the part could enter, the one nearest the arm's base first (best reach and view), and try them in that order (step 5).
   Holding the part, park the fingertips ~0.12-0.15 m in
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
   disagree (common: device textures show several slot-like marks and a small opening is a few
   pixels), do NOT keep re-viewing: the HEIGHT (z) is usually consistent across views, the position
   ALONG the face is not. Opening z = the MEDIAN z of ALL candidates from all views on that face
   (drop any > 5 mm from the median - a single odd candidate is a mis-hit); opening y = the mean of
   the candidates' y. Print both. The touch line search in step 4 finds it along the face
   (`search_line`, covers +-2.5 cm) but needs z right within ~1 mm (2.5 mm off fails even with
   coarse extra rows - tested), so z is the number to get right.
   **Reject edge hits**: also locate the face's top edge (`locate_with_wrist_camera("top edge of
   the laptop side face")`, same views). Candidates within 3 mm of that edge's z are the edge, not
   the opening (an opening sits below the top surface by at least its half-height + a wall). Fewer
   than 4 candidates left: take 2 more views with the camera lower (~0.05 m above the opening
   height) so the face is seen head-on, and redo the median.
3. **Calibrate the in-hand offset by touch** (the wrist camera cannot see the held tip, and closing /
   lifting shifts the part in the fingers by up to ~1 cm). First `gripper_holding()` must be True -
   a touch with an empty hand measures the fingers, not the part; False = it fell: find it and re-pick. Use `move_until_contact` on the device
   itself, next to the opening; the camera that measured the opening measures the surfaces too,
   so its calibration error cancels. Targets below are fingertip-centre positions (= `eef_pos`):
   command pose z = target z + 0.06 (same rule as the pick); re-command until `eef_pos` is within 1 mm.
   ```python
   face_x = port[0]                         # face plane along the axis (the opening lies in it)
   side_y = port[1] + 0.03                  # 3 cm sideways: flat face, no opening
   # a) face touch -> tip position along the axis, ~7 mm below the opening centre (solid face; no
   #    separate deck-height estimate needed - the opening z from step 2 is the reference)
   # execute_trajectory to fingertips [face_x - 0.040, side_y, port[2] - 0.007] (4 cm in front)
   r = move_until_contact([1, 0, 0], max_travel=0.04)          # 'contact' expected
   off_x = face_x - r["eef_pos"][0]                            # eef -> tip end, along the axis
   # back off 1 cm along -axis
   # b) height: from the touched grasp (step 1; recorded across subtasks) - tip centred on the body height
   g = get_grasp_info()      # {"grasp_eef_pos", "touched_top_z"}
   off_z = g["touched_top_z"] - body_h / 2 - g["grasp_eef_pos"][2]   # touched_top_z None: redo step 1's touch
   offset = [off_x, 0.0, off_z]                                # eef -> tip centre (y: fingers centre it)
   ```
   (Rotate the example for other axes: touch the face along `axis`.)
   'no_contact' on a): the face is farther - start 2 cm closer and retry.
   **Off-centre grasp -> regrasp.** Expected off_x = middle of the part -> tip end (about half its
   length). More than 5 mm beyond that = held near the back end: the long lever lets the part pitch in
   the fingers and wedge a few mm into the opening. Put it back (fingertips over the spot where it
   was, 1.5 cm high; `move_until_contact([0, 0, -1], max_travel=0.03, force=3.0)` until it rests on the
   table; `open_gripper()`; rise 4 cm), shift the grasp along the part axis by (off_x - expected) toward
   the tip, descend to the same `eef_pos` z, `close_gripper()`, lift, and redo touch a).
   Do NOT press the held tip down onto a surface or edge-scan it against the deck to measure its height:
   at ~2 N it pivots / slides in the fingers. off_z comes from `get_grasp_info()` only; if that has
   no `touched_top_z`, put the part down and redo step 1 (touch + grasp).
   Then align: commanded pose target = opening - offset + axis * (-0.004) (tip **3-5 mm in front of
   the opening**, at the opening's height); pass `tool_offset=offset` to `insert_with_touch`
   (then `res["tip_pos"]` is the real tip). Approach the last ~3 cm slowly along the axis. Never
   push into the face yourself. The arm settles a few mm off target: read
   `get_robot_state()["eef_pos"]` and re-command `target + error` until < 1 mm.
   First touch contact far from the opening height (`res["tip_pos"]` z vs the opening z off by
   > 5 mm): the offset is wrong - redo the calibration, do not just retry.
4. Call (gripper stays closed; do not open it before this returns):
   ```python
   res = insert_with_touch(axis=[1, 0, 0], tool_length=0.025)  # tool_length: fingertip centre -> tip, m
   ```
   It probes, spirals (3 mm radius; search_radius=0.005 widens it, slower) on the face to find the opening, and pushes the part in (~20 N max).
   Opening position along the face uncertain by more than ~3 mm (views disagree, or a spiral attempt
   returned `search_exhausted`): search by feel along the face instead (seated in sim with 2 cm aim error, ~20-60 s):
   ```python
   res = insert_with_touch(axis=[1, 0, 0], tool_offset=offset, search_line=[0, 1, 0], search_half=0.025)
   ```
   `search_line` = the face direction across the axis along which the aim is uncertain (horizontal
   along a laptop side); the height must be right within ~1 mm (the line passes are 0.4 mm apart).
   A line search that exhausted at a height you doubt by 1-2 mm: retry once with `search_cross=0.002`
   (passes cover +-2 mm in height, ~2x slower). Tested: it finds the opening but entries off-height
   can jam after 2-4 mm, so a correct z (step 2) matters more.
   Do not hand-write your own scan: this keeps the force control and drop detection.
5. `res["outcome"]`: `seated` = in: open the gripper, back off along -axis. Anything else = NOT in:
   NEVER open the gripper. `search_exhausted` / `jammed` / `timeout` after a spiral: back off 1 cm
   along -axis and retry with `search_line` (above) from the mean aim. A line search that ends
   `search_exhausted` after a full sweep at a verified height (off_z from step 3, opening z consistent
   across views) means **there is NO opening on that face**: device textures often paint port-like
   marks on several faces (e.g. both sides of a laptop), and only the real one accepts the part. Do not
   retry the same face: move to the next candidate FACE (opposite side, front, back), wrist-locate the
   port there, re-touch a) on that face, and line-search it. Only re-check the height (step 3) if the
   touch contacts were far from the opening z.
   `no_contact` / `slipped` (advanced past the part length =
   nothing was hit): the opening is not where you aimed - back off, re-locate with the wrist camera,
   check `gripper_holding()`. `unsupported`/`disabled`: report failure.
