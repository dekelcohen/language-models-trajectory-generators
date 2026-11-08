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
   **The held part turns with the hand.** Keep the pick rotation (`get_robot_state()["rotation"]`)
   in every later pose, start poses included; turn the hand only by the angle that brings the
   part's tip axis onto the insertion axis. A pick at 1.571 then poses at 0 turned the plug 90 deg,
   so its side, not its tip, met the face (every early LLM run).
   Small parts: take x,y from the affordance point, not the detection box centre (the box may merge the stand).
   **Measure the top by touch first** (depth on a small part is a few mm off; a grasp below the body's
   mid-height lets it pivot or slide in the fingers later): `close_gripper()`, move the fingertips
   3 cm above the guessed top at the part's x,y, then
   `r = move_until_contact([0, 0, -1], max_travel=0.05, force=3.0)`;
   `top_z = r["finger_bottom_z"]`. Rise 3 cm, `open_gripper()`, grasp at centre z = top_z - body
   height / 2 (command z = centre z + 0.06). body height = the HOUSING only (USB-A plug: 8 mm, so
   centre = top_z - 0.004), never the detection box / depth height (it merges a stand or cable);
   no stand/floor heuristics. **Sanity-check the touch**: a contact > 5 mm ABOVE the located
   point's z (or > body height + 5 mm above the surface it lies on) hit something else - the
   edge of the support it fell off, a neighbour - not the part: do not grasp there (it closes on
   air). Re-locate from straight above (wrist camera looking down, 15-20 cm over the point) and
   touch again; a dropped part usually lies on the floor next to its support. The arm settles a few mm LOW under load: before
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
   Finish a face (touch insert at its candidates) before viewing the next one. Skip faces > 0.8 m
   from the arm base (reach limit: pushes there stall or knock the part out of the grasp).
   Holding the part, park the fingertips ~0.12-0.15 m in
   front of the face (opposite the axis), ~0.10 m above the opening's guessed height, offset ~0.1 m
   sideways from the guessed opening so the gripper does not hide it, then:
   ```python
   cands = locate_with_wrist_camera("USB-A port opening on the laptop side")  # best first
   ```
   Each entry: `{"label", "position": [x, y, z], "pixel"}` (~1 mm accurate). A device can show
   several similar openings (only one may fit): keep all candidates. None found or the gripper
   hides the face: shift the hand 0.1 m sideways / higher and call again. The wrist camera turns
   with the hand: if the palm / held part fills the view at the carry rotation, take the views at
   carry rotation +-1.571 (the part turns too - harmless in free space), then turn back to the carry
   rotation before step 3 (check `get_robot_state()["rotation"]`).
   **Collect views, then aggregate with `estimate_opening`** (do not hand-roll the median): take
   3-4 views ~5 cm apart sideways (still not hiding the face), each calling the port locate AND the
   face's top edge (`locate_with_wrist_camera("top edge of the laptop side face")`), and pass ALL
   candidates of all views - outliers (other faces, odd heights, rim hits) are rejected inside:
   ```python
   port_c, edge_c = [], []
   # for each view pose: port_c += locate_with_wrist_camera("USB-A port opening on the laptop side")
   #                     edge_c += locate_with_wrist_camera("top edge of the laptop side face")
   est = estimate_opening(port_c, edge_c, axis=[1, 0, 0])   # prints its result
   port = est["position"]          # aim point [x, y, z]; None -> no usable hit: new views
   ```
   It returns `position`, `search_line`, `search_half`, `search_cross` (use them in step 4),
   `n_used`, `z_spread`, `confident`, `notes`. Why: pointing at a small opening is consistent in
   HEIGHT across views but scatters a few cm ALONG the face; the touch line search in step 4 finds
   the along-face position by feel, but needs z right within ~1-2 mm, so z is the number to get
   right. `confident` False (fewer than 3 consistent hits): take 2 more views with the camera lower
   (~0.05 m above the opening height) so the face is seen head-on, and call it again on ALL
   candidates. Never fall back to a single early candidate.
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
   returned `search_exhausted`): search by feel along the face instead, with the numbers from
   `estimate_opening` (seated in sim with 2 cm aim error, ~20-60 s):
   ```python
   res = insert_with_touch(axis=[1, 0, 0], tool_offset=offset, search_line=est["search_line"],
                           search_half=est["search_half"], search_cross=est["search_cross"])
   ```
   `search_line` = the face direction across the axis along which the aim is uncertain (horizontal
   along a laptop side); the height must be right within ~1 mm (the line passes are 0.4 mm apart).
   A line search that exhausted at a height you doubt by 1-2 mm: retry once with `search_cross=0.002`
   (passes cover +-2 mm in height, ~2x slower). The search also finds height by feel: a push that
   sinks 0.4-2 mm but is then blocked is a "partial catch" on an edge of the opening; two catches
   ~one opening width apart give its centre, then a short column raster finds the height (logged as
   "edges along the line ... column raster"). Tested: seated with a 2 mm aim error in both axes.
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
   `no_contact` / `slipped` (advanced past the part length, or the sideways seat check moved = the
   part slid / pivoted in the grasp instead of entering): the part is not in. Back off, check
   `gripper_holding()`, re-measure the in-hand offset (step 3) before retrying, and re-locate with
   the wrist camera. `unsupported`/`disabled`: report failure.
