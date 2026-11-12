---
name: touch-guided-insertion
scope: subtask
description: >
  Pick up a small plug / peg / connector (USB-A, USB-C, HDMI, charger cable, peg, mains plug) and
  insert it into its tight hole, port or socket using force feedback, on a device standing at any
  heading. Load it for the PICK-UP subtask too (grasp depth rule). Vision alone is ~1-2 mm off; this
  finds the opening by touch. Gives locate_with_wrist_camera(), estimate_face_normal(),
  face_normal_from_touches(), estimate_opening(), move_until_contact(), get_grasp_info(), insert_with_touch().
license: MIT
metadata:
  category: manipulation
  tags: [insert, plug, connector, usb, hdmi, socket, peg-in-hole, touch, force]
  requires: [touch]
---

# Touch-guided insertion

1. Grasp the part across its body (not the tip), fingertip centre near the middle of the BODY/housing
   (not of the whole part: the finger pads reach ~1 cm toward the tip and must stay clear of the face
   when the tip is fully in - step 3 "Off-centre grasp"); lift; check
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
   (guessed top = the HIGHEST affordance / depth point z, else the detection's Position z + 0.06 = its
   box top. The box may include a stand / pedestal / support under the part, so the part can sit
   cm above the floor even when a description says "on the floor": never start the touch below that
   top, and never replace a touched top with a floor-based guess.)
   `r = move_until_contact([0, 0, -1], max_travel=0.05, force=3.0)`;
   `top_z = r["finger_bottom_z"]`. Rise 3 cm, `open_gripper()`, grasp at centre z = top_z - body
   height / 2 (command z = centre z + 0.06). body height = the HOUSING only, from the part type's
   standard size (a category prior: USB-A plug housing ~8 mm, USB-C ~6.5 mm, HDMI ~10 mm, unknown
   connector ~8 mm; check it looks right in the wrist view), so e.g. USB-A centre = top_z - 0.004.
   Lengths (same kind of prior, used for the step 3 grasp gate): insertable tip `tip_len` USB-A ~12 mm,
   USB-C ~7 mm, HDMI ~11 mm; housing `housing_len` ~20-25 mm (check it against the image).
   Never the detection box / depth height (it merges a stand or cable), and never derive it from the
   grasp itself (body_h = 2 * grasp_depth_below_top is circular: it always gives off_z = 0 later);
   no stand/floor heuristics. The arm settles a few mm LOW under load: before
   `close_gripper()` read `get_robot_state()["eef_pos"]` and re-command `target + error` until
   within 1 mm (a grasp 4 mm below the centre lets the part pivot during insertion).
   `close_gripper()` records the grasp; a later subtask
   reads it with `get_grasp_info()`. Right after closing, check
   `get_grasp_info()["grasp_depth_below_top"]` ≈ body height / 2; off by > 3 mm =
   bad grasp: open, rise, redo the grasp with the command shifted by the depth error (repeating the
   same command gives the same depth); do not lift a part held below its housing.
2. Locate the opening with the WRIST camera (the head camera cannot resolve a port; detect_object on
   it returns the whole face/edge, cm off).
   Holding the part, park the fingertips ~0.12-0.15 m in
   front of the face (opposite the axis), ~0.10 m above the opening's guessed height, offset ~0.1 m
   sideways from the guessed opening so the gripper does not hide it, then:
   ```python
   cands = locate_with_wrist_camera("HDMI port opening on the laptop side")  # name YOUR part's port type
   ```
   Each entry: `{"label", "position": [x, y, z], "pixel"}` (~1 mm accurate). Points a few cm apart along the face at the same height are usually ONE opening scattered by
   pointing, not two: treat them as separate openings only when one wrist image clearly shows two
   distinct holes; otherwise pass them all to `estimate_opening` and line-search (step 4). None found or the gripper
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
   w_open = ...   # the opening's width along the face (m), the prior for its type (step 4)
   # for each view pose: port_c += locate_with_wrist_camera("<part type> port opening on the <device> side")
   #                     edge_c += locate_with_wrist_camera("top edge of the <device> side face")
   fn = estimate_face_normal(port_c[0]["position"])  # head depth: {"normal", "axis", "along", ...}
   axis = fn["axis"]               # insertion direction = into the face; NEVER assume a world axis
   est = estimate_opening(port_c, edge_c, axis=axis, opening_width=w_open)   # prints its result
   port = est["position"]          # aim point [x, y, z]; None -> no usable hit: new views
   ```
   `estimate_face_normal` None (face not seen by the head camera): take the axis from the two
   face touches of step 3 (`face_normal_from_touches`), starting from your best view-based guess.
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
   a, t = np.array(axis), np.array(fn["along"])   # into the face; along the face (horizontal)
   # a) face touches -> tip position along the axis, ~7 mm below the opening centre (solid face; no
   #    separate deck-height estimate needed - the opening z from step 2 is the reference),
   #    3 cm to each side of the opening (flat face). Fingertips 4 cm in front of each point:
   hits = []
   for s in (+0.03, -0.03):
       side = np.array(port) + s * t + [0, 0, -0.007]
       # execute_trajectory to fingertips side - 0.040 * a (carry rotation), then:
       r = move_until_contact(axis, max_travel=0.04)            # 'contact' expected
       hits.append(np.array(r["eef_pos"]))
       # back off 1 cm along -axis
   # the face line from the two touches refines the depth-fitted normal (both see the same face)
   n_t = face_normal_from_touches(hits[0], hits[1], outward_hint=-a)
   if np.dot(n_t, -a) > math.cos(math.radians(10)):      # agree within 10 deg: trust the touches
       a = -np.array(n_t); axis = a.tolist()
   off_a = float(np.dot(np.array(port) - (hits[0] + hits[1]) / 2, a))   # eef -> tip end, along the axis
   # b) height: from the touched grasp (step 1; recorded across subtasks) - tip centred on the body height
   g = get_grasp_info()      # {"grasp_eef_pos", "touched_top_z"}
   # body_h = the housing height prior of step 1 (NOT 2 * grasp_depth_below_top - that cancels to 0)
   off_z = g["touched_top_z"] - body_h / 2 - g["grasp_eef_pos"][2]   # touched_top_z None: redo step 1's touch
   offset = (off_a * a + np.array([0.0, 0.0, off_z])).tolist()   # eef -> tip centre (fingers centre it across)
   # c) grasp-position gate (tip_len / housing_len = the part type's standard sizes, step 1 priors)
   expected = tip_len + housing_len / 2
   lo = min(tip_len + 0.012, expected - 0.003)   # short housings: never reject a grasp at `expected`
   if not (lo <= off_a <= expected + 0.005):
       regrasp(shift = off_a - expected)   # see "Off-centre grasp" below; do NOT insert with this grasp
       # This regrasp is REQUIRED even if the task text says "do not re-grasp / do not open the gripper"
       # (that only means: do not drop the part). Do not insert anyway, flip, or re-touch instead -
       # inserting with off_a below the gate stalls at the finger pads and pivots/flings the part.
   ```
   Touches > 10 deg off the depth normal: a touch hit something else (the opening, an edge) -
   repeat it 1 cm further along the face.
   'no_contact' on a): the face is farther - start 2 cm closer and retry.
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
   res = insert_with_touch(axis=axis, tool_offset=offset, insert_force=f_ins, opening_width=w_open)
   ```
   It probes, spirals (3 mm radius; search_radius=0.005 widens it, slower) on the face to find the
   opening, and pushes the part in. Part-specific numbers - set them, then ADAPT them per retry (step 5):
   `insert_force` (N; start 18 for a small connector, ~10 for a loose peg, ~30 for a mains plug,
   always below what the grip holds: `grip_force` 20 N slips at ~24 N) and `opening_width` (m: the
   opening's width along the face, from the wrist views; USB-A ~0.012, HDMI ~0.014, USB-C ~0.009).
   Opening position along the face uncertain by more than ~3 mm (views disagree, or a spiral attempt
   returned `search_exhausted`): search by feel along the face instead, with the numbers from
   `estimate_opening` (seated in sim with 2 cm aim error, ~20-60 s):
   ```python
   res = insert_with_touch(axis=axis, tool_offset=offset, search_line=est["search_line"],
                           search_half=est["search_half"], search_cross=est["search_cross"],
                           insert_force=f_ins, opening_width=w_open)
   ```
   `search_line` = the face direction across the axis along which the aim is uncertain (horizontal
   along the face, = `fn["along"]`); the height must be right within ~1 mm (the line passes are 0.4 mm apart).
   A line search that exhausted at a height you doubt by 1-2 mm: retry once with `search_cross=0.002`
   (passes cover +-2 mm in height, ~2x slower). The search also finds height by feel: a push that
   sinks 0.4-2 mm but is then blocked is a "partial catch" on an edge of the opening; two catches
   ~one opening width apart give its centre, then a short column raster finds the height (logged as
   "edges along the line ... column raster"). Tested: seated with a 2 mm aim error in both axes.
   Do not hand-write your own scan: this keeps the force control and drop detection.
5. `res["outcome"]`: `seated` = in: open the gripper, back off along -axis. Anything else = NOT in:
   NEVER open the gripper. `search_exhausted` / `jammed` / `timeout` after a spiral: back off 1 cm
   along -axis and retry with `search_line` (above) from the mean aim.    `no_contact` / `slipped` (advanced past the part length, or the sideways seat check moved = the
   part slid / pivoted in the grasp instead of entering): the part is not in. Back off, check
   `gripper_holding()`, re-measure the in-hand offset (step 3) before retrying, and re-locate with
   the wrist camera. `unsupported`/`disabled`: report failure.
