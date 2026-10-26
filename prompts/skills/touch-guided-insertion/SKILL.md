---
name: touch-guided-insertion
scope: subtask
description: >
  Insert a HELD peg/plug (USB stick, connector, pin) into a tight hole or port using force
  feedback. Vision alone is ~1-2 mm off; this finds the opening by touch. Gives insert_with_touch().
license: MIT
metadata:
  category: manipulation
  tags: [insert, usb, plug, peg-in-hole, touch, force]
  requires: [touch]
---

# Touch-guided insertion

1. Grasp the part across its body (not the tip), fingertip centre near its middle; lift; check `gripper_holding`.
2. Align by vision: the part's tip must point along the insertion `axis` (world unit vector, into
   the hole). Stop with the tip **3-5 mm in front of the opening**, centred on it; approach the
   last ~3 cm slowly along the axis. Never push into the face yourself.
   The arm settles a few mm off target: read `eef_pos` and re-command `target + error` until < 1 mm.
   Measure where the part sits in the hand with the wrist camera after lifting (closing can shift it).
3. Call (gripper stays closed; do not open it before this returns):
   ```python
   res = insert_with_touch(axis=[1, 0, 0], tool_length=0.025)  # tool_length: fingertip centre -> tip, m
   ```
   It probes, spirals on the face to find the opening, and pushes the part in (~20 N max).
4. `res["outcome"]`: `seated` = in (open the gripper, back off along -axis). `search_exhausted` /
   `jammed`: back off 1 cm, re-check alignment with the wrist camera, retry once from step 2.
   `no_contact`: the tip was > 2 cm from the face; re-approach closer. `unsupported`/`disabled`: report failure.
