"""Force-feedback (touch) results of an attempt -> reviewer prompt text.

Built only from what the arm itself measured during ``insert_with_touch`` / ``pull_with_touch``
(joint torques -> estimated external force, encoder travel, gripper width). No sim state.
"""

INSERT_WORDS = {
    "seated": "the part went IN: it advanced past the contact point and then stopped against the "
              "bottom of the hole (seated)",
    "jammed": "the part touched the face but stalled repeatedly before full depth (jammed) - NOT inserted",
    "search_exhausted": "the part touched the face but the spiral search never found the opening - "
                        "NOT inserted",
    "no_contact": "the hand travelled the whole approach without feeling any contact - the part was "
                  "not in front of the hole (or not held)",
    "slipped": "the hand advanced farther than the part is long - the part slid in the fingers; "
               "insertion unconfirmed",
    "over_force": "the contact force exceeded the safety limit; aborted - NOT inserted",
    "timeout": "ran out of time before a result - insertion unconfirmed",
}

PULL_WORDS = {
    "opened": "the mechanism moved with the hand to the requested opening (OPENED)",
    "end_stop": "the mechanism moved with the hand until it hit its end stop (fully open)",
    "blocked": "the hand pulled but the mechanism did not yield (BLOCKED)",
    "slipped": "the handle slipped out of the fingers during the pull",
    "timeout": "ran out of time while still pulling",
    "regrasp_failed": "re-grasping the moved handle failed",
}


def _event_text(ev):
    if isinstance(ev, (list, tuple)) and ev:
        return str(ev[-1])
    return str(ev)


def format_touch_evidence(records):
    """``records``: [{"kind": "insert"|"pull", "result": dict}, ...] of THIS attempt -> text or None."""
    lines = []
    for r in records or []:
        res = r.get("result") or {}
        out = res.get("outcome")
        if out in (None, "disabled", "unsupported", "error"):
            continue
        if r.get("kind") == "contact":
            # calibration / contact checks: not evidence of task success by themselves
            if out == "contact":
                lines.append(f"  - move_until_contact: touched a surface after {res.get('travel_mm')} mm "
                             f"({res.get('contact_N')} N) - a probe, not a task step.")
            continue
        if r.get("kind") == "insert":
            evs = [_event_text(e) for e in res.get("events") or []]
            detail = next((e for e in reversed(evs) if "advance" in e or "contact" in e), None)
            lines.append(f"  - insert_with_touch: outcome '{out}': {INSERT_WORDS.get(out, out)}"
                         + (f" (last event: {detail})" if detail else "") + ".")
        else:
            bits = []
            if res.get("travel_cm") is not None:
                bits.append(f"hand travelled {res['travel_cm']} cm while holding")
            if res.get("swept_deg"):
                bits.append(f"swept {res['swept_deg']} deg about a fitted hinge")
            if res.get("pull_force_N") is not None:
                bits.append(f"pull force {res['pull_force_N']} N")
            lines.append(f"  - pull_with_touch: outcome '{out}': {PULL_WORDS.get(out, out)}"
                         + (f" ({', '.join(bits)})" if bits else "") + ".")
    if not lines:
        return None
    return ("FORCE FEEDBACK (measured by the arm's joint torque sensing and encoders during the "
            "touch-guided motions of THIS attempt; a direct physical measurement, more reliable "
            "than millimetre details in the low-resolution clips):\n" + "\n".join(lines)
            + "\n  Use this for the parts the camera cannot resolve (a plug's last millimetres, "
              "whether a handle really moved), and the clips for everything else.")
