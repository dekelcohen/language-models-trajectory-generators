"""Grasp-and-pull evidence from the arm's own sensors (no camera, no scene state).

A real arm logs, for every executed trajectory, where its controller was told to take the
hand, where the joint encoders say the hand went, and the gripper's finger opening. That is
enough to tell the three ways a pull on a handle can end:

  * the handle came along     - the fingers still hold it and the hand travelled freely;
  * the handle stayed put     - the fingers slid off it (holding -> fully closed) or the
                                hand stalled short of its target while holding (blocked);
  * there was nothing to pull - the fingers closed on air before the pull.

``summarize_pull_proprio`` turns the per-trajectory records produced by
``env._trajectory_proprio`` into those facts; ``format_pull_proprio`` is the reviewer's
prompt block; ``measured_no_motion_veto`` is the deterministic guard used to overrule a
VLM "success" when the grasped part provably never moved with the hand; and
``alternative_directions`` / ``retry_direction_hint`` make the next attempt try a
direction it has not tried yet (pull straight out -> slide sideways -> lift/lower ->
push), with no hinge / slide-axis prior.
"""

import re

import numpy as np

#: A closed-gripper motion shorter than this is a squeeze/adjustment, not a pull.
MIN_PULL_M = 0.04
#: The held part counts as "moved with the hand" past this net carried distance.
CARRIED_MIN_M = 0.03
#: Fingers slide this far along a bar before it pops out, while still reading "holding":
#: hand travel up to the moment the grip was lost is discounted by it.
SLIP_ALLOWANCE_M = 0.04
#: Holding, but the hand reached less than this fraction of its commanded travel = blocked.
BLOCKED_RATIO = 0.5
#: Two directions within ~45 deg are "the same direction" for the retry bookkeeping.
SAME_DIR_COS = 0.7
#: An open/empty-hand motion counts as a retreat (cannot have pushed the part) when it
#: backs away along the reversed approach by at least this cosine.
RETREAT_COS = 0.5
#: Hand motions shorter than this are ignored entirely.
MIN_MOTION_M = 0.02

ARTICULATED_TASK_RE = re.compile(
    r"\b(open|close|shut|pull|push|slide|door|drawer|cabinet|cupboard|handle|lid|hatch|hinge|"
    r"fridge|oven|microwave|dishwasher)\b", re.IGNORECASE)
#: Peg/plug insertion wording ("push the plug into the port") is not an articulated pull.
INSERT_TASK_RE = re.compile(r"\b(insert\w*|plug|socket|peg|connector|usb)\b", re.IGNORECASE)


def is_articulated_task(text):
    text = str(text or "")
    return bool(ARTICULATED_TASK_RE.search(text)) and not INSERT_TASK_RE.search(text)


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float).reshape(3)


def _unit(v):
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else np.asarray(v, dtype=float) / n


def _horizontal(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return _unit(v)


def _vec(v):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


def _motion(rec):
    """Commanded/achieved hand motion of one record -> dict (``None`` if unusable)."""
    start, end = _arr(rec.get("ee_start")), _arr(rec.get("ee_end"))
    if start is None or end is None:
        return None
    cmd_end = _arr(rec.get("cmd_end"))
    cmd = None if cmd_end is None else cmd_end - start
    act = end - start
    cmd_travel = float(np.linalg.norm(cmd)) if cmd is not None else float(np.linalg.norm(act))
    cmd_dir = _unit(cmd) if cmd is not None else _unit(act)
    along = float(act @ cmd_dir) if cmd_dir is not None else 0.0
    return {"desc": rec.get("desc") or "", "closed": bool(rec.get("gripper_closed")),
            "holding_start": rec.get("holding_start"), "holding_end": rec.get("holding_end"),
            "start": start, "end": end, "cmd_dir": cmd_dir, "cmd_travel": cmd_travel,
            "act_travel": float(np.linalg.norm(act)), "along": along,
            "last_holding": _arr(rec.get("ee_last_holding")),
            "traj_start": _arr(rec.get("traj_start"))}


def summarize_pull_proprio(records, logger=None):
    """Per-trajectory proprio records of ONE attempt -> facts dict, or ``None``.

    ``None`` when the gripper never closed during the attempt or the gripper reports no
    width (holding unknown): there was no grasp-and-pull to judge.
    """
    motions = [m for m in (_motion(r) for r in (records or [])) if m is not None]
    closed = [i for i, m in enumerate(motions) if m["closed"]]
    if not closed or all(motions[i]["holding_start"] is None for i in closed):
        return None
    first_closed = closed[0]

    # approach = the last real open-hand motion before the first close (onto the handle)
    approach = None
    for m in reversed(motions[:first_closed]):
        if not m["closed"] and m["act_travel"] >= MIN_MOTION_M:
            approach = _unit(m["end"] - m["start"])
            break

    pulls, carried_max, outcome_parts = [], 0.0, []
    grasped = False
    grasp_hand = grasp_cmd = last_held = None
    for i in closed:
        m = motions[i]
        if not m["holding_start"]:
            continue
        grasped = True
        if grasp_hand is None:
            grasp_hand, grasp_cmd = m["start"], m["traj_start"]
        last_held = m["end"] if m["holding_end"] else (
            m["last_holding"] if m["last_holding"] is not None else m["start"])
        if m["cmd_travel"] < MIN_PULL_M:
            continue
        if m["holding_end"]:
            blocked = m["along"] < BLOCKED_RATIO * m["cmd_travel"]
            # a stalled hand still creeps (finger pads slide on the bar): discount it
            carried = max(0.0, m["along"] - (SLIP_ALLOWANCE_M if blocked else 0.0))
            result = "blocked" if blocked else "carried"
        else:
            held_to = m["last_holding"]
            run = float(np.linalg.norm(held_to - m["start"])) if held_to is not None else 0.0
            carried = max(0.0, run - SLIP_ALLOWANCE_M)
            result = "slipped"
        carried_max = max(carried_max, carried)
        pulls.append({"desc": m["desc"], "dir": np.round(m["cmd_dir"], 3).tolist(),
                      "cmd_m": round(m["cmd_travel"], 3), "reached_m": round(max(0.0, m["along"]), 3),
                      "carried_m": round(carried, 3), "result": result})
        outcome_parts.append(result)

    # could an open/empty hand have pushed the part after the first pull? (anything that is
    # not a plain retreat back along the approach after the hand let go)
    first_pull = next((i for i in closed if motions[i]["holding_start"]
                       and motions[i]["cmd_travel"] >= MIN_PULL_M), None)
    possible_push = []
    if first_pull is not None:
        back = -approach if approach is not None else None
        for m in motions[first_pull + 1:]:
            held = m["closed"] and m["holding_start"]
            if held or m["act_travel"] < MIN_MOTION_M:
                continue
            d = _unit(m["end"] - m["start"])
            if back is None or float(d @ back) < RETREAT_COS:
                possible_push.append(m["desc"])

    if not grasped:
        outcome = "closed_on_nothing"
    elif not pulls:
        outcome = "no_pull"
    elif carried_max >= CARRIED_MIN_M:
        outcome = "part_moved_with_hand"
    else:
        outcome = "part_did_not_move"
    facts = {
        "outcome": outcome, "grasped": grasped, "pulls": pulls,
        "max_carried_m": round(carried_max, 3),
        "approach_dir": None if approach is None else np.round(approach, 3).tolist(),
        "possible_push_after_pull": possible_push,
        "pull_dirs": [p["dir"] for p in pulls],
        # directions in which the held part demonstrably travelled with the hand
        "moved_dirs": [p["dir"] for p in pulls if p["carried_m"] >= CARRIED_MIN_M],
        "handle_now": None,
    }
    if outcome == "part_moved_with_hand" and grasp_cmd is not None and last_held is not None:
        # grasp point in trajectory coordinates, shifted by how far the hand carried it
        facts["handle_now"] = np.round(grasp_cmd + (last_held - grasp_hand), 3).tolist()
    if logger is not None:
        logger(f"[pull_proprio] outcome={outcome} grasped={grasped} max_carried={carried_max:.3f} "
               f"pulls={[(p['desc'][:30], p['result'], p['cmd_m'], p['carried_m']) for p in pulls]} "
               f"approach={facts['approach_dir']} possible_push={possible_push} "
               f"handle_now={facts['handle_now']}")
    return facts


def measured_no_motion_veto(facts, task_text):
    """Reason string if a VLM "success" must be overruled, else ``None``.

    Only when it is physically unambiguous: an articulated open/close task, the fingers
    really held the part, at least one pull was commanded, the part never travelled with
    the hand (slipped out / blocked), and no later open/empty-hand motion could have pushed
    it (every later motion is a retreat back along the approach).
    """
    if not facts or not is_articulated_task(task_text):
        return None
    if facts["outcome"] != "part_did_not_move" or facts["possible_push_after_pull"]:
        return None
    if facts.get("approach_dir") is None:
        return None
    how = ", ".join(f"'{p['desc']}': {p['result']} after {p['cmd_m'] * 100:.0f} cm commanded, "
                    f"{p['carried_m'] * 100:.0f} cm carried" for p in facts["pulls"])
    return ("MEASURED (arm encoders + gripper width): the gripper held the handle, but the handle "
            f"never travelled with the hand (max {facts['max_carried_m'] * 100:.0f} cm < "
            f"{CARRIED_MIN_M * 100:.0f} cm) - {how}. Nothing pushed it afterwards (only a retreat), "
            "so the door/drawer/panel did not open.")


def alternative_directions(tried_dirs, approach_dir=None):
    """Untried horizontal/vertical opening directions, best-first -> list of (label, vec).

    Reference = the outward direction (reverse of the approach onto the handle, else the
    first tried pull). Order: straight out, the two sideways directions along the face
    (sliding panels), up/down (lift-up flaps, vertical sliders), then push in. Anything
    within ~45 deg of a direction already tried without moving the part is dropped.
    """
    tried = [d for d in (_unit(t) for t in (tried_dirs or [])) if d is not None]
    out = _horizontal(-np.asarray(approach_dir, dtype=float)) if approach_dir is not None else None
    if out is None and tried:
        out = _horizontal(tried[0])
    if out is None:
        return []
    side = np.array([-out[1], out[0], 0.0])
    cands = [("pull straight out (away from the face)", out),
             ("slide sideways along the face", side),
             ("slide sideways along the face (other side)", -side),
             ("lift up along the face", np.array([0.0, 0.0, 1.0])),
             ("push down along the face", np.array([0.0, 0.0, -1.0])),
             ("push in (into the face)", -out)]
    return [(label, np.round(v, 3).tolist()) for label, v in cands
            if not any(float(v @ t) >= SAME_DIR_COS for t in tried)]


def _moved_hint(facts, history_dirs):
    moved = facts.get("moved_dirs") or []
    if not moved:
        return None
    net = _unit(np.sum([_unit(d) for d in moved], axis=0))
    if net is None:
        return None
    avoid = [d for d in history_dirs if _unit(d) is not None and float(_unit(d) @ net) < SAME_DIR_COS]
    where = (f" The handle is now near {_vec(facts['handle_now'])} (trajectory coordinates: the "
             "grasp point shifted by the measured hand travel) - the handle coordinates detected "
             "before this attempt are STALE; grasp it there, not at the old position."
             if facts.get("handle_now") else "")
    return ("MEASURED (arm encoders + gripper width): the handle stayed in the gripper and travelled "
            f"~{facts['max_carried_m'] * 100:.0f} cm with the hand along {_vec(net)} (world) - the "
            "mechanism DOES move that way." + where + " If it is not open far enough, continue in "
            "that same direction; do not switch to a different opening motion"
            + (f" (these did NOT move it: {', '.join(_vec(d) for d in avoid)})" if avoid else "")
            + ". When moving away from the cabinet, first open the gripper and back straight away "
            "from the handle so the fingers do not drag it back.")


def retry_direction_hint(facts, history_dirs=()):
    """Retry-prompt line from the measured pull result, else ``None``.

    ``part_did_not_move``: the pull direction is wrong -> recommend untried directions.
    ``part_moved_with_hand`` (yet the attempt failed): the direction is right -> keep it and
    re-grasp the handle where the hand left it.
    ``history_dirs``: pull directions of EARLIER attempts that also did not move the part.
    """
    if not facts:
        return None
    if facts["outcome"] == "part_moved_with_hand":
        return _moved_hint(facts, list(history_dirs))
    if facts["outcome"] != "part_did_not_move":
        return None
    tried = list(history_dirs) + list(facts["pull_dirs"])
    alts = alternative_directions(tried, facts.get("approach_dir"))
    tried_txt = ", ".join(_vec(d) for d in tried)
    if not alts:
        return (f"MEASURED: the held handle did not move under pulls in every basic direction tried "
                f"({tried_txt}). Re-check the grasp point / the part identity instead of the direction.")
    first = alts[0]
    rest = "; ".join(f"{label} {_vec(v)}" for label, v in alts[1:3])
    return ("MEASURED (arm encoders + gripper width): the gripper held the handle but the part did NOT "
            f"move when pulled along {tried_txt} (world) - the mechanism does not open that way. Do "
            f"NOT repeat that pull. Next attempt: keep a firm grasp and move the handle "
            f"{first[0]}: world direction {_vec(first[1])}"
            + (f" (if that also does not move it: {rest})" if rest else "")
            + ". Decide the sideways sense from the scene: a sliding panel moves toward its free/"
            "open side of the track, away from the edge where its handle sits.")


def format_pull_proprio(facts):
    """Reviewer prompt block (measured facts only)."""
    words = {
        "closed_on_nothing": "the gripper closed but its fingers closed fully every time: nothing "
                             "was HELD between them. This does NOT show the part stayed put - closed "
                             "fingers moving against a handle can still push it along; judge from "
                             "the clips whether the part moved.",
        "no_pull": "the gripper held something but the hand was not moved while holding it.",
        "part_moved_with_hand": f"the held part travelled with the hand (up to "
                                f"{facts['max_carried_m'] * 100:.0f} cm) - it DID move; do not claim "
                                "it stayed shut. Judge from the clips whether it moved far enough "
                                "and the opening/closing way (a panel that slid sideways open IS "
                                "open, even if the task text expected a hinged swing).",
        "part_did_not_move": f"the held part did NOT travel with the hand (max "
                             f"{facts['max_carried_m'] * 100:.0f} cm): it slipped out of the fingers "
                             "or blocked the arm. The door/drawer/panel did not start to open.",
    }
    lines = ["PULL PROPRIOCEPTION (measured by the arm's joint encoders and the gripper width sensor "
             "for every pull while the fingers were closed; a direct physical measurement):"]
    for p in facts["pulls"]:
        res = {"carried": "hand travelled freely while still holding",
               "blocked": "hand STALLED short of its target while holding (the part resisted)",
               "slipped": "the fingers LOST the part during the pull (closed fully)"}[p["result"]]
        lines.append(f"  - '{p['desc']}': commanded {p['cmd_m'] * 100:.0f} cm along {_vec(p['dir'])}, "
                     f"reached {p['reached_m'] * 100:.0f} cm; {res}; held part carried "
                     f"~{p['carried_m'] * 100:.0f} cm.")
    lines.append("  - Verdict: " + words[facts["outcome"]])
    if facts["outcome"] == "part_did_not_move":
        lines.append("  - If the part did not move, the task is NOT achieved, whatever the arm's "
                     "motion looks like in the clips.")
    return "\n".join(lines)
