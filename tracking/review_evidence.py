"""Handle-pull evidence for the VLM reviewer: tracked points -> motion -> verdict facts.

Pure numpy, no simulator. Input is what the tracker already produces per reference frame:
the fused ``(K, 3)`` world points of the tracked handle/door part (``nan`` rows = not seen)
and the gripper world position. Output is a dict of *measured* facts plus a short text block
that is pasted into the review prompt, so the reviewer no longer has to guess from pixels
whether a door moved 8 deg or 0 deg, or whether the arm dragged the handle or lost it.

Pipeline (one call to :func:`summarize_pull`):

  1. Pick the widest-apart pair of frames that still share >= 3 seen points (the gripper
     occludes the handle at grasp time, so the window's literal ends are often empty).
  2. ``estimate_motion`` on that pair -> kind / hinge axis / angle; if the screw fit is not
     confident, a plain rigid fit of the same points gives the angle.
  3. gripper<->handle distance trace -> "handle left the gripper" time (slip / lost).
  4. gripper travel direction vs the handle's own motion (door moved) or vs the door
     front's outward normal (door did not move) -> "pulling the wrong way".
  5. a recommended pull direction for the next attempt (arc tangent at the handle's latest
     position / handle motion / face normal) - the retry prompt's actionable fix.

No hinge prior is used anywhere: neither the sim nor the real robot knows where the hinge
is. The only geometric assumption is that furniture fronts are vertical (horizontal pull
normal), switchable with ``horizontal_normal=False``.
"""

import re

import numpy as np

from tracking import motion as mo

#: Opening-angle bands for the one-word label shown to the reviewer.
ANGLE_CLOSED_DEG = 3.0
ANGLE_SLIGHT_DEG = 15.0
ANGLE_PARTIAL_DEG = 35.0
#: Handle farther than this (beyond its engaged offset) from the gripper = it left the hand.
DETACH_EXTRA_M = 0.05
#: ... in this many consecutive *measured* samples (unmeasured frames are skipped, not "far":
#: buffering trackers such as CoTracker3 only deliver a trusted pose every few frames).
DETACH_SAMPLES = 2
#: Gripper must have travelled this far before "handle did not follow" is claimed.
MIN_GRIPPER_TRAVEL_M = 0.04
#: Handle moved less than this fraction of the gripper's travel = it did not follow.
FOLLOW_RATIO_MIN = 0.4
#: cos(gripper travel, opening direction) below this (~60 deg off) = pulling the wrong way.
WRONG_DIR_COS = 0.5
#: Handle travel below this = the door/drawer did not move (no direction evidence from it).
HANDLE_MOVED_M = 0.02
#: Arc tangent must be within (this + half the opening angle) of the handle's chord, and
#: the hinge axis must be (nearly) perpendicular to the chord, or the screw fit is ignored.
ARC_CHORD_TOL_DEG = 25.0
AXIS_CHORD_MAX_COS = 0.4
#: Handle unseen for the last this-many seconds = its track is stale for the recommendation.
STALE_TRACK_S = 1.0
#: Rigid-fit rmse above this = the tracked points are not one rigid body; no angle from it.
RIGID_FIT_MAX_RMSE_M = 0.01


def angle_label(angle_deg):
    if angle_deg < ANGLE_CLOSED_DEG:
        return "closed"
    if angle_deg < ANGLE_SLIGHT_DEG:
        return "slightly open"
    if angle_deg < ANGLE_PARTIAL_DEG:
        return "partly open"
    return "wide open"


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else np.asarray(v, dtype=float) / n


def _first_run(flags, length):
    """Index where ``flags`` first stays True for ``length`` consecutive entries, else None."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= length:
            return i - length + 1
    return None


def _track_centroid(frames):
    """Per-frame handle centre that does not jump when the visible subset of points changes.

    A plain ``nanmean`` moves by the points' spread (~4 cm) whenever a point drops in or out,
    which looks exactly like a slip. Instead each point keeps its offset from the centre at
    the frame it was first seen, and the centre is the mean of (point - offset).
    """
    offsets, out = {}, []
    for pts in frames:
        pts = np.asarray(pts, dtype=float)
        vis = np.flatnonzero(np.isfinite(pts).all(axis=1))
        known = [j for j in vis if j in offsets]
        if known:
            c = np.mean([pts[j] - offsets[j] for j in known], axis=0)
        elif len(vis):
            c = pts[vis].mean(axis=0)
        else:
            out.append(np.full(3, np.nan))
            continue
        for j in vis:
            offsets.setdefault(j, pts[j] - c)
        out.append(c)
    return np.array(out).reshape(-1, 3)


def _widest_pair(frames, min_common=3):
    """``(i, j)`` with the largest ``j - i`` whose frames share >= ``min_common`` seen points.

    The first/last frames of a pull are exactly when the gripper occludes the handle, so
    comparing only the window's end frames throws the whole measurement away.
    """
    valid = [np.isfinite(np.asarray(p, dtype=float)).all(axis=1) for p in frames]
    best = None
    for i in range(len(frames)):
        if valid[i].sum() < min_common:
            continue
        for j in range(len(frames) - 1, i, -1):
            if best is not None and j - i <= best[1] - best[0]:
                break
            if (valid[i] & valid[j]).sum() >= min_common:
                best = (i, j)
                break
    return best


def _horizontal(v):
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return _unit(v)


def face_normal(points, toward=None, horizontal=True):
    """Outward normal of the door/drawer front from tracked points -> ``(normal, conf, src)``.

    A least-squares plane through the points (measured on the object, e.g. the seed points
    at ``track_objects`` time while nothing occludes it), oriented towards ``toward`` (the
    camera or the gripper, i.e. the side the robot stands on). ``horizontal`` drops the z
    component: furniture fronts are vertical, so their pull normal is horizontal even when
    a small handle-only cloud gives a tilted plane. This is the only "model" used - no
    hinge location, side or radius is assumed.
    """
    pts = None if points is None else np.asarray(points, dtype=float).reshape(-1, 3)
    if pts is not None:
        pts = pts[np.isfinite(pts).all(axis=1)]
    if pts is None or len(pts) < 3:
        return None, 0.0, None
    fit = mo.fit_plane(pts, view_point=toward)
    if fit.normal is None:
        return None, 0.0, None
    n = _horizontal(fit.normal) if horizontal else _unit(fit.normal)
    if n is None:
        return None, 0.0, None
    if toward is not None and n @ (np.asarray(toward, dtype=float) - pts.mean(axis=0)) < 0:
        n = -n
    return n, float(fit.confidence), "plane_fit"


def _recent_chord(cents, seen, min_m):
    """Handle motion direction at the END of the window: last seen centre minus the latest
    earlier seen centre at least ``min_m`` away (the arc's local direction, not its chord)."""
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return None
    last = cents[idx[-1]]
    for j in idx[-2::-1]:
        if np.linalg.norm(last - cents[j]) >= min_m:
            return _unit(last - cents[j])
    return _unit(last - cents[idx[0]])


def summarize_pull(handle_frames, gripper_positions, dt=0.1, face_points=None,
                   view_point=None, horizontal_normal=True, logger=None):
    """Measured facts about one pull attempt + the direction to pull on the next attempt.

    No hinge prior: nothing about where the hinge is (or whether it is a door or a drawer)
    is given. Everything comes from the tracked points and the gripper trajectory.

    ``handle_frames``: sequence of ``(K, 3)`` arrays (``nan`` for unseen points).
    ``gripper_positions``: sequence of ``(3,)`` (same length) or ``None``.
    ``face_points``: ``(K, 3)`` points on the door front before the pull (the tracker's
    seed points); default = the first frame with >= 3 seen points.
    ``view_point``: camera / robot position used to orient the face normal outwards;
    default = the gripper sample farthest from the face plane (robot side).

    Direction logic:
      * the handle moved (>= ``HANDLE_MOVED_M``): it shows the opening direction itself.
        A confident revolute screw fit gives the hinge (estimated) and the arc tangent at
        the handle's latest position; otherwise the handle's recent motion. The gripper
        pull is judged against the handle's motion.
      * the handle did not move: the gripper pull is judged against the door-face outward
        normal (the first-pull direction for both doors and drawers); a pull more than
        ~60 deg off it is "wrong direction", and the normal is recommended.
    """
    n = len(handle_frames)
    pair = _widest_pair(handle_frames)
    buf = mo.MotionBuffer(capacity=2)
    if pair is not None:
        for k in pair:
            buf.push(handle_frames[k], frame_idx=k)
    est = mo.estimate_motion(buf, dt=dt * (pair[1] - pair[0] if pair else 1))

    cents = _track_centroid(handle_frames)
    seen = np.isfinite(cents).all(axis=1)
    idx_seen = np.flatnonzero(seen)
    handle_travel = 0.0
    if len(idx_seen) >= 2:
        handle_travel = float(np.linalg.norm(cents[idx_seen[-1]] - cents[idx_seen[0]]))
    handle_moved = handle_travel >= HANDLE_MOVED_M

    grip = None
    if gripper_positions is not None and len(gripper_positions) == n \
            and all(g is not None for g in gripper_positions):
        grip = np.asarray(gripper_positions, dtype=float).reshape(n, 3)
    gripper_travel = float(np.linalg.norm(grip[-1] - grip[0])) if grip is not None else None

    # -- opening angle (hinge-free) --------------------------------------------------
    angle_deg, angle_source = None, "unmeasured"
    if est.kind == mo.REVOLUTE and est.confidence > 0:
        angle_deg, angle_source = float(np.degrees(est.angle_rad)), "screw_fit"
    elif len(idx_seen) >= 2 and not handle_moved:
        angle_deg, angle_source = 0.0, "no_handle_motion"
    elif pair is not None:
        fit = mo.fit_rigid_motion(handle_frames[pair[0]], handle_frames[pair[1]])
        if np.isfinite(fit.rmse) and fit.rmse <= RIGID_FIT_MAX_RMSE_M:
            angle_deg = float(np.degrees(mo.rotation_angle_axis(fit.R)[0]))
            angle_source = "rigid_fit"
    facts = {
        "kind": est.kind, "confidence": round(float(est.confidence), 3),
        "angle_deg": None if angle_deg is None else round(angle_deg, 1),
        "angle_label": "unknown" if angle_deg is None else angle_label(angle_deg),
        "angle_source": angle_source,
        "handle_travel_m": round(handle_travel, 3),
        "handle_seen_frac": round(float(seen.mean()), 2) if n else 0.0,
        "handle_measured_frames": int(seen.sum()), "n_frames": int(n),
        "gripper_travel_m": None if gripper_travel is None else round(gripper_travel, 3),
        "axis_dir": None if est.axis_dir is None else np.round(est.axis_dir, 3).tolist(),
        "flags": [],
    }

    # -- door front normal (measured, not assumed) -----------------------------------
    if face_points is None:
        first = next((np.asarray(f, dtype=float) for f in handle_frames
                      if np.isfinite(np.asarray(f, dtype=float)).all(axis=1).sum() >= 3), None)
        face_points = first
    toward = view_point
    if toward is None and grip is not None and face_points is not None:
        # the gripper is always on the robot's side of the door: orient the normal towards
        # its sample farthest from the face plane (its first sample may sit on the handle)
        fp = np.asarray(face_points, dtype=float).reshape(-1, 3)
        fp = fp[np.isfinite(fp).all(axis=1)]
        plane = mo.fit_plane(fp) if len(fp) >= 3 else None
        if plane is not None and plane.normal is not None:
            toward = grip[int(np.argmax(np.abs((grip - plane.point) @ plane.normal)))]
    if toward is None and grip is not None:
        toward = grip[0]
    normal, n_conf, n_src = face_normal(face_points, toward=toward, horizontal=horizontal_normal)
    if normal is not None:
        facts["face_normal"] = np.round(normal, 3).tolist()
        facts["face_normal_confidence"] = round(n_conf, 2)

    # -- handle motion direction / hinge estimate ------------------------------------
    handle_dir, rec, rec_src = None, None, None
    if handle_moved:
        chord = _unit(cents[idx_seen[-1]] - cents[idx_seen[0]])
        recent = _recent_chord(cents, seen, HANDLE_MOVED_M)
        if est.kind == mo.REVOLUTE and est.confidence > 0 and est.hinge is not None:
            axis_dir, axis_point = est.hinge
            last = cents[idx_seen[-1]]
            tangent = _unit(mo.pull_direction(axis_dir, axis_point, last))
            if tangent is not None and tangent @ chord < 0:
                tangent = -tangent
            # Sparse/noisy tracks can give a confident but wrong screw (axis lying along
            # the motion, tiny radius). The tangent at the end of an arc of angle theta is
            # theta/2 off the chord and the axis is perpendicular to it; else distrust it.
            tol = np.radians(ARC_CHORD_TOL_DEG) + 0.5 * float(est.angle_rad)
            consistent = (tangent is not None and abs(float(axis_dir @ chord)) < AXIS_CHORD_MAX_COS
                          and float(tangent @ chord) >= np.cos(min(tol, np.pi / 2)))
            if tangent is not None and not consistent and logger is not None:
                logger(f"[review_evidence] screw fit inconsistent with the handle chord "
                       f"(axis.chord={float(axis_dir @ chord):.2f}, "
                       f"tangent.chord={float(tangent @ chord):.2f}) -> handle motion")
            if consistent:
                rec, rec_src = tangent, "arc_tangent"
                r_vec = last - axis_point
                r_vec = r_vec - (r_vec @ axis_dir) * axis_dir
                facts["hinge_estimate"] = {
                    "axis_dir": np.round(axis_dir, 3).tolist(),
                    "axis_point": np.round(axis_point, 3).tolist(),
                    "radius_m": round(float(np.linalg.norm(r_vec)), 3)}
        if rec is None and recent is not None:
            rec, rec_src = recent, "handle_motion"
        handle_dir = chord

    if grip is not None:
        # -- detach / follow ---------------------------------------------------------
        dist = np.linalg.norm(cents - grip, axis=1)
        ok = np.isfinite(dist)
        half = max(1, n // 2)
        baseline = float(np.nanmin(dist[:half])) if ok[:half].any() else None
        if baseline is not None:
            idx = np.flatnonzero(ok)
            idx = idx[idx >= half // 2]
            far = dist[idx] > baseline + DETACH_EXTRA_M
            k = _first_run(far, DETACH_SAMPLES)
            if k is not None:
                facts["detached_at_s"] = round(int(idx[k]) * dt, 2)
                facts["flags"].append("handle_left_gripper")
        # compare over the span the handle was actually measured in (a track lost early
        # must not read as "the handle stopped while the gripper went on")
        span = (float(np.linalg.norm(grip[idx_seen[-1]] - grip[idx_seen[0]]))
                if len(idx_seen) >= 2 else gripper_travel)
        if (span >= MIN_GRIPPER_TRAVEL_M
                and handle_travel < FOLLOW_RATIO_MIN * span):
            facts["flags"].append("handle_did_not_follow_gripper")

        # -- was the gripper pulling the right way? ----------------------------------
        move = _unit(grip[-1] - grip[0])
        ref, ref_src = (handle_dir, "handle_motion") if handle_dir is not None else \
            (normal, "door_face_normal")
        if move is not None and ref is not None and gripper_travel >= MIN_GRIPPER_TRAVEL_M:
            facts["gripper_pull_dir"] = np.round(move, 3).tolist()
            cos = float(move @ ref)
            facts["pull_alignment_cos"] = round(cos, 2)
            facts["pull_alignment_ref"] = ref_src
            if cos < WRONG_DIR_COS:
                facts["flags"].append("pull_direction_wrong")

        # The handle followed the gripper (no slip) but its track went stale long before
        # the end: the gripper's own late motion is the best measurement of where the
        # handle went (the door was carried along with it).
        stale = len(idx_seen) and (n - 1 - idx_seen[-1]) * dt >= STALE_TRACK_S
        if (handle_moved and stale and not facts["flags"]):
            g_dir = _recent_chord(grip, np.ones(n, dtype=bool), HANDLE_MOVED_M)
            if g_dir is not None:
                rec, rec_src = g_dir, "gripper_motion_while_holding"

    # -- recommendation for the next attempt -----------------------------------------
    if rec is None and normal is not None:
        aligned_no_motion = (not handle_moved and facts.get("pull_alignment_cos") is not None
                             and "pull_direction_wrong" not in facts["flags"])
        if aligned_no_motion:
            # pulled straight out of the face and nothing moved: the part does not open
            # that way (sliding panel / lift-up flap) - recommend sliding along the face
            side = _unit(np.array([-normal[1], normal[0], 0.0]))
            if side is not None:
                rec, rec_src = side, "face_lateral_after_no_motion"
                facts["alternative_dirs"] = [np.round(-side, 3).tolist(), [0.0, 0.0, 1.0],
                                             np.round(-normal, 3).tolist()]
        if rec is None:
            rec_src = ("face_normal_after_wrong_dir" if "pull_direction_wrong" in facts["flags"]
                       else "face_normal")
            rec = normal
    if rec is not None:
        facts["recommended_pull_dir"] = np.round(rec, 3).tolist()
        facts["recommended_pull_dir_source"] = rec_src
    if logger is not None:
        logger(f"[review_evidence] angle={facts['angle_deg']} ({angle_source}) "
               f"handle_travel={handle_travel:.3f} moved={handle_moved} "
               f"normal={facts.get('face_normal')} align={facts.get('pull_alignment_cos')}"
               f"/{facts.get('pull_alignment_ref')} flags={facts['flags']} "
               f"rec={facts.get('recommended_pull_dir')} ({rec_src})")
    return facts


def _vec(v):
    return "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"


def recommendation_text(facts):
    """One line for the retry prompt: where to pull next time and why (``None`` if unknown)."""
    rec = facts.get("recommended_pull_dir")
    if rec is None:
        return None
    src = facts.get("recommended_pull_dir_source")
    hinge = facts.get("hinge_estimate")
    if src == "arc_tangent" and hinge:
        why = (f"the door turned about an estimated hinge through {_vec(hinge['axis_point'])} "
               f"(axis {_vec(hinge['axis_dir'])}, handle radius {hinge['radius_m']} m); this "
               "is the arc tangent at the handle's latest position - follow the arc "
               "(re-aim the pull as the door swings) instead of a straight line")
    elif src == "handle_motion":
        why = "the direction the handle itself moved at the end of the attempt (keep following it)"
    elif src == "gripper_motion_while_holding":
        why = ("the gripper's late motion while it still carried the handle (the handle track "
               "was lost before the end)")
    elif src == "face_normal_after_wrong_dir":
        why = ("straight out of the door front (its outward normal); the last pull went "
               "sideways/along the face and the door did not move")
    elif src == "face_lateral_after_no_motion":
        alts = facts.get("alternative_dirs") or []
        why = ("sideways along the door front: the last pull went straight out of the front "
               "and the door did not move at all, so it probably slides in a track rather than "
               "swinging out (if it has a latch/lever, actuate that first). Pick the sideways "
               "sense toward the panel's free/open side (away from the edge its handle sits on)"
               + (f"; if it does not move either, try {_vec(alts[0])}, then lifting "
                  f"{_vec(alts[1])}, then pushing {_vec(alts[2])}" if len(alts) >= 3 else ""))
    else:
        why = ("straight out of the door front (its outward normal). The door did not move: "
               "make sure the handle/latch is actuated and the grasp is firm before pulling; "
               "if it still does not move, try pushing (the opposite direction)")
    return f"Recommended pull direction for the next attempt (world, unit vector): {_vec(rec)} - {why}."


def format_evidence(facts):
    """Prompt block for the reviewer: measured numbers, then the flags in plain words."""
    ang = facts.get("angle_deg")
    ang_txt = (f"opening angle {ang} deg -> '{facts['angle_label']}'" if ang is not None
               else "opening angle not measurable")
    lines = ["TRACKING EVIDENCE (measured by the 3D point tracker, not by a VLM; it is "
             "independent evidence - weigh it against the video):",
             f"- Tracked door/handle motion: {facts['kind']} "
             f"(confidence {facts['confidence']}), {ang_txt}; handle travelled "
             f"{facts['handle_travel_m']} m; handle position measured in "
             f"{facts.get('handle_measured_frames', '?')} of {facts.get('n_frames', '?')} frames."]
    if facts["gripper_travel_m"] is not None:
        line = f"- Gripper travelled {facts['gripper_travel_m']} m during the pull"
        if facts.get("gripper_pull_dir") is not None:
            line += f", direction {_vec(facts['gripper_pull_dir'])} (world)"
        lines.append(line + ".")
    if facts.get("face_normal") is not None:
        lines.append(f"- Door front outward normal (plane fit on the tracked points): "
                     f"{_vec(facts['face_normal'])}.")
    if facts.get("hinge_estimate"):
        h = facts["hinge_estimate"]
        lines.append(f"- Estimated hinge from the observed rotation: through {_vec(h['axis_point'])}, "
                     f"axis {_vec(h['axis_dir'])}, handle radius {h['radius_m']} m.")
    if facts.get("pull_alignment_cos") is not None:
        ref = {"handle_motion": "the handle's own motion",
               "door_face_normal": "the door front's outward normal"}[facts["pull_alignment_ref"]]
        ok = facts["pull_alignment_cos"] >= WRONG_DIR_COS
        lines.append(f"- Pull direction check: cos(gripper motion, {ref}) = "
                     f"{facts['pull_alignment_cos']} -> "
                     + ("ALIGNED (the gripper pulled the right way)." if ok else "NOT aligned."))
    words = {
        "handle_left_gripper": "the handle separated from the gripper at "
                               f"t={facts.get('detached_at_s')}s (slip / lost handle)",
        "handle_did_not_follow_gripper": "the handle barely moved while the gripper did "
                                         "(grasp not holding the handle, or the door is latched)",
        "pull_direction_wrong": f"the gripper did not pull along the opening direction "
                                f"(cos={facts.get('pull_alignment_cos')}): pulling the wrong way",
    }
    for f in facts["flags"]:
        lines.append(f"- WARNING: {words[f]}.")
    if not facts["flags"]:
        lines.append("- No grasp/direction anomalies detected.")
    if facts.get("angle_source") == "rigid_fit":
        lines.append("- Note: the angle comes from a rigid fit of the tracked points; the "
                     "hinge rotation fit itself was not confident.")
    elif ang is None:
        lines.append("- Note: the handle was not tracked well enough to measure the opening; "
                     "judge the door state from the video.")
    elif facts["confidence"] == 0:
        lines.append("- Note: motion confidence is 0 (little/uncertain motion); the angle is "
                     "'no measurable opening', not a precise value.")
    rec = recommendation_text(facts)
    if rec:
        lines.append("- " + rec.replace("Recommended pull direction for the next attempt",
                                        "RECOMMENDED NEXT PULL DIRECTION"))
    return "\n".join(lines)


# -- production adapter: TrackingReporter frames -> summarize_pull ------------------------
#: Tasks whose tracked object is an articulated part being pulled/pushed open or closed:
#: only for those is a "pull direction" meaningful.
ARTICULATED_TASK_RE = re.compile(
    r"\b(open|close|shut|pull|push|door|drawer|cabinet|cupboard|handle|lid|hatch|hinge|"
    r"fridge|oven|microwave|dishwasher)\b", re.IGNORECASE)


def _quat_xyzw_to_R(q):
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def evidence_from_tracking_frames(frames, seeds, name, start_step=0, logger=None):
    """``TrackingReporter.frames`` (dicts) of one attempt -> ``(facts, text)`` or ``None``.

    Per frame (``trajectory_step >= start_step``) the object's ``K`` seed points are carried
    by its **trusted** pose (measured, not tentative) when the 3D provider fits one; the
    fused ``world_point`` (if measured, not predicted/lost) is appended as point ``K`` so
    centroid-only trackers still give travel / detach / direction evidence. Predictions
    (Kalman coast, gripper FK) are never used as measurements. The seeds double as the door
    front points for the outward normal.
    """
    seeds = np.asarray(seeds, dtype=float).reshape(-1, 3)
    k = len(seeds)
    pts, grip, times = [], [], []
    for fr in frames or []:
        if int(fr.get("trajectory_step", 0)) < int(start_step):
            continue
        obj = (fr.get("objects") or {}).get(name)
        gp = (fr.get("gripper") or {}).get("world_pos")
        if obj is None or gp is None:
            continue
        row = np.full((k + 1, 3), np.nan)
        pose = obj.get("pose")
        if (pose and str(pose.get("source", "")).startswith("measured")
                and not pose.get("tentative") and k):
            row[:k] = seeds @ _quat_xyzw_to_R(pose["quat_xyzw"]).T + np.asarray(pose["t"], dtype=float)
        if obj.get("world_point") is not None and not obj.get("predicted") and not obj.get("lost"):
            row[k] = np.asarray(obj["world_point"], dtype=float)
        pts.append(row)
        grip.append(np.asarray(gp, dtype=float))
        times.append(fr.get("sim_time"))
    if len(pts) < 3:
        if logger is not None:
            logger(f"[review_evidence] '{name}': only {len(pts)} tracked frames since step "
                   f"{start_step} - no tracking evidence for the reviewer")
        return None
    dt = 1.0
    if all(t is not None for t in times):
        d = np.diff(np.asarray(times, dtype=float))
        d = d[d > 0]
        if len(d):
            dt = float(np.median(d))
    facts = summarize_pull(pts, grip, dt=dt, face_points=seeds, logger=logger)
    facts["object"] = name
    return facts, format_evidence(facts)
