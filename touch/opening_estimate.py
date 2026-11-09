"""Robust opening (port / hole) position from many noisy wrist-camera pointing candidates.

Pointing at a small opening on a side face is consistent in HEIGHT across views but scatters
several cm ALONG the face (texture marks, the opening is a few pixels), and single views
sometimes hit something else entirely. The insertion search finds the along-face position by
feel (`search_line`), but needs the height right within ~1 mm - so the height is what this
estimates carefully. Pure function of camera-derived candidates (Sim2Real safe).
"""
import statistics


def _pos(c):
    p = c.get("position") if isinstance(c, dict) else c
    return [float(v) for v in p[:3]] if p is not None and len(p) >= 3 else None


def estimate_opening(candidates, edge_candidates=None, axis=(1, 0, 0), face_tol=0.01,
                     z_tol=0.004, edge_gap=0.003):
    """Aggregate opening candidates (all views of one face) into one aim point.

    candidates: `locate_with_wrist_camera(...)` results (dicts with "position") or [x, y, z]
        lists, from ALL views of the face - pass every one, outliers are rejected here.
    edge_candidates: optional hits of the face's top edge; port hits within `edge_gap` of the
        edge height are dropped (the opening sits below the top surface).
    axis: insertion axis (the face normal, any heading); hits are judged in the face's own frame.

    Returns dict: position [x, y, z] (None if no candidate survives), n_used, n_total,
    z_spread / along_spread (max-min of the kept), search_half (suggested `search_half` for
    insert_with_touch's line search), search_cross (0.002 when the kept heights spread > 2 mm,
    else None), search_line (the along-face direction), confident
    (>= 3 kept and z spread <= 4 mm), notes (list of str).
    """
    notes = []
    pts = [p for p in (_pos(c) for c in (candidates or [])) if p is not None]
    # face frame: n = the axis (the face normal; its horizontal part for a side face), t = along the
    # face (horizontal), z = up. Works for any face heading, not only world-axis-aligned faces.
    n = [float(v) for v in axis]
    if abs(n[2]) < 0.9:
        n = [n[0], n[1], 0.0]
    nn = sum(v * v for v in n) ** 0.5
    n = [v / nn for v in n]
    t = [-n[1], n[0], 0.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0]
    tn = sum(v * v for v in t) ** 0.5
    t = [v / tn for v in t]
    if t[max(range(3), key=lambda i: abs(t[i]))] < 0:
        t = [-v for v in t]
    dot = lambda p, d: sum(p[i] * d[i] for i in range(3))  # noqa: E731
    out = {"position": None, "n_used": 0, "n_total": len(pts), "z_spread": None, "along_spread": None,
           "search_half": 0.025, "search_line": [round(v, 4) for v in t],
           "search_cross": None, "confident": False, "notes": notes}
    if not pts:
        notes.append("no candidates")
        return out

    # 1) the face plane: the coordinate along the axis is where most hits agree
    plane = statistics.median(dot(p, n) for p in pts)
    on_face = [p for p in pts if abs(dot(p, n) - plane) <= face_tol]
    if len(on_face) < len(pts):
        notes.append(f"dropped {len(pts) - len(on_face)} off-face hit(s) (> {face_tol * 100:.0f} cm from the face plane)")

    # 2) height: median, reject far hits, median again
    z0 = statistics.median(p[2] for p in on_face)
    kept = [p for p in on_face if abs(p[2] - z0) <= z_tol]
    if len(kept) < len(on_face):
        notes.append(f"dropped {len(on_face) - len(kept)} height outlier(s) (> {z_tol * 1000:.0f} mm from median z {z0:.4f})")

    # 3) top-edge hits: only trusted when the edge is ABOVE the opening (else the edge pointing is wrong)
    edges = [p for p in (_pos(c) for c in (edge_candidates or [])) if p is not None and abs(dot(p, n) - plane) <= face_tol]
    if edges and kept:
        ez = statistics.median(p[2] for p in edges)
        mz = statistics.median(p[2] for p in kept)
        if ez > mz:
            near = [p for p in kept if abs(p[2] - ez) <= edge_gap]
            if near and len(near) < len(kept):
                kept = [p for p in kept if abs(p[2] - ez) > edge_gap]
                notes.append(f"dropped {len(near)} hit(s) within {edge_gap * 1000:.0f} mm of the top edge z {ez:.4f}")
        else:
            notes.append(f"top-edge z {ez:.4f} is not above the opening z {mz:.4f}: edge hits ignored")
    if not kept:
        notes.append("no candidate survived")
        return out

    # rebuild from the face-frame medians (a per-coordinate median is off the face when it is turned)
    mn, mt = statistics.median(dot(p, n) for p in kept), statistics.median(dot(p, t) for p in kept)
    pos = [mn * n[i] + mt * t[i] for i in range(2)] + [statistics.median(p[2] for p in kept)]
    zs = [p[2] for p in kept]
    al = [dot(p, t) for p in kept]
    z_spread = max(zs) - min(zs)
    along_spread = max(al) - min(al)
    # the line search is centred on the median, so it must reach the FARTHEST kept hit (+ ~half a
    # port width); spread/2 missed the one correct hit in usb_u5 (median 0.4713, hits 0.462..0.498)
    reach = max(abs(v - statistics.median(al)) for v in al)
    out.update(position=[round(v, 4) for v in pos], n_used=len(kept), z_spread=round(z_spread, 4),
               along_spread=round(along_spread, 4),
               search_half=round(min(0.04, max(0.015, reach + 0.006)), 3),
               search_cross=0.002 if z_spread > 0.002 else None,
               confident=len(kept) >= 3 and z_spread <= 0.004)
    if len(kept) < 3:
        notes.append("fewer than 3 consistent hits: take 2 more views (camera ~5 cm above the opening height)")
    return out


def format_estimate(est):
    p = est.get("position")
    ps = "None" if p is None else "[" + ", ".join(f"{v:.4f}" for v in p) + "]"
    return (f"estimate_opening: position={ps} used {est['n_used']}/{est['n_total']} "
            f"z_spread={est['z_spread']} along_spread={est['along_spread']} "
            f"search_line={est['search_line']} search_half={est['search_half']} "
            f"search_cross={est['search_cross']} confident={est['confident']}"
            + ("".join(f"\n  - {n}" for n in est["notes"]) if est["notes"] else ""))
