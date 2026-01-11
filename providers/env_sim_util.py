import numpy as np


def _rotmat_to_quat_xyzw(R):
    """
    Convert a 3x3 rotation matrix to a normalized quaternion [x, y, z, w].
    Extracted from metaworld_server for reuse across Mujoco-based envs.
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def _get_objects_pose(env, names):
    """
    Return pose and approximate dimensions for requested Mujoco entities.
    If names is empty/None, include all geoms in the scene.

    For each item we report:
      - pos: world position (3,)
      - quat: orientation as [x,y,z,w]
      - dims: approximate oriented box dimensions [dx,dy,dz]
      - aabb_min/aabb_max: world axis-aligned bounding box corners (3,) if dims available
      - kind: "geom"|"site"|"body"
      - geom_type: Mujoco geom type integer when applicable
    """
    out = {}
    include_all = not names

    # Helper to add an oriented box entry and optional AABB
    def _add(name, pos_arr, R, dims, kind, geom_type=None):
        quat = _rotmat_to_quat_xyzw(R)
        entry = {
            "pos": np.asarray(pos_arr, dtype=np.float64).tolist(),
            "quat": quat.tolist(),
            "kind": kind,
        }
        if geom_type is not None:
            entry["geom_type"] = int(geom_type)
        if dims is not None:
            dx, dy, dz = [float(x) for x in dims]
            entry["dims"] = [dx, dy, dz]
            # Compute world AABB by transforming 8 corners
            hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
            corners = np.array([
                [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
                [-hx, -hy, hz], [hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz],
            ], dtype=np.float64)
            wc = (R @ corners.T).T + np.asarray(pos_arr, dtype=np.float64)
            aabb_min = wc.min(axis=0).tolist()
            aabb_max = wc.max(axis=0).tolist()
            entry["aabb_min"] = [float(v) for v in aabb_min]
            entry["aabb_max"] = [float(v) for v in aabb_max]
        out[name] = entry

    try:
        import mujoco
    except Exception:
        mujoco = None

    # 1) Geoms: preferred source (have sizes/types)
    try:
        ngeom = int(env.model.ngeom)
        for gid in range(ngeom):
            # Resolve geom name via MuJoCo core API for reliability
            gname = None
            try:
                if mujoco is not None:
                    gname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            except Exception:
                gname = None
            if not gname:
                try:
                    gname = env.model.geom_id2name(gid)
                except Exception:
                    gname = None
            if not gname or len(str(gname)) == 0:
                gname = f"geom_{gid}"
            if (not include_all) and (gname not in (names or [])):
                continue
            pos = env.data.geom_xpos[gid]
            R = env.data.geom_xmat[gid].reshape(3, 3)
            gtype = int(env.model.geom_type[gid]) if mujoco is not None else None
            size = env.model.geom_size[gid]
            dims = None
            try:
                if mujoco is not None:
                    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        hx, hy, hz = size[0], size[1], size[2]
                        dims = [2 * hx, 2 * hy, 2 * hz]
                    elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                        r, hl = size[0], size[1]
                        dims = [2 * r, 2 * r, 2 * hl]
                    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                        r = size[0]
                        dims = [2 * r, 2 * r, 2 * r]
                    else:
                        # Mesh/plane/others: skip dims (no simple closed-form)
                        dims = None
            except Exception:
                dims = None
            # Prefer real geom name; if missing, include body name to help identification
            try:
                bid = int(env.model.geom_bodyid[gid])
                bname = None
                if mujoco is not None:
                    bname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if not bname:
                    bname = env.model.body_id2name(bid)
            except Exception:
                bid, bname = None, None
            key = gname if (isinstance(gname, str) and len(gname) > 0) else f"geom_{gid}"
            _add(key, pos, R, dims, kind="geom", geom_type=gtype)
            if bname and (include_all or bname in (names or [])) and bname not in out:
                # Add a light-weight alias for the parent body for discoverability
                _add(bname, pos, R, None, kind="body", geom_type=None)
    except Exception:
        pass

    # 2) Sites: include all sites with their names (often used for semantic markers like 'handle')
    try:
        nsite = int(env.model.nsite)
        for sid in range(nsite):
            # Resolve site name via MuJoCo API
            sname = None
            try:
                if mujoco is not None:
                    sname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, sid)
            except Exception:
                sname = None
            if not sname:
                try:
                    sname = env.model.site_id2name(sid)
                except Exception:
                    sname = None
            if not sname or len(str(sname)) == 0:
                sname = f"site_{sid}"
            if (not include_all) and (sname not in (names or [])):
                continue
            pos = env.data.site_xpos[sid]
            R = env.data.site_xmat[sid].reshape(3, 3)
            # Try to extract site size as dims when possible
            dims = None
            try:
                size = env.model.site_size[sid]
                if size is not None and len(size) >= 3:
                    hx, hy, hz = float(size[0]), float(size[1]), float(size[2])
                    dims = [2 * hx, 2 * hy, 2 * hz]
            except Exception:
                dims = None
            key = sname if (isinstance(sname, str) and len(sname) > 0) else f"site_{sid}"
            _add(key, pos, R, dims, kind="site", geom_type=None)
    except Exception:
        pass

    return out

