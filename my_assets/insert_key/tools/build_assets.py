"""Stage 2: build the insert_key scene assets (meshes, textures, URDFs).

Run from the repo root with a python that has numpy, trimesh, scipy, PIL and
fast_simplification (e.g. the ``vlm_genesis`` conda env)::

    python my_assets/insert_key/tools/build_assets.py

Inputs: ``cache/robodojo/extracted/key.npz`` + ``key__extracted_image_2.jpg`` produced
by ``extract_robodojo_usdz.py`` (stage 1). Everything else is procedural.

Outputs (committed): ``my_assets/insert_key/meshes/*`` (OBJ/MTL + textures) and the URDFs
``key.urdf``, ``lock.urdf``, ``key_stand.urdf``, ``table.urdf``.

Design notes
------------
* Only the key mesh comes from RoboDojo (Apache-2.0). The lock, stand and table are
  generated here so the lock can have a real revolute plug (RoboDojo's slot is a
  static puck).
* Visual meshes are built from convex pieces (rounded boxes / cylinders as convex
  hulls) - no boolean engine is needed. Keyway openings are drawn as dark inset
  patches; the *collision* keyways are real holes made of boxes.
* Every visual is a single-material OBJ so PyBullet (one texture per visual) and
  Genesis render it identically.
"""

import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
CACHE = os.path.join(REPO, "cache", "robodojo", "extracted")
MESH_DIR = os.path.join(ROOT, "meshes")
# Textures sit next to the OBJs: trimesh (Genesis) will not resolve a "../" map_Kd path.
TEX_DIR = os.path.join(ROOT, "meshes")

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Geometry constants (metres). Shared with sim_envs/pybullet/insert_key.py via
# my_assets/insert_key/geometry.json so the profile never hard-codes them twice.
# ---------------------------------------------------------------------------
TABLE_TOP_Z = 0.75
TABLE_SIZE = (1.20, 1.00, 0.04)
TABLE_CENTER_XY = (0.0, 0.30)

# Key (RoboDojo mesh frame: z up, blade towards -z, bow faces +-y).
KEY_BLADE = dict(x=(-0.0060, 0.0068), y=(-0.0052, 0.0052), z=(-0.0457, 0.0))
KEY_SHOULDER = dict(x=(-0.012, 0.012), y=(-0.0050, 0.0050), z=(0.0, 0.015))
KEY_BOW = dict(x=(-0.020, 0.020), y=(-0.0046, 0.0046), z=(0.015, 0.0455))
KEY_MASS = 0.030

# Keyway clearance hole (x along the blade width, y along its thickness).
SLOT_HX, SLOT_HY = 0.0080, 0.0066
# Collision-only: the mesh blade is a chunky 12.5x10.3 mm prism that slips past the keyway
# walls under ~1.6 mm of (soft, Genesis) contact penetration. A flat 3 mm collision blade
# in a 4.4 mm keyway needs ~4.5 mm to slip, so turning the key reliably drives the plug.
KEY_BLADE_COLL_HY = 0.0015
KEYWAY_COLL_HY = 0.0022

# Lock (link frames: base origin on the table top, plug joint on the housing top).
PLINTH = (0.13, 0.13, 0.025)
HOUSING_R, HOUSING_TOP = 0.042, 0.095
FLANGE_R, FLANGE_H = 0.049, 0.008
PLUG_R, PLUG_PROUD, PLUG_DEPTH = 0.026, 0.003, 0.055
PLUG_SQ = 0.0180            # half-size of the square plug collision body
RING_INNER = 0.026          # housing collision ring leaves room for the turning plug
FUNNEL_TOP_HX, FUNNEL_TOP_HY, FUNNEL_DEPTH = 0.0165, 0.0135, 0.008
KEYWAY_DEPTH = 0.050        # plug top -> keyway floor
PLUG_LIMIT = 1.62           # rad, ~93 deg each way
PLUG_DAMPING = 0.02

# Key stand.
STAND = (0.075, 0.055, 0.050)
STAND_SLOT_DEPTH = 0.048


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------
def _noise(shape, sigma, rng=RNG):
    n = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    n -= n.min()
    return n / (n.max() + 1e-9)


def _save(img, name):
    path = os.path.join(TEX_DIR, name)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return name


def wood_texture(name, light, dark, rings=22.0, size=512, figure=0.35):
    """Plain-sawn wood: wavy growth rings along v, fine pores, slight colour drift."""
    v, u = np.mgrid[0:size, 0:size] / size
    warp = _noise((size, size), (size / 6, size / 40)) * 2.0 + _noise((size, size), (size / 30, size / 8)) * 0.4
    ring = np.sin(2 * np.pi * (u * rings + warp * 2.2))
    ring = (ring * 0.5 + 0.5) ** 2.2
    pores = _noise((size, size), (0.6, 6.0))
    pores = (pores > 0.82).astype(float) * 0.35
    drift = _noise((size, size), size / 5)
    t = np.clip(ring * figure + drift * 0.35 + pores, 0, 1)[..., None]
    img = np.asarray(light)[None, None, :] * (1 - t) + np.asarray(dark)[None, None, :] * t
    return _save(img, name)


def brushed_texture(name, base, streak=18.0, size=512, speckle=4.0):
    """Brushed metal: long 1-D streaks + faint speckle."""
    s = _noise((size, size), (0.5, size / 3))
    fine = _noise((size, size), (0.3, 40))
    sp = _noise((size, size), 1.0)
    lum = (s - 0.5) * streak + (fine - 0.5) * streak * 0.6 + (sp - 0.5) * speckle
    img = np.asarray(base)[None, None, :] + lum[..., None]
    return _save(img, name)


def build_textures():
    os.makedirs(TEX_DIR, exist_ok=True)
    return {
        "oak": wood_texture("oak.png", (200, 158, 112), (164, 117, 74), rings=26.0, figure=0.28),
        "walnut": wood_texture("walnut.png", (110, 70, 45), (58, 34, 22), rings=20.0, figure=0.45),
        "steel": brushed_texture("steel.png", (182, 186, 190), streak=22.0),
        "brass": brushed_texture("brass.png", (204, 160, 74), streak=16.0),
        "dark_steel": brushed_texture("dark_steel.png", (58, 60, 64), streak=10.0),
    }


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def rounded_box(size, radius, center=(0, 0, 0), segs=6):
    """Convex hull of 8 corner spheres -> box with filleted edges."""
    sx, sy, sz = (np.asarray(size, float) / 2.0) - radius
    sph = trimesh.creation.icosphere(subdivisions=2, radius=radius).vertices
    pts = [sph + np.array([cx * sx, cy * sy, cz * sz])
           for cx in (-1, 1) for cy in (-1, 1) for cz in (-1, 1)]
    hull = trimesh.convex.convex_hull(np.vstack(pts))
    hull.apply_translation(center)
    return hull


def rounded_cylinder(radius, z0, z1, fillet=0.0, sections=64, center_xy=(0, 0), fillet_bottom=False):
    """Cylinder whose top (and optionally bottom) rim is filleted."""
    rings = []
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    c, s = np.cos(th), np.sin(th)

    def ring(r, z):
        return np.stack([c * r + center_xy[0], s * r + center_xy[1], np.full_like(c, z)], 1)

    if fillet > 0 and fillet_bottom:
        for a in np.linspace(0, np.pi / 2, 5):
            rings.append(ring(radius - fillet + fillet * np.sin(a), z0 + fillet - fillet * np.cos(a)))
    else:
        rings.append(ring(radius, z0))
    if fillet > 0:
        for a in np.linspace(0, np.pi / 2, 6):
            rings.append(ring(radius - fillet + fillet * np.cos(a), z1 - fillet + fillet * np.sin(a)))
    else:
        rings.append(ring(radius, z1))
    return trimesh.convex.convex_hull(np.vstack(rings))


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2.0)
    return m


def shade(mesh, crease_deg=35.0, uv_scale=1.0, cyl_axis=None):
    """Split a mesh at creases, return (V, N, UV, F) ready for OBJ export.

    Normals are area-weighted over faces within ``crease_deg`` of the corner's face,
    so flat panels stay flat while fillets shade smoothly. UVs are tri-planar
    (by face normal) or cylindrical around z when ``cyl_axis`` is given.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    fn = mesh.face_normals
    area = mesh.area_faces
    cos_t = np.cos(np.radians(crease_deg))
    vf = mesh.vertex_faces
    F = mesh.faces
    V_out, N_out, UV_out = [], [], []
    for fi, face in enumerate(F):
        for v in face:
            adj = vf[v][vf[v] >= 0]
            ok = adj[(fn[adj] @ fn[fi]) > cos_t]
            n = (fn[ok] * area[ok, None]).sum(0)
            n = n / (np.linalg.norm(n) + 1e-12)
            p = mesh.vertices[v]
            if cyl_axis is not None and abs(fn[fi][2]) < 0.7:
                ang = np.arctan2(p[1] - cyl_axis[1], p[0] - cyl_axis[0])
                uv = (ang * 0.05, p[2])
            else:
                ax = int(np.argmax(np.abs(fn[fi])))
                uv = (p[1], p[2]) if ax == 0 else (p[0], p[2]) if ax == 1 else (p[0], p[1])
            V_out.append(p)
            N_out.append(n)
            UV_out.append(np.asarray(uv) * uv_scale)
    Fo = np.arange(len(V_out)).reshape(-1, 3)
    return np.array(V_out), np.array(N_out), np.array(UV_out), Fo


def concat(parts):
    V, N, UV, F, off = [], [], [], [], 0
    for v, n, uv, f in parts:
        V.append(v); N.append(n); UV.append(uv); F.append(f + off)
        off += len(v)
    return np.vstack(V), np.vstack(N), np.vstack(UV), np.vstack(F)


def write_obj(name, parts, texture=None, kd=(1, 1, 1), ks=0.25, ns=60):
    """Write ``meshes/<name>.obj`` + ``.mtl`` with a single material."""
    V, N, UV, F = concat(parts) if isinstance(parts, list) else parts
    os.makedirs(MESH_DIR, exist_ok=True)
    mtl = f"{name}.mtl"
    with open(os.path.join(MESH_DIR, mtl), "w") as fh:
        fh.write(f"newmtl {name}\n")
        fh.write(f"Ka 0.2 0.2 0.2\nKd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
        fh.write(f"Ks {ks:.3f} {ks:.3f} {ks:.3f}\nNs {ns}\nd 1.0\nillum 2\n")
        if texture:
            fh.write(f"map_Kd {texture}\n")
    lines = [f"mtllib {mtl}", f"o {name}"]
    lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in V]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in UV]
    lines += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in N]
    lines.append(f"usemtl {name}")
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in (F + 1)]
    with open(os.path.join(MESH_DIR, f"{name}.obj"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return f"meshes/{name}.obj"


# ---------------------------------------------------------------------------
# Key: decimate RoboDojo mesh, keep its UVs
# ---------------------------------------------------------------------------
def build_key_mesh(target_faces=12000):
    import fast_simplification as fs

    d = np.load(os.path.join(CACHE, "key.npz"))
    V, F, cuv = d["vertices"], d["faces"], d["corner_uv"]
    uv = np.zeros((len(V), 2))
    for k in range(3):
        uv[F[:, k]] = cuv[:, k]
    reduction = 1.0 - target_faces / len(F)
    _v, _f, collapses = fs.simplify(V.astype(np.float32), F.astype(np.int32), reduction,
                                   return_collapses=True)
    V2, F2, mapping = fs.replay_simplification(V.astype(np.float32), F.astype(np.int32), collapses)
    uv2 = np.zeros((len(V2), 2))
    uv2[mapping] = uv  # every original vertex lands on a surviving one; last write wins
    m = trimesh.Trimesh(V2, F2, process=False)
    N2 = m.vertex_normals
    print(f"[key] {len(F)} -> {len(F2)} faces")

    img = Image.open(os.path.join(CACHE, "key__extracted_image_2.jpg")).convert("RGB")
    img = img.resize((1024, 1024), Image.LANCZOS)
    os.makedirs(TEX_DIR, exist_ok=True)
    img.save(os.path.join(TEX_DIR, "key_basecolor.jpg"), quality=92)
    return write_obj("key", (np.asarray(V2, float), N2, uv2, np.asarray(F2)),
                     texture="key_basecolor.jpg", ks=0.6, ns=120)


# ---------------------------------------------------------------------------
# URDF helpers
# ---------------------------------------------------------------------------
def _xyz(v):
    return " ".join(f"{float(a):.5f}" for a in v)


def coll_box(lo, hi, rpy=(0, 0, 0), center=None, size=None):
    if center is None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        center, size = (lo + hi) / 2.0, hi - lo
    return (f'    <collision><origin xyz="{_xyz(center)}" rpy="{_xyz(rpy)}"/>'
            f'<geometry><box size="{_xyz(size)}"/></geometry></collision>')


def vis_mesh(path, rgba=None, name=None):
    mat = ""
    if rgba is not None:
        mat = f'<material name="{name}"><color rgba="{_xyz(rgba)}"/></material>'
    return (f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><mesh filename="{path}" scale="1 1 1"/></geometry>{mat}</visual>')


def inertial(mass, size, com=(0, 0, 0)):
    x, y, z = size
    ixx, iyy, izz = (mass / 12 * (y * y + z * z), mass / 12 * (x * x + z * z), mass / 12 * (x * x + y * y))
    return (f'    <inertial><origin xyz="{_xyz(com)}" rpy="0 0 0"/><mass value="{mass}"/>'
            f'<inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}"/></inertial>')


def write_urdf(name, body):
    path = os.path.join(ROOT, f"{name}.urdf")
    with open(path, "w") as fh:
        fh.write(f'<?xml version="1.0"?>\n<!-- Generated by tools/build_assets.py - do not edit. -->\n'
                 f'<robot name="{name}">\n{body}\n</robot>\n')
    print(f"[urdf] {path}")


def keyway_collisions(top, depth, hx, hy, outer, floor=0.006):
    """Square block of half-size ``outer`` with a hx*hy rectangular hole from ``top``
    down ``depth``, closed by a floor."""
    z0, z1 = top - depth, top
    return [
        coll_box((hx, -outer, z0), (outer, outer, z1)),
        coll_box((-outer, -outer, z0), (-hx, outer, z1)),
        coll_box((-hx, hy, z0), (hx, outer, z1)),
        coll_box((-hx, -outer, z0), (hx, -hy, z1)),
        coll_box((-outer, -outer, z0 - floor), (outer, outer, z0)),
    ]


def funnel_collisions(top, depth, hx0, hy0, hx1, hy1, thick=0.003):
    """Four tilted plates forming a pyramidal lead-in from (hx1,hy1) at ``top`` to the
    (hx0,hy0) slot at ``top-depth``."""
    out = []
    zc = top - depth / 2.0
    for sign in (1, -1):
        # +-x plates: tilt about y.
        dx = hx1 - hx0
        L = np.hypot(dx, depth)
        ang = np.arctan2(dx, depth)
        nx = np.cos(ang)  # outward normal x-component
        cx = sign * ((hx0 + hx1) / 2.0 + nx * thick / 2.0)
        cz = zc + np.sin(ang) * thick / 2.0 * -1
        out.append(coll_box(None, None, rpy=(0, sign * ang, 0), center=(cx, 0, cz),
                            size=(thick, 2 * hy1, L)))
        dy = hy1 - hy0
        L = np.hypot(dy, depth)
        ang = np.arctan2(dy, depth)
        ny = np.cos(ang)
        cy = sign * ((hy0 + hy1) / 2.0 + ny * thick / 2.0)
        cz = zc + np.sin(ang) * thick / 2.0 * -1
        out.append(coll_box(None, None, rpy=(-sign * ang, 0, 0), center=(0, cy, cz),
                            size=(2 * hx1, thick, L)))
    return out


# ---------------------------------------------------------------------------
# Scene pieces
# ---------------------------------------------------------------------------
def build_table(tex):
    tx, ty, tz = TABLE_SIZE
    cx, cy = TABLE_CENTER_XY
    top = rounded_box((tx, ty, tz), 0.012, center=(cx, cy, TABLE_TOP_Z - tz / 2))
    top_path = write_obj("table_top", [shade(top, uv_scale=2.2)], texture=tex["oak"], ks=0.15, ns=30)
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = cx + sx * (tx / 2 - 0.07), cy + sy * (ty / 2 - 0.07)
            legs.append(shade(rounded_cylinder(0.024, 0.0, TABLE_TOP_Z - tz - 0.03, 0.0, 32, (lx, ly)),
                              cyl_axis=(lx, ly), uv_scale=4.0))
    apron_z = TABLE_TOP_Z - tz - 0.05
    for sy in (-1, 1):
        legs.append(shade(rounded_box((tx - 0.14, 0.03, 0.06), 0.004,
                                      center=(cx, cy + sy * (ty / 2 - 0.07), apron_z)), uv_scale=4.0))
    for sx in (-1, 1):
        legs.append(shade(rounded_box((0.03, ty - 0.14, 0.06), 0.004,
                                      center=(cx + sx * (tx / 2 - 0.07), cy, apron_z)), uv_scale=4.0))
    legs_path = write_obj("table_frame", legs, texture=tex["dark_steel"], ks=0.5, ns=80)
    body = (
        '  <link name="table">\n'
        + inertial(20.0, (tx, ty, tz), (cx, cy, TABLE_TOP_Z - tz / 2)) + "\n"
        + vis_mesh(top_path) + "\n" + vis_mesh(legs_path) + "\n"
        + coll_box((cx - tx / 2, cy - ty / 2, TABLE_TOP_Z - tz), (cx + tx / 2, cy + ty / 2, TABLE_TOP_Z)) + "\n"
        + "  </link>"
    )
    write_urdf("table", body)


def build_key_stand(tex):
    sx, sy, sz = STAND
    block = rounded_box((sx, sy, sz), 0.006, center=(0, 0, sz / 2))
    p = write_obj("key_stand", [shade(block, uv_scale=3.0)], texture=tex["walnut"], ks=0.3, ns=50)
    # Dark slot opening drawn just above the top face, with a lighter bevel ring.
    bevel = box((-SLOT_HX - 0.0025, -SLOT_HY - 0.0025, sz), (SLOT_HX + 0.0025, SLOT_HY + 0.0025, sz + 0.0004))
    hole = box((-SLOT_HX, -SLOT_HY, sz), (SLOT_HX, SLOT_HY, sz + 0.0007))
    pb = write_obj("key_stand_bevel", [shade(bevel)], kd=(0.16, 0.10, 0.07), ks=0.1)
    ph = write_obj("key_stand_hole", [shade(hole)], kd=(0.03, 0.03, 0.03), ks=0.0)
    outer_x, outer_y = sx / 2, sy / 2
    z0 = sz - STAND_SLOT_DEPTH
    colls = [
        coll_box((-outer_x, -outer_y, 0), (outer_x, outer_y, z0)),
        coll_box((SLOT_HX, -outer_y, z0), (outer_x, outer_y, sz)),
        coll_box((-outer_x, -outer_y, z0), (-SLOT_HX, outer_y, sz)),
        coll_box((-SLOT_HX, SLOT_HY, z0), (SLOT_HX, outer_y, sz)),
        coll_box((-SLOT_HX, -outer_y, z0), (SLOT_HX, -SLOT_HY, sz)),
    ]
    body = (
        '  <link name="key_stand">\n'
        + inertial(0.6, STAND, (0, 0, sz / 2)) + "\n"
        + vis_mesh(p) + "\n"
        + vis_mesh(pb, (0.16, 0.10, 0.07, 1), "stand_bevel") + "\n"
        + vis_mesh(ph, (0.03, 0.03, 0.03, 1), "stand_hole") + "\n"
        + "\n".join(colls) + "\n  </link>"
    )
    write_urdf("key_stand", body)


def build_key(tex):
    path = build_key_mesh()
    colls = [coll_box((KEY_BLADE["x"][0], -KEY_BLADE_COLL_HY, KEY_BLADE["z"][0]),
                      (KEY_BLADE["x"][1], KEY_BLADE_COLL_HY, KEY_BLADE["z"][1])),
             coll_box((KEY_SHOULDER["x"][0], KEY_SHOULDER["y"][0], KEY_SHOULDER["z"][0]),
                      (KEY_SHOULDER["x"][1], KEY_SHOULDER["y"][1], KEY_SHOULDER["z"][1])),
             coll_box((KEY_BOW["x"][0], KEY_BOW["y"][0], KEY_BOW["z"][0]),
                      (KEY_BOW["x"][1], KEY_BOW["y"][1], KEY_BOW["z"][1]))]
    body = (
        '  <link name="key">\n'
        + inertial(KEY_MASS, (0.03, 0.01, 0.09)) + "\n"
        + vis_mesh(path) + "\n" + "\n".join(colls) + "\n  </link>"
    )
    write_urdf("key", body)


def build_lock(tex):
    px, py, pz = PLINTH
    plinth = rounded_box(PLINTH, 0.008, center=(0, 0, pz / 2))
    p_plinth = write_obj("lock_plinth", [shade(plinth, uv_scale=3.0)], texture=tex["walnut"], ks=0.3, ns=50)

    flange = rounded_cylinder(FLANGE_R, pz - 0.001, pz + FLANGE_H, 0.003)
    housing = rounded_cylinder(HOUSING_R, pz + FLANGE_H - 0.001, HOUSING_TOP, 0.006)
    p_housing = write_obj("lock_housing", [shade(flange, cyl_axis=(0, 0), uv_scale=3.0),
                                           shade(housing, cyl_axis=(0, 0), uv_scale=3.0)],
                          texture=tex["steel"], ks=0.8, ns=150)

    # Indicator dots on the housing face: red = locked (plug at 0), green = open (+-90 deg).
    r_dot = 0.0345
    red = rounded_cylinder(0.0036, HOUSING_TOP - 0.001, HOUSING_TOP + 0.0009, 0.0006, 24, (0, r_dot))
    green = [rounded_cylinder(0.0036, HOUSING_TOP - 0.001, HOUSING_TOP + 0.0009, 0.0006, 24, (sx * r_dot, 0))
             for sx in (-1, 1)]
    p_red = write_obj("lock_dot_red", [shade(red)], kd=(0.85, 0.08, 0.06), ks=0.4)
    p_green = write_obj("lock_dot_green", [shade(g) for g in green], kd=(0.10, 0.75, 0.25), ks=0.4)

    # Plug (child link frame: origin on the housing top face).
    plug = rounded_cylinder(PLUG_R, -0.012, PLUG_PROUD, 0.0015)
    p_plug = write_obj("lock_plug", [shade(plug, cyl_axis=(0, 0), uv_scale=5.0)],
                       texture=tex["brass"], ks=0.9, ns=180)
    funnel = box((-FUNNEL_TOP_HX, -FUNNEL_TOP_HY, PLUG_PROUD), (FUNNEL_TOP_HX, FUNNEL_TOP_HY, PLUG_PROUD + 0.0004))
    hole = box((-SLOT_HX, -SLOT_HY, PLUG_PROUD), (SLOT_HX, SLOT_HY, PLUG_PROUD + 0.0007))
    # Engraved index line on the plug face pointing at the red dot (+y) when locked.
    notch = box((-0.0009, FUNNEL_TOP_HY + 0.002, PLUG_PROUD), (0.0009, PLUG_R - 0.003, PLUG_PROUD + 0.0005))
    p_funnel = write_obj("lock_plug_funnel", [shade(funnel)], kd=(0.42, 0.30, 0.10), ks=0.5)
    p_hole = write_obj("lock_plug_hole", [shade(hole), shade(notch)], kd=(0.03, 0.025, 0.02), ks=0.0)

    base_colls = [coll_box((-px / 2, -py / 2, 0), (px / 2, py / 2, pz))]
    lower_top = HOUSING_TOP - PLUG_DEPTH
    base_colls.append(coll_box((-HOUSING_R, -HOUSING_R, pz), (HOUSING_R, HOUSING_R, lower_top)))
    r_in, r_out = RING_INNER, HOUSING_R
    base_colls += [
        coll_box((r_in, -r_out, lower_top), (r_out, r_out, HOUSING_TOP)),
        coll_box((-r_out, -r_out, lower_top), (-r_in, r_out, HOUSING_TOP)),
        coll_box((-r_in, r_in, lower_top), (r_in, r_out, HOUSING_TOP)),
        coll_box((-r_in, -r_out, lower_top), (r_in, -r_in, HOUSING_TOP)),
    ]
    plug_top = PLUG_PROUD
    funnel_bottom = plug_top - FUNNEL_DEPTH
    plug_colls = keyway_collisions(funnel_bottom, KEYWAY_DEPTH - FUNNEL_DEPTH, SLOT_HX, KEYWAY_COLL_HY, PLUG_SQ)
    # Solid rim around the funnel up to the plug face.
    plug_colls += [
        coll_box((FUNNEL_TOP_HX, -PLUG_SQ, funnel_bottom), (PLUG_SQ, PLUG_SQ, plug_top)),
        coll_box((-PLUG_SQ, -PLUG_SQ, funnel_bottom), (-FUNNEL_TOP_HX, PLUG_SQ, plug_top)),
        coll_box((-FUNNEL_TOP_HX, FUNNEL_TOP_HY, funnel_bottom), (FUNNEL_TOP_HX, PLUG_SQ, plug_top)),
        coll_box((-FUNNEL_TOP_HX, -PLUG_SQ, funnel_bottom), (FUNNEL_TOP_HX, -FUNNEL_TOP_HY, plug_top)),
    ]
    plug_colls += funnel_collisions(plug_top, FUNNEL_DEPTH, SLOT_HX, KEYWAY_COLL_HY, FUNNEL_TOP_HX, FUNNEL_TOP_HY)

    body = (
        '  <link name="lock_base">\n'
        + inertial(2.0, (px, py, HOUSING_TOP), (0, 0, HOUSING_TOP / 2)) + "\n"
        + vis_mesh(p_plinth) + "\n" + vis_mesh(p_housing) + "\n"
        + vis_mesh(p_red, (0.85, 0.08, 0.06, 1), "dot_red") + "\n"
        + vis_mesh(p_green, (0.10, 0.75, 0.25, 1), "dot_green") + "\n"
        + "\n".join(base_colls) + "\n  </link>\n"
        '  <link name="plug">\n'
        # COM kept at the link frame: PyBullet reports link/base poses at the COM, so
        # this keeps get_link_pose/get_base_pose identical across simulators.
        + inertial(0.08, (2 * PLUG_R, 2 * PLUG_R, PLUG_DEPTH)) + "\n"
        + vis_mesh(p_plug) + "\n"
        + vis_mesh(p_funnel, (0.42, 0.30, 0.10, 1), "funnel") + "\n"
        + vis_mesh(p_hole, (0.03, 0.025, 0.02, 1), "keyway") + "\n"
        + "\n".join(plug_colls) + "\n  </link>\n"
        '  <joint name="plug_joint" type="revolute">\n'
        f'    <parent link="lock_base"/><child link="plug"/>\n'
        f'    <origin xyz="0 0 {HOUSING_TOP:.5f}" rpy="0 0 0"/><axis xyz="0 0 1"/>\n'
        f'    <limit lower="{-PLUG_LIMIT}" upper="{PLUG_LIMIT}" effort="5" velocity="3"/>\n'
        f'    <dynamics damping="{PLUG_DAMPING}" friction="0.0"/>\n'
        "  </joint>"
    )
    write_urdf("lock", body)


def write_geometry_json():
    geo = {
        "table_top_z": TABLE_TOP_Z,
        "table_size": TABLE_SIZE,
        "table_center_xy": TABLE_CENTER_XY,
        "key_blade": KEY_BLADE, "key_shoulder": KEY_SHOULDER, "key_bow": KEY_BOW,
        "slot_half": [SLOT_HX, SLOT_HY],
        "lock_plug_joint_z": HOUSING_TOP,
        "lock_plug_top_local_z": PLUG_PROUD,
        "lock_keyway_depth": KEYWAY_DEPTH,
        "lock_plug_limit": PLUG_LIMIT,
        "stand_size": STAND,
        "stand_slot_depth": STAND_SLOT_DEPTH,
    }
    with open(os.path.join(ROOT, "geometry.json"), "w") as fh:
        json.dump(geo, fh, indent=2)


def main():
    if not os.path.exists(os.path.join(CACHE, "key.npz")):
        sys.exit("Run tools/extract_robodojo_usdz.py first (needs usd-core).")
    tex = build_textures()
    build_table(tex)
    build_key_stand(tex)
    build_key(tex)
    build_lock(tex)
    write_geometry_json()


if __name__ == "__main__":
    main()
