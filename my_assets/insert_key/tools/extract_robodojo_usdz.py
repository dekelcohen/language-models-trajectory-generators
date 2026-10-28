"""Stage 1 of the insert_key asset build: RoboDojo USDZ -> plain numpy meshes + textures.

RoboDojo ships its objects only as Isaac Sim USDZ. This script downloads the two
``insert_key`` objects from the RoboDojo HuggingFace dataset (Apache-2.0) and dumps
their visual meshes (triangulated, with UVs when present) to ``.npz`` plus the
texture files, so stage 2 (``build_assets.py``) needs no USD library.

Needs ``usd-core`` (Pixar's USD bindings). Run it once, in any env that has it::

    conda activate env_paperqa   # pip install usd-core numpy
    python my_assets/insert_key/tools/extract_robodojo_usdz.py

Output: ``cache/robodojo/extracted/`` (gitignored).
"""

import argparse
import json
import os
import ssl
import urllib.request
import zipfile

import numpy as np

HF_BASE = ("https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/main/"
           "Assets/Object/RoboDojo")
ASSETS = {
    # name: (path inside the dataset, visual mesh prim)
    "key": ("Clutter/key/00000", "/root/visual/model"),
    "key_slot": ("Geometry/key_slot/00000", "/KeySlot/visual/mesh"),
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "cache", "robodojo")


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # Corporate proxies sometimes break certificate revocation checks.
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:
            f.write(r.read())


def triangulate(counts, indices):
    """Fan-triangulate a USD polygon mesh. Returns (faces, corner index per triangle corner)."""
    faces, corners = [], []
    offset = 0
    for c in counts:
        poly = indices[offset:offset + c]
        for i in range(1, c - 1):
            faces.append([poly[0], poly[i], poly[i + 1]])
            corners.append([offset, offset + i, offset + i + 1])
        offset += c
    return np.asarray(faces, dtype=np.int64), np.asarray(corners, dtype=np.int64)


def extract_mesh(usdz_path, prim_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usdz_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
    xf = np.array(UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()), dtype=np.float64)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    pts = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3] * UsdGeom.GetStageMetersPerUnit(stage)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    faces, corners = triangulate(counts, indices)
    out = {"vertices": pts, "faces": faces}

    uv_pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("uv")
    if uv_pv and uv_pv.HasValue():
        uv = np.array(uv_pv.ComputeFlattened(), dtype=np.float64)
        interp = uv_pv.GetInterpolation()
        if interp == UsdGeom.Tokens.faceVarying:
            out["corner_uv"] = uv[corners]          # (F, 3, 2)
        elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            out["corner_uv"] = uv[faces]
        print(f"  uv: {interp}, {len(uv)} values")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "extracted"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, (ds_path, prim) in ASSETS.items():
        usdz = os.path.join(CACHE_DIR, f"{name}.usdz")
        os.makedirs(CACHE_DIR, exist_ok=True)
        download(f"{HF_BASE}/{ds_path}/object.usdz", usdz)
        download(f"{HF_BASE}/{ds_path}/metadata.json", os.path.join(CACHE_DIR, f"{name}_meta.json"))

        print(f"{name}: extracting {prim}")
        data = extract_mesh(usdz, prim)
        np.savez_compressed(os.path.join(args.out, f"{name}.npz"), **data)
        print(f"  {len(data['vertices'])} vertices, {len(data['faces'])} triangles")

        # Textures travel inside the usdz (a plain zip).
        tex = []
        with zipfile.ZipFile(usdz) as z:
            for info in z.infolist():
                if info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    dest = os.path.join(args.out, f"{name}__{os.path.basename(info.filename)}")
                    with open(dest, "wb") as f:
                        f.write(z.read(info))
                    tex.append(os.path.basename(dest))
        with open(os.path.join(args.out, f"{name}_textures.json"), "w") as f:
            json.dump(tex, f, indent=1)
        print(f"  textures: {tex}")


if __name__ == "__main__":
    main()
