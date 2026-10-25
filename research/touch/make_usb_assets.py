"""Build the USB-insertion assets from my_assets/79-obj.rar (low-poly HP notebook, Blender OBJ).

The source OBJ is Y-up and ~7.9x real size, with absolute texture paths and no USB hole. This
script bakes a real-size, Z-up copy whose left side faces -x (towards the robot), with the base
centred on x/y and its bottom at z=0, and writes the mtl with relative texture paths. The USB port
itself is not in the mesh: the env builds it from collision boxes (see sim_envs/usb_insert.py).

  python research/touch/make_usb_assets.py   (needs the rar extracted to research/touch/assets/laptop_src)
"""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research/touch/assets/laptop_src/obj"
DST = ROOT / "sim_envs/assets/usb_laptop"
REAL_WIDTH = 0.416  # 17" HP notebook, side to side (m)


def remap(v):
    # old (x: front->back, y: up, z: side) -> new (x: side, y: front->back, z: up). Cyclic, so a proper rotation.
    return np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1)


def main():
    lines = (SRC / "Lowpoly_Notebook_2.obj").read_text(encoding="utf-8", errors="ignore").splitlines()
    vi = [i for i, l in enumerate(lines) if l.startswith("v ")]
    ni = [i for i, l in enumerate(lines) if l.startswith("vn ")]
    V = np.array([[float(t) for t in lines[i].split()[1:4]] for i in vi])
    N = np.array([[float(t) for t in lines[i].split()[1:4]] for i in ni])
    # body vertices = those referenced by faces after "usemtl Lowpoly_Body"
    body_idx, cur = set(), None
    for l in lines:
        if l.startswith("usemtl "):
            cur = l.split(maxsplit=1)[1]
        elif l.startswith("f ") and cur and "Body" in cur:
            body_idx.update(int(t.split("/")[0]) - 1 for t in l.split()[1:])
    Vb = V[sorted(body_idx)]
    scale = REAL_WIDTH / np.ptp(Vb[:, 2])
    Vn, Vbn = remap(V) * scale, remap(Vb) * scale
    lo, hi = Vbn.min(0), Vbn.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    Vn += shift
    Nn = remap(N)
    for k, i in enumerate(vi):
        lines[i] = "v {:.6f} {:.6f} {:.6f}".format(*Vn[k])
    for k, i in enumerate(ni):
        lines[i] = "vn {:.4f} {:.4f} {:.4f}".format(*Nn[k])
    lines = [("mtllib laptop.mtl" if l.startswith("mtllib") else l) for l in lines]
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "laptop.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DST / "laptop.mtl").write_text(
        "newmtl Lowpoly_Body\nKa 1 1 1\nKd 1 1 1\nKs 0.3 0.3 0.3\nNs 50\nd 1\nillum 2\nmap_Kd laptop_body.jpg\n\n"
        "newmtl Lowpoly_Screen\nKa 1 1 1\nKd 1 1 1\nKs 0.3 0.3 0.3\nNs 50\nd 1\nillum 2\nmap_Kd laptop_screen.jpg\n",
        encoding="utf-8")
    shutil.copyfile(SRC / "textures/Lowpoly_Laptop_2.jpg", DST / "laptop_body.jpg")
    shutil.copyfile(SRC / "textures/Lowpoly_Laptop_1.jpg", DST / "laptop_screen.jpg")
    size = (hi - lo).tolist()
    # Port centre measured from a straight-on render of the -x side: the first textured USB-A port
    # (26 x 10 px at 2198 px/m = 11.8 x 4.5 mm). The env cuts the collision pocket here.
    meta = {"source": "my_assets/79-obj.rar (3dhaupt.com low-poly HP notebook)", "scale": float(scale),
            "base_size_m": size, "port_face": "-x", "front": "-y", "screen": "+y",
            "usb_port_center_local": [-size[0] / 2, -0.0541, 0.0190]}
    (DST / "laptop_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
