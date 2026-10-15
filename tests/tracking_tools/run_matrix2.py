"""Run the tracking regression matrix through drift_probe.py.

usage: python run_matrix2.py OUT_NAME [ROOT] [-- run keys...]   env CFG="key=val ..." for config overrides
"""
import os, subprocess, sys, concurrent.futures as cf
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_OUT = os.path.join(_REPO, "outputs", "tracking_probes")
S = os.path.dirname(os.path.abspath(__file__))
GW = "--wrist-standoff 0.30 --wrist-height 0.30 --wrist-azimuth 60"
DW = "--wrist-standoff 0.30 --wrist-height 0.15 --wrist-azimuth 60"
RUNS = {
    "D4 door headocc DEFAULTwrist": "--scene door --occlusion 15 25 --occluder-cam head --tracker3d pose_tracker",
    "D5 door alloccl DEFAULTwrist": "--scene door --occlusion 15 25 --occluder-cam all --tracker3d pose_tracker",
    "D5v door alloccl vgrip": "--scene door --occlusion 15 25 --occluder-cam all --tracker3d pose_tracker --virtual-grip",
    "G5 grasp alloccl": "--scene grasp --occlusion 12 19 --occluder-cam all --tracker3d pose_tracker",
    "G5v grasp alloccl vgrip": "--scene grasp --occlusion 12 19 --occluder-cam all --tracker3d pose_tracker --virtual-grip",
    "G6 grasp alloccl rot3": "--scene grasp --occlusion 12 19 --occluder-cam all --rot-step 3 --tracker3d pose_tracker",
    "D1 door headocc aim": f"--scene door --occlusion 15 25 --occluder-cam head {DW} --tracker3d pose_tracker",
    "D3 door headocc aim KLT": f"--scene door --occlusion 15 25 --occluder-cam head {DW} --provider klt --tracker3d pose_tracker",
    "G1 grasp headocc aim": f"--scene grasp --occlusion 12 19 --occluder-cam head {GW} --tracker3d pose_tracker",
    "G4 grasp headocc aim KLT": f"--scene grasp --occlusion 12 19 --occluder-cam head {GW} --provider klt --tracker3d pose_tracker",
    "D0 door noocc aim": f"--scene door --occlusion {DW} --tracker3d pose_tracker",
    "G0 grasp noocc aim": f"--scene grasp --occlusion {GW} --tracker3d pose_tracker",
}
out_name = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else \
    _REPO
sel = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
runs = {k: v for k, v in RUNS.items() if not sel or k.split()[0] in sel}
env = dict(os.environ, TRK_ROOT=root)


def go(item):
    k, a = item
    a += " " + os.environ.get("EXTRA", "")
    if os.environ.get("CFG"):
        a += " --cfg " + os.environ["CFG"]
    r = subprocess.run([sys.executable, os.path.join(S, "drift_probe.py"), *a.split()],
                       capture_output=True, text=True, cwd=root, env=env)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.startswith(("pre", "occl", "post", "aggregate", "recover", "Traceback", "      ")) or "Error" in l]
    return f"## {k} [{os.environ.get('CFG', '')}]\n" + "\n".join(lines)


with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", 4))) as ex:
    res = list(ex.map(go, runs.items()))
out = "\n".join(res)
print(out)
os.makedirs(_OUT, exist_ok=True) or open(os.path.join(_OUT, out_name), "w").write(out + "\n")
