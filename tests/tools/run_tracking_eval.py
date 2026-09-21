"""Run the tracking evaluation harness on the PyBullet grasp scene.

Boots the ``grasp`` scene headlessly (the same boot the tracking integration test uses),
runs :mod:`tests.tracking_eval` for one or more provider combinations, prints a comparison
table and writes each run's JSON payload under ``outputs/tracking_eval/``.

    python tests\\tools\\run_tracking_eval.py
    python tests\\tools\\run_tracking_eval.py --tracker3d depth_fusion triangulate
"""

import argparse
import logging
import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TESTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS)

import tracking_eval as te  # noqa: E402
from test_tracking_pybullet import _boot  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "outputs", "tracking_eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", nargs="+", default=["template"],
                        help="2D tracker provider(s): template | csrt | remote")
    parser.add_argument("--tracker3d", nargs="+", default=["depth_fusion"],
                        help="3D lift provider(s): depth_fusion | triangulate | lapa")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--occlusion", type=int, nargs=2, default=[12, 19],
                        help="[start, end) frame indices of the painted occlusion window")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT, help="directory for the JSON payloads")
    parser.add_argument("--no-frames-trace", action="store_true",
                        help="omit the per-frame trace from the JSON payload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sim, env, robot = _boot()
    scene = te.GraspSceneDriver(sim, env, robot, env.simenv.object_id, n_frames=args.frames,
                                occlusion=tuple(args.occlusion))

    payloads = []
    for provider in args.provider:
        for lift in args.tracker3d:
            payload = te.run_eval(scene, tracker_provider=provider, tracker3d=lift,
                                  cameras=args.cameras, keep_frames=not args.no_frames_trace)
            path = os.path.join(args.out, f"grasp_{provider}_{lift}.json")
            te.write_report(payload, path)
            payloads.append(payload)

    print()
    print(te.render_comparison_table(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
