"""Run the tracking evaluation harness on a PyBullet scene (grasp / door) or the synthetic one.

Boots the scene headlessly (the same boot the tracking integration tests use),
runs :mod:`tests.tracking_eval` for one or more provider combinations, prints a comparison
table and writes each run's JSON payload under ``outputs/tracking_eval/``.

    python tests\\tools\\run_tracking_eval.py
    python tests\\tools\\run_tracking_eval.py --tracker3d depth_fusion triangulate
    python tests\\tools\\run_tracking_eval.py --scene door --seeding grid --tracker3d pose_tracker
    python tests\\tools\\run_tracking_eval.py --scenario handoff --tracker3d pose_tracker
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

DEFAULT_OUT = os.path.join(ROOT, "outputs", "tracking_eval")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    te.add_eval_args(parser)
    out = parser.add_argument_group("output")
    out.add_argument("--out", default=DEFAULT_OUT, help="directory for the JSON payloads")
    out.add_argument("--tag", default="", help="suffix for output file names")
    out.add_argument("--no-frames-trace", action="store_true",
                     help="omit the per-frame trace from the JSON payload")
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scene, kwargs_by_provider = te.build_scene(args)

    payloads = []
    for provider in args.provider:
        for lift in args.tracker3d:
            payload = te.run_eval(scene, tracker_provider=provider, tracker3d=lift,
                                  cameras=args.cameras, keep_frames=not args.no_frames_trace,
                                  tracker_kwargs=kwargs_by_provider.get(provider),
                                  **te.timing_kwargs(args))
            stem = f"{args.scene}_{provider}_{lift}" + (f"_{args.tag}" if args.tag else "")
            te.write_report(payload, os.path.join(args.out, stem + ".json"))
            payloads.append(payload)

    print()
    print(te.render_comparison_table(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
