"""Render annotated tracking videos: what the tracker saw, marked up, frame by frame.

Runs the same ground-truth eval harness as ``run_tracking_eval.py`` but attaches a frame
hook that paints markers on every camera frame and encodes an mp4 per configuration:

* green dot   - a tracked 2D point the provider calls visible
* orange ring - a tracked 2D point the provider calls hidden
* cyan cross  - the fused 3D estimate, reprojected into that camera
* red circle  - ground truth, reprojected into that camera
* magenta line- the 3D error, drawn in pixels
* yellow frame + "OCCLUDED" - the painted occlusion window

    python tests\\tools\\render_tracking_video.py
    python tests\\tools\\render_tracking_video.py --tracker3d depth_fusion triangulate rigid_refine
    python tests\\tools\\render_tracking_video.py --scene synthetic     # no simulator needed

Videos land in ``outputs/tracking_video/`` next to the JSON payload of the same run, so the
numbers in the table and the pixels in the video always come from one execution.
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
from tracking import visualize  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "outputs", "tracking_video")

log = logging.getLogger("render_tracking_video")


class VideoRecorder:
    """Frame hook that accumulates composed overlay frames for one eval run."""

    def __init__(self, scene, obj, cameras, scale=2, extra_lines=()):
        self.scene = scene
        self.obj = obj
        self.cameras = cameras
        self.scale = scale
        self.extra_lines = list(extra_lines)
        self.frames = []

    def __call__(self, frame_idx, views, report, samples):
        sample = samples.get(self.obj)
        state = None if report is None else report.objects.get(self.obj)
        expected = None if sample is None else sample.expected
        lines = list(self.extra_lines)
        if sample is not None and sample.t is not None:
            # Real time: the overlay is the latest *published* estimate, computed on an
            # older frame; "age" is how old - that gap is the lag you see.
            computed_on = None if report is None else report.camera_frame
            lines.append(f"t={sample.t:.2f}s  estimate from frame {computed_on}  "
                         f"age={1000.0 * (sample.age_s or 0.0):.0f} ms")
        image = visualize.compose_frame(
            views, state=state, expected_world=expected, cameras=self.cameras,
            frame_idx=frame_idx, scale=self.scale,
            occluded=bool(sample is not None and sample.occluded),
            extra_lines=lines)
        if image is not None:
            self.frames.append(image)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    te.add_eval_args(parser)
    out = parser.add_argument_group("output")
    out.add_argument("--fps", type=int, default=None,
                     help="playback fps (default: the camera fps in real-time runs, i.e. "
                          "real speed; 4 in lock-step runs)")
    out.add_argument("--scale", type=int, default=2, help="pixel upscale for legibility")
    out.add_argument("--png", action="store_true", help="also dump per-frame PNGs")
    out.add_argument("--tag", default="", help="suffix for output file names")
    out.add_argument("--out", default=DEFAULT_OUT)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not visualize.available():
        log.error("OpenCV is not importable - cannot render overlays")
        return 2

    scene, kwargs_by_provider = te.build_scene(args)
    obj = scene.objects[0]
    payloads, videos = [], []
    timing = te.timing_kwargs(args)
    # Real time: play back at the camera rate, so 1 s of video = 1 s of sim time.
    fps = args.fps or (int(round(timing["camera_fps"])) if timing else 4)
    log.info("[video] timing=%s playback fps=%s", timing or "lockstep", fps)

    for provider in args.provider:
        for lift in args.tracker3d:
            label = f"{provider}+{lift}"
            recorder = VideoRecorder(scene, obj, args.cameras,
                                     scale=args.scale,
                                     extra_lines=[f"{args.scene} | 2D={provider} 3D={lift}"])
            payload = te.run_eval(scene, tracker_provider=provider, tracker3d=lift,
                                  cameras=args.cameras, keep_frames=True,
                                  tracker_kwargs=kwargs_by_provider.get(provider),
                                  frame_hook=recorder, **timing)
            stem = f"{args.scene}_{provider}_{lift}" + (f"_{args.tag}" if args.tag else "")
            te.write_report(payload, os.path.join(args.out, stem + ".json"))
            path = visualize.write_video(recorder.frames,
                                         os.path.join(args.out, stem + ".mp4"), fps=fps)
            if args.png:
                visualize.write_frames(recorder.frames, os.path.join(args.out, stem + "_png"))
            agg = payload.get("aggregate", {})
            log.info("[video] %-28s frames=%d median=%s p95=%s occl=%s point=%s pose=%s -> %s",
                     label, len(recorder.frames), agg.get("median_l2_m"), agg.get("p95_l2_m"),
                     agg.get("err_during_occlusion_m"), agg.get("median_point_err_m"),
                     agg.get("median_pose_err_m"), path)
            payloads.append(payload)
            if path:
                videos.append(path)

    print()
    print(te.render_comparison_table(payloads))
    print()
    for path in videos:
        print("video:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
