#!/usr/bin/env python3
"""Run MolmoPoint-8B with a different prompt per row.

molmo_run.py applies one fixed prompt to a whole tile set. Here every
spreadsheet row carries its own pointing query, and the same image appears
under several different prompts, so the unit of work is the (image, prompt)
pair rather than the image.

point_once is imported rather than reimplemented: MolmoPoint emits special
point tokens that only decode correctly via model.extract_image_points with the
preprocessor metadata, and that call resolves coordinates against each image's
own size, so mixed image dimensions need no rescaling here.
"""
import argparse
import json
import os
import time

from molmo_run import load, point_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="allenai/MolmoPoint-8B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    with open(args.jobs) as handle:
        jobs = json.load(handle)
    print(f"{len(jobs)} (image, prompt) pairs", flush=True)

    model, processor = load(args.checkpoint, args.device, args.dtype)

    results, failures = {}, 0
    started = time.time()
    for index, job in enumerate(jobs, start=1):
        key = str(job["id"])
        call_started = time.time()
        try:
            points, text = point_once(model, processor, job["path"],
                                      job["prompt"], args.max_new_tokens)
            results[key] = {"points": points, "raw": text,
                            "latency": time.time() - call_started,
                            "prompt": job["prompt"],
                            "image": os.path.basename(job["path"])}
        except Exception as exc:
            failures += 1
            results[key] = {"points": [], "raw": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "latency": None, "prompt": job["prompt"],
                            "image": os.path.basename(job["path"])}
            print(f"  !! row {key}: {type(exc).__name__}: {exc}", flush=True)

        n = len(results[key]["points"])
        print(f"[{index}/{len(jobs)}] {results[key]['image']:<24} "
              f"{n:>2} points | {job['prompt'][:52]}", flush=True)

        tmp = args.out + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(results, handle, indent=1)
        os.replace(tmp, args.out)

    wall = time.time() - started
    got = sum(1 for r in results.values() if r["points"])
    print(f"\ndone: {len(results)} rows, {got} with points, "
          f"{failures} errors | {wall / 60:.1f} min "
          f"({wall / len(jobs):.2f} s/row)")
    print("DEKEL_DONE", flush=True)


if __name__ == "__main__":
    main()
