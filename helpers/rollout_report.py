"""Package a main.py rollout into a small, browsable folder.

    python -m helpers.rollout_report <run_dir> --name door_hidden_terra_success --out outputs/2026-11-02 \
        [--notes "root cause / fix text"]

Creates ``<out>/<name>/`` with:
  - ``videos/``: the per-attempt head/wrist clips and full-run videos;
  - ``00_start.png``: first head frame, annotated with the command;
  - ``NN_s<k>_a<j>_<SUCCESS|FAIL|ACCEPTED>.png``: head + wrist at the end of each attempt,
    annotated with the reviewer verdict and reason;
  - ``99_finish_<RESULT>.png``: last head + wrist frame with the overall result;
  - ``summary.md``: args, per-subtask/attempt timeline, verdicts, reasons, measured evidence,
    sim world-state (EVALUATION ONLY - never shown to the agent) and the optional notes.
Everything is parsed from ``vlm_traj.log`` and the run's ``trajectory/`` frames; nothing reruns.
"""
import argparse
import os
import re
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = r"^(\d\d/\d\d \d\d:\d\d) \| \w+\s+\| "
RE_ARGS = re.compile(TS + r"Args: (.*)$")
RE_CMD = re.compile(TS + r"Enter a command: (.*)$")
RE_SUBTASK = re.compile(TS + r"\[planner\] execute_subtasks\[\d+\] \(max_attempts=(\d+)\) prompt=(.*)$")
RE_VIDEO = re.compile(TS + r"\[Success\] Saved .*[\\/]rgb_image_attempt_(\d+)_(\d+)\.mp4")
RE_REVIEW = re.compile(TS + r"Review: success=(True|False)")
RE_REASON = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"')
RE_ACCEPT = re.compile(TS + r"task_completed final attempt (\d+)/(\d+)")
RE_OVERRULE = re.compile(TS + r"Review: success verdict OVERRULED by (.*)$")
RE_WORLD = re.compile(TS + r"\[world-state\] (attempt \d+ end \(\w+\)|subtask end|subtask start): (.*)$")
RE_EVIDENCE = re.compile(TS + r"(Review: (displacement|force feedback|gripper evidence|pull proprio)[^\n]*|"
                         r"\[pull_proprio\] outcome=\S+|Touch insertion finished: \S+|"
                         r"Touch pull finished: \S+)")
RE_END = re.compile(TS + r"(Planner: overall command completed\.|Planner: stopped after .*|"
                    r"Planner: overall command failed.*|.*plan_failed\(.*)$")


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(frames, title, body, color, out_path, width=512):
    """Frames side by side (scaled to ``width`` each) + a caption banner."""
    ims = [Image.open(f).convert("RGB") for f in frames if f and os.path.exists(f)]
    if not ims:
        return None
    ims = [im.resize((width, int(im.height * width / im.width))) for im in ims]
    w = sum(im.width for im in ims)
    h = max(im.height for im in ims)
    f_title, f_body = _font(22), _font(16)
    lines = []
    for para in str(body or "").split("\n"):
        lines += textwrap.wrap(para, width=max(40, w // 9)) or [""]
    lines = lines[:14]
    banner = 40 + 22 * len(lines)
    canvas = Image.new("RGB", (w, h + banner), "white")
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.width
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, h, w, h + 34], fill=color)
    d.text((8, h + 5), title, fill="white", font=f_title)
    for i, ln in enumerate(lines):
        d.text((8, h + 38 + 22 * i), ln, fill="black", font=f_body)
    canvas.save(out_path)
    return out_path


def parse_log(path):
    run = {"args": "", "command": "", "subtasks": [], "end": None, "evidence": []}
    cur_sub = None
    cur_att = None
    last_reason = ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = ANSI.sub("", line.rstrip("\n"))
            m = RE_REASON.search(line)
            if m and not re.match(TS, line):
                last_reason = m.group(1).replace('\\"', '"')
                continue
            if not re.match(TS, line):
                continue
            if (m := RE_ARGS.match(line)):
                run["args"] = m.group(2)
            elif (m := RE_CMD.match(line)) and not run["command"]:
                run["command"] = m.group(2)
            elif (m := RE_SUBTASK.match(line)):
                cur_sub = {"prompt": m.group(3).strip("'\""), "max_attempts": int(m.group(2)),
                           "attempts": [], "world_start": None, "world_end": None}
                run["subtasks"].append(cur_sub)
                cur_att = None
            elif (m := RE_VIDEO.match(line)):
                if cur_sub is None:
                    cur_sub = {"prompt": run["command"], "max_attempts": 0, "attempts": [],
                               "world_start": None, "world_end": None}
                    run["subtasks"].append(cur_sub)
                cur_att = {"range": (int(m.group(2)), int(m.group(3))), "verdict": None, "reason": "",
                           "evidence": [], "world": None, "time": m.group(1)}
                cur_sub["attempts"].append(cur_att)
                last_reason = ""
            elif (m := RE_REVIEW.match(line)) and cur_att is not None:
                cur_att["verdict"] = "SUCCESS" if m.group(2) == "True" else "FAIL"
                cur_att["reason"] = last_reason
            elif (m := RE_OVERRULE.match(line)) and cur_att is not None:
                cur_att["evidence"].append("OVERRULED by " + m.group(2)[:300])
            elif (m := RE_ACCEPT.match(line)) and cur_att is not None:
                cur_att["verdict"] = "ACCEPTED_UNREVIEWED"
            elif (m := RE_EVIDENCE.match(line)) and cur_att is not None:
                cur_att["evidence"].append(m.group(2)[:300])
            elif (m := RE_WORLD.match(line)):
                kind, state = m.group(2), m.group(3)
                if cur_sub is None:
                    continue
                if kind == "subtask start":
                    cur_sub["world_start"] = state
                elif kind == "subtask end":
                    cur_sub["world_end"] = state
                    if cur_att is not None and cur_att["world"] is None:
                        cur_att["world"] = state
                elif cur_att is not None:
                    cur_att["world"] = state
            elif (m := RE_END.match(line)):
                run["end"] = m.group(2)
    return run


def overall_result(run):
    end = run.get("end") or ""
    if "completed" in end:
        return "SUCCESS"
    if "stopped" in end or "failed" in end:
        return "FAIL"
    return "INCOMPLETE"


COLORS = {"SUCCESS": (20, 140, 40), "FAIL": (190, 30, 30), "ACCEPTED_UNREVIEWED": (200, 120, 0),
          None: (90, 90, 90), "INCOMPLETE": (90, 90, 90)}


def _encode_unfinished_tail(traj, vid_dir):
    """Clips are written only when an attempt ends, so a killed/timed-out run has no video of its last
    attempt. Encode the trajectory frames after the last clip into `<cam>_image_attempt_<s>_<e>_unfinished.mp4`."""
    if not os.path.isdir(traj):
        return
    for cam in ("rgb", "wrist"):
        idx = [int(m.group(1)) for f in os.listdir(traj) if (m := re.match(cam + r"_image_(\d+)\.png$", f))]
        if not idx:
            continue
        ends = [int(m.group(1)) for f in os.listdir(vid_dir)
                if (m := re.match(cam + r"_image_attempt_\d+_(\d+)\.mp4$", f))]
        start = max(ends) + 1 if ends else 0
        if max(idx) - start < 5:
            continue
        try:
            from debug.dbg_utils import create_video_from_images
            create_video_from_images(folder_path=traj, output_video_folder_path=vid_dir, base_name=f"{cam}_image",
                                     start_idx=start, end_idx=float("inf"), fps=15,
                                     output_filename=f"{cam}_image_attempt_{start}_{{end}}_unfinished.mp4")
        except Exception as e:  # report packaging must not fail on video encoding
            print(f"[rollout_report] could not encode unfinished {cam} clip: {e}")


def package(run_dir, name, out_root, notes=""):
    log = os.path.join(run_dir, "vlm_traj.log")
    run = parse_log(log)
    dst = os.path.join(out_root, name)
    os.makedirs(os.path.join(dst, "videos"), exist_ok=True)
    vids = os.path.join(run_dir, "videos")
    if os.path.isdir(vids):
        for f in sorted(os.listdir(vids)):
            if f.endswith(".mp4"):
                shutil.copy2(os.path.join(vids, f), os.path.join(dst, "videos", f))
    traj = os.path.join(run_dir, "trajectory")
    _encode_unfinished_tail(traj, os.path.join(dst, "videos"))

    def frame(kind, i):
        p = os.path.join(traj, f"{kind}_image_{i}.png")
        return p if os.path.exists(p) else None

    last_idx = -1
    if os.path.isdir(traj):
        idx = [int(m.group(1)) for f in os.listdir(traj) if (m := re.match(r"rgb_image_(\d+)\.png$", f))]
        last_idx = max(idx) if idx else -1
    result = overall_result(run)
    images = []
    p = annotate([frame("rgb", 0), frame("wrist", 0)], f"START  |  {name}",
                 f"Command: {run['command']}", (40, 70, 140), os.path.join(dst, "00_start.png"))
    if p:
        images.append(os.path.basename(p))
    n = 1
    for si, sub in enumerate(run["subtasks"], 1):
        for ai, att in enumerate(sub["attempts"], 1):
            v = att["verdict"] or "NO_VERDICT"
            end = min(att["range"][1], last_idx) if last_idx >= 0 else att["range"][1]
            title = f"subtask {si} attempt {ai}: {v}"
            body = f"Subtask: {sub['prompt'][:220]}\nReviewer: {att['reason'][:600] or '(no VLM review)'}"
            out = os.path.join(dst, f"{n:02d}_s{si}_a{ai}_{v}.png")
            if annotate([frame("rgb", end), frame("wrist", end)], title, body, COLORS.get(att["verdict"]), out):
                images.append(os.path.basename(out))
            n += 1
    if last_idx >= 0:
        p = annotate([frame("rgb", last_idx), frame("wrist", last_idx)], f"FINISH: {result}",
                     f"Planner end: {run.get('end') or '(not reached)'}\n{notes[:500]}",
                     COLORS.get(result, (90, 90, 90)), os.path.join(dst, f"99_finish_{result}.png"))
        if p:
            images.append(os.path.basename(p))

    md = [f"# Rollout `{name}` - {result}", "",
          f"- Source: `{run_dir}`", f"- Command: `{run['command']}`",
          f"- Planner end: {run.get('end') or '(not reached)'}", ""]
    if run["args"]:
        md += ["<details><summary>Args</summary>", "", "```", run["args"], "```", "</details>", ""]
    if notes:
        md += ["## Analysis (root cause / fix)", "", notes, ""]
    md += ["## Timeline", ""]
    for si, sub in enumerate(run["subtasks"], 1):
        md += [f"### Subtask {si} (max_attempts={sub['max_attempts']})", "", f"> {sub['prompt'][:600]}", ""]
        for ai, att in enumerate(sub["attempts"], 1):
            md += [f"- **Attempt {ai}: {att['verdict'] or 'NO_VERDICT'}** (frames {att['range'][0]}-"
                   f"{att['range'][1]}, {att['time']})"]
            if att["reason"]:
                md += [f"  - Reviewer: {att['reason'][:900]}"]
            for e in att["evidence"][:12]:
                md += [f"  - Evidence: {e}"]
            if att["world"]:
                md += [f"  - Sim world-state (EVALUATION ONLY, not seen by the agent): `{att['world'][:400]}`"]
        md += [""]
    md += ["## Files", "", *[f"- `{i}`" for i in images], "- `videos/` (per-attempt head/wrist clips + full run)", ""]
    with open(os.path.join(dst, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return dst, result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    dst, result = package(a.run_dir, a.name, a.out, a.notes)
    print(f"{result}: {dst}")


if __name__ == "__main__":
    main()
