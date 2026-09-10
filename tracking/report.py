"""Per-frame tracking output: JSONL log, in-memory buffer, reviewer summary.

Two consumers:
  * offline - ``outputs/tracking/<run_id>/track.jsonl``, one JSON object per tracked
    frame, for post-hoc debugging and for a future "accumulate tracks and review later"
    workflow;
  * online - the in-memory buffer, which the agent pulls over IPC at the end of a
    trajectory (``GET_TRACKING_REPORT``) to decide whether to retry, and whose flagged
    frames are handed to the VLM reviewer.
"""

import json
import os
import time

import config
from tracking.types import STATUS_ABORT, STATUS_RECORD, STATUS_SEVERITY


class TrackingReporter:
    """Collects :class:`tracking.types.TrackFrameReport` objects."""

    def __init__(self, run_id=None, output_dir=None, write_jsonl=True, logger=None,
                 max_frames_in_memory=2000):
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir or config.tracking_output_dir, self.run_id)
        self.write_jsonl = write_jsonl
        self.logger = logger
        self.max_frames_in_memory = int(max_frames_in_memory)
        self.frames = []
        self.flagged = []          # frames whose monitor said record/abort
        self.reseeds = []
        self.worst_status = "ok"
        self.abort_reason = None
        self._fh = None
        if self.write_jsonl:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                self._fh = open(os.path.join(self.output_dir, config.tracking_log_name),
                                "a", encoding="utf-8")
            except OSError as exc:      # tracking must never break a rollout
                self._log(f"[tracking] could not open the JSONL log: {exc}")
                self._fh = None

    # -- ingestion ---------------------------------------------------------
    def add(self, report):
        payload = report.to_dict()
        if len(self.frames) < self.max_frames_in_memory:
            self.frames.append(payload)
        status = report.monitor.status
        if STATUS_SEVERITY.get(status, 0) > STATUS_SEVERITY.get(self.worst_status, 0):
            self.worst_status = status
            if status == STATUS_ABORT:
                self.abort_reason = report.monitor.reason
        if status in (STATUS_RECORD, STATUS_ABORT):
            self.flagged.append({"frame_idx": report.frame_idx,
                                 "trajectory_step": report.trajectory_step,
                                 "status": status,
                                 "reason": report.monitor.reason,
                                 "rgb_paths": dict(report.rgb_paths)})
        for obj in report.objects.values():
            for ev in obj.reseeds:
                self.reseeds.append({"frame_idx": report.frame_idx, **ev.to_dict()})
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(payload) + "\n")
                self._fh.flush()
            except (OSError, TypeError) as exc:
                self._log(f"[tracking] JSONL write failed: {exc}")
                self._fh = None

    # -- output ------------------------------------------------------------
    def summary(self):
        """Compact, IPC- and prompt-friendly digest of the whole rollout."""
        per_object = {}
        for frame in self.frames:
            for name, obj in frame["objects"].items():
                entry = per_object.setdefault(name, {
                    "frames": 0, "frames_lost": 0, "reseeds": 0,
                    "min_gripper_dist": None, "max_gripper_dist": None,
                    "max_disagreement": None, "cams_used": set(),
                })
                entry["frames"] += 1
                if obj["lost"]:
                    entry["frames_lost"] += 1
                entry["reseeds"] += len(obj["reseeds"])
                if obj["disagreement"] is not None:
                    prev = entry["max_disagreement"]
                    entry["max_disagreement"] = obj["disagreement"] if prev is None else max(prev, obj["disagreement"])
                for cam, track in obj["cams"].items():
                    if track["seeded"] and track["world_point"] is not None:
                        entry["cams_used"].add(cam)
                gp = frame["gripper"]["world_pos"]
                op = obj["world_point"]
                if gp is not None and op is not None:
                    dist = sum((a - b) ** 2 for a, b in zip(gp, op)) ** 0.5
                    lo, hi = entry["min_gripper_dist"], entry["max_gripper_dist"]
                    entry["min_gripper_dist"] = dist if lo is None else min(lo, dist)
                    entry["max_gripper_dist"] = dist if hi is None else max(hi, dist)

        for entry in per_object.values():
            entry["cams_used"] = sorted(entry["cams_used"])
            for key in ("min_gripper_dist", "max_gripper_dist"):
                if entry[key] is not None:
                    entry[key] = round(entry[key], 4)

        return {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "num_frames": len(self.frames),
            "status": self.worst_status,
            "abort_reason": self.abort_reason,
            "objects": per_object,
            "num_reseeds": len(self.reseeds),
            "flagged_frames": self.flagged[-20:],
        }

    def describe(self):
        """Human/LLM readable one-paragraph summary, printed back to the agent."""
        s = self.summary()
        lines = [f"Tracking: {s['num_frames']} frames, status={s['status']}, "
                 f"{s['num_reseeds']} camera re-seeds."]
        for name, obj in s["objects"].items():
            lines.append(
                f"  - '{name}': tracked in {', '.join(obj['cams_used']) or 'no camera'}; "
                f"lost in {obj['frames_lost']}/{obj['frames']} frames; "
                f"gripper distance {obj['min_gripper_dist']}..{obj['max_gripper_dist']} m; "
                f"{obj['reseeds']} re-seeds.")
        if s["abort_reason"]:
            lines.append(f"  ABORT: {s['abort_reason']}")
        elif s["flagged_frames"]:
            lines.append(f"  {len(s['flagged_frames'])} frame(s) flagged for review.")
        return "\n".join(lines)

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self.write_jsonl:
            try:
                path = os.path.join(self.output_dir, config.tracking_summary_name)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(self.summary(), fh, indent=2)
            except (OSError, TypeError) as exc:
                self._log(f"[tracking] could not write the summary: {exc}")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
