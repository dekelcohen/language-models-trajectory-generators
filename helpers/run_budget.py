"""Execution budgets for LLM-written robot code (no LLM cost inside a code block, so these are
the only brakes on a runaway loop).

Three layers, all measured from proprioception / wall clock only:
  * per code block: at most ``max_motion_calls_per_block`` robot actions, and a "churn" guard -
    ``churn_window`` consecutive actions whose measured hand positions all stay inside a
    ``churn_box_m`` cube = the code is busy without getting anywhere (oscillating corrections,
    a loop waiting for a condition the arm cannot meet).
  * per attempt: a wall-clock budget learned from past attempts of the same task (attempts that
    ended normally); before enough history exists, a default. Outliers time out, so a short task
    is stopped after minutes while a long one gets what it usually needs.
  * per command: a fixed wall-clock budget for the whole planner run.

Exceeding a block or time budget raises ``BudgetExceeded`` inside the running code (the LLM
sees it as an error); the agent loop then ends an expired attempt for review and the planner
stops dispatching when the command budget is spent.
"""

import json
import os
import time
from collections import deque

import numpy as np


class BudgetExceeded(BaseException):
    """BaseException (like KeyboardInterrupt): LLM code's ``except Exception`` cannot swallow it
    and keep looping."""


class RunBudget:
    #: commanded-orientation component span (quaternion / Euler units, ~3-6 deg) below which a
    #: window of in-place actions also did not rotate the hand
    CHURN_ROT_SPAN = 0.05

    def __init__(self, task_key="default", history_path=None, *, max_motion_calls_per_block=300,
                 churn_window=40, churn_box_m=0.01, attempt_default_s=1200.0, attempt_min_s=240.0,
                 attempt_max_s=2700.0, attempt_factor=2.0, min_history=5, history_keep=50,
                 command_s=5400.0, clock=time.monotonic):
        self.task_key = str(task_key)
        self.history_path = history_path
        self.max_motion_calls_per_block = int(max_motion_calls_per_block)
        self.churn_window, self.churn_box_m = int(churn_window), float(churn_box_m)
        self.attempt_default_s, self.attempt_min_s = float(attempt_default_s), float(attempt_min_s)
        self.attempt_max_s, self.attempt_factor = float(attempt_max_s), float(attempt_factor)
        self.min_history, self.history_keep = int(min_history), int(history_keep)
        self.command_s = float(command_s)
        self.clock = clock
        self.command_start = self.attempt_start = None
        self.attempt_budget_s = None
        self.start_block()

    # --- lifecycle -------------------------------------------------------------------------
    def start_command(self):
        self.command_start = self.clock()

    def start_attempt(self):
        self.attempt_start = self.clock()
        self.attempt_budget_s = self.learned_attempt_budget()
        self.start_block()
        return self.attempt_budget_s

    def start_block(self):
        self.block_calls = 0
        self.tripped = None
        self.block_positions = deque(maxlen=max(self.churn_window, 1))

    def _trip(self, msg):
        self.tripped = msg
        raise BudgetExceeded(msg)

    def end_attempt(self, ended_normally):
        """Record the attempt duration (only attempts that ended on their own form the
        'reasonable execution time' distribution)."""
        if self.attempt_start is None:
            return None
        dur = self.clock() - self.attempt_start
        self.attempt_start = None
        if ended_normally:
            hist = self._load()
            durs = (hist.get(self.task_key) or []) + [round(dur, 1)]
            hist[self.task_key] = durs[-self.history_keep:]
            self._save(hist)
        return dur

    # --- budgets ---------------------------------------------------------------------------
    def learned_attempt_budget(self):
        durs = self._load().get(self.task_key) or []
        if len(durs) < self.min_history:
            return self.attempt_default_s
        p90 = float(np.percentile(np.asarray(durs, dtype=float), 90))
        return float(min(max(self.attempt_factor * p90, self.attempt_min_s), self.attempt_max_s))

    def expired(self):
        """Reason string if the attempt or command wall-clock budget is spent, else None."""
        now = self.clock()
        if self.command_start is not None and now - self.command_start > self.command_s:
            return (f"the command time budget ({self.command_s / 60:.0f} min) is spent")
        if self.attempt_start is not None and self.attempt_budget_s is not None \
                and now - self.attempt_start > self.attempt_budget_s:
            return (f"this attempt's time budget ({self.attempt_budget_s / 60:.1f} min, from past attempts "
                    f"of this task) is spent")
        return None

    def command_expired(self):
        return self.command_start is not None and self.clock() - self.command_start > self.command_s

    def on_robot_action(self, name, eef_pos=None, target=None):
        """Called after every robot action of LLM code. ``target``: the commanded end pose (its
        components after xyz = orientation; a tilt-in-place loop changes them and is not churn).
        Raises BudgetExceeded."""
        if self.tripped:
            raise BudgetExceeded(self.tripped + " (still in force: end this code block)")
        self.block_calls += 1
        if self.block_calls > self.max_motion_calls_per_block:
            self._trip(
                f"{name}: {self.block_calls} robot actions in ONE code block (limit "
                f"{self.max_motion_calls_per_block}). A loop is not converging - stop it, print what "
                "was measured, and change the plan.")
        reason = self.expired()
        if reason:
            self._trip(f"{name}: {reason}. Stop here; the attempt is ended for review.")
        if eef_pos is None:
            return
        rot = None if target is None else np.asarray(list(target), dtype=float)[3:]
        self.block_positions.append((np.asarray(eef_pos, dtype=float)[:3], rot))
        if len(self.block_positions) >= self.churn_window:
            pts = np.stack([p for p, _ in self.block_positions])
            span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            rots = [r for _, r in self.block_positions if r is not None and r.size]
            rot_span = 0.0
            if len(rots) == len(self.block_positions) and len({r.shape for r in rots}) == 1:
                R = np.stack(rots)
                rot_span = float(np.max(R.max(axis=0) - R.min(axis=0)))
            if span < self.churn_box_m and rot_span < self.CHURN_ROT_SPAN:
                self._trip(
                    f"{name}: the last {self.churn_window} robot actions of this code block kept the "
                    f"hand inside a {span * 1000:.1f} mm box - busy without progress (oscillating "
                    "corrections or an unreachable exit condition). Accept the measured pose or change "
                    "the plan.")

    # --- history ---------------------------------------------------------------------------
    def _load(self):
        if not self.history_path or not os.path.exists(self.history_path):
            return {}
        try:
            with open(self.history_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, hist):
        if not self.history_path:
            return
        try:
            os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=1)
        except Exception:
            pass


def from_config(config, task_key):
    """Build a RunBudget from config.py values."""
    g = lambda k, d: getattr(config, k, d)  # noqa: E731
    return RunBudget(
        task_key=task_key,
        history_path=g("run_budget_history_path", os.path.join("outputs", "run_budget_history.json")),
        max_motion_calls_per_block=g("max_motion_calls_per_block", 300),
        churn_window=g("churn_window_calls", 40),
        churn_box_m=g("churn_box_m", 0.01),
        attempt_default_s=g("attempt_budget_default_s", 1200.0),
        attempt_min_s=g("attempt_budget_min_s", 240.0),
        attempt_max_s=g("attempt_budget_max_s", 2700.0),
        attempt_factor=g("attempt_budget_factor", 2.0),
        command_s=g("command_budget_s", 5400.0),
    )
