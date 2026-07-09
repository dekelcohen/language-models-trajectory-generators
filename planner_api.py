"""PlannerAPI: exec-based tool surface for the agentic recovery planner.

Analogous to `api.API` (whose tools drive the low-level subtask agent via exec'd
code blocks), `PlannerAPI` exposes *planner-level* tools that the recovery planner
LLM invokes by emitting ```python blocks:

- execute_subtasks(subtasks) -> dict : run a list of subtasks in order (in a loop
  while they succeed) and get a batch summary (this is "subtasks as a tool").
- plan_completed()                  : the overall user command is now satisfied.
- plan_failed()                     : give up; remaining goal is unreachable.

The planner drives one continuous agentic conversation: it dispatches a decomposed
plan via execute_subtasks, observes the printed batch result, replans on failure,
and always terminates by calling plan_completed() or plan_failed().
"""
import math
import numpy as np

from config import OK, PROGRESS, WARNING, FAIL, ENDC


class PlannerAPI:
    def __init__(self, ctx, execute_task_fn, logger):
        self.ctx = ctx
        self._execute_task = execute_task_fn
        self.logger = logger
        # Control flags read by the agentic loop
        self.plan_completed_flag = False
        self.plan_failed_flag = False
        # History of subtasks the planner ran (each: {prompt, max_attempts, result})
        self.subtask_results = []

    # --- Tools (injected into the planner exec environment) ---------------
    def execute_subtasks(self, subtasks):
        """Run `subtasks` in order, in a loop, continuing WHILE each succeeds.

        `subtasks` is a list of dicts: {"prompt": str, "max_attempts": int}.
        Stops at the first failing subtask. Prints and returns a summary dict:
          {"all_succeeded": bool,
           "completed": [prompt, ...],
           "failed": {"index": int, "prompt": str, "result": <summary dict>} or None,
           "remaining": [subtask, ...]}
        The printed output flows back to the planner LLM's next turn (mirroring how
        subtask tools surface printed output).
        """
        if isinstance(subtasks, dict):
            subtasks = [subtasks]
        completed = []
        failed = None
        subtasks = list(subtasks or [])
        for i, st in enumerate(subtasks):
            prompt = str(st.get("prompt", "")).strip()
            max_attempts = st.get("max_attempts")
            if not prompt:
                self.logger.info(WARNING + f"[planner] skipping subtask {i} with empty prompt" + ENDC)
                continue
            self.logger.info(PROGRESS + f"[planner] execute_subtasks[{i}] (max_attempts={max_attempts}) prompt={prompt!r}" + ENDC)
            result = self._execute_task(self.ctx, prompt, max_attempts=max_attempts)
            summary = result.as_summary_dict()
            self.subtask_results.append({"prompt": prompt, "max_attempts": max_attempts, "result": summary})
            if summary["success"]:
                completed.append(prompt)
                continue
            failed = {"index": i, "prompt": prompt, "result": summary}
            break

        remaining = subtasks[(failed["index"] + 1):] if failed else []
        out = {
            "all_succeeded": failed is None,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
        }
        # Concise print for the planner LLM's next turn
        print(f"execute_subtasks: all_succeeded={out['all_succeeded']} completed={len(completed)}/{len(subtasks)}")
        if failed is not None:
            fr = failed["result"]
            print(f"failed at index {failed['index']}: {failed['prompt']!r}")
            print(f"  success={fr['success']} attempts={fr['attempts']} accepted_without_review={fr['accepted_without_review']}")
            if fr["reviewer_reason"]:
                print(f"  reviewer_reason: {fr['reviewer_reason']}")
            if fr["improvement_steps"]:
                print(f"  improvement_steps: {fr['improvement_steps']}")
            print(f"  remaining_subtasks: {len(remaining)}")
        return out

    def plan_completed(self):
        """Signal the overall user command is satisfied; ends the planner loop."""
        self.plan_completed_flag = True
        self.logger.info(OK + "[planner] plan_completed()" + ENDC)

    def plan_failed(self):
        """Signal the overall goal is unreachable; ends the planner loop."""
        self.plan_failed_flag = True
        self.logger.info(FAIL + "[planner] plan_failed()" + ENDC)

    def detect_object(self, object_or_object_part):
        """Observation passthrough to the low-level API's detector (optional tool)."""
        return self.ctx.api.detect_object(object_or_object_part)


def get_planner_exec_locals(planner_api, logger):
    """Locals injected into the planner's exec environment (mirrors get_exec_locals)."""
    return {
        "execute_subtasks": planner_api.execute_subtasks,
        "plan_completed": planner_api.plan_completed,
        "plan_failed": planner_api.plan_failed,
        "detect_object": planner_api.detect_object,
        "planner": planner_api,
        "math": math,
        "np": np,
        "logger": logger,
    }
