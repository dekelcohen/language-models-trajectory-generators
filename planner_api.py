"""PlannerAPI: exec-based tool surface for the agentic recovery planner.

Analogous to `api.API` (whose tools drive the low-level subtask agent via exec'd
code blocks), `PlannerAPI` exposes *planner-level* tools that the recovery planner
LLM invokes by emitting ```python blocks:

- execute_subtasks(subtasks) -> dict : run the NEXT (first) subtask from the list,
  then STOP and return so the planner loop can re-perceive + re-plan on the updated
  world state (this is "subtasks as a tool", stepped one subtask at a time).
- plan_completed()                  : the overall user command is now satisfied.
- plan_failed()                     : give up; remaining goal is unreachable.

The planner drives one continuous agentic conversation: it dispatches the next
subtask via execute_subtasks, then the loop re-runs scene perception on the new
state and re-invokes the planner. This lets the planner react to state changes
between subtasks - e.g. after clearing an occluder it may insert a "move the arm
out of the way" subtask before the next manipulation, or replan on failure. It
always terminates by calling plan_completed() or plan_failed().
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
        """Run ONLY the NEXT (first) subtask, then STOP and return control.

        `subtasks` is a list of dicts: {"prompt": str, "max_attempts": int}. Only the
        FIRST subtask is executed; the rest are returned as `remaining`. After this
        returns, the planner loop re-runs scene perception on the (possibly changed)
        world state and re-invokes the planner, so the planner can reevaluate before
        the next subtask - e.g. insert a "move the arm out of the way" subtask once an
        occluder is cleared, or replan on failure. Prints and returns a summary dict:
          {"executed": prompt or None,
           "success": bool,
           "result": <summary dict> or None,
           "remaining": [subtask, ...]}
        The printed output flows back to the planner LLM's next turn (mirroring how
        subtask tools surface printed output).
        """
        if isinstance(subtasks, dict):
            subtasks = [subtasks]
        subtasks = list(subtasks or [])

        # Skip any leading empty-prompt subtasks.
        idx = 0
        while idx < len(subtasks) and not str(subtasks[idx].get("prompt", "")).strip():
            self.logger.info(WARNING + f"[planner] skipping subtask {idx} with empty prompt" + ENDC)
            idx += 1

        if idx >= len(subtasks):
            out = {"executed": None, "success": True, "result": None, "remaining": []}
            print("execute_subtasks: no runnable subtask (all prompts empty)")
            return out

        st = subtasks[idx]
        prompt = str(st.get("prompt", "")).strip()
        max_attempts = st.get("max_attempts")
        remaining = subtasks[idx + 1:]

        self.logger.info(PROGRESS + f"[planner] execute_subtasks[{idx}] (max_attempts={max_attempts}) prompt={prompt!r}" + ENDC)
        result = self._execute_task(self.ctx, prompt, max_attempts=max_attempts)
        summary = result.as_summary_dict()
        self.subtask_results.append({"prompt": prompt, "max_attempts": max_attempts, "result": summary})

        out = {
            "executed": prompt,
            "success": bool(summary["success"]),
            "result": summary,
            "remaining": remaining,
        }
        # Concise print for the planner LLM's next turn
        print(f"execute_subtasks: executed={prompt!r} success={out['success']} remaining_subtasks={len(remaining)}")
        print(f"  attempts={summary['attempts']} accepted_without_review={summary['accepted_without_review']}")
        if not out["success"]:
            if summary["reviewer_reason"]:
                print(f"  reviewer_reason: {summary['reviewer_reason']}")
            if summary["improvement_steps"]:
                print(f"  improvement_steps: {summary['improvement_steps']}")
        print("  NOTE: perception will re-run on the updated scene before your next turn; "
              "reevaluate (occlusions, arm position, new blockers) before dispatching the next subtask.")
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
