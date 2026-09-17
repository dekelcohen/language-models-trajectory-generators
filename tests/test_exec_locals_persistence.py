# -*- coding: utf-8 -*-
"""
Unit tests for the LLM exec namespace (agent_runner.execute_python_blocks /
_run_planner_code_blocks).

Contract under test - "persists across turns, never across rollouts":
  * names the LLM defines survive from one assistant response to the next WITHIN the
    same attempt, with ordinary Python module semantics (live closures, working `del`);
  * the namespace is wiped when the attempt changes (retry / rollout) and when a new
    subtask starts, so no stale pose can silently drive the next attempt;
  * injected tool handles always win over anything the LLM bound earlier.

No LLM, simulator or network is used: `ctx` is a stub and the ```python blocks are
plain strings.

Run:
    python -m unittest tests.test_exec_locals_persistence -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import agent_runner
from task_state import TaskState


class _StubLogger:
    def info(self, *_a, **_kw):
        pass

    warning = error = debug = info


class _StubCtx:
    """Minimal AgentContext stand-in: execute_python_blocks only reads .logger/.exec_locals."""

    def __init__(self, exec_locals=None):
        self.logger = _StubLogger()
        self.exec_locals = {"MARKER": "injected", "helper": lambda: "real-helper"}
        if exec_locals:
            self.exec_locals.update(exec_locals)


def block(code):
    """Wrap raw python source the way an assistant response carries it."""
    return "```python\n" + code + "\n```"


class ExecEnvPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.ctx = _StubCtx()
        self.task = TaskState(command="unit test", max_attempts=3)

    def run_turn(self, code, task=None):
        task = self.task if task is None else task
        return agent_runner.execute_python_blocks(self.ctx, task, block(code))

    # --- persistence within one attempt ---------------------------------
    def test_variable_survives_to_next_turn(self):
        self.run_turn("gx = 1.5")
        feedback = self.run_turn("print('gx=', gx)")
        self.assertIn("gx= 1.5", feedback)

    def test_function_defined_earlier_sees_current_values(self):
        """A carried function must read the LIVE namespace, not a turn-1 snapshot."""
        self.run_turn("z = 10\ndef pose():\n    return z")
        feedback = self.run_turn("z = 99\nprint('pose=', pose())")
        self.assertIn("pose= 99", feedback)

    def test_del_removes_the_name_for_good(self):
        self.run_turn("z = 10")
        self.run_turn("del z")
        feedback = self.run_turn("print(z)")
        self.assertIn("NameError", feedback)

    def test_partial_assignments_before_an_exception_are_kept(self):
        feedback = self.run_turn("a = 1\nraise ValueError('boom')")
        self.assertIn("ValueError", feedback)
        self.assertIn("a= 1", self.run_turn("print('a=', a)"))

    # --- isolation: no leak across attempts / subtasks -------------------
    def test_namespace_is_wiped_on_a_new_attempt(self):
        self.run_turn("gx = 1.5")
        self.task.attempt_number += 1  # what api.task_completed() does before a retry
        feedback = self.run_turn("print(gx)")
        self.assertIn("NameError", feedback)
        self.assertIn("gx", feedback)

    def test_namespace_is_wiped_even_when_a_function_was_defined(self):
        self.run_turn("z = 10\ndef pose():\n    return z")
        self.task.attempt_number += 1
        self.assertIn("NameError", self.run_turn("pose()"))

    def test_fresh_task_state_starts_empty(self):
        self.run_turn("gx = 1.5")
        other = TaskState(command="next subtask", max_attempts=3)
        self.assertIn("NameError", self.run_turn("print(gx)", task=other))

    def test_attempt_marker_follows_the_attempt_number(self):
        self.run_turn("gx = 1.5")
        self.assertEqual(self.task.exec_env_attempt, 0)
        self.task.attempt_number = 2
        self.run_turn("gy = 2.5")
        self.assertEqual(self.task.exec_env_attempt, 2)
        self.assertNotIn("gx", self.task.exec_env)
        self.assertIn("gy", self.task.exec_env)

    # --- injected handles always win ------------------------------------
    def test_rebinding_an_injected_name_lasts_one_response_only(self):
        self.run_turn("MARKER = 'llm-value'\nprint('during=', MARKER)")
        self.assertIn("during= llm-value", self.run_turn("MARKER = 'llm-value'\nprint('during=', MARKER)"))
        self.assertIn("after= injected", self.run_turn("print('after=', MARKER)"))

    def test_late_injected_helper_beats_an_earlier_llm_guess(self):
        """A skill loaded later must not be shadowed by a name the LLM invented before."""
        self.run_turn("side_grasp_pose = 'llm-guess'")
        self.ctx.exec_locals["side_grasp_pose"] = "real-skill-helper"  # e.g. load_skill()
        self.assertIn("real-skill-helper", self.run_turn("print(side_grasp_pose)"))

    # --- unchanged behaviour --------------------------------------------
    def test_no_python_block_returns_the_nudge(self):
        feedback = agent_runner.execute_python_blocks(self.ctx, self.task, "just prose")
        self.assertEqual(feedback, agent_runner.NO_TOOL_CALL_PROMPT)

    def test_blocks_of_one_response_still_share_a_namespace(self):
        feedback = agent_runner.execute_python_blocks(
            self.ctx, self.task, block("q = 7") + "\ntext\n" + block("print('q=', q)")
        )
        self.assertIn("q= 7", feedback)


class PlannerExecEnvTest(unittest.TestCase):
    """Planner scope: one namespace per run_plan() call, isolated between runs."""

    def _turn(self, code, carried):
        messages = [{"role": "assistant", "content": block(code)}]
        return agent_runner._run_planner_code_blocks(messages, {"MARKER": "injected"}, None, carried)

    def test_persists_within_one_planner_run(self):
        carried = {}
        self._turn("plan_step = 3", carried)
        self.assertIn("plan_step= 3", self._turn("print('plan_step=', plan_step)", carried))

    def test_separate_runs_are_isolated(self):
        first = {}
        self._turn("plan_step = 3", first)
        self.assertIn("NameError", self._turn("print(plan_step)", {}))

    def test_none_carried_keeps_per_response_scope(self):
        self._turn("plan_step = 3", None)
        self.assertIn("NameError", self._turn("print(plan_step)", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
