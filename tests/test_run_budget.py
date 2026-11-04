"""helpers.run_budget: per-block action cap, churn guard, learned attempt time budget."""

import pytest

from helpers.run_budget import BudgetExceeded, RunBudget


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _budget(tmp_path, **kw):
    clock = Clock()
    b = RunBudget("pybullet:door", str(tmp_path / "hist.json"), clock=clock, **kw)
    return b, clock


def test_churn_in_small_box_stops_and_stays_tripped(tmp_path):
    b, _ = _budget(tmp_path, churn_window=10)
    with pytest.raises(BudgetExceeded):
        for i in range(10):
            b.on_robot_action("execute_trajectory", [0.3 + 0.001 * (i % 2), 0.0, 0.5], [0.3, 0, 0.56, 0, 0, 0, 1])
    # LLM code that swallows the error cannot keep going in the same block
    try:
        b.on_robot_action("execute_trajectory", [0.5, 0.0, 0.5])
    except Exception:
        pytest.fail("BudgetExceeded must not be an Exception subclass")
    except BudgetExceeded:
        pass
    b.start_block()
    b.on_robot_action("execute_trajectory", [0.5, 0.0, 0.5])


def test_tilting_in_place_and_real_progress_are_not_churn(tmp_path):
    b, _ = _budget(tmp_path, churn_window=10)
    for i in range(30):    # pouring: hand fixed, orientation changing
        b.on_robot_action("execute_trajectory", [0.3, 0.0, 0.5], [0.3, 0, 0.56, 0.02 * i, 0, 0, 1])
    b.start_block()
    for i in range(30):    # door arc: 2 mm steps
        b.on_robot_action("execute_trajectory", [0.3 + 0.002 * i, 0.0, 0.5], [0, 0, 0, 0, 0, 0, 1])


def test_action_cap_per_block(tmp_path):
    b, _ = _budget(tmp_path, max_motion_calls_per_block=5)
    for _ in range(5):
        b.on_robot_action("close_gripper")
    with pytest.raises(BudgetExceeded, match="ONE code block"):
        b.on_robot_action("close_gripper")


def test_attempt_budget_learned_from_normal_attempts(tmp_path):
    b, clock = _budget(tmp_path, attempt_default_s=1200, attempt_min_s=60, min_history=3)
    assert b.start_attempt() == 1200        # no history yet
    for dur in (100, 120, 110):
        b.start_attempt()
        clock.t += dur
        b.end_attempt(ended_normally=True)
    b.start_attempt()
    clock.t += 5000
    b.end_attempt(ended_normally=False)     # forced ends are not part of the distribution
    budget = b.start_attempt()
    assert 200 < budget < 260               # 2 x p90 of (100, 110, 120)
    assert b.expired() is None
    clock.t += budget + 1
    assert "time budget" in b.expired()
    with pytest.raises(BudgetExceeded, match="time budget"):
        b.on_robot_action("execute_trajectory", [0, 0, 0])
