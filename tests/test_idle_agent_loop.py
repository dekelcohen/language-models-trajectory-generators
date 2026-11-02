from types import SimpleNamespace

import agent_runner


def test_idle_agent_loop_is_ended_for_review(monkeypatch):
    """An LLM that never moves the robot nor calls task_completed() must not loop forever."""
    task = SimpleNamespace(completed_task=False, failed_task=False,
                           conversation_messages=[{"role": "assistant", "content": "```python\npass\n```"}])
    calls = {"turns": 0, "completed": 0}

    def task_completed():
        calls["completed"] += 1
        task.completed_task = True

    api = SimpleNamespace(trajectory_step=42, task_completed=task_completed)
    ctx = SimpleNamespace(api=api, logger=SimpleNamespace(info=lambda *_: None))
    monkeypatch.setattr(agent_runner, "execute_python_blocks", lambda *a: "")

    def turn(*a):
        calls["turns"] += 1
        assert calls["turns"] < 50, "loop never ended"
        return task.conversation_messages

    monkeypatch.setattr(agent_runner, "continue_task_turn", turn)
    monkeypatch.setattr(agent_runner.config, "max_idle_agent_turns", 4)
    agent_runner.run_task_agent_loop(ctx, task, "p", "", [])
    assert calls["completed"] == 1 and calls["turns"] == 3


def test_endless_moving_attempt_is_ended_for_review(monkeypatch):
    """An LLM that keeps moving the robot (retrying forever) is also ended after N turns."""
    task = SimpleNamespace(completed_task=False, failed_task=False,
                           conversation_messages=[{"role": "assistant", "content": "```python\npass\n```"}])
    calls = {"turns": 0, "completed": 0}

    def task_completed():
        calls["completed"] += 1
        task.completed_task = True

    api = SimpleNamespace(trajectory_step=0, task_completed=task_completed)
    ctx = SimpleNamespace(api=api, logger=SimpleNamespace(info=lambda *_: None))

    def moving_turn(*a):
        api.trajectory_step += 5
        return ""

    monkeypatch.setattr(agent_runner, "execute_python_blocks", moving_turn)

    def turn(*a):
        calls["turns"] += 1
        assert calls["turns"] < 50, "loop never ended"
        return task.conversation_messages

    monkeypatch.setattr(agent_runner, "continue_task_turn", turn)
    monkeypatch.setattr(agent_runner.config, "max_idle_agent_turns", 4)
    monkeypatch.setattr(agent_runner.config, "max_agent_turns_per_attempt", 6)
    agent_runner.run_task_agent_loop(ctx, task, "p", "", [])
    assert calls["completed"] == 1 and calls["turns"] == 5
