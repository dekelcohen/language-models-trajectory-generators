"""Final attempt of a planner subtask is still VLM-reviewed (fresh verdict for the planner).

Run with::

    python -m pytest tests/test_final_attempt_review.py -q
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api as api_mod  # noqa: E402
from task_state import TaskState  # noqa: E402


class _Conn:
    def send(self, msg):
        pass

    def recv(self):
        return ["done"]


def _make_api(overall_command, review_provider="vlm:some-model"):
    a = object.__new__(api_mod.API)
    a.args = SimpleNamespace(attempts=2, replay_log=None, review_provider=review_provider)
    a.logger = SimpleNamespace(info=lambda *_: None)
    a.main_connection = _Conn()
    a.trajectory_step = 10
    a.task = TaskState(command="open the door", max_attempts=2, start_trajectory_step=0)
    a.task.attempt_number = 1  # task_completed() bumps it to the final attempt (2)
    a.task.overall_command = overall_command
    a.task.review_reason = "stale reason from attempt 1"
    return a


class FinalAttemptReviewTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(api_mod, "create_trajectory_videos", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def test_planner_subtask_final_attempt_gets_fresh_review(self):
        a = _make_api("open the door")

        def review():
            a.task.review_reason = "door is open"
            a.task.review_succeeded = True
            a.task.completed_task = True
        a.run_vlm_review = review
        a.task_completed()
        self.assertTrue(a.task.completed_task)
        self.assertFalse(a.task.failed_task)
        self.assertTrue(a.task.review_succeeded)
        self.assertEqual(a.task.review_reason, "door is open")

    def test_failed_final_review_still_ends_the_loop(self):
        a = _make_api("open the door")

        def review():
            a.task.review_reason = "door still closed"
            a.task.failed_task = True
        a.run_vlm_review = review
        a.task_completed()
        self.assertTrue(a.task.completed_task)
        self.assertFalse(a.task.failed_task)
        self.assertFalse(a.task.review_succeeded)
        self.assertEqual(a.task.review_reason, "door still closed")

    def test_without_planner_final_attempt_is_accepted_unreviewed(self):
        a = _make_api(None)
        a.run_vlm_review = mock.Mock()
        a._pull_proprio_review_evidence = lambda *_: (None, "")
        a.task_completed()
        a.run_vlm_review.assert_not_called()
        self.assertTrue(a.task.accepted_without_review)


if __name__ == "__main__":
    unittest.main()
