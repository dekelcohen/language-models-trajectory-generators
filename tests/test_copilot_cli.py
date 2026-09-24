# -*- coding: utf-8 -*-
"""Unit tests for the Copilot CLI provider (providers/llms/copilot_cli.py).

The `copilot` binary is never launched: `_run` and `_resolve_cli` are patched, so
the tests assert on the argv/prompt the provider *would* have used. What matters
here is session management - the copilot CLI keeps conversation state on its side
(`--session-id` / `--resume`), so the provider must send each turn exactly once
and never let two conversations share a session.

Run:
    python -m unittest tests.test_copilot_cli -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from providers.llms import copilot_cli
from providers.llms.message_media import append_to_messages


class CopilotCliTestBase(unittest.TestCase):
    MODEL = "claude-haiku-4.5"

    def setUp(self):
        copilot_cli._SESSIONS.clear()
        self.calls = []  # one (args, prompt) tuple per copilot invocation

        self._orig_run = copilot_cli._run
        self._orig_resolve = copilot_cli._resolve_cli

        def fake_run(exe, args, prompt, timeout):
            self.calls.append((args, prompt))
            return f"REPLY-{len(self.calls)}"

        copilot_cli._run = fake_run
        copilot_cli._resolve_cli = lambda: "copilot"

    def tearDown(self):
        copilot_cli._run = self._orig_run
        copilot_cli._resolve_cli = self._orig_resolve
        copilot_cli._SESSIONS.clear()

    # -- helpers ---------------------------------------------------------
    def _session_of(self, call_index):
        """Session id used by call #call_index, and whether it was a resume."""
        args = self.calls[call_index][0]
        for arg in args:
            if arg.startswith("--resume="):
                return arg[len("--resume="):], True
        return args[args.index("--session-id") + 1], False

    def _attachments_of(self, call_index):
        args = self.calls[call_index][0]
        return [args[i + 1] for i, a in enumerate(args) if a == "--attachment"]

    def _turn(self, messages, text, key=None):
        """Append a user turn, call the provider, append the reply (as models.py does)."""
        messages.append({"role": "user", "content": text})
        reply = copilot_cli.call_copilot(messages, model=self.MODEL, conversation_key=key)
        messages.append({"role": "assistant", "content": reply})
        return messages


class TestSessionReuse(CopilotCliTestBase):

    def test_first_call_starts_a_new_session(self):
        copilot_cli.call_copilot([{"role": "user", "content": "hi"}], model=self.MODEL)
        sid, resumed = self._session_of(0)
        self.assertFalse(resumed, "a fresh conversation must create a session, not resume one")
        self.assertTrue(sid)

    def test_followup_resumes_and_sends_only_the_new_turn(self):
        msgs = self._turn([], "first question")
        self._turn(msgs, "second question")

        sid0, resumed0 = self._session_of(0)
        sid1, resumed1 = self._session_of(1)
        self.assertFalse(resumed0)
        self.assertTrue(resumed1, "a continued conversation must resume its session")
        self.assertEqual(sid0, sid1)
        # Only the new turn is sent: the CLI still holds the earlier ones.
        self.assertIn("second question", self.calls[1][1])
        self.assertNotIn("first question", self.calls[1][1])

    def test_new_messages_list_starts_a_new_session(self):
        """A brand-new conversation (e.g. agent_runner starting from []) must not
        latch onto an existing session."""
        self._turn([], "first question")
        copilot_cli.call_copilot([{"role": "user", "content": "unrelated"}], model=self.MODEL)

        sid0, _ = self._session_of(0)
        sid1, resumed1 = self._session_of(1)
        self.assertFalse(resumed1)
        self.assertNotEqual(sid0, sid1)

    def test_branching_off_a_shared_prefix_does_not_reuse_the_session(self):
        """Two branches share a prefix; the first one advances the session, so the
        second must get its own instead of reading the first branch's turns."""
        base = self._turn([], "shared question")
        branch_a = list(base)
        branch_b = list(base)

        self._turn(branch_a, "branch A")
        self._turn(branch_b, "branch B")

        sid_a, _ = self._session_of(1)
        sid_b, resumed_b = self._session_of(2)
        self.assertNotEqual(sid_a, sid_b)
        self.assertFalse(resumed_b, "branch B must start a fresh session")
        # Having missed, branch B replays its whole conversation.
        self.assertIn("shared question", self.calls[2][1])
        self.assertIn("branch B", self.calls[2][1])
        self.assertNotIn("branch A", self.calls[2][1])


class TestConversationKeys(CopilotCliTestBase):

    def test_planner_and_task_never_share_a_session(self):
        planner, task = [], []
        self._turn(planner, "plan the task", key="planner")
        self._turn(task, "execute the subtask", key="task-abcd1234")
        # ...and both keep going, interleaved, as they do in agent_runner.
        self._turn(planner, "next subtask?", key="planner")
        self._turn(task, "next step?", key="task-abcd1234")

        planner_sid, _ = self._session_of(0)
        task_sid, _ = self._session_of(1)
        self.assertNotEqual(planner_sid, task_sid)

        planner_sid2, resumed_p = self._session_of(2)
        task_sid2, resumed_t = self._session_of(3)
        self.assertTrue(resumed_p)
        self.assertTrue(resumed_t)
        self.assertEqual(planner_sid, planner_sid2)
        self.assertEqual(task_sid, task_sid2)

    def test_identical_conversations_under_different_keys_are_isolated(self):
        a, b = [], []
        self._turn(a, "same text", key="planner")
        self._turn(b, "same text", key="task-1")
        sid_a, _ = self._session_of(0)
        sid_b, resumed_b = self._session_of(1)
        self.assertFalse(resumed_b)
        self.assertNotEqual(sid_a, sid_b)


class TestAttachments(CopilotCliTestBase):
    IMAGE = os.path.join(os.path.dirname(__file__), "..", "outputs", "bugs", "door_rgb_image_head.png")

    def setUp(self):
        super().setUp()
        if not os.path.exists(self.IMAGE):
            self.skipTest(f"test image not found: {self.IMAGE}")

    def test_image_is_written_from_message_bytes_and_attached_as_a_path(self):
        msgs = append_to_messages("what is this?", [self.IMAGE], [], "user")
        copilot_cli.call_copilot(msgs, model=self.MODEL)

        attachments = self._attachments_of(0)
        self.assertEqual(len(attachments), 1)
        # A real file path (the CLI cannot take base64), holding the same bytes...
        self.assertTrue(os.path.exists(attachments[0]))
        with open(attachments[0], "rb") as f:
            written = f.read()
        with open(self.IMAGE, "rb") as f:
            self.assertEqual(written, f.read())
        # ...but NOT the caller's path, which the pipeline overwrites every step.
        self.assertNotEqual(os.path.abspath(attachments[0]), os.path.abspath(self.IMAGE))
        self.assertIn("[image 1 attached]", self.calls[0][1])

    def test_image_is_not_resent_on_a_followup_turn(self):
        msgs = append_to_messages("what is this?", [self.IMAGE], [], "user")
        reply = copilot_cli.call_copilot(msgs, model=self.MODEL)
        msgs.append({"role": "assistant", "content": reply})
        self._turn(msgs, "and what color is it?")

        self.assertEqual(len(self._attachments_of(0)), 1)
        self.assertEqual(self._attachments_of(1), [],
                         "the resumed session already holds the image; re-attaching wastes tokens")

    def test_video_parts_are_dropped(self):
        """The CLI attaches images/PDFs only (models.model_supports_video reports
        False for copilot-*, so this is only a safety net)."""
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "review this"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }]
        copilot_cli.call_copilot(msgs, model=self.MODEL)
        self.assertEqual(self._attachments_of(0), [])
        self.assertIn("non-attachable media omitted", self.calls[0][1])


class TestCliInvocation(CopilotCliTestBase):

    def test_reasoning_effort_maps_to_effort_flag(self):
        copilot_cli.call_copilot([{"role": "user", "content": "hi"}], model=self.MODEL,
                                 reasoning_effort="high")
        args = self.calls[0][0]
        self.assertIn("--effort", args)
        self.assertEqual(args[args.index("--effort") + 1], "high")

    def test_runs_non_interactively_without_tools(self):
        copilot_cli.call_copilot([{"role": "user", "content": "hi"}], model=self.MODEL)
        args = self.calls[0][0]
        for flag in ("--allow-all-tools", "--available-tools=none", "--no-ask-user", "--silent"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--model") + 1], self.MODEL)

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            copilot_cli.call_copilot([], model=self.MODEL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
