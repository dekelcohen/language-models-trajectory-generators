# -*- coding: utf-8 -*-
"""
Unit tests for the lazily-loaded agent skills (skill_registry.py) and their wiring
into the prompt builders.

No LLM, simulator or network is used:
  * Skill trees are written into a temp folder removed at teardown.
  * Prompt assembly is exercised through agent_runner._build_main_prompt with a stub ctx.

Run:
    python -m unittest tests.test_skill_registry -v
"""

import os
import re
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import skill_registry
from skill_registry import SkillSession


def write_skill(root, name, scope_line, description="Does a thing. Use when a thing is needed.",
                body="# Body\n\nDetailed instructions.\n", name_line=None, extra_head=""):
    """Create <root>/<name>/SKILL.md with a hand-built frontmatter head."""
    skill_dir = os.path.join(root, name)
    os.makedirs(skill_dir, exist_ok=True)
    name_line = f"name: {name}" if name_line is None else name_line
    text = (
        "---\n"
        f"{name_line}\n"
        f"description: {description}\n"
        f"{scope_line}\n"
        f"{extra_head}"
        "---\n\n"
        f"{body}"
    )
    path = os.path.join(skill_dir, "SKILL.md")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


class SkillDiscoveryTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="skills_test_")
        skill_registry.clear_cache()

    def tearDown(self):
        skill_registry.clear_cache()
        shutil.rmtree(self.root, ignore_errors=True)

    def _names(self, scope):
        return [m.name for m in skill_registry.discover(self.root, scope, use_cache=False)]

    def test_scope_filters_both_directions(self):
        write_skill(self.root, "planner-only", "scope: planner")
        write_skill(self.root, "subtask-only", "scope: subtask")
        write_skill(self.root, "shared-one", "scope: both")

        self.assertEqual(sorted(self._names("planner")), ["planner-only", "shared-one"])
        self.assertEqual(sorted(self._names("subtask")), ["shared-one", "subtask-only"])

    def test_subtask_description_never_reaches_planner_index(self):
        secret = "SECRETSUBTASKDESCRIPTION about grasping levers."
        write_skill(self.root, "subtask-only", "scope: subtask", description=secret)
        write_skill(self.root, "planner-only", "scope: planner")

        index = skill_registry.render_index(
            skill_registry.discover(self.root, "planner", use_cache=False), "planner")
        self.assertIn("planner-only", index)
        self.assertNotIn("SECRETSUBTASKDESCRIPTION", index)
        self.assertNotIn("subtask-only", index)

    def test_lenient_scope_prefilter_variants(self):
        # Indented, quoted, upper-case, and declared after other keys - all accepted.
        write_skill(self.root, "indented-scope", "   scope: planner")
        write_skill(self.root, "quoted-scope", 'scope: "planner"')
        write_skill(self.root, "quoted-key", "'scope': planner")
        write_skill(self.root, "upper-scope", "scope: PLANNER")
        write_skill(self.root, "trailing-space", "scope: planner   ")

        self.assertEqual(
            sorted(self._names("planner")),
            ["indented-scope", "quoted-key", "quoted-scope", "trailing-space", "upper-scope"],
        )

    def test_lenient_scope_prefilter_crlf_and_no_opening_fence(self):
        crlf_dir = os.path.join(self.root, "crlf-skill")
        os.makedirs(crlf_dir)
        with open(os.path.join(crlf_dir, "SKILL.md"), "w", encoding="utf-8", newline="") as f:
            f.write("---\r\nname: crlf-skill\r\ndescription: A skill.\r\nscope: planner\r\n---\r\n\r\n# Body\r\n")

        no_fence_dir = os.path.join(self.root, "no-fence-skill")
        os.makedirs(no_fence_dir)
        with open(os.path.join(no_fence_dir, "SKILL.md"), "w", encoding="utf-8", newline="") as f:
            f.write("name: no-fence-skill\ndescription: A skill.\nscope: planner\n")

        self.assertEqual(sorted(self._names("planner")), ["crlf-skill", "no-fence-skill"])

    def test_invalid_skills_are_skipped_not_fatal(self):
        write_skill(self.root, "no-scope", "other: value")
        write_skill(self.root, "bad-scope", "scope: robot")
        write_skill(self.root, "no-description", "scope: subtask", description="")
        write_skill(self.root, "too-long", "scope: subtask", description="x" * 2000)
        write_skill(self.root, "good-one", "scope: subtask")

        names = self._names("subtask")
        self.assertIn("good-one", names)
        self.assertIn("too-long", names)  # description truncated, skill kept
        self.assertNotIn("no-scope", names)
        self.assertNotIn("bad-scope", names)
        self.assertNotIn("no-description", names)

        truncated = next(m for m in skill_registry.discover(self.root, "subtask", use_cache=False)
                         if m.name == "too-long")
        self.assertLessEqual(len(truncated.description), skill_registry.MAX_DESCRIPTION_CHARS)

    def test_name_mismatch_falls_back_to_directory(self):
        write_skill(self.root, "real-dir-name", "scope: subtask", name_line="name: other-name")
        metas = skill_registry.discover(self.root, "subtask", use_cache=False)
        self.assertEqual([m.name for m in metas], ["real-dir-name"])

    def test_malformed_yaml_is_recovered_leniently(self):
        # Unquoted colon inside the description is invalid YAML but common in the wild.
        write_skill(self.root, "colon-skill", "scope: subtask",
                    description="Opens things: doors, drawers and cabinets.")
        metas = skill_registry.discover(self.root, "subtask", use_cache=False)
        self.assertEqual([m.name for m in metas], ["colon-skill"])
        self.assertIn("doors", metas[0].description)

    def test_category_subdirectory_is_discovered(self):
        nested = os.path.join(self.root, "manipulation")
        write_skill(nested, "nested-skill", "scope: subtask")
        self.assertIn("nested-skill", self._names("subtask"))

    def test_read_body_strips_frontmatter_and_lists_resources(self):
        write_skill(self.root, "res-skill", "scope: subtask", body="# Title\n\nStep one.\n")
        ref_dir = os.path.join(self.root, "res-skill", "references")
        os.makedirs(ref_dir)
        with open(os.path.join(ref_dir, "extra.md"), "w", encoding="utf-8") as f:
            f.write("extra details")

        meta = skill_registry.discover(self.root, "subtask", use_cache=False)[0]
        body = skill_registry.read_body(meta)
        self.assertIn("<skill_content name=\"res-skill\">", body)
        self.assertIn("Step one.", body)
        self.assertNotIn("scope: subtask", body)
        self.assertIn("res-skill/references/extra.md", body)


class SkillSessionTest(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="skills_sess_")
        skill_registry.clear_cache()
        write_skill(self.root, "door-skill", "scope: subtask", body="# Door\n\n" + "detail. " * 500)
        self.session = SkillSession(self.root, "subtask")

    def tearDown(self):
        skill_registry.clear_cache()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_load_skill_buffers_body_and_dedups(self):
        self.session.load_skill("door-skill")
        pending = self.session.drain_pending()
        self.assertIn("# Door", pending)
        self.assertGreater(len(pending), 2000)  # would be dropped by the stdout print cap
        self.assertEqual(self.session.drain_pending(), "")

        self.session.load_skill("door-skill")
        self.assertEqual(self.session.drain_pending(), "", "already-loaded skill must not be re-injected")
        self.assertIn("door-skill", self.session.loaded_block())

    def test_load_skill_unknown_name_lists_valid_names(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.session.load_skill("nope")
        out = buf.getvalue()
        self.assertIn("unknown skill", out)
        self.assertIn("door-skill", out)
        self.assertEqual(self.session.drain_pending(), "")

    def test_read_file_sandbox_rejects_escapes(self):
        outside = os.path.join(tempfile.gettempdir(), "skills_outside_secret.txt")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("secret")
        self.addCleanup(os.remove, outside)

        for bad in ("../skills_outside_secret.txt", outside, "door-skill", "door-skill/missing.md"):
            with self.assertRaises(ValueError):
                skill_registry.resolve_in_root(self.root, bad)

        # And through the tool: no exception, nothing buffered, error printed instead.
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.session.read_file("../skills_outside_secret.txt")
        self.assertIn("escapes the skills directory", buf.getvalue())
        self.assertEqual(self.session.drain_pending(), "")

    def test_read_file_reads_bundled_resource(self):
        ref_dir = os.path.join(self.root, "door-skill", "references")
        os.makedirs(ref_dir)
        with open(os.path.join(ref_dir, "hinge.md"), "w", encoding="utf-8") as f:
            f.write("hinge maths")

        self.session.read_file("door-skill/references/hinge.md")
        pending = self.session.drain_pending()
        self.assertIn("hinge maths", pending)
        self.assertIn("<skill_file", pending)

    def test_disabled_session_is_inert(self):
        session = SkillSession(None, "subtask", enabled=False)
        self.assertEqual(session.index_text(), "")
        session.load_skill("door-skill")
        self.assertEqual(session.drain_pending(), "")


class PromptWiringTest(unittest.TestCase):
    """The index and loaded bodies must actually reach the assembled prompts."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="skills_prompt_")
        skill_registry.clear_cache()
        write_skill(self.root, "door-skill", "scope: subtask",
                    description="Opens doors. Use when opening a door.")

    def tearDown(self):
        skill_registry.clear_cache()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_main_prompt_contains_index_tools_and_loaded_skills(self):
        import agent_runner

        session = SkillSession(self.root, "subtask")
        session.load_skill("door-skill")

        prompt = agent_runner._build_main_prompt(
            "detect tool", "detect initial", [0, 0, 0], "open the door", "coords", "",
            skills_index=session.index_text(), loaded_skills=session.loaded_block(),
        )
        self.assertIn("AVAILABLE SKILLS", prompt)
        self.assertIn("door-skill", prompt)
        self.assertIn("load_skill(skill_name: str)", prompt)
        self.assertIn("SKILLS LOADED EARLIER IN THIS TASK", prompt)
        self.assertNotIn("[INSERT SKILLS]", prompt)
        self.assertNotIn("[INSERT SKILL TOOLS]", prompt)

    def test_prompts_are_clean_when_skills_disabled(self):
        import agent_runner

        prompt = agent_runner._build_main_prompt(
            "detect tool", "detect initial", [0, 0, 0], "pick up the box", "coords", "")
        self.assertNotIn("AVAILABLE SKILLS", prompt)
        self.assertNotIn("[INSERT SKILLS]", prompt)
        self.assertNotIn("[INSERT SKILL TOOLS]", prompt)
        self.assertNotIn("load_skill", prompt)

    def test_planner_prompt_contains_planner_index_only(self):
        import agent_runner

        write_skill(self.root, "planner-skill", "scope: planner",
                    description="Plans around occluders.")
        skill_registry.clear_cache()

        class Ctx:
            coords_section = "coords"
            ee_pos_for_prompt = [0, 0, 0]

        session = SkillSession(self.root, "planner")
        prompt = agent_runner._build_planner_prompt(Ctx(), "open the door",
                                                    skills_index=session.index_text())
        self.assertIn("planner-skill", prompt)
        self.assertNotIn("door-skill", prompt)
        self.assertNotIn("[INSERT SKILLS]", prompt)
        self.assertNotIn("[INSERT SKILL TOOLS]", prompt)

    def test_subtask_batches_load_skill_with_detect_object(self):
        import agent_runner

        session = SkillSession(self.root, "subtask")
        index = session.index_text()

        first = agent_runner._build_main_prompt(
            "detect tool", "DETECT INITIAL.", [0, 0, 0], "open the door", "coords", "",
            skills_index=index,
        )
        self.assertIn("SAME code block", first)
        self.assertIn("DETECT INITIAL. In that same code block, ALSO call load_skill", first)

        # Retries have no detect_object tool, so there is nothing to batch into.
        retry = agent_runner._build_main_prompt(
            "detect tool", "DETECT INITIAL.", [0, 0, 0], "open the door", "coords", "",
            skills_index=index, detect_object_available=False,
        )
        self.assertNotIn("ALSO call load_skill", retry)

    def test_planner_does_not_batch_load_skill(self):
        import agent_runner

        write_skill(self.root, "planner-skill", "scope: planner",
                    description="Plans around occluders.")
        skill_registry.clear_cache()

        class Ctx:
            coords_section = "coords"
            ee_pos_for_prompt = [0, 0, 0]

        session = SkillSession(self.root, "planner")
        prompt = agent_runner._build_planner_prompt(Ctx(), "open the door",
                                                    skills_index=session.index_text())
        self.assertIn("in its own code block and end the turn", prompt)
        self.assertNotIn("detect_object", prompt)
        self.assertIn("Do NOT call execute_subtasks(...) in that same turn", prompt)

    def test_grasp_in_context_example_still_always_loaded(self):
        from prompts.main_prompt import MAIN_PROMPT, IN_CONTEXT_EXAMPLE

        self.assertIn("[INSERT IN CONTEXT EXAMPLE]", MAIN_PROMPT)
        self.assertTrue(IN_CONTEXT_EXAMPLE.strip(), "the grasp in-context example must stay always-on")


class ShippedSkillsTest(unittest.TestCase):
    """The skills committed to prompts/skills must be valid and correctly scoped."""

    def setUp(self):
        skill_registry.clear_cache()

    def tearDown(self):
        skill_registry.clear_cache()

    def test_shipped_skills_parse(self):
        root = skill_registry.DEFAULT_SKILLS_DIR
        self.assertTrue(os.path.isdir(root), root)
        subtask = {m.name: m for m in skill_registry.discover(root, "subtask", use_cache=False)}
        planner = {m.name: m for m in skill_registry.discover(root, "planner", use_cache=False)}

        self.assertIn("open-close-door-cabinet-drawer", subtask)
        self.assertNotIn("open-close-door-cabinet-drawer", planner)

        body = skill_registry.read_body(subtask["open-close-door-cabinet-drawer"])
        self.assertIn("angle_long", body)
        self.assertIn("Sliding cabinet", body)

    def test_command_history_seed_file_exists_after_move(self):
        from helpers.command_utils import REPO_HISTORY_FILE

        self.assertTrue(os.path.isfile(REPO_HISTORY_FILE), REPO_HISTORY_FILE)
        self.assertIn(os.path.join("prompts", "user_prompts"), REPO_HISTORY_FILE)


class FeedbackPathTest(unittest.TestCase):
    """A long skill body must survive execute_python_blocks, which drops 2000+ char stdout."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="skills_feedback_")
        skill_registry.clear_cache()
        write_skill(self.root, "long-skill", "scope: subtask", body="# Long\n\n" + "word " * 2000)

    def tearDown(self):
        skill_registry.clear_cache()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_long_skill_body_reaches_the_next_turn(self):
        import agent_runner
        from task_state import TaskState

        class Logger:
            def info(self, *_a, **_k):
                pass

        class Ctx:
            pass

        ctx = Ctx()
        ctx.logger = Logger()
        ctx.exec_locals = {}

        task = TaskState(command="open the door", max_attempts=1)
        task.skills = SkillSession(self.root, "subtask")

        response = 'Loading the skill.\n```python\nload_skill("long-skill")\n```'
        feedback = agent_runner.execute_python_blocks(ctx, task, response)

        self.assertIn("# Long", feedback)
        self.assertGreater(len(feedback), 2000)
        self.assertIn("<skill_content", feedback)


if __name__ == "__main__":
    unittest.main()
