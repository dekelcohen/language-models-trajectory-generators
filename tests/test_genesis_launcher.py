"""Unit tests for Genesis child-process interpreter resolution.

These run under ``vlm_traj`` and must NOT require Genesis to be installed: they exercise
the *resolution* logic with injected environments and temp directories. The one test that
touches the real machine skips itself when the env is absent.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import genesis_launcher as gl


def _make_fake_env(envs_dir, name):
    """Create ``<envs_dir>/<name>/python(.exe)`` and return its path."""
    prefix = os.path.join(envs_dir, name)
    interpreter = gl._interpreter_in(prefix)
    os.makedirs(os.path.dirname(interpreter), exist_ok=True)
    with open(interpreter, "w", encoding="utf-8") as handle:
        handle.write("")
    return interpreter


class TestResolveGenesisPython(unittest.TestCase):
    def test_explicit_genesis_python_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = _make_fake_env(tmp, "some_other_env")
            other = _make_fake_env(tmp, gl.DEFAULT_GENESIS_CONDA_ENV)
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]):
                resolved = gl.resolve_genesis_python({gl.GENESIS_PYTHON_ENV_VAR: explicit})
            self.assertEqual(resolved, explicit)
            self.assertNotEqual(resolved, other)

    def test_explicit_genesis_python_is_quote_and_space_tolerant(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = _make_fake_env(tmp, "quoted_env")
            resolved = gl.resolve_genesis_python({gl.GENESIS_PYTHON_ENV_VAR: f'  "{explicit}"  '})
            self.assertEqual(resolved, explicit)

    def test_explicit_but_missing_path_fails_loudly(self):
        """Must NOT silently fall back to sys.executable (the METAWORLD_PYTHON trap).

        Falling back would launch Genesis under the main app's env, where `import genesis`
        blows up inside the child with a traceback that hides the real cause.
        """
        bogus = os.path.join(tempfile.gettempdir(), "definitely-not-here", "python.exe")
        with self.assertRaises(gl.GenesisInterpreterNotFound) as ctx:
            gl.resolve_genesis_python({gl.GENESIS_PYTHON_ENV_VAR: bogus})
        message = str(ctx.exception)
        self.assertIn(gl.GENESIS_PYTHON_ENV_VAR, message)
        self.assertIn(bogus, message)
        self.assertNotIn(sys.executable, message)

    def test_discovers_default_conda_env_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _make_fake_env(tmp, gl.DEFAULT_GENESIS_CONDA_ENV)
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]):
                self.assertEqual(gl.resolve_genesis_python({}), expected)

    def test_conda_env_name_is_overridable(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _make_fake_env(tmp, "genesis_custom")
            _make_fake_env(tmp, gl.DEFAULT_GENESIS_CONDA_ENV)
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]):
                resolved = gl.resolve_genesis_python({gl.GENESIS_CONDA_ENV_VAR: "genesis_custom"})
            self.assertEqual(resolved, expected)

    def test_first_matching_envs_dir_wins(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            preferred = _make_fake_env(first, gl.DEFAULT_GENESIS_CONDA_ENV)
            _make_fake_env(second, gl.DEFAULT_GENESIS_CONDA_ENV)
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[first, second]):
                self.assertEqual(gl.resolve_genesis_python({}), preferred)

    def test_falls_back_to_current_interpreter_only_if_genesis_importable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]), \
                 mock.patch.object(gl, "_conda_base_via_cli", return_value=None), \
                 mock.patch.object(gl, "_current_interpreter_has_genesis", return_value=True):
                self.assertEqual(gl.resolve_genesis_python({}), sys.executable)

    def test_error_message_is_actionable_when_nothing_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]), \
                 mock.patch.object(gl, "_conda_base_via_cli", return_value=None), \
                 mock.patch.object(gl, "_current_interpreter_has_genesis", return_value=False):
                with self.assertRaises(gl.GenesisInterpreterNotFound) as ctx:
                    gl.resolve_genesis_python({})
        message = str(ctx.exception)
        for expected in (gl.DEFAULT_GENESIS_CONDA_ENV, gl.GENESIS_PYTHON_ENV_VAR,
                         gl.GENESIS_CONDA_ENV_VAR, "conda create", "genesis-world"):
            self.assertIn(expected, message)

    def test_describe_never_raises(self):
        with mock.patch.object(gl, "resolve_genesis_python",
                               side_effect=gl.GenesisInterpreterNotFound("nope\nmore detail")):
            self.assertEqual(gl.describe_genesis_python({}), "<unresolved: nope>")


class TestGenesisEndpoint(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(gl.genesis_endpoint({}), (gl.DEFAULT_GENESIS_HOST, gl.DEFAULT_GENESIS_PORT))

    def test_overrides(self):
        host, port = gl.genesis_endpoint({gl.GENESIS_HOST_ENV_VAR: "0.0.0.0",
                                          gl.GENESIS_PORT_ENV_VAR: "9911"})
        self.assertEqual((host, port), ("0.0.0.0", 9911))

    def test_blank_port_falls_back_to_default(self):
        _, port = gl.genesis_endpoint({gl.GENESIS_PORT_ENV_VAR: "   "})
        self.assertEqual(port, gl.DEFAULT_GENESIS_PORT)

    def test_bad_port_raises(self):
        with self.assertRaises(ValueError):
            gl.genesis_endpoint({gl.GENESIS_PORT_ENV_VAR: "not-a-port"})


class TestGenesisChildEnv(unittest.TestCase):
    def test_repo_root_is_prepended_to_pythonpath(self):
        child = gl.genesis_child_env({"PYTHONPATH": "/already/there"}, repo_root="/repo")
        self.assertEqual(child["PYTHONPATH"].split(os.pathsep)[0], "/repo")
        self.assertIn("/already/there", child["PYTHONPATH"].split(os.pathsep))

    def test_parent_venv_vars_are_scrubbed(self):
        """The child runs a different interpreter; inherited prefixes shadow its site-packages."""
        child = gl.genesis_child_env(
            {"CONDA_PREFIX": "/x/vlm_traj", "VIRTUAL_ENV": "/x/venv",
             "PYTHONHOME": "/x/home", "CONDA_DEFAULT_ENV": "vlm_traj", "KEEP_ME": "1"},
            repo_root="/repo",
        )
        for scrubbed in ("CONDA_PREFIX", "VIRTUAL_ENV", "PYTHONHOME", "CONDA_DEFAULT_ENV"):
            self.assertNotIn(scrubbed, child)
        self.assertEqual(child["KEEP_ME"], "1")

    def test_child_is_unbuffered(self):
        self.assertEqual(gl.genesis_child_env({}, repo_root="/repo")["PYTHONUNBUFFERED"], "1")

    def test_caller_env_is_not_mutated(self):
        source = {"CONDA_PREFIX": "/x", "PYTHONPATH": "/p"}
        gl.genesis_child_env(source, repo_root="/repo")
        self.assertEqual(source, {"CONDA_PREFIX": "/x", "PYTHONPATH": "/p"})

    def test_default_repo_root_is_this_repository(self):
        child = gl.genesis_child_env({})
        root = child["PYTHONPATH"].split(os.pathsep)[0]
        self.assertTrue(os.path.exists(os.path.join(root, "env.py")),
                        f"repo root {root!r} does not look like the lmtg checkout")


class _FakeArgs:
    task = "door"
    robot = "franka"
    mode = "default"
    gui = False


class TestGenesisChildCommand(unittest.TestCase):
    def test_resolved_interpreter_is_argv0(self):
        cmd = gl.genesis_child_command(_FakeArgs(), "127.0.0.1", 8770,
                                       python_exe="/envs/vlm_genesis/python", repo_root="/repo")
        self.assertEqual(cmd[0], "/envs/vlm_genesis/python")

    def test_child_script_and_endpoint_are_passed(self):
        cmd = gl.genesis_child_command(_FakeArgs(), "0.0.0.0", 9999,
                                       python_exe="/py", repo_root="/repo")
        self.assertEqual(cmd[1], os.path.join("/repo", gl.GENESIS_CHILD_SCRIPT))
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "9999")

    def test_task_robot_and_mode_are_forwarded(self):
        cmd = gl.genesis_child_command(_FakeArgs(), "h", 1, python_exe="/py", repo_root="/repo")
        self.assertEqual(cmd[cmd.index("--task") + 1], "door")
        self.assertEqual(cmd[cmd.index("--robot") + 1], "franka")
        self.assertEqual(cmd[cmd.index("--mode") + 1], "default")
        self.assertNotIn("--gui", cmd)

    def test_gui_flag(self):
        args = _FakeArgs()
        args.gui = True
        cmd = gl.genesis_child_command(args, "h", 1, python_exe="/py", repo_root="/repo")
        self.assertIn("--gui", cmd)

    def test_missing_optional_args_are_omitted(self):
        class Bare:
            pass

        cmd = gl.genesis_child_command(Bare(), "h", 1, python_exe="/py", repo_root="/repo")
        for flag in ("--task", "--robot", "--mode", "--gui"):
            self.assertNotIn(flag, cmd)

    def test_launch_uses_the_resolved_interpreter_not_sys_executable(self):
        """The whole point: the child must run under vlm_genesis, never the parent's python."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_python = _make_fake_env(tmp, gl.DEFAULT_GENESIS_CONDA_ENV)
            repo = os.path.join(tmp, "repo")
            script = os.path.join(repo, gl.GENESIS_CHILD_SCRIPT)
            os.makedirs(os.path.dirname(script), exist_ok=True)
            with open(script, "w", encoding="utf-8") as handle:
                handle.write("")

            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]), \
                 mock.patch.object(gl.subprocess, "Popen") as popen:
                popen.return_value = object()
                _proc, host, port, python_exe = gl.launch_genesis_child(
                    _FakeArgs(), logger=None, env={}, repo_root=repo
                )

            self.assertEqual(python_exe, fake_python)
            self.assertEqual((host, port), (gl.DEFAULT_GENESIS_HOST, gl.DEFAULT_GENESIS_PORT))
            cmd = popen.call_args.args[0]
            self.assertEqual(cmd[0], fake_python)
            self.assertNotEqual(os.path.normcase(cmd[0]), os.path.normcase(sys.executable))
            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(child_env["PYTHONPATH"].split(os.pathsep)[0], repo)

    def test_launch_fails_clearly_when_child_script_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_env(tmp, gl.DEFAULT_GENESIS_CONDA_ENV)
            with mock.patch.object(gl, "_conda_env_dirs", return_value=[tmp]):
                with self.assertRaises(gl.GenesisInterpreterNotFound) as ctx:
                    gl.launch_genesis_child(_FakeArgs(), logger=None, env={},
                                            repo_root=os.path.join(tmp, "no-repo"))
        self.assertIn(gl.GENESIS_CHILD_SCRIPT.replace("/", os.sep), str(ctx.exception))


class TestRealMachineDiscovery(unittest.TestCase):
    def test_finds_the_real_genesis_env_if_present(self):
        found = gl.find_conda_env_python(gl.DEFAULT_GENESIS_CONDA_ENV)
        if not found:
            self.skipTest(f"conda env {gl.DEFAULT_GENESIS_CONDA_ENV!r} is not installed on this machine")
        self.assertTrue(os.path.exists(found))
        self.assertNotEqual(os.path.normcase(found), os.path.normcase(sys.executable),
                            "Genesis must not resolve to the main app's interpreter")


if __name__ == "__main__":
    unittest.main()
