"""Resolve the Python interpreter that runs the Genesis child process.

Genesis cannot share an interpreter with the main app: ``genesis-world`` pulls in
``Pillow>11``, ``numpy>=2``, ``mujoco``, ``numba``, ``pyrender`` and friends, which collide
with the pinned ``requirements.txt`` (``Pillow==10.1.0``, ``numpy==1.26.2``). It therefore
lives in its own conda env (``vlm_genesis`` by default) and is launched as a subprocess,
mirroring how ``agent_runner._setup_metaworld_ws`` launches the Metaworld server via
``METAWORLD_PYTHON``.

Configuration, highest precedence first:

``GENESIS_PYTHON``
    Absolute path to the interpreter. Wins over everything.
``GENESIS_CONDA_ENV``
    Name of the conda env to auto-discover (default ``vlm_genesis``).

Unlike ``METAWORLD_PYTHON``, an explicitly-configured interpreter is **never** silently
replaced by ``sys.executable``: doing so would launch Genesis under the main app's env,
where ``import genesis`` fails with a confusing traceback from inside the child. We fail
fast with an actionable message instead.
"""

import os
import subprocess
import sys

GENESIS_PYTHON_ENV_VAR = "GENESIS_PYTHON"
GENESIS_CONDA_ENV_VAR = "GENESIS_CONDA_ENV"
DEFAULT_GENESIS_CONDA_ENV = "vlm_genesis"

GENESIS_HOST_ENV_VAR = "GENESIS_HOST"
GENESIS_PORT_ENV_VAR = "GENESIS_PORT"
DEFAULT_GENESIS_HOST = "127.0.0.1"
DEFAULT_GENESIS_PORT = 8770


class GenesisInterpreterNotFound(RuntimeError):
    """Raised when no usable Genesis interpreter can be located."""


def _interpreter_in(prefix):
    """Path to the interpreter inside an env rooted at ``prefix``."""
    if os.name == "nt":
        return os.path.join(prefix, "python.exe")
    return os.path.join(prefix, "bin", "python")


def _conda_env_dirs():
    """Directories that may hold named conda envs, most likely first, de-duplicated."""
    candidates = []

    # Running inside `<root>/envs/<name>` -> siblings live in `<root>/envs`.
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        parent = os.path.dirname(os.path.normpath(prefix))
        if os.path.basename(parent).lower() == "envs":
            candidates.append(parent)

    # Running from a conda *root* (base env) -> envs live in `<root>/envs`.
    for root in (os.environ.get("CONDA_ROOT"), os.environ.get("MAMBA_ROOT_PREFIX"), sys.prefix):
        if root:
            candidates.append(os.path.join(root, "envs"))

    for raw in (os.environ.get("CONDA_ENVS_PATH") or os.environ.get("CONDA_ENVS_DIRS") or "").split(os.pathsep):
        if raw.strip():
            candidates.append(raw.strip())

    seen = set()
    unique = []
    for path in candidates:
        key = os.path.normcase(os.path.normpath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _conda_base_via_cli():
    """Last resort: ask conda itself where its base is. Slow, so tried only at the end."""
    executable = os.environ.get("CONDA_EXE")
    if not executable or not os.path.exists(executable):
        return None
    try:
        out = subprocess.run(
            [executable, "info", "--base"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    base = (out.stdout or "").strip()
    return base or None


def find_conda_env_python(env_name):
    """Return the interpreter of conda env ``env_name``, or ``None`` if not found."""
    for envs_dir in _conda_env_dirs():
        candidate = _interpreter_in(os.path.join(envs_dir, env_name))
        if os.path.exists(candidate):
            return candidate

    base = _conda_base_via_cli()
    if base:
        candidate = _interpreter_in(os.path.join(base, "envs", env_name))
        if os.path.exists(candidate):
            return candidate
    return None


def _current_interpreter_has_genesis():
    try:
        import genesis  # noqa: F401
    except Exception:
        return False
    return True


def resolve_genesis_python(env=None):
    """Return the interpreter that should run the Genesis child process.

    Order: ``GENESIS_PYTHON`` -> conda env ``GENESIS_CONDA_ENV`` (default
    ``vlm_genesis``) -> the current interpreter, but only if it can actually import
    ``genesis``. Raises :class:`GenesisInterpreterNotFound` otherwise.
    """
    env = os.environ if env is None else env

    configured = (env.get(GENESIS_PYTHON_ENV_VAR) or "").strip().strip('"')
    if configured:
        if not os.path.exists(configured):
            # Plain, not !r: repr() double-escapes backslashes and makes Windows paths
            # unreadable in the very message that is meant to help the user fix the path.
            raise GenesisInterpreterNotFound(
                f"{GENESIS_PYTHON_ENV_VAR} is set to '{configured}' but that file does not exist. "
                f"Point it at the python.exe of your Genesis conda env, or unset it to "
                f"auto-discover the '{DEFAULT_GENESIS_CONDA_ENV}' env."
            )
        return configured

    env_name = (env.get(GENESIS_CONDA_ENV_VAR) or DEFAULT_GENESIS_CONDA_ENV).strip()
    found = find_conda_env_python(env_name)
    if found:
        return found

    if _current_interpreter_has_genesis():
        return sys.executable

    searched = "\n  ".join(os.path.join(d, env_name) for d in _conda_env_dirs()) or "  (no conda env dirs detected)"
    raise GenesisInterpreterNotFound(
        f"Could not find a Python interpreter with Genesis installed.\n"
        f"Looked for conda env '{env_name}' in:\n  {searched}\n\n"
        f"Fix it either way:\n"
        f"  1. Create the env:   conda create -n {env_name} python=3.11 -y\n"
        f"                       pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        f"                       pip install -e <path-to>/genesis-world\n"
        f"  2. Or point at an existing one:  set {GENESIS_PYTHON_ENV_VAR}=<...>\\envs\\{env_name}\\python.exe\n"
        f"  3. Or rename the env to look for: set {GENESIS_CONDA_ENV_VAR}=<name>"
    )


def genesis_endpoint(env=None):
    """Return the ``(host, port)`` the Genesis child should serve on."""
    env = os.environ if env is None else env
    host = (env.get(GENESIS_HOST_ENV_VAR) or DEFAULT_GENESIS_HOST).strip()
    raw_port = (env.get(GENESIS_PORT_ENV_VAR) or "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_GENESIS_PORT
    except ValueError:
        raise ValueError(
            f"{GENESIS_PORT_ENV_VAR}={raw_port!r} is not an integer port number."
        )
    return host, port


def genesis_child_env(env=None, repo_root=None):
    """Environment for the child: this repo on PYTHONPATH, parent's venv scrubbed.

    The child runs a *different* interpreter, so any inherited ``VIRTUAL_ENV`` /
    ``CONDA_PREFIX`` / ``PYTHONHOME`` from the parent would point at the wrong
    site-packages and can shadow the Genesis install.
    """
    env = os.environ if env is None else env
    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    child = dict(env)
    for key in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONSTARTUP"):
        child.pop(key, None)

    existing = child.get("PYTHONPATH", "")
    parts = [repo_root] + [q for q in existing.split(os.pathsep) if q.strip()]
    child["PYTHONPATH"] = os.pathsep.join(parts)
    # Unbuffered so the child's crash tracebacks reach us before the pipe closes.
    child["PYTHONUNBUFFERED"] = "1"
    # ``pybullet_data`` is not installed in the Genesis env, but a few assets
    # (plane.urdf, robotiq_2f_85) still live there. Hand the child the path we
    # resolved on the parent side so it can load them as plain files.
    if ASSET_ROOT_ENV_VAR not in child:
        asset_root = _pybullet_data_path()
        if asset_root:
            child[ASSET_ROOT_ENV_VAR] = asset_root
    return child


ASSET_ROOT_ENV_VAR = "LMTG_ASSET_ROOT"


def _pybullet_data_path():
    """``pybullet_data.getDataPath()`` or ``None`` if PyBullet isn't importable."""
    try:
        import pybullet_data
    except Exception:
        return None
    try:
        return pybullet_data.getDataPath()
    except Exception:
        return None


def describe_genesis_python(env=None):
    """Human-readable resolution result for startup logs. Never raises."""
    try:
        return resolve_genesis_python(env)
    except GenesisInterpreterNotFound as exc:
        return f"<unresolved: {exc.args[0].splitlines()[0]}>"


# --- Child process launch ------------------------------------------------

GENESIS_CHILD_SCRIPT = os.path.join("sim_envs", "genesis", "genesis_env.py")


def genesis_child_command(args, host, port, python_exe, repo_root=None):
    """Build the argv for the Genesis child process.

    Kept as a pure function so the launch contract is unit-testable without Genesis
    installed, and so there is exactly one place that decides how the child is invoked.
    """
    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, GENESIS_CHILD_SCRIPT)
    cmd = [python_exe, script, "--host", str(host), "--port", str(port)]
    for flag, value in (("--task", getattr(args, "task", None)),
                        ("--robot", getattr(args, "robot", None)),
                        ("--mode", getattr(args, "mode", None))):
        if value:
            cmd += [flag, str(value)]
    if getattr(args, "gui", False):
        cmd.append("--gui")
    return cmd


def launch_genesis_child(args, logger=None, env=None, repo_root=None):
    """Resolve the interpreter, log it, and start the Genesis child process.

    Returns ``(process, host, port, python_exe)``.
    """
    env = os.environ if env is None else env
    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    python_exe = resolve_genesis_python(env)
    host, port = genesis_endpoint(env)
    cmd = genesis_child_command(args, host, port, python_exe, repo_root)

    script = cmd[1]
    if not os.path.exists(script):
        raise GenesisInterpreterNotFound(
            f"Genesis interpreter resolved to '{python_exe}', but the child entry point "
            f"'{script}' does not exist yet."
        )

    if logger is not None:
        # The interpreter is the single most common source of confusing Genesis failures,
        # so state it explicitly at startup rather than leaving it implicit.
        logger.info(f"[Genesis] interpreter : {python_exe}")
        logger.info(f"[Genesis] endpoint    : {host}:{port}")
        logger.info(f"[Genesis] command     : {' '.join(cmd)}")

    process = subprocess.Popen(cmd, cwd=repo_root, env=genesis_child_env(env, repo_root))
    return process, host, port, python_exe


if __name__ == "__main__":
    print(f"{GENESIS_PYTHON_ENV_VAR}      = {os.environ.get(GENESIS_PYTHON_ENV_VAR)!r}")
    print(f"{GENESIS_CONDA_ENV_VAR}   = {os.environ.get(GENESIS_CONDA_ENV_VAR)!r} (default {DEFAULT_GENESIS_CONDA_ENV!r})")
    print(f"conda env dirs        = {_conda_env_dirs()}")
    try:
        print(f"resolved interpreter  = {resolve_genesis_python()}")
        print(f"endpoint              = {genesis_endpoint()}")
    except GenesisInterpreterNotFound as exc:
        print(f"\nUNRESOLVED:\n{exc}")
        sys.exit(1)
