"""Bash-like command history for the interactive `Enter a command:` prompt.

History is merged from two sources (duplicates removed, order preserved):
  1. A repo-committed seed file: <repo root>/prompts/user_prompts/vlm_traj_user_commands.txt
  2. A per-user file in the home dir: ~/vlm_traj_user_commands.txt (newly recorded
     commands are persisted here; repo-seeded commands are not duplicated into it).

On Windows the built-in input() does NOT route through pyreadline3, so callers use
`read_command()` which invokes pyreadline3's line editor directly; on Unix the stdlib
readline hook already instruments input().
"""

import os
import sys

from config import WARNING, ENDC

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_HISTORY_FILE = os.path.join(_REPO_ROOT, "prompts", "user_prompts", "vlm_traj_user_commands.txt")
HOME_HISTORY_FILE = os.path.join(os.path.expanduser("~"), "vlm_traj_user_commands.txt")


def _read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()]
    except (FileNotFoundError, OSError):
        return []


def _write_lines(path, lines):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    except OSError:
        pass


def _dedup(seq):
    """Deduplicate preserving order (keeps first occurrence)."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _rebuild_history(readline, items):
    try:
        readline.clear_history()
    except Exception:
        return
    for cmd in items:
        try:
            readline.add_history(cmd)
        except Exception:
            pass


def init_command_history(logger, repo_file=REPO_HISTORY_FILE, home_file=HOME_HISTORY_FILE):
    """Load + merge repo and home history into readline. Returns readline or None."""
    try:
        import readline
    except ImportError:
        logger.info(WARNING + "readline unavailable; command history disabled." + ENDC)
        return None

    merged = _dedup(_read_lines(repo_file) + _read_lines(home_file))
    _rebuild_history(readline, merged)
    try:
        readline.set_history_length(1000)
    except Exception:
        pass
    return readline


def record_command(readline, command, home_file=HOME_HISTORY_FILE, repo_file=REPO_HISTORY_FILE):
    """Persist a newly entered command to the home history file.

    No-op if the command is already in history. The line editor auto-adds the
    accepted line, so we treat the trailing item as the current entry and dedup
    against the rest. Repo-seeded commands are not written back to the home file.
    """
    if readline is None or not command:
        return
    try:
        n = readline.get_current_history_length()
        items = [readline.get_history_item(i) for i in range(1, n + 1)]
    except Exception:
        return

    prior = items[:-1] if (items and items[-1] == command) else items
    if command in prior:
        _rebuild_history(readline, _dedup(prior))  # drop trailing duplicate
        return

    merged = _dedup(prior + [command])
    _rebuild_history(readline, merged)
    repo_cmds = set(_read_lines(repo_file))
    _write_lines(home_file, [c for c in merged if c not in repo_cmds])


def read_command(prompt, readline=None):
    """Read a line with bash-like editing/history (arrow-up recall).

    On Windows built-in input() bypasses pyreadline3, so we call its line editor
    (`readline.rl.readline`) directly. On Unix the stdlib readline hook already
    instruments input(). Falls back to input() when no line editor is available or
    stdin is not a TTY (piped/replay input).
    """
    rl = getattr(readline, "rl", None) if readline is not None else None
    if rl is not None and hasattr(rl, "readline"):
        try:
            if sys.stdin.isatty():
                return rl.readline(prompt)
        except Exception:
            pass
    return input(prompt)
