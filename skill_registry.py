"""Agent Skills registry: filesystem-backed, lazily-loaded prompt know-how.

Implements the `SKILL.md` convention (agentskills.io / Claude Code) adapted to this
repo's exec-based tool layer, with three disclosure levels:

  1. INDEX     name + description of every skill matching the calling agent's scope.
               Rendered into the system prompt on every turn (cheap, ~50-100 tokens each).
  2. BODY      the full SKILL.md markdown, pulled in only when the LLM calls
               load_skill("<name>") - the text then enters the conversation exactly as if
               it had been concatenated into the prompt.
  3. RESOURCES bundled files (scripts/, references/, assets/) read on demand via
               read_file("<name>/references/x.md"), sandboxed to the skills root.

Layout (default root: ./prompts/skills):

    prompts/skills/<skill-name>/SKILL.md        # required: frontmatter + markdown body
    prompts/skills/<skill-name>/references/*.md # optional level-3 resources
    prompts/skills/<category>/<skill-name>/SKILL.md   # optional one category level

Frontmatter (YAML between --- fences):

    name        required, must match the directory name, [a-z0-9-], <= 64 chars
    description required, <= 1024 chars: what it does AND when to use it
    scope       required, one of: planner | subtask | both

`scope` is what keeps the two agents apart: the planner LLM must never see subtask skill
descriptions and vice versa. Discovery therefore runs a cheap, lenient regex over the
frontmatter head FIRST and skips non-matching files without parsing YAML or reading their
descriptions at all.
"""
import os
import re
import glob

import yaml

from config import OK, PROGRESS, WARNING, FAIL, ENDC

# Default discovery root, relative to the repo root (this file's directory).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SKILLS_DIR = os.path.join(_REPO_ROOT, "prompts", "skills")

SKILL_FILENAME = "SKILL.md"
VALID_SCOPES = ("planner", "subtask", "both")

# Only the frontmatter head is read for discovery; a SKILL.md whose head is larger than
# this is malformed (missing closing fence) and is skipped.
MAX_HEAD_BYTES = 8192
MAX_DESCRIPTION_CHARS = 1024
MAX_NAME_CHARS = 64
# Cap for level-2 bodies and level-3 resource reads (chars) so one file cannot blow up the
# context window; the tool reports the truncation to the LLM.
MAX_BODY_CHARS = 60000

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Lenient scope prefilter: tolerates indentation, surrounding whitespace, quoted keys and
# values, any case, CRLF line endings, and `scope` appearing anywhere in the frontmatter
# head rather than only as the first key.
_SCOPE_RE = re.compile(
    r"""^[ \t]*['"]?scope['"]?[ \t]*:[ \t]*['"]?(planner|subtask|both)['"]?[ \t]*\r?$""",
    re.IGNORECASE | re.MULTILINE,
)

_FENCE_RE = re.compile(r"^---[ \t]*\r?$", re.MULTILINE)

# Directory names inside a skill folder that hold level-3 resources.
RESOURCE_DIRS = ("scripts", "references", "assets", "prompts")

# discover() results cached per (root, scope); skills are static for a process run.
_discover_cache = {}


class SkillMeta:
    """One discovered skill: its level-1 index entry plus where to find the rest."""

    def __init__(self, name, description, scope, path, metadata=None):
        self.name = name
        self.description = description
        self.scope = scope
        self.path = path                       # absolute path of SKILL.md
        self.dir = os.path.dirname(path)       # skill directory (level-3 resource root)
        self.metadata = metadata or {}

    def matches_scope(self, scope):
        return self.scope == "both" or self.scope == scope

    def __repr__(self):
        return f"SkillMeta(name={self.name!r}, scope={self.scope!r}, path={self.path!r})"


def _log(logger, msg):
    if logger is not None:
        logger.info(msg)


def _read_head(path):
    """Return the frontmatter head text, or None when the file has no usable head.

    Only the first MAX_HEAD_BYTES are read - discovery must stay cheap even when the body
    is long. A leading `---` fence is expected but tolerated when missing (in which case
    the whole head chunk up to the first fence, or all of it, is treated as frontmatter).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            chunk = f.read(MAX_HEAD_BYTES)
    except Exception:
        return None
    if not chunk:
        return None

    fences = list(_FENCE_RE.finditer(chunk))
    if fences and fences[0].start() <= len(chunk) - len(chunk.lstrip()):
        # Normal case: head is between the first and second `---` fence.
        if len(fences) >= 2:
            return chunk[fences[0].end():fences[1].start()]
        return chunk[fences[0].end():]
    # No opening fence (lenient): treat everything up to the first fence as the head.
    return chunk[:fences[0].start()] if fences else chunk


_KEY_VALUE_RE = re.compile(
    r"""^[ \t]*['"]?(?P<key>name|description|scope)['"]?[ \t]*:[ \t]*(?P<value>.*?)[ \t]*\r?$""",
    re.IGNORECASE | re.MULTILINE,
)


def _regex_fallback(head):
    """Last-resort extraction of name/description/scope when the head is not valid YAML.

    Covers real-world breakage the YAML parser rejects outright - an indented top-level key,
    stray tabs, unbalanced quotes - so a skill is never dropped over frontmatter cosmetics.
    Block scalars (`description: >`) are reassembled from the following indented lines.
    """
    lines = head.splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = _KEY_VALUE_RE.match(line)
        if not m:
            continue
        key = m.group("key").lower()
        value = m.group("value").strip().strip("'\"")
        if value in (">", "|", ">-", "|-", ">+", "|+", ""):
            # Block scalar: gather the following more-indented lines.
            indent = len(line) - len(line.lstrip())
            parts = []
            for cont in lines[i + 1:]:
                if not cont.strip():
                    continue
                if len(cont) - len(cont.lstrip()) <= indent:
                    break
                parts.append(cont.strip())
            value = " ".join(parts)
        out.setdefault(key, value)
    return out or None


def _parse_head(head):
    """yaml.safe_load a frontmatter head, retrying leniently on malformed YAML.

    Third-party skills often contain an unquoted colon inside `description:`, which is not
    valid YAML. Rather than dropping the skill, retry with such values quoted, then fall
    back to a plain regex extraction.
    """
    try:
        data = yaml.safe_load(head)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fixed_lines = []
    for line in head.splitlines():
        m = re.match(r"^([ \t]*)([A-Za-z_][\w-]*)[ \t]*:[ \t]*(\S.*)$", line)
        if m and not m.group(3).startswith(("'", '"', "|", ">", "[", "{", "#")):
            value = m.group(3).strip().replace("'", "''")
            fixed_lines.append(f"{m.group(1)}{m.group(2)}: '{value}'")
        else:
            fixed_lines.append(line)
    try:
        data = yaml.safe_load("\n".join(fixed_lines))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _regex_fallback(head)


def _validate(data, path, logger):
    """Turn parsed frontmatter into a SkillMeta, or None (with a warning) if invalid."""
    rel = os.path.basename(os.path.dirname(path)) + "/" + SKILL_FILENAME

    scope = str(data.get("scope", "")).strip().lower()
    if scope not in VALID_SCOPES:
        _log(logger, WARNING + f"[skills] {rel}: scope must be one of {VALID_SCOPES}, got {scope!r} - skipped" + ENDC)
        return None

    name = str(data.get("name", "")).strip()
    dir_name = os.path.basename(os.path.dirname(path))
    if not name:
        name = dir_name
    if len(name) > MAX_NAME_CHARS or not _NAME_RE.match(name):
        _log(logger, WARNING + f"[skills] {rel}: invalid name {name!r} (lowercase, digits and single hyphens, <= {MAX_NAME_CHARS} chars) - skipped" + ENDC)
        return None
    if name != dir_name:
        _log(logger, WARNING + f"[skills] {rel}: name {name!r} does not match directory {dir_name!r} - using the directory name" + ENDC)
        name = dir_name

    description = str(data.get("description", "") or "").strip()
    if not description:
        _log(logger, WARNING + f"[skills] {rel}: missing description - skipped" + ENDC)
        return None
    if len(description) > MAX_DESCRIPTION_CHARS:
        _log(logger, WARNING + f"[skills] {rel}: description is {len(description)} chars (max {MAX_DESCRIPTION_CHARS}) - truncated" + ENDC)
        description = description[:MAX_DESCRIPTION_CHARS].rstrip()
    description = " ".join(description.split())

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return SkillMeta(name=name, description=description, scope=scope, path=path, metadata=metadata)


def _skill_files(root):
    """SKILL.md paths: one level (root/<skill>/) plus one category level (root/<cat>/<skill>/)."""
    patterns = (
        os.path.join(root, "*", SKILL_FILENAME),
        os.path.join(root, "*", "*", SKILL_FILENAME),
    )
    seen = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            ap = os.path.abspath(path)
            if ap not in seen:
                seen.append(ap)
    return seen


def discover(root, scope, logger=None, use_cache=True):
    """Return the SkillMeta list for `scope` ('planner' or 'subtask') under `root`.

    Skills whose scope does not match are rejected by a regex over the frontmatter head
    before any YAML parsing, so their descriptions never reach the caller (and therefore
    never reach the other agent's prompt).
    """
    if not root:
        return []
    root = os.path.abspath(root)
    key = (root, scope)
    if use_cache and key in _discover_cache:
        return _discover_cache[key]

    if not os.path.isdir(root):
        _log(logger, WARNING + f"[skills] skills dir not found: {root}" + ENDC)
        _discover_cache[key] = []
        return []

    metas = []
    by_name = {}
    for path in _skill_files(root):
        head = _read_head(path)
        if head is None:
            continue
        m = _SCOPE_RE.search(head)
        if not m:
            # Either the wrong scope or no scope at all. Only warn for the latter, and only
            # once (when scanning the 'subtask' scope) to avoid duplicate noise.
            if not re.search(r"^[ \t]*['\"]?scope['\"]?[ \t]*:", head, re.IGNORECASE | re.MULTILINE) and scope == "subtask":
                _log(logger, WARNING + f"[skills] {path}: missing mandatory 'scope' frontmatter key - skipped" + ENDC)
            continue
        found_scope = m.group(1).lower()
        if found_scope != "both" and found_scope != scope:
            continue

        data = _parse_head(head)
        if not data:
            _log(logger, WARNING + f"[skills] {path}: unparseable YAML frontmatter - skipped" + ENDC)
            continue
        meta = _validate(data, path, logger)
        if meta is None:
            continue
        if meta.name in by_name:
            _log(logger, WARNING + f"[skills] duplicate skill name {meta.name!r}: keeping {by_name[meta.name].path}, ignoring {meta.path}" + ENDC)
            continue
        by_name[meta.name] = meta
        metas.append(meta)

    _discover_cache[key] = metas
    return metas


def clear_cache():
    """Drop the discovery cache (tests, or after editing skills in a live session)."""
    _discover_cache.clear()


def render_index(metas, scope):
    """Render the level-1 catalog block, or "" when no skill matches the scope."""
    if not metas:
        return ""
    lines = ["<available_skills>"]
    for meta in metas:
        lines.append("  <skill>")
        lines.append(f"    <name>{meta.name}</name>")
        lines.append(f"    <description>{meta.description}</description>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def list_resources(meta):
    """Relative paths (skill-name/...) of the skill's bundled level-3 resource files."""
    out = []
    for sub in RESOURCE_DIRS:
        sub_dir = os.path.join(meta.dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(sub_dir):
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                out.append(f"{meta.name}/{os.path.relpath(full, meta.dir)}".replace(os.sep, "/"))
    return out


def strip_frontmatter(text):
    """Return the markdown body: everything after the closing frontmatter fence."""
    fences = list(_FENCE_RE.finditer(text))
    if fences and text[:fences[0].start()].strip() == "" and len(fences) >= 2:
        return text[fences[1].end():].lstrip("\r\n")
    return text


def read_body(meta):
    """Level-2 load: the SKILL.md body wrapped so it is unmistakable in the transcript."""
    with open(meta.path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    body = strip_frontmatter(text).strip()
    truncated = ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS]
        truncated = f"\n[... truncated at {MAX_BODY_CHARS} characters ...]"

    resources = list_resources(meta)
    res_block = ""
    if resources:
        listed = "\n".join(f"  <file>{r}</file>" for r in resources)
        res_block = (
            "\n<skill_resources>\n" + listed + "\n</skill_resources>\n"
            "Read any of these with read_file(\"<path shown above>\") when the instructions above call for it."
        )
    return (
        f"<skill_content name=\"{meta.name}\">\n"
        f"{body}{truncated}\n"
        f"{res_block}\n"
        f"</skill_content>"
    )


def resolve_in_root(root, rel_path):
    """Resolve `rel_path` inside the skills root, refusing to escape it.

    Returns the absolute path. Raises ValueError on traversal attempts, absolute paths
    pointing outside the root, symlink escapes, missing files and directories.
    """
    if not root:
        raise ValueError("skills are disabled - no readable skills directory")
    root_real = os.path.realpath(os.path.abspath(root))
    rel = str(rel_path or "").strip().replace("\\", "/")
    if not rel:
        raise ValueError("empty path")

    candidate = rel if os.path.isabs(rel) else os.path.join(root_real, rel)
    candidate = os.path.realpath(os.path.abspath(candidate))

    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise ValueError(f"path escapes the skills directory: {rel_path!r}")
    if os.path.isdir(candidate):
        raise ValueError(f"{rel_path!r} is a directory, not a file")
    if not os.path.isfile(candidate):
        raise ValueError(f"file not found: {rel_path!r}")
    return candidate


def read_resource(root, rel_path):
    """Level-3 read: sandboxed file text, wrapped for the transcript."""
    path = resolve_in_root(root, rel_path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_BODY_CHARS + 1)
    truncated = ""
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS]
        truncated = f"\n[... truncated at {MAX_BODY_CHARS} characters ...]"
    shown = os.path.relpath(path, os.path.realpath(os.path.abspath(root))).replace(os.sep, "/")
    return f"<skill_file path=\"{shown}\">\n{text.strip()}{truncated}\n</skill_file>"


class SkillSession:
    """Per-agent (per planner run / per subtask) skill loading state and tools.

    The LLM calls load_skill(...) / read_file(...) from inside a ```python block. Their
    text output must NOT travel through stdout: agent_runner.execute_python_blocks drops
    captured stdout of 2000+ chars, which would silently swallow a skill body. Instead the
    text is buffered here and the exec loops append drain_pending() verbatim to the next
    user turn - so the skill lands in the conversation exactly as if it had been part of
    the prompt. The tools themselves only print a short confirmation line.
    """

    def __init__(self, root, scope, logger=None, enabled=True):
        self.root = os.path.abspath(root) if (root and enabled) else None
        self.scope = scope
        self.logger = logger
        self.enabled = bool(enabled and root)
        self.index = discover(self.root, scope, logger) if self.enabled else []
        self.loaded = {}     # name -> wrapped body text (order preserved, py3.7+)
        self.pending = []    # text blocks awaiting injection into the next user turn

    # --- prompt-side helpers --------------------------------------------
    def index_text(self):
        return render_index(self.index, self.scope)

    def loaded_block(self):
        """All bodies loaded so far - used to re-inject skills into a rebuilt prompt."""
        if not self.loaded:
            return ""
        return "\n\n".join(self.loaded.values())

    def drain_pending(self):
        """Return and clear the text buffered since the last turn."""
        if not self.pending:
            return ""
        out = "\n\n".join(self.pending)
        self.pending = []
        return out

    # --- tools injected into the exec environment -----------------------
    def load_skill(self, name):
        """Load the full instructions of a named skill into the conversation."""
        name = str(name or "").strip()
        valid = [m.name for m in self.index]
        if not self.enabled or not valid:
            print("load_skill: no skills are available in this session.")
            return
        meta = next((m for m in self.index if m.name == name), None)
        if meta is None:
            print(f"load_skill: unknown skill {name!r}. Available skills: {', '.join(valid)}")
            _log(self.logger, WARNING + f"[skills:{self.scope}] load_skill: unknown skill {name!r}" + ENDC)
            return
        if name in self.loaded:
            print(f"load_skill: skill '{name}' is already loaded and still applies - follow the instructions already in this conversation.")
            return
        try:
            body = read_body(meta)
        except Exception as e:
            print(f"load_skill: failed to read skill '{name}': {e}")
            _log(self.logger, FAIL + f"[skills:{self.scope}] failed to read {meta.path}: {e}" + ENDC)
            return
        self.loaded[name] = body
        self.pending.append(body)
        _log(self.logger, OK + f"[skills:{self.scope}] loaded skill '{name}' ({len(body)} chars)" + ENDC)
        print(f"load_skill: loaded skill '{name}'. Its full instructions follow below; apply them to this task.")

    def read_file(self, path):
        """Read a file bundled with a skill (scripts/, references/, assets/)."""
        if not self.enabled:
            print("read_file: skills are disabled in this session.")
            return
        try:
            text = read_resource(self.root, path)
        except Exception as e:
            print(f"read_file: {e}")
            _log(self.logger, WARNING + f"[skills:{self.scope}] read_file({path!r}) failed: {e}" + ENDC)
            return
        self.pending.append(text)
        _log(self.logger, OK + f"[skills:{self.scope}] read_file('{path}') ({len(text)} chars)" + ENDC)
        print(f"read_file: loaded '{path}'. Its contents follow below.")

    def exec_locals(self):
        """Tools merged into the agent's exec environment."""
        return {"load_skill": self.load_skill, "read_file": self.read_file}
