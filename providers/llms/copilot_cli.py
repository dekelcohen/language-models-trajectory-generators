# copilot_cli.py
"""GitHub Copilot CLI as an LLM provider (models prefixed `copilot-`).

Why this module looks different from the other providers
--------------------------------------------------------
Every other provider here POSTs a JSON body containing the whole `messages`
array. The `copilot` CLI has no such entry point:

  * `-p <text>` / piped stdin take exactly ONE prompt string - there is no way
    to hand it an array of messages / a prior conversation.
  * `--attachment <path>` takes image/PDF FILE PATHS. It never accepts base64.
  * Multi-turn is managed by the CLI itself: `--session-id <uuid>` starts a
    session and `--resume=<uuid>` appends another turn to it, with the CLI
    keeping the full prior context server-side. This was verified end to end:
    an image attached on turn 3 was still described correctly on turn 4 without
    the file being attached again.

So the conversation is replayed into a copilot *session* instead of a request
body. `_SESSIONS` maps (conversation_key, conversation-state signature) -> copilot
session id:

  * cold path (no matching signature - i.e. a brand new / branched `messages`
    list, such as the fresh `[]` conversations started by agent_runner): a new
    session id is created and the WHOLE conversation is flattened into one
    prompt, with every image attached.
  * warm path (the signature of `messages[:k]` is a session we ourselves
    produced): resume that session and send ONLY messages[k:], attaching only
    that turn's media.

A session id is popped from `_SESSIONS` the moment it is resumed, because after
the new turn it no longer represents the old prefix. That keeps one session
bound to exactly one conversation state, so two branches that share a prefix
can never end up appending into (and reading from) the same contaminated
session - the second branch simply misses and starts a fresh session.

`conversation_key` (threaded from callers via the `conversation_key` option, see
`models.CHATGPT_DEFAULT_OPTIONS`) namespaces the registry so independent, but
interleaved, conversations always land in separate copilot sessions - notably
the planner conversation ("planner") and each task/main-prompt conversation
("task-<id>"), which alternate turns throughout a run. Callers that pass no key
still get correct isolation from the content signature alone; the key just makes
the separation explicit and immune to two conversations sharing a prefix.

Media handling
--------------
Attachments are rebuilt from the base64 data URL stored in `messages`, NOT from
the caller's original file path. The pipeline overwrites paths such as
`rgb_image_head.png` on every step, so a path is only truthful for the turn it
was captured in; the bytes carried in the message are the only reliable source.
Files are written to a temp dir under a content-hashed name, so the same frame
is never written (or attached) twice.
"""
import base64
import hashlib
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid

logger = logging.getLogger(__name__)

MODEL_PREFIX = "copilot-"

# Copilot CLI supports images/PDFs only - no video attachments.
_ATTACHABLE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".heic", ".heif"}

# Hard ceiling on a single non-interactive copilot run.
DEFAULT_TIMEOUT_SEC = 900

# (conversation_key, conversation-state signature) -> copilot session id (see module docstring).
_SESSIONS: dict[tuple[str, str], str] = {}
_MAX_SESSIONS = 256

_ATTACH_DIR = os.path.join(tempfile.gettempdir(), "lmtg_copilot_attachments")

_COLD_PREAMBLE = (
    "Below is an existing conversation between a system, a user and an assistant. "
    "Continue it: reply with the next assistant message only.\n"
    "Answer directly from the conversation. Do not use any tools, do not read or write files, "
    "and do not describe what you are doing - output only the assistant reply itself.\n"
)


def _resolve_cli() -> str:
    """Absolute path of the `copilot` executable (raises if not installed)."""
    exe = os.environ.get("COPILOT_CLI_PATH") or shutil.which("copilot")
    if not exe:
        raise RuntimeError(
            "The `copilot` CLI was not found on PATH. Install GitHub Copilot CLI "
            "(https://docs.github.com/copilot/how-tos/copilot-cli) or set COPILOT_CLI_PATH "
            "to use a copilot-* model."
        )
    return exe


def _split_content(content):
    """Split an OpenAI-style message content into (text, media_parts)."""
    if isinstance(content, str):
        return content, []
    text_chunks, media = [], []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text_chunks.append(part.get("text") or "")
        elif part.get("type") in ("image_url", "video_url") or "image_url" in part or "video_url" in part:
            media.append(part)
    return "\n".join(c for c in text_chunks if c), media


def _part_data_url(part):
    """Return the `data:` URL carried by a media part, or None."""
    for key in ("image_url", "video_url"):
        holder = part.get(key)
        if isinstance(holder, dict) and isinstance(holder.get("url"), str):
            return holder["url"]
        if isinstance(holder, str):
            return holder
    return None


def _media_to_file(part):
    """Materialize a media part as a file on disk and return its path.

    The filename is derived from a hash of the decoded bytes, so identical
    frames reuse one file across turns and runs (and are only written once).
    Returns None for media the copilot CLI cannot attach (e.g. video).
    """
    url = _part_data_url(part)
    if not url or not url.startswith("data:") or "base64," not in url:
        return None
    header, b64 = url.split(",", 1)
    mime = header[len("data:"):].split(";")[0]
    if not mime.startswith("image/") and mime != "application/pdf":
        logger.warning("copilot_cli: dropping unsupported attachment mime %r (CLI takes images/PDFs only)", mime)
        return None

    ext = mimetypes.guess_extension(mime) or ""
    if ext == ".jpe":
        ext = ".jpg"
    if ext.lower() not in _ATTACHABLE_EXT:
        logger.warning("copilot_cli: dropping attachment with unsupported extension %r (mime %r)", ext, mime)
        return None

    try:
        blob = base64.b64decode(b64)
    except Exception as e:
        logger.warning("copilot_cli: failed to decode attachment (%s); skipping it", e)
        return None

    os.makedirs(_ATTACH_DIR, exist_ok=True)
    path = os.path.join(_ATTACH_DIR, hashlib.sha256(blob).hexdigest()[:32] + ext)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(blob)
    return path


def _render_turns(messages, attachments, label_roles):
    """Render `messages` as prompt text, appending their media to `attachments`.

    Each attached file is announced inline ("[image 1 attached]") so the model
    can tell which turn an attachment belongs to - the CLI itself attaches
    everything to the single prompt it receives.
    """
    lines = []
    for msg in messages:
        role = (msg.get("role") or "user") if isinstance(msg, dict) else "user"
        text, media = _split_content(msg.get("content") if isinstance(msg, dict) else msg)
        refs = []
        for part in media:
            path = _media_to_file(part)
            if path is None:
                refs.append("[non-attachable media omitted]")
                continue
            attachments.append(path)
            refs.append(f"[image {len(attachments)} attached]")
        body = "\n".join([t for t in ([text] + refs) if t])
        lines.append(f"[{role}]\n{body}" if label_roles else body)
    return "\n\n".join(lines)


def _signature(messages) -> str:
    """Stable hash of a conversation state.

    Media parts contribute only their data URL hash, so the signature stays
    small while still distinguishing different images.
    """
    h = hashlib.sha256()
    for msg in messages or []:
        role = (msg.get("role") or "user") if isinstance(msg, dict) else "user"
        text, media = _split_content(msg.get("content") if isinstance(msg, dict) else msg)
        h.update(b"|role|")
        h.update(role.encode("utf-8", "replace"))
        h.update(b"|text|")
        h.update((text or "").encode("utf-8", "replace"))
        for part in media:
            h.update(b"|media|")
            h.update(hashlib.sha256((_part_data_url(part) or "").encode("utf-8", "replace")).digest())
    return h.hexdigest()


def _remember(messages, session_id, conversation_key):
    """Bind `session_id` to the conversation state it now holds."""
    if len(_SESSIONS) >= _MAX_SESSIONS:
        _SESSIONS.clear()
        logger.info("copilot_cli: session registry full, cleared it")
    _SESSIONS[(conversation_key, _signature(messages))] = session_id


def _find_resumable(messages, conversation_key):
    """Longest already-sent prefix of `messages`: (session_id, index) or (None, 0).

    Only sessions registered under the same `conversation_key` are considered, so
    e.g. the planner conversation never resumes a task conversation's session.

    The entry is popped: once resumed, the session advances past that prefix, so
    it must never be handed out again (a branch sharing the prefix would then
    read another branch's turns).
    """
    for k in range(len(messages) - 1, 0, -1):
        sid = _SESSIONS.pop((conversation_key, _signature(messages[:k])), None)
        if sid:
            return sid, k
    return None, 0


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _run(exe, args, prompt, timeout):
    """Run copilot non-interactively, feeding `prompt` on stdin."""
    proc = subprocess.run(
        [exe] + args,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=_workdir(),
    )
    out = _ANSI_RE.sub("", proc.stdout or "").strip()
    err = _ANSI_RE.sub("", proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"copilot CLI failed (exit {proc.returncode}): {err or out}")
    if not out:
        raise RuntimeError(f"copilot CLI returned an empty response (stderr: {err})")
    return out


def _workdir():
    """Empty scratch cwd so the agent never touches the repo working tree."""
    d = os.path.join(tempfile.gettempdir(), "lmtg_copilot_cwd")
    os.makedirs(d, exist_ok=True)
    return d


def call_copilot(messages, model, max_tokens=None, reasoning_effort=None,
                 conversation_key=None, timeout=DEFAULT_TIMEOUT_SEC):
    """Run a conversation through the Copilot CLI and return the assistant text.

    Args:
        messages: OpenAI-style message dicts (text + base64 image parts).
        model: copilot model name WITHOUT the `copilot-` prefix (e.g. "claude-opus-5").
        max_tokens: accepted for signature parity - the CLI exposes no such flag.
        reasoning_effort: mapped to `--effort`.
        conversation_key: logical conversation this call belongs to (e.g. "planner",
            "task-3f2a"). Sessions are never shared across different keys, keeping
            the interleaved planner and task/main-prompt conversations separate.
        timeout: seconds before the subprocess is killed.
    """
    exe = _resolve_cli()
    if not messages:
        raise ValueError("call_copilot: messages is empty")
    if max_tokens is not None:
        logger.debug("copilot_cli: max_tokens=%s ignored (the copilot CLI has no equivalent flag)", max_tokens)

    conversation_key = conversation_key or "default"
    session_id, sent = _find_resumable(messages, conversation_key)
    attachments: list[str] = []

    if session_id:
        # Warm path: the CLI still holds everything up to `sent`.
        prompt = _render_turns(messages[sent:], attachments, label_roles=len(messages) - sent > 1)
        session_args = [f"--resume={session_id}"]
        logger.info("copilot_cli[%s]: resuming session %s, sending %d new message(s), %d attachment(s)",
                    conversation_key, session_id, len(messages) - sent, len(attachments))
    else:
        session_id = str(uuid.uuid4())
        body = _render_turns(messages, attachments, label_roles=True)
        prompt = f"{_COLD_PREAMBLE}\n=== conversation ===\n{body}\n=== end conversation ===\n"
        session_args = ["--session-id", session_id]
        logger.info("copilot_cli[%s]: new session %s for a fresh conversation (%d message(s), %d attachment(s))",
                    conversation_key, session_id, len(messages), len(attachments))

    args = session_args + [
        "--model", model,
        "--allow-all-tools",      # required for non-interactive mode
        "--available-tools=none",  # ...but expose no tools: we want a plain completion
        "--no-ask-user",
        "--no-custom-instructions",
        "--disable-builtin-mcps",
        "--no-color",
        "--silent",
        "--log-level", "none",
    ]
    if reasoning_effort:
        args += ["--effort", reasoning_effort]
    for path in attachments:
        args += ["--attachment", path]

    output = _run(exe, args, prompt, timeout)

    # The session now also holds this turn's reply: bind it to the resulting state.
    _remember(list(messages) + [{"role": "assistant", "content": output}], session_id, conversation_key)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from providers.llms.message_media import append_to_messages

    MODEL = "claude-haiku-4.5"

    print("=" * 60, "\nTest 1: text-only\n", "=" * 60)
    msgs = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    reply = call_copilot(msgs, model=MODEL)
    print("Response:", reply)

    print("=" * 60, "\nTest 2: follow-up reuses the copilot session\n", "=" * 60)
    msgs.append({"role": "assistant", "content": reply})
    msgs.append({"role": "user", "content": "Multiply that by 10. Just the number."})
    print("Response:", call_copilot(msgs, model=MODEL))

    image_path = r"./outputs/bugs/door_rgb_image_head.png"
    if os.path.exists(image_path):
        print("=" * 60, "\nTest 3: image attachment\n", "=" * 60)
        img_msgs = append_to_messages("What color is the door? One word.", [image_path], [], "user")
        print("Response:", call_copilot(img_msgs, model=MODEL))
    else:
        print(f"Skipped image test: {image_path} not found")
