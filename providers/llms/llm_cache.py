# -*- coding: utf-8 -*-
"""
llm_cache.py

Two-level, on-disk cache for LLM responses.

Layout:
    <cache_dir>/<model>/<text_params_hash>/<state_hash>.json

Level 1 (folder <text_params_hash>):
    sha256 over: model + llm params (max_tokens, reasoning_effort, temperature, ...)
    + all text values and dict keys found in the `messages` list.
    Image bytes (base64 data URLs) are deliberately EXCLUDED so that identical
    text/params reuse the same folder regardless of re-rendered images.

Level 2 (file <state_hash>.json):
    Each file stores one {"env_state": ..., "response": ...} entry.
    On get(), every entry in the L1 folder is scanned and its stored env_state
    is "smart matched" against the current env_state: structures must match and
    floats may differ by up to `float_tolerance` (to absorb physics-settling
    jitter). The first matching entry is a cache hit.

Semantics:
    get(...) is read-insert-if-not-exists. No deletion, no eviction.
"""

import os
import re
import io
import json
import hashlib


class LLMCache:
    """Standard cache interface: get(key_material, producer) -> value.

    Here the "key" is composed of (model, messages, params) for level 1 and
    env_state for the level-2 smart match.
    """

    def __init__(self, cache_dir="./cache", float_tolerance=1e-2, logger=None):
        self.cache_dir = cache_dir
        self.float_tolerance = float(float_tolerance)
        self.logger = logger
        os.makedirs(self.cache_dir, exist_ok=True)

    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        print(msg)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get(self, model, messages, params, env_state, producer):
        """Read-insert-if-not-exists.

        Args:
            model: model identifier string.
            messages: list of chat messages (text values + keys are hashed,
                image data URLs are excluded).
            params: dict of llm params affecting the output (max_tokens,
                reasoning_effort, temperature, ...).
            env_state: JSON-serializable dict describing the sim/env state, or
                None. Matched with float tolerance across cached entries.
            producer: zero-arg callable returning the response string on a miss.

        Returns:
            The cached or freshly produced response value.
        """
        folder = self._level1_folder(model, messages, params)
        had_l1 = os.path.isdir(folder) and any(
            n.endswith(".json") for n in os.listdir(folder)
        )
        os.makedirs(folder, exist_ok=True)

        hit, reason = self._scan_for_match(folder, env_state)
        if hit is not None:
            self._log(f"LLMCache HIT model={model} (text+params+env-state matched)")
            return hit

        if not had_l1:
            miss_reason = "text/params mismatch (no cached entry for this text+params)"
        else:
            miss_reason = f"env-state mismatch ({reason})" if reason else "env-state mismatch"
        self._log(f"LLMCache MISS model={model} reason={miss_reason}")

        value = producer()
        self._insert(folder, env_state, value)
        return value

    # ------------------------------------------------------------------ #
    # Level 1: text + params hash -> folder
    # ------------------------------------------------------------------ #
    def _level1_folder(self, model, messages, params):
        text_hash = self._text_params_hash(model, messages, params)
        return os.path.join(self.cache_dir, self._sanitize(model), text_hash)

    def _text_params_hash(self, model, messages, params):
        h = hashlib.sha256()
        h.update(("model=" + str(model)).encode("utf-8"))
        for k in sorted((params or {}).keys()):
            h.update(("|param|" + str(k) + "=" + str(params[k])).encode("utf-8"))
        h.update(b"|messages|")
        h.update(self._messages_text_signature(messages).encode("utf-8"))
        return h.hexdigest()

    def _messages_text_signature(self, messages):
        """Deterministic string of all dict keys + text values in messages,
        excluding image data (base64 data URLs)."""
        buf = io.StringIO()
        self._walk(messages, buf)
        return buf.getvalue()

    def _walk(self, node, buf):
        if isinstance(node, dict):
            for k in sorted(node.keys(), key=str):
                v = node[k]
                # Skip image payloads: image_url dicts and data: URLs.
                if k == "image_url":
                    buf.write("<k:image_url>")
                    continue
                buf.write("<k:" + str(k) + ">")
                self._walk(v, buf)
        elif isinstance(node, (list, tuple)):
            for item in node:
                buf.write("<i>")
                self._walk(item, buf)
        elif isinstance(node, str):
            if node.startswith("data:") and "base64," in node:
                return  # exclude raw image bytes
            buf.write(node)
        else:
            buf.write(repr(node))

    # ------------------------------------------------------------------ #
    # Level 2: env_state smart match
    # ------------------------------------------------------------------ #
    def _scan_for_match(self, folder, env_state):
        """Return (response, last_mismatch_reason).

        response is None on a miss; last_mismatch_reason describes why the
        closest cached entry failed the env-state smart match (best-effort;
        reflects the last scanned entry)."""
        last_reason = None
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return None, last_reason
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
            except (OSError, ValueError):
                continue
            reason = self._smart_diff(entry.get("env_state"), env_state)
            if reason is None:
                return entry.get("response"), None
            last_reason = reason
        return None, last_reason

    def _insert(self, folder, env_state, value):
        state_hash = self._state_hash(env_state)
        path = os.path.join(folder, state_hash + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"env_state": env_state, "response": value}, f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _state_hash(self, env_state):
        try:
            canonical = json.dumps(env_state, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = repr(env_state)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _smart_match(self, a, b):
        """Recursive structural compare; floats/ints within float_tolerance."""
        return self._smart_diff(a, b) is None

    def _smart_diff(self, a, b, path="env_state"):
        """Return None if a and b smart-match, else a human-readable reason
        describing the first difference (key mismatch or float beyond
        tolerance) and where it occurred."""
        if isinstance(a, bool) or isinstance(b, bool):
            return None if a == b else f"value mismatch at {path}: {a!r} != {b!r}"
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = abs(float(a) - float(b))
            if diff <= self.float_tolerance:
                return None
            return (f"float beyond tolerance at {path}: |{a} - {b}| = "
                    f"{diff:.4g} > {self.float_tolerance:g}")
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                missing = set(b.keys()) - set(a.keys())
                extra = set(a.keys()) - set(b.keys())
                return (f"different keys at {path}: "
                        f"cached_only={sorted(extra)} current_only={sorted(missing)}")
            for k in a:
                reason = self._smart_diff(a[k], b[k], f"{path}.{k}")
                if reason is not None:
                    return reason
            return None
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return f"different lengths at {path}: {len(a)} != {len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                reason = self._smart_diff(x, y, f"{path}[{i}]")
                if reason is not None:
                    return reason
            return None
        return None if a == b else f"value mismatch at {path}: {a!r} != {b!r}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize(name):
        return re.sub(r"[^A-Za-z0-9._-]", "_", str(name)) or "unknown"
