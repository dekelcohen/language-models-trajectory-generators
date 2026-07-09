# -*- coding: utf-8 -*-
"""
Unit tests for the LLM response cache (providers/llms/llm_cache.py) and the
caching wrapper (models.call_llm_cached).

No real LLM or simulator is used:
  * models.call_llm_provider is monkeypatched to a counting stub.
  * A FakeConn emulates the GET_STATE round-trip (env state) without a sim.
All cache data lives in a temp folder removed at teardown.

Run:
    python -m unittest tests.test_llm_cache -v
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import models
from providers.llms.llm_cache import LLMCache


class FakeConn:
    """Emulates the sim connection: replies to GET_STATE with a scripted env state."""

    def __init__(self, env_state=None):
        self.env_state = env_state if env_state is not None else {"eef_pos": [0.1, 0.2, 0.3]}
        self.get_state_calls = 0

    def send(self, payload):
        # payload is [config.GET_STATE]
        self.get_state_calls += 1

    def recv(self):
        return dict(self.env_state)


class LLMCacheTestBase(unittest.TestCase):
    MODEL = "azure-gpt-5-test"

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="llm_cache_test_")
        self.cache = LLMCache(cache_dir=self.tmp_dir, float_tolerance=1e-2)

        # Count provider invocations and return a deterministic, unique response.
        self.provider_calls = 0
        self._orig_provider = models.call_llm_provider

        def fake_provider(client, model, messages, max_tokens, reasoning_effort, file):
            self.provider_calls += 1
            return f"RESP-{self.provider_calls}"

        models.call_llm_provider = fake_provider

    def tearDown(self):
        models.call_llm_provider = self._orig_provider
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _opts(self, **overrides):
        opts = {"cache": self.cache, "max_tokens": 100, "reasoning_effort": None}
        opts.update(overrides)
        return opts


class TestTextOnlyCalls(LLMCacheTestBase):
    def test_first_call_is_a_miss(self):
        conn = FakeConn()
        msgs = models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user",
                                      options=self._opts())
        self.assertEqual(self.provider_calls, 1)
        self.assertEqual(msgs[-1]["content"], "RESP-1")

    def test_second_identical_call_is_a_hit(self):
        conn = FakeConn()
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts())
        msgs2 = models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts())
        # Provider called only once; second served from cache.
        self.assertEqual(self.provider_calls, 1)
        self.assertEqual(msgs2[-1]["content"], "RESP-1")

    def test_text_only_never_queries_env_state(self):
        conn = FakeConn()
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts())
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts())
        # No GET_STATE round-trips for text-only calls.
        self.assertEqual(conn.get_state_calls, 0)

    def test_different_prompt_is_a_miss(self):
        conn = FakeConn()
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts())
        msgs2 = models.call_llm_cached(conn, None, self.MODEL, "goodbye", [], "user", options=self._opts())
        self.assertEqual(self.provider_calls, 2)
        self.assertEqual(msgs2[-1]["content"], "RESP-2")

    def test_different_params_is_a_miss(self):
        conn = FakeConn()
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts(max_tokens=100))
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=self._opts(max_tokens=200))
        self.assertEqual(self.provider_calls, 2)


class TestImageCalls(LLMCacheTestBase):
    def _img_message(self, b64="AAAA"):
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]

    def test_image_call_queries_env_state(self):
        conn = FakeConn(env_state={"eef_pos": [0.10, 0.20, 0.30]})
        models.call_llm_cached(conn, None, self.MODEL, None, self._img_message(), "user",
                               options=self._opts())
        # Image-bearing call fetches env state via GET_STATE.
        self.assertEqual(conn.get_state_calls, 1)
        self.assertEqual(self.provider_calls, 1)

    def test_image_call_hit_when_env_state_within_tolerance(self):
        conn1 = FakeConn(env_state={"eef_pos": [0.100, 0.200, 0.300]})
        conn2 = FakeConn(env_state={"eef_pos": [0.104, 0.203, 0.298]})  # within 1e-2
        models.call_llm_cached(conn1, None, self.MODEL, None, self._img_message(), "user", options=self._opts())
        msgs2 = models.call_llm_cached(conn2, None, self.MODEL, None, self._img_message(), "user", options=self._opts())
        # Same text/params + near-identical env state -> cache hit.
        self.assertEqual(self.provider_calls, 1)
        self.assertEqual(msgs2[-1]["content"], "RESP-1")

    def test_image_call_miss_when_env_state_beyond_tolerance(self):
        conn1 = FakeConn(env_state={"eef_pos": [0.10, 0.20, 0.30]})
        conn2 = FakeConn(env_state={"eef_pos": [0.90, 0.20, 0.30]})  # far -> miss
        models.call_llm_cached(conn1, None, self.MODEL, None, self._img_message(), "user", options=self._opts())
        msgs2 = models.call_llm_cached(conn2, None, self.MODEL, None, self._img_message(), "user", options=self._opts())
        self.assertEqual(self.provider_calls, 2)
        self.assertEqual(msgs2[-1]["content"], "RESP-2")

    def test_image_bytes_excluded_from_level1_key(self):
        # Same text + same env state but different image bytes -> still a hit.
        conn1 = FakeConn(env_state={"eef_pos": [0.10, 0.20, 0.30]})
        conn2 = FakeConn(env_state={"eef_pos": [0.10, 0.20, 0.30]})
        models.call_llm_cached(conn1, None, self.MODEL, None, self._img_message("AAAA"), "user", options=self._opts())
        msgs2 = models.call_llm_cached(conn2, None, self.MODEL, None, self._img_message("ZZZZ"), "user", options=self._opts())
        self.assertEqual(self.provider_calls, 1)
        self.assertEqual(msgs2[-1]["content"], "RESP-1")


class TestCacheDisabled(LLMCacheTestBase):
    def test_cache_none_always_calls_provider(self):
        conn = FakeConn()
        opts = {"cache": None, "max_tokens": 100, "reasoning_effort": None}
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=opts)
        models.call_llm_cached(conn, None, self.MODEL, "hello", [], "user", options=opts)
        # No caching -> provider called every time, and no GET_STATE round-trips.
        self.assertEqual(self.provider_calls, 2)
        self.assertEqual(conn.get_state_calls, 0)


class TestLLMCacheUnit(LLMCacheTestBase):
    """Direct tests of the LLMCache primitives."""

    def test_layout_is_two_level(self):
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        msgs = [{"role": "user", "content": "hi"}]
        self.cache.get(self.MODEL, msgs, params, {"eef_pos": [0.0, 0.0, 0.0]},
                       lambda: "R")
        model_dir = os.path.join(self.tmp_dir, self.cache._sanitize(self.MODEL))
        self.assertTrue(os.path.isdir(model_dir))
        l1_folders = os.listdir(model_dir)
        self.assertEqual(len(l1_folders), 1)
        l2_files = [f for f in os.listdir(os.path.join(model_dir, l1_folders[0])) if f.endswith(".json")]
        self.assertEqual(len(l2_files), 1)

    def test_smart_match_float_tolerance(self):
        self.assertTrue(self.cache._smart_match({"a": [0.100, 0.200]}, {"a": [0.104, 0.198]}))
        self.assertFalse(self.cache._smart_match({"a": [0.10, 0.20]}, {"a": [0.90, 0.20]}))
        self.assertFalse(self.cache._smart_match({"a": 1}, {"b": 1}))  # key mismatch
        self.assertTrue(self.cache._smart_match(None, None))

    def test_none_env_state_single_entry(self):
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        msgs = [{"role": "user", "content": "hi"}]
        r1 = self.cache.get(self.MODEL, msgs, params, None, lambda: "R1")
        r2 = self.cache.get(self.MODEL, msgs, params, None, lambda: "R2")
        self.assertEqual(r1, "R1")
        self.assertEqual(r2, "R1")


class TestCacheLogging(LLMCacheTestBase):
    """Verify hit/miss reasons are logged via the injected logger."""

    class RecordingLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            self.messages.append(str(msg))

    def _make_cache(self):
        rec = self.RecordingLogger()
        cache = LLMCache(cache_dir=self.tmp_dir, float_tolerance=1e-2, logger=rec)
        return cache, rec

    def test_logs_text_params_miss(self):
        cache, rec = self._make_cache()
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        cache.get(self.MODEL, [{"role": "user", "content": "hi"}], params, None, lambda: "R")
        self.assertTrue(any("MISS" in m and "text/params mismatch" in m for m in rec.messages), rec.messages)

    def test_logs_hit(self):
        cache, rec = self._make_cache()
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        msgs = [{"role": "user", "content": "hi"}]
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.0, 0.0, 0.0]}, lambda: "R")
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.001, 0.0, 0.0]}, lambda: "R2")
        self.assertTrue(any("HIT" in m for m in rec.messages), rec.messages)

    def test_logs_env_state_float_tolerance_miss(self):
        cache, rec = self._make_cache()
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        msgs = [{"role": "user", "content": "hi"}]
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.0, 0.0, 0.0]}, lambda: "R")
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.9, 0.0, 0.0]}, lambda: "R2")
        self.assertTrue(any("env-state mismatch" in m and "beyond tolerance" in m for m in rec.messages), rec.messages)

    def test_logs_env_state_key_mismatch_miss(self):
        cache, rec = self._make_cache()
        params = {"max_tokens": 100, "reasoning_effort": None, "temperature": 0}
        msgs = [{"role": "user", "content": "hi"}]
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.0, 0.0, 0.0]}, lambda: "R")
        cache.get(self.MODEL, msgs, params, {"eef_pos": [0.0, 0.0, 0.0], "extra": 1}, lambda: "R2")
        self.assertTrue(any("env-state mismatch" in m and "different keys" in m for m in rec.messages), rec.messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
