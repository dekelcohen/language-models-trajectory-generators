"""Trivial module for tests and transport benchmarks (no model, no GPU)."""

import time

from gpu_server.modules.base import GpuModule, GpuSession


class EchoSession(GpuSession):
    def __init__(self, start=0):
        self.count = int(start)

    def m_add(self, args, arrays):
        self.count += int(args.get("n", 1))
        return {"count": self.count}, {}


class EchoModule(GpuModule):
    name = "echo"
    description = "returns its inputs; 'sleep' simulates a slow model; sessions count"
    stateful = True

    def load(self):
        # config load_s simulates a slow model load (preload / launcher tests)
        time.sleep(float(self.config.get("load_s", 0.0)))

    def m_echo(self, args, arrays):
        return {"args": args}, dict(arrays)

    def m_sleep(self, args, arrays):
        time.sleep(float(args.get("s", 0.1)))
        return {}, {}

    def m_fail(self, args, arrays):
        raise RuntimeError(args.get("msg", "boom"))

    def open_session(self, **params):
        return EchoSession(**params)
