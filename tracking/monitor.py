"""Running a user/LLM supplied invariant against each tracked frame.

A monitor is any callable ``monitor(state) -> dict | None`` where ``state`` is a
:class:`tracking.types.TrackFrameReport`. It may also be supplied as source text (the
agent process ships ``inspect.getsource`` of the LLM's function across IPC, because a
closure cannot be pickled into the simulator process).

Return contract::

    None or {"status": "ok"}                  -> continue
    {"status": "warn",   "reason": str}       -> log only
    {"status": "record", "reason": str}       -> keep going, flag the frame for the reviewer
    {"status": "abort",  "reason": str}       -> stop the trajectory, fail the sub-task

Anything else (bad status, exception, wrong type) is contained and downgraded to
``warn``: a broken monitor must never take down a rollout.
"""

import traceback

from tracking.types import (
    STATUS_ABORT,
    STATUS_OK,
    STATUS_RECORD,
    STATUS_SEVERITY,
    STATUS_WARN,
    MonitorResult,
)

VALID_STATUSES = (STATUS_OK, STATUS_WARN, STATUS_RECORD, STATUS_ABORT)


def normalise(value) -> MonitorResult:
    """Coerce whatever a monitor returned into a :class:`MonitorResult`."""
    if value is None:
        return MonitorResult(STATUS_OK)
    if isinstance(value, MonitorResult):
        return value
    if isinstance(value, str):
        status = value.lower()
        return MonitorResult(status if status in VALID_STATUSES else STATUS_WARN,
                             "" if status in VALID_STATUSES else f"unknown status {value!r}")
    if isinstance(value, bool):
        # `True` from a naive invariant means "still fine".
        return MonitorResult(STATUS_OK if value else STATUS_ABORT,
                             "" if value else "monitor returned False")
    if isinstance(value, dict):
        status = str(value.get("status", STATUS_OK)).lower()
        reason = str(value.get("reason", ""))
        if status not in VALID_STATUSES:
            return MonitorResult(STATUS_WARN, f"unknown status {status!r}", dict(value))
        data = {k: v for k, v in value.items() if k not in ("status", "reason")}
        return MonitorResult(status, reason, data)
    return MonitorResult(STATUS_WARN, f"monitor returned unsupported type {type(value).__name__}")


def compile_monitor(source, func_name=None, extra_globals=None):
    """Compile monitor *source text* into a callable.

    The source runs with the normal builtins plus ``numpy``, ``math`` and the built-in
    monitor library, matching the environment the LLM already writes code in elsewhere
    in this repo (``agent_runner.execute_python_blocks`` also ``exec``s model code).
    ``func_name`` defaults to the last function defined in the source.
    """
    import math

    import numpy as np

    from tracking import monitors as monitors_lib

    namespace = {"__builtins__": __builtins__, "np": np, "numpy": np, "math": math}
    for name in monitors_lib.__all__:
        namespace[name] = getattr(monitors_lib, name)
    if extra_globals:
        namespace.update(extra_globals)

    exec(compile(source, "<monitor>", "exec"), namespace)

    if func_name:
        fn = namespace.get(func_name)
        if not callable(fn):
            raise ValueError(f"monitor source defines no callable named {func_name!r}")
        return fn

    candidates = [v for v in namespace.values()
                  if callable(v)
                  and getattr(v, "__code__", None) is not None
                  and getattr(v.__code__, "co_filename", "") == "<monitor>"]
    if not candidates:
        raise ValueError("monitor source defines no function")
    return candidates[-1]


class MonitorRunner:
    """Calls a monitor safely and remembers the worst status seen."""

    def __init__(self, monitor=None, logger=None):
        self.logger = logger
        self.monitor = monitor
        self.worst = MonitorResult(STATUS_OK)
        self.error_count = 0

    @classmethod
    def from_spec(cls, spec, logger=None):
        """Build from a callable, source text, ``{"source":..., "name":...}``, or ``None``.

        ``None`` selects :func:`tracking.monitors.default_monitor`, so tracking always has
        a useful invariant even when the agent supplies nothing.
        """
        from tracking import monitors as monitors_lib

        if spec is None:
            return cls(monitors_lib.default_monitor(), logger=logger)
        if callable(spec):
            return cls(spec, logger=logger)
        if isinstance(spec, str):
            return cls(compile_monitor(spec), logger=logger)
        if isinstance(spec, dict):
            if "source" in spec:
                return cls(compile_monitor(spec["source"], spec.get("name")), logger=logger)
            if "builtin" in spec:
                factory = getattr(monitors_lib, spec["builtin"], None)
                if factory is None:
                    raise ValueError(f"unknown built-in monitor {spec['builtin']!r}")
                return cls(factory(**spec.get("kwargs", {})), logger=logger)
        raise TypeError(f"Unsupported monitor spec of type {type(spec).__name__}")

    def __call__(self, state) -> MonitorResult:
        if self.monitor is None:
            return MonitorResult(STATUS_OK)
        try:
            result = normalise(self.monitor(state))
        except Exception as exc:
            self.error_count += 1
            detail = traceback.format_exc(limit=3)
            if self.logger is not None and self.error_count <= 3:
                self.logger.info(f"[tracking] monitor raised {type(exc).__name__}: {exc}\n{detail}")
            result = MonitorResult(STATUS_WARN, f"monitor error: {type(exc).__name__}: {exc}")
        if STATUS_SEVERITY.get(result.status, 0) > self.worst.severity:
            self.worst = result
        return result
