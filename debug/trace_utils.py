"""Opt-in tracing of the sim-layer interface, for before/after refactor comparison.

Disabled unless the ``LMTG_TRACE`` environment variable points at an output file, so
there is zero cost on normal runs (no permanent hot-loop logging). Enable with::

    $env:LMTG_TRACE = "outputs/trace_pybullet_door.jsonl"

Every record is one JSON object on its own line::

    {"seq": 0, "kind": "call", "name": "Robot.move", "ctx": {...}, "data": {...}}

Timestamps are deliberately omitted by default so two runs of the same scene produce
byte-identical trace files, which is what makes the regression diff meaningful.
"""

import functools
import json
import os
import threading

import numpy as np

TRACE_ENV_VAR = "LMTG_TRACE"
TRACE_TIME_ENV_VAR = "LMTG_TRACE_TIME"

# Floats are rounded before serialisation so that harmless last-bit noise does not
# swamp a diff. 12 decimals is far tighter than the 1e-9 regression gate.
FLOAT_ROUND_DECIMALS = 12

_lock = threading.Lock()
_state = {
    "path": None,
    "handle": None,
    "seq": 0,
    "context": {},
    "checked": False,
}


def is_enabled():
    """True when tracing is on. Cheap enough to call in inner loops."""
    if not _state["checked"]:
        _init_from_env()
    return _state["handle"] is not None


def _init_from_env():
    _state["checked"] = True
    path = os.environ.get(TRACE_ENV_VAR, "").strip()
    if path:
        _open(path)


def _open(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    _state["path"] = path
    _state["handle"] = open(path, "w", encoding="utf-8")
    _state["seq"] = 0


def start(path):
    """Begin (or restart) a trace at ``path``, regardless of the environment variable."""
    with _lock:
        _close_locked()
        _state["checked"] = True
        _open(path)
    return path


def stop():
    """Flush and close the current trace."""
    with _lock:
        _close_locked()


def _close_locked():
    handle = _state["handle"]
    if handle is not None:
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
    _state["handle"] = None
    _state["path"] = None
    _state["context"] = {}


def set_context(**kwargs):
    """Attach key/value tags (task name, phase, ...) to every subsequent record."""
    if not is_enabled():
        return
    with _lock:
        _state["context"].update({str(k): encode(v) for k, v in kwargs.items()})


class context:
    """Context manager form of :func:`set_context` that restores the previous tags."""

    def __init__(self, **kwargs):
        self._new = kwargs
        self._saved = None

    def __enter__(self):
        if is_enabled():
            with _lock:
                self._saved = dict(_state["context"])
            set_context(**self._new)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._saved is not None:
            with _lock:
                _state["context"] = self._saved
        return False


def encode(obj, _depth=0):
    """Convert ``obj`` into something ``json.dumps`` accepts, deterministically.

    numpy arrays keep their dtype and shape so a reshaped-but-equal array still shows
    up as a difference (that is exactly the kind of camera-matrix regression we are
    guarding against).
    """
    if _depth > 8:
        return "<max-depth>"

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return _round(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _round(float(obj))

    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": _round_nested(obj.tolist()),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, dict):
        return {str(k): encode(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        # Tuples become lists; the pybullet API returns tuples where the ported code
        # will return lists, and that difference is not meaningful.
        return [encode(v, _depth + 1) for v in obj]

    for attr in ("tolist", "item"):
        if hasattr(obj, attr):
            try:
                return encode(getattr(obj, attr)(), _depth + 1)
            except Exception:
                pass

    return f"<{type(obj).__name__}>"


def _round(value):
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    return round(value, FLOAT_ROUND_DECIMALS)


def _round_nested(value):
    if isinstance(value, list):
        return [_round_nested(v) for v in value]
    if isinstance(value, float):
        return _round(value)
    return value


def emit(kind, name, data):
    """Write one record. No-op when tracing is disabled."""
    if not is_enabled():
        return
    with _lock:
        handle = _state["handle"]
        if handle is None:
            return
        record = {
            "seq": _state["seq"],
            "kind": kind,
            "name": name,
        }
        if _state["context"]:
            record["ctx"] = dict(_state["context"])
        record["data"] = data
        if os.environ.get(TRACE_TIME_ENV_VAR, "0") == "1":
            import time

            record["t"] = time.time()
        _state["seq"] += 1
        try:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
        except Exception:
            # Tracing must never break the simulation.
            pass


def trace_value(name, value=None, **fields):
    """Record a named value (or a bag of named fields) at this point in the run."""
    if not is_enabled():
        return
    if fields:
        payload = {k: encode(v) for k, v in fields.items()}
        if value is not None:
            payload["value"] = encode(value)
    else:
        payload = encode(value)
    emit("value", name, payload)


def traced(name=None, args=True, result=True, skip_self=True):
    """Decorator recording a function's inputs and outputs.

    ``args``/``result`` can be turned off for calls whose payload is huge (image
    buffers) but whose invocation order still matters.
    """

    def decorate(func):
        label = name or getattr(func, "__qualname__", getattr(func, "__name__", "?"))

        @functools.wraps(func)
        def wrapper(*call_args, **call_kwargs):
            if not is_enabled():
                return func(*call_args, **call_kwargs)

            if args:
                positional = call_args[1:] if (skip_self and call_args) else call_args
                emit(
                    "call",
                    label,
                    {
                        "args": [encode(a) for a in positional],
                        "kwargs": {str(k): encode(v) for k, v in call_kwargs.items()},
                    },
                )
            else:
                emit("call", label, None)

            out = func(*call_args, **call_kwargs)
            emit("return", label, encode(out) if result else None)
            return out

        return wrapper

    return decorate
