"""Remote point-tracker client (stub).

Purpose: run a heavy point tracker (CoTracker, TAPIR, SAM2 video) on a GPU server while
the rollout runs locally. This file fixes the wire protocol so a server can be written
independently; it deliberately does not ship a client implementation yet, because there
is no server to test against and a silently-wrong tracker is worse than an explicit error.

Transport follows the existing repo pattern in ``providers/ws_connection.py``: a
WebSocket carrying JSON messages, an asyncio loop on a daemon thread, and a thread-safe
queue for replies.

Protocol
--------
Client -> server::

    {"cmd": "init",   "session": "<uuid>", "cam": "head",
     "points": [[x, y], ...], "obj_id": "mug",
     "image": "<base64 png>", "seq": 0}
    {"cmd": "update", "session": "<uuid>", "cam": "head",
     "image": "<base64 png>", "seq": 1}
    {"cmd": "close",  "session": "<uuid>"}

Server -> client::

    {"ok": true, "seq": 1,
     "points":  [[x, y], ...],     # same order/length as the init points
     "visible": [true, false, ...],
     "scores":  [0.93, 0.0, ...]}
    {"ok": false, "error": "..."}

Notes for the implementer
-------------------------
* Frames are RGB uint8 PNG-encoded then base64'd (``message_media.encode_media`` already
  does this for the LLM providers - reuse it).
* ``seq`` must be echoed so a late reply can be dropped rather than mis-attributed.
* The session is stateful per ``(session, cam)``: re-sending ``init`` re-seeds that camera,
  which is exactly what ``tracking.health`` asks for on a cross-camera re-seed.
* Latency budget: the rollout hook runs at ~5 FPS, so a round trip under ~150 ms keeps
  tracking synchronous. Beyond that, buffer frames and report the newest available result
  (``TrackResult.meta["stale_frames"]``).
"""

from providers.trackers.base import PointTracker

PROTOCOL_VERSION = 1
DEFAULT_URL = "ws://127.0.0.1:8765/track"

NOT_IMPLEMENTED_MSG = (
    "The remote tracker provider is a documented stub - no server implementation exists "
    "yet. See the module docstring in providers/trackers/remote_tracker.py for the wire "
    "protocol, or run with --tracker-provider template."
)


class RemoteTracker(PointTracker):
    name = "remote"

    def __init__(self, url=DEFAULT_URL, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def init(self, frame, points, obj_id=None):  # pragma: no cover - unreachable
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def update(self, frame):  # pragma: no cover - unreachable
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)
