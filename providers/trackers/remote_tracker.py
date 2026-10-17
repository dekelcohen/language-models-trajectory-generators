"""Remote point tracker: a thin proxy to a tracker session on :mod:`gpu_server`.

Runs a heavy tracker (CoTracker3 today) on a GPU server - another machine, or another
conda env on this machine - while the rollout runs where the robot/sim is. The server-side
module wraps the same provider class as the local one (``gpu_server/modules/cotracker.py``
wraps :class:`~providers.trackers.cotracker_tracker.CoTrackerTracker`), so window, flush,
staleness (``meta["stale_frames"]``, ``meta["flushes"]``) and visibility semantics are
identical; only the transport is added.

Calls are synchronous: ``update`` returns when the server has processed the frame. With
CoTracker a non-flush frame is just buffered (~1 ms round trip on localhost) and a flush
frame carries the model time, so the wall time the session measures is the real latency.

Settings (constructor kwarg > environment > ``config``):

    url          GPU_SERVER_URL        config.tracker_remote_url      ws://127.0.0.1:8765/v1/ws
    module       GPU_SERVER_MODULE     config.tracker_remote_module   cotracker
    token        GPU_SERVER_TOKEN      -
    image_codec  -                     config.tracker_remote_image_codec  raw | png
    timeout      -                     config.tracker_remote_timeout_s

If nothing answers at a local URL, the first ``init`` starts a server through
:mod:`gpu_server.launcher` (policy ``--tracking-server``: it is left running by default and
exits after 30 idle minutes). ``init`` reconnects once if the server went away in between.

Wire protocol: :mod:`gpu_server.protocol` (JSON header + raw arrays, no base64).
"""

import logging
import os

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

logger = logging.getLogger(__name__)

DEFAULT_URL = "ws://127.0.0.1:8765/v1/ws"


class RemoteTracker(PointTracker):
    name = "remote"

    def __init__(self, url=None, module=None, token=None, image_codec=None, timeout=None,
                 session_params=None, client=None, autolaunch=None, **kwargs):
        super().__init__(**kwargs)
        self.url = (url or os.environ.get("GPU_SERVER_URL")
                    or getattr(config, "tracker_remote_url", DEFAULT_URL))
        self.module = (module or os.environ.get("GPU_SERVER_MODULE")
                       or getattr(config, "tracker_remote_module", "cotracker"))
        self.token = token
        self.image_codec = (image_codec or os.environ.get("GPU_SERVER_CODEC")
                            or getattr(config, "tracker_remote_image_codec", "raw"))
        self.timeout = float(timeout or getattr(config, "tracker_remote_timeout_s", 60.0))
        self.session_params = dict(session_params or {})
        # None = follow gpu_server.launcher's policy (launch a local server when none answers,
        # unless --tracking-server no-launch); False = never launch.
        self.autolaunch = autolaunch
        self._client = client
        self._own_client = client is None
        # The server-side CoTracker keeps its window memory across occlusions too.
        self.keeps_memory_through_occlusion = self.module in ("cotracker",)
        self._session = None
        self._updates = 0
        self._rtt_sum = 0.0
        # Mirrors the server tracker's window-flush counter: tracking.session reads
        # ``tracker._flushes`` before an update to tell flush frames from buffered ones.
        self._flushes = 0

    # -- connection --------------------------------------------------------
    def _ensure_session(self):
        if self._session is not None:
            return self._session
        if self._client is None:
            from gpu_server import launcher
            from gpu_server.client import GpuClient
            if self.autolaunch is not False:
                launcher.ensure_server(self.url, self.module)
            self._client = GpuClient(self.url, timeout=self.timeout, token=self.token,
                                     image_codec=self.image_codec)
            launcher.register_client(self)
        self._session = self._client.open(self.module, **self.session_params)
        logger.info("[remote] session %s on %s (%s, codec=%s)", self._session.sid, self.url,
                    self.module, self.image_codec)
        return self._session

    def _drop_connection(self):
        self._session = None
        if self._client is not None and self._own_client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001 - the server may already be gone
                pass
            self._session = None
        if self._client is not None and self._own_client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- PointTracker --------------------------------------------------------
    def init(self, frame, points, obj_id=None):
        pts = self._as_points(points)
        args = dict(frame=np.ascontiguousarray(frame), points=pts, obj_id=obj_id)
        try:
            reply, _ = self._ensure_session().call("init", **args)
        except ConnectionError as exc:
            # A new stream starts here, so nothing is lost by reconnecting: the server may have
            # exited after its idle timeout (or been restarted) since the last session.
            if not self._own_client:
                raise
            logger.warning("[remote] %s lost (%s); reconnecting", self.url, exc)
            self._drop_connection()
            reply, _ = self._ensure_session().call("init", **args)
        self.n_points = int(reply.get("n_points", len(pts)))
        self.initialised = True
        self._flushes = 0

    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        reply, out = self._session.call("update", frame=np.ascontiguousarray(frame))
        rtt = self._client.last_rtt_ms
        self._updates += 1
        self._rtt_sum += rtt or 0.0
        meta = dict(reply.get("meta") or {})
        if meta.get("flushes") is not None:
            self._flushes = int(meta["flushes"])
        meta.update({"remote_module": self.module, "remote_rtt_ms": round(rtt, 3),
                     "remote_server_ms": reply.get("server_ms"),
                     "transport": self._client.scheme})
        if meta.get("fresh"):
            logger.debug("[remote] flush %s: rtt %.1f ms (server %.1f ms)", meta.get("flushes"),
                         rtt, reply.get("server_ms") or -1)
        return TrackResult(points=np.array(out["points"], dtype=float),
                           visible=np.array(out["visible"], dtype=bool),
                           scores=np.array(out["scores"], dtype=float), meta=meta)

    def reset(self):
        super().reset()
        self._flushes = 0
        if self._session is not None:
            self._session.call("reset")

    @property
    def mean_rtt_ms(self):
        return self._rtt_sum / self._updates if self._updates else None
