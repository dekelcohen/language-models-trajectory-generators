"""CoTracker3 point tracker (opt-in, needs torch + a one-off model download).

Why online, not offline
-----------------------
CoTracker3 ships two entry points. ``cotracker3_offline`` wants the whole clip up front,
which a live rollout does not have, so this provider uses ``cotracker3_online``: the model
keeps internal state and consumes the video as a sliding window (``window = 2 * step``,
typically 16 frames advancing 8 at a time).

Window semantics (important for callers)
----------------------------------------
The online model does **not** emit a fresh prediction on every frame - it emits one every
``step`` frames, covering the last ``window`` frames, whose final row is the newest frame
in that chunk. Between flushes this tracker returns *the most recent available
prediction* (option (a): honest, never extrapolated) and reports how old it is in
``TrackResult.meta["stale_frames"]`` (0 = computed from this very frame). Per-point scores
are additionally decayed by ``stale_decay`` per stale frame so
:attr:`TrackResult.confidence` degrades on its own and the session down-weights or
re-seeds a camera that is coasting on an old window.

Warm start
----------
A cold window would mean no prediction at all for the first ``window`` frames. With
``warm_start=True`` (default) ``init`` pre-fills the buffer with copies of the seed frame,
so the very first ``update`` flushes a real window and ``stale_frames`` starts at 0. The
cost is that the first few windows contain a static prefix; positions are still correct
because the queries sit on that same frame.

Re-seeding
----------
``init`` may be called at any time (the session re-seeds a camera from its partner). The
cost path is: drop the frame buffer, drop the model's online state and re-run the
``is_first_step`` pass with the new queries on the next flush. The model itself is *not*
reloaded - that is the expensive part and it is kept. With ``warm_start`` the next
``update`` already produces a fresh prediction; without it there is a ``window``-frame
warm-up during which the seeded points are returned with a rising ``stale_frames``.

Cost (measured, RTX A1000 4 GB, torch 2.0.1+cu117, 320x240 input)
-----------------------------------------------------------------
~2.0 s per window flush (the first one ~6 s, cuDNN warm-up) and ~0 ms on the frames in
between, i.e. ~0.25 s per frame amortised at ``step = 8``. Fresh predictions were accurate
to ~0.2 px on a synthetic translating target; the stale frames in between carry the target's
own motion as error (5 px/frame motion => up to ~38 px just before the next flush), which is
exactly what ``stale_frames`` and the score decay advertise to the session.

Weights / offline use
---------------------
``torch.hub.load`` clones ``facebookresearch/co-tracker`` and downloads the checkpoint into
``TORCH_HOME`` (defaults to ``<repo>/cache/torch``, which is git-ignored). After one online
run everything is cached, so later runs work offline as long as ``TORCH_HOME`` points at
the same directory::

    set TORCH_HOME=D:\\...\\language-models-trajectory-generators\\cache\\torch
    python main.py --tracking --tracker-provider cotracker

Select with ``--tracker-provider cotracker``; the provider is imported lazily, so a repo
without torch or without network is unaffected unless it is actually selected.
"""

import logging
import os

import numpy as np

import config
from providers.trackers.base import PointTracker, TrackResult

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TORCH_HOME = os.path.join(REPO_ROOT, "cache", "torch")
HUB_REPO = "facebookresearch/co-tracker"

TORCH_HINT = (
    "The cotracker provider needs PyTorch. Install torch in the active environment (or "
    "run with --tracker-provider template)."
)


def _load_hint(torch_home, variant, detail):
    return (
        f"Could not load CoTracker ({variant}) from torch.hub: {detail}\n"
        f"The first run downloads '{HUB_REPO}' + the checkpoint into TORCH_HOME, which is "
        f"currently '{torch_home}'.\n"
        "Fixes:\n"
        f"  * run once with network access so the model is cached under TORCH_HOME;\n"
        f"  * or point TORCH_HOME at a machine that already has the cache "
        f"(set TORCH_HOME=<dir> containing hub/checkpoints), e.g. "
        f"set TORCH_HOME={DEFAULT_TORCH_HOME};\n"
        "  * or run with --tracker-provider template (no download, no torch)."
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(f"{TORCH_HINT} ({exc})") from exc
    return torch


def ensure_torch_home(torch_home=None):
    """Make ``TORCH_HOME`` explicit so the hub cache is reused across runs."""
    resolved = (torch_home
                or os.environ.get("TORCH_HOME")
                or getattr(config, "tracker_cotracker_torch_home", None)
                or DEFAULT_TORCH_HOME)
    resolved = os.path.abspath(os.path.expanduser(str(resolved)))
    os.makedirs(resolved, exist_ok=True)
    os.environ["TORCH_HOME"] = resolved
    return resolved


def resolve_device(device=None):
    """Auto -> ``cuda:0`` when CUDA works, else ``cpu``. Never ``cuda:1``."""
    torch = _import_torch()
    if device is None:
        device = getattr(config, "tracker_cotracker_device", None)
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    name = str(device).lower().strip()
    if name.startswith("cuda"):
        # The second GPU on this machine cannot run the model; pin to the first one.
        return "cuda:0"
    return name


def _hub_load(variant):
    """Thin seam around ``torch.hub.load`` so tests can fake the download."""
    torch = _import_torch()
    return torch.hub.load(HUB_REPO, variant)


def load_cotracker_model(variant=None, device=None, torch_home=None):
    """Fetch (or reuse the cached) CoTracker predictor, moved to ``device``."""
    variant = variant or getattr(config, "tracker_cotracker_variant", "cotracker3_online")
    if "online" not in variant:
        raise ValueError(
            f"CoTrackerTracker requires an online variant (got {variant!r}); the offline "
            "model needs the whole clip up front and cannot track live.")
    resolved_home = ensure_torch_home(torch_home)
    dev = resolve_device(device)
    logger.info("[cotracker] loading %s (device=%s, TORCH_HOME=%s)", variant, dev, resolved_home)
    try:
        model = _hub_load(variant)
    except Exception as exc:  # noqa: BLE001 - any hub/network/git failure must be readable
        raise RuntimeError(_load_hint(resolved_home, variant, f"{type(exc).__name__}: {exc}")) from exc
    try:
        model = model.to(dev)
    except Exception as exc:  # noqa: BLE001 - e.g. a CUDA driver too old for the wheel
        raise RuntimeError(
            f"CoTracker loaded but could not be moved to {dev}: {type(exc).__name__}: {exc}. "
            "Pass device='cpu' or run with --tracker-provider template.") from exc
    logger.info("[cotracker] model ready on %s (step=%s)", dev, getattr(model, "step", "?"))
    return model, dev


class CoTrackerTracker(PointTracker):
    """One instance per camera. See the module docstring for the window semantics."""

    name = "cotracker"

    def __init__(self, device=None, variant=None, torch_home=None, model=None, step=None,
                 vis_threshold=None, stale_decay=None, warm_start=True, **kwargs):
        super().__init__(**kwargs)
        self.torch = _import_torch()
        self.vis_threshold = float(vis_threshold if vis_threshold is not None
                                   else getattr(config, "tracker_cotracker_vis_threshold", 0.5))
        self.stale_decay = float(stale_decay if stale_decay is not None
                                 else getattr(config, "tracker_cotracker_stale_decay", 0.05))
        self.warm_start = bool(warm_start)
        if model is None:
            self._model, self.device = load_cotracker_model(variant, device, torch_home)
        else:
            # Injected predictor (tests, or a model loaded by the caller). The predictor
            # holds per-stream online state, so it must not be shared between cameras.
            self._model = model
            self.device = resolve_device(device) if device is not None else "cpu"
        self.step = int(step if step is not None else getattr(self._model, "step", 8) or 8)
        self.window = 2 * self.step

        self._buffer = []
        self._queries = None
        self._seed_points = None
        self._frame_shape = None
        self._started = False
        self._since_flush = 0
        self._stale = 0
        self._flushes = 0
        self._last_points = None
        self._last_visible = None
        self._last_scores = None
        self._obj_id = None

    # -- lifecycle ---------------------------------------------------------
    def init(self, frame, points, obj_id=None):
        rgb = self._as_rgb(frame)
        h, w = rgb.shape[:2]
        pts = self._as_points(points)
        keep = [p for p in pts if 0.0 <= float(p[0]) <= w - 1 and 0.0 <= float(p[1]) <= h - 1]
        if not keep:
            raise ValueError("CoTrackerTracker.init: no point lies inside the image")
        seed = np.asarray(keep, dtype=float)

        reseed = self.initialised
        self._reset_stream()
        self._seed_points = seed
        self._frame_shape = rgb.shape[:2]
        self._obj_id = obj_id
        self.n_points = len(seed)
        self.initialised = True
        # (t, x, y) queries, all anchored on the seed frame at local time 0.
        queries = np.concatenate([np.zeros((len(seed), 1)), seed], axis=1)
        self._queries = self.torch.as_tensor(queries, dtype=self.torch.float32,
                                             device=self.device)[None]
        self._last_points = np.array(seed, dtype=float, copy=True)
        self._last_visible = np.ones(len(seed), dtype=bool)
        self._last_scores = np.ones(len(seed), dtype=float)

        # With warm_start the sliding window is pre-filled with the seed frame, so the
        # next update already has a full window and returns a fresh prediction.
        self._buffer = [rgb] * (self.window - 1) if self.warm_start else [rgb]
        logger.info("[cotracker] %s %d point(s) on %s (obj=%s, device=%s, window=%d, step=%d, "
                    "warm_start=%s)", "re-seeded" if reseed else "seeded", self.n_points,
                    f"{w}x{h}", obj_id, self.device, self.window, self.step, self.warm_start)

    def reset(self):
        super().reset()
        self._reset_stream()
        self._seed_points = None
        self._queries = None
        self._frame_shape = None
        self._obj_id = None

    def _reset_stream(self):
        """Drop buffered frames and the model's online state (kept: the loaded weights)."""
        self._buffer = []
        self._started = False
        self._since_flush = 0
        self._stale = 0
        self._flushes = 0
        self._last_points = None
        self._last_visible = None
        self._last_scores = None

    # -- per-frame ---------------------------------------------------------
    def update(self, frame):
        if not self.initialised:
            return self._empty_result()
        rgb = self._as_rgb(frame)
        if rgb.shape[:2] != self._frame_shape:
            raise ValueError(
                f"CoTrackerTracker: frame size changed {self._frame_shape} -> {rgb.shape[:2]}; "
                "call init() again to re-seed at the new resolution")

        self._buffer.append(rgb)
        if len(self._buffer) > self.window:
            del self._buffer[:-self.window]
        self._since_flush += 1

        fresh = False
        if len(self._buffer) >= self.window and (not self._started or self._since_flush >= self.step):
            fresh = self._flush()

        if fresh:
            self._stale = 0
        else:
            self._stale += 1

        decay = max(0.0, 1.0 - self.stale_decay * self._stale)
        points = np.array(self._last_points, dtype=float, copy=True)
        visible = np.array(self._last_visible, dtype=bool, copy=True)
        scores = np.array(self._last_scores, dtype=float, copy=True) * decay
        scores[~visible] = 0.0
        return TrackResult(points=points, visible=visible, scores=scores,
                           meta={"provider": self.name, "stale_frames": int(self._stale),
                                 "fresh": bool(fresh), "device": self.device,
                                 "window": self.window, "step": self.step,
                                 "flushes": int(self._flushes)})

    # -- internals ---------------------------------------------------------
    def _flush(self):
        """Run the model on the newest ``window`` frames. Returns True on a prediction."""
        chunk = self._chunk_tensor()
        no_grad = getattr(self.torch, "no_grad", None)
        context = no_grad() if callable(no_grad) else _NullContext()
        with context:
            if not self._started:
                self._model(video_chunk=chunk, is_first_step=True, queries=self._queries,
                            grid_size=0)
                self._started = True
                logger.info("[cotracker] online state initialised with %d quer(ies) (obj=%s)",
                            self.n_points, self._obj_id)
            out = self._model(video_chunk=chunk)
        # Count from the attempt, not the outcome: the model state advances by ``step``
        # frames per call, so a call must not be retried on the very next frame.
        self._since_flush = 0
        tracks, visibility = (out if isinstance(out, (tuple, list)) else (out, None))
        if tracks is None:
            logger.debug("[cotracker] window flush produced no tracks (warm-up)")
            return False

        points = self._to_numpy(tracks)[0, -1].reshape(-1, 2).astype(float)
        vis_raw = None if visibility is None else self._to_numpy(visibility)[0, -1].reshape(-1)
        n = self.n_points
        points = points[:n]
        if len(points) < n:
            logger.warning("[cotracker] model returned %d of %d points; marking the rest lost",
                           len(points), n)
            pad = np.full((n - len(points), 2), np.nan)
            points = np.concatenate([points, pad], axis=0)

        if vis_raw is None:
            scores = np.ones(n, dtype=float)
            visible = np.ones(n, dtype=bool)
        else:
            is_mask = vis_raw.dtype == bool
            vis_raw = np.asarray(vis_raw, dtype=float)[:n]
            if len(vis_raw) < n:
                vis_raw = np.concatenate([vis_raw, np.zeros(n - len(vis_raw))])
            if is_mask:
                # Some predictor versions already threshold visibility into a bool mask;
                # there is no soft score left, so a visible point gets full confidence.
                visible = vis_raw > 0.5
                scores = visible.astype(float)
            else:
                visible = vis_raw >= self.vis_threshold
                scores = np.clip(vis_raw, 0.0, 1.0)

        h, w = self._frame_shape
        inside = (np.isfinite(points).all(axis=1)
                  & (points[:, 0] >= 0) & (points[:, 0] <= w - 1)
                  & (points[:, 1] >= 0) & (points[:, 1] <= h - 1))
        # A point that left the frame is lost, whatever the model says about visibility.
        visible = visible & inside
        scores = np.where(visible, scores, 0.0)
        points = np.where(np.isfinite(points), points,
                          self._last_points if self._last_points is not None else np.nan)

        self._last_points = points
        self._last_visible = visible
        self._last_scores = scores
        self._since_flush = 0
        self._flushes += 1
        logger.debug("[cotracker] flush %d: %d/%d visible (obj=%s)", self._flushes,
                     int(np.count_nonzero(visible)), n, self._obj_id)
        return True

    def _chunk_tensor(self):
        frames = self._buffer[-self.window:]
        stacked = np.stack(frames).astype(np.float32)          # (T, H, W, 3)
        stacked = np.transpose(stacked, (0, 3, 1, 2))           # (T, 3, H, W)
        return self.torch.as_tensor(stacked, device=self.device)[None]

    @staticmethod
    def _to_numpy(tensor):
        detach = getattr(tensor, "detach", None)
        if callable(detach):
            tensor = detach().cpu().numpy()
        return np.asarray(tensor)

    @staticmethod
    def _as_rgb(frame):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"CoTrackerTracker expects an RGB frame, got shape {np.shape(frame)}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)


class _NullContext:  # pragma: no cover - only used by torch stubs without no_grad
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
