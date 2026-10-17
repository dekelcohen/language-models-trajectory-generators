"""Point-tracker factory.

Mirrors the provider-selection style used elsewhere in the repo (a name -> constructor
lookup, lazy import so an optional dependency never breaks startup).
"""

import config

SUPPORTED = ("template", "klt", "csrt", "cotracker", "remote", "hybrid", "hybrid_remote")
# Providers whose model runs in gpu_server (the launcher starts / attaches to it).
REMOTE_PROVIDERS = ("remote", "hybrid_remote")


def uses_gpu_server(name):
    return str(name or "").lower() in REMOTE_PROVIDERS


def get_tracker(name=None, **kwargs):
    """Return a fresh :class:`providers.trackers.base.PointTracker`.

    ``name`` defaults to ``config.tracker_provider_default``. Unknown names raise, so a
    typo in ``--tracker-provider`` fails loudly instead of silently tracking nothing.
    """
    name = (name or config.tracker_provider_default).lower()
    if name == "template":
        from providers.trackers.template_tracker import TemplateTracker
        return TemplateTracker(**kwargs)
    if name == "klt":
        from providers.trackers.klt_tracker import KLTTracker
        return KLTTracker(**kwargs)
    if name == "csrt":
        from providers.trackers.csrt_tracker import CSRTTracker
        return CSRTTracker(**kwargs)
    if name == "cotracker":
        # Lazy: importing this pulls in torch, and constructing it may hit torch.hub.
        # Neither must happen unless the provider is actually selected.
        from providers.trackers.cotracker_tracker import CoTrackerTracker
        return CoTrackerTracker(**kwargs)
    if name == "remote":
        from providers.trackers.remote_tracker import RemoteTracker
        return RemoteTracker(**kwargs)
    if name in ("hybrid", "hybrid_remote"):
        # CoTracker anchor (in-process or gpu_server) + per-frame KLT bridge between flushes.
        from providers.trackers.hybrid_tracker import HybridTracker
        return HybridTracker(anchor="remote" if name == "hybrid_remote" else None, **kwargs)
    raise ValueError(f"Unknown tracker provider {name!r}. Supported: {SUPPORTED}")
