"""Multi-camera 3D lift factory.

Mirrors ``providers/trackers/factory.py``: a name -> constructor lookup with lazy imports,
so the LAPA provider's heavy optional dependencies (torch, DINOv2, the external clone)
never cost anything unless it is actually selected.
"""

import config

SUPPORTED = ("depth_fusion", "triangulate", "weighted_triangulate", "lapa", "rigid_refine")


def get_tracker3d(name=None, **kwargs):
    """Return a fresh :class:`providers.tracker3d.base.MultiCamTracker3D`.

    ``name`` defaults to ``config.tracker3d_provider_default``. Unknown names raise, so a
    typo fails loudly rather than silently falling back to the old behaviour and quietly
    invalidating an evaluation run.
    """
    name = (name or getattr(config, "tracker3d_provider_default", "depth_fusion")).lower()
    if name == "depth_fusion":
        from providers.tracker3d.depth_fusion import DepthFusionTracker3D
        return DepthFusionTracker3D(**kwargs)
    if name == "triangulate":
        from providers.tracker3d.triangulate import TriangulateTracker3D
        return TriangulateTracker3D(**kwargs)
    if name == "weighted_triangulate":
        from providers.tracker3d.weighted_triangulate import WeightedTriangulateTracker3D
        return WeightedTriangulateTracker3D(**kwargs)
    if name == "lapa":
        from providers.tracker3d.lapa_tracker3d import LapaTracker3D
        return LapaTracker3D(**kwargs)
    if name == "rigid_refine":
        # Decorator over another provider: ``base`` names the one it wraps and defaults to
        # ``triangulate``, so ``rigid_refine`` alone is "triangulation + shape constraint".
        from providers.tracker3d.rigid_refine import RigidRefineTracker3D
        return RigidRefineTracker3D(**kwargs)
    raise ValueError(f"Unknown 3D tracker provider {name!r}. Supported: {SUPPORTED}")
