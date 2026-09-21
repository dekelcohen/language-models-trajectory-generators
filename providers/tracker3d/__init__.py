"""Multi-camera 3D lift providers.

A :class:`~providers.tracker3d.base.MultiCamTracker3D` turns **all cameras' 2D tracks for
one object** into a single world point. It is the counterpart to
:class:`providers.trackers.base.PointTracker`, which handles one camera's 2D points.

Why a second abstraction instead of extending ``PointTracker``: a point tracker is
inherently per-camera and per-frame, while a 3D lift is inherently *joint* over cameras -
triangulation has nothing to do until it has seen every view. Bolting that onto the 2D ABC
would force every implementation to carry a fake "which camera am I" identity.

Providers:
  * ``depth_fusion`` - the historical behaviour: deproject each camera's points through its
    own depth buffer, then weighted-average. Depth noise at object edges is its weakness.
  * ``triangulate``  - classic multi-view DLT from 2D + calibration only. **No depth
    buffer**, which is the point: background depth bleed stops being in the loop.
  * ``weighted_triangulate`` - the same DLT with *hand-written* per-view weights: depth
    consistency against the previous accepted point (the one signal that still works at two
    cameras), tracker confidence, staleness, and the reprojection residual from 3 views up.
    The no-DINOv2, CPU-only control for LAPA's learned view weighting.
  * ``lapa``         - LAPA's learned view-weighting on top of the same triangulation.
  * ``rigid_refine`` - a *decorator* over any of the above: fits the object's known shape to
    the per-point output and replaces points that disagree with it. The only mechanism in
    the stack that can catch a bad point at exactly 2 cameras, where the reprojection
    residual is structurally blind.
"""

from providers.tracker3d.base import Lift3DResult, MultiCamTracker3D  # noqa: F401
from providers.tracker3d.factory import SUPPORTED, get_tracker3d  # noqa: F401
