"""Motion semantics from multi-point 3D tracks.

This is the module that pays for the migration from a bbox/centroid tracker to a
**multi-point** one. A single centroid can only ever give you a displacement; three or
more *corresponded* points give you an **axis**, a **plane** and a **rotation**, i.e. the
things a manipulation policy actually needs:

  * door handle  -> pull direction + hinge axis (direction *and* location)
  * drawer       -> slide direction + how perpendicular it is to the drawer front
  * free object  -> "this is not moving on a constraint, stop guessing"

Scope and contracts:

* **Self-contained.** numpy only - no session, no provider, no simulator import. The
  wiring into the monitors/report lives elsewhere (``p7-monitors``).
* **Correspondence is the caller's job.** Point index ``i`` in one camera is not
  necessarily the same physical point as index ``i`` in another (see
  ``providers/tracker3d/base.py``). Feed :meth:`MotionBuffer.push` the *seed* indices
  (``point_index``, ``-1`` = centroid fallback / no correspondence) and the buffer keeps
  each physical point in its own slot; points may appear and disappear freely.
* **Every estimator returns a residual and a confidence, never a bare vector.** A unit
  vector derived from a near-zero displacement is pure noise pointing confidently in a
  random direction, so those cases return ``direction=None`` and ``confidence=0.0``.
* **Robustness gates** (all in :class:`MotionThresholds`): at least 3 corresponded,
  non-collinear points (judged from the singular spectrum); at least
  ``min_direction_m`` (~1.5 cm) of travel before *any* direction is emitted; frames whose
  per-camera disagreement exceeds ``max_disagreement_m`` are refused by the buffer at push
  time; and ``confidence`` is driven by the fit residual **relative to** both the observed
  motion and the cloud extent, so it tracks the actual angular error instead of looking
  good on a large but noisy motion. :class:`MotionSmoother` EMA-smooths the direction and
  the (sign-ambiguous) axis with sign alignment.

Conventions match the rest of the repo: world coordinates in metres, Z-up.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

STATIC = "static"
PRISMATIC = "prismatic"
REVOLUTE = "revolute"
FREE = "free"

MOTION_KINDS = (STATIC, PRISMATIC, REVOLUTE, FREE)


def _as_list(value, digits=6):
    """JSON-serialisable copy of ``value`` (mirrors ``tracking.types._as_list``)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return round(float(arr), digits)
    return [round(float(v), digits) for v in arr.reshape(-1)]


def _round(value, digits=6):
    return None if value is None else round(float(value), digits)


def _unit(vec):
    """Unit vector, or ``None`` when the input is too short to have a direction."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return v / norm


def _log(logger, msg):
    if logger is not None:
        logger.info(msg)


# -- tunables --------------------------------------------------------------


@dataclass
class MotionThresholds:
    """Decision thresholds, all in metres / radians over the analysis window.

    ``*_enter`` is what an *incoming* label must clear; ``*_exit`` is the looser bound the
    *current* label only has to stay inside. That gap, plus ``switch_frames``, is the
    hysteresis that stops the label flickering on borderline input.
    """

    angle_enter_rad: float = 0.05        # ~2.9 deg over the window
    angle_exit_rad: float = 0.02
    disp_enter_m: float = 0.005
    disp_exit_m: float = 0.002
    #: Rigid-fit rmse above this (or above ``free_residual_frac`` of the motion) means the
    #: points did not move rigidly together -> unconstrained/unreliable.
    residual_free_m: float = 0.02
    free_residual_frac: float = 0.5
    #: A rotation is only claimed when the rigid fit explains the data this much better
    #: than a translation-only fit. The ``_exit`` variant keeps an existing revolute label.
    revolute_gain: float = 0.5
    revolute_gain_exit: float = 0.8
    #: Below this the displacement is indistinguishable from tracking jitter.
    min_motion_m: float = 0.002
    #: **Direction gate.** No direction/axis is emitted at all until the object has moved
    #: this far over the window. Between ``min_motion_m`` and here the motion is real but
    #: the *direction* derived from it is dominated by the per-frame noise: a 3 mm slide
    #: measured with 2 mm jitter points confidently in a random direction. Such a window
    #: comes back with ``confidence=0`` and ``reason="near_zero_motion"``.
    min_direction_m: float = 0.015
    #: sigma_2 / sigma_1 of the centred point cloud; below this the points are collinear
    #: and no plane (and no reliable rotation) can be recovered from them.
    collinear_ratio: float = 0.02
    #: Absolute floor for the robust outlier threshold, so an exact fit (residuals ~1e-16)
    #: does not reject its own points.
    outlier_floor_m: float = 1e-4
    outlier_sigma: float = 3.0
    #: rmse this fraction of the motion scale drives confidence to 0. Bounds the error of
    #: the *direction*: the residual is what the motion vector could not explain.
    residual_frac: float = 0.5
    #: rmse this fraction of the cloud extent (sigma_1) drives confidence to 0. Bounds the
    #: error of the *axis*: tilting an axis by ``phi`` moves the outermost point by
    #: ``phi * sigma_1``, so ``phi ~ rmse / sigma_1`` is the angle the data cannot resolve.
    #: Calibrated so 4 mm point noise on a ~12 cm cloud (measured axis error ~6 deg) lands
    #: near 0.55 rather than the 0.91 a residual-only model produced.
    cond_residual_frac: float = 0.15
    #: Points needed before the point-count term saturates.
    full_points: int = 4
    #: Consecutive frames a new label must win before it is adopted.
    switch_frames: int = 2
    #: EMA weight of the newest direction sample in :class:`DirectionSmoother`.
    direction_ema_alpha: float = 0.4
    #: Frames whose per-camera 3D disagreement exceeds this are refused by
    #: :meth:`MotionBuffer.push`. Mirrors ``config.track_disagree_m``; kept as a local
    #: tunable because this module must stay importable without the session/config.
    max_disagreement_m: float = 0.08


DEFAULT_THRESHOLDS = MotionThresholds()


# -- result types ----------------------------------------------------------


@dataclass
class RigidFit:
    """Output of :func:`fit_rigid_motion`. Unpacks as ``R, t, rmse``."""

    R: np.ndarray
    t: np.ndarray
    rmse: float
    inliers: np.ndarray                 # (N,) bool over the *input* rows
    n_used: int

    def __iter__(self):
        yield self.R
        yield self.t
        yield self.rmse

    def apply(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ self.R.T + self.t

    def to_dict(self):
        return {"R": [_as_list(row) for row in self.R], "t": _as_list(self.t),
                "rmse": _round(self.rmse), "n_used": int(self.n_used)}


@dataclass
class ScrewMotion:
    """Output of :func:`screw_decompose`."""

    angle_rad: float
    axis_dir: Optional[np.ndarray] = None       # unit; ``None`` for pure translation
    axis_point: Optional[np.ndarray] = None     # a point on the axis line
    translation: Optional[np.ndarray] = None    # the full t of the rigid transform
    axis_translation: float = 0.0               # screw pitch: metres along ``axis_dir``
    residual: float = 0.0                       # ||(I - R) c - t_perp||

    @property
    def is_rotation(self) -> bool:
        return self.axis_dir is not None

    def to_dict(self):
        return {
            "angle_rad": _round(self.angle_rad),
            "axis_dir": _as_list(self.axis_dir),
            "axis_point": _as_list(self.axis_point),
            "axis_translation": _round(self.axis_translation),
            "residual": _round(self.residual),
        }


@dataclass
class PlaneFit:
    """Output of :func:`fit_plane`."""

    normal: Optional[np.ndarray] = None
    point: Optional[np.ndarray] = None
    residual: float = 0.0
    confidence: float = 0.0
    n_points: int = 0

    def to_dict(self):
        return {"normal": _as_list(self.normal), "point": _as_list(self.point),
                "residual": _round(self.residual), "confidence": _round(self.confidence, 4),
                "n_points": int(self.n_points)}


@dataclass
class LineFit:
    """Output of :func:`centroid_direction` - a total-least-squares trajectory line."""

    direction: Optional[np.ndarray] = None
    point: Optional[np.ndarray] = None
    speed: float = 0.0              # metres per unit time along ``direction``
    travel: float = 0.0             # end-to-end distance covered in the window
    residual: float = 0.0
    confidence: float = 0.0
    n_frames: int = 0

    def to_dict(self):
        return {"direction": _as_list(self.direction), "point": _as_list(self.point),
                "speed": _round(self.speed), "travel": _round(self.travel),
                "residual": _round(self.residual), "confidence": _round(self.confidence, 4),
                "n_frames": int(self.n_frames)}


@dataclass
class MotionEstimate:
    """What one analysis window says about an object's motion.

    The two headline outputs a planner consumes:

    * **Door handle** - ``axis_dir``/``axis_point`` are the *hinge line*, so the whole arc
      can be predicted instead of chased, and ``direction`` is the pull direction
      ``omega_hat x (p - p_axis)`` evaluated at ``query_point`` (the tracked handle when
      the caller passes one, otherwise the centroid). Use :meth:`pull_direction_at` to
      re-evaluate it anywhere else on the body - the two differ, which is precisely why
      the handle must be named rather than assumed to be the centroid.
    * **Drawer** - ``perpendicularity`` = ``|direction . plane_normal|`` (1.0 = pulling
      straight out of the face) and ``perpendicularity_deg`` = the same thing as an angle
      between the pull direction and the surface normal (0 deg = straight out, 90 deg =
      sliding along the face).

    ``confidence == 0`` means *do not use the geometry*: too few points, collinear points,
    or a displacement small enough that any direction derived from it would be noise. In
    that case ``direction``/``axis_dir`` are ``None`` rather than a plausible-looking lie,
    and ``reason`` names the gate that fired. ``kind`` is still reported (a confident
    ``static`` reads as ``kind=static, confidence=0, reason=near_zero_motion``: nothing
    moved, so there is no geometry to be confident *about*).
    """

    kind: str = STATIC
    direction: Optional[np.ndarray] = None      # unit; tangential direction for revolute
    speed: float = 0.0                          # metres per unit time (see ``dt``)
    axis_dir: Optional[np.ndarray] = None
    axis_point: Optional[np.ndarray] = None
    angle_rad: float = 0.0
    displacement: float = 0.0                   # centroid travel over the window
    plane_normal: Optional[np.ndarray] = None
    perpendicularity: Optional[float] = None    # |direction . plane_normal|, 1 = straight out
    perpendicularity_deg: Optional[float] = None  # angle to the normal, 0 deg = straight out
    query_point: Optional[np.ndarray] = None    # where ``direction`` is evaluated
    confidence: float = 0.0
    n_points: int = 0
    window_frames: int = 0
    residual: float = 0.0                       # rigid-fit rmse, metres
    reason: str = ""                            # diagnostic; why confidence was reduced
    plane: Optional[PlaneFit] = field(default=None, repr=False)
    line: Optional[LineFit] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.confidence > 0.0

    @property
    def hinge(self):
        """``(axis_dir, axis_point)`` of the hinge, or ``None`` when there is no rotation."""
        if self.axis_dir is None or self.axis_point is None:
            return None
        return np.asarray(self.axis_dir, dtype=float), np.asarray(self.axis_point, dtype=float)

    def pull_direction_at(self, point):
        """Pull direction at an arbitrary body point (the handle the gripper will grasp).

        For a revolute joint this is ``omega_hat x (p - p_axis)``, which genuinely depends
        on ``p``; for a slide it is the (single) slide direction. ``None`` whenever the
        estimate is not trustworthy, so a caller can never get a confident-looking vector
        out of a rejected window.
        """
        if self.confidence <= 0.0:
            return None
        if self.axis_dir is not None and self.axis_point is not None:
            return pull_direction(self.axis_dir, self.axis_point, point)
        return None if self.direction is None else _unit(self.direction)

    def to_dict(self):
        return {
            "kind": self.kind,
            "direction": _as_list(self.direction),
            "speed": _round(self.speed),
            "axis_dir": _as_list(self.axis_dir),
            "axis_point": _as_list(self.axis_point),
            "angle_rad": _round(self.angle_rad),
            "displacement": _round(self.displacement),
            "plane_normal": _as_list(self.plane_normal),
            "perpendicularity": _round(self.perpendicularity, 4),
            "perpendicularity_deg": _round(self.perpendicularity_deg, 2),
            "query_point": _as_list(self.query_point),
            "confidence": _round(self.confidence, 4),
            "n_points": int(self.n_points),
            "window_frames": int(self.window_frames),
            "residual": _round(self.residual),
            **({"reason": self.reason} if self.reason else {}),
        }


# -- ring buffer -----------------------------------------------------------


class MotionBuffer:
    """Last ``capacity`` frames of fused 3D points, ``(K, N, 3)`` + an ``(K, N)`` mask.

    Slot ``j`` is one physical point across time, so a point that disappears for a few
    frames and comes back lands in its own slot again - provided the caller passes the
    seed ``point_index``. Missing samples are ``nan`` and masked ``False``; nothing in this
    module ever reads a masked-out entry.
    """

    def __init__(self, capacity=12, n_points=0, logger=None, max_disagreement=None,
                 thresholds=None):
        self.capacity = int(max(2, capacity))
        self.logger = logger
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        #: Frames whose per-camera 3D disagreement exceeds this are refused outright.
        #: ``None`` disables the gate; pass ``config.track_disagree_m`` from the session.
        self.max_disagreement = None if max_disagreement is None else float(max_disagreement)
        self.rejected_frames = 0
        self._rejecting = False
        n = max(int(n_points), 0)
        self._points = np.full((self.capacity, n, 3), np.nan, dtype=float)
        self._mask = np.zeros((self.capacity, n), dtype=bool)
        self._frames = np.zeros(self.capacity, dtype=float)
        self._count = 0

    # -- state ------------------------------------------------------------
    def __len__(self):
        return int(min(self._count, self.capacity))

    @property
    def n_points(self) -> int:
        return int(self._points.shape[1])

    @property
    def frames_seen(self) -> int:
        return int(self._count)

    def reset(self):
        self._points[:] = np.nan
        self._mask[:] = False
        self._frames[:] = 0.0
        self._count = 0
        self.rejected_frames = 0
        self._rejecting = False

    def _grow(self, n):
        extra = int(n) - self.n_points
        if extra <= 0:
            return
        self._points = np.concatenate(
            [self._points, np.full((self.capacity, extra, 3), np.nan)], axis=1)
        self._mask = np.concatenate(
            [self._mask, np.zeros((self.capacity, extra), dtype=bool)], axis=1)

    # -- writing ----------------------------------------------------------
    def accepts(self, disagreement=None):
        """``True`` when a frame with this per-camera disagreement may be stored.

        A frame whose cameras disagree by more than ``max_disagreement`` metres is not a
        noisy measurement of the object, it is a *different* object (or a bad match) in at
        least one view. Fitting it into the rigid motion drags the axis with it, so the
        frame is refused at push time rather than down-weighted later.
        """
        if self.max_disagreement is None or disagreement is None:
            return True
        d = float(disagreement)
        return not np.isfinite(d) or d <= self.max_disagreement

    def _refuse(self, disagreement, frame_idx):
        self.rejected_frames += 1
        if not self._rejecting:
            # Log the onset of a bad run only - one line per run, never per frame.
            _log(self.logger,
                 f"[motion] buffer rejected frame {frame_idx}: disagreement="
                 f"{float(disagreement):.4f}m > {self.max_disagreement:.4f}m")
        self._rejecting = True
        return self

    def push(self, points, mask=None, point_index=None, frame_idx=None, disagreement=None):
        """Add one frame.

        ``points`` is ``(M, 3)``. ``mask`` is an optional ``(M,)`` per-point validity flag
        (non-finite rows are dropped regardless). ``point_index`` is the ``(M,)`` seed
        index of each row - rows with ``-1`` have no correspondence and are ignored, since
        a centroid fallback cannot be tracked as a physical point. ``disagreement`` is the
        fused per-camera spread in metres (``TrackedObjectState.disagreement``); the frame
        is **dropped** when it exceeds ``max_disagreement``.
        """
        if not self.accepts(disagreement):
            return self._refuse(disagreement, frame_idx)
        self._rejecting = False

        pts = np.asarray(points, dtype=float).reshape(-1, 3) if points is not None \
            else np.zeros((0, 3))
        m = pts.shape[0]

        valid = np.isfinite(pts).all(axis=1)
        if mask is not None and m:
            valid &= np.asarray(mask, dtype=bool).reshape(-1)[:m]

        if point_index is not None and m:
            slots = np.asarray(point_index, dtype=int).reshape(-1)[:m]
            valid &= slots >= 0
        else:
            slots = np.arange(m, dtype=int)

        row = self._count % self.capacity
        self._points[row] = np.nan
        self._mask[row] = False
        known = slots[slots >= 0] if m else slots
        if known.size:
            # Reserve a slot for every point the caller knows about, even if it is invalid
            # this frame, so the mask width is the object's point count and not "whatever
            # happened to be visible first".
            self._grow(int(known.max()) + 1)
        if np.any(valid):
            self._points[row, slots[valid]] = pts[valid]
            self._mask[row, slots[valid]] = True
        self._frames[row] = float(self._count if frame_idx is None else frame_idx)
        self._count += 1
        return self

    def push_lift(self, result, frame_idx=None, disagreement=None):
        """Push a ``Lift3DResult``-shaped object (duck-typed, no provider import).

        Falls back to the fused ``world_point`` in slot 0 when the provider produced no
        per-point cloud, which keeps the buffer usable (translation only) with a
        centroid-style provider.
        """
        if disagreement is None:
            disagreement = getattr(result, "disagreement", None)
        per_point = getattr(result, "per_point", None)
        if per_point is None:
            world = getattr(result, "world_point", None)
            if world is None:
                return self.push(np.zeros((0, 3)), frame_idx=frame_idx,
                                 disagreement=disagreement)
            return self.push(np.asarray(world, dtype=float).reshape(1, 3), frame_idx=frame_idx,
                             disagreement=disagreement)
        return self.push(per_point, point_index=getattr(result, "per_point_index", None),
                         frame_idx=frame_idx, disagreement=disagreement)

    def push_state(self, state, points=None, point_index=None, frame_idx=None):
        """Push a ``TrackedObjectState``-shaped object (duck-typed).

        Applies both frame-level gates the state carries: a ``lost`` frame and a frame
        whose cameras disagree by more than ``max_disagreement`` are refused. ``points``
        overrides the geometry (a per-point cloud from the 3D provider); without it the
        fused ``world_point`` goes into slot 0.
        """
        disagreement = getattr(state, "disagreement", None)
        if getattr(state, "lost", False):
            self.rejected_frames += 1
            if not self._rejecting:
                _log(self.logger, f"[motion] buffer rejected frame {frame_idx}: target lost")
            self._rejecting = True
            return self
        if points is None:
            world = getattr(state, "world_point", None)
            points = np.zeros((0, 3)) if world is None else \
                np.asarray(world, dtype=float).reshape(1, 3)
        return self.push(points, point_index=point_index, frame_idx=frame_idx,
                         disagreement=disagreement)

    # -- reading ----------------------------------------------------------
    def _rows(self, window=None):
        have = len(self)
        k = have if window is None else int(min(max(window, 0), have))
        start = self._count - k
        return [(start + i) % self.capacity for i in range(k)]

    def window(self, window=None):
        """Oldest-first ``(points (K, N, 3), mask (K, N), frames (K,))``."""
        rows = self._rows(window)
        if not rows:
            return (np.zeros((0, self.n_points, 3)), np.zeros((0, self.n_points), dtype=bool),
                    np.zeros(0))
        return self._points[rows].copy(), self._mask[rows].copy(), self._frames[rows].copy()

    def common_valid(self, window=None):
        """Slots that are valid in **every** frame of the window (the corresponded set)."""
        _pts, mask, _frames = self.window(window)
        if mask.shape[0] == 0:
            return np.zeros(self.n_points, dtype=bool)
        return mask.all(axis=0)

    def latest(self):
        """``(points (n, 3), slots (n,))`` of the most recent frame's valid points."""
        if len(self) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=int)
        row = (self._count - 1) % self.capacity
        keep = self._mask[row]
        return self._points[row][keep], np.nonzero(keep)[0]

    def centroids(self, window=None):
        """Per-frame centroid of the valid points -> ``(C (K, 3), frames (K,), counts (K,))``.

        Frames with no valid point are dropped, so a dropout does not inject a nan into a
        line fit.
        """
        pts, mask, frames = self.window(window)
        if pts.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
        counts = mask.sum(axis=1)
        keep = counts > 0
        filled = np.where(mask[..., None], np.nan_to_num(pts), 0.0)
        sums = filled.sum(axis=1)
        cents = np.divide(sums, np.maximum(counts, 1)[:, None])
        return cents[keep], frames[keep], counts[keep].astype(int)


# -- geometry primitives ---------------------------------------------------


def point_spread(points):
    """Singular values of the centred point cloud, descending ``(3,)``.

    The spectrum - not an ad-hoc cross product - is how collinearity and coincidence are
    detected: ``s[1] << s[0]`` is a line, ``s[2] << s[1]`` is a plane.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.zeros(3)
    centred = pts - pts.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    out = np.zeros(3)
    out[:s.shape[0]] = s
    return out


def is_collinear(points, ratio=DEFAULT_THRESHOLDS.collinear_ratio):
    """``True`` when the points span (at most) a line, so no rotation/plane is observable."""
    s = point_spread(points)
    if s[0] <= 1e-12:
        return True
    return bool(s[1] <= ratio * s[0])


def _kabsch(P0, P1, weights=None):
    """Least-squares rotation+translation mapping ``P0`` onto ``P1`` (no outlier handling).

    The ``det(R) < 0`` correction is the classic Kabsch trap: the raw ``V U^T`` is the best
    *orthogonal* matrix, which for near-degenerate or mirrored data is a **reflection**.
    Flipping the sign of the last singular vector yields the best matrix in ``SO(3)``.
    """
    A = np.asarray(P0, dtype=float).reshape(-1, 3)
    B = np.asarray(P1, dtype=float).reshape(-1, 3)
    if weights is None:
        w = np.ones(A.shape[0])
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        w = np.ones(A.shape[0])
        total = float(w.sum())

    mu_a = (w[:, None] * A).sum(axis=0) / total
    mu_b = (w[:, None] * B).sum(axis=0) / total
    H = (w[:, None] * (A - mu_a)).T @ (B - mu_b)

    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0.0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_b - R @ mu_a
    return R, t


def fit_rigid_motion(P0, P1, weights=None, thresholds=None, max_iter=4, min_points=3,
                     logger=None):
    """Robust Kabsch/Umeyama fit: ``R, t, rmse`` mapping ``P0`` onto ``P1``.

    Iteratively fits, measures per-point residuals, drops the single worst point when it
    exceeds a robust (median + ``sigma`` * 1.4826 * MAD, floored) threshold, and refits.
    One bad correspondence is enough to rotate the whole estimate, and a plain mean
    residual cannot see it - the median/MAD can.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    A = np.asarray(P0, dtype=float).reshape(-1, 3)
    B = np.asarray(P1, dtype=float).reshape(-1, 3)
    if A.shape != B.shape:
        raise ValueError(f"P0 {A.shape} and P1 {B.shape} must have the same shape")

    n = A.shape[0]
    keep = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    if int(keep.sum()) < min_points:
        return RigidFit(np.eye(3), np.zeros(3), float("inf"), keep, int(keep.sum()))

    R, t = np.eye(3), np.zeros(3)
    rmse = float("inf")
    for _ in range(max(1, int(max_iter))):
        w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[keep]
        R, t = _kabsch(A[keep], B[keep], w)
        resid = np.linalg.norm(A[keep] @ R.T + t - B[keep], axis=1)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        if int(keep.sum()) <= min_points:
            break
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        limit = max(med + th.outlier_sigma * 1.4826 * mad, th.outlier_floor_m)
        worst = int(np.argmax(resid))
        if resid[worst] <= limit:
            break
        idx = np.nonzero(keep)[0][worst]
        keep[idx] = False

    n_used = int(keep.sum())
    if n_used < n and logger is not None:
        _log(logger, f"[motion] rigid fit dropped {n - n_used}/{n} point(s), rmse={rmse:.5f}m")
    return RigidFit(R, t, rmse, keep, n_used)


def rotation_angle_axis(R, eps=1e-9):
    """``(theta, n_hat)`` from a rotation matrix; ``n_hat`` is ``None`` when ``theta ~ 0``.

    ``theta`` is always in ``[0, pi]`` and ``n_hat`` carries the sense of rotation
    (right-handed about ``n_hat``).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < eps:
        return 0.0, None

    sin_theta = np.sin(theta)
    if sin_theta > 1e-6:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return theta, axis / (2.0 * sin_theta)

    # theta ~ pi: the skew part vanishes, but R + I = 2 n n^T, so any non-degenerate
    # column of it is parallel to the axis. The sign is genuinely ambiguous at pi.
    M = R + np.eye(3)
    col = int(np.argmax(np.linalg.norm(M, axis=0)))
    axis = _unit(M[:, col])
    return theta, axis


def screw_decompose(R, t, ref_point=None, eps=1e-9):
    """Rigid transform -> :class:`ScrewMotion` (angle, axis direction, axis *location*).

    A rigid motion is a rotation about a line plus a translation along it. Splitting ``t``
    into its component along the axis (the screw pitch) and perpendicular to it leaves
    ``(I - R) c = t_perp`` for any point ``c`` on the axis. ``(I - R)`` is singular along
    the axis, so this is solved with ``lstsq``: the minimum-norm solution is the axis point
    closest to the origin.

    ``ref_point`` (use the tracked centroid) re-parametrises that line and returns the axis
    point closest to it instead - far more useful than the origin's foot point, which for a
    hinge a metre away from the world origin lands nowhere near the object.

    Pure translation (``theta -> 0``) has **no axis**: ``axis_dir``/``axis_point`` come back
    ``None`` rather than the garbage a pseudo-inverse would happily produce.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)

    theta, n_hat = rotation_angle_axis(R, eps=eps)
    if n_hat is None:
        return ScrewMotion(angle_rad=0.0, axis_dir=None, axis_point=None, translation=t,
                           axis_translation=float(np.linalg.norm(t)), residual=0.0)

    pitch = float(n_hat @ t)
    t_perp = t - pitch * n_hat
    A = np.eye(3) - R
    c, *_ = np.linalg.lstsq(A, t_perp, rcond=None)
    residual = float(np.linalg.norm(A @ c - t_perp))

    if ref_point is not None:
        ref = np.asarray(ref_point, dtype=float).reshape(3)
        c = c + float((ref - c) @ n_hat) * n_hat

    return ScrewMotion(angle_rad=theta, axis_dir=n_hat, axis_point=c, translation=t,
                       axis_translation=pitch, residual=residual)


def pull_direction(axis_dir, axis_point, query_point):
    """Instantaneous pull direction at ``query_point``: ``omega_hat x (p - p_axis)``.

    This is the door-handle answer. It is **not** the centroid velocity: the centroid's
    frame-to-frame step is the *chord* of an arc (off by half the swept angle) and, worse,
    it is a single vector for the whole body, whereas every point of a rotating body moves
    in a different direction. The gripper grasps the handle, so the direction has to be
    evaluated at the handle.

    Returns ``None`` when there is no axis, or when the query point lies *on* the axis -
    a point on the hinge line does not move, so it has no pull direction to report.
    """
    if axis_dir is None or axis_point is None or query_point is None:
        return None
    n_hat = _unit(axis_dir)
    if n_hat is None:
        return None
    p = np.asarray(query_point, dtype=float).reshape(3)
    radius = p - np.asarray(axis_point, dtype=float).reshape(3)
    radius = radius - float(radius @ n_hat) * n_hat
    return _unit(np.cross(n_hat, radius))


def fit_plane(points, view_point=None, view_dir=None, thresholds=None):
    """Least-squares plane -> :class:`PlaneFit` (normal, point on the plane, residual).

    The normal is the smallest right singular vector of the centred cloud. Its sign is
    meaningless on its own, so it is disambiguated to point **towards the observer**: give
    either ``view_point`` (camera position) or ``view_dir`` (the direction the camera
    looks). Without one of those the sign is left as SVD produced it.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return PlaneFit(n_points=int(pts.shape[0]))

    centre = pts.mean(axis=0)
    _U, s, Vt = np.linalg.svd(pts - centre)
    if s[0] <= 1e-12 or s[1] <= th.collinear_ratio * s[0]:
        return PlaneFit(point=centre, n_points=int(pts.shape[0]))

    normal = Vt[2]
    normal = normal / float(np.linalg.norm(normal))

    if view_point is not None:
        towards = np.asarray(view_point, dtype=float).reshape(3) - centre
        if float(normal @ towards) < 0.0:
            normal = -normal
    elif view_dir is not None:
        if float(normal @ np.asarray(view_dir, dtype=float).reshape(3)) > 0.0:
            normal = -normal

    residual = float(np.sqrt(np.mean(((pts - centre) @ normal) ** 2)))
    # A cloud that is flat (s[2] << s[1]) gives a trustworthy normal; a fat blob does not.
    flatness = 1.0 - float(s[2] / max(s[1], 1e-12))
    confidence = float(np.clip(flatness, 0.0, 1.0))
    return PlaneFit(normal=normal, point=centre, residual=residual, confidence=confidence,
                    n_points=int(pts.shape[0]))


def centroid_direction(buffer, window=None, dt=1.0, thresholds=None):
    """Total-least-squares (PCA) line fit through the centroid trajectory.

    Why not frame-to-frame differencing: a difference uses exactly two samples, so the full
    per-frame tracking jitter ``sigma`` enters the direction divided by a one-frame
    baseline; over ``K`` frames the line fit averages that jitter down to roughly
    ``sigma / (sqrt(K) * span)``, and it degrades gracefully when one frame is bad instead
    of producing a wild direction for that frame.

    ``buffer`` may be a :class:`MotionBuffer` or a raw ``(K, 3)`` array of centroids.
    """
    th = thresholds or DEFAULT_THRESHOLDS

    if hasattr(buffer, "centroids"):
        cents, frames, _counts = buffer.centroids(window)
    else:
        cents = np.asarray(buffer, dtype=float).reshape(-1, 3)
        if window is not None:
            cents = cents[-int(window):]
        cents = cents[np.isfinite(cents).all(axis=1)]
        frames = np.arange(cents.shape[0], dtype=float)

    k = int(cents.shape[0])
    if k < 2:
        return LineFit(n_frames=k)

    travel = float(np.linalg.norm(cents[-1] - cents[0]))
    centre = cents.mean(axis=0)
    _U, s, Vt = np.linalg.svd(cents - centre)
    direction = Vt[0] / float(np.linalg.norm(Vt[0]))
    if float(direction @ (cents[-1] - cents[0])) < 0.0:
        direction = -direction

    proj = (cents - centre) @ direction
    times = (frames - frames[0]) * float(dt)
    span = float(times[-1] - times[0])
    if span > 1e-12:
        # Slope of the projection against time: a regression, not a two-sample difference.
        slope = float(np.polyfit(times, proj, 1)[0])
    else:
        slope = 0.0

    perp = (cents - centre) - proj[:, None] * direction
    residual = float(np.sqrt(np.mean((perp ** 2).sum(axis=1))))

    if travel < th.min_motion_m:
        # Near-zero travel: the "direction" would be the principal axis of the jitter.
        return LineFit(direction=None, point=centre, speed=0.0, travel=travel,
                       residual=residual, confidence=0.0, n_frames=k)

    ramp = float(np.clip((travel - th.min_motion_m) / max(th.min_motion_m, 1e-12), 0.0, 1.0))
    straight = float(np.clip(1.0 - residual / max(th.residual_frac * travel, 1e-12), 0.0, 1.0))
    return LineFit(direction=direction, point=centre, speed=slope, travel=travel,
                   residual=residual, confidence=ramp * straight, n_frames=k)


# -- classification --------------------------------------------------------


def classify_motion(angle_rad, displacement, rigid_residual, prismatic_residual,
                    motion_scale=None, previous=None, thresholds=None):
    """``static | prismatic | revolute | free`` for one window.

    Stateless. ``previous`` (the label currently held) selects the looser ``*_exit``
    thresholds, so a label that is already established survives borderline frames; the
    frame-count dwell lives in :class:`MotionClassifier`.

    ``motion_scale`` is how far the *points* moved (mean over correspondences). It differs
    from ``displacement`` (centroid travel) for a rotation about the centroid, where the
    centroid does not move at all but every point does.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    scale = float(displacement if motion_scale is None else motion_scale)
    angle = float(angle_rad)
    rigid_residual = float(rigid_residual)
    prismatic_residual = float(prismatic_residual)

    holding = previous in MOTION_KINDS
    angle_gate = th.angle_exit_rad if (holding and previous == REVOLUTE) else th.angle_enter_rad
    move_gate = th.disp_exit_m if (holding and previous != STATIC) else th.disp_enter_m
    gain = th.revolute_gain_exit if (holding and previous == REVOLUTE) else th.revolute_gain

    if scale < move_gate and angle < angle_gate:
        return STATIC
    if not np.isfinite(rigid_residual):
        return FREE
    if rigid_residual > max(th.residual_free_m, th.free_residual_frac * scale):
        return FREE
    if angle >= angle_gate and rigid_residual <= gain * prismatic_residual:
        return REVOLUTE
    return PRISMATIC


class MotionClassifier:
    """:func:`classify_motion` plus frame hysteresis and transition logging.

    A candidate label must win ``switch_frames`` consecutive windows before it is adopted.
    Combined with the enter/exit threshold gap this keeps a door that is being opened from
    oscillating ``revolute``/``prismatic`` while the gripper jitters.
    """

    def __init__(self, thresholds=None, switch_frames=None, logger=None, label=STATIC):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.switch_frames = int(self.thresholds.switch_frames if switch_frames is None
                                 else switch_frames)
        self.logger = logger
        self.label = label
        self.candidate = label
        self.streak = 0
        self.transitions = 0
        self._last_reason = None

    def reset(self, label=STATIC):
        self.label = label
        self.candidate = label
        self.streak = 0
        self._last_reason = None

    def update(self, angle_rad, displacement, rigid_residual, prismatic_residual,
               motion_scale=None):
        candidate = classify_motion(angle_rad, displacement, rigid_residual,
                                    prismatic_residual, motion_scale=motion_scale,
                                    previous=self.label, thresholds=self.thresholds)
        if candidate == self.label:
            self.candidate, self.streak = candidate, 0
            return self.label

        if candidate == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = candidate, 1

        if self.streak >= self.switch_frames:
            scale = displacement if motion_scale is None else motion_scale
            _log(self.logger,
                 f"[motion] {self.label} -> {candidate} "
                 f"(angle={float(angle_rad):.4f}rad, scale={float(scale):.4f}m, "
                 f"rigid_rmse={float(rigid_residual):.5f}m, "
                 f"prismatic_rmse={float(prismatic_residual):.5f}m)")
            self.label = candidate
            self.streak = 0
            self.transitions += 1
        return self.label

    def note_rejection(self, reason, detail=""):
        """Log a low-confidence rejection once per distinct reason (never per frame)."""
        if reason and reason != self._last_reason:
            _log(self.logger, f"[motion] low confidence: {reason} {detail}".rstrip())
        self._last_reason = reason

    def clear_rejection(self):
        self._last_reason = None


# -- temporal smoothing ----------------------------------------------------


class DirectionSmoother:
    """Sign-aligned exponential moving average of a direction or axis.

    **Why the sign alignment is not optional.** An axis direction is inherently
    sign-ambiguous: ``n`` and ``-n`` describe the same line, and both the SVD in
    :func:`fit_plane` and the eigen branch of :func:`rotation_angle_axis` hand back
    whichever sign the solver happened to land on. Averaging the raw samples then
    *cancels* them - ``mean(n, -n, n, -n) ~ 0`` - and the "smoothed" axis is a tiny,
    numerically meaningless vector pointing nowhere. Each incoming sample is therefore
    flipped into the hemisphere of the running mean (``dot < 0`` -> negate) *before* it is
    blended.

    For a true motion direction the sign is meaningful, so a genuine reversal (a door
    being closed instead of opened) would be silently folded onto the old direction. Call
    :meth:`reset` when the classification changes - that is the event which means "this is
    a different motion now".
    """

    def __init__(self, alpha=None, thresholds=None, logger=None, name="direction"):
        th = thresholds or DEFAULT_THRESHOLDS
        self.alpha = float(th.direction_ema_alpha if alpha is None else alpha)
        self.logger = logger
        self.name = name
        self.value = None
        self.n_updates = 0
        self.flips = 0

    def reset(self):
        self.value = None
        self.n_updates = 0
        self.flips = 0

    def update(self, vec):
        """Blend one sample in and return the smoothed unit vector (or ``None``)."""
        v = _unit(vec)
        if v is None:
            return self.value
        if self.value is None:
            self.value = v
            self.n_updates = 1
            return self.value

        dot = float(v @ self.value)
        if dot < 0.0:
            v = -v
            self.flips += 1
            if self.flips == 1:
                _log(self.logger,
                     f"[motion] {self.name}: sign-flipped sample (dot={dot:.3f}) onto the "
                     f"running mean; averaging it raw would cancel the estimate")
        blended = _unit(self.alpha * v + (1.0 - self.alpha) * self.value)
        if blended is not None:
            self.value = blended
        self.n_updates += 1
        return self.value


class MotionSmoother:
    """The pair of :class:`DirectionSmoother` an estimate needs (motion + axis)."""

    def __init__(self, alpha=None, thresholds=None, logger=None):
        self.direction = DirectionSmoother(alpha, thresholds, logger, name="direction")
        self.axis = DirectionSmoother(alpha, thresholds, logger, name="axis")
        self._kind = None

    def reset(self):
        self.direction.reset()
        self.axis.reset()
        self._kind = None

    def apply(self, estimate):
        """Smooth ``estimate`` in place. Rejected windows are never blended in."""
        if estimate is None or estimate.confidence <= 0.0:
            return estimate
        if self._kind is not None and estimate.kind != self._kind:
            # A different motion: the history describes something else.
            self.reset()
        self._kind = estimate.kind
        if estimate.direction is not None:
            estimate.direction = self.direction.update(estimate.direction)
        if estimate.axis_dir is not None:
            estimate.axis_dir = self.axis.update(estimate.axis_dir)
        if estimate.direction is not None and estimate.plane_normal is not None:
            estimate.perpendicularity, estimate.perpendicularity_deg = _perpendicularity(
                estimate.direction, estimate.plane_normal)
        return estimate


# -- the composed estimate -------------------------------------------------


def _perpendicularity(direction, normal):
    """``(|d . n|, angle-to-normal in degrees)`` - the drawer answer.

    The absolute value is deliberate: the plane normal's sign is a *viewing* convention
    (it is flipped to face the camera), while the pull direction's sign is physical, so a
    drawer pulled straight out of a face whose normal happens to point inwards must still
    read 1.0. 1.0 / 0 deg = straight out of the face, 0.0 / 90 deg = sliding along it.
    """
    d = _unit(direction)
    n = _unit(normal)
    if d is None or n is None:
        return None, None
    dot = float(np.clip(abs(d @ n), 0.0, 1.0))
    return dot, float(np.degrees(np.arccos(dot)))


def _window_plane(pts, mask, R, t, view_point=None, view_dir=None, thresholds=None,
                  logger=None):
    """Surface plane in the **latest** pose, with a documented fallback for dropouts.

    The normal has to describe the surface as it is *now*, because that is the pose the
    direction is reported in, so the primary source is the latest frame's points. When a
    point was visible at the start of the window but has dropped out by its end, it is
    carried forward through the fitted rigid transform (``p1 = R p0 + t``, exact for a
    rigid body) instead of being thrown away - a drawer front that loses half its points
    to occlusion still gets a normal fitted on the full set.

    If even that leaves fewer than three non-collinear points, ``normal`` stays ``None``
    and the caller reports ``perpendicularity=None`` rather than inventing a surface.
    """
    cloud = pts[-1][mask[-1]]
    carried = mask[0] & ~mask[-1]
    n_carried = int(carried.sum())
    if n_carried:
        mapped = pts[0][carried] @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)
        cloud = np.vstack([cloud, mapped])
    plane = fit_plane(cloud, view_point=view_point, view_dir=view_dir, thresholds=thresholds)
    if n_carried and plane.normal is not None:
        _log(logger, f"[motion] plane fit carried {n_carried} dropped point(s) forward "
                     f"through the fitted rigid transform ({plane.n_points} points total)")
    return plane


def _confidence(motion_scale, residual, n_used, spread, th):
    """Confidence in the *emitted geometry*, in ``[0, 1]``.

    Driven by the fit residual **relative to** the two scales that actually bound the
    angular error - a raw residual says nothing on its own:

    * ``rmse / motion_scale`` - 1 mm of residual on a 2 mm motion is garbage; the same
      1 mm on a 200 mm motion is excellent. This bounds the *direction* error.
    * ``rmse / sigma_1`` (cloud extent) - tilting the axis by ``phi`` moves the outermost
      point by ``phi * sigma_1``, so ``phi ~ rmse / sigma_1`` is exactly the angular error
      the data cannot resolve. This bounds the *axis* error, and it is the term the
      previous residual-only model was missing: it reported 0.91 at 4 mm point noise where
      the measured axis error was ~6 deg.

    Multiplied by the point count (more correspondences average the noise down) and by the
    conditioning of the cloud (near-collinear points cannot pin an axis at all).
    """
    if motion_scale < th.min_direction_m or not np.isfinite(residual):
        return 0.0
    ramp = float(np.clip((motion_scale - th.min_direction_m) / max(th.min_direction_m, 1e-12),
                         0.0, 1.0))
    rel_motion = residual / max(motion_scale, 1e-12)
    rel_shape = residual / max(float(spread[0]), 1e-12)
    fit = float(np.clip(1.0 - rel_motion / max(th.residual_frac, 1e-12), 0.0, 1.0))
    cond = float(np.clip(1.0 - rel_shape / max(th.cond_residual_frac, 1e-12), 0.0, 1.0))
    pts = float(np.clip(n_used / float(max(th.full_points, 1)), 0.0, 1.0))
    shape = 0.0 if spread[0] <= 1e-12 else float(
        np.clip((spread[1] / spread[0]) / max(th.collinear_ratio, 1e-12), 0.0, 1.0))
    return float(ramp * fit * cond * pts * shape)


def estimate_motion(buffer, window=None, view_point=None, view_dir=None, dt=1.0,
                    classifier=None, thresholds=None, logger=None, query_point=None,
                    query_index=None, smoother=None):
    """Turn a :class:`MotionBuffer` window into a :class:`MotionEstimate`.

    Compares the oldest and newest frame of the window over the points valid in **both**
    (the corresponded set), fits a robust rigid motion, decomposes it into a screw, fits
    the surface plane on the latest frame and fits the centroid trajectory line.

    For ``revolute`` the reported ``direction`` is the *tangential* pull direction
    ``omega_hat x (p - p_axis)`` (see :func:`pull_direction`), evaluated at the **query
    point** - pass ``query_index`` (the buffer slot of the tracked handle) or
    ``query_point`` (its world position) to get the direction *at the handle*; without
    either it falls back to the centroid. For a door those are different vectors, and the
    handle is the one the gripper has to follow. The hinge line itself comes back in
    ``axis_dir``/``axis_point`` so a planner can predict the whole arc instead of chasing
    it one frame at a time.

    ``smoother`` is an optional :class:`MotionSmoother`; it sign-aligns and EMA-blends the
    direction and axis across calls.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    logger = logger if logger is not None else getattr(classifier, "logger", None)

    pts, mask, frames = buffer.window(window) if hasattr(buffer, "window") else (
        np.asarray(buffer[0], dtype=float), np.asarray(buffer[1], dtype=bool),
        np.asarray(buffer[2], dtype=float))
    k = int(pts.shape[0])

    def _reject(reason, detail="", **kwargs):
        if classifier is not None:
            classifier.note_rejection(reason, detail)
        elif logger is not None:
            _log(logger, f"[motion] low confidence: {reason} {detail}".rstrip())
        kind = kwargs.pop("kind", FREE)
        return MotionEstimate(kind=kind, confidence=0.0, window_frames=k, reason=reason,
                              **kwargs)

    if k < 2:
        return _reject("insufficient_frames", f"have {k}")

    both = mask[0] & mask[-1]
    n_common = int(both.sum())
    span = float(frames[-1] - frames[0]) * float(dt)
    span = span if span > 1e-12 else float(dt)

    if n_common == 0:
        return _reject("no_correspondence", f"{int(mask[-1].sum())} point(s) this frame")

    P0 = pts[0][both]
    P1 = pts[-1][both]
    centroid_travel = float(np.linalg.norm(P1.mean(axis=0) - P0.mean(axis=0)))
    point_travel = float(np.mean(np.linalg.norm(P1 - P0, axis=1)))
    motion_scale = max(centroid_travel, point_travel)

    if n_common < 3:
        kind = STATIC if motion_scale < th.min_motion_m else FREE
        return _reject("too_few_points", f"{n_common} corresponded point(s)", kind=kind,
                       n_points=n_common, displacement=centroid_travel)

    spread = point_spread(P0)
    if spread[0] <= 1e-12 or spread[1] <= th.collinear_ratio * spread[0]:
        kind = STATIC if motion_scale < th.min_motion_m else FREE
        return _reject("collinear_points",
                       f"sigma=({spread[0]:.4g}, {spread[1]:.4g}, {spread[2]:.4g})",
                       kind=kind, n_points=n_common, displacement=centroid_travel)

    fit = fit_rigid_motion(P0, P1, thresholds=th)
    inl = fit.inliers
    n_used = int(fit.n_used)
    A, B = P0[inl], P1[inl]
    centroid_travel = float(np.linalg.norm(B.mean(axis=0) - A.mean(axis=0)))
    point_travel = float(np.mean(np.linalg.norm(B - A, axis=1)))
    motion_scale = max(centroid_travel, point_travel)

    # Translation-only fit: the null hypothesis a rotation has to beat.
    delta = B - A
    prismatic_residual = float(np.sqrt(np.mean(((delta - delta.mean(axis=0)) ** 2).sum(axis=1))))

    # Where the direction is evaluated: the tracked handle when the caller names one,
    # otherwise the centroid. For a rotation these are genuinely different answers.
    anchor = None
    if query_index is not None:
        j = int(query_index)
        if 0 <= j < mask.shape[1] and mask[-1][j]:
            anchor = pts[-1][j].copy()
        else:
            _log(logger, f"[motion] query_index {j} is not valid this frame; "
                         f"falling back to the centroid")
    if anchor is None and query_point is not None:
        anchor = np.asarray(query_point, dtype=float).reshape(3)
    if anchor is None:
        anchor = B.mean(axis=0)

    screw = screw_decompose(fit.R, fit.t, ref_point=anchor)
    angle = float(screw.angle_rad)

    if classifier is not None:
        kind = classifier.update(angle, centroid_travel, fit.rmse, prismatic_residual,
                                 motion_scale=motion_scale)
    else:
        kind = classify_motion(angle, centroid_travel, fit.rmse, prismatic_residual,
                               motion_scale=motion_scale, thresholds=th)

    line = centroid_direction(buffer, window=window, dt=dt, thresholds=th) \
        if hasattr(buffer, "centroids") else LineFit(n_frames=k)
    plane = _window_plane(pts, mask, fit.R, fit.t, view_point=view_point, view_dir=view_dir,
                          thresholds=th, logger=logger)

    confidence = _confidence(motion_scale, fit.rmse, n_used, spread, th)

    direction = None
    speed = 0.0
    axis_dir = screw.axis_dir
    axis_point = screw.axis_point

    if motion_scale < th.min_direction_m or kind == STATIC:
        # Gate: below the direction floor the object has moved less than the tracking noise
        # can resolve, so *any* unit vector here would be a confident lie about jitter.
        direction, axis_dir, axis_point = None, None, None
        speed = 0.0
        confidence = 0.0
    elif kind == REVOLUTE and axis_dir is not None:
        direction = pull_direction(axis_dir, axis_point, anchor)
        radius = anchor - axis_point
        radius = radius - float(radius @ axis_dir) * axis_dir
        arc = angle * float(np.linalg.norm(radius))
        speed = float(np.hypot(arc, screw.axis_translation) / span)
        if direction is None:
            # The query point sits on the axis: the rotation is real, but that point does
            # not move, so it has no pull direction of its own.
            confidence *= 0.5
    else:
        if kind == PRISMATIC:
            # A slide has no axis of rotation to report.
            axis_dir, axis_point = None, None
        if line.direction is not None:
            direction, speed = line.direction, abs(float(line.speed))
            confidence *= max(line.confidence, 0.0)
        else:
            direction = _unit(B.mean(axis=0) - A.mean(axis=0))
            speed = centroid_travel / span
        if direction is None:
            confidence = 0.0

    perpendicularity, perpendicularity_deg = _perpendicularity(direction, plane.normal)

    estimate = MotionEstimate(
        kind=kind, direction=direction, speed=speed, axis_dir=axis_dir, axis_point=axis_point,
        angle_rad=angle, displacement=centroid_travel,
        plane_normal=plane.normal, perpendicularity=perpendicularity,
        perpendicularity_deg=perpendicularity_deg,
        query_point=(anchor if direction is not None else None),
        confidence=float(np.clip(confidence, 0.0, 1.0)), n_points=n_used, window_frames=k,
        residual=float(fit.rmse), plane=plane, line=line)

    if estimate.confidence <= 0.0:
        reason = "near_zero_motion" if motion_scale < th.min_direction_m else "poor_fit"
        estimate.reason = reason
        detail = f"scale={motion_scale:.5f}m rmse={fit.rmse:.5f}m n={n_used}"
        if classifier is not None:
            classifier.note_rejection(reason, detail)
        elif logger is not None:
            _log(logger, f"[motion] low confidence: {reason} {detail}")
    elif classifier is not None:
        classifier.clear_rejection()

    if smoother is not None:
        smoother.apply(estimate)

    return estimate
