"""Shared, simulator-agnostic scenario helpers for the tracking integration tests.

Both ``test_tracking_pybullet.py`` and ``test_tracking_genesis.py`` drive the *real*
:class:`tracking.session.TrackingSession` against a *real* rendered scene - no LLM, no
agent loop. Objects are moved directly with ``sim.set_base_pose``, so every frame has an
exact ground-truth world position to compare the fused estimate against.

Nothing here imports a simulator: the caller passes an already-connected
:class:`sim_adapter.base.SimAdapter`.

The scenario is also parameterised over the **3D lift provider**
(``providers/tracker3d``): every test body takes a ``tracker3d`` argument, the plain test
methods leave it at ``None`` (the session default, i.e. ``depth_fusion``, so the default
path is unchanged), and ``test_scenario_holds_for_every_tracker3d_provider`` replays the
whole scenario once per provider that can actually be constructed here.
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from providers.tracker3d import factory as tracker3d_factory  # noqa: E402
from sim_adapter import camera_math  # noqa: E402
from tracking import geometry, monitors  # noqa: E402
from tracking.session import TrackingSession  # noqa: E402
from tracking.types import CameraView  # noqa: E402

log = logging.getLogger("tracking_scenario")

RES = 256
FOV = 60.0


def render_view(sim, name, eye, target, up=(0.0, 0.0, 1.0), width=RES, height=RES):
    """Render one camera through the adapter and wrap it as a :class:`CameraView`.

    Mirrors what ``Robot.capture_camera_view`` does in production, including the
    metric-depth conversion that hides the PyBullet/Genesis encoding difference.
    """
    view_matrix = sim.compute_view_matrix(eye=list(eye), target=list(target), up=list(up))
    projection = sim.compute_projection_matrix(FOV, float(width) / float(height),
                                               config.near_plane, config.far_plane)
    frame = sim.render_camera(width, height, view_matrix, projection)
    depth = camera_math.depth_to_metric(frame.depth, sim.depth_encoding,
                                        config.near_plane, config.far_plane)
    return CameraView(name=name, rgb=np.asarray(frame.rgb),
                      depth=np.asarray(depth, dtype=np.float32),
                      view_matrix=geometry.mat4(view_matrix),
                      projection_matrix=geometry.mat4(projection),
                      near=float(config.near_plane), far=float(config.far_plane),
                      position=list(eye))


def surface_seed_points(view, world_center, spread_px=5, count=5):
    """Seed points that lie on the object's *visible surface*, as detect_object would.

    Deprojecting the object's centre would place a point inside the mesh, which the
    occlusion test then (correctly) rejects. Instead the centre is projected to a pixel,
    a small cross of neighbouring pixels is sampled from the depth buffer, and those are
    deprojected back - exactly the "2D points + depth -> 3D" path the tracker itself uses.

    Returns ``(world_points (N, 3), pixels (N, 2))``; raises when the object is not
    visible, which in a test means the scene is set up wrong.
    """
    pixel, _z_eye = geometry.project_world_to_pixel(view, world_center)
    if pixel is None or not geometry.in_bounds(view, pixel):
        raise AssertionError(f"object at {world_center} is not in the '{view.name}' view")

    offsets = [(0, 0), (-spread_px, 0), (spread_px, 0), (0, -spread_px), (0, spread_px)]
    world_points, pixels = [], []
    for dx, dy in offsets[:count]:
        candidate = np.array([pixel[0] + dx, pixel[1] + dy], dtype=float)
        depth = geometry.sample_depth(view, candidate)
        if depth is None:
            continue
        world_points.append(geometry.deproject_pixel_to_world(view, candidate, depth))
        pixels.append(candidate)
    if not world_points:
        raise AssertionError(f"no valid depth around {pixel} in the '{view.name}' view")
    return np.asarray(world_points), np.asarray(pixels)


def paint_occluder(view, world_pos, radius=40, standoff=0.25):
    """Simulate the robot arm passing between the camera and the object.

    Overwrites a disc of the depth buffer with a much closer surface, so
    ``geometry.is_visible`` reports the object as occluded in that camera. Doing it on
    the depth buffer keeps the test deterministic - no dependence on where the arm
    happens to swing - while exercising exactly the code path a real occlusion hits.
    """
    pixel, z_eye = geometry.project_world_to_pixel(view, world_pos)
    if pixel is None:
        return
    cx, cy = int(round(pixel[0])), int(round(pixel[1]))
    ys, xs = np.ogrid[:view.height, :view.width]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    view.depth[mask] = max(float(z_eye) - standoff, config.track_depth_min * 2.0)
    # The occluder must look different too, or the tracker would happily keep matching
    # the object's texture through it.
    view.rgb[mask] = 30


def object_body(env):
    """Handle of the ``grasp`` scene's target object, in either simulator."""
    body = getattr(env.simenv, "object_id", None)
    if body is None:
        body = env.simenv.get_state().get("object_id")
    if body is None:
        raise AssertionError("the active sim-env exposes no grasp object to track")
    return body


def tracking_error(report, name, ground_truth, seed_offset):
    """Distance between the fused estimate and the expected point.

    ``seed_offset`` is the constant vector from the object's origin to the centroid of the
    seed points (which sit on the surface facing the camera). Comparing against
    ``ground_truth + seed_offset`` measures tracking error rather than that fixed bias.
    """
    obj = report.objects[name]
    if obj.world_point is None:
        return None
    expected = np.asarray(ground_truth, dtype=float) + np.asarray(seed_offset, dtype=float)
    return float(np.linalg.norm(np.asarray(obj.world_point, dtype=float) - expected))


SETTLE_STEPS = 120
OBJECT_START = [-0.2, 0.4, 0.1]
# Seen by the head camera but outside the wrist camera's frustum, so the wrist track can
# only ever start through a cross-camera bootstrap.
OUT_OF_WRIST_VIEW = [0.45, 0.45, 0.1]
# The seed points sit on the object's camera-facing surface, so the estimate carries a
# fixed offset from the body origin; only tracking *error* on top of that is asserted.
POSITION_TOL = 0.04


class ProviderBand:
    """The accuracy a given 3D lift provider is *expected* to reach on this scene.

    Bands are two-sided on purpose. An upper bound catches a regression; the lower bound
    on a deliberately weak provider catches the opposite - a provider silently becoming
    accurate means it is no longer the thing the test thinks it is measuring, and the
    documented finding below would need re-deriving rather than quietly disappearing.
    """

    def __init__(self, median_max, occluded_max, median_min=0.0, occluded_min=0.0,
                 note=""):
        self.median_max = float(median_max)
        self.occluded_max = float(occluded_max)
        self.median_min = float(median_min)
        self.occluded_min = float(occluded_min)
        self.note = note


# Measured on this scenario (grasp scene, head+wrist, 256x256, PyBullet):
#   depth_fusion  scripted median  9.8 mm | head-camera blackout  14.6 mm
#   triangulate   scripted median  6.8 mm | head-camera blackout 141.4 mm
#   rigid_refine  scripted median  2.2 mm | head-camera blackout 141.4 mm
# With a clear line of sight the triangulators are as good as (or better than) depth
# fusion. Under the painted occluder they are an order of magnitude worse, and that is
# structural rather than a bug: at exactly two cameras the DLT is an exactly-determined
# system, so the reprojection residual a triangulator would use to reject a bad
# correspondence is ~0 by construction (a 47 px drift moves the 3D point >1 cm while the
# residual stays at 0.18 px). Depth fusion still has the depth buffer to disagree with.
# The weakness is therefore asserted as a two-sided expectation instead of being hidden by
# loosening the shared bands: if triangulation ever became occlusion-robust at 2 views,
# this band must be re-derived rather than silently drift.
TRACKER3D_BANDS = {
    "depth_fusion": ProviderBand(median_max=POSITION_TOL, occluded_max=2 * POSITION_TOL),
    "triangulate": ProviderBand(median_max=POSITION_TOL, occluded_max=0.30,
                                occluded_min=0.05,
                                note="2-view DLT is exactly determined, so its "
                                     "residual-based outlier rejection is structurally "
                                     "blind during the head camera's blackout"),
    # Triangulation plus a rigid-shape constraint; still a 2-view system under occlusion,
    # so only an upper bound is asserted while that provider is being tuned.
    "rigid_refine": ProviderBand(median_max=POSITION_TOL, occluded_max=0.30),
}
#: Anything registered later (e.g. ``rigid_refine``) gets a deliberately wide band until
#: it has been measured, so a new provider joins the sweep without being mis-asserted.
DEFAULT_BAND = ProviderBand(median_max=0.15, occluded_max=0.40)


def band_for(name):
    return TRACKER3D_BANDS.get(name, DEFAULT_BAND)


def available_tracker3d(names=None):
    """Providers from ``factory.SUPPORTED`` that can actually be constructed here.

    Discovery rather than a hard-coded list, so a newly registered provider is swept
    automatically; one whose optional dependencies are missing (LAPA's torch/DINOv2
    clone) is skipped instead of failing the suite.
    """
    out = []
    for name in (names or tracker3d_factory.SUPPORTED):
        try:
            tracker3d_factory.get_tracker3d(name)
        except Exception as exc:
            log.info("[scenario] 3D provider '%s' unavailable: %s: %s",
                     name, type(exc).__name__, exc)
            continue
        out.append(name)
    return out


class TrackingScenarioMixin:
    """The whole tracking scenario, shared by the PyBullet and Genesis test cases.

    Deliberately not a ``TestCase`` so pytest does not collect it on its own: each
    simulator subclass supplies ``sim``/``env``/``robot``/``object_id`` from its own boot
    and inherits the identical assertions, which is what makes the two suites comparable.
    """

    # -- helpers -----------------------------------------------------------
    def place(self, position, settle=3):
        """Teleport the tracked object and let the render catch up."""
        self.sim.set_base_pose(self.object_id, position, [0.0, 0.0, 0.0, 1.0])
        for _ in range(settle):
            self.sim.step()
        return np.asarray(position, dtype=float)

    def make_session(self, monitor=None, tracker3d=None, **kwargs):
        """Build the session under test.

        ``tracker3d=None`` deliberately does *not* forward anything, so the default path
        constructs exactly the session it did before this scenario was parameterised.
        """
        if tracker3d is not None:
            kwargs["tracker3d"] = tracker3d
        return TrackingSession(robot=self.robot, env=self.env, provider="template",
                               monitor=monitor, write_jsonl=False, **kwargs)

    def seed(self, session, name, world_center):
        """Seed a target from the head camera's view of it, as ``detect_object`` would.

        Returns the constant offset between the object origin and the seed centroid.
        """
        head = self.robot.capture_camera_view("head", self.env)
        points, _pixels = surface_seed_points(head, world_center)
        session.add_target(name, points)
        return points.mean(axis=0) - np.asarray(world_center, dtype=float)

    def gripper_pose(self):
        pos, quat = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return {"position": list(pos), "orientation_q": list(quat)}

    # -- tests -------------------------------------------------------------
    def test_capture_camera_view_is_metric(self):
        """The tracker's view of the world must agree with the simulator's ground truth."""
        centre = self.place(OBJECT_START)
        for cam in ("head", "wrist"):
            view = self.robot.capture_camera_view(cam, self.env)
            self.assertEqual(view.depth.shape[:2], view.rgb.shape[:2])
            finite = view.depth[np.isfinite(view.depth)]
            self.assertGreater(finite.min(), 0.0, f"{cam} depth must be metric metres")
            self.assertLess(finite.min(), config.far_plane)

        head = self.robot.capture_camera_view("head", self.env)
        pixel, z_eye = geometry.project_world_to_pixel(head, centre)
        self.assertTrue(geometry.in_bounds(head, pixel))
        self.assertGreater(z_eye, 0.0)
        # Deprojecting the object's own surface depth must land back on the object.
        depth = geometry.sample_depth(head, pixel)
        self.assertIsNotNone(depth)
        back = geometry.deproject_pixel_to_world(head, pixel, depth)
        self.assertLess(float(np.linalg.norm(back - centre)), 0.1)

    def test_tracks_scripted_object_motion(self):
        """Fused world coordinates follow a scripted motion in both cameras."""
        self._run_scripted_motion()

    def _run_scripted_motion(self, tracker3d=None, band=None):
        """The scripted-motion assertions; returns the median error in metres."""
        band = band or band_for(tracker3d or "depth_fusion")
        start = self.place(OBJECT_START)
        session = self.make_session(monitor=monitors.object_not_lost("cube", patience=6),
                                    tracker3d=tracker3d)
        offset = self.seed(session, "cube", start)

        errors, seeded_frames = [], {"head": None, "wrist": None}
        for step in range(12):
            truth = self.place(start + np.array([0.012 * step, 0.0, 0.004 * step]))
            report = session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=step)
            self.assertIsNotNone(report, f"frame {step} was dropped")
            state = report.objects["cube"]
            for cam, track in state.cams.items():
                if track.seeded and seeded_frames[cam] is None:
                    seeded_frames[cam] = step
            if state.world_point is not None:
                errors.append(tracking_error(report, "cube", truth, offset))

        self.assertEqual(session.errors, 0, "tracking raised internally")
        self.assertGreaterEqual(len(errors), 8, "object was lost for most of the run")
        median = float(np.median(errors))
        log.info("[scenario] scripted motion | tracker3d=%s median=%.4f m band=[%.3f, %.3f]",
                 session.tracker3d_name, median, band.median_min, band.median_max)
        self.assertLess(median, band.median_max,
                        f"median world-position error {median:.3f} m too large "
                        f"for '{session.tracker3d_name}'")
        self.assertGreaterEqual(median, band.median_min,
                                f"'{session.tracker3d_name}' is unexpectedly accurate "
                                f"({median:.3f} m) - re-derive its band: {band.note}")
        self.assertIsNotNone(seeded_frames["head"], "head camera never seeded")
        self.assertFalse(session.aborted)

        summary = session.summary()
        self.assertEqual(summary["frames"], 12)
        self.assertIn("cube", summary["objects"])
        return median

    def test_wrist_camera_is_seeded_from_head(self):
        """The wrist cam starts blind to the object and is bootstrapped by the head cam."""
        self._run_cross_camera_bootstrap()

    def _run_cross_camera_bootstrap(self, tracker3d=None, band=None):
        band = band or band_for(tracker3d or "depth_fusion")
        start = self.place(OUT_OF_WRIST_VIEW)
        session = self.make_session(tracker3d=tracker3d)
        offset = self.seed(session, "cube", start)

        # Frame 0 with the object still at its seed pose: only the head can see it.
        first = session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=0)
        self.assertTrue(first.objects["cube"].cams["head"].seeded)
        self.assertFalse(first.objects["cube"].cams["wrist"].seeded,
                         "the wrist camera was supposed to start blind to the object")

        gripper = np.asarray(self.gripper_pose()["position"], dtype=float)
        end = gripper + np.array([0.0, 0.0, -0.05])
        steps = 24
        wrist_seed_frame, wrist_events, last_err = None, [], None
        for step in range(1, steps + 1):
            # Walk the object towards the gripper, which is what the wrist camera looks
            # at - so it enters that view partway through the run.
            truth = self.place(start + (step / steps) * (end - start))
            report = session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=step)
            self.assertIsNotNone(report)
            state = report.objects["cube"]
            wrist_events += [e for e in state.reseeds if e.cam == "wrist" and e.applied]
            if state.cams["wrist"].seeded and wrist_seed_frame is None:
                wrist_seed_frame = step
            if state.world_point is not None:
                last_err = tracking_error(report, "cube", truth, offset)

        self.assertEqual(session.errors, 0)
        self.assertIsNotNone(wrist_seed_frame,
                             "wrist camera was never seeded from the head camera")
        self.assertGreater(wrist_seed_frame, 0, "wrist seeding was not a late bootstrap")
        self.assertTrue(any(e.reason == "unseeded" and e.donor == "head" for e in wrist_events),
                        "wrist seeding did not come from the head camera")
        self.assertIsNotNone(last_err, "object was lost by the end of the run")
        log.info("[scenario] cross-camera bootstrap | tracker3d=%s wrist_seed_frame=%d "
                 "final_error=%.4f m", session.tracker3d_name, wrist_seed_frame, last_err)
        self.assertLess(last_err, 2 * band.median_max)
        return wrist_seed_frame, last_err

    def test_occluded_camera_is_reseeded_from_the_other(self):
        """Blinding the head camera must repair its track from the wrist camera."""
        self._run_painted_occluder()

    def _run_painted_occluder(self, tracker3d=None, band=None):
        """Returns the error (m) at the end of the head camera's blackout."""
        band = band or band_for(tracker3d or "depth_fusion")
        # At the spawn pose both cameras see the object, which is the precondition for
        # testing what happens when one of them is blinded.
        centre = self.place(OBJECT_START)
        session = self.make_session(tracker3d=tracker3d)
        offset = self.seed(session, "cube", centre)

        for step in range(3):
            report = session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=step)
            self.assertIsNotNone(report)
        cams = session.last_report.objects["cube"].cams
        self.assertTrue(cams["head"].seeded and cams["wrist"].seeded,
                        "both cameras must hold a track before occlusion can be tested")

        # Now something passes right in front of the head camera. Painting the depth
        # buffer is deterministic, and hits exactly the code path a real arm occlusion
        # does: is_visible() rejects the object, the head track degrades, and the policy
        # must repair it from the wrist camera rather than latch onto the occluder.
        head_reseeds = []
        for step in range(3, 14):
            views = session.capture_views()
            paint_occluder(views["head"], centre + offset)
            report = session.on_frame(views=views, gripper_pose=self.gripper_pose(),
                                      trajectory_step=step)
            self.assertIsNotNone(report)
            head_reseeds += [e for e in report.objects["cube"].reseeds if e.cam == "head"]

        self.assertEqual(session.errors, 0)
        self.assertTrue(head_reseeds, "occlusion never triggered a re-seed decision")
        self.assertTrue(all(e.donor == "wrist" for e in head_reseeds),
                        "the head track was not repaired from the wrist camera")
        # A re-seed must never be applied onto the occluding surface.
        for event in head_reseeds:
            self.assertFalse(event.applied, f"head re-seeded onto the occluder: {event.detail}")
        # The wrist camera keeps the object alive through the head camera's blackout.
        state = session.last_report.objects["cube"]
        self.assertIsNotNone(state.world_point,
                             "object was lost even though one camera still saw it")
        self.assertIn("wrist", state.visible_cams)
        error = tracking_error(session.last_report, "cube", centre, offset)
        log.info("[scenario] painted occluder | tracker3d=%s occluded_error=%.4f m "
                 "band=[%.3f, %.3f] reseeds=%d", session.tracker3d_name, error,
                 band.occluded_min, band.occluded_max, len(head_reseeds))
        self.assertLess(error, band.occluded_max,
                        f"'{session.tracker3d_name}' drifted {error:.3f} m under occlusion")
        self.assertGreaterEqual(error, band.occluded_min,
                                f"'{session.tracker3d_name}' held {error:.3f} m under "
                                f"occlusion, better than its documented band: {band.note}")
        return error

    def test_detach_trips_the_attachment_invariant(self):
        """``attached_to_gripper`` aborts the rollout when the object leaves the gripper."""
        self._run_detach_abort()

    def _run_detach_abort(self, tracker3d=None):
        gripper = np.asarray(self.gripper_pose()["position"], dtype=float)
        held = self.place(gripper + np.array([0.0, 0.0, -0.05]))

        session = self.make_session(
            monitor=monitors.attached_to_gripper("cube", max_dist=0.12, grace_frames=1),
            tracker3d=tracker3d)
        self.seed(session, "cube", held)

        for step in range(3):
            report = session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=step)
            self.assertIsNotNone(report)
            self.assertFalse(session.aborted, "aborted while the object was still attached")

        # Drop it: the object falls away from the gripper while the gripper stays put.
        for step in range(3, 10):
            self.place(held + np.array([0.0, 0.0, -0.05 * (step - 2)]))
            session.on_frame(gripper_pose=self.gripper_pose(), trajectory_step=step)
            if session.aborted:
                break

        self.assertEqual(session.errors, 0)
        self.assertTrue(session.aborted, "detached object did not trip the invariant")
        self.assertIn("cube", session.abort_reason)
        self.assertEqual(session.summary()["status"], "abort")

    # -- provider sweep ----------------------------------------------------
    def test_scenario_holds_for_every_tracker3d_provider(self):
        """Replay the whole scenario against each constructible 3D lift provider.

        The three behavioural assertions (cross-camera bootstrap, occluder repair, detach
        abort) are provider-agnostic and must hold everywhere; only the numeric accuracy
        band varies, per :data:`TRACKER3D_BANDS`.
        """
        providers = available_tracker3d()
        if not providers:
            self.skipTest("no 3D lift provider is constructible in this environment")
        log.info("[scenario] sweeping 3D lift providers: %s", providers)

        measured = {name: {} for name in providers}
        for name in providers:
            band = band_for(name)
            with self.subTest(tracker3d=name, case="scripted_motion"):
                measured[name]["median_m"] = self._run_scripted_motion(name, band)
            with self.subTest(tracker3d=name, case="cross_camera_bootstrap"):
                self._run_cross_camera_bootstrap(name, band)
            with self.subTest(tracker3d=name, case="painted_occluder"):
                measured[name]["occluded_m"] = self._run_painted_occluder(name, band)
            with self.subTest(tracker3d=name, case="detach_abort"):
                self._run_detach_abort(name)
        log.info("[scenario] 3D provider accuracy: %s",
                 {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in measured.items()})

