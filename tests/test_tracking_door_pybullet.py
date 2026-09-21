"""Tracking the Adroit door handle through a hinge + latch arc on PyBullet.

The grasp scenario (``tests/test_tracking_pybullet.py``) teleports a free body, so ground
truth is whatever pose was written. The door is the opposite case and the one the hinge-axis
motion estimation work needs: the handle is **articulated**, its world path is an arc the
test does not choose, and the simulator reports it for free through
``SimEnvDoor.get_state()["door_handle_pos"]``.

What is exercised end to end:

* seeding from a **2D affordance point** ``{"point": [x, y], "label": "door_handle"}`` through
  the patch-depth-sampling path of ``tests/test_affordance_depth_sampling.py``
  (``utils.sample_surface_depth`` -> deproject), never from a known 3D position;
* the real :class:`tracking.session.TrackingSession` over the real ``head``/``wrist`` renders
  while the hinge swings ~43 deg and the latch lever turns to its stop;
* both 3D lift providers, ``depth_fusion`` and ``triangulate``.

**The seed offset is the subtlety this file is really about.** The tracker seeds on the
*surface* the camera images (~0.17 m from the latch link origin here); ground truth is the
link origin. That constant is not tracking error - but it is constant only in the *handle's*
frame, because the handle rotates about the hinge. Subtracting a fixed world-frame offset
instead inflates the error to 0.133 m by the end of the arc (asserted in
``test_a_fixed_world_offset_would_fake_a_growing_error``), which would be the test measuring
its own bug rather than the tracker.

Run with::

    python -m pytest tests/test_tracking_door_pybullet.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import utils  # noqa: E402
from tracking import geometry, monitors  # noqa: E402
from tracking.session import TrackingSession  # noqa: E402

try:
    import pybullet as p
except ImportError:                                  # pragma: no cover - env without pybullet
    p = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOOR_URDF = os.path.join(REPO_ROOT, "my_assets", "adroit_door", "adroit_door.urdf")

OBJECT = "door_handle"
#: The affordance point a pointing VLM would emit. Only the ``point`` is ever consumed; it is
#: filled in per run by projecting the true handle pose into the head camera, so the test
#: follows the render resolution (``config.image_width``) instead of hard-coding pixels.
AFFORDANCE_TEMPLATE = {"point": None, "label": OBJECT}

#: Cross of pixels around the affordance point, each patch-sampled into its own seed point.
#: A point *set* is what gives the trackers multi-point outlier rejection and gives
#: ``triangulate`` per-seed correspondence across cameras. +/-2 px keeps the whole cross on
#: the lever, which is ~9 px across at this range once the door has swung and foreshortened it.
SEED_PIXEL_OFFSETS = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2))

#: Where on the ``latch`` link to aim the affordance point, in the link's own frame, so the
#: pixel is *derived* from the simulator's pose rather than hard-coded.
#:
#: Not the link origin: ``adroit_door.urdf`` puts the latch axle (a r=0.05 l=0.3 cylinder along
#: the link's y) through the door, so the origin's pixel shows the door panel's wooden
#: cylinder *beside* the handle - measured 0.09 m away, and rigid to the **panel**, not to the
#: latch. Seeding there would attach the offset to the wrong body and the correction below
#: would be undone by the lever's own rotation. ``[0, -0.15, 0]`` is the axle's end cap, where
#: the lever bar is rooted; the sign is chosen per run by picking the cap facing the camera.
LATCH_AXLE_HALF_LENGTH = 0.15
#: ``get_link_pose`` reports PyBullet's COM frame; the URDF's ``latch`` inertial origin is the
#: offset between it and the link frame the geometry above is expressed in.
LATCH_INERTIAL_ORIGIN = np.array([-0.017, 0.013, 0.0])

SETTLE_STEPS = 120
FRAMES = 40
SUBSTEPS = 8
#: Hinge/latch targets per frame. Measured: the hinge reaches ~0.75 rad (43 deg), the latch its
#: 0.8 rad stop by frame 27, and the handle sweeps a ~0.37 m arc. Deliberately slow: at twice
#: this rate the handle's appearance changes faster than an NCC template can follow and the
#: track dies on the revealed background instead of measuring anything.
HINGE_STEP = 0.02
LATCH_STEP = 0.03
LATCH_MAX = 0.8

#: The handle is a ~9 px feature at this range, so the defaults (``track_patch_half`` 12,
#: ``track_search_scale`` 3.0) give a 25x25 template that is mostly door-panel wood grain and
#: a 73x73 search window. Measured: that template latches onto the static background revealed
#: by the opening door within 11 frames and ends 2.0 m out. A patch that is the handle, and a
#: search window a few times the 1.7 px/frame motion, tracks the whole arc.
TRACKER_KWARGS = {"patch_half": 5, "search_scale": 2.0}

#: Thresholds, all from measured runs (PyBullet 3.2.x, 256x256 renders, template tracker):
#:   depth_fusion  median 0.064 m  max 0.082 m  lost 0/40  max jump 0.014 m
#:   triangulate   median 0.054 m  max 0.079 m  lost 0/40  max jump 0.014 m
#: The residual is the NCC template lagging ~8-10 px behind a handle that is simultaneously
#: rotating and foreshortening; the depth/geometry half of the pipeline is exact to 0.041 m
#: (see ``test_perfect_tracker_control_scores_near_zero``).
MEDIAN_TOL = 0.075
MAX_TOL = 0.095
#: Largest ground-truth handle step between two frames is ~0.010 m and the largest measured
#: estimate step is 0.028 m (one-pixel NCC quantisation at ~0.0055 m/px, amplified by the
#: depth patch), so anything past this is the tracker jumping, not the door.
MAX_FRAME_JUMP = 0.04
#: The perfect-tracker control: ground-truth-projected pixels through the same patch-depth
#: path. Measured median 0.004 m, max 0.041 m - the residual appears only once the lever has
#: turned, and is the lever bar's own diameter (r=0.02): a *material* point on a round bar
#: rotates away from the surface the camera images. Score the same recoveries against a
#: frozen world offset and the median jumps to 0.064 m, peaking at 0.097 m.
CONTROL_TOL = 0.05


def _door_asset_available():
    return p is not None and os.path.exists(DOOR_URDF)


class _Args:
    mode = "default"
    robot = "franka"
    task = "door"
    save_grasp_inputs = False
    tracking = True
    tracker_provider = "template"
    track_interval = 1
    track_save_depth = False
    track_log_dir = None


def _boot():
    """Bring up the ``door`` scene headlessly through the production classes."""
    import env as env_module
    from robot import Robot
    from debug.dbg_utils import init_loguru_logger
    from sim_adapter import get_adapter

    if p.isConnected():
        p.disconnect()
    sim = get_adapter("pybullet")
    sim.connect(gui=False)
    sim.set_asset_search_path()
    sim.set_gravity(0, 0, -9.81)
    sim.load_urdf("plane.urdf")

    args = _Args()
    utils.args = args
    environment = env_module.Environment(args, sim)
    environment.simenv.configure_robot_pose()
    environment.load()
    robot = Robot(args, init_loguru_logger("tracking_door_pybullet.log"), sim)
    sim.build()
    for _ in range(SETTLE_STEPS):
        environment.update()
    return sim, environment, robot


def rotation(quat):
    return np.asarray(p.getMatrixFromQuaternion(list(quat)), dtype=float).reshape(3, 3)


def seed_from_affordance_point(view, affordance):
    """``{"point": [x, y], "label": ...}`` -> world seed points, the production way.

    Exactly the path ``tests/test_affordance_depth_sampling.py`` pins down: the *near*
    surface inside a patch around each pixel (``utils.sample_surface_depth``), deprojected
    with the camera's own matrices. A single ``depth[y, x]`` read would land on the door face
    behind the lever whenever the point misses the ~7 px bar by a pixel or two.

    Returns ``(world_points (N, 3), pixels (N, 2))``.
    """
    x0, y0 = int(round(float(affordance["point"][0]))), int(round(float(affordance["point"][1])))
    world_points, pixels = [], []
    for dx, dy in SEED_PIXEL_OFFSETS:
        x, y = x0 + dx, y0 + dy
        if not geometry.in_bounds(view, (x, y)):
            continue
        depth, _info = utils.sample_surface_depth(view.depth, x, y)
        if depth is None:
            continue
        world_points.append(geometry.deproject_pixel_to_world(view, (x, y), depth))
        pixels.append((x, y))
    if not world_points:
        raise AssertionError(f"no usable depth around {(x0, y0)} in the '{view.name}' view")
    return np.asarray(world_points, dtype=float), np.asarray(pixels, dtype=float)


class TestTrackingDoorPyBullet(unittest.TestCase):
    """One booted door scene for the whole class; each test re-closes the door itself."""

    @classmethod
    def setUpClass(cls):
        if not _door_asset_available():
            raise unittest.SkipTest("pybullet or my_assets/adroit_door/adroit_door.urdf missing")
        cls.sim, cls.env, cls.robot = _boot()
        state = cls.env.simenv.get_state()
        if state.get("door_id") is None or state.get("door_handle_pos") is None:
            raise unittest.SkipTest("the door scene did not load a handle to track")
        cls.door_id = state["door_id"]
        cls.hinge_index = state["door_hinge_index"]
        cls.latch_index = state["latch_index"]
        cls.handle_link = state["door_handle_latch"]

    @classmethod
    def tearDownClass(cls):
        if p is not None and p.isConnected():
            p.disconnect()

    # -- scene helpers -----------------------------------------------------
    def setUp(self):
        self.close_door()

    def close_door(self):
        for joint in (self.hinge_index, self.latch_index):
            self.sim.reset_joint_state(self.door_id, joint, 0.0)
            self.sim.set_joint_position(self.door_id, joint, target=0.0, force=200)
        for _ in range(SUBSTEPS * 4):
            self.sim.step()

    def drive_door(self, step):
        """Open the hinge and turn the latch lever by one frame's worth."""
        self.sim.set_joint_position(self.door_id, self.hinge_index,
                                    target=HINGE_STEP * step, force=200)
        self.sim.set_joint_position(self.door_id, self.latch_index,
                                    target=min(LATCH_MAX, LATCH_STEP * step), force=200)
        for _ in range(SUBSTEPS):
            self.sim.step()

    def handle_pose(self):
        """Ground truth: the latch link's world pose, straight from the simulator."""
        pos, quat = self.sim.get_link_pose(self.door_id, self.handle_link)
        return np.asarray(pos, dtype=float), rotation(quat)

    def handle_aim_point(self, view):
        """World point on the *latch link* that the affordance pixel should look at.

        Derived from the link's current pose, so it follows the door wherever the scene puts
        it. Both ends of the axle are candidates; the one nearer the camera is the one whose
        lever root is actually imaged.
        """
        pos, rot = self.handle_pose()
        best, best_z = None, float("inf")
        for sign in (-1.0, 1.0):
            local = np.array([0.0, sign * LATCH_AXLE_HALF_LENGTH, 0.0]) - LATCH_INERTIAL_ORIGIN
            world = pos + rot @ local
            _pixel, z_eye = geometry.project_world_to_pixel(view, world)
            if z_eye > 0.0 and z_eye < best_z:
                best, best_z = world, z_eye
        self.assertIsNotNone(best, "neither end of the latch axle is in front of the camera")
        return best

    def joint_angles(self):
        return (self.sim.get_joint_state(self.door_id, self.hinge_index).position,
                self.sim.get_joint_state(self.door_id, self.latch_index).position)

    def gripper_pose(self):
        pos, quat = self.sim.get_link_pose(self.robot.id, self.robot.ee_index)
        return {"position": list(pos), "orientation_q": list(quat)}

    # -- seeding -----------------------------------------------------------
    def affordance_point(self, view):
        """Derive the pixel a pointing model would produce, from the true handle pose.

        Only the *pixel* leaves this method: everything downstream sees a 2D affordance
        point and nothing else, which is the whole point of the exercise.
        """
        aim = self.handle_aim_point(view)
        pixel, z_eye = geometry.project_world_to_pixel(view, aim)
        self.assertIsNotNone(pixel, "the handle is behind the head camera")
        self.assertGreater(z_eye, 0.0)
        margin = max(o for pair in SEED_PIXEL_OFFSETS for o in pair)
        self.assertTrue(geometry.in_bounds(view, pixel, margin=margin),
                        f"handle projects to {np.round(pixel, 1)}, outside the usable "
                        f"{view.width}x{view.height} image")
        point = [int(round(float(pixel[0]))), int(round(float(pixel[1])))]
        return dict(AFFORDANCE_TEMPLATE, point=point)

    def seed(self, session, view, affordance):
        """Seed the session and return the seed offset *in the handle's own frame*.

        The returned offset is local, not world: the handle swings about the hinge, so the
        vector from the link origin to the seeded surface point rotates with it. Re-applying
        it through the link's current rotation each frame is what keeps a constant bias
        constant instead of letting it masquerade as drift.
        """
        points, pixels = seed_from_affordance_point(view, affordance)
        session.add_target(OBJECT, points)
        pos, rot = self.handle_pose()
        local_offset = rot.T @ (points.mean(axis=0) - pos)
        print(f"[door-track] seed pixel={affordance['point']} label={affordance['label']} "
              f"seeds={len(points)} px_span={np.ptp(pixels, axis=0).tolist()} "
              f"local_offset={np.round(local_offset, 4).tolist()} "
              f"|offset|={np.linalg.norm(local_offset):.4f} m")
        return local_offset

    # -- the arc -----------------------------------------------------------
    def run_arc(self, tracker3d, frames=FRAMES, monitor=None):
        """Seed from a 2D affordance point, open the door, track it, and score every frame."""
        self.close_door()
        head = self.robot.capture_camera_view("head", self.env)
        affordance = self.affordance_point(head)

        session = TrackingSession(robot=self.robot, env=self.env, provider="template",
                                  tracker3d=tracker3d, monitor=monitor, write_jsonl=False,
                                  tracker_kwargs=dict(TRACKER_KWARGS))
        local_offset = self.seed(session, head, affordance)
        seed_pos, seed_rot = self.handle_pose()
        world_offset = seed_rot @ local_offset        # the naive, frozen-in-world version

        run = {"errors": [], "naive_errors": [], "estimates": [], "truth": [], "lost": 0,
               "lift_meta": [], "wrist_seen": 0, "session": session,
               "affordance": affordance, "local_offset": local_offset}
        for step in range(frames):
            self.drive_door(step)
            pos, rot = self.handle_pose()
            expected = pos + rot @ local_offset
            views = session.capture_views()
            run["wrist_seen"] += int(self.handle_is_imaged(views.get("wrist"), expected))
            report = session.on_frame(views=views, gripper_pose=self.gripper_pose(),
                                      trajectory_step=step)
            self.assertIsNotNone(report, f"frame {step} was dropped")
            state = report.objects[OBJECT]
            run["lift_meta"].append(dict(state.lift_meta or {}))
            run["truth"].append(expected)
            if state.world_point is None:
                run["lost"] += 1
                run["estimates"].append(None)
                continue
            estimate = np.asarray(state.world_point, dtype=float)
            run["estimates"].append(estimate)
            run["errors"].append(float(np.linalg.norm(estimate - expected)))
            run["naive_errors"].append(float(np.linalg.norm(estimate - (pos + world_offset))))
            if step % 10 == 0:
                hinge, latch = self.joint_angles()
                print(f"[door-track] {tracker3d} frame {step:2d} hinge={hinge:+.3f} "
                      f"latch={latch:+.3f} err={run['errors'][-1]:.4f} m "
                      f"cams={state.visible_cams}")

        run["truth"] = np.asarray(run["truth"], dtype=float)
        run["path_len"] = float(np.sum(np.linalg.norm(np.diff(run["truth"], axis=0), axis=1)))
        tracked = np.asarray([e for e in run["estimates"] if e is not None], dtype=float)
        jumps = (np.linalg.norm(np.diff(tracked, axis=0), axis=1) if len(tracked) > 1
                 else np.zeros(1))
        run["jumps"] = jumps
        print(f"[door-track] {tracker3d}: n={len(run['errors'])}/{frames} "
              f"median={np.median(run['errors']):.4f} max={np.max(run['errors']):.4f} m "
              f"lost={run['lost']} wrist_seen={run['wrist_seen']}/{frames} "
              f"max_jump={np.max(run['jumps']):.4f} m "
              f"handle_path={run['path_len']:.3f} m | the same estimates scored against a "
              f"frozen world offset: median={np.median(run['naive_errors']):.4f} "
              f"max={np.max(run['naive_errors']):.4f} m")
        return run

    def handle_is_imaged(self, view, world_point):
        """Does this camera actually see the handle? (recorded, never asserted per frame)"""
        if view is None:
            return False
        return bool(geometry.is_visible(view, world_point)[0])

    def assert_arc_is_tracked(self, run, median_tol=MEDIAN_TOL, max_tol=MAX_TOL):
        session, frames = run["session"], len(run["truth"])
        self.assertEqual(session.errors, 0, "tracking raised internally")
        self.assertFalse(session.aborted, f"session aborted: {session.abort_reason}")
        # The motion must be worth measuring: a door that barely moves passes anything.
        self.assertGreater(run["path_len"], 0.30,
                           "the handle hardly moved; the hinge is not being driven")
        self.assertGreaterEqual(len(run["errors"]), int(0.8 * frames),
                                f"object was lost on {run['lost']}/{frames} frames")
        self.assertLess(float(np.median(run["errors"])), median_tol)
        self.assertLess(float(np.max(run["errors"])), max_tol)

        estimates = [e for e in run["estimates"] if e is not None]
        self.assertLess(float(np.max(run["jumps"])), MAX_FRAME_JUMP,
                        f"estimate jumped {np.max(run['jumps']):.3f} m between frames")
        self.assertEqual(len(estimates), len(run["errors"]))

    # -- tests -------------------------------------------------------------
    def test_affordance_pixel_is_derived_and_seeds_on_the_handle(self):
        """The 2D point must be a real, in-bounds pixel that deprojects onto the handle."""
        head = self.robot.capture_camera_view("head", self.env)
        affordance = self.affordance_point(head)
        x, y = affordance["point"]
        self.assertEqual(affordance["label"], OBJECT)
        self.assertTrue(0 <= x < config.image_width and 0 <= y < config.image_height,
                        f"affordance pixel {(x, y)} outside the "
                        f"{config.image_width}x{config.image_height} render")
        self.assertEqual((head.width, head.height), (config.image_width, config.image_height))

        points, pixels = seed_from_affordance_point(head, affordance)
        self.assertEqual(len(points), len(SEED_PIXEL_OFFSETS))
        truth, _rot = self.handle_pose()
        aim = self.handle_aim_point(head)
        # The seeds sit on the imaged surface, so they are offset from the link origin - but
        # they must be on the handle, not on the door face behind it or the room beyond.
        centroid = points.mean(axis=0)
        print(f"[door-track] seed centroid {np.round(centroid, 4).tolist()} is "
              f"{np.linalg.norm(centroid - aim):.4f} m from the aimed lever root and "
              f"{np.linalg.norm(centroid - truth):.4f} m from the latch origin "
              f"{np.round(truth, 4).tolist()}")
        self.assertLess(float(np.linalg.norm(centroid - aim)), 0.05,
                        "the affordance point did not deproject onto the lever root")
        self.assertLess(float(np.max(np.linalg.norm(points - centroid, axis=1))), 0.05,
                        "the seed cross straddles two surfaces")

    def test_perfect_tracker_control_scores_near_zero(self):
        """Control for the *scoring*, not the tracker: ground-truth pixels in, ~0 error out.

        Each seed point is carried forward through the link's rotation, projected into the
        head camera and patch-sampled back through the same path that seeded it. Any error
        left is the renderer's and the lever's round cross-section; if this fails, the offset
        transform - not the tracker - is wrong, and every other assertion in this file would
        be measuring that bug.
        """
        self.close_door()
        head = self.robot.capture_camera_view("head", self.env)
        affordance = self.affordance_point(head)
        points, _pixels = seed_from_affordance_point(head, affordance)
        pos, rot = self.handle_pose()
        local = (points - pos) @ rot                     # rows of R^T @ v
        world_offset = points.mean(axis=0) - pos         # the frozen (wrong) convention

        errors, naive_errors, missed = [], [], 0
        for step in range(FRAMES):
            self.drive_door(step)
            view = self.robot.capture_camera_view("head", self.env)
            pos, rot = self.handle_pose()
            expected = pos + local @ rot.T
            recovered = []
            for want in expected:
                pixel, _z = geometry.project_world_to_pixel(view, want)
                if pixel is None or not geometry.in_bounds(view, pixel):
                    continue
                depth, _info = utils.sample_surface_depth(view.depth, int(round(pixel[0])),
                                                          int(round(pixel[1])))
                if depth is None:
                    continue
                recovered.append(geometry.deproject_pixel_to_world(view, pixel, depth))
            if not recovered:
                missed += 1
                continue
            centroid = np.mean(recovered, axis=0)
            errors.append(float(np.linalg.norm(centroid - expected.mean(axis=0))))
            naive_errors.append(float(np.linalg.norm(centroid - (pos + world_offset))))
            if step % 10 == 0:
                print(f"[door-track] control frame {step:2d} err={errors[-1]:.4f} m "
                      f"({len(recovered)}/{len(expected)} seeds imaged)")

        print(f"[door-track] control: median={np.median(errors):.4f} "
              f"max={np.max(errors):.4f} m over {len(errors)} frames | the same recoveries "
              f"scored against a frozen world offset: median={np.median(naive_errors):.4f} "
              f"max={np.max(naive_errors):.4f} m")
        self.assertEqual(missed, 0, "the handle left the head camera during the arc")
        self.assertLess(float(np.max(errors)), CONTROL_TOL,
                        "a perfect tracker does not score near zero, so the seed-offset "
                        "transform is wrong and the tracking thresholds are meaningless")
        # Same recoveries, wrong convention: measured median 0.064 m vs 0.004 m, peaking at
        # 0.097 m. Compare medians, not maxima - both peak on the same late frames, where the
        # lever's round cross-section dominates either convention.
        self.assertGreater(float(np.median(naive_errors)), 5.0 * float(np.median(errors)),
                           "the rotated and frozen offsets score the same, so this control "
                           "no longer proves anything about the transform")
        self.assertGreater(float(np.max(naive_errors)), MAX_TOL,
                           "a frozen offset would still pass the tracking thresholds")

    def test_a_fixed_world_offset_would_fake_a_growing_error(self):
        """Why the offset is rotated: the naive version invents ~10 cm of error by itself.

        Pure kinematics, no tracker involved - this is the bug the control above rules out.
        """
        self.close_door()
        head = self.robot.capture_camera_view("head", self.env)
        points, _pixels = seed_from_affordance_point(head, self.affordance_point(head))
        pos0, rot0 = self.handle_pose()
        world_offset = points.mean(axis=0) - pos0
        local_offset = rot0.T @ world_offset

        discrepancies = []
        for step in range(FRAMES):
            self.drive_door(step)
            pos, rot = self.handle_pose()
            discrepancies.append(float(np.linalg.norm((pos + rot @ local_offset)
                                                      - (pos + world_offset))))
        hinge, _latch = self.joint_angles()
        print(f"[door-track] fixed-world-offset discrepancy at hinge={hinge:.3f} rad: "
              f"{discrepancies[-1]:.4f} m (max {np.max(discrepancies):.4f} m)")
        self.assertLess(discrepancies[0], 0.01, "the door was not closed at seed time")
        # Measured 0.133 m at 0.765 rad: far past MAX_TOL, so a fixed world offset would fail
        # the tracking tests for reasons that have nothing to do with tracking.
        self.assertGreater(float(np.max(discrepancies)), MAX_TOL,
                           "the door no longer rotates the handle enough for this test to "
                           "distinguish the two offset conventions")

    def test_tracks_the_handle_with_depth_fusion(self):
        """The default 3D lift must follow the handle through the whole swing."""
        run = self.run_arc("depth_fusion",
                           monitor=monitors.object_not_lost(OBJECT, patience=8))
        self.assert_arc_is_tracked(run)
        self.assertEqual(run["session"].lift_errors, 0)

    def test_tracks_the_handle_with_triangulate(self):
        """Same arc, pure-geometry lift - no depth buffer in the 3D step."""
        run = self.run_arc("triangulate")
        self.assert_arc_is_tracked(run)

    def test_triangulate_falls_back_at_two_cameras(self):
        """Recorded, not papered over: what ``triangulate`` actually does on this scene.

        ``tracking_cameras`` is ``("head", "wrist")``. Both seed, but the wrist camera is
        mounted on a robot that stays parked for this task and the handle sits within a few
        pixels of the bottom edge of its frame, so it drops out part-way through the swing:
        measured, the triangulator has its ``min_cameras = 2`` on only ~18 of 40 frames and
        the session falls back to ``depth_fusion`` for the other ~22.

        And where two views *are* available it proves little: at exactly two views the DLT is
        exactly determined, so its reprojection residual carries no outlier signal (see
        ``tests/test_tracker3d.py::test_two_view_residual_is_blind_to_drift``).

        So this arc cannot really discriminate between the two providers - they score within
        0.01 m of each other above - and a third camera is what the comparison needs.
        """
        run = self.run_arc("triangulate")
        providers = [m.get("provider") for m in run["lift_meta"]]
        fell_back = [m for m in run["lift_meta"] if m.get("fell_back_from")]
        seeded_cams = [c for c, t in run["session"].last_report.objects[OBJECT].cams.items()
                       if t.seeded]
        print(f"[door-track] triangulate: {len(fell_back)}/{len(run['lift_meta'])} frames fell "
              f"back to depth_fusion; providers={sorted(set(providers))}; "
              f"first reason={fell_back[0].get('why') if fell_back else None}; "
              f"seeded cams={seeded_cams}; wrist saw the handle on "
              f"{run['wrist_seen']}/{len(run['lift_meta'])} frames")
        self.assertTrue(set(providers) <= {"triangulate", "depth_fusion"})
        self.assertEqual(run["session"].lift_errors, 0,
                         "triangulate raised instead of returning an empty result")
        self.assertTrue(fell_back, "triangulate never fell back; re-read the note above")
        self.assertIn("2 cams", str(fell_back[0].get("why")))
        # Whatever the mix, the object must survive: the fallback exists so an experimental
        # provider can never lose the object.
        self.assertGreaterEqual(len(run["errors"]), int(0.8 * len(run["lift_meta"])))


if __name__ == "__main__":
    unittest.main()
