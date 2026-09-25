"""Camera-visibility scenarios on the PyBullet grasp scene: who sees the object, and when.

The scenarios are blackout schedules (``tests.tracking_eval.blind_schedule``): a blind camera
renders black RGB and invalid depth, so it can neither seed, track nor lift.

* ``head_only``  - the wrist never sees the object; the whole run rests on the head.
* ``wrist_only`` - the head never sees it, so detection at t=0 falls back to the wrist.
* ``handoff``    - seeded in the head only; the wrist must be seeded **from the head's
  estimate** (design Step 0.5, :func:`tracking.health.decide_reseeds`) as soon as it can see
  the object, then the head goes blind and tracking continues on the wrist alone.

The painted disc occluder of the grasp scene (frames 12-19, head) is on in every scenario, so
``handoff`` also covers "the head is occluded while the wrist is still being trusted".

Ground truth is the body pose plus the body-frame offset of each seed point, so besides the
centroid error the harness scores every tracked point against *its own* seed point. That
per-point score is what pins down correspondence: the hand-over used to lose it because
a 2D tracker dropped the seeds it could not track and every later row slid onto the wrong
template point.

Run with::

    python -m pytest tests/test_tracking_scenarios_pybullet.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tracking_eval as te  # noqa: E402

try:
    import pybullet as p
except ImportError:                                  # pragma: no cover - env without pybullet
    p = None

N_FRAMES = 30
OCCLUSION = (12, 19)
WRIST_SEEDABLE_FROM = N_FRAMES // 3                  # handoff: wrist blind before this
HEAD_BLIND_FROM = N_FRAMES // 2                      # handoff: head blind from here on
#: handoff: last frame (exclusive) the template-tracked wrist is still trusted (measured 24).
WRIST_TRUSTED_UNTIL = 23

#: Measured (PyBullet 3.2.x, 256x256, template tracker, grid seeding 20 points, pose_tracker):
#:   head_only   median 0.0031  p95 0.0060  lost 0/30
#:   wrist_only  median 0.0114  p95 0.0436  lost 0/30  (the wrist closes in on the cube and
#:               the fixed-size NCC templates drift under the scale change)
#:   handoff     median 0.0087  p95 0.0128  lost 0/30
#: Before the one-row-per-seed tracker contract the hand-over's wrist-only half fitted at
#: 2-5 cm RMSE (red/black) and wrist_only never produced a pose at all.
TOL = {
    "head_only": {"median": 0.010, "p95": 0.015},
    "wrist_only": {"median": 0.025, "p95": 0.070},
    "handoff": {"median": 0.015, "p95": 0.030},
}
#: Per-point error of the wrist right after its hand-over seed (frames 10-14): measured
#: 0.007-0.012 m. Wrong correspondence shows up here as 5-30 cm.
HANDOFF_POINT_TOL = 0.025


class _Stub(te.SceneDriver):
    name = "stub"
    n_frames = 4
    objects = ()
    cameras = ("head", "wrist")


def _view(value=200):
    from tracking.types import CameraView
    return CameraView(name="head", rgb=np.full((4, 4, 3), value, dtype=np.uint8),
                      depth=np.full((4, 4), 0.5, dtype=np.float32),
                      view_matrix=np.eye(4), projection_matrix=np.eye(4), near=0.01, far=5.0)


class TestBlindSchedule(unittest.TestCase):
    def test_named_scenarios(self):
        self.assertIsNone(te.blind_schedule("occlusion", 30))
        self.assertIsNone(te.blind_schedule(None, 30))
        self.assertEqual(te.blind_schedule("head_only", 30), {"wrist": [(0, None)]})
        self.assertEqual(te.blind_schedule("wrist_only", 30), {"head": [(0, None)]})
        self.assertEqual(te.blind_schedule("handoff", 30),
                         {"wrist": [(0, 10)], "head": [(15, None)]})

    def test_extra_windows_are_appended(self):
        sched = te.blind_schedule("head_only", 30, ["head:3:5", "shoulder:20:"])
        self.assertEqual(sched["wrist"], [(0, None)])
        self.assertEqual(sched["head"], [(3, 5)])
        self.assertEqual(sched["shoulder"], [(20, None)])

    def test_unknown_scenario_is_an_error(self):
        with self.assertRaises(ValueError):
            te.blind_schedule("nope", 30)

    def test_is_blind_windows_are_half_open(self):
        stub = _Stub()
        stub.blind = {"head": [(2, 4)], "wrist": [(0, None)]}
        self.assertEqual([stub.is_blind("head", f) for f in range(5)],
                         [False, False, True, True, False])
        self.assertTrue(stub.is_blind("wrist", 10_000))
        self.assertFalse(stub.is_blind("shoulder", 0))

    def test_seed_camera_falls_back_to_the_first_seeing_camera(self):
        stub = _Stub()
        self.assertEqual(stub.seed_camera(), "head")
        stub.blind = {"head": [(0, None)]}
        self.assertEqual(stub.seed_camera(), "wrist")
        stub.blind = {"head": [(0, 5)], "wrist": [(0, 5)]}
        self.assertEqual(stub.seed_camera(), "head")      # nobody sees: keep the default

    def test_apply_blind_blacks_out_rgb_and_invalidates_depth(self):
        import config
        stub = _Stub()
        stub.blind = {"head": [(1, None)]}
        views = stub.apply_blind({"head": _view()}, 0)
        self.assertEqual(int(views["head"].rgb.max()), 200)
        views = stub.apply_blind({"head": _view()}, 1)
        self.assertEqual(int(views["head"].rgb.max()), 0)
        self.assertTrue(np.all(views["head"].depth < config.track_depth_min))


@unittest.skipIf(p is None, "pybullet is not installed")
class TestVisibilityScenariosPyBullet(unittest.TestCase):
    """One booted grasp scene; each scenario is a fresh driver + session over it."""

    @classmethod
    def setUpClass(cls):
        from test_tracking_pybullet import _boot

        cls.sim, cls.env, cls.robot = _boot()
        cls._cache = {}

    @classmethod
    def tearDownClass(cls):
        if p is not None and p.isConnected():
            p.disconnect()

    def run_scenario(self, scenario, tracker3d="pose_tracker"):
        key = (scenario, tracker3d)
        if key not in self._cache:
            scene = te.GraspSceneDriver(self.sim, self.env, self.robot,
                                        self.env.simenv.object_id, n_frames=N_FRAMES,
                                        occlusion=OCCLUSION, seeding="grid",
                                        blind=te.blind_schedule(scenario, N_FRAMES))
            self._cache[key] = te.run_eval(scene, tracker_provider="template",
                                           tracker3d=tracker3d, keep_frames=True)
        payload = self._cache[key]
        return payload, payload["frames"]["cube"]

    def assert_accuracy(self, scenario, payload):
        agg = payload["aggregate"]
        tol = TOL[scenario]
        self.assertEqual(agg["pct_lost"], 0.0, f"{scenario}: object lost")
        self.assertLess(agg["median_l2_m"], tol["median"], f"{scenario}: {agg}")
        self.assertLess(agg["p95_l2_m"], tol["p95"], f"{scenario}: {agg}")
        self.assertEqual(payload["session"]["lift_errors"], 0)

    # -- head only ---------------------------------------------------------
    def test_head_only_tracks_on_the_head_alone(self):
        payload, frames = self.run_scenario("head_only")
        self.assert_accuracy("head_only", payload)
        self.assertTrue(all(f["cam_status"]["wrist"] == "unseeded" for f in frames),
                        "a blind wrist must never be seeded")

    # -- wrist only --------------------------------------------------------
    def test_wrist_only_seeds_from_the_wrist(self):
        payload, frames = self.run_scenario("wrist_only")
        self.assert_accuracy("wrist_only", payload)
        self.assertTrue(all(f["cam_status"]["head"] == "unseeded" for f in frames))
        self.assertEqual(frames[0]["cam_status"]["wrist"], "ok")
        modes = [f["lift"].get("mode") for f in frames]
        self.assertTrue(all(m is not None for m in modes),
                        f"pose_tracker must fit a pose on the wrist alone: {modes}")
        self.assertEqual(modes[0], "green")

    # -- hand-over ---------------------------------------------------------
    def test_handoff_seeds_the_wrist_from_the_head(self):
        _payload, frames = self.run_scenario("handoff")
        before = [f["cam_status"]["wrist"] for f in frames[:WRIST_SEEDABLE_FROM]]
        self.assertTrue(all(s == "unseeded" for s in before), before)
        self.assertNotEqual(frames[WRIST_SEEDABLE_FROM]["cam_status"]["wrist"], "unseeded",
                            "the wrist must be seeded from the head as soon as it can see")

    def test_handoff_keeps_tracking_on_the_wrist_after_the_head_goes_blind(self):
        payload, frames = self.run_scenario("handoff")
        self.assert_accuracy("handoff", payload)
        for f in frames[HEAD_BLIND_FROM:]:
            self.assertIn("head", f["blind_cams"])
        # Known limit, measured: from ~frame 24 the wrist is a few cm from the cube and its
        # fixed-size NCC templates drift under the scale change until the jump gate rejects
        # them; the pose then coasts on the Kalman filter (error stays ~1 cm, never lost).
        for f in frames[HEAD_BLIND_FROM:WRIST_TRUSTED_UNTIL]:
            self.assertEqual(f["cam_status"]["wrist"], "ok", f)

    def test_handoff_wrist_points_keep_their_correspondence(self):
        _payload, frames = self.run_scenario("handoff")
        # Head blind -> every scored point is a wrist point, judged against its own seed.
        errs = [f["point_err_m"] for f in frames[HEAD_BLIND_FROM:HEAD_BLIND_FROM + 5]]
        self.assertTrue(all(e is not None for e in errs), errs)
        self.assertLess(float(np.median(errs)), HANDOFF_POINT_TOL, errs)

    def test_handoff_seeding_does_not_depend_on_the_3d_lift(self):
        _payload, frames = self.run_scenario("handoff", tracker3d="depth_fusion")
        self.assertNotEqual(frames[WRIST_SEEDABLE_FROM]["cam_status"]["wrist"], "unseeded")


if __name__ == "__main__":
    unittest.main()
