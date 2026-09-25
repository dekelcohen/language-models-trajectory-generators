"""Real-time pipeline model (tracking/realtime.py) and the session's tick/defer contract.

Simulator-free and deterministic: latency comes from ``zero`` / ``fixed_*`` profiles or
explicit wall times, never from the machine running the test.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from tracking.realtime import LatencyModel, RealtimePipeline, policy_for  # noqa: E402


class LatencyModelTest(unittest.TestCase):
    def test_zero_measured_and_scale(self):
        self.assertEqual(LatencyModel("zero").cost(0.5), 0.0)
        self.assertAlmostEqual(LatencyModel("measured").cost(0.04), 0.04)
        self.assertAlmostEqual(LatencyModel("measured", scale=2.0).cost(0.04), 0.08)

    def test_fixed_profile_ignores_wall_time(self):
        model = LatencyModel("fixed_20ms")
        self.assertTrue(model.deterministic)
        self.assertAlmostEqual(model.cost(5.0), 0.020)

    def test_device_profile_swaps_flush_cost_only(self):
        model = LatencyModel("a1000_fp16")
        # 4.1 s measured of which 4.0 s was one CPU flush: 0.1 s CPU stays, flush -> 0.223 s
        cost = model.cost(4.1, {"flushes": 1, "flush_s": 4.0})
        self.assertAlmostEqual(cost, 0.1 + config.track_latency_profiles["a1000_fp16"]["cotracker_flush_s"])
        self.assertAlmostEqual(model.cost(0.01, {"flushes": 0, "flush_s": 0.0}), 0.01)

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            LatencyModel("warp_drive")

    def test_policy_for(self):
        self.assertEqual(policy_for("cotracker"), "queue")
        self.assertEqual(policy_for("template"), "drop")
        self.assertEqual(policy_for("klt"), "drop")


class PipelineTest(unittest.TestCase):
    def _drive(self, pipe, wall_s, sim_dt=1.0 / 240.0, duration=0.5, timing=None):
        """Poll like the robot does (every physics step); return accepted frames/publishes."""
        accepted, published = [], []
        n = int(round(duration / sim_dt))
        for step in range(n + 1):
            now = step * sim_dt
            published += [(now, r.frame_idx) for r in pipe.publish(now)]
            frame = pipe.next_frame(now)
            if frame is not None and pipe.accepts(frame[1]):
                pipe.submit(frame[0], frame[1], wall_s, timing)
                accepted.append(frame[0])
                published += [(now, r.frame_idx) for r in pipe.publish(now)]
        return accepted, published

    def test_camera_clock_starts_at_first_poll(self):
        pipe = RealtimePipeline(camera_fps=10, latency="zero")
        self.assertEqual(pipe.next_frame(12.34), (0, 12.34))
        self.assertIsNone(pipe.next_frame(12.38))
        idx, t = pipe.next_frame(12.44)
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(t, 12.44)
        self.assertEqual(pipe.stats["frames_missed"], 0)

    def test_skipping_polls_counts_missed_frames(self):
        pipe = RealtimePipeline(camera_fps=10, latency="zero")
        pipe.next_frame(0.0)
        self.assertEqual(pipe.next_frame(0.35)[0], 3)       # frames 1, 2 were stepped over
        self.assertEqual(pipe.stats["frames_missed"], 2)

    def test_zero_latency_takes_and_publishes_every_frame(self):
        pipe = RealtimePipeline(camera_fps=30, latency="zero")
        accepted, published = self._drive(pipe, wall_s=1.0, duration=0.5)
        self.assertEqual(accepted, list(range(16)))
        self.assertEqual([f for _t, f in published], accepted)
        for now, idx in published:
            self.assertAlmostEqual(now, pipe.frame_time(idx), places=6)

    def test_drop_policy_skips_frames_while_busy(self):
        # 100 ms per frame at 30 fps: a latest-frame loop takes every 3rd-4th frame.
        pipe = RealtimePipeline(camera_fps=30, latency="fixed_100ms", policy="drop")
        accepted, published = self._drive(pipe, wall_s=0.0, duration=1.0)
        gaps = np.diff(accepted)
        self.assertTrue(np.all(gaps >= 3), accepted)
        self.assertTrue(np.all(gaps <= 4), accepted)
        self.assertGreater(pipe.stats["frames_dropped_busy"], 15)
        for now, idx in published:          # each result appears ~100 ms after its frame
            self.assertGreaterEqual(now - pipe.frame_time(idx), 0.100 - 1e-6)
            self.assertLess(now - pipe.frame_time(idx), 0.100 + 1.0 / 240.0 + 1e-6)

    def test_queue_policy_feeds_every_frame_and_backs_up(self):
        # A tracker slower than the camera: every frame is processed, the queue (and the
        # age of the published result) grows - what a too-slow device really does.
        pipe = RealtimePipeline(camera_fps=30, latency="fixed_100ms", policy="queue")
        accepted, published = self._drive(pipe, wall_s=0.0, duration=1.0)
        self.assertEqual(accepted, list(range(31)))
        self.assertEqual(pipe.stats["frames_dropped_busy"], 0)
        waits = [r.queue_wait_s for r in pipe.pending]
        self.assertGreater(pipe.stats["max_queue_wait_s"], 0.5)
        self.assertEqual([f for _t, f in published], sorted(f for _t, f in published))
        self.assertTrue(waits == sorted(waits))

    def test_queue_policy_keeps_up_when_fast_enough(self):
        pipe = RealtimePipeline(camera_fps=30, latency="fixed_20ms", policy="queue")
        self._drive(pipe, wall_s=0.0, duration=1.0)
        self.assertAlmostEqual(pipe.stats["max_queue_wait_s"], 0.0)

    def test_min_period_caps_the_rate(self):
        pipe = RealtimePipeline(camera_fps=30, latency="zero", min_period_s=0.1)
        accepted, _ = self._drive(pipe, wall_s=0.0, duration=1.0)
        self.assertTrue(np.all(np.diff(accepted) >= 3), accepted)

    def test_age(self):
        pipe = RealtimePipeline(camera_fps=10, latency="fixed_20ms")
        self.assertIsNone(pipe.age(0.0))
        pipe.next_frame(0.0)
        pipe.submit(0, 0.0, 0.0)
        pipe.publish(0.01)
        self.assertIsNone(pipe.age(0.01))                 # not available yet
        pipe.publish(0.02)
        self.assertAlmostEqual(pipe.age(0.05), 0.05)

    def test_fps_zero_rejected(self):
        with self.assertRaises(ValueError):
            RealtimePipeline(camera_fps=0)


class KalmanTimeUnitsTest(unittest.TestCase):
    """The pose filter steps in reference frames of sim time, not in tracked frames."""

    def test_pose_tracker_dt_from_sim_time(self):
        from providers.tracker3d.pose_tracker import PoseTracker3D, _ObjState
        from tracking.kalman import RigidBodyKF

        tracker = PoseTracker3D()
        st = _ObjState(template=np.zeros((4, 3)), centroid=np.zeros(3), kf=RigidBodyKF())
        self.assertEqual(tracker._step_dt(st), 1.0)            # no clock: 1 per lift
        tracker.set_time(1.0)
        self.assertEqual(tracker._step_dt(st), 1.0)            # first timed lift
        tracker.set_time(1.0 + 1.0 / 30.0)
        self.assertAlmostEqual(tracker._step_dt(st), config.track_kf_ref_hz / 30.0)

    def test_coast_budget_is_time_not_frames(self):
        from tracking.kalman import RigidBodyKF

        kf = RigidBodyKF()
        kf.init(np.zeros(3), np.eye(3))
        for _ in range(3 * int(kf.cfg.max_coast)):            # 30 fps for max_coast ref-frames
            kf.predict(1.0 / 3.0)
        self.assertFalse(kf.stale)
        kf.predict(1.0 / 3.0)
        self.assertTrue(kf.stale)

    def test_same_motion_same_prediction_at_any_rate(self):
        from tracking.kalman import RigidBodyKF

        def run(steps_per_ref):
            kf = RigidBodyKF()
            kf.init(np.zeros(3), np.eye(3))
            dt = 1.0 / steps_per_ref
            for k in range(1, 10 * steps_per_ref + 1):
                kf.predict(dt)
                kf.update(np.array([0.01 * k * dt, 0.0, 0.0]), np.eye(3))
            return kf.velocity[0]

        # 1 cm per reference frame, sampled at 1x and 3x: the velocity estimate agrees
        self.assertAlmostEqual(run(1), 0.01, delta=0.002)
        self.assertAlmostEqual(run(3), 0.01, delta=0.002)


class SessionTickTest(unittest.TestCase):
    """The session commits a result only when its latency has elapsed."""

    def setUp(self):
        import tracking_eval as te
        from tracking.session import TrackingSession

        self.scene = te.SyntheticSphereScene(n_frames=10, occlusion=None)
        self.scene.reset()
        self.make = lambda **kw: TrackingSession(cameras=("head", "wrist"), provider="template",
                                                 write_jsonl=False, **kw)

    def _seeded(self, session):
        session.add_target("ball", self.scene.seed_points("ball"))
        return session

    def test_no_pipeline_tick_is_on_frame(self):
        session = self._seeded(self.make())
        self.assertIsNone(session.pipeline)
        report = session.tick(0.0, views_fn=lambda: self.scene.views(0))
        self.assertIs(session.last_report, report)
        self.assertIsNone(report.sim_time)

    def test_deferred_publish(self):
        session = self._seeded(self.make(camera_fps=10, latency="fixed_100ms"))
        self.assertEqual(session.pipeline.policy, "drop")
        report = session.tick(0.0, views_fn=lambda: self.scene.views(0))
        self.assertIsNotNone(report)
        self.assertIsNone(session.last_report)               # computed, not yet usable
        self.assertAlmostEqual(report.available_at, 0.1)
        self.assertAlmostEqual(report.latency_s, 0.1)
        self.assertIn("track_s", report.timing)
        self.assertIsNone(session.tick(0.05, views_fn=lambda: self.scene.views(0)))
        self.assertIsNone(session.last_report)
        session.tick(0.1, views_fn=lambda: self.scene.views(1))
        self.assertIs(session.last_report, report)            # published at t = 0.1
        self.assertAlmostEqual(session.estimate_age(0.1), 0.1)
        d = report.to_dict()
        self.assertAlmostEqual(d["latency_s"], 0.1)
        self.assertEqual(d["camera_frame"], 0)
        self.assertIn("realtime", session.summary())

    def test_abort_takes_effect_at_publish_time(self):
        from tracking.types import STATUS_ABORT, MonitorResult

        session = self._seeded(self.make(camera_fps=10, latency="fixed_100ms",
                                         monitor=lambda state: MonitorResult(STATUS_ABORT, "test")))
        session.tick(0.0, views_fn=lambda: self.scene.views(0))
        self.assertFalse(session.aborted)                     # decided, not yet known
        session.tick(0.1, views_fn=lambda: self.scene.views(1))
        self.assertTrue(session.aborted)
        self.assertEqual(session.abort_reason, "test")

    def test_drain_publishes_in_flight(self):
        session = self._seeded(self.make(camera_fps=10, latency="fixed_100ms"))
        report = session.tick(0.0, views_fn=lambda: self.scene.views(0))
        session.drain()
        self.assertIs(session.last_report, report)


class HarnessRealtimeTest(unittest.TestCase):
    """The harness's real-time loop on the simulator-free scene."""

    def _run(self, **kw):
        import tracking_eval as te

        scene = te.SyntheticSphereScene(n_frames=12, occlusion=None)
        return te.run_eval(scene, tracker_provider="template", **kw)

    def test_lockstep_equivalent(self):
        # camera fps == motion rate with zero latency is the historical loop, frame for frame
        base = self._run()
        rt = self._run(camera_fps=10, latency="zero", motion_rate_hz=10)
        self.assertEqual(rt["aggregate"]["n_frames"], base["aggregate"]["n_frames"])
        self.assertAlmostEqual(rt["aggregate"]["median_l2_m"], base["aggregate"]["median_l2_m"],
                               places=9)
        self.assertEqual(rt["realtime"]["n_warmup_frames"], 0)
        self.assertEqual(rt["config"]["timing"]["mode"], "realtime")
        self.assertEqual(base["config"]["timing"]["mode"], "lockstep")

    def test_latency_costs_accuracy_and_shows_as_age(self):
        fast = self._run(camera_fps=30, latency="zero")
        slow = self._run(camera_fps=30, latency="fixed_100ms")
        self.assertGreater(slow["realtime"]["stats"]["frames_dropped_busy"], 0)
        self.assertGreater(slow["realtime"]["n_warmup_frames"], 0)
        self.assertGreater(slow["aggregate"]["age_s_median"], fast["aggregate"]["age_s_median"])
        self.assertGreater(slow["aggregate"]["median_l2_m"], fast["aggregate"]["median_l2_m"])
        frames = slow["frames"]["ball"]
        self.assertIn("age_s", frames[0])


if __name__ == "__main__":
    unittest.main()
