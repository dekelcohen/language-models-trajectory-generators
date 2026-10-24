"""Do tracked-motion facts separate the door-pull outcomes a VLM reviewer must tell apart?

Pure math (no sim/GPU/VLM): synthetic door pulls with 2 mm point noise, scored against the
hinge angle used to generate them. No hinge prior is given to the code under test (neither
the sim nor the real robot knows where the hinge is); the hinge below only generates data.
Covers the cases that motivate feeding tracking to the reviewer:

  * closed / slightly open / wide open door - reported angle band must match,
  * arm pulls the wrong way (radially instead of along the hinge arc) and loses the handle,
  * grasp slips mid-arc (door stops, gripper keeps going),
  * latched door pulled straight out: right direction, door did not move,
  * control: a correct full pull raises no warning (no false alarms),
  * the recommended next pull direction (arc tangent / face normal).

Run with::

    python -m pytest tests/test_tracking_review_evidence.py -q
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.review_evidence import format_evidence, summarize_pull  # noqa: E402

HINGE_POINT = np.zeros(3)
R_HANDLE = 0.4
#: 6 points on the handle/door plate, ~10 cm across, in the door plane (x-z at angle 0)
#: with a little out-of-plane relief (the handle). The door opens towards +y.
CLOUD = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.01], [0.0, 0.01, 0.09],
                  [-0.08, 0.0, 0.04], [0.05, 0.02, -0.05], [0.04, 0.0, 0.07]]) \
    + np.array([R_HANDLE, 0.0, 1.0])
#: camera / robot side of the door
VIEW = np.array([R_HANDLE, 1.5, 1.2])
OUT_NORMAL = np.array([0.0, 1.0, 0.0])
N_FRAMES, DT, NOISE = 40, 0.1, 0.002


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def door_pull(angles_deg, grip_fn=None, lost_after=None, seed=0):
    """Frames of the handle cloud for a per-frame hinge angle; gripper follows by default."""
    rng = np.random.default_rng(seed)
    frames, grip = [], []
    for i, a in enumerate(angles_deg):
        pts = (rot_z(np.radians(a)) @ CLOUD.T).T + rng.normal(0, NOISE, CLOUD.shape)
        if lost_after is not None and i >= lost_after:
            pts = np.full_like(pts, np.nan)
        frames.append(pts)
        base = pts.mean(0) if lost_after is None or i < lost_after \
            else (rot_z(np.radians(a)) @ CLOUD.mean(0))
        grip.append(grip_fn(i, base) if grip_fn else base)
    return frames, grip


def ramp(final_deg, start=5, end=30):
    return [float(np.interp(i, [start, end], [0.0, final_deg])) for i in range(N_FRAMES)]


def facts_for(frames, grip):
    return summarize_pull(frames, grip, dt=DT, view_point=VIEW)


def tangent_at(deg):
    """True opening direction at the handle for hinge angle ``deg`` (scoring only)."""
    return rot_z(np.radians(deg)) @ OUT_NORMAL


def cos_to(vec, ref):
    v = np.asarray(vec, dtype=float)
    return float(v @ ref / np.linalg.norm(v))


class TestOpeningAngleBand(unittest.TestCase):
    def test_closed_door_is_reported_closed(self):
        f = facts_for(*door_pull([0.0] * N_FRAMES))
        self.assertEqual(f["angle_label"], "closed", f)

    def test_slightly_open_door_is_not_called_closed_or_wide(self):
        f = facts_for(*door_pull(ramp(9.0)))
        self.assertEqual(f["angle_label"], "slightly open", f)
        self.assertAlmostEqual(f["angle_deg"], 9.0, delta=2.5)
        self.assertEqual(f["flags"], [], "a successful slight opening must not raise warnings")

    def test_wide_open_door_is_reported_wide(self):
        f = facts_for(*door_pull(ramp(45.0)))
        self.assertEqual(f["angle_label"], "wide open", f)
        self.assertAlmostEqual(f["angle_deg"], 45.0, delta=3.0)


class TestPullFailures(unittest.TestCase):
    def test_correct_pull_has_no_false_alarm(self):
        for final in (9.0, 25.0, 45.0):
            f = facts_for(*door_pull(ramp(final), seed=int(final)))
            self.assertEqual(f["flags"], [], (final, f))

    def test_wrong_direction_pull_and_lost_handle_is_flagged(self):
        # Gripper drags along the door face (+x, radially away from the hinge) instead of
        # out of it; the door barely moves and the handle drops out of the gripper then
        # out of view. Judged against the measured face normal - no hinge prior.
        def radial(i, base):
            return base + np.array([min(max(i - 5, 0), 25) * 0.007, 0.0, 0.0])

        frames, grip = door_pull(ramp(1.0), grip_fn=radial, lost_after=30)
        f = facts_for(frames, grip)
        self.assertIn("pull_direction_wrong", f["flags"], f)
        self.assertEqual(f["pull_alignment_ref"], "door_face_normal", f)
        self.assertIn("handle_did_not_follow_gripper", f["flags"], f)
        self.assertIn("handle_left_gripper", f["flags"], f)
        self.assertEqual(f["angle_label"], "closed")
        # next attempt: pull straight out of the door front
        self.assertEqual(f["recommended_pull_dir_source"], "face_normal_after_wrong_dir")
        self.assertGreater(cos_to(f["recommended_pull_dir"], OUT_NORMAL), 0.95, f)

    def test_latched_door_pulled_straight_out_is_right_direction(self):
        # Door stays latched; the gripper pulls along the outward normal and the handle
        # does not follow. Direction is right - the failure is the latch/grasp.
        def straight_out(i, base):
            return base + OUT_NORMAL * min(max(i - 5, 0), 25) * 0.007

        f = facts_for(*door_pull([0.0] * N_FRAMES, grip_fn=straight_out))
        self.assertNotIn("pull_direction_wrong", f["flags"], f)
        self.assertIn("handle_did_not_follow_gripper", f["flags"], f)
        self.assertGreater(f["pull_alignment_cos"], 0.9, f)
        self.assertEqual(f["recommended_pull_dir_source"], "face_normal")
        self.assertIn("pushing", format_evidence(f))

    def test_slip_mid_arc_reports_partial_angle_and_detach(self):
        angles = [float(np.interp(i, [5, 15, 16, 30], [0, 12, 12, 12])) for i in range(N_FRAMES)]

        def slipping(i, base):
            return base + np.array([0.0, max(i - 15, 0) * 0.012, 0.0])

        f = facts_for(*door_pull(angles, grip_fn=slipping))
        self.assertIn("handle_left_gripper", f["flags"], f)
        self.assertEqual(f["angle_label"], "slightly open")
        self.assertLessEqual(f["detached_at_s"], 2.6)
        # the door started opening: recommend its arc direction at the handle's last pose
        self.assertGreater(cos_to(f["recommended_pull_dir"], tangent_at(12.0)), 0.95, f)

    def test_visible_subset_change_is_not_a_slip(self):
        # Points dropping in/out of view shift a naive centroid by the cloud's spread
        # (here ~9 cm: gripper at point 1, alternating views of {0, 1} and {0, 3, 4, 5}).
        angles = ramp(9.0)
        frames, _ = door_pull(angles)
        grip = [rot_z(np.radians(a)) @ CLOUD[1] for a in angles]
        for i, pts in enumerate(frames):
            pts[2] = np.nan
            if (i // 4) % 2:
                pts[1] = np.nan
            else:
                pts[3:] = np.nan
        f = facts_for(frames, grip)
        self.assertEqual(f["flags"], [], f)

    def test_sparse_measurements_do_not_fake_a_detach(self):
        # Buffering trackers (CoTracker3) give a trusted pose only every ~8 frames.
        frames, grip = door_pull(ramp(9.0))
        for i, pts in enumerate(frames):
            if i % 8:
                pts[:] = np.nan
        f = facts_for(frames, grip)
        self.assertEqual(f["flags"], [], f)
        self.assertEqual(f["angle_label"], "slightly open", f)


class TestRecommendation(unittest.TestCase):
    def test_opened_door_recommends_arc_tangent_and_estimates_hinge(self):
        f = facts_for(*door_pull(ramp(45.0)))
        self.assertEqual(f["recommended_pull_dir_source"], "arc_tangent", f)
        self.assertGreater(cos_to(f["recommended_pull_dir"], tangent_at(45.0)), 0.97, f)
        h = f["hinge_estimate"]
        self.assertLess(np.linalg.norm(np.asarray(h["axis_point"])[:2] - HINGE_POINT[:2]), 0.05, h)
        self.assertGreater(abs(h["axis_dir"][2]), 0.98, h)
        self.assertAlmostEqual(h["radius_m"], R_HANDLE, delta=0.05)

    def test_face_normal_is_measured_outwards(self):
        f = facts_for(*door_pull([0.0] * N_FRAMES))
        self.assertGreater(cos_to(f["face_normal"], OUT_NORMAL), 0.95, f)

    def test_face_normal_orientation_without_a_view_point(self):
        # production has no camera position here: the gripper (always on the robot side,
        # here pulling straight out) orients the normal even though it starts on the handle
        def straight_out(i, base):
            return base + OUT_NORMAL * min(max(i - 5, 0), 25) * 0.007

        f = summarize_pull(*door_pull([0.0] * N_FRAMES, grip_fn=straight_out), dt=DT)
        self.assertGreater(cos_to(f["face_normal"], OUT_NORMAL), 0.95, f)
        self.assertNotIn("pull_direction_wrong", f["flags"], f)


class TestTrackingFramesAdapter(unittest.TestCase):
    """Production path: TrackingReporter frame dicts (pose + gripper FK) -> evidence."""

    @staticmethod
    def _quat_z(deg):
        h = np.radians(deg) / 2.0
        return [0.0, 0.0, float(np.sin(h)), float(np.cos(h))]

    def test_pose_frames_give_the_same_facts_and_skip_predictions(self):
        from tracking.review_evidence import evidence_from_tracking_frames
        angles = ramp(30.0)
        frames = []
        for i, a in enumerate(angles):
            grip = rot_z(np.radians(a)) @ CLOUD.mean(0)
            source = "predicted" if i % 5 == 4 else "measured"     # coasting frames ignored
            frames.append({
                "trajectory_step": 100 + i, "sim_time": i * DT,
                "gripper": {"world_pos": grip.tolist()},
                "objects": {"door_handle": {
                    "world_point": None, "predicted": False, "lost": False,
                    "pose": {"quat_xyzw": self._quat_z(a), "t": [0.0, 0.0, 0.0],
                             "source": source, "tentative": False}}}})
        stale = dict(frames[0], trajectory_step=5)               # previous attempt
        facts, text = evidence_from_tracking_frames([stale] + frames, CLOUD, "door_handle",
                                                    start_step=100)
        self.assertEqual(facts["n_frames"], N_FRAMES)
        self.assertEqual(facts["flags"], [], facts)
        self.assertAlmostEqual(facts["angle_deg"], 30.0, delta=3.0)
        self.assertGreater(cos_to(facts["recommended_pull_dir"], tangent_at(30.0)), 0.95, facts)
        self.assertIn("RECOMMENDED NEXT PULL DIRECTION", text)


class TestEvidenceText(unittest.TestCase):
    def test_text_carries_numbers_and_warnings(self):
        def radial(i, base):
            return base + np.array([min(max(i - 5, 0), 25) * 0.007, 0.0, 0.0])

        txt = format_evidence(facts_for(*door_pull(ramp(1.0), grip_fn=radial, lost_after=30)))
        self.assertIn("TRACKING EVIDENCE", txt)
        self.assertIn("WARNING", txt)
        self.assertIn("wrong way", txt)

    def test_clean_pull_text_has_no_warning(self):
        txt = format_evidence(facts_for(*door_pull(ramp(30.0))))
        self.assertNotIn("WARNING", txt)
        self.assertIn("ALIGNED", txt)
        self.assertIn("RECOMMENDED NEXT PULL DIRECTION", txt)


if __name__ == "__main__":
    unittest.main()
