import unittest

import numpy as np

from helpers import object_displacement as od


def _scene(pole=True, arm=False):
    rng = np.random.default_rng(0)
    rgb = (rng.integers(80, 200, (64, 64, 3))).astype(np.uint8)   # textured door/background
    depth = np.full((64, 64), 2.0, np.float32)
    if pole:
        rgb[10:50, 28:36] = (90, 90, 95)
        rgb[10:50:4, 28:36] = (60, 60, 60)
        depth[10:50, 28:36] = 1.2
    if arm:
        rgb[5:55, 20:44] = (240, 240, 240)
        depth[5:55, 20:44] = 0.6
    return rgb, depth


class ObjectDisplacementTest(unittest.TestCase):
    def setUp(self):
        rgb, depth = _scene()
        mask = np.zeros((64, 64), bool)
        mask[10:50, 28:36] = True
        self.snap = od.snapshot("gray cylinder", rgb, depth, mask, camera_position=[0, 0, 1])

    def test_removed_object_left_place(self):
        f = od.compare(self.snap, *_scene(pole=False), camera_position=[0, 0, 1])
        self.assertEqual(f["verdict"], "left_place")
        self.assertIn("LEFT ITS ORIGINAL PLACE", od.format_facts([f]))

    def test_untouched_object_in_place(self):
        self.assertEqual(od.compare(self.snap, *_scene())["verdict"], "in_place")

    def test_arm_in_front_is_not_reported_as_gone(self):
        self.assertEqual(od.compare(self.snap, *_scene(arm=True))["verdict"], "hidden")

    def test_moved_camera_gives_no_evidence(self):
        self.assertIsNone(od.compare(self.snap, *_scene(pole=False), camera_position=[0.1, 0, 1]))

    def test_retry_hint_only_for_relocate_subtasks(self):
        f = od.compare(self.snap, *_scene(pole=False))
        self.assertIn("task_completed", od.retry_hint([f], "Move the gray cylinder out of the way of the door"))
        self.assertIsNone(od.retry_hint([f], "Open the wooden door"))

    def test_same_blob_for_two_labels_is_flagged(self):
        rgb, depth = _scene()
        big = np.zeros((64, 64), bool)
        big[5:55, 20:44] = True
        door = od.snapshot("door", rgb, depth, big)
        facts = [od.compare(s, *_scene(pole=False)) for s in (self.snap, door)]
        od.mark_shared_regions([self.snap, door], facts)
        self.assertEqual(facts[0]["same_region_as"], ["door"])
        self.assertIn("SAME image region", od.format_facts(facts))

    def test_run_baseline_reports_object_moved_by_earlier_subtask(self):
        from types import SimpleNamespace
        from unittest import mock
        import api as api_mod
        from task_state import TaskState
        mask = np.zeros((64, 64), bool)
        mask[10:50, 28:36] = True
        a = object.__new__(api_mod.API)
        a.logger = SimpleNamespace(info=lambda *_: None)
        a.cam_info, a.head_camera_position = {}, [0, 0, 1]
        a.trajectory_step = 0
        a.task = TaskState(command="clear the pole", start_trajectory_step=0)
        with mock.patch.object(od, "metric_depth", side_effect=lambda d, _c: d):
            a._snapshot_detections(*_scene(), [mask], ["gray cylinder"])       # before any motion
            a.trajectory_step = 7                                               # subtask 1 moved it
            a.task = TaskState(command="open the door", start_trajectory_step=7)
            a._snapshot_detections(*_scene(pole=False), [mask], ["gray cylinder"])
            a._capture_head_image_and_depth = lambda: _scene(pole=False)
            facts, text = a._displacement_review_evidence()
        self.assertEqual(facts[0]["verdict"], "in_place")       # unchanged within this subtask
        self.assertIn("SINCE THE START OF THE WHOLE COMMAND", text)
        self.assertIn("'gray cylinder': LEFT ITS ORIGINAL PLACE", text.split("SINCE THE START")[1])


class TouchEvidenceTest(unittest.TestCase):
    def test_insert_and_pull_outcomes_are_reported(self):
        from helpers.touch_evidence import format_touch_evidence
        txt = format_touch_evidence([
            {"kind": "insert", "result": {"outcome": "jammed",
                                          "events": [(1.0, "probe", "contact F=6.0N after 3.0mm")]}},
            {"kind": "pull", "result": {"outcome": "opened", "travel_cm": 21.0, "swept_deg": 40.0}},
            {"kind": "insert", "result": {"outcome": "disabled"}}])
        self.assertIn("NOT inserted", txt)
        self.assertIn("contact F=6.0N", txt)
        self.assertIn("OPENED", txt)
        self.assertNotIn("disabled", txt)
        self.assertIsNone(format_touch_evidence([{"kind": "pull", "result": {"outcome": "disabled"}}]))


if __name__ == "__main__":
    unittest.main()
