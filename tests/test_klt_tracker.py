"""KLT (pyramidal Lucas-Kanade) provider: accuracy on a translating texture and the
one-row-per-seed contract (providers/trackers/base.py)."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from providers.trackers.factory import SUPPORTED, get_tracker  # noqa: E402


def _texture(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(h // 4, w // 4), dtype=np.uint8)
    import cv2

    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def _shift(img, dx, dy):
    import cv2

    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


class KLTTrackerTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("klt", SUPPORTED)
        self.assertEqual(get_tracker("klt").name, "klt")

    def test_follows_translation(self):
        base = _texture()
        seeds = np.array([[100.0, 80.0], [160.0, 120.0], [220.0, 150.0], [130.0, 170.0]])
        tracker = get_tracker("klt")
        tracker.init(base, seeds)
        for k in range(1, 11):
            result = tracker.update(_shift(base, 2.5 * k, 1.0 * k))
        self.assertTrue(np.all(result.visible))
        err = np.linalg.norm(result.points - (seeds + [25.0, 10.0]), axis=1)
        self.assertLess(float(err.max()), 0.5)
        self.assertEqual(result.meta["provider"], "klt")

    def test_one_row_per_seed_and_dead_slots(self):
        base = _texture()
        # the last seed starts near the right edge and leaves the image as the scene shifts
        seeds = np.array([[100.0, 80.0], [160.0, 120.0], [312.0, 120.0]])
        tracker = get_tracker("klt")
        tracker.init(base, seeds)
        for k in range(1, 6):
            result = tracker.update(_shift(base, 3.0 * k, 0.0))
            self.assertEqual(result.points.shape, (3, 2))
            self.assertEqual(result.visible.shape, (3,))
        self.assertFalse(result.visible[2])
        self.assertTrue(np.all(result.visible[:2]))
        self.assertEqual(result.meta["alive"], 2)

    def test_forward_backward_rejects_a_scene_cut(self):
        tracker = get_tracker("klt")
        tracker.init(_texture(seed=0), np.array([[100.0, 80.0], [160.0, 120.0]]))
        result = tracker.update(_texture(seed=7))       # unrelated image
        self.assertLess(int(np.count_nonzero(result.visible)), 2)

    def test_uninitialised_update_is_empty(self):
        result = get_tracker("klt").update(_texture())
        self.assertEqual(len(result.points), 0)


if __name__ == "__main__":
    unittest.main()
