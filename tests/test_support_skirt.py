"""``utils._drop_support_skirt``: floor points bleeding into a small object's cluster are
dropped (USB plug on a stand), while a tight object cluster is returned unchanged."""

import logging
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402

utils.logger = logging.getLogger("test_support_skirt")


def _box(center, size, n, rng):
    return np.asarray(center) + (rng.random((n, 3)) - 0.5) * np.asarray(size)


class DropSupportSkirtTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_floor_skirt_is_dropped(self):
        plug = _box([-0.12, 0.45, 0.054], [0.025, 0.016, 0.008], 150, self.rng)
        stand = _box([-0.12, 0.45, 0.025], [0.02, 0.01, 0.05], 100, self.rng)
        # Floor patch spilling toward the camera (-y), as in the real capture.
        floor = _box([-0.12, 0.41, 0.001], [0.10, 0.03, 0.002], 200, self.rng)
        out = utils._drop_support_skirt(np.vstack([plug, stand, floor]))
        self.assertGreater(float(out[:, 2].min()), 0.002)  # no floor points left
        self.assertGreater(len(out), 200)
        self.assertAlmostEqual(float(np.mean(out[:, 1])), 0.45, delta=0.005)

    def test_tight_cluster_unchanged(self):
        cube = _box([0.3, 0.0, 0.03], [0.06, 0.06, 0.06], 400, self.rng)
        out = utils._drop_support_skirt(cube)
        self.assertEqual(len(out), len(cube))


if __name__ == "__main__":
    unittest.main()
