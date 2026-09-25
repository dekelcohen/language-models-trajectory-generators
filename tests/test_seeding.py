"""Unit tests for ``tracking.seeding`` (mask-grid seeding, affordance first)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_tracker3d import make_view  # noqa: E402

from tracking.seeding import erode, farthest_point_order, mask_grid_seed, sim_link_mask  # noqa: E402


class TestSimLinkMask(unittest.TestCase):
    def test_decodes_pybullet_object_and_link(self):
        view = make_view("head", (0.9, 1.1, 0.8))
        seg = np.full(view.depth.shape, -1, dtype=np.int64)
        seg[10:20, 10:20] = 7 + ((2 + 1) << 24)         # body 7, link 2
        seg[30:40, 30:40] = 7                            # body 7, base link (-1)
        seg[50:60, 50:60] = 3 + ((2 + 1) << 24)          # other body, link 2
        view.segmentation = seg
        self.assertEqual(int(sim_link_mask(view, 7, 2).sum()), 100)
        self.assertEqual(int(sim_link_mask(view, 7, -1).sum()), 100)
        self.assertEqual(int(sim_link_mask(view, 7, None).sum()), 200)

    def test_no_segmentation_is_none(self):
        self.assertIsNone(sim_link_mask(make_view("head", (0.9, 1.1, 0.8)), 1))


class TestMaskGridSeed(unittest.TestCase):
    def setUp(self):
        self.view = make_view("head", (0.9, 1.1, 0.8))
        self.view.depth[:] = 1.2
        self.mask = np.zeros(self.view.depth.shape, dtype=bool)
        self.mask[100:140, 80:200] = True

    def test_affordance_first_then_spread_inside_eroded_mask(self):
        world, px, info = mask_grid_seed(self.view, self.mask, affordance_px=[(150, 120)],
                                         n_points=12, erode_px=2)
        self.assertEqual(len(world), 12)
        np.testing.assert_allclose(px[0], [150, 120])
        core = erode(self.mask, 2)
        self.assertTrue(all(core[int(y), int(x)] for x, y in px))
        # spread: covers most of the 120 px wide mask, not a cluster
        self.assertGreater(np.ptp(px[:, 0]), 90)
        self.assertGreater(info["extent_m"], 0.05)

    def test_skips_pixels_without_depth(self):
        self.view.depth[:, 80:150] = 0.0                  # invalid half
        _world, px, _info = mask_grid_seed(self.view, self.mask, n_points=10)
        self.assertTrue(np.all(px[:, 0] >= 150))

    def test_too_small_mask_returns_none(self):
        tiny = np.zeros_like(self.mask)
        tiny[5, 5] = True
        world, px, info = mask_grid_seed(self.view, tiny)
        self.assertIsNone(world)
        self.assertIn("why", info)

    def test_farthest_point_order_keeps_first(self):
        pts = np.array([[0, 0], [1, 0], [10, 0], [5, 5]], dtype=float)
        self.assertEqual(farthest_point_order(pts, [1], 3)[:2], [1, 2])


if __name__ == "__main__":
    unittest.main()
