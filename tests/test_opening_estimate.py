"""touch.opening_estimate: robust port aim point from noisy wrist-camera pointing candidates."""
import unittest

from touch.opening_estimate import estimate_opening

# Real candidates from rollout usb_u3 (2026-11-08), all views of the laptop's -x side face.
# The first view's single hit (0.096, 0.4242, 0.0549) became the LLM's fallback aim: 1.4 cm low.
U3_PORT = [[0.096, 0.4242, 0.0549], [0.0908, 0.5416, 0.0707], [0.0907, 0.5184, 0.0698],
           [0.0905, 0.6125, 0.0658], [-0.1311, 0.4523, 0.0413], [0.0907, 0.5405, 0.0693],
           [0.0906, 0.5198, 0.0682], [0.0909, 0.489, 0.0713], [0.0918, 0.4555, 0.0721],
           [-0.0357, 0.3366, 0.1119], [0.09, 0.4966, 0.07], [0.0904, 0.459, 0.0705]]
U3_EDGE = [[0.0905, 0.598, 0.0651], [0.0905, 0.5707, 0.0664], [0.0904, 0.5665, 0.0644],
           [0.0951, 0.6095, 0.076], [0.0971, 0.5451, 0.0551], [0.0904, 0.4649, 0.0645],
           [0.0903, 0.6504, 0.0619], [0.2357, 0.4075, 0.0729], [0.0904, 0.473, 0.0652]]


class EstimateOpeningTest(unittest.TestCase):
    def test_u3_height_within_2mm_and_line_covers_the_port(self):
        est = estimate_opening([{"position": p} for p in U3_PORT], U3_EDGE)
        x, y, z = est["position"]
        self.assertAlmostEqual(z, 0.069, delta=0.002)          # eval truth 0.069
        self.assertAlmostEqual(x, 0.092, delta=0.003)
        self.assertLessEqual(abs(y - 0.4959), est["search_half"])  # the line search reaches it
        self.assertEqual(est["search_line"], [0.0, 1.0, 0.0])
        self.assertTrue(est["confident"])
        # edge hits here are BELOW the port hits (bad edge pointing) -> must not drop real port hits
        self.assertTrue(any("edge hits ignored" in n for n in est["notes"]))

    def test_u5_search_reaches_the_farthest_kept_hit(self):
        # usb_u5: median 0.4713, the one correct hit 0.498 (truth 0.4959) was outside spread/2+5mm
        port = [[0.0931, 0.4643, 0.0507], [-0.1311, 0.4494, 0.0405], [0.0909, 0.462, 0.07],
                [0.0907, 0.498, 0.0708], [0.0909, 0.4713, 0.0698]]
        edge = [[0.0957, 0.4798, 0.0593]]
        est = estimate_opening(port, edge)
        y = est["position"][1]
        self.assertGreaterEqual(est["search_half"], abs(0.498 - y) + 0.005)
        self.assertLessEqual(abs(0.4959 - y), est["search_half"])

    def test_edge_hits_above_drop_rim_candidates(self):
        port = [[0.09, 0.50, 0.069], [0.09, 0.51, 0.070], [0.09, 0.49, 0.0685], [0.09, 0.50, 0.072]]
        est = estimate_opening(port, edge_candidates=[[0.09, 0.5, 0.0745], [0.09, 0.52, 0.075]])
        self.assertEqual(est["n_used"], 3)
        self.assertAlmostEqual(est["position"][2], 0.069, delta=0.0006)

    def test_too_few_hits_is_not_confident(self):
        est = estimate_opening([[0.09, 0.5, 0.07]])
        self.assertFalse(est["confident"])
        self.assertTrue(any("more views" in n for n in est["notes"]))

    def test_turned_face_keeps_the_estimate_on_the_face(self):
        """A face 30 deg off the world axes: off-face hits are judged along its normal, and the search
        line runs along the face, not along world y."""
        import math
        n = [math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0]
        t = [-n[1], n[0], 0.0]
        port = [0.1, 0.5, 0.07]
        hits = [[port[i] + s * t[i] for i in range(3)] for s in (-0.01, 0.0, 0.004, 0.012)]
        hits.append([port[i] + 0.03 * n[i] for i in range(3)])      # 3 cm off the face
        est = estimate_opening(hits, axis=n)
        self.assertEqual(est["n_used"], 4)
        self.assertAlmostEqual(est["search_line"][0], t[0], delta=1e-3)
        off = sum((est["position"][i] - port[i]) * n[i] for i in range(3))
        self.assertLess(abs(off), 1e-3)

    def test_wide_scatter_widens_the_search(self):
        # plug p5 (2026-11-10): port hits 0.466..0.627 (truth 0.4959), median 0.537; a 4 cm cap
        # stopped 1 mm short of the port
        port = [[0.0909, 0.538, 0.0717], [0.0905, 0.6269, 0.0661], [0.0909, 0.4657, 0.0709],
                [0.0905, 0.4888, 0.0695], [0.0908, 0.5573, 0.0695], [0.0907, 0.5364, 0.0694]]
        est = estimate_opening(port)
        self.assertLessEqual(abs(0.4959 - est["position"][1]) + 0.006, est["search_half"])
        self.assertTrue(any("along the face" in n for n in est["notes"]))

    def test_bimodal_heights_keep_the_densest_cluster(self):
        # plug p6 (2026-11-10): the median z fell between two height clusters and rejected every hit
        port = [[0.096, 0.4607, 0.0542], [0.0946, 0.6409, 0.0518], [0.0903, 0.4798, 0.0447],
                [0.0908, 0.4577, 0.0697], [0.0911, 0.4596, 0.0719], [0.0907, 0.4955, 0.0702]]
        est = estimate_opening(port)
        self.assertIsNotNone(est["position"])
        self.assertAlmostEqual(est["position"][2], 0.069, delta=0.003)

    def test_line_search_reaches_both_edges_of_the_far_opening(self):
        # plug p8 (2026-11-10): half a width past the farthest hit ended the sweep on the far edge of
        # the opening, so the edge finder never bracketed it
        port = [[0.0909, 0.4657, 0.0703], [0.0897, 0.4958, 0.0696], [0.0898, 0.4942, 0.0685],
                [0.0907, 0.4673, 0.0688]]
        est = estimate_opening(port, opening_width=0.012)
        far_edge = abs(0.4959 - est["position"][1]) + 0.006
        self.assertGreaterEqual(est["search_half"], far_edge + 0.003)

    def test_empty(self):
        self.assertIsNone(estimate_opening([])["position"])


if __name__ == "__main__":
    unittest.main()
