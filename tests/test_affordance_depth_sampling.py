"""Affordance-point depth sampling: patch vs the single pixel it replaced.

The VLM points at a *thin* feature. Seen from the head camera 1.2 m away at 5.5 mm/px, the
adroit door's lever bar is about 7 px thick, so a 1-2 px pointing error - well inside what
any pointing model produces - made the old ``depth_array[yi, xi]`` return the depth of the
door face behind the bar, or of the room behind the door. The 3D point then landed 20 cm
away from anything graspable and nothing downstream could tell it had happened.

The in-sim tests below pin that down with real renders: they project the lever's known world
midpoint into the head camera, then compare the two sampling strategies at pixels that just
miss the bar. Measured on PyBullet, a 3 px miss costs the single pixel 0.22 m of 3D error
while the patch stays within 0.05 m.

Layout:
  * ``SampleSurfaceDepth`` / ``DepthMatchesObject`` - pure numpy, no sim, always run.
  * ``AffordanceDepthInSimMixin`` - the real assertions; subclassed for PyBullet (default)
    and Genesis (skipped unless its interpreter resolves).
"""

import multiprocessing
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import utils  # noqa: E402
from config import CAPTURE_IMAGES, GET_STATE  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPLY_TIMEOUT = float(os.environ.get("LMTG_IPC_TEST_TIMEOUT", "300"))
BOOT_TIMEOUT = float(os.environ.get("LMTG_IPC_TEST_BOOT_TIMEOUT", "600"))

#: World midpoint of the adroit door's lever bar, derived from ``adroit_door.urdf``: the
#: ``latch`` link carries a r=0.02 l=0.2 cylinder at ``[0.1, -0.15, 0]`` rpy ``(0, 1.5708, 0)``,
#: and the door spawns at a fixed pose (``sim_envs/pybullet/door.py``), so the scene is
#: deterministic. ``_LATCH_ORIGIN`` is the same link's origin as reported by GET_STATE and is
#: asserted before use, so the tests skip loudly if the scene is ever moved instead of
#: quietly testing the wrong pixels.
MID_LEVER = np.array([-0.2773, 0.0092, 0.6724])
_LATCH_ORIGIN = np.array([-0.07745, -0.00880, 0.67238])


class _Args:
    mode = "default"
    robot = "franka"
    task = "door"
    sim = "pybullet"
    gui = False
    save_grasp_inputs = False


def _pybullet_child(connection):
    import traceback
    sys.path.insert(0, REPO_ROOT)
    try:
        import env as envmod
        envmod.run_simulation_environment(_Args, connection, None)
    except Exception:
        traceback.print_exc()
        raise


class SampleSurfaceDepth(unittest.TestCase):
    """``utils.sample_surface_depth`` on hand-built depth maps with known answers."""

    def setUp(self):
        # A 1 px wide "bar" at 0.30 in front of a wall at 0.80. Deliberately thinner than
        # the patch, which is the case that breaks single-pixel sampling.
        self.depth = np.full((21, 21), 0.80, dtype=np.float32)
        self.depth[:, 10] = 0.30

    def test_reads_the_near_surface_when_pointed_at_it(self):
        z, info = utils.sample_surface_depth(self.depth, 10, 10, radius=2)
        self.assertAlmostEqual(z, 0.30, places=5)
        self.assertEqual(info["n_used"], 25)

    def test_recovers_the_near_surface_after_a_pointing_miss(self):
        """The whole point: 2 px off the bar, the centre pixel is wall, the patch is bar."""
        self.assertAlmostEqual(float(self.depth[10, 12]), 0.80, places=5)
        z, info = utils.sample_surface_depth(self.depth, 12, 10, radius=2)
        self.assertAlmostEqual(z, 0.30, places=5)
        self.assertAlmostEqual(info["center"], 0.80, places=5)

    def test_gives_up_when_the_miss_exceeds_the_patch(self):
        """No silent guessing: outside the window there is no evidence of the bar."""
        z, _ = utils.sample_surface_depth(self.depth, 14, 10, radius=2)
        self.assertAlmostEqual(z, 0.80, places=5)

    def test_returns_a_measured_depth_never_an_interpolation(self):
        """A value between bar and wall would be a surface that does not exist."""
        for x in range(8, 14):
            z, _ = utils.sample_surface_depth(self.depth, x, 10, radius=2)
            self.assertIn(round(float(z), 5), (0.30, 0.80), f"interpolated depth at x={x}")

    def test_a_single_bad_pixel_does_not_win(self):
        """Why a percentile and not min(): one speck of noise would capture every point."""
        noisy = self.depth.copy()
        noisy[4, 4] = 0.01  # e.g. an anti-aliased silhouette edge
        z, _ = utils.sample_surface_depth(noisy, 4, 4, radius=2)
        self.assertAlmostEqual(z, 0.80, places=5)

    def test_percentile_zero_is_plain_min(self):
        noisy = self.depth.copy()
        noisy[4, 4] = 0.01
        z, _ = utils.sample_surface_depth(noisy, 4, 4, radius=2, percentile=0.0)
        self.assertAlmostEqual(z, 0.01, places=5)

    def test_non_finite_pixels_are_skipped(self):
        broken = self.depth.copy()
        broken[9:12, 9:12] = np.nan
        z, info = utils.sample_surface_depth(broken, 10, 10, radius=2)
        self.assertTrue(np.isfinite(z))
        self.assertLess(info["n_used"], info["n_total"])

    def test_all_non_finite_reports_unusable(self):
        z, _ = utils.sample_surface_depth(np.full((9, 9), np.nan), 4, 4, radius=2)
        self.assertIsNone(z)

    def test_out_of_bounds_is_none(self):
        for x, y in ((-1, 5), (5, -1), (21, 5), (5, 21)):
            self.assertIsNone(utils.sample_surface_depth(self.depth, x, y)[0])

    def test_window_is_clipped_at_the_image_edge(self):
        z, info = utils.sample_surface_depth(self.depth, 0, 0, radius=2)
        self.assertAlmostEqual(z, 0.80, places=5)
        self.assertEqual(info["n_used"], 9)

    def test_mask_confines_the_patch(self):
        """With the bar masked out, the patch may only see the wall."""
        mask = np.ones_like(self.depth, dtype=bool)
        mask[:, 10] = False
        z, info = utils.sample_surface_depth(self.depth, 10, 10, radius=2, mask=mask)
        self.assertAlmostEqual(z, 0.80, places=5)
        self.assertTrue(info["mask_used"])

    def test_mask_of_the_wrong_shape_is_ignored_not_fatal(self):
        z, info = utils.sample_surface_depth(self.depth, 10, 10, radius=2,
                                             mask=np.ones((5, 5), dtype=bool))
        self.assertAlmostEqual(z, 0.30, places=5)
        self.assertFalse(info["mask_used"])

    def test_empty_mask_falls_back_to_the_centre_pixel(self):
        z, _ = utils.sample_surface_depth(self.depth, 10, 10, radius=2,
                                          mask=np.zeros_like(self.depth, dtype=bool))
        self.assertAlmostEqual(z, 0.30, places=5)

    def test_works_for_either_depth_encoding(self):
        """Both sims are monotonic in distance, so 'low percentile' means 'near' in both."""
        opengl = np.full((11, 11), 0.9940, dtype=np.float32)
        opengl[:, 5] = 0.9930
        z, _ = utils.sample_surface_depth(opengl, 6, 5, radius=2)
        self.assertAlmostEqual(z, 0.9930, places=6)


class DepthMatchesObject(unittest.TestCase):
    """``utils.depth_matches_object`` - a guard against a point that is off the object."""

    def setUp(self):
        self.depth = np.full((21, 21), 0.80, dtype=np.float32)
        self.depth[8:13, 8:13] = 0.30  # the object
        self.mask = np.zeros_like(self.depth, dtype=bool)
        self.mask[8:13, 8:13] = True

    def test_accepts_a_depth_on_the_object(self):
        ok, info = utils.depth_matches_object(0.30, self.depth, self.mask)
        self.assertTrue(ok)
        self.assertTrue(info["checked"])

    def test_rejects_the_backdrop_depth(self):
        self.assertFalse(utils.depth_matches_object(0.80, self.depth, self.mask)[0])

    def test_accepts_any_part_of_a_deep_object(self):
        """An object spans a range of depths; every one of them is still the object."""
        slanted = self.depth.copy()
        slanted[8:13, 8:13] = np.linspace(0.30, 0.45, 5)
        for z in (0.30, 0.37, 0.45):
            self.assertTrue(utils.depth_matches_object(z, slanted, self.mask)[0], z)
        self.assertFalse(utils.depth_matches_object(0.80, slanted, self.mask)[0])

    def test_cannot_run_means_accept(self):
        """A missing mask is a missing check, not a reason to discard a good point."""
        for args in ((0.30, self.depth, None), (None, self.depth, self.mask),
                     (0.30, None, self.mask)):
            self.assertTrue(utils.depth_matches_object(*args)[0])
        ok, info = utils.depth_matches_object(0.30, self.depth,
                                              np.zeros_like(self.mask))
        self.assertTrue(ok)
        self.assertFalse(info["checked"])

    def test_mismatched_mask_shape_is_ignored(self):
        ok, info = utils.depth_matches_object(0.80, self.depth, np.ones((4, 4), dtype=bool))
        self.assertTrue(ok)
        self.assertFalse(info["checked"])


class AffordanceDepthInSimMixin:
    """Real renders, real depth encodings. Subclasses supply ``self.connection``."""

    connection = None
    handshake = None
    expected_depth_encoding = None
    #: How far the reported latch origin may sit from ``_LATCH_ORIGIN`` before the scene is
    #: assumed to have changed. Subclasses loosen it where the simulator uses a different
    #: link reference frame (see the Genesis subclass).
    latch_origin_tolerance = 5e-3

    min_single_pixel_error = 0.15
    #: ...while the patch stays within this much of the lever axis.
    max_patch_error = 0.08
    #: How much worse than the single pixel the patch is allowed to be where the single
    #: pixel already hit the bar. The patch reports the NEAR surface by design, so on a
    #: curved feature it can legitimately pick a point up to one radius closer than the
    #: exact pixel - the lever bar is r=0.02, so anything beyond ~3 cm is a real defect.
    max_patch_regression = 0.03

    @classmethod
    def _read_handshake(cls, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cls.connection.poll(0.5):
                return cls.connection.recv()
        raise unittest.SkipTest(f"no handshake from the environment within {timeout}s")

    def _exchange(self, message):
        self.connection.send(message)
        deadline = time.time() + REPLY_TIMEOUT
        while time.time() < deadline:
            if self.connection.poll(0.2):
                return self.connection.recv()
        self.fail(f"no reply to command {message[0]} within {REPLY_TIMEOUT}s")

    def setUp(self):
        utils.args = _Args  # get_world_point_world_frame reads this module global

        state = self._exchange([GET_STATE])
        sim_state = state.get("sim_state", state) if isinstance(state, dict) else {}
        origin = sim_state.get("door_handle_pos")
        if origin is None:
            self.skipTest("the environment did not report door_handle_pos")
        if not np.allclose(np.asarray(origin, dtype=float), _LATCH_ORIGIN,
                           atol=self.latch_origin_tolerance):
            self.skipTest(f"door moved ({np.round(origin, 4)} != {_LATCH_ORIGIN}); "
                          "MID_LEVER would no longer be the lever")

        reply = self._exchange([CAPTURE_IMAGES])
        self.assertIsInstance(reply, list)
        self.assertGreaterEqual(len(reply), 6, "CAPTURE_IMAGES must return 6 elements")
        self.head_pos, self.head_quat = reply[0], reply[1]
        self.cam_info = reply[5]
        self.assertEqual(self.cam_info.get("depth_encoding"), self.expected_depth_encoding,
                         "wrong simulator answered this test")

        npy = os.path.splitext(config.depth_image_head_path)[0] + ".npy"
        if not os.path.exists(npy):
            self.skipTest("no raw depth .npy; the 8-bit PNG is too coarse for this test")
        self.depth = np.load(npy).astype(np.float32)
        self.size = (config.image_width, config.image_height)

        self.lever_px = utils.project_3d_world_pos_to_2d_pixel(
            self.head_pos, self.head_quat, "head", self.size, list(MID_LEVER), self.cam_info)
        x, y = int(self.lever_px[0]), int(self.lever_px[1])
        height, width = self.depth.shape[:2]
        if not (3 <= x < width - 3 and 3 <= y < height - 3):
            self.skipTest(f"the lever projects to {(x, y)}, outside the usable image")
        self.lever_px = (x, y)

    def _unproject(self, x, y, depth_value):
        world = utils.get_world_point_world_frame(
            self.head_pos, self.head_quat, "head", self.size, [x, y, float(depth_value)],
            cam_info=self.cam_info)
        return np.asarray(world, dtype=float).flatten()[:3]

    def _errors(self, x, y):
        """3D distance to the lever axis for single-pixel and patch sampling."""
        single = self._unproject(x, y, self.depth[y, x])
        patch_z, info = utils.sample_surface_depth(self.depth, x, y)
        patch = self._unproject(x, y, patch_z)
        return (float(np.linalg.norm(single - MID_LEVER)),
                float(np.linalg.norm(patch - MID_LEVER)), info)

    def test_patch_sampling_finds_the_lever_not_the_door_behind_it(self):
        x, y = self.lever_px
        _, patch_error, _ = self._errors(x, y)
        self.assertLess(patch_error, self.max_patch_error,
                        f"patch sampling at the lever pixel {self.lever_px} landed "
                        f"{patch_error:.3f} m from the lever axis")

    def test_a_pointing_miss_that_ruins_the_single_pixel_is_survived(self):
        """The regression this feature exists for.

        Pixels a few rows below the bar still see it inside a 5x5 window, but their own
        depth is the door face or the room. At least one such miss must exist next to a
        7 px feature - if none does, the render or the geometry changed and the test is no
        longer proving anything.
        """
        x, y = self.lever_px
        rescued = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                single_error, patch_error, info = self._errors(x + dx, y + dy)
                if single_error < self.min_single_pixel_error:
                    continue
                rescued.append((dx, dy, single_error, patch_error))
                self.assertLess(patch_error, self.max_patch_error,
                                f"offset ({dx:+d},{dy:+d}): single pixel was "
                                f"{single_error:.3f} m off and the patch did not recover "
                                f"({patch_error:.3f} m, {info['n_used']} px sampled)")
                self.assertLess(patch_error * 2.0, single_error,
                                f"offset ({dx:+d},{dy:+d}): the patch has to be decisively "
                                f"better, got {patch_error:.3f} vs {single_error:.3f} m")
        self.assertTrue(rescued,
                        "no pointing miss in the +/-3 px neighbourhood cost the single "
                        "pixel real error - the lever is no longer a thin feature here, so "
                        "this test would silently pass forever")

    def test_patch_sampling_never_makes_a_good_point_much_worse(self):
        """Where the single pixel already hit the bar, the patch stays on the bar too."""
        x, y = self.lever_px
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                single_error, patch_error, _ = self._errors(x + dx, y + dy)
                if single_error >= self.min_single_pixel_error:
                    continue
                self.assertLess(patch_error, single_error + self.max_patch_regression,
                                f"offset ({dx:+d},{dy:+d}): patch {patch_error:.3f} m is "
                                f"worse than the single pixel {single_error:.3f} m by more "
                                f"than the lever's radius")

    def test_the_lever_really_is_in_front_of_the_door(self):
        """Guards the premise: without a near/far split there is nothing to recover."""
        x, y = self.lever_px
        _, info = utils.sample_surface_depth(self.depth, x, y, radius=4)
        near = self._unproject(x, y, info["near"])
        far = self._unproject(x, y, info["far"])
        self.assertGreater(float(np.linalg.norm(near - far)), 0.03,
                           "the lever and its backdrop are less than 3 cm apart in this "
                           "render; a pointing miss could not be detected")


class TestPyBulletAffordanceDepth(AffordanceDepthInSimMixin, unittest.TestCase):

    expected_depth_encoding = "opengl"

    @classmethod
    def setUpClass(cls):
        cls.connection, child = multiprocessing.Pipe()
        cls.process = multiprocessing.Process(target=_pybullet_child, args=(child,))
        cls.process.daemon = True
        cls.process.start()
        cls.handshake = cls._read_handshake(BOOT_TIMEOUT)
        if not cls.process.is_alive():
            raise unittest.SkipTest("the PyBullet environment subprocess died on startup")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.connection.close()
        finally:
            cls.process.terminate()
            cls.process.join(timeout=20)


def _genesis_available():
    try:
        from providers.genesis_launcher import resolve_genesis_python
        resolve_genesis_python()
        return True
    except Exception:
        return False


@unittest.skipUnless(_genesis_available(),
                     "Genesis interpreter not resolvable; set GENESIS_PYTHON or create the "
                     "vlm_genesis conda env")
class TestGenesisAffordanceDepth(AffordanceDepthInSimMixin, unittest.TestCase):
    """The same claims against linear-metre depth, to prove the sampler is encoding-agnostic.

    Genesis renders the same URDF through a different rasteriser, so the bar's silhouette
    lands on slightly different pixels; the tolerances are loosened rather than tuned to
    one renderer's anti-aliasing.
    """

    expected_depth_encoding = "linear_metric"
    min_single_pixel_error = 0.15
    max_patch_error = 0.10
    #: GenesisAdapter warns that ``link_ref_frame`` is unavailable, so it reports the latch
    #: link's URDF-origin pose where PyBullet reports its centre of mass - a fixed ~2 cm XY
    #: offset, not a moved door (the z matches exactly). The guard still catches a real
    #: relocation, which would be far larger.
    latch_origin_tolerance = 3e-2

    @classmethod
    def setUpClass(cls):
        from providers.genesis_launcher import launch_genesis_child
        from providers.json_ipc import JsonIpcConnection

        class _GenesisArgs(_Args):
            sim = "genesis"

        cls.process, host, port, _python = launch_genesis_child(_GenesisArgs)
        cls.connection = JsonIpcConnection(host, port)
        try:
            cls.connection.wait_until_ready(process=cls.process, timeout=BOOT_TIMEOUT)
        except Exception as exc:
            cls.process.terminate()
            raise unittest.SkipTest(f"Genesis child failed to start: {exc}")
        cls.handshake = cls._read_handshake(BOOT_TIMEOUT)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.connection.close()
        finally:
            cls.process.terminate()
            # launch_genesis_child returns a subprocess.Popen, not a multiprocessing.Process.
            cls.process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()

