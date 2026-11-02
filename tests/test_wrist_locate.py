"""locate_with_wrist_camera: VLM wrist-image points -> 3D candidates via wrist depth + calibration."""
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import config
import utils
from api import API
from sim_adapter import camera_math

W = H = 64
NEAR, FAR = 0.01, 100.0


def _perspective(fov_deg=60.0):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (FAR + NEAR) / (NEAR - FAR), 2 * FAR * NEAR / (NEAR - FAR)],
                     [0, 0, -1, 0]])


class WristLocateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.patches = [
            mock.patch.object(config, "rgb_image_wrist_path", os.path.join(d, "rgb_image_wrist.png")),
            mock.patch.object(config, "depth_image_wrist_path", os.path.join(d, "depth_image_wrist.png")),
            mock.patch.object(config, "images_folder", d),
        ]
        for p in self.patches:
            p.start()
        Image.new("RGB", (W, H)).save(config.rgb_image_wrist_path)
        # Camera at the origin looking down -z (identity view); a wall 0.5 m in front.
        np.save(os.path.join(d, "depth_image_wrist.npy"), np.full((H, W), 0.5, np.float32))
        cam_info = {"wrist": {"viewMatrix": np.eye(4).flatten(order="F").tolist(),
                              "projectionMatrix": _perspective().flatten(order="F").tolist(),
                              "znear": NEAR, "zfar": FAR},
                    "depth_encoding": camera_math.DEPTH_LINEAR_METRIC, "new_3d_proj": True}

        api = API.__new__(API)
        api.args = types.SimpleNamespace(planner_perception_vlm="or-google/gemini-3.8-flash",
                                         max_tokens=100, reasoning_effort=None, save_grasp_inputs=False)
        api.logger = mock.MagicMock()
        api.main_connection = api.client = api.llm_cache = None
        api.trajectory_step = 1
        api.task = types.SimpleNamespace()

        def capture(capture=True):
            api.wrist_camera_position, api.wrist_camera_orientation_q = [0, 0, 0], [0, 0, 0, 1]
            api.cam_info = cam_info
            return None, None

        api._capture_head_image_and_depth = capture
        self.api = api
        utils.args, utils.logger = api.args, api.logger

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _locate(self, reply):
        with mock.patch("models.call_llm_cached", return_value=[{"role": "assistant", "content": reply}]):
            return self.api.locate_with_wrist_camera("usb port")

    def test_points_become_world_candidates_best_first(self):
        # yx normalised 0-1000: image centre, then a point right of centre.
        res = self._locate('```json\n[{"point": [500, 500], "label": "usb A"},'
                           ' {"point": [500, 750], "label": "usb B"}]\n```')
        self.assertEqual([r["label"] for r in res], ["usb A", "usb B"])
        np.testing.assert_allclose(res[0]["position"], [0, 0, -0.5], atol=1e-3)
        self.assertGreater(res[1]["position"][0], 0.05)
        self.assertAlmostEqual(res[1]["position"][2], -0.5, places=3)

    def test_unparseable_or_out_of_image_points_are_dropped(self):
        self.assertEqual(self._locate("no port visible"), [])
        self.assertEqual(self._locate('[{"point": [500, 1500]}, {"label": "no point"}]'), [])


if __name__ == "__main__":
    unittest.main()
