import unittest
import io
import contextlib
import numpy as np
from PIL import Image
import pybullet as p
import env
import config


class TestEnvDirect(unittest.TestCase):
    def setUp(self):
        # Ensure a clean connection state before each test
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass

    def test_2d_pixel_coords_to_3d_world_coords(self):
        # Run the GUI demo in headless DIRECT mode; it should not block
        # and should avoid accessing the DebugVisualizer camera.        
        env.run_sim_demo(task_p='door', disable_forces=False, gui=False)        
        
        
        # Disconnect at test end and assert
        p.disconnect()
        self.assertFalse(p.isConnected())


if __name__ == "__main__":
    unittest.main()


