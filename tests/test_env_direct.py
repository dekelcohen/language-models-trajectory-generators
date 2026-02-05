import unittest
import io
import contextlib
import pybullet as p
import env


class TestEnvDirect(unittest.TestCase):
    def setUp(self):
        # Ensure a clean connection state before each test
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass

    def test_run_gui_demo_direct(self):
        # Run the GUI demo in headless DIRECT mode; it should not block
        # and should avoid accessing the DebugVisualizer camera.        
        env.run_gui_demo(task_p='door', disable_forces=False, connection_mode=p.DIRECT)        
        
        # Ensure PyBullet disconnected
        self.assertFalse(p.isConnected())


if __name__ == "__main__":
    unittest.main()
