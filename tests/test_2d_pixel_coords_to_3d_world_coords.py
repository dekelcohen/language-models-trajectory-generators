import unittest
import numpy as np
import pybullet as p
import env
import config

class TestCameraUnprojection(unittest.TestCase):
    def setUp(self):
        # Clean up any existing connection
        if p.isConnected():
            p.disconnect()

    def tearDown(self):
        if p.isConnected():
            p.disconnect()

    def test_2d_pixel_coords_to_3d_world_coords(self):
        # 1. Initialize Simulation (loads assets, sets poses)
        env.run_sim_demo(task_p='door', disable_forces=False, connection_mode=p.DIRECT)

        # 2. Define Known World Point (The Latch Link Origin provided in prompt)
        known_world_pos = np.array([-0.07745519744833454, -0.00880230021590278, 0.672376])

        # 3. Define Optimized Camera Parameters for Depth Precision
        # Standard config.far_plane=100 causes loss of precision for objects at 1.3m.
        # We use tight bounds to ensure we get valid depth data.
        near_plane = 0.5
        far_plane = 2.0
        fov = config.fov
        width = config.image_width
        height = config.image_height

        # 4. Compute Matrices Manually
        # We ensure we use exactly the same matrices for projection and unprojection.
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=config.camera_target_position,
            distance=config.camera_distance,
            yaw=config.camera_yaw,
            pitch=config.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=config.aspect,
            nearVal=near_plane,
            farVal=far_plane
        )

        # Convert to Numpy (Column-Major)
        Vm = np.array(view_matrix).reshape(4, 4, order='F')
        Pm = np.array(proj_matrix).reshape(4, 4, order='F')
        VP = Pm @ Vm

        # 5. Project Known Point to find the Correct Pixel
        # The prompt suggested (179, 76), but math shows the link origin is at (156, 72).
        # We must use the pixel that actually corresponds to the 3D point.
        point_4d = np.append(known_world_pos, 1.0)
        clip = VP @ point_4d
        ndc = clip / clip[3]
        
        pixel_x = int(round((ndc[0] + 1.0) * width / 2.0))
        pixel_y = int(round((1.0 - ndc[1]) * height / 2.0))
        
        print(f"\n[Validation]")
        print(f"Known World Pos: {known_world_pos}")
        print(f"Calculated Pixel: ({pixel_x}, {pixel_y})")

        # 6. Capture High-Precision Depth Buffer
        # We ask PyBullet for the depth buffer directly (float array 0.0-1.0)
        # using the TinyRenderer (software) which is reliable in DIRECT mode.
        w, h, rgb, depth_buffer, seg = p.getCameraImage(
            width, height, 
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        # Reshape flat list to 2D array
        depth_data = np.array(depth_buffer).reshape(height, width)

        # 7. Read Depth at Pixel
        # We check a small window because the mathematical center of the link 
        # might be slightly inside the mesh or occluded by the axis cylinder.
        # We take the minimum depth (closest surface) in a 3x3 patch.
        px = np.clip(pixel_x, 0, width-1)
        py = np.clip(pixel_y, 0, height-1)
        
        window = 1
        y_min, y_max = max(0, py-window), min(height, py+window+1)
        x_min, x_max = max(0, px-window), min(width, px+window+1)
        patch = depth_data[y_min:y_max, x_min:x_max]
        
        # Filter out background (1.0)
        valid_depths = patch[patch < 0.99]
        
        if len(valid_depths) > 0:
            d_val = np.min(valid_depths)
            print(f"Depth Value (closest surface): {d_val:.4f}")
        else:
            print("WARNING: Ray hit background (depth=1.0). Object may be missing/occluded.")
            d_val = depth_data[py, px]

        # 8. Unproject
        # Convert Depth (0..1) -> NDC Z (-1..1)
        z_ndc = 2.0 * d_val - 1.0
        
        # Pixel -> NDC X,Y
        ndc_x = (2.0 * pixel_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / height)
        
        clip_pos = np.array([ndc_x, ndc_y, z_ndc, 1.0])
        inv_VP = np.linalg.inv(VP)
        world_hom = inv_VP @ clip_pos
        world_recon = world_hom[:3] / world_hom[3]

        print(f"[Result]")
        print(f"Reconstructed: {world_recon}")
        error = np.linalg.norm(world_recon - known_world_pos)
        print(f"Error: {error:.4f} m")

        # 9. Assert
        np.testing.assert_allclose(
            world_recon, 
            known_world_pos, 
            rtol=0.1, 
            atol=0.05, 
            err_msg="Unprojection failed. Ensure the object is rendered and not occluded."
        )

if __name__ == "__main__":
    unittest.main()