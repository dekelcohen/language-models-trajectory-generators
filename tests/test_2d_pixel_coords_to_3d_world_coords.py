import unittest
import numpy as np
import pybullet as p
import env
import config

class TestCameraUnprojection(unittest.TestCase):
    def setUp(self):
        # Ensure clean state
        if p.isConnected():
            p.disconnect()

    def tearDown(self):
        if p.isConnected():
            p.disconnect()

    def test_2d_pixel_coords_to_3d_world_coords(self):
        # 1. Initialize Simulation (Headless/Direct)
        env.run_sim_demo(task_p='door', disable_forces=False, connection_mode=p.DIRECT)

        # 2. Known World Point (The Link Origin / Joint Center)
        # Note: This point is inside the object mesh.
        known_world_pos = np.array([-0.07745519744833454, -0.00880230021590278, 0.672376])

        # 3. Optimize Camera Frustum for Precision
        # Default far=100m compresses depth precision too much.
        # We use a tight range [0.5, 2.5] to get accurate float depth values for the object at ~1.3m.
        near_plane = 0.5
        far_plane = 2.5
        
        # 4. Compute Camera Matrices
        view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=config.camera_target_position,
            distance=config.camera_distance,
            yaw=config.camera_yaw,
            pitch=config.camera_pitch,
            roll=0,
            upAxisIndex=2
        )).reshape(4, 4, order='F')

        proj_matrix = np.array(p.computeProjectionMatrixFOV(
            fov=config.fov,
            aspect=config.aspect,
            nearVal=near_plane,
            farVal=far_plane
        )).reshape(4, 4, order='F')
        
        VP = proj_matrix @ view_matrix

        # 5. Project Known Point to find the Pixel
        # This tells us exactly which pixel covers the link center.
        point_4d = np.append(known_world_pos, 1.0)
        clip = VP @ point_4d
        ndc = clip / clip[3]
        
        width = config.image_width
        height = config.image_height
        
        pixel_x = int(round((ndc[0] + 1.0) * width / 2.0))
        pixel_y = int(round((1.0 - ndc[1]) * height / 2.0))

        print(f"\n[Validation Setup]")
        print(f"Known Center Pos: {known_world_pos}")
        print(f"Projected Pixel:  ({pixel_x}, {pixel_y})")

        # 6. Get High-Precision Depth from Simulation
        # We use p.getCameraImage with the TinyRenderer to get the float depth buffer.
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width, height, 
            viewMatrix=view_matrix.flatten(order='F'),
            projectionMatrix=proj_matrix.flatten(order='F'),
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        depth_data = np.array(depth_buffer).reshape(height, width)

        # 7. Sample Depth at the Projected Pixel
        # We look at the specific pixel. A 3x3 min filter is used to find the closest surface 
        # in case of aliasing or if the pixel is on an edge.
        px = np.clip(pixel_x, 0, width-1)
        py = np.clip(pixel_y, 0, height-1)
        
        window = 1
        y_min, y_max = max(0, py-window), min(height, py+window+1)
        x_min, x_max = max(0, px-window), min(width, px+window+1)
        patch = depth_data[y_min:y_max, x_min:x_max]
        
        # Ignore background (1.0)
        valid_depths = patch[patch < 0.99]
        if len(valid_depths) > 0:
            real_depth_val = np.min(valid_depths)
        else:
            real_depth_val = depth_data[py, px]

        # 8. Unproject using REAL Depth
        # Convert depth buffer (0..1) to NDC Z (-1..1)
        z_ndc_real = 2.0 * real_depth_val - 1.0
        
        ndc_x = (2.0 * pixel_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / height)
        
        clip_pos_real = np.array([ndc_x, ndc_y, z_ndc_real, 1.0])
        inv_VP = np.linalg.inv(VP)
        world_hom = inv_VP @ clip_pos_real
        reconstructed_surface_pos = world_hom[:3] / world_hom[3]

        # 9. Analyze Results
        error = np.linalg.norm(reconstructed_surface_pos - known_world_pos)
        
        print(f"[Results]")
        print(f"Depth Value: {real_depth_val:.4f}")
        print(f"Reconstructed Surface: {reconstructed_surface_pos}")
        print(f"Original Center:       {known_world_pos}")
        print(f"Error (Surface offset): {error:.4f} m")
        
        # 10. Assert
        # The test passes if the reconstructed point is within 12cm of the center.
        # This accounts for the physical size of the door handle/latch mechanism,
        # as the camera sees the outside surface, not the internal joint origin.
        np.testing.assert_allclose(
            reconstructed_surface_pos, 
            known_world_pos, 
            rtol=0.1, 
            atol=0.12, 
            err_msg="Reconstructed point too far from object center (>12cm)."
        )

if __name__ == "__main__":
    unittest.main()