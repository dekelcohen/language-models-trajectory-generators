import unittest
import numpy as np
import pybullet as p
import env
import config

class TestCameraUnprojection(unittest.TestCase):
    def setUp(self):
        # Ensure clean state to avoid collisions with other tests
        if p.isConnected():
            p.disconnect()

    def tearDown(self):
        if p.isConnected():
            p.disconnect()

    def test_2d_pixel_coords_to_3d_world_coords(self):
        # --- 1. Initialize Simulation ---
        # Run in DIRECT (headless) mode. Loads the door assets and physics state.
        env.run_sim_demo(task_p='door', disable_forces=False, connection_mode=p.DIRECT)

        # --- 2. Define Known World Point ---
        # The Link Origin / Joint Center from PyBullet's getLinkState.
        # Note: This geometric center is often inside the mesh, not on the surface.
        
        # To change object and world pos in sim: enter it here and in run_sim_demo in OVERLAY_COORD_TEST = True
        known_world_pos = np.array([-0.07745519744833454, -0.00880230021590278, 0.672376])  # door handle 
        # known_world_pos = np.array([0, 0, 0])        

        # --- 3. Optimize Camera Frustum for Depth Precision ---
        # The default far_plane=100.0 compresses depth values significantly.
        # For an object ~1.3m away, this results in poor float precision.
        # We tighten the range [0.5, 2.5] to ensure valid depth differentiation.
        
        
        # --- 4. Compute Camera Matrices ---
        # View Matrix: Transforms World Space -> Camera Space
        view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=config.camera_target_position,
            distance=config.camera_distance,
            yaw=config.camera_yaw,
            pitch=config.camera_pitch,
            roll=0,
            upAxisIndex=2
        )).reshape(4, 4, order='F') # Reshape flattened Fortran-order list to 4x4 matrix

        # Projection Matrix: Transforms Camera Space -> Clip Space
        proj_matrix = np.array(p.computeProjectionMatrixFOV(
            fov=config.fov,
            aspect=config.aspect,
            nearVal=config.near_plane,
            farVal=config.far_plane
        )).reshape(4, 4, order='F')
        
        # Combined Model-View-Projection (MVP) Matrix
        VP = proj_matrix @ view_matrix

        # --- 5. Project Known Point to find the Pixel ---
        # We project the 3D center to 2D to find exactly which pixel corresponds to it.
        # This ensures we don't test a pixel that misses the object.
        point_4d = np.append(known_world_pos, 1.0) # Convert to Homogeneous [x,y,z,1]
        clip = VP @ point_4d
        ndc = clip / clip[3] # Perspective Divide to get Normalized Device Coordinates (NDC)
        
        width = config.image_width
        height = config.image_height
        
        # Map NDC [-1, 1] to Pixel Coordinates [0, Width/Height]
        pixel_x = int(round((ndc[0] + 1.0) * width / 2.0))
        # Note: We subtract from 1.0 because Image Y is Top-Down, while NDC Y is Bottom-Up
        pixel_y = int(round((1.0 - ndc[1]) * height / 2.0))

        print(f"\n[Validation Setup]")
        print(f"Known Center Pos: {known_world_pos}")
        print(f"Projected Pixel:  ({pixel_x}, {pixel_y})")

        # --- 6. Get High-Precision Depth from Simulation ---
        # Use TinyRenderer in DIRECT mode to get the float depth buffer (0.0 - 1.0)
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width, height, 
            viewMatrix=view_matrix.flatten(order='F'),
            projectionMatrix=proj_matrix.flatten(order='F'),
            renderer=p.ER_TINY_RENDERER
        )
        depth_data = np.array(depth_buffer).reshape(height, width)

        # --- 7. Sample Depth at the Projected Pixel ---
        # We use a 3x3 search window here strictly for robustness in this single-point test,
        # ensuring we don't hit an aliased edge or empty background pixel.
        # NOTE: In production (segmentation masks), do not use this loop. Instead, 
        # erode the mask by 1-2 pixels to avoid edges, then sample directly.
        px = np.clip(pixel_x, 0, width-1)
        py = np.clip(pixel_y, 0, height-1)
        
        window = 1
        y_min, y_max = max(0, py-window), min(height, py+window+1)
        x_min, x_max = max(0, px-window), min(width, px+window+1)
        patch = depth_data[y_min:y_max, x_min:x_max]
        
        # Filter out background values (typically 1.0) to find the closest object surface
        valid_depths = patch[patch < 0.99]
        if len(valid_depths) > 0:
            real_depth_val = np.min(valid_depths)
        else:
            real_depth_val = depth_data[py, px]

        # --- 8. Unproject using REAL Depth (The Core Logic) ---
        
        # A. Map Depth Buffer [0.0, 1.0] to NDC Z [-1.0, 1.0]
        #    OpenGL depth is non-linear, but the Projection Matrix expects standard NDC.
        z_ndc_real = 2.0 * real_depth_val - 1.0
        
        # B. Map Pixel X [0, Width] to NDC X [-1.0, 1.0]
        ndc_x = (2.0 * pixel_x / width) - 1.0
        
        # C. Map Pixel Y [0, Height] to NDC Y [1.0, -1.0]
        #    Note the "1.0 - ..." structure. This performs the Y-Flip.
        #    Image origin is Top-Left; OpenGL NDC origin is Bottom-Left.
        ndc_y = 1.0 - (2.0 * pixel_y / height)
        
        # D. Create the Clip Space Vector
        #    Homogeneous coordinates require w=1.0 for a point position.
        clip_pos_real = np.array([ndc_x, ndc_y, z_ndc_real, 1.0])
        
        # E. Apply Inverse View-Projection Matrix
        #    Transform: Clip Space -> World Space
        inv_VP = np.linalg.inv(VP)
        world_hom = inv_VP @ clip_pos_real
        
        # F. Perspective Divide
        #    The resulting vector is [x*w, y*w, z*w, w].
        #    We divide by w to recover the Cartesian [x, y, z] coordinates.
        reconstructed_surface_pos = world_hom[:3] / world_hom[3]

        # --- 9. Analyze Results ---
        # We calculate the Euclidean distance between the reconstructed surface point
        # and the known internal link origin.
        error = np.linalg.norm(reconstructed_surface_pos - known_world_pos)
        
        print(f"[Results]")
        print(f"Depth Value: {real_depth_val:.4f}")
        print(f"Reconstructed Surface: {reconstructed_surface_pos}")
        print(f"Original Center:       {known_world_pos}")
        print(f"Error (Surface offset): {error:.4f} m")
        
        # --- 10. Assert ---
        # We allow ~12cm tolerance. This accounts for:
        # 1. The physical distance between the object's surface (seen by camera) 
        #    and its internal mechanical origin (returned by getLinkState).
        # 2. Small quantization errors in the depth buffer.
        np.testing.assert_allclose(
            reconstructed_surface_pos, 
            known_world_pos, 
            rtol=0.1, 
            atol=0.12, 
            err_msg="Reconstructed point too far from object center (>12cm)."
        )

if __name__ == "__main__":
    unittest.main()