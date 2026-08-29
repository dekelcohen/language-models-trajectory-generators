import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os
import config
from config import OK, PROGRESS, FAIL, ENDC
from PIL import Image
from shapely.geometry import MultiPoint, Polygon, polygon
from sklearn.cluster import DBSCAN
from sim_adapter import camera_math

logger = None
args = None
_grasp_inputs_saved_cameras = set()  # tracks cameras whose matrices have already been saved
def get_segmentation_mask(model_predictions, segmentation_threshold):

    masks = []

    for model_prediction in model_predictions:
        # Support both torch.Tensor and numpy arrays for mask predictions
        if hasattr(model_prediction, "detach"):
            model_prediction_np = model_prediction.detach().cpu().numpy()
            thr = np.max(model_prediction_np) - segmentation_threshold * (np.max(model_prediction_np) - np.min(model_prediction_np))
            model_prediction[model_prediction < thr] = False
            model_prediction[model_prediction >= thr] = True
            masks.append(model_prediction)
        else:
            mp = np.asarray(model_prediction)
            thr = np.max(mp) - segmentation_threshold * (np.max(mp) - np.min(mp))
            bin_mask = (mp >= thr).astype(np.uint8)
            masks.append(bin_mask)

    return masks



def get_max_contour(image, image_width, image_height):

    ret, thresh = cv.threshold(image, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)

    if not contours:
        return None

    # Choose the contour with the largest area to be robust to thin rectangles
    max_idx = None
    max_area = 0.0
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_idx = i

    return contours[max_idx] if max_idx is not None else None



def _quat_to_rotmat(q):
    # Expect [x, y, z, w]
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ])
    return R





def save_xmem_image(masks):

    xmem_array = np.array(Image.open(config.xmem_input_path).convert("L"))
    xmem_array = np.unique(xmem_array, return_inverse=True)[1].reshape(xmem_array.shape)

    for mask in masks:
        mask_index = np.max(xmem_array) + 1
        if hasattr(mask, "detach"):
            mask_bool = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_bool = np.asarray(mask).astype(bool)
        xmem_array[mask_bool] = mask_index

    max_val = np.max(xmem_array)
    norm = xmem_array / max_val if max_val > 0 else xmem_array
    Image.fromarray((norm * 255).astype(np.uint8)).save(config.xmem_input_path)

def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, segmentation_count, cam_info=None):

    image_width, image_height = image.size
    bounding_cubes = []
    bounding_cubes_orientations = []
    
    logger.info(PROGRESS + f"-- Enter get_bounding_cube_from_point_cloud(...)" + ENDC)
        
    for i, mask in enumerate(masks):
        save_mask_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)

        contour = get_max_contour(mask_np, image_width, image_height)
        if contour is not None:
            
            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) >= 0]
            
            contour_world_points = [get_world_point_world_frame(camera_position, camera_orientation_q, "head", image.size, pixel_point, cam_info=cam_info) for pixel_point in contour_pixel_points]
            
            if len(contour_world_points) == 0:
                continue

            contour_world_points_np = np.array(contour_world_points)

            # ====================================================================
            # FIX 1: DBSCAN CLUSTERING TO REMOVE MASK BLEEDING (BACKGROUND REMOVAL)
            # ====================================================================
            # eps=0.025 means any points further than 2.5cm apart belong to different clusters.
            clustering = DBSCAN(eps=0.025, min_samples=5).fit(contour_world_points_np)
            labels = clustering.labels_
            
            # Separate points into their distinct clusters (ignoring -1, which DBSCAN flags as noise)
            clusters = [contour_world_points_np[labels == cluster_id] for cluster_id in set(labels) if cluster_id != -1]
            
            if len(clusters) > 0:
                # The handle is the object protruding *closest* to the camera.
                # We calculate the average Euclidean distance from the camera to each cluster and pick the minimum.
                cam_pos_np = np.array(camera_position)
                target_points = min(clusters, key=lambda c: np.mean(np.linalg.norm(c - cam_pos_np, axis=1)))
            else:
                # Fallback if clustering completely fails
                target_points = contour_world_points_np

            # ====================================================================
            # FIX 2: HYBRID TOP-SURFACE AND VOLUME BOUNDING BOX
            # ====================================================================
            max_z_coordinate = np.max(target_points[:, 2])
            min_z_coordinate = np.min(target_points[:, 2])
            
            # 1. Grasp rigid objects (top-down): Apply original top-surface filter to avoid table bleeding 
            top_surface_world_points = target_points[target_points[:, 2] > (max_z_coordinate - config.point_cloud_top_surface_filter)]
            
            # Safety check in case the top surface filter removes too many points
            if len(top_surface_world_points) < 3:
                top_surface_world_points = target_points

            rect = MultiPoint(top_surface_world_points[:, :2]).minimum_rotated_rectangle
            
            if isinstance(rect, Polygon):
                temp_box = np.array(rect.exterior.coords[:-1])
                width = np.linalg.norm(temp_box[1] - temp_box[0])
                length = np.linalg.norm(temp_box[2] - temp_box[1])
                
                # 2. Check if the object got squashed flat (like a door handle ridge).
                # If the footprint is thinner than 1.5cm, use ALL the clean cluster points!
                if min(width, length) < 0.015:  
                    rect = MultiPoint(target_points[:, :2]).minimum_rotated_rectangle

            # ====================================================================
            # BUILD FINAL 3D BOX
            # ====================================================================
            if isinstance(rect, Polygon):
                rect = polygon.orient(rect, sign=-1)
                box = rect.exterior.coords
                box = np.array(box[:-1])
                box_min_x = np.argmin(box[:, 0])
                
                box = np.roll(box, -box_min_x, axis=0)
                box_top = [list(point) + [max_z_coordinate] for point in box]
                box_btm = [list(point) + [min_z_coordinate] for point in box]
                
                box_top.append(list(np.mean(box_top, axis=0)))
                box_btm.append(list(np.mean(box_btm, axis=0)))
                
                bounding_cubes.append(box_top + box_btm)

                # Calculating rotation in world frame
                bounding_cubes_orientation_width = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
                bounding_cubes_orientation_length = np.arctan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
                bounding_cubes_orientations.append([bounding_cubes_orientation_width, bounding_cubes_orientation_length])

    bounding_cubes = np.array(bounding_cubes)

    return bounding_cubes, bounding_cubes_orientations

def save_mask_image(mask, path):
    """
    Save a binary or float mask to disk as a grayscale PNG, supporting both
    torch.Tensor and numpy arrays. Avoids importing torchvision to keep
    SAM3 runs torch-free.
    """
    if hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    # Convert to binary 0/255 for robustness
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(path)

def _depth_sample_to_ndc_z(depth_sample, named_cam_info=None, camera=None):
    """One depth sample -> OpenGL clip-space z, honouring the simulator's encoding.

    ``cam_info[camera]["depth_encoding"]`` is set by the sim adapter:
    ``"opengl"`` (PyBullet, default) or ``"linear_metric"`` (Genesis). Defaulting to
    ``"opengl"`` keeps every existing PyBullet payload - including the recorded
    goldens - bit-identical.
    """
    info = {}
    if isinstance(named_cam_info, dict):
        entry = named_cam_info.get(camera)
        if isinstance(entry, dict):
            info = entry
        # ``depth_encoding`` is a property of the *simulator*, so env.py puts it once at
        # the top level rather than repeating it per camera; per-camera wins if present.
        if "depth_encoding" not in info and "depth_encoding" in named_cam_info:
            info = dict(info, depth_encoding=named_cam_info["depth_encoding"])

    encoding = info.get("depth_encoding", camera_math.DEPTH_OPENGL)
    if encoding == camera_math.DEPTH_OPENGL:
        return 2.0 * depth_sample - 1.0

    near = float(info.get("znear", config.near_plane))
    far = float(info.get("zfar", config.far_plane))
    z_ndc = float(camera_math.metric_to_ndc_z(depth_sample, near, far))
    if os.environ.get("DEBUG_PINHOLE", "0") == "1":
        logger.info(PROGRESS + f"[depth] encoding={encoding} metric={float(depth_sample):.6f} "
                    f"near={near} far={far} -> z_ndc={z_ndc:.6f}" + ENDC)
    return z_ndc


def get_intrinsics_extrinsics(image_height, camera, camera_position, camera_orientation_q, cam_info=None):
    """
    Returns (K, Rt, view_matrix).
    - K: 3x3 intrinsics. Uses cam_info['K'] when present, else computes from config.fov.
    - Rt: 4x4 camera-to-world. If cam_info['viewMatrix'] exists, Rt = inv(viewMatrix) (column-major).
    - view_matrix: 4x4 view matrix from cam_info when provided; otherwise None.
    """
    named_cam_info = cam_info.get(camera, None) if isinstance(cam_info, dict) else {}
    view_matrix = None
    if not named_cam_info.get("viewMatrix", None) is None:   
        view_matrix = np.array(named_cam_info.get("viewMatrix"), dtype=float).reshape(4, 4, order='F')
        #logger.info(PROGRESS + f"########### view_matrix.shape= {view_matrix.shape} view_matrix={view_matrix}" + ENDC)
    
    projection_matrix = None    
    K = None
    if not named_cam_info.get("projectionMatrix", None) is None:
        projection_matrix = np.array(named_cam_info.get("projectionMatrix"), dtype=float).reshape(4, 4, order='F')
        #logger.info(PROGRESS + f"########### projection_matrix.shape= {projection_matrix.shape} projection_matrix={projection_matrix}" + ENDC)
    else:
        fov = (config.fov / 360) * 2 * math.pi
        f_x = f_y = image_height / (2 * math.tan(fov / 2))
        # Keep principal point at (0,0); caller subtracts image center, matching existing pipeline
        K = np.array([[f_x, 0, 0], [0, f_y, 0], [0, 0, 1]])

    # TODO:Delete: Legacy inv_view matrix 
    R_np = _quat_to_rotmat(camera_orientation_q)         
    Rt = np.hstack((R_np, np.array(camera_position).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return K, Rt, projection_matrix, view_matrix

def project_3d_world_pos_to_2d_pixel(camera_position, camera_orientation_q, camera, image_size, world_pos, cam_info):
    """
    Calc 2D projection in pixel image space from a 3D x,y,z world pos 
    Note: depth info is not required, as all the 3D world pos on a ray from a certain pixel x_0,y_0 would be projected on the same x_0, y_0
    """
    image_width, image_height = image_size
    K, Rt, projection_matrix, view_matrix = get_intrinsics_extrinsics(image_height, camera, camera_position, camera_orientation_q, cam_info=cam_info)
    pixel_2d = []
    if not view_matrix is None and not projection_matrix is None:
        # 3. Construct the View-Projection Matrix
        # VP = Projection @ View
        VP = projection_matrix @ view_matrix
        point_4d = np.append(world_pos, 1.0) # Convert to Homogeneous [x,y,z,1]
        clip = VP @ point_4d
        ndc = clip / clip[3] # Perspective Divide to get Normalized Device Coordinates (NDC)
                
        # Map NDC [-1, 1] to Pixel Coordinates x in [0, image_width],  y in [0, image_height]
        pixel_x = int(round((ndc[0] + 1.0) * image_width / 2.0))
        # Note: We subtract from 1.0 because Image Y is Top-Down, while NDC Y is Bottom-Up
        pixel_y = int(round((1.0 - ndc[1]) * image_height / 2.0))
        pixel_2d = [pixel_x, pixel_y]
    return pixel_2d
    
def get_world_point_world_frame(camera_position, camera_orientation_q, camera, image_size, point, cam_info=None):
    """
    Calc 3D world pos x,y,z from 2D pixel_point [x=point[0],y=point[1]] + depth value (point[2])
    Uses view and projection matrices obtained from Sim + depth value from depth map
    """
    image_width, image_height = image_size

    K, Rt, projection_matrix, view_matrix = get_intrinsics_extrinsics(image_height, camera, camera_position, camera_orientation_q, cam_info=cam_info)
    if args.save_grasp_inputs and camera not in _grasp_inputs_saved_cameras:
        try:
            if projection_matrix is not None:
                np.save(os.path.join(config.images_folder, f"{camera}_projection_matrix.npy"), projection_matrix)
            if view_matrix is not None:
                np.save(os.path.join(config.images_folder, f"{camera}_view_matrix.npy"), view_matrix)
            _grasp_inputs_saved_cameras.add(camera)
        except Exception as e:
            logger.info(PROGRESS + f"Warning: failed to save grasp input matrices: {e}" + ENDC)
    if os.environ.get("DEBUG_PINHOLE", "0") == "1":
        # Observability: log view/projection and pixel depth for this query
        try:
            logger.info(PROGRESS + f"[get_world_point_world_frame] Projected pixel(u={point[0]}, v={point[1]}), depth={float(point[2]):.6f}" + ENDC)
            if projection_matrix is not None:
                logger.info(PROGRESS + f"[get_world_point_world_frame] projection_matrix.shape: {projection_matrix.shape} projection_matrix: \n{projection_matrix}" + ENDC)
            else:
                logger.info(PROGRESS + "[get_world_point_world_frame] projection_matrix: None (using K intrinsics)" + ENDC)
            if view_matrix is not None:
                logger.info(PROGRESS + f"[get_world_point_world_frame] view_matrix.shape: {view_matrix.shape} view_matrix  =\n{view_matrix}" + ENDC)
            else:
                logger.info(PROGRESS + "[get_world_point_world_frame] view_matrix: None" + ENDC)
        except Exception as e:
            try:
                print("[utils.get_world_point_world_frame] log failure:", e)
            except Exception:
                pass


    if isinstance(cam_info, dict) and cam_info.get("new_3d_proj", False):
        # --- Implementation based on TestCameraUnprojection ---
        
        # 1. Map Pixel coordinates and Depth to Normalized Device Coordinates (NDC)
        # NDC Range: [-1, 1] for x, y, z
        
        # Map Pixel X [0, Width] -> NDC X [-1.0, 1.0]
        ndc_x = (2.0 * point[0] / image_width) - 1.0
        
        # Map Pixel Y [0, Height] -> NDC Y [1.0, -1.0]
        # Note: Image origin is Top-Left, OpenGL NDC origin is Bottom-Left.
        ndc_y = 1.0 - (2.0 * point[1] / image_height)
        
        # Map Depth Buffer -> NDC Z [-1.0, 1.0]
        # PyBullet ("opengl") hands back the non-linear z-buffer in [0, 1], which maps
        # directly. Genesis ("linear_metric") hands back metres along the optical axis,
        # which has to be re-projected through the frustum first - otherwise the point
        # lands at a plausible-looking but wrong depth. Doing the conversion here keeps
        # the single inverse view-projection code path below shared by both sims.
        z_ndc = _depth_sample_to_ndc_z(point[2], named_cam_info=cam_info, camera=camera)
        
        # 2. Create the Clip Space Vector [x, y, z, w]
        clip_pos = np.array([ndc_x, ndc_y, z_ndc, 1.0])
        
        # 3. Construct the View-Projection Matrix
        # VP = Projection @ View
        VP = projection_matrix @ view_matrix
        
        # 4. Apply Inverse View-Projection Matrix
        # Transform: Clip Space -> World Space
        inv_VP = np.linalg.inv(VP)
        world_hom = inv_VP @ clip_pos
        
        # 5. Perspective Divide
        # Recover Cartesian [x, y, z] from Homogeneous [xw, yw, zw, w]
        world_point_world_frame = world_hom[:3] / world_hom[3]
        if os.environ.get("DEBUG_PINHOLE", "0") == "1":
            logger.info(PROGRESS + f"[get_world_point_world_frame] 3D world point: {world_point_world_frame}" + ENDC)

    else:        
        # Legacy PyBullet path: recenter and apply axis flips
        pixel_point = np.array([[point[0] - (image_width / 2)], [(image_height / 2) - point[1]], [1.0]])
        if camera == "wrist":
            pixel_point = [pixel_point[1], pixel_point[0], pixel_point[2]]
        elif camera == "head":
            pixel_point = [-pixel_point[1], -pixel_point[0], pixel_point[2]]

        # logger.info(PROGRESS + f"########### |(np.linalg.inv(K) @ pixel_point)| ={np.linalg.norm(np.linalg.inv(K) @ pixel_point)}" + ENDC)
        world_point_camera_frame = (np.linalg.inv(K) @ pixel_point) * point[2]
        world_point_world_frame = Rt @ np.vstack((world_point_camera_frame, np.array([1.0])))
        world_point_world_frame = world_point_world_frame.squeeze()[:-1]
 
    return world_point_world_frame
