import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os
import config
from config import OK, PROGRESS, FAIL, ENDC
from PIL import Image
from shapely.geometry import MultiPoint, Polygon, polygon

logger = None 
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

    contour_index = None
    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [(c, r) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
        if len(contour_points) > max_length:
            contour_index = c
            max_length = len(contour_points)

    if contour_index is None:
        return None

    return contours[contour_index]



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
    logger.info(PROGRESS + f"------------------ Enter get_bounding_cube_from_point_cloud(...)" + ENDC)
    
    for i, mask in enumerate(masks):

        save_mask_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)

        contour = get_max_contour(mask_np, image_width, image_height)
        if contour is not None:
            
            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
            if len(contour_pixel_points) > 0:
                _mean_px = np.mean(np.array(contour_pixel_points, dtype=np.float64), axis=0)
                logger.info(PROGRESS + f"[Contour] mean pixel_point (u,v,depth)={[_mean_px[0], _mean_px[1], _mean_px[2]]}" + ENDC)
            else:
                logger.info(PROGRESS + "[Contour] No pixels inside contour; mean undefined" + ENDC)
            logger.info(PROGRESS + f"++++++++++++++++++ Before get_world_point_world_frame len(contour_pixel_points)={len(contour_pixel_points)}" + ENDC)            
            contour_world_points = [get_world_point_world_frame(camera_position, camera_orientation_q, "head", image, pixel_point, cam_info=cam_info) for pixel_point in contour_pixel_points]
            # Optional depth statistics within the mask to probe Z handling
            if os.environ.get("DEBUG_DEPTH", "0") == "1":
                try:
                    _d = np.array([pt[2] for pt in contour_pixel_points], dtype=np.float32)
                    if _d.size > 0:
                        self_mean = float(_d.mean()); self_min = float(_d.min()); self_max = float(_d.max())
                        print(f"[Depth] mask idx={i} min={self_min:.3f} max={self_max:.3f} mean={self_mean:.3f}")
                except Exception:
                    pass
            max_z_coordinate = np.max(np.array(contour_world_points)[:, 2])
            min_z_coordinate = np.min(np.array(contour_world_points)[:, 2])
            top_surface_world_points = [world_point for world_point in contour_world_points if world_point[2] > max_z_coordinate - config.point_cloud_top_surface_filter]

            rect = MultiPoint([world_point[:2] for world_point in top_surface_world_points]).minimum_rotated_rectangle
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


def get_world_point_world_frame(camera_position, camera_orientation_q, camera, image, point, cam_info=None):
    image_width, image_height = image.size

    K, Rt, projection_matrix, view_matrix = get_intrinsics_extrinsics(image_height, camera, camera_position, camera_orientation_q, cam_info=cam_info)

    if isinstance(cam_info, dict) and cam_info.get("new_3d_proj", False):
        # --- Implementation based on TestCameraUnprojection ---
        
        # 1. Map Pixel coordinates and Depth to Normalized Device Coordinates (NDC)
        # NDC Range: [-1, 1] for x, y, z
        
        # Map Pixel X [0, Width] -> NDC X [-1.0, 1.0]
        ndc_x = (2.0 * point[0] / image_width) - 1.0
        
        # Map Pixel Y [0, Height] -> NDC Y [1.0, -1.0]
        # Note: Image origin is Top-Left, OpenGL NDC origin is Bottom-Left.
        ndc_y = 1.0 - (2.0 * point[1] / image_height)
        
        # Map Depth Buffer [0.0, 1.0] -> NDC Z [-1.0, 1.0]
        # Assumes point[2] is the non-linear depth buffer value.
        z_ndc = 2.0 * point[2] - 1.0
        
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
