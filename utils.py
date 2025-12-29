import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import os
import config
from PIL import Image
from shapely.geometry import MultiPoint, Polygon, polygon

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


def get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q, K_override=None):
    """
    Returns intrinsics K and extrinsics Rt. If K_override is provided (from server), use it.
    When running under PyBullet, optionally compare PyBullet's getMatrixFromQuaternion with our pure-numpy
    version under DEBUG_DIFF flag for quick validation.
    """
    if K_override is not None:
        K = np.array(K_override, dtype=float)
    else:
        fov = (config.fov / 360) * 2 * math.pi
        f_x = f_y = image_height / (2 * math.tan(fov / 2))
        # Keep principal point at (0,0); caller subtracts image center, matching existing pipeline
        K = np.array([[f_x, 0, 0], [0, f_y, 0], [0, 0, 1]])

    R_np = _quat_to_rotmat(camera_orientation_q)

    # Optional: compare with PyBullet's quaternion->matrix if available and DEBUG_DIFF set
    if os.environ.get("DEBUG_DIFF", "0") == "1":
        try:
            import pybullet as p
            R_pb = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
            diff = np.abs(R_pb - R_np).max()
            if diff > 1e-6:
                print(f"[DEBUG_DIFF] Rotation matrix diff max: {diff}")
        except Exception as e:
            print(f"[DEBUG_DIFF] PyBullet compare failed: {e}")

    Rt = np.hstack((R_np, np.array(camera_position).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return K, Rt



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



def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, segmentation_count, K_override=None):

    image_width, image_height = image.size

    bounding_cubes = []
    bounding_cubes_orientations = []

    for i, mask in enumerate(masks):

        save_mask_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)

        contour = get_max_contour(mask_np, image_width, image_height)

        if contour is not None:

            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
            contour_world_points = [get_world_point_world_frame(camera_position, camera_orientation_q, "head", image, pixel_point, K_override=K_override) for pixel_point in contour_pixel_points]
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



def get_world_point_world_frame(camera_position, camera_orientation_q, camera, image, point, K_override=None):

    image_width, image_height = image.size

    if K_override is not None:
        K_use = np.array(K_override, dtype=float)
    else:
        K_use = None
    K, Rt = get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q, K_override=K_use)

    if K_override is not None:
        # Use pixel coordinates directly (u, v, 1) and rely on provided K
        pixel_point = np.array([[point[0]], [point[1]], [1.0]])
    else:
        # Legacy PyBullet path: recenter and apply axis flips
        pixel_point = np.array([[point[0] - (image_width / 2)], [(image_height / 2) - point[1]], [1.0]])
        if camera == "wrist":
            pixel_point = [pixel_point[1], pixel_point[0], pixel_point[2]]
        elif camera == "head":
            pixel_point = [-pixel_point[1], -pixel_point[0], pixel_point[2]]

    world_point_camera_frame = (np.linalg.inv(K) @ pixel_point) * point[2]
    world_point_world_frame = Rt @ np.vstack((world_point_camera_frame, np.array([1.0])))
    world_point_world_frame = world_point_world_frame.squeeze()[:-1]

    return world_point_world_frame


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
