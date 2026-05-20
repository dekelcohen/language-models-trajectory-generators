import numpy as np
import os


def get_grasp_pose_candidates(object_name="open_door"):
    """Load pre-computed grasp pose candidates from an .npz file.

    Args:
        object_name: Name used to locate the file at ./outputs/graspgen/grasp_poses_{object_name}.npz

    Returns:
        poses: np.ndarray of shape (N, 4, 4) – 4x4 homogeneous transformation matrices.
        scores: np.ndarray of shape (N,) – grasp quality scores (higher is better).
    """
    npz_path = os.path.join(".", "outputs", "graspgen", f"grasp_poses_{object_name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Grasp poses file not found: {npz_path}")

    data = np.load(npz_path)
    poses = data["poses"]   # (N, 4, 4)
    scores = data["scores"] # (N,)
    return poses, scores
