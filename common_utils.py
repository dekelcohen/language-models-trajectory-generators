import os
import shutil
from typing import Iterable
from config import images_folder, trajectory_folder, overlay_folder, video_folder

class Trajectory:
    def __init__(self, points, desc):
        self.points = points # a straight-line end-effector trajectory between two 4D poses [x,y,z,theta]
        self.desc = desc # short sentence to describe the motion and its end_pose

def ensure_image_dirs_exist(delete: bool = False, extra_dirs: Iterable[str] | None = None) -> None:
    """Ensure image directories exist; optionally delete existing contents first.

    Uses the same directory set as tests and runtime:
    - ./images
    - ./images/trajectory
    - ./images/overlay
    - ./images/videos

    If delete is True, removes these directories (if present) before recreating.
    Extra directories can be provided via extra_dirs.
    """
    dirs = [
        images_folder,
        trajectory_folder,
        overlay_folder,
        video_folder,
    ]
    if extra_dirs:
        for d in extra_dirs:
            if d not in dirs:
                dirs.append(d)

    if delete:
        for d in dirs:
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d)
            except Exception:
                # Best effort cleanup; leave to creation phase
                pass

    for d in dirs:
        os.makedirs(d, exist_ok=True)

