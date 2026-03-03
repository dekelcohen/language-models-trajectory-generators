import os
import shutil
from typing import Iterable
from config import images_folder, trajectory_folder, overlay_folder

def ensure_image_dirs_exist(delete: bool = False, extra_dirs: Iterable[str] | None = None) -> None:
    """Ensure image directories exist; optionally delete existing contents first.

    Uses the same directory set as tests and runtime:
    - ./images
    - ./images/trajectory
    - ./images/overlay

    If delete is True, removes these directories (if present) before recreating.
    Extra directories can be provided via extra_dirs.
    """
    dirs = [
        images_folder,
        trajectory_folder,
        overlay_folder,
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

