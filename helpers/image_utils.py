import os
# Unified lookahead helper used at start and mid-sequence
def find_available_frame(
    folder_path: str,
    base_name: str,
    current_idx: int,
    end_idx: int,
    ext: str,
    lookahead_max: int,
    include_current: bool = True,
):
    """
    Find first existing frame index in [current_idx, current_idx+lookahead_max]
    (or (current_idx+1, ...) when include_current=False). Returns (path, idx) or (None, None).
    """
    start = int(current_idx) + (0 if include_current else 1)
    stop = int(current_idx) + int(lookahead_max)
    if end_idx != float('inf'):
        stop = min(stop, int(end_idx))
    for idx in range(start, stop + 1):
        candidate = os.path.join(folder_path, f"{base_name}_{idx}.{ext}")
        if os.path.exists(candidate):
            return candidate, idx
    return None, None


def list_file_paths(
    root: str = "./images/trajectory",
    base_name: str = "rgb_image",
    start_idx: int = 0,
    end_idx: int = float("inf"),
    ext: str = "png",
    skip: int = 5,
    lookahead_max: int = 10,
):
    """
    List image paths in fixed index steps.
    If an exact frame is missing, use find_available_frame() to look ahead.

    Example:
        0,5,10,15...
        If 5 missing → maybe 6 (if within lookahead_max)

    Returns:
        List[str]
    """

    if not os.path.isdir(root):
        print(f"[Error] Folder not found: {root}")
        return []

    paths = []
    used_indices = set()

    current_idx = int(start_idx)

    while current_idx <= end_idx:

        candidate = os.path.join(root, f"{base_name}_{current_idx}.{ext}")

        if os.path.exists(candidate):
            if current_idx not in used_indices:
                paths.append(candidate)
                used_indices.add(current_idx)

        else:
            # --- Use your helper ---
            next_path, next_idx = find_available_frame(
                root,
                base_name,
                current_idx,
                end_idx,
                ext,
                lookahead_max,
                include_current=False,
            )

            if next_path is not None and next_idx not in used_indices:
                paths.append(next_path)
                used_indices.add(next_idx)

            elif end_idx == float("inf"):
                # No more frames in lookahead → sequence likely ended
                break

        current_idx += int(skip)

    print(f"[Info] Found {len(paths)} file(s) with skip={skip} (lookahead={lookahead_max}).")
    return paths