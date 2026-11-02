import numpy as np
from PIL import Image

from helpers.review_frames import pick_by_change, select_key_frames


def test_pick_keeps_first_last_and_concentrates_on_motion():
    diffs = [0.0] + [0.01] * 60 + [5.0] * 20 + [0.01] * 19  # motion in 61..80
    keep = pick_by_change(diffs, 12)
    assert keep[0] == 0 and keep[-1] == 99 and len(keep) == 12
    in_motion = sum(61 <= k <= 80 for k in keep)
    assert in_motion >= 6


def test_select_from_folder_skips_missing_start_and_keeps_final(tmp_path):
    # frames start at 63, not at start_idx=60 (the case that made stride selection give 4 frames)
    for i in range(63, 200):
        v = 255 if i > 150 else 0
        Image.fromarray(np.full((16, 16, 3), v, np.uint8)).save(tmp_path / f"rgb_image_{i}.png")
    sel = select_key_frames(str(tmp_path), "rgb_image", 60, 10)
    idx = [i for i, _ in sel]
    assert len(idx) == 10 and idx[0] == 63 and idx[-1] == 199
    assert any(149 <= i <= 152 for i in idx)  # the change is captured
