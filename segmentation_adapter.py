"""
Segmentation provider adapter.

Required installs for LangSAM provider:
    pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
    pip install numpy==1.2.6  # pybullet 3.2.6 compiled against older numpy

This adapter normalizes different segmentation providers to the
get_langsam_output-like triple: (masks, boxes, text_labels), so
api.py requires minimal changes.
"""
from __future__ import annotations

import os
import tempfile
import re
from typing import List, Tuple
import numpy as np

# Logger
from config import OK, PROGRESS, WARNING, FAIL, ENDC
logger = None


# Optional bbox override supplied via CLI (wired from api.py)
# When all four are set to integers, the adapter will ignore provider
# boxes and use this rectangle for both boxes and masks.
_ovr_x1 = None
_ovr_x2 = None
_ovr_y1 = None
_ovr_y2 = None

# Optional object-match regex: when set, apply the bbox override
# only to predictions whose text label matches the regex.
_ovr_obj_regex = None  # type: re.Pattern | None
def override_bbox_from_globals(H: int, W: int):
    """
    If all override globals are set, return integer (x1,y1,x2,y2) as provided.
    No clipping is performed. If values violate image bounds/order, raise.
    """
    vals = (_ovr_x1, _ovr_y1, _ovr_x2, _ovr_y2)
    if any(v is None for v in vals):
        return None
    try:
        x1, y1, x2, y2 = map(int, vals)
    except Exception as e:
        raise ValueError(f"Failed to parse override bbox values {_ovr_x1,_ovr_y1,_ovr_x2,_ovr_y2}: {e}")

    problems = []
    if not (0 <= x1 <= W):
        problems.append(f"x1={x1} not in [0,{W}]")
    if not (0 <= x2 <= W):
        problems.append(f"x2={x2} not in [0,{W}]")
    if not (0 <= y1 <= H):
        problems.append(f"y1={y1} not in [0,{H}]")
    if not (0 <= y2 <= H):
        problems.append(f"y2={y2} not in [0,{H}]")
    if x2 <= x1 or y2 <= y1:
        problems.append(f"order invalid (need x1<x2 and y1<y2), got {(x1,y1,x2,y2)}")

    if problems:
        guidance = (
            "Provide a bbox within image bounds and correct order: "
            f"0 <= x1 < x2 <= {W}, 0 <= y1 < y2 <= {H}. "
            "Adjust your --ovr-bbox values accordingly."
        )
        raise ValueError(
            "Override bbox violates constraints: " + "; ".join(problems) + ". " + guidance
        )

    if logger:
        logger.info(WARNING + f"Using --ovr-bbox manual override instead of segmentation results: x1={x1} y1={y1} x2={x2} y2={y2}" + ENDC)
    return x1, y1, x2, y2


def _sam3_predict(
    image_pil,
    prompts: List[str],
):
    """
    Call SAM3 segmentation provider via robotic_perception and convert
    results to (masks, boxes, text_labels) where masks and boxes are torch tensors
    to match existing downstream expectations.
    """
    # Local import of provider factory (PYTHONPATH includes robotic_perception)
    from features_markers.segmentation_providers.segmentation_provider_factory import get_segmentation_provider

    # Pass in-memory image directly to provider (supports path or image)
    prov = get_segmentation_provider("sam3")
    results = prov.segment(image_pil, prompts or [])

    # Parse provider response
    # Expected shape: List[ {"sam": {"predictions": [ { np_mask, x,y,width,height, class } ] } } ]
    preds = []
    for item in results or []:
        sam = item.get("sam") if isinstance(item, dict) else None
        if not isinstance(sam, dict):
            continue
        preds.extend(sam.get("predictions", []) or [])

    if not preds:
        # Produce empty outputs consistent with downstream code
        empty_masks_np = np.empty((0, image_pil.height, image_pil.width), dtype=np.uint8)
        empty_boxes_np = np.empty((0, 4), dtype=np.float32)
        return empty_masks_np, empty_boxes_np, []

    # Build masks (H,W) and boxes (x1,y1,x2,y2)
    masks_np = []
    boxes_np = []
    labels: List[str] = []

    W = int(preds[0].get("rle_mask", {}).get("size", [image_pil.height, image_pil.width])[1])
    H = int(preds[0].get("rle_mask", {}).get("size", [image_pil.height, image_pil.width])[0])

    for p in preds:
        np_mask = p.get("np_mask")
        if np_mask is None:
            # Skip predictions without mask
            continue
        # Ensure shape is (H,W) uint8 -> float
        m = np.array(np_mask).astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        # Clip to image bounds
        m = m[:H, :W]
        masks_np.append(m)

        # Convert center box to [x1,y1,x2,y2]
        cx = float(p.get("x", 0.0))
        cy = float(p.get("y", 0.0))
        bw = float(p.get("width", 0.0))
        bh = float(p.get("height", 0.0))
        x1 = max(0.0, cx - bw / 2.0)
        y1 = max(0.0, cy - bh / 2.0)
        x2 = min(float(W), cx + bw / 2.0)
        y2 = min(float(H), cy + bh / 2.0)
        boxes_np.append([x1, y1, x2, y2])

        labels.append(str(p.get("class", "")))

    # Return NumPy to keep SAM3 torch-free
    masks_np_stacked = np.stack(masks_np, axis=0) if masks_np else np.empty((0, H, W), dtype=np.uint8)
    boxes_np_arr = np.array(boxes_np, dtype=np.float32) if boxes_np else np.empty((0, 4), dtype=np.float32)
    return masks_np_stacked, boxes_np_arr, labels


def _moondream_predict(image_pil, prompts: List[str]):
    """
    Call Moondream segmentation provider via robotic_perception and convert
    results to (masks, boxes, text_labels).
    - Masks: binary rectangles derived from provider bbox (no SVG rasterization).
    - Boxes: [x1,y1,x2,y2] in pixel coordinates.
    - Labels: prompt strings aligned with predictions.
    """
    from features_markers.segmentation_providers.segmentation_provider_factory import get_segmentation_provider

    prov = get_segmentation_provider("moondream")
    result = prov.segment(image_pil, prompts or [])

    preds = (result or {}).get("predictions", []) if isinstance(result, dict) else []
    if not preds:
        empty_masks_np = np.empty((0, image_pil.height, image_pil.width), dtype=np.uint8)
        empty_boxes_np = np.empty((0, 4), dtype=np.float32)
        return empty_masks_np, empty_boxes_np, []

    H, W = image_pil.height, image_pil.width
    masks_np = []
    boxes_np = []
    labels: List[str] = []

    # If user supplied a full override bbox, consider applying per-match
    override_rect = override_bbox_from_globals(H, W)

    for p in preds:
        bbox_px = p.get("bbox_pixels") or []
        if len(bbox_px) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_px]
        else:
            raise ValueError(f'bbox_pixels should be 4 ints {bbox_px}')

        label_str = str(p.get("class", ""))
        # Apply override only when set and either regex not provided (apply all) or label matches regex
        if override_rect is not None and (_ovr_obj_regex is None or _ovr_obj_regex.search(label_str or "")):
            x1, y1, x2, y2 = override_rect

        if x2 <= x1 or y2 <= y1:
            continue

        # Rectangle mask
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        masks_np.append(m)
        boxes_np.append([x1, y1, x2, y2])
        labels.append(str(p.get("class", "")))

    masks_np_stacked = np.stack(masks_np, axis=0) if masks_np else np.empty((0, H, W), dtype=np.uint8)
    boxes_np_arr = np.array(boxes_np, dtype=np.float32) if boxes_np else np.empty((0, 4), dtype=np.float32)
    return masks_np_stacked, boxes_np_arr, labels


def get_segmentation_output(
    image,
    langsam_model,
    segmentation_texts: List[str],
    segmentation_count: int,
    provider: str = "langsam",
):
    """
    Unified segmentation entry point with a signature similar to
    models.get_langsam_output(...).

    Returns (masks, boxes, text_labels) where masks/boxes are torch tensors
    to keep downstream code unchanged.
    """
    provider = (provider or "langsam").lower()
    if provider == "langsam":
        # Delegate to existing implementation
        import models
        return models.get_langsam_output(image, langsam_model, segmentation_texts, segmentation_count)
    elif provider in ("sam3", "roboflow-sam3", "sam"):
        return _sam3_predict(image, segmentation_texts)
    elif provider in ("moondream", "md", "moondreamvl"):
        return _moondream_predict(image, segmentation_texts)
    else:
        raise ValueError(f"Unknown segmentation provider: {provider}")

def set_override_bbox_from_string(ovr_str: str | None):
    """Parse a string "x1,y1,x2,y2" and set override globals.
    Accepts ints or floats; rounds to nearest int. None clears overrides.
    Raises ValueError on bad format.
    """
    global _ovr_x1, _ovr_y1, _ovr_x2, _ovr_y2
    if not ovr_str:
        _ovr_x1 = _ovr_y1 = _ovr_x2 = _ovr_y2 = None
        return
    parts = [p.strip() for p in str(ovr_str).split(',') if p.strip() != '']
    if len(parts) != 4:
        raise ValueError(f"--ovr-bbox must have 4 comma-separated values (x1,y1,x2,y2). Got: {ovr_str}")
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in parts]
    except Exception as e:
        raise ValueError(f"Failed to parse --ovr-bbox values from '{ovr_str}': {e}")
    _ovr_x1, _ovr_y1, _ovr_x2, _ovr_y2 = x1, y1, x2, y2


def set_override_object_regex(regex_str: str | None):
    """
    Set an optional regex. When provided, --ovr-bbox is applied only
    to predictions whose text label (provider 'class') matches the regex.
    If None or empty, the override applies to all predictions (legacy behavior).
    Raises ValueError when the regex is invalid.
    """
    global _ovr_obj_regex
    if not regex_str:
        _ovr_obj_regex = None
        return
    try:
        _ovr_obj_regex = re.compile(str(regex_str))
    except Exception as e:
        raise ValueError(f"Invalid --ovr-obj regex '{regex_str}': {e}")
