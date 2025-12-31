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
from typing import List, Tuple

import numpy as np


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

    for p in preds:
        bbox_px = p.get("bbox_pixels") or []
        if len(bbox_px) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox_px]
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        # Rectangle mask
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = 1
        masks_np.append(m)
        boxes_np.append([float(x1), float(y1), float(x2), float(y2)])
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
