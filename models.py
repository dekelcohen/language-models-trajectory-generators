import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
import utils
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

sys.path.append("./XMem/")

def get_langsam_output(image, model, segmentation_texts, segmentation_count):
    """
    Updated to handle new LangSAM output format:
    model.predict([image], [text_prompt]) → list of result dicts.
    """

    # Ensure segmentation_texts is a list of strings.
    if isinstance(segmentation_texts, str):
        segmentation_texts = [segmentation_texts]

    # Run LangSAM
    results = model.predict([image], segmentation_texts)

    # LangSAM returns a list (one per image). We only passed one image → results[0]
    result = results[0]

    # Extract arrays
    masks_np = result["masks"]            # shape: (N, H, W)
    boxes_np = result["boxes"]            # shape: (N, 4)
    phrases = result["text_labels"]       # list of N strings

    # Convert numpy masks and boxes to torch tensors for visualization
    masks = torch.from_numpy(masks_np).bool()        # (N, H, W)
    boxes = torch.from_numpy(boxes_np).float()       # (N, 4)

    # Return predictions for downstream use; visualization handled elsewhere
    return masks.float(), boxes, phrases


def visualize_segmentation_overlay(image, masks_any, boxes_any, labels, out_path):
    """
    Overlay segmentation masks and bounding boxes on an RGB image and save.
    - Accepts masks/boxes as either numpy arrays or torch tensors.
    - Uses repo's thresholding via utils.get_segmentation_mask for consistency.
    - Always saves an image; if no masks/bboxes, saves the original image.
    Returns a dict with keys: saved, had_masks, had_boxes, filename.
    """
    # Normalize masks to binary list using existing thresholding utility
    bin_masks = utils.get_segmentation_mask(masks_any, getattr(config, 'segmentation_threshold', 0.2))

    to_tensor = transforms.PILToTensor()
    to_pil = transforms.ToPILImage()
    image_tensor = to_tensor(image)

    # Convert boxes to torch tensor if provided (use int for robust drawing)
    boxes_t = None
    try:
        if boxes_any is not None:
            if isinstance(boxes_any, np.ndarray):
                boxes_t = torch.from_numpy(boxes_any).to(dtype=torch.int64)
            elif isinstance(boxes_any, torch.Tensor):
                boxes_t = boxes_any.to(dtype=torch.int64)
    except Exception:
        boxes_t = None

    # Determine presence
    had_masks = False
    for m in bin_masks or []:
        arr = None
        if hasattr(m, 'detach'):
            arr = m.detach().cpu().numpy()
        else:
            try:
                arr = np.asarray(m)
            except Exception:
                arr = None
        if arr is not None and np.any(arr):
            had_masks = True
            break

    had_boxes = False
    if boxes_t is not None and boxes_t.numel() > 0 and boxes_t.shape[0] > 0:
        had_boxes = True

    # Draw masks (semi-transparent) and all boxes if present
    if had_masks:
        for m in bin_masks:
            if hasattr(m, 'detach'):
                mask_t = m.detach().cpu().bool()
            else:
                mask_t = torch.from_numpy(np.asarray(m)).bool()
            image_tensor = draw_segmentation_masks(image_tensor, mask_t, alpha=0.5, colors="cyan")  # type: ignore
    if had_boxes:
        image_tensor = draw_bounding_boxes(image_tensor, boxes_t, colors=["red"], width=3)

    # Save overlay (or original if nothing to draw)
    out_image = to_pil(image_tensor) if (had_masks or had_boxes) else image
    out_image.save(out_path)
    return {"saved": True, "had_masks": had_masks, "had_boxes": had_boxes, "filename": out_path}




def log_messages(messages, file, prefix):
    """messages to log
    - file: stream to print warnings to
    - prefix: filename prefix (e.g., "vlm_review" or "chat")
    """
    try:
        import json
        import os
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(config.images_folder, f"{prefix}_messages_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to log messages JSON: {e}", file=file)
        
def get_chatgpt_output(client, model, new_prompt, messages, role, file=sys.stdout, image_paths=None, log_msgs=False):
    """
    Call LLM (model - for zure or client - openai client) with new_prompt, existing conversation messages, role, image_paths - to attach images.
    new_prompt - optional: can be None or empty (only if image_paths is also None/empty)
    image_paths - optional: can be None or empty 
    messages - existing conversation messages. will be updated with assistant response and returned.
      You can optionally add to messages before calling and pass new_prompt=None, image_paths=None
    """
    print(role + ":", file=file)
    print(new_prompt, file=file)
    from azure_openai import append_to_messages
    messages_before = list(messages) if messages is not None else []
    messages = append_to_messages(new_prompt, image_paths, messages, role)
    # ----------------------------------------------------------------------
    # 1. Azure OpenAI mode: model name starts with "azure-"
    # ----------------------------------------------------------------------
    if model.startswith("azure-"):
        from azure_openai import call_llm
        deployment = model[len("azure-"):]
        print("assistant:", file=file)
        new_output = call_llm(messages, azure_deployment_model=deployment, max_tokens=60000)
        print(new_output, file=file)
        if not new_output or len(new_output) < 5:
            print(f"Warning: Model response is empty or very short {new_output}. messages: {messages}")        
    else:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            stream=True
        )
        print("assistant:", file=file)
        new_output = ""
        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content
            finish_reason = chunk.choices[0].finish_reason
            if chunk_content is not None:
                print(chunk_content, end="", file=file)
                new_output += chunk_content
            else:
                print("finish_reason:", finish_reason, file=file)
    messages.append({"role": "assistant", "content": new_output})
    if log_msgs:
        log_messages(messages, file, prefix=("vlm_review" if image_paths else "chat"))
    return messages

    # ----------------------------------------------------------------------
    # 2. OpenAI normal mode (original code)
    # ----------------------------------------------------------------------
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role": "assistant", "content": new_output})
    return messages



def get_xmem_output(model, device, trajectory_length):
    # Import XMem utilities lazily to avoid hard dependency when tracking is disabled
    try:
        from XMem.inference.inference_core import InferenceCore  # type: ignore
        from XMem.inference.interact.interactive_utils import (
            image_to_torch,
            index_numpy_to_one_hot_torch,
            torch_prob_to_numpy_mask,
            overlay_davis,
        )  # type: ignore
    except Exception as e:
        raise RuntimeError("XMem components are unavailable. Ensure submodule and deps are installed.") from e

    mask = np.array(Image.open(config.xmem_input_path).convert("L"))
    mask = np.unique(mask, return_inverse=True)[1].reshape(mask.shape)
    num_objects = len(np.unique(mask)) - 1

    torch.cuda.empty_cache()

    processor = InferenceCore(model, config.xmem_config)
    processor.set_all_labels(range(1, num_objects + 1))

    masks = []

    with torch.cuda.amp.autocast(enabled=True):

        for i in range(0, trajectory_length + 1, config.xmem_output_every):

            frame = np.array(Image.open(config.rgb_image_trajectory_path.format(step=i)).convert("RGB"))

            frame_torch, _ = image_to_torch(frame, device)
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                prediction = processor.step(frame_torch)

            prediction = torch_prob_to_numpy_mask(prediction)
            masks.append(prediction)

            if i % config.xmem_visualise_every == 0:
                visualisation = overlay_davis(frame, prediction)
                output = Image.fromarray(visualisation)
                output.save(config.xmem_output_path.format(step=i))

    return masks

