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

logger = None

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




def _strip_images_from_messages(messages, image_paths):
    """Return a deep copy of messages with base64 image_url blobs replaced by a
    reference to the copied image file under the 'images/' subfolder."""
    import copy
    import os
    stripped = copy.deepcopy(messages)
    img_iter = iter(image_paths or [])
    for msg in stripped:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    item.pop("image_url", None)
                    item["type"] = "image_ref"
                    src = next(img_iter, None)
                    item["image"] = f"images/{os.path.basename(src)}" if src else None
    return stripped


def log_messages(messages, file, prefix, image_paths=None):
    """messages to log
    - file: stream to print warnings to
    - prefix: filename prefix (e.g., "vlm_review" or "chat")
    - image_paths: images attached to the prompt; copied into an 'images/' subfolder

    Layout: ./images/vlm_images_prompts/{prefix}_prompt_<date_time>/
        {prefix}_messages.json   (conversation without embedded base64 images)
        images/                  (copies of all files in image_paths)
    """
    try:
        import json
        import os
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(config.images_folder, "vlm_images_prompts", f"{prefix}_prompt_{ts}")
        os.makedirs(base_dir, exist_ok=True)

        if image_paths:
            images_dir = os.path.join(base_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            for src in image_paths:
                try:
                    shutil.copy2(src, os.path.join(images_dir, os.path.basename(src)))
                except Exception as e:
                    logger.info(f"Warning: failed to copy prompt image {src}: {e}", file=file)

        out_path = os.path.join(base_dir, f"{prefix}_messages.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_strip_images_from_messages(messages, image_paths), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.info(f"Warning: failed to log messages JSON: {e}", file=file)
        
# Default options for _call_llm_provider_wrapper; callers may override any subset.
CHATGPT_DEFAULT_OPTIONS = {
    "log_msgs": False,
    "max_tokens": 60000,
    "reasoning_effort": None,
    "cache": None,          # LLMCache instance, or None to disable caching
    "cache_env_state": None,  # sim/env state dict used for level-2 smart match
}


def call_llm_provider(client, model, messages, max_tokens, reasoning_effort):
    """Dispatch to the correct provider and return the assistant response string"""
    if model.startswith("azure-"):
        from providers.llms.azure_openai import call_llm
        deployment = model[len("azure-"):]
        new_output = call_llm(messages, azure_deployment_model=deployment, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    elif model.startswith("or-"):
        from providers.llms.openrouter import call_openrouter
        openrouter_model = model[len("or-"):]
        new_output = call_openrouter(messages, model=openrouter_model, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    elif model.startswith("aws-"):
        from providers.llms.aws_bedrock import call_llm as call_bedrock
        bedrock_model_id = model[len("aws-"):]
        new_output = call_bedrock(messages, bedrock_model_id=bedrock_model_id, max_tokens=max_tokens, temperature=0, reasoning_effort=reasoning_effort)
    elif model.startswith("gemini-"):
        from providers.llms.gemini import call_gemini
        new_output = call_gemini(messages, model=model, max_tokens=max_tokens, temperature=0, reasoning_effort=reasoning_effort)
    else:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            stream=True
        )
        new_output = ""
        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                new_output += chunk_content
    return new_output
def _call_llm_provider_wrapper(client, model, new_prompt, messages, role, file=None, image_paths=None, options=None):
    """
    Call LLM (model - for azure or client - openai client) with new_prompt, existing conversation messages, role, image_paths - to attach images.
    new_prompt - optional: can be None or empty (only if image_paths is also None/empty)
    image_paths - optional: can be None or empty
    messages - existing conversation messages. will be updated with assistant response and returned.
      You can optionally add to messages before calling and pass new_prompt=None, image_paths=None
    options - optional dict merged over CHATGPT_DEFAULT_OPTIONS:
        log_msgs (bool): dump conversation JSON after the call.
        max_tokens (int): max completion tokens for the response.
        reasoning_effort (str|None): "high"/"medium"/"low" for reasoning models.
        cache (LLMCache|None): when None, caching is disabled. When set, the
            response is served from / stored to the cache keyed on model +
            text/params (level 1) and env state smart-match (level 2).
        cache_env_state (dict|None): sim/env state used for the level-2 match.
    """
    opts = dict(CHATGPT_DEFAULT_OPTIONS)
    if options:
        opts.update(options)
    log_msgs = opts["log_msgs"]
    max_tokens = opts["max_tokens"]
    reasoning_effort = opts["reasoning_effort"]
    cache = opts["cache"]
    cache_env_state = opts["cache_env_state"]
    
    logger.info(f"{role}:\n{new_prompt}")    
    
    from providers.llms.azure_openai import append_to_messages
    messages = append_to_messages(new_prompt, image_paths, messages, role)

    produced = {"called": False}
    def _producer():
        produced["called"] = True
        return call_llm_provider(client, model, messages, max_tokens, reasoning_effort)

    # Cache key is built AFTER images and everything are appended to messages.
    if cache is not None:
        params = {
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "temperature": 0,
        }
        new_output = cache.get(model, messages, params, cache_env_state, _producer)
    else:
        new_output = _producer()

    logger.info(f"assistant:\n{new_output}")
    if not new_output or len(new_output) < 5:
        logger.info(f"Warning: Model response is empty or very short {new_output}. messages: {messages}")
    messages.append({"role": "assistant", "content": new_output})
    if log_msgs:
        log_messages(messages, file, prefix=("vlm_review" if image_paths else "chat"), image_paths=image_paths)
    return messages


def fetch_env_state(main_connection):
    """Query the simulator for its current state via the GET_STATE command.

    Works with both the pybullet Pipe connection and the metaworld WS
    connection. Returns a JSON-serializable dict (empty on any failure).
    """
    if main_connection is None:
        return {}
    try:
        main_connection.send([config.GET_STATE])
        resp = main_connection.recv()
        return resp if isinstance(resp, dict) else {}
    except Exception as e:
        logger.info(f"Warning: fetch_env_state failed: {e}")
        return {}


def messages_have_images(messages):
    """True if any message carries image content (image_url part or data: URL)."""
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url" or "image_url" in part:
                        return True
        elif isinstance(content, str) and content.startswith("data:") and "base64," in content:
            return True
    return False


def call_llm_cached(main_connection, client, model, new_prompt, messages, role, file=None, image_paths=None, options=None):
    """Wrapper that fetches the current env state (GET_STATE) and threads it
    into the cache, then delegates to _call_llm_provider_wrapper.

    The GET_STATE round-trip and level-2 env-state comparison are only used
    when images are involved (image_paths given or messages already contain
    image content). Text-only calls are deterministic w.r.t. the env, so a
    single cache entry (env_state=None) is sufficient.

    If options["cache"] is None the GET_STATE round-trip is skipped entirely.
    """
    opts = dict(options or {})
    if opts.get("cache") is not None and opts.get("cache_env_state") is None:
        if image_paths or messages_have_images(messages):
            opts["cache_env_state"] = fetch_env_state(main_connection)
    return _call_llm_provider_wrapper(client, model, new_prompt, messages, role, file=file, image_paths=image_paths, options=opts)



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


