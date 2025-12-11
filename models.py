import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import config
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

sys.path.append("./XMem/")

from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

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

    # --- Visualization identical to your old code ---
    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + 5 * len(masks), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)

    to_tensor = transforms.PILToTensor()
    to_pil = transforms.ToPILImage()

    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        image_tensor = to_tensor(image)
        box = box.unsqueeze(0)

        image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
        image_pil = to_pil(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(
            box[0][0].item(),
            box[0][1].item() - 15,
            phrase,
            color="red",
            bbox={"facecolor": "white", "edgecolor": "red", "boxstyle": "square"}
        )

    plt.savefig(config.langsam_image_path.format(object=segmentation_count))
    plt.show()

    # Return exactly what your downstream code expects
    return masks.float(), boxes, phrases




def get_chatgpt_output(client, model, new_prompt, messages, role, file=sys.stdout):
    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role": role, "content": new_prompt})

    # ----------------------------------------------------------------------
    # 1. Azure OpenAI mode: model name starts with "azure-"
    # ----------------------------------------------------------------------
    if model.startswith("azure-"):
        from azure_openai import call_llm   # import your Azure helper

        # deployment name is the substring after "azure-"
        deployment = model[len("azure-"):]

        print("assistant:", file=file)

        # Call Azure endpoint (non-streaming)
        new_output = call_llm(messages, azure_deployment_model=deployment)

        # Convert LLM JSON response to text
        #new_output = json.dumps(azure_response, ensure_ascii=False)

        print(new_output, file=file)

        messages.append({"role": "assistant", "content": new_output})
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
