# gemini.py
import os
from dotenv import load_dotenv
from providers.llms.azure_openai import post_with_retries

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# reasoning_effort values accepted by the CLI:
# "xhigh", "high", "medium", "low", "minimal", "none"
# Gemini 3+ models use thinkingConfig.thinkingLevel: minimal, low, medium, high
# Gemini 2.5 models use thinkingConfig.thinkingBudget (an integer token budget)
_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
_THINKING_BUDGETS = {
    "none": 0,
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 24576,
    "xhigh": 32768,
}


def _oai_content_to_gemini(content):
    """
    Convert OpenAI-style message content to Gemini `parts`.

    OpenAI image format:  {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
    Gemini image format:  {"inline_data": {"mime_type": "<mime>", "data": "<data>"}}
    OpenAI video format:  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<data>"}}
    Gemini video format:  same inline_data block (mime_type "video/mp4"); Gemini samples
                          it at 1 FPS. Inline is capped at <100 MB per request; larger
                          clips would need the Files API.
    """
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append({"text": item["text"]})
            elif item.get("type") in ("image_url", "video_url"):
                url = item[item["type"]]["url"]
                if url.startswith("data:"):
                    # data:<mime_type>;base64,<data>
                    header, data = url.split(",", 1)
                    mime_type = header.split(";")[0][len("data:"):]
                    parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
                else:
                    # Gemini supports remote files / YouTube URLs via file_data
                    parts.append({"file_data": {"file_uri": url}})
        return parts

    return [{"text": str(content)}]


def _thinking_config(reasoning_effort, model):
    """Map a CLI reasoning_effort value to a Gemini thinkingConfig dict, or None.

    Gemini 2.5 models expect an integer `thinkingBudget`; newer (3+) models
    expect a string `thinkingLevel`.
    """
    if not reasoning_effort:
        return None
    effort = reasoning_effort.lower()

    # Gemini 2.5 family only supports thinkingBudget.
    if model.startswith("gemini-2"):
        budget = _THINKING_BUDGETS.get(effort)
        if budget is None:
            return None
        return {"thinkingBudget": budget}

    if effort == "none":
        return {"thinkingBudget": 0}
    if effort == "xhigh":
        effort = "high"
    if effort in _THINKING_LEVELS:
        return {"thinkingLevel": effort}
    return None


def call_gemini(messages, model, max_tokens=60000, temperature=0, reasoning_effort=None):
    """
    Call the Gemini generateContent REST endpoint.

    Args:
        messages: List of OpenAI-style message dicts (role + content).
                  Supports text and image_url (base64 data URIs) content items.
        model: Full Gemini model name, including the "gemini-" prefix
               (e.g. "gemini-2.5-flash", "gemini-2.5-pro").
        max_tokens: Maximum output tokens for the response.
        temperature: Sampling temperature (0-1).
        reasoning_effort: Optional reasoning effort ("high", "medium", "low", etc.).

    Returns:
        str: The assistant message text.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    # Gemini separates the system prompt (system_instruction) from contents,
    # and uses role "model" instead of "assistant".
    system_parts = []
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.extend(_oai_content_to_gemini(content))
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": _oai_content_to_gemini(content)})

    # Gemini requires at least one content entry; if only a system prompt was
    # provided, promote it to a user turn.
    if not contents and system_parts:
        contents = [{"role": "user", "parts": system_parts}]
        system_parts = []

    generation_config = {
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    thinking_cfg = _thinking_config(reasoning_effort, model)
    if thinking_cfg is not None:
        generation_config["thinkingConfig"] = thinking_cfg

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    response = post_with_retries(url, headers=headers, payload=payload)
    result = response.json()

    candidates = result.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {result}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    if not texts:
        raise RuntimeError(f"Gemini returned no text content: {result}")

    return "\n".join(texts)


if __name__ == "__main__":
    from providers.llms.azure_openai import encode_image

    MODEL = "gemini-2.5-flash"

    # --- Test 1: Simple text query ---
    print("=" * 60)
    print(f"Test 1: Text-only query  (model: {MODEL})")
    print("=" * 60)
    messages_text = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2+2? Answer in one word."}]}
    ]
    response_text = call_gemini(messages_text, model=MODEL)
    print(f"Response: {response_text}\n")

    # --- Test 2: Text + image query ---
    print("=" * 60)
    print(f"Test 2: Text + image query  (model: {MODEL})")
    print("=" * 60)
    image_path = "./images/rgb_image_head.png"
    if os.path.exists(image_path):
        mime_type, base64_image = encode_image(image_path)
        messages_image = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]
        response_image = call_gemini(messages_image, model=MODEL)
        print(f"Response: {response_image}\n")
    else:
        print(f"Skipped: image not found at {image_path}")
        print("Run the simulation first or provide an image path.\n")

    print("Done.")
