# openrouter.py
import os
import requests
from dotenv import load_dotenv
from azure_openai import post_with_retries

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(messages, model, max_tokens=60000, temperature=0):
    """
    Call OpenRouter chat completion endpoint.

    Args:
        messages: List of message dicts (supports text and image_url content).
        model: OpenRouter model identifier (e.g. "google/gemini-2.5-flash").
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.

    Returns:
        str: The assistant message content.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    # Most providers don't support image_url in system messages — promote to user
    sanitized = []
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") == "system" and isinstance(content, list):
            has_image = any(
                item.get("type") == "image_url" for item in content if isinstance(item, dict)
            )
            if has_image:
                sanitized.append({**msg, "role": "user"})
                continue
        sanitized.append(msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": sanitized,
        "max_tokens": max_tokens,
        "temperature": temperature,        
    }

    response = post_with_retries(OPENROUTER_BASE_URL, headers=headers, payload=payload)
    result = response.json()

    if "choices" not in result or len(result["choices"]) == 0:
        raise RuntimeError(f"OpenRouter returned no choices: {result}")

    return result["choices"][0]["message"]["content"]


if __name__ == "__main__":
    from azure_openai import encode_image

    MODEL = 'openai/gpt-5.5' # "google/gemini-2.5-flash"

    # --- Test 1: Simple text query ---
    print("=" * 60)
    print(f"Test 1: Text-only query  (model: {MODEL})")
    print("=" * 60)
    messages_text = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2+2? Answer in one word."}]}
    ]
    response_text = call_openrouter(messages_text, model=MODEL)
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
        response_image = call_openrouter(messages_image, model=MODEL)
        print(f"Response: {response_image}\n")
    else:
        print(f"Skipped: image not found at {image_path}")
        print("Run the simulation first or provide an image path.\n")

    print("Done.")
