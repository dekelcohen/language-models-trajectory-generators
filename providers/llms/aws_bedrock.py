# aws_bedrock.py
import os
import json
import subprocess
import time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Bedrock error codes that are safe to retry
_RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "RequestTimeoutException",
    "InternalServerException",
}

# Model IDs available via Bedrock cross-region inference profiles:
# "eu.anthropic.claude-sonnet-4-6"
# "eu.anthropic.claude-opus-4-8"
# "eu.anthropic.claude-opus-4-7"
# "openai.gpt-oss-120b-1:0"
# See: https://eu-central-1.console.aws.amazon.com/bedrock/home?region=eu-central-1#/inference-profiles


def _is_sso_expiry(exc) -> bool:
    """Return True if the exception looks like an expired SSO session."""
    msg = str(exc).lower()
    return "expired" in msg and ("token" in msg or "credentials" in msg)


def _is_access_denied(exc) -> bool:
    msg = str(exc).lower()
    return "access" in msg and "denied" in msg


def _handle_sso_expiry(exc):
    """Print the error, notify the user, and launch `aws sso login`."""
    print(f"\n{exc}")
    print("SSO Session expired or invalid. Launching browser login...")
    profile = os.environ.get("AWS_PROFILE", "default")
    subprocess.run(["aws", "sso", "login", "--profile", profile], check=False)


def _build_client():
    """Create a Bedrock Runtime boto3 client from environment variables."""
    region = os.environ.get("AWS_REGION", "eu-central-1")
    session_kwargs = {"region_name": region}

    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        session_kwargs["profile_name"] = aws_profile

    return boto3.Session(**session_kwargs).client("bedrock-runtime")


def _oai_content_to_bedrock(content):
    """
    Convert OpenAI-style message content to Bedrock/Anthropic native format.

    OpenAI image format:  {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
    Bedrock image format: {"type": "image", "source": {"type": "base64", "media_type": "<mime>", "data": "<data>"}}
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        result = []
        for item in content:
            if item.get("type") == "text":
                result.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:"):
                    # data:<mime_type>;base64,<data>
                    header, data = url.split(",", 1)
                    media_type = header.split(";")[0][len("data:"):]
                    result.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                else:
                    # URL-based image
                    result.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
        return result

    return [{"type": "text", "text": str(content)}]


def call_llm(messages, bedrock_model_id=None, max_tokens=60000, temperature=0, reasoning_effort=None, max_retries=5):
    """
    Call AWS Bedrock (Anthropic Messages API via boto3) with the given messages.

    Args:
        messages: List of OpenAI-style message dicts (role + content).
                  Supports text and image_url (base64 data URIs) content items.
        bedrock_model_id: Bedrock model or cross-region inference profile ID.
                          Falls back to env var AWS_BEDROCK_MODEL_ID, then
                          "eu.anthropic.claude-opus-4-7".
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature (0–1).
        reasoning_effort: Reserved for future extended-thinking support.
        max_retries: Retry attempts on transient errors.

    Returns:
        str: The assistant response text.
    """    
    if bedrock_model_id is None:
        bedrock_model_id = os.environ.get("AWS_BEDROCK_MODEL_ID")
    if bedrock_model_id is None:
        raise ValueError(
            "bedrock_model_id is required. Pass it explicitly or set the AWS_BEDROCK_MODEL_ID environment variable."
        )

    client = _build_client()

    # Bedrock Anthropic API separates system prompt from the messages array
    system_parts = []
    anthropic_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    item["text"] for item in content if item.get("type") == "text"
                )
        else:
            anthropic_messages.append({
                "role": role,
                "content": _oai_content_to_bedrock(content),
            })

    # Bedrock requires at least one user message; if only system messages were
    # provided (e.g. first-turn system-prompt-only calls), promote the combined
    # system text to a user message so the API call is valid.
    if not anthropic_messages and system_parts:
        anthropic_messages = [{"role": "user", "content": [{"type": "text", "text": "\n".join(system_parts)}]}]
        system_parts = []

    # opus and openai models reject the temperature parameter entirely
    _temperature_unsupported = "opus" in bedrock_model_id or "openai" in bedrock_model_id

    native_request = {
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if not _temperature_unsupported:
        native_request["temperature"] = temperature
    if "anthropic" in bedrock_model_id:
        native_request["anthropic_version"] = "bedrock-2023-05-31"
    if system_parts:
        native_request["system"] = "\n".join(system_parts)

    invoke_kwargs = {
        "modelId": bedrock_model_id,
        "body": json.dumps(native_request),
        "trace": "ENABLED",
    }
    # Optional guardrail settings from environment
    guardrail_id = os.environ.get("AWS_BEDROCK_GUARDRAIL_ID")
    if guardrail_id:
        invoke_kwargs["guardrailIdentifier"] = guardrail_id
        invoke_kwargs["guardrailVersion"] = os.environ.get("AWS_BEDROCK_GUARDRAIL_VERSION", "DRAFT")

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(**invoke_kwargs)
            model_response = json.loads(response["body"].read())
            if bedrock_model_id.startswith("openai"):
                return model_response["choices"][0]["message"]["content"]
            return model_response["content"][0]["text"]

        except ClientError as e:
            if _is_sso_expiry(e):
                _handle_sso_expiry(e)
                raise
            if _is_access_denied(e):
                print(f"\n{e}")
                raise
            error_code = e.response["Error"]["Code"]
            is_last = attempt == max_retries - 1
            print(f"\nModel Request {attempt + 1} failed ({error_code}): {e}")
            if error_code not in _RETRYABLE_ERROR_CODES or is_last:
                raise
            sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

        except Exception as e:
            if _is_sso_expiry(e):
                _handle_sso_expiry(e)
                raise
            if _is_access_denied(e):
                print(f"\n{e}")
                raise
            print(f"\nAttempt {attempt + 1} unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    from providers.llms.azure_openai import encode_image

    MODEL = os.environ.get("AWS_BEDROCK_MODEL_ID", "eu.anthropic.claude-opus-4-8")
    if MODEL is None:
        raise ValueError("Set AWS_BEDROCK_MODEL_ID environment variable to run tests.")

    # --- Test 1: Text-only query ---
    if False:
        print("=" * 60)
        print(f"Test 1: Text-only query  (model: {MODEL})")
        print("=" * 60)
        messages_text = [
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ]
        response_text = call_llm(messages_text, bedrock_model_id=MODEL)
        print(f"Response: {response_text}\n")

    # --- Test 2: Text + image query ---
    print("=" * 60)
    print(f"Test 2: Text + image query  (model: {MODEL})")
    print("=" * 60)
    image_path = "./images/rgb_image_head.png" 
    # "./outputs/longhorizon/dark_gray_pillar_blocks_door_less.png" #"./images/rgb_image_head.png"
    if os.path.exists(image_path):
        mime_type, base64_image = encode_image(image_path)
        messages_image = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: Open the door - grasp door handle. SubTasks: 1) Describe all objects in this image. 2) If the target object affordance is visible - say so and finish. If not, identify and describe where do you estimate that the target object and its affordance may be and the top objects the may occlude it. If the robot arm, on the way to the main target object affordance, may collide with objects in scene (that are not easy to bypass), state them"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]
        response_image = call_llm(messages_image, bedrock_model_id=MODEL)
        print(f"Response: {response_image}\n")
    else:
        print(f"Skipped: image not found at {image_path}")
        print("Run the simulation first or provide an image path.\n")

    print("Done.")