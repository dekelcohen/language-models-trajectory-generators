# -*- coding: utf-8 -*-

# azure_openai.py
import os
import base64
import json
import time
import requests
from dotenv import load_dotenv

# Disable SSL verify=False warning 
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env
load_dotenv()

# Canonical multimodal message building lives in message_media.py (shared by every
# provider). Re-exported here for backwards compatibility with existing imports.
# NOTE: Azure OpenAI chat completions accept images only - video input is not
# supported (see models.model_supports_video), so callers fall back to key frames.
from providers.llms.message_media import (  # noqa: F401
    encode_media,
    encode_image,
    append_images,
    append_videos,
    append_to_messages,
)


def post_with_retries(url, headers, payload, max_retries=5, timeout=(10,220)):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=False,
            )
            response.raise_for_status()
            return response  # ✅ success

        except requests.exceptions.RequestException as e:
            is_last_attempt = attempt == max_retries - 1

            print(f"\nModel Request {attempt + 1} failed:")

            # Print response details if available
            if hasattr(e, "response") and e.response is not None:
                print("Status Code:", e.response.status_code)
                print("Response Body:", e.response.text)
                status = e.response.status_code
            else:
                print("Error:", str(e))
                status = None

            # Decide if retryable
            retryable = isinstance(e, (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout
            )) or (status is not None and 500 <= status < 600)

            if not retryable or is_last_attempt:
                raise Exception(f"Request failed after {attempt + 1} attempts: {e}")

            sleep_time = 2 ** attempt
            print(f"Retrying in {sleep_time} seconds...\n")
            time.sleep(sleep_time)    
            
def call_llm(messages, azure_deployment_model = None, max_tokens=2048, temperature=0.1, reasoning_effort=None):
    """
    Call Azure OpenAI's chat completion endpoint with the given messages and max_tokens.

    Args:
        azure_deployment_model - name of azure model deployment (not always gpt-4 as in openai)
        messages (list): List of message objects for the conversation.
        max_tokens (int): Maximum tokens for the response.
        temperature : 0-1
        reasoning_effort: Optional reasoning effort level ("high", "medium", "low").

    Returns:
        dict: The parsed JSON response from the LLM.
    """
    # Retrieve configuration variables from the environment
    api_key = os.environ['AZURE_OPENAI_API_KEY']
    azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
    api_version = os.environ['AZURE_OPENAI_API_VERSION']
    if azure_deployment_model is None:
        azure_deployment_model = os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] # default model
    

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    # Build the payload
    payload = {
        "messages": messages,
        "max_completion_tokens": max_tokens,        
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Construct the Azure OpenAI endpoint URL
    GPT_ENDPOINT_URL = (
        f"{azure_endpoint}/openai/deployments/{azure_deployment_model}"
        f"/chat/completions?api-version={api_version}"
    )

    # Make the POST request
    response = post_with_retries(GPT_ENDPOINT_URL, headers=headers, payload=payload)
    
    # Parse the JSON response
    response_json = response.json()
    
    # Extract the message content from the first choice
    message_content = response_json["choices"][0]["message"]["content"]
    
    # Convert the content string to a JSON object (if necessary)
    final_response = message_content # json.loads(message_content)
    
    return final_response

if __name__ == "__main__":
    messages = [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an expert NLP and Search AI assistant that helps people summarize and search for information"
        }
      ]
    },
    {
        "role": "user",
        "content": "Extract the PLO entities from the following document: Roni went to the sea with Dani",
    },
    {
        "role": "assistant",
        "content": "Ron - PER, Dani - PER",
    },
    {
        "role": "user",
        "content": "Extract the PLO entities from the following document: Dudu went to the desert with Roei",
    }]
    ATTACH_IMAGES = False
    if ATTACH_IMAGES:
        from helpers.image_utils import list_file_paths    
        key_frames = list_file_paths()
        messages = append_to_messages('Did the robot succeeded in Task: grasp door handle? Please reason step by step and analyze what you see in the frames, regarding robot arm and gripper positions relative to the door and door handle', key_frames)
    response = call_llm(messages, azure_deployment_model = 'gpt-5')
    # Handle the response as needed (e.g., print or process)
    print(response)
