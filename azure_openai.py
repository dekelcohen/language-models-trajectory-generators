# -*- coding: utf-8 -*-

# azure_openai.py
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def call_llm(messages, azure_deployment_model = None, max_tokens=2048, temperature=0.1):
    """
    Call Azure OpenAI's chat completion endpoint with the given messages and max_tokens.

    Args:
        azure_deployment_model - name of azure model deployment (not always gpt-4 as in openai)
        messages (list): List of message objects for the conversation.
        max_tokens (int): Maximum tokens for the response.
        temperature : 0-1
        

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

    # Construct the Azure OpenAI endpoint URL
    GPT_ENDPOINT_URL = (
        f"{azure_endpoint}/openai/deployments/{azure_deployment_model}"
        f"/chat/completions?api-version={api_version}"
    )

    # Make the POST request
    try:
        response = requests.post(GPT_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for non-2xx responses
    except requests.RequestException as e:        
        if e.response is not None:
            print("Status Code:", e.response.status_code)
            print("Response Body:", e.response.text)
        raise SystemExit(f"Failed to make the request. Error: {e}")

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
    response = call_llm(messages, azure_deployment_model = 'gpt-5')
    # Handle the response as needed (e.g., print or process)
    print(response.json())
