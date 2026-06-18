import os
import requests
import json

# 1. Retrieve the API token from the OS environment variables
# Make sure you exported this in your terminal, e.g., export BEDROCK_API_TOKEN="your_token"
api_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

if not api_token:
    print("Error: BEDROCK_API_TOKEN environment variable not set.")
    exit(1)

model_id = 'eu.anthropic.claude-opus-4-7'
# 2. Define the URL
# NOTE: If using an API Token, ensure this is your company's API Gateway proxy URL.
# If you hit amazonaws.com directly with a Bearer token, it will return an AccessDenied error.
url = f"https://bedrock-runtime.eu-central-1.amazonaws.com/model/{model_id}/invoke"

# 3. Set up the exact Headers you requested
headers = {
    # Standard Bearer auth (Adjust if your company uses a different header like 'x-api-key')
    "Authorization": f"Bearer {api_token}", 
    
    # Bedrock specific headers
    "X-Amzn-Bedrock-GuardrailIdentifier": "jbidmu6tsbf0",  # Replace with actual ID
    "X-Amzn-Bedrock-GuardrailVersion": "DRAFT",
    "X-Amzn-Bedrock-Trace": "ENABLED",
    
    # Standard HTTP headers
    "Accept": "*/*",
    "Content-Type": "application/json"
}

# 4. Define the Body
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        { "role": "user", "content": "What is the date today?" }
    ],
    "max_tokens": 200
}

# 5. Make the POST Request
try:
    print(f"Sending POST request to {url}...")
    response = requests.post(url, headers=headers, json=body)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Print the raw JSON response
    print("\n--- SUCCESS ---")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.HTTPError as err:
    print(f"\n--- HTTP ERROR ---")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")