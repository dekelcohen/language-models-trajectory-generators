import os
import boto3
import json
from botocore.exceptions import ClientError

# ---------------------------------------------------------
# CREDENTIALS STEP
# Replace these strings with the temporary credentials 
# you copied from the AWS Access Portal in your browser.
# ---------------------------------------------------------
my_access_key = os.environ["AWS_ACCESS_KEY_ID"]
my_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
my_session_token = os.environ["AWS_SESSION_TOKEN"]

# Create a boto3 Session using your temporary credentials
session = boto3.Session(
    aws_access_key_id=my_access_key,
    aws_secret_access_key=my_secret_key,
    aws_session_token=my_session_token,
    region_name="eu-central-1"  # Replace if your Bedrock models are in a different region
)

# Use the session to create the Bedrock Runtime client
client = session.client("bedrock-runtime")

# Set model_id Inference profile ID from:
#https://eu-central-1.console.aws.amazon.com/bedrock/home?region=eu-central-1#/inference-profiles

model_id = "openai.gpt-oss-120b-1:0"
# "eu.anthropic.claude-sonnet-4-6"
# "eu.anthropic.claude-opus-4-8"
# "eu.anthropic.claude-opus-4-7" # "anthropic.claude-opus-4-7"

# Define the prompt for the model.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Format the request payload using the model's native Anthropic Messages API structure.
native_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,    
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ],
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    print(f"Invoking {model_id}...")
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, 
                                   body=request,
                                   guardrailIdentifier="jbidmu6tsbf0", # Header X-Amzn-Bedrock-GuardrailIdentifier
                                   guardrailVersion="DRAFT",           # Header X-Amzn-Bedrock-GuardrailVersion
                                   trace="ENABLED",                    # Header X-Amzn-Bedrock-Trace
                                  )

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["content"][0]["text"]
print("\n--- RESPONSE ---")
print(response_text)