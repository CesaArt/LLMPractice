import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file

API_TOKEN = os.getenv("HF_API_TOKEN")
if not API_TOKEN:
    raise ValueError("Hugging Face API token not found. Add HF_API_TOKEN to .env file.")


# Define the model and the API endpoint
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
api_url = f"https://api-inference.huggingface.co/models/{model_name}"

# Set up headers with your API token
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Define the input prompt
data = {
    "inputs": "tell a joke about a dog and a cricket",
    "parameters": {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }
}

# Make the API request
response = requests.post(api_url, headers=headers, json=data)

# Check response and print the generated text
if response.status_code == 200:
    result = response.json()
    print(result[0]["generated_text"])
else:
    print(f"Error: {response.status_code}, {response.json()}")
