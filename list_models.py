import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# List available models
def list_models():
    url = "https://api.generativeai.google/v1beta/models"
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        models = response.json()
        print("Available models:", models)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Call the function to list models
list_models()
