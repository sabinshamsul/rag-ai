# Take a prompt, return the model's text answer

import os 

from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None
MODEL = "gemini-2.5-flash"

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("NO GEMINI_API_KEY found!")
        _client = genai.Client(api_key=api_key)
    return _client

# Send a prompt to LLM and return its text answer
def generate(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text