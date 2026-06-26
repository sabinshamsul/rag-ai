# To confirm Gemini API key works by loading it from .env file

import os

from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("No GEMINI_API_KEY found!")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in 1 sentence",
)

print("Gemini replied:")
print (response.text)