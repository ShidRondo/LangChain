from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key:
    print("Claude API key begins with:", api_key[:5], "...")
else:
    print("No API key found. Check your .env file.")