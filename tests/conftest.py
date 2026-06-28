import os

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Propagate GOOGLE_API_KEY to GEMINI_API_KEY for Google GenAI SDK compatibility
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
