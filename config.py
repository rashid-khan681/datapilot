import os
import sys

from dotenv import load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

# Load environment variables from .env
load_dotenv()

# Propagate GOOGLE_API_KEY to GEMINI_API_KEY for Google GenAI SDK compatibility
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(..., min_length=1, validation_alias="GOOGLE_API_KEY")
    MCP_SERVER_URL: str = Field(..., min_length=1, validation_alias="MCP_SERVER_URL")

    class Config:
        env_file = ".env"
        extra = "ignore"

def get_missing_key_guidance(key_name: str) -> str:
    if key_name == "GOOGLE_API_KEY":
        return (
            "👉 GOOGLE_API_KEY is missing or empty!\n"
            "   Please check your .env file or environment variables.\n"
            "   You can get a Gemini API Key from Google AI Studio at:\n"
            "   https://aistudio.google.com/"
        )
    elif key_name == "MCP_SERVER_URL":
        return (
            "👉 MCP_SERVER_URL is missing or empty!\n"
            "   Please check your .env file or environment variables.\n"
            "   This URL points to your Model Context Protocol (MCP) server.\n"
            "   Typically, it should be set to: http://localhost:8000"
        )
    return f"👉 {key_name} is missing or empty."

def check_security():
    """
    Verifies that all configurations are properly set before pipeline starts.
    Prints green 'Config OK' if all found, or red 'Config Error' and exits if missing.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    try:
        s = Settings()
        print(f"{GREEN}Config OK{RESET}")
        return s
    except ValidationError as e:
        print(f"{RED}Config Error{RESET}", file=sys.stderr)

        missing = []
        for err in e.errors():
            loc = err.get("loc", [])
            if loc:
                missing.append(str(loc[0]))

        if not missing:
            if not os.getenv("GOOGLE_API_KEY"):
                missing.append("GOOGLE_API_KEY")
            if not os.getenv("MCP_SERVER_URL"):
                missing.append("MCP_SERVER_URL")

        for key in missing:
            print(get_missing_key_guidance(key), file=sys.stderr)

        sys.exit(1)

# Validate on startup (when imported)
settings = None
try:
    settings = Settings()
    print("\033[92mConfig OK\033[0m")
except ValidationError as e:
    print("\033[91mConfig Error\033[0m", file=sys.stderr)
    missing = []
    for err in e.errors():
        loc = err.get("loc", [])
        if loc:
            missing.append(str(loc[0]))
    if not missing:
        if not os.getenv("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
        if not os.getenv("MCP_SERVER_URL"):
            missing.append("MCP_SERVER_URL")

    for key in missing:
        print(get_missing_key_guidance(key), file=sys.stderr)
    sys.exit(1)
