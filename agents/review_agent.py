import logging
from typing import Any

import requests
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datapilot.review_agent")

class ReviewTaskInput(BaseModel):
    file_path: str = Field(description="The absolute or relative path to the CSV file to review")
    target: str | None = Field(default=None, description="The target column name if specified")
    code: str | None = Field(default=None, description="The python machine learning training code string")
    domain: str | None = Field(default=None, description="The detected domain of the dataset")

class ReviewAgentOutput(BaseModel):
    security_score: int = Field(description="Overall security score out of 100")
    issues_list: list[dict[str, Any]] = Field(description="List of security and privacy issues discovered")
    auto_fixes: list[str] = Field(description="Actionable fix recommendations for each issue")
    safe_to_proceed: bool = Field(description="Whether the dataset is safe to proceed with ML training")
    status: str = Field(default="complete", description="Completion status of the security review stage")

def send_status(message: str, status: str = "running"):
    """Helper to send progress status updates to the FastAPI MCP server for WebSocket broadcast."""
    url = "http://localhost:8000/status/broadcast"
    try:
        requests.post(url, json={
            "agent": "DataPilot_Security_Agent",
            "message": message,
            "status": status
        }, timeout=2)
    except Exception as e:
        logger.warning(f"Failed to broadcast status update: {e}")

def run_security_review(file_path: str, target: str | None = None, code: str | None = None) -> dict:
    """Runs a complete security and privacy audit on the dataset by calling the MCP server.

    Args:
        file_path: Path to the dataset CSV file.
        target: Optional target column name.
        code: Optional python code to scan.
    """
    send_status("Security Agent activated...", "running")
    send_status("Running Data Privacy scan...", "running")
    send_status("Checking for Data Leakage...", "running")
    send_status("Verifying Model Fairness...", "running")
    send_status("Performing Input Validation...", "running")

    url = "http://localhost:8000/tools/review"
    try:
        r = requests.post(url, json={
            "file_path": file_path,
            "target": target,
            "code": code
        }, timeout=20)

        if r.status_code == 200:
            response_json = r.json()
            data = response_json.get("data", {})
            send_status("Security Review Complete!", "complete")
            return data
        else:
            err_msg = f"MCP server error: {r.text}"
            send_status(f"Security Review Failed: {err_msg}", "error")
            return {"error": err_msg}
    except Exception as e:
        err_msg = f"Failed to connect to MCP server: {e}"
        send_status(f"Security Review Failed: {err_msg}", "error")
        return {"error": err_msg}

def get_model():
    return Gemini(
        model="gemini-1.5-flash",
        retry_options=types.HttpRetryOptions(attempts=1),
    )

review_agent = Agent(
    name="DataPilot_Security_Agent",
    model=get_model(),
    tools=[run_security_review],
    input_schema=ReviewTaskInput,
    output_schema=ReviewAgentOutput,
    description="Security Audit Agent. Scans datasets for PII leakage, data leakage, demographic fairness, and column injection vulnerabilities.",
    instruction="""You are a Data Security Expert working inside the DataPilot platform.
Your task is to run dataset privacy, leakage, and fairness audits before model training proceeds.

Your workflow:
1. Call the 'run_security_review' tool, passing the dataset 'file_path', 'target', and optional ML 'code' parameters.
2. Based on the returned review results:
   - Formulate clear explanations of privacy risks, leakage parameters, or fairness warnings without being alarming. Use terminology aligned with the 'domain' context if provided.
   - Propose clear auto-fix actions (e.g. 'Drop CustomerID column').
3. Populate the output schema fields:
   - 'security_score': Map directly to the overall security score (e.g. 90).
   - 'issues_list': Populate with the issues details (severity, category, message, suggestion).
   - 'auto_fixes': Extract only the actionable recommendations into a clean list of suggestions.
   - 'safe_to_proceed': Set to the boolean safe flag returned.
   - 'status': Always set to "complete".

Ensure all risk explanations are easy to read and thorough.
"""
)
