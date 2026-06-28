import logging
from typing import Any

import requests
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datapilot.eda_agent")

class DataScienceTaskInput(BaseModel):
    file_path: str = Field(description="The absolute or relative path to the CSV file to process")
    goal: str = Field(description="The user's analytical or machine learning goal")
    domain: str | None = Field(default=None, description="The detected domain of the dataset (e.g. MEDICAL, FINANCE, CUSTOMER, HR, EDUCATION, GENERAL)")

class EDAAgentOutput(BaseModel):
    raw_results: dict[str, Any] = Field(description="Raw analysis results returned from the MCP run_eda tool")
    insights: str = Field(description="Agent's natural language interpretation and pattern analysis of the data")
    warnings: list[str] = Field(description="List of security, schema, or missing value warnings discovered")
    recommendations: list[str] = Field(description="Suggested next steps for data cleaning, preprocessing, or ML training")
    status: str = Field(default="complete", description="Completion status of the EDA stage")

def send_status(message: str, status: str = "running"):
    """Helper to send progress status updates to the FastAPI MCP server for WebSocket broadcast."""
    url = "http://localhost:8000/status/broadcast"
    try:
        requests.post(url, json={
            "agent": "DataPilot_EDA_Agent",
            "message": message,
            "status": status
        }, timeout=2)
    except Exception as e:
        logger.warning(f"Failed to broadcast status update: {e}")

def run_eda(file_path: str) -> dict:
    """Runs EDA analysis on the dataset by calling the MCP server.

    Args:
        file_path: Path to the CSV file to analyze.
    """
    send_status("EDA Agent started...", "running")
    send_status("Analyzing data structure...", "running")

    url = "http://localhost:8000/tools/eda"
    try:
        r = requests.post(url, json={"file_path": file_path}, timeout=20)
        if r.status_code == 200:
            send_status("Finding patterns...", "running")
            response_json = r.json()
            data = response_json.get("data", {})
            send_status("EDA Complete!", "complete")
            return data
        else:
            err_msg = f"MCP server error: {r.text}"
            send_status(f"EDA Failed: {err_msg}", "error")
            return {"error": err_msg}
    except Exception as e:
        err_msg = f"Failed to connect to MCP server: {e}"
        send_status(f"EDA Failed: {err_msg}", "error")
        return {"error": err_msg}

def get_model():
    return Gemini(
        model="gemini-2.0-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    )

eda_agent = Agent(
    name="DataPilot_EDA_Agent",
    model=get_model(),
    tools=[run_eda],
    input_schema=DataScienceTaskInput,
    output_schema=EDAAgentOutput,
    description="Exploratory Data Analysis Agent. Analyzes schemas, missing values, correlations, and registers Plotly HTML charts.",
    instruction="""You are an expert Data Analyst working inside the DataPilot platform.
When given a dataset path, call the 'run_eda' tool to analyze the dataset.
Once the tool execution completes, evaluate the returned results and format your response by populating the output schema:
- 'raw_results': Populate directly with the dictionary returned by the 'run_eda' tool.
- 'insights': Summarize key patterns, correlations, and dataset features in simple, clear English with emojis and clean sections. Explain what the data is telling us in relation to the goal. Adjust your analysis tone and terms based on the 'domain' context (e.g. risk-focused for FINANCE, clinical for MEDICAL).
- 'warnings': List all warnings, anomalies, missing values, or potential PII issues found.
- 'recommendations': Suggest concrete next steps for data cleaning, preprocessing, or ML model training based on the insights.
- 'status': Always set to "complete".

Ensure all insights are detailed and easy to read. Use clean headers and emojis for premium readability.
"""
)
