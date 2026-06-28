import logging
from typing import Any

import requests
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datapilot.ml_agent")

class MLTaskInput(BaseModel):
    file_path: str = Field(description="The absolute or relative path to the CSV file to train on")
    goal: str = Field(description="The user's analytical or machine learning goal")
    eda_insights: str | None = Field(default=None, description="Exploratory Data Analysis insights and warnings")
    test_file_path: str | None = Field(default=None, description="Optional path to test dataset for predictions")
    domain: str | None = Field(default=None, description="The detected domain of the dataset")
    force_continue: bool = Field(default=False, description="Whether to proceed despite low confidence target warnings")

class MLAgentOutput(BaseModel):
    model_name: str = Field(description="Name of the winning model selected")
    accuracy_score: float = Field(description="Performance score (Accuracy/F1 for classification, R2 for regression)")
    performance_summary: str = Field(description="Plain English explanation of model performance and metrics")
    feature_importance_top5: list[dict[str, Any]] = Field(description="List of top 5 features and their importance scores")
    business_insight: str = Field(description="Actionable business insights and recommendations suggested by the model results")
    predictions_file_path: str = Field(default="", description="File path to the predictions CSV if generated")
    status: str = Field(default="complete", description="Completion status of the ML stage")

def send_status(message: str, status: str = "running"):
    """Helper to send progress status updates to the FastAPI MCP server for WebSocket broadcast."""
    url = "http://localhost:8000/status/broadcast"
    try:
        requests.post(url, json={
            "agent": "DataPilot_ML_Agent",
            "message": message,
            "status": status
        }, timeout=2)
    except Exception as e:
        logger.warning(f"Failed to broadcast status update: {e}")

def train_model(file_path: str, target: str = None, test_file_path: str = None, goal: str = None, force_continue: bool = False) -> dict:
    """Trains the best ML model on the dataset by calling the MCP server.
    
    Args:
        file_path: Path to the training dataset.
        target: Target column name. If None, auto-detected.
        test_file_path: Optional path to the test dataset.
        goal: The user's goal string to guide target column detection.
        force_continue: Boolean to force continue despite low confidence target warnings.
    """
    send_status("ML Agent activated...", "running")
    send_status("Selecting target column...", "running")

    # Broadcast intermediate training stages sequentially
    send_status("Training Random Forest...", "running")
    send_status("Training XGBoost...", "running")

    url = "http://localhost:8000/tools/train"
    try:
        r = requests.post(url, json={
            "file_path": file_path,
            "target": target,
            "test_file_path": test_file_path,
            "goal": goal,
            "force_continue": force_continue
        }, timeout=120)

        send_status("Selecting best model...", "running")

        if r.status_code == 200:
            response_json = r.json()
            data = response_json.get("data", {})
            send_status("ML Training Complete!", "complete")
            return data
        else:
            err_msg = f"MCP server error: {r.text}"
            send_status(f"ML Training Failed: {err_msg}", "error")
            return {"error": err_msg}
    except Exception as e:
        err_msg = f"Failed to connect to MCP server: {e}"
        send_status(f"ML Training Failed: {err_msg}", "error")
        return {"error": err_msg}

def get_model():
    return Gemini(
        model="gemini-2.0-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    )

ml_agent = Agent(
    name="DataPilot_ML_Agent",
    model=get_model(),
    tools=[train_model],
    input_schema=MLTaskInput,
    output_schema=MLAgentOutput,
    description="Machine Learning Engineer Agent. Preprocesses dataset, trains models, evaluates metrics, and outputs serializations.",
    instruction="""You are an expert Machine Learning Engineer working inside the DataPilot platform.
You receive the dataset path, target, goal, optional domain, and optional EDA insights.

Your workflow:
1. Read the input fields. If the EDA insights suggest a target column or if you can infer the target column from the goal, identify it.
2. Call the 'train_model' tool, passing the dataset 'file_path', the identified 'target' column, the 'goal', and 'force_continue' flag. Pass 'test_file_path' if provided.
3. Once training completes, explain the results in simple terms that even a non-technical person can understand. Adopt a tone suitable for the specified 'domain' context (e.g. risk-focused for FINANCE, clinical for MEDICAL).
4. Populate the output schema fields:
   - 'model_name': Populate with the winning model name (e.g. Random Forest, Logistic Regression).
   - 'accuracy_score': Populate with the best model's primary metric (e.g. Accuracy/F1 or R2).
   - 'performance_summary': Write a plain English explanation of what this score means in real life.
   - 'feature_importance_top5': Populate with the top 5 features and their values as returned in 'top_features' from the tool (e.g. [{'name': 'MonthlyCharges', 'importance': 0.26}, ...]).
   - 'business_insight': Write an actionable business decision recommendation based on which features matter most.
   - 'predictions_file_path': The path of predictions CSV (if generated, else empty string).
   - 'status': Always set to "complete".

Ensure your explanations are clear and use headers.
"""
)
