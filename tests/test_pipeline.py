import os
import requests
import pytest
import asyncio
from agents.orchestrator import run_pipeline

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

@pytest.mark.asyncio
async def test_titanic_pipeline():
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uploads_dir = os.path.join(workspace_root, "uploads")
    outputs_dir = os.path.join(workspace_root, "outputs")
    
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    dataset_path = os.path.join(uploads_dir, "titanic.csv")
    
    # 1. Download Titanic dataset if not present
    if not os.path.exists(dataset_path):
        print(f"Downloading Titanic dataset from {TITANIC_URL}...")
        response = requests.get(TITANIC_URL, timeout=15)
        response.raise_for_status()
        with open(dataset_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
        
    # 2. Run the complete multi-agent pipeline on Titanic with force_continue=True
    # to make sure we bypass the sensitive demographic column warnings (Age, Sex) and run ML
    goal = "Predict which passengers survived"
    result = await run_pipeline(dataset_path, goal, force_continue=True)
    
    # 3. Assertions
    # Check overall success status
    assert result["status"] == "success", f"Pipeline failed: {result.get('message')}"
    
    # Assert EDA completes successfully
    assert "eda_results" in result
    assert result["eda_results"].get("status") == "complete"
    assert "patterns_insights" in result["eda_results"].get("raw_results", {})
    
    # Assert ML model achieves at least 75% accuracy
    assert "ml_results" in result
    assert result["ml_results"].get("status") == "complete"
    accuracy = result["ml_results"].get("accuracy_score", 0.0)
    print(f"Model accuracy/metric score: {accuracy:.4f}")
    
    # If accuracy is between 0 and 1, check against 0.75, if between 0 and 100 check against 75
    if accuracy <= 1.0:
        assert accuracy >= 0.75, f"Model accuracy/metric score is too low: {accuracy}"
    else:
        assert accuracy >= 75.0, f"Model accuracy/metric score is too low: {accuracy}"
    
    # Assert Security review passes (runs and generates results)
    assert "security_results" in result
    assert "security_score" in result["security_results"]
    assert "issues_list" in result["security_results"]
    
    # Assert Report generates properly
    assert result.get("report_markdown") != ""
    assert result.get("report_html") != ""
    
    # Assert all expected output files exist
    assert os.path.exists(os.path.join(outputs_dir, "report.md"))
    assert os.path.exists(os.path.join(outputs_dir, "report.html"))
    assert os.path.exists(os.path.join(outputs_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(outputs_dir, "best_model.pkl"))
    assert os.path.exists(os.path.join(outputs_dir, "security_audit.txt"))
    assert os.path.exists(os.path.join(outputs_dir, "metrics.json"))
    
    # Check that Plotly charts were generated
    assert os.path.exists(os.path.join(outputs_dir, "eda_correlation_heatmap.html"))
    assert os.path.exists(os.path.join(outputs_dir, "ml_feature_importance.html"))
