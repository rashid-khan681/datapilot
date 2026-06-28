import asyncio
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

# Load env variables and propagate API key for Google GenAI SDK compatibility
load_dotenv()
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Suppress ADK/GenAI internal tracebacks and logging warnings to keep the demo clean
import logging

logging.disable(logging.WARNING)

from agents.orchestrator import run_pipeline

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def print_header(title):
    print(f"\n{BOLD}{MAGENTA}=== {title} ==={RESET}\n")

async def main():
    print_header("DataPilot Autonomous Multi-Agent Platform Demo")

    workspace_root = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(workspace_root, "uploads")
    outputs_dir = os.path.join(workspace_root, "outputs")

    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    dataset_path = os.path.join(uploads_dir, "titanic.csv")

    # 1. Download dataset
    if not os.path.exists(dataset_path):
        print(f"{CYAN}• Download Sample Titanic Dataset...{RESET}")
        try:
            r = requests.get(TITANIC_URL, timeout=15)
            r.raise_for_status()
            with open(dataset_path, "wb") as f:
                f.write(r.content)
            print(f"{GREEN}✓ Titanic dataset downloaded successfully.{RESET}")
        except Exception as e:
            print(f"{RED}✗ Failed to download dataset: {e}{RESET}")
            sys.exit(1)
    else:
        print(f"{GREEN}✓ Found local Titanic dataset at {dataset_path}{RESET}")

    # 2. Run Pipeline
    print_header("Running Multi-Agent Orchestration Pipeline")
    print(f"{CYAN}• Dataset size: {os.path.getsize(dataset_path) / 1024:.2f} KB")
    print(f"{CYAN}• Goal: Predict which passengers survived{RESET}")
    print(f"{CYAN}• Auto-retry & 30s timeouts active on all agents.{RESET}")
    print(f"{CYAN}• Running pipeline...{RESET}")

    start_time = time.time()

    # Run the pipeline with force_continue=True to complete fully
    # Redirect stderr to suppress GenAI/ADK task cleanup tracebacks from console
    import contextlib
    import io

    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        result = await run_pipeline(dataset_path, "Predict which passengers survived", force_continue=True)

    elapsed = time.time() - start_time

    # 3. Print Results
    print_header("Pipeline Completed Execution")

    if result.get("status") == "success":
        print(f"{GREEN}{BOLD}✓ Pipeline finished successfully in {elapsed:.2f} seconds!{RESET}")

        # ML Result
        ml = result.get("ml_results", {})
        print(f"\n{GREEN}{BOLD}[ML Agent Findings]{RESET}")
        print(f"  • Best Model: {ml.get('model_name')}")
        print(f"  • Accuracy/Metric Score: {ml.get('accuracy_score')}")
        print("  • Feature Importance Top 5:")
        for feat in ml.get("feature_importance_top5", []):
            print(f"    - {feat.get('name')}: {feat.get('importance')}")
        print(f"  • Business Insight: {ml.get('business_insight')}")

        # Security Result
        sec = result.get("security_results", {})
        print(f"\n{YELLOW}{BOLD}[Security Agent Findings]{RESET}")
        print(f"  • Security Score: {sec.get('security_score')}/100")
        print(f"  • Is Safe to Proceed: {sec.get('safe_to_proceed')}")
        print("  • Audited Issues:")
        for issue in sec.get("issues_list", []):
            print(f"    - [{issue.get('severity')}] {issue.get('message')} (Suggestion: {issue.get('suggestion')})")

        # Verify output files
        print_header("Verifying Output Files")
        expected_files = [
            "report.md",
            "report.html",
            "predictions.csv",
            "best_model.pkl",
            "security_audit.txt",
            "metrics.json",
            "eda_correlation_heatmap.html",
            "ml_feature_importance.html"
        ]

        for file in expected_files:
            file_path = os.path.join(outputs_dir, file)
            if os.path.exists(file_path):
                print(f"{GREEN}✓ Found output file: {file}{RESET}")
            else:
                print(f"{RED}✗ Missing output file: {file}{RESET}")

        # Print metrics summary
        metrics_file = os.path.join(outputs_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
            print_header("Performance Summary")
            print(f"{CYAN}• Dataset size processed: {metrics.get('dataset_size_mb')} MB")
            print(f"{CYAN}• Total elapsed time: {metrics.get('total_elapsed_seconds')} seconds")
            print(f"{CYAN}• Stage durations:")
            for stage, dur in metrics.get("agent_durations", {}).items():
                print(f"  - {stage}: {dur} seconds")
    else:
        print(f"{RED}{BOLD}✗ Pipeline execution failed: {result.get('message')}{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
