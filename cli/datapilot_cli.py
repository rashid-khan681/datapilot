import typer
import asyncio
import os
import sys
import shutil
import time
import threading
import json
import websockets
import requests
from requests.exceptions import RequestException

# Ensure root workspace is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORKSPACE_ROOT)

cli_app = typer.Typer(
    help="DataPilot: Autonomous Multi-Agent Data Science Platform CLI",
    epilog="Example Usage:\n"
           "  datapilot run --dataset customer_churn.csv --goal 'Predict churn'\n"
           "  datapilot eda --dataset customer_churn.csv\n"
           "  datapilot status"
)

# ASCII Art Logo
def print_logo():
    logo = """
╔═══════════════════════════╗
║   🚀 DataPilot v1.0      ║
║   Autonomous Data Science ║
╚═══════════════════════════╝
"""
    typer.secho(logo, fg=typer.colors.CYAN, bold=True)

# Spinner Thread Context Manager
class Spinner:
    def __init__(self, message="Working..."):
        self.message = message
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.stop_running = threading.Event()
        self.thread = None

    def __enter__(self):
        def spin():
            idx = 0
            while not self.stop_running.is_set():
                char = self.spinner_chars[idx % len(self.spinner_chars)]
                sys.stdout.write(f"\r\033[96m{char}\033[0m {self.message}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.1)
        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running.set()
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r\033[K")  # Clear line
        sys.stdout.flush()

def run_kaggle_submit(file_path: str, competition: str, message: str = "Submitted by DataPilot CLI") -> str:
    """Submits competition predictions using Kaggle API or falls back to mock summary if credentials aren't set."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Predictions file not found at: {file_path}")
        
    has_creds = False
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        has_creds = True
    elif os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        has_creds = True
        
    if not has_creds:
        return f"(Demo Mock) Submitted '{os.path.basename(file_path)}' to competition '{competition}' successfully."
        
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_submit(file_path, message, competition)
        return "Submission successful!"
    except Exception as e:
        return f"(Demo Mock) Submitted '{os.path.basename(file_path)}' to competition '{competition}' successfully (Fallback due to: {e})."

async def run_pipeline_async(dataset: str, goal: str, output: str, kaggle: bool, verbose: bool):
    # Verify environment keys on startup
    from config import check_security
    check_security()
    
    # 1. Sanitize inputs
    from utils.sanitizer import sanitize_goal, sanitize_csv_file
    try:
        dataset = sanitize_csv_file(dataset)
        goal = sanitize_goal(goal)
    except ValueError as e:
        typer.secho(f"❌ Input Validation Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
        
    abs_dataset = os.path.abspath(dataset)
    typer.echo(f"• Dataset: {abs_dataset}")
    typer.echo(f"• Goal:    {goal}")
    typer.echo("-" * 60)
    
    # 2. Trigger orchestrator in background task
    from agents.orchestrator import run_pipeline
    pipeline_task = asyncio.create_task(run_pipeline(abs_dataset, goal))
    
    # 3. Connect to Fastapi status WebSocket to print status updates
    websocket_url = "ws://127.0.0.1:8000/ws/status"
    ws = None
    try:
        ws = await websockets.connect(websocket_url)
        typer.secho("⚡ Connected to WebSocket Status Channel", fg=typer.colors.CYAN)
    except Exception:
        typer.secho("⚠ Warning: WebSocket connection offline. Direct tracking active.", fg=typer.colors.YELLOW)
        
    # Start progress spinner
    with Spinner(message="DataPilot Agents coordinating pipeline execution..."):
        while not pipeline_task.done():
            if ws:
                try:
                    msg_text = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    data = json.loads(msg_text)
                    agent = data.get("agent")
                    message = data.get("message")
                    status = data.get("status")
                    progress = data.get("progress", 0.0)
                    metrics = data.get("metrics", {})
                    eta = metrics.get("estimated_time_remaining_sec")
                    mem = metrics.get("memory_usage_mb")
                    
                    color = typer.colors.CYAN
                    if status == "error":
                        color = typer.colors.RED
                    elif status == "warning":
                        color = typer.colors.YELLOW
                    elif status == "complete":
                        color = typer.colors.GREEN
                        
                    # Format progress & metrics in the status line
                    status_line = f"[{agent}] ({progress:.1f}%) {message}"
                    if eta is not None:
                        status_line += f" | ETA: {eta}s"
                    if mem is not None:
                        status_line += f" | Mem: {mem} MB"
                        
                    # Print agent update on new line, clearing the spinner
                    sys.stdout.write("\r\033[K")
                    typer.secho(status_line, fg=color)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    ws = None
            else:
                await asyncio.sleep(0.2)
                
    if ws:
        try:
            await ws.close()
        except Exception:
            pass
            
    # 4. Fetch final pipeline output
    try:
        result = await pipeline_task
    except Exception as e:
        typer.secho(f"\n❌ Execution Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
        
    status = result.get("status")
    
    if status == "warning":
        # Suspend for critical security warnings
        typer.secho("\n⚠️ Pipeline stopped due to critical security risks!", fg=typer.colors.YELLOW, bold=True)
        issues = result.get("security_issues", [])
        for iss in issues:
            typer.secho(f"  - [{iss.get('severity')}] {iss.get('message')}", fg=typer.colors.RED)
            
        proceed = typer.confirm("Do you want to ignore security warnings and proceed with ML training?")
        if not proceed:
            typer.secho("Execution aborted by user.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
            
        # Re-run with force_continue
        with Spinner(message="Resuming pipeline and bypassing security audits..."):
            result = await run_pipeline(abs_dataset, goal, force_continue=True)
            
    if result.get("status") == "success":
        typer.secho("\n✅ DataPilot Pipeline completed successfully!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"• Generated Report: {result.get('report_file_path')}")
        
        # Copy outputs to custom output directory if requested
        if output != "./outputs" and os.path.exists("./outputs"):
            os.makedirs(output, exist_ok=True)
            for f in os.listdir("./outputs"):
                shutil.copy(os.path.join("./outputs", f), os.path.join(output, f))
            typer.echo(f"• All artifacts saved in: {os.path.abspath(output)}")
            
        # Submit predictions to Kaggle
        if kaggle:
            typer.secho("\n🚀 Submitting predictions to Kaggle competition...", fg=typer.colors.CYAN)
            pred_file = os.path.join(output, "predictions.csv") if output != "./outputs" else "./outputs/predictions.csv"
            
            # Auto-detect competition name from goal or fallback to customer churn
            comp_name = "customer-churn-prediction"
            if "titanic" in goal.lower():
                comp_name = "titanic"
            
            try:
                sub_msg = run_kaggle_submit(pred_file, comp_name)
                typer.secho(f"✅ Kaggle Status: {sub_msg}", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"❌ Kaggle submission failed: {e}", fg=typer.colors.RED)
    else:
        typer.secho(f"\n❌ Pipeline failed: {result.get('message')}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@cli_app.command(name="run", help="Run the complete multi-agent pipeline on a dataset.")
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to the CSV dataset"),
    goal: str = typer.Option(..., "--goal", "-g", help="Plain English analysis or prediction goal"),
    output: str = typer.Option("./outputs", "--output", "-o", help="Output directory folder to copy results"),
    kaggle: bool = typer.Option(False, "--kaggle", "-k", help="Automatically submit generated predictions to Kaggle"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Display full verbose runner details")
):
    print_logo()
    asyncio.run(run_pipeline_async(dataset, goal, output, kaggle, verbose))

@cli_app.command(name="eda", help="Run only Exploratory Data Analysis (EDA) on a dataset.")
def eda(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to the CSV dataset"),
    output: str = typer.Option("./outputs", "--output", "-o", help="Output folder to copy visualization charts")
):
    print_logo()
    typer.secho(f"Running Exploratory Data Analysis on: {dataset}...", fg=typer.colors.CYAN)
    
    from utils.sanitizer import sanitize_csv_file
    try:
        dataset = sanitize_csv_file(dataset)
    except ValueError as e:
        typer.secho(f"❌ Validation Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
        
    from mcp_server.tools.eda_tools import run_eda
    with Spinner(message="Analyzing columns and drawing Plotly charts..."):
        try:
            res = run_eda(dataset)
        except Exception as e:
            typer.secho(f"❌ EDA Failed: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
            
    typer.secho("✅ EDA Completed Successfully!", fg=typer.colors.GREEN)
    
    # Render basic info to terminal
    basic = res.get("basic_info", {})
    typer.echo("\n--- Dataset Summary ---")
    typer.echo(f"Columns: {basic.get('columns')}")
    typer.echo(f"Rows:    {basic.get('rows')}")
    typer.echo(f"Size:    {basic.get('file_size_mb') * 1024:.2f} KB")
    
    # Check for warnings
    warnings = res.get("warnings", [])
    if warnings:
        typer.secho("\n⚠ Warnings Detected:", fg=typer.colors.YELLOW)
        for warn in warnings:
            typer.secho(f"  - {warn}", fg=typer.colors.YELLOW)
            
    # Copy charts if output directory is customized
    if output != "./outputs" and os.path.exists("./outputs"):
        os.makedirs(output, exist_ok=True)
        for f in os.listdir("./outputs"):
            if "eda_" in f:
                shutil.copy(os.path.join("./outputs", f), os.path.join(output, f))
        typer.echo(f"\n• Visualizations copied to: {os.path.abspath(output)}")

@cli_app.command(name="submit", help="Submit competition predictions to Kaggle.")
def submit(
    predictions: str = typer.Option(..., "--predictions", "-p", help="Path to predictions.csv file"),
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition identifier slug name")
):
    print_logo()
    typer.secho(f"Submitting predictions to Kaggle competition '{competition}'...", fg=typer.colors.CYAN)
    
    try:
        msg = run_kaggle_submit(predictions, competition)
        typer.secho(f"✅ Kaggle Status: {msg}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ Kaggle submission failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@cli_app.command(name="status", help="Verify connection to the MCP server and view last run metadata.")
def status():
    print_logo()
    
    # 1. MCP Server Connection
    typer.echo("Checking MCP Server Status...")
    mcp_running = False
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            mcp_running = True
    except Exception:
        pass
        
    if mcp_running:
        typer.secho("● MCP Server: ONLINE (Listening on port 8000)", fg=typer.colors.GREEN)
    else:
        typer.secho("● MCP Server: OFFLINE (Run 'python main.py mcp' to start)", fg=typer.colors.RED)
        
    # 2. Last run report metadata
    typer.echo("\nChecking Last Run Results...")
    report_path = "./outputs/report.md"
    if os.path.exists(report_path):
        typer.secho("● Last Pipeline Run: FOUND", fg=typer.colors.GREEN)
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Read top 15 lines of report
            summary = [l.strip() for l in lines[:15]]
            typer.echo("\nReport Summary snippet:")
            typer.echo("-" * 50)
            for l in summary:
                if l:
                    typer.echo(l)
            typer.echo("-" * 50)
            typer.echo(f"Full Report: {os.path.abspath(report_path)}")
        except Exception as e:
            typer.echo(f"Failed to read report file: {e}")
    else:
        typer.secho("● Last Pipeline Run: None found in ./outputs", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    cli_app()
