import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ensure the root folder is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Detect Docker container environment or environment override flags
    is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_RUN") or os.environ.get("PORT") or os.environ.get("K_SERVICE")
    default_host = "0.0.0.0" if is_docker else "127.0.0.1"


    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "mcp":
            # Run MCP Server
            print(f"Starting DataPilot MCP Server on {default_host}...")
            import uvicorn
            uvicorn.run("mcp_server.server:app", host=default_host, port=8000)
        elif command == "cli":
            # Run Typer CLI
            from cli.datapilot_cli import cli_app
            sys.argv = [sys.argv[0], *sys.argv[2:]]
            cli_app()
        elif command == "ui":
            # Run UI
            ui_port = int(os.environ.get("PORT", "8080"))
            print(f"Starting DataPilot Gradio Web UI on {default_host} (port {ui_port})...")
            from ui.app import WORKSPACE_ROOT, launch_ui
            demo = launch_ui()
            demo.queue().launch(server_name=default_host, server_port=ui_port, allowed_paths=[WORKSPACE_ROOT])
        else:
            print("Unknown command. Available commands: mcp, cli, ui")
            print("Usage:")
            print("  python main.py ui                 # Starts the Gradio Web UI (Default)")
            print("  python main.py mcp                # Starts the FastMCP Server")
            print("  python main.py cli --file F --goal G  # Runs the pipeline in CLI")
    else:
        # Default to launching Web UI
        ui_port = int(os.environ.get("PORT", "8080"))
        print(f"Starting DataPilot Gradio Web UI on {default_host} (port {ui_port})...")
        from ui.app import WORKSPACE_ROOT, launch_ui
        demo = launch_ui()
        demo.queue().launch(server_name=default_host, server_port=ui_port, allowed_paths=[WORKSPACE_ROOT])


if __name__ == "__main__":
    main()
