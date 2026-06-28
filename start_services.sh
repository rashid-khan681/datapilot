#!/bin/bash
# Start DataPilot MCP Fastapi Server in the background on port 8000
python main.py mcp &

# Give the server a few seconds to boot up
sleep 3

# Start the Gradio Web UI on port 7860
PORT=7860 python main.py ui
