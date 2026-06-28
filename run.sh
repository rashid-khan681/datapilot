#!/bin/bash

# Clean termination of all background processes spawned by this script on exit
trap "kill 0" EXIT

echo "=================================================="
echo "🚀 Starting DataPilot Platform Services..."
echo "=================================================="

# 1. Start MCP Server in the background
uv run python main.py mcp &
MCP_PID=$!

# 2. Wait for server to initialize
echo "Waiting 3 seconds for MCP server to be ready..."
sleep 3

# 3. Check health status
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ DataPilot MCP Server is healthy and online!"
else
    echo "⚠️ Warning: MCP Server health check did not respond. Attempting to proceed..."
fi

# 4. Open browser to Gradio UI after a short delay
(sleep 2 && open http://localhost:8080) &

echo "--------------------------------------------------"
echo "🎉 DataPilot is ready! Opening http://localhost:8080..."
echo "--------------------------------------------------"

# 5. Launch Gradio UI in the foreground
uv run python main.py ui
