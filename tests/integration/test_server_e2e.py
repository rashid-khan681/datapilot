import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}


def log_output(pipe: Any, log_func: Any) -> None:
    """Log the output from the given pipe."""
    for line in iter(pipe.readline, ""):
        log_func(line.strip())


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "mcp_server.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env["INTEGRATION_TEST"] = "TRUE"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    # Start threads to log stdout and stderr in real-time
    threading.Thread(
        target=log_output, args=(process.stdout, logger.info), daemon=True
    ).start()
    threading.Thread(
        target=log_output, args=(process.stderr, logger.error), daemon=True
    ).start()

    return process


def wait_for_server(timeout: int = 30, interval: int = 1) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f"Server did not become ready within {timeout} seconds")
    return False


@pytest.fixture(scope="session")
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:
    """Pytest fixture to start and stop the server for testing."""
    logger.info("Starting server process")

    # Check if a server is already running on port 8000
    already_running = False
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            logger.info("FastAPI server is already running on port 8000, skipping spawning subprocess.")
            already_running = True
    except RequestException:
        pass

    if already_running:
        yield None
        return

    server_process = start_server()
    if not wait_for_server():
        pytest.fail("Server failed to start")
    logger.info("Server process started")

    def stop_server() -> None:
        logger.info("Stopping server process")
        server_process.terminate()
        server_process.wait()
        logger.info("Server process stopped")

    request.addfinalizer(stop_server)
    yield server_process


def test_health_endpoint(server_fixture: Any) -> None:
    """Test the /health endpoint of the FastAPI MCP Server."""
    logger.info("Starting health endpoint test")
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["status"] == "healthy"


def test_status_broadcast_endpoint(server_fixture: Any) -> None:
    """Test the /status/broadcast endpoint."""
    logger.info("Starting status broadcast endpoint test")
    payload = {
        "agent": "TestAgent",
        "message": "Testing integration",
        "status": "running"
    }
    response = requests.post(f"{BASE_URL}/status/broadcast", json=payload, headers=HEADERS, timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_report_endpoint(server_fixture: Any) -> None:
    """Test the /tools/report endpoint."""
    logger.info("Starting report endpoint test")
    payload = {
        "content": "# Test Report\nThis is a test report content."
    }
    response = requests.post(f"{BASE_URL}/tools/report", json=payload, headers=HEADERS, timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
