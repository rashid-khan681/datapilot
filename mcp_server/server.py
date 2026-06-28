import datetime
import logging
import os
import secrets
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure root workspace is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tools import (
    execute_python_code,
    read_dataset_info,
    review_code,
    run_eda,
    save_report,
    scan_code_security,
    train_model,
)
from monitoring.monitor import ws_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datapilot-mcp")

# Internal broadcast token — agents use this to authenticate status broadcasts
BROADCAST_TOKEN = os.environ.get("DATAPILOT_BROADCAST_TOKEN", secrets.token_hex(16))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    print(f"{BLUE}" + "="*60 + f"{RESET}")
    print(f"{GREEN}🚀 DataPilot MCP Server Starting on Port 8000...{RESET}")
    print("Available Tools:")
    print("  • [POST] /tools/eda     -> Exploratory Data Analysis tool")
    print("  • [POST] /tools/train   -> Python Model Training sandbox")
    print("  • [POST] /tools/review  -> Security Static Code scanner")
    print("  • [POST] /tools/report  -> Save Markdown Report tool")
    print("WebSocket Endpoint:")
    print("  • [WS]   /ws/status     -> Live agents progress status channel")
    print(f"{BLUE}" + "="*60 + f"{RESET}")
    yield
    # Shutdown
    rate_limit_records.clear()


app = FastAPI(title="DataPilot MCP Server", lifespan=lifespan)

# CORS: restrict to localhost origins only (Fix: was wildcard "*" with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiting dictionary (IP -> list of request timestamps)
rate_limit_records: dict[str, list[float]] = defaultdict(list)
_last_rate_limit_cleanup = time.time()

def rate_limiter(request: Request):
    global _last_rate_limit_cleanup
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Periodic cleanup of stale IPs (every 5 minutes) to prevent memory leak
    if now - _last_rate_limit_cleanup > 300:
        stale_ips = [ip for ip, ts_list in rate_limit_records.items()
                     if not ts_list or (now - max(ts_list)) > 120]
        for ip in stale_ips:
            del rate_limit_records[ip]
        _last_rate_limit_cleanup = now

    # Filter timestamps within the last 60 seconds
    rate_limit_records[client_ip] = [t for t in rate_limit_records[client_ip] if now - t < 60]

    if len(rate_limit_records[client_ip]) >= 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 100 requests per minute."
        )

    rate_limit_records[client_ip].append(now)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    if path.startswith("/tools"):
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        logger.info(f"[{timestamp}] Tool execution request: {request.method} {path} from {request.client.host if request.client else 'unknown'}")
    response = await call_next(request)
    return response

# Standard JSON response format helper
def make_response(status: str, data: dict = None, message: str = "") -> dict:
    return {
        "status": status,
        "data": data or {},
        "message": message,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }

# Exception handlers for clean JSON errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=make_response("error", message=exc.detail)
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc!s}", exc_info=True)
    # Do NOT leak internal error details to clients
    return JSONResponse(
        status_code=500,
        content=make_response("error", message="Internal Server Error. Check server logs for details.")
    )

# Pydantic Schemas for input validation
class EDARequest(BaseModel):
    file_path: str

class TrainRequest(BaseModel):
    file_path: str
    code: str | None = None
    target: str | None = None
    test_file_path: str | None = None
    goal: str | None = None
    force_continue: bool = False

class ReviewRequest(BaseModel):
    file_path: str | None = None
    target: str | None = None
    code: str | None = None

class ReportRequest(BaseModel):
    content: str

# Health check
@app.get("/health")
def health_check():
    return make_response("success", {"status": "healthy"}, "DataPilot MCP Server is healthy")

@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Read-only: WS clients can only receive status updates, not broadcast
            # This prevents external clients from injecting fake status messages
            await websocket.receive_text()  # keep connection alive by consuming pings
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.post("/status/broadcast")
async def rest_broadcast_status(request: Request, payload: Dict[str, Any]):
    """Internal endpoint for agents to broadcast status updates. Protected by token."""
    # Validate broadcast token from internal agents (localhost requests are also accepted)
    client_host = request.client.host if request.client else ""
    is_localhost = client_host in ("127.0.0.1", "::1", "localhost")
    token = request.headers.get("X-Broadcast-Token", "")
    
    if not is_localhost and token != BROADCAST_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized broadcast request.")
    
    await ws_manager.broadcast(payload)
    return make_response("success", message="Status broadcasted successfully")

# Tool Endpoint Groups
@app.post("/tools/eda", dependencies=[Depends(rate_limiter)])
def run_eda_endpoint(payload: EDARequest):
    try:
        from utils.sanitizer import validate_safe_path
        safe_path = validate_safe_path(payload.file_path)
        result = run_eda(safe_path)
        # Add markdown report to result dictionary for backwards compatibility
        result["report"] = read_dataset_info(safe_path)
        return make_response("success", result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tools/train", dependencies=[Depends(rate_limiter)])
def run_train(payload: TrainRequest):
    try:
        from utils.sanitizer import validate_safe_path
        safe_path = validate_safe_path(payload.file_path)
        safe_test_path = validate_safe_path(payload.test_file_path) if payload.test_file_path else None
        
        if payload.code is not None:
            result = execute_python_code(payload.code, safe_path)
            if result.get("exit_code") != 0:
                return make_response("error", result, f"Training code execution failed with exit code {result.get('exit_code')}")
            return make_response("success", result)

        result = train_model(safe_path, payload.target, safe_test_path, goal=payload.goal, force_continue=payload.force_continue)
        return make_response("success", result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tools/review", dependencies=[Depends(rate_limiter)])
def run_review(payload: ReviewRequest):
    try:
        from utils.sanitizer import validate_safe_path
        if payload.file_path is not None:
            safe_path = validate_safe_path(payload.file_path)
            result = review_code(safe_path, payload.target, payload.code)
            return make_response("success", result)

        result = scan_code_security(payload.code or "")
        return make_response("success", result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tools/report", dependencies=[Depends(rate_limiter)])
def run_report(payload: ReportRequest):
    result = save_report(payload.content)
    if result.startswith("Error"):
        raise HTTPException(status_code=400, detail=result)
    return make_response("success", {"message": result})

if __name__ == "__main__":
    import uvicorn
    # Restrict binding to 127.0.0.1 for local security unless running in Docker
    host = "0.0.0.0" if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_RUN") else "127.0.0.1"
    uvicorn.run("mcp_server.server:app", host=host, port=8000, reload=True)
