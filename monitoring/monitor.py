import os
import json
import time
import resource
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

# Setup path logic
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUTS_DIR, "pipeline.log")
METRICS_FILE_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")

# Define levels and ANSI color codes
ANSI_COLORS = {
    "INFO": "\033[94m",    # Blue
    "SUCCESS": "\033[92m", # Green
    "WARNING": "\033[93m", # Yellow
    "ERROR": "\033[91m",   # Red
    "RESET": "\033[0m"
}

class LiveLogger:
    def __init__(self, log_file: str = LOG_FILE_PATH, max_in_memory: int = 50):
        self.log_file = log_file
        self.max_in_memory = max_in_memory
        self.logs_buffer: List[str] = []

    def log(self, level: str, agent: str, message: str):
        level = level.upper()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plain_log = f"[{timestamp}] [{level}] [{agent}] {message}"
        
        # Save to buffer
        self.logs_buffer.append(plain_log)
        if len(self.logs_buffer) > self.max_in_memory:
            self.logs_buffer.pop(0)
            
        # Append to log file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(plain_log + "\n")
        except Exception as e:
            print(f"Error writing to pipeline log file: {e}", file=sys.stderr)

    def get_last_logs(self) -> List[str]:
        return self.logs_buffer

    def clear(self):
        self.logs_buffer.clear()
        if os.path.exists(self.log_file):
            try:
                os.remove(self.log_file)
            except Exception:
                pass

class PipelineTracker:
    def __init__(self, logger_instance: LiveLogger):
        self.logger = logger_instance
        self.current_agent: Optional[str] = None
        self.progress: float = 0.0  # 0 to 100
        self.start_time: Optional[float] = None
        self.warnings: List[str] = []
        self.dataset_size_mb: float = 0.0
        
        # Expected durations in seconds for ETA calculations
        self.expected_durations = {
            "DataPilot_EDA_Agent": 15.0,
            "DataPilot_ML_Agent": 25.0,
            "DataPilot_Security_Agent": 15.0,
            "DataPilot_Report_Agent": 10.0
        }

    def reset(self, dataset_size_mb: float = 0.0):
        self.current_agent = None
        self.progress = 0.0
        self.start_time = time.time()
        self.warnings.clear()
        self.dataset_size_mb = dataset_size_mb
        self.logger.clear()

    def add_warning(self, agent: str, warning_msg: str):
        self.warnings.append(warning_msg)
        self.logger.log("WARNING", agent, warning_msg)

    def get_memory_usage_mb(self) -> float:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # MacOS ru_maxrss is in bytes, Linux is in KB
            if sys.platform == "darwin":
                return round(usage / (1024 * 1024), 2)
            else:
                return round(usage / 1024, 2)
        except Exception:
            return 0.0

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return round(time.time() - self.start_time, 2)

    def get_estimated_time_remaining(self) -> float:
        """Calculates ETA based on expected remaining stages."""
        if self.start_time is None:
            return 0.0
        
        # Map agent names to index/order
        agent_order = [
            "DataPilot_EDA_Agent",
            "DataPilot_ML_Agent",
            "DataPilot_Security_Agent",
            "DataPilot_Report_Agent"
        ]
        
        if not self.current_agent or self.current_agent not in agent_order:
            return sum(self.expected_durations.values())
            
        current_idx = agent_order.index(self.current_agent)
        
        # Estimate remaining duration of current agent
        # Overall progress ranges from 0 to 100. Let's estimate stage progress:
        # 0-25%: EDA, 25-50%: ML, 50-75%: Security, 75-100%: Report
        stage_progress = (self.progress % 25.0) / 25.0
        current_agent_expected = self.expected_durations.get(self.current_agent, 10.0)
        remaining_current = (1.0 - stage_progress) * current_agent_expected
        
        # Add remaining stages completely
        remaining_stages_sum = sum(self.expected_durations[a] for a in agent_order[current_idx + 1:])
        
        remaining_total = remaining_current + remaining_stages_sum
        return round(remaining_total, 2)

    def get_status_dict(self, message: str, status: str = "running") -> dict:
        # Standardize agent names for UI/CLI badges if necessary, but keep original for internal
        display_agent = self.current_agent
        if display_agent == "DataPilot_EDA_Agent":
            display_agent = "EDA Agent"
        elif display_agent == "DataPilot_ML_Agent":
            display_agent = "ML Agent"
        elif display_agent == "DataPilot_Security_Agent":
            display_agent = "Security Agent"
        elif display_agent == "DataPilot_Report_Agent":
            display_agent = "Report Agent"

        return {
            "type": "agent_status",
            "agent": display_agent or "System",
            "status": status,
            "message": message,
            "progress": round(self.progress, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "elapsed_time_sec": self.get_elapsed_time(),
                "estimated_time_remaining_sec": self.get_estimated_time_remaining(),
                "memory_usage_mb": self.get_memory_usage_mb(),
                "warnings_count": len(self.warnings),
                "dataset_size_mb": self.dataset_size_mb
            }
        }

class PerformanceMetrics:
    def __init__(self, metrics_file: str = METRICS_FILE_PATH):
        self.metrics_file = metrics_file
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
        self.total_start_time: Optional[float] = None
        self.dataset_size_mb: float = 0.0

    def start_pipeline(self, dataset_size_mb: float):
        self.total_start_time = time.time()
        self.dataset_size_mb = dataset_size_mb
        self.durations.clear()
        self.start_times.clear()

    def start_stage(self, stage_name: str):
        self.start_times[stage_name] = time.time()

    def end_stage(self, stage_name: str):
        if stage_name in self.start_times:
            self.durations[stage_name] = round(time.time() - self.start_times[stage_name], 2)

    def save_summary(self):
        if self.total_start_time is None:
            return
        total_duration = round(time.time() - self.total_start_time, 2)
        summary = {
            "dataset_size_mb": self.dataset_size_mb,
            "total_elapsed_seconds": total_duration,
            "agent_durations": self.durations,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Error writing performance metrics: {e}", file=sys.stderr)

# Global instances
live_logger = LiveLogger()
pipeline_tracker = PipelineTracker(live_logger)
perf_metrics = PerformanceMetrics()

class WebSocketManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message):
        import json
        if isinstance(message, dict):
            message_str = json.dumps(message)
        else:
            message_str = str(message)
        
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message_str)
            except Exception:
                self.disconnect(connection)

ws_manager = WebSocketManager()

