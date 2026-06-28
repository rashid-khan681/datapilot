import os
import json
import time
import pytest
from monitoring.monitor import live_logger, pipeline_tracker, perf_metrics, LOG_FILE_PATH, METRICS_FILE_PATH

def test_live_logger():
    # Clear logs first
    live_logger.clear()
    assert len(live_logger.get_last_logs()) == 0
    assert not os.path.exists(LOG_FILE_PATH)

    # Log some messages
    live_logger.log("INFO", "EDA Agent", "Starting statistical analysis")
    live_logger.log("WARNING", "Security Agent", "Sensitive data detected")
    live_logger.log("SUCCESS", "ML Agent", "Model trained successfully")

    # Verify buffers
    logs = live_logger.get_last_logs()
    assert len(logs) == 3
    assert "INFO" in logs[0] and "EDA Agent" in logs[0] and "Starting statistical analysis" in logs[0]
    assert "WARNING" in logs[1] and "Sensitive data detected" in logs[1]
    assert "SUCCESS" in logs[2] and "Model trained successfully" in logs[2]

    # Verify file was written
    assert os.path.exists(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
        file_content = f.read()
    assert "Starting statistical analysis" in file_content

def test_pipeline_tracker():
    pipeline_tracker.reset(dataset_size_mb=4.2)
    assert pipeline_tracker.progress == 0.0
    assert pipeline_tracker.dataset_size_mb == 4.2
    assert len(pipeline_tracker.warnings) == 0

    # Simulate progress and warnings
    pipeline_tracker.current_agent = "DataPilot_EDA_Agent"
    pipeline_tracker.progress = 25.0
    pipeline_tracker.add_warning("DataPilot_EDA_Agent", "High cardinality columns found")

    assert len(pipeline_tracker.warnings) == 1
    assert pipeline_tracker.warnings[0] == "High cardinality columns found"

    # Get status dictionary and verify format
    status = pipeline_tracker.get_status_dict("EDA complete", "running")
    assert status["type"] == "agent_status"
    assert status["agent"] == "EDA Agent"  # Display agent name mapping
    assert status["progress"] == 25.0
    assert status["status"] == "running"
    assert "metrics" in status
    assert status["metrics"]["dataset_size_mb"] == 4.2
    assert status["metrics"]["warnings_count"] == 1
    assert status["metrics"]["memory_usage_mb"] >= 0.0

def test_performance_metrics():
    # Remove old metrics file if exists
    if os.path.exists(METRICS_FILE_PATH):
        os.remove(METRICS_FILE_PATH)

    perf_metrics.start_pipeline(dataset_size_mb=12.5)
    
    perf_metrics.start_stage("DataPilot_EDA_Agent")
    time.sleep(0.1)
    perf_metrics.end_stage("DataPilot_EDA_Agent")
    
    perf_metrics.start_stage("DataPilot_ML_Agent")
    time.sleep(0.1)
    perf_metrics.end_stage("DataPilot_ML_Agent")
    
    perf_metrics.save_summary()

    assert os.path.exists(METRICS_FILE_PATH)
    with open(METRICS_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_size_mb"] == 12.5
    assert data["total_elapsed_seconds"] > 0.0
    assert "DataPilot_EDA_Agent" in data["agent_durations"]
    assert "DataPilot_ML_Agent" in data["agent_durations"]
