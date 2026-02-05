"""
from __future__ import annotations

Logging and Monitoring Module

Provides structured logging, prediction monitoring, latency tracking,
and error logging for the LLM pipeline.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Union

import structlog
from structlog.processors import JSONRenderer

from src.config import settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.

    Sets up both console and file handlers with JSON formatting.
    """
    # Ensure log directory exists
    log_dir = settings.logging.log_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                JSONRenderer()
                if settings.logging.format == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        settings.logging.log_file,
        maxBytes=settings.logging.max_file_size,
        backupCount=settings.logging.backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


class PredictionMonitor:
    """
    Monitor for tracking model predictions, latency, and errors.

    Logs all prediction events to a structured log file for analysis.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        enabled: bool = True,
    ):
        """
        Initialize the prediction monitor.

        Args:
            log_file: Path to the prediction log file.
            enabled: Whether monitoring is enabled.
        """
        self.log_file = log_file or settings.logging.log_file
        self.enabled = enabled
        self.logger = structlog.get_logger("prediction_monitor")

        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self._prediction_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._start_time = datetime.now()

    def log_prediction(
        self,
        input_text: str,
        output: Any,
        model_version: str,
        latency_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a prediction event.

        Args:
            input_text: Input text (truncated for privacy).
            output: Model output.
            model_version: Version of the model used.
            latency_ms: Inference latency in milliseconds.
            metadata: Additional metadata to log.
        """
        if not self.enabled or not settings.logging.log_predictions:
            return

        self._prediction_count += 1
        self._total_latency_ms += latency_ms

        # Truncate input for logging
        truncated_input = (
            input_text[:100] + "..." if len(input_text) > 100 else input_text
        )

        log_entry = {
            "event": "prediction",
            "timestamp": datetime.now().isoformat(),
            "input_preview": truncated_input,
            "input_length": len(input_text),
            "output_type": type(output).__name__,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 2),
        }

        if isinstance(output, dict):
            # Add classification results if available
            if "predicted_class" in output:
                log_entry["predicted_class"] = output["predicted_class"]
                log_entry["confidence"] = output.get("confidence")

        if metadata:
            log_entry["metadata"] = metadata

        self._write_log(log_entry)

        if settings.logging.log_latency:
            self.logger.info(
                "prediction_logged",
                latency_ms=round(latency_ms, 2),
                model_version=model_version,
            )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        input_text: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log an error event.

        Args:
            error_type: Type/class of the error.
            error_message: Error message.
            input_text: Input that caused the error.
            stack_trace: Full stack trace.
            metadata: Additional metadata.
        """
        if not self.enabled or not settings.logging.log_errors:
            return

        self._error_count += 1

        log_entry = {
            "event": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
        }

        if input_text:
            log_entry["input_preview"] = (
                input_text[:100] + "..." if len(input_text) > 100 else input_text
            )

        if stack_trace:
            log_entry["stack_trace"] = stack_trace

        if metadata:
            log_entry["metadata"] = metadata

        self._write_log(log_entry)

        self.logger.error(
            "error_logged",
            error_type=error_type,
            error_message=error_message,
        )

    def log_model_load(
        self,
        model_version: str,
        model_path: str,
        load_time_ms: float,
        device: str,
    ) -> None:
        """
        Log a model loading event.

        Args:
            model_version: Version of the model.
            model_path: Path to the model.
            load_time_ms: Time to load the model.
            device: Device the model is loaded on.
        """
        if not self.enabled:
            return

        log_entry = {
            "event": "model_load",
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "model_path": model_path,
            "load_time_ms": round(load_time_ms, 2),
            "device": device,
        }

        self._write_log(log_entry)

        self.logger.info(
            "model_loaded",
            model_version=model_version,
            device=device,
            load_time_ms=round(load_time_ms, 2),
        )

    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        client_ip: Optional[str] = None,
    ) -> None:
        """
        Log an API request.

        Args:
            endpoint: API endpoint path.
            method: HTTP method.
            status_code: Response status code.
            latency_ms: Request latency.
            client_ip: Client IP address.
        """
        if not self.enabled:
            return

        log_entry = {
            "event": "api_request",
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        }

        if client_ip:
            log_entry["client_ip"] = client_ip

        self._write_log(log_entry)

    def _write_log(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Fallback to standard logging if file write fails
            logging.error(f"Failed to write to log file: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current monitoring metrics.

        Returns:
            Dictionary containing monitoring metrics.
        """
        uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        avg_latency = (
            self._total_latency_ms / self._prediction_count
            if self._prediction_count > 0
            else 0
        )

        return {
            "predictions_total": self._prediction_count,
            "errors_total": self._error_count,
            "average_latency_ms": round(avg_latency, 2),
            "uptime_seconds": round(uptime_seconds, 2),
            "predictions_per_second": (
                round(self._prediction_count / uptime_seconds, 4)
                if uptime_seconds > 0
                else 0
            ),
            "error_rate": (
                round(self._error_count / self._prediction_count, 4)
                if self._prediction_count > 0
                else 0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset all monitoring metrics."""
        self._prediction_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._start_time = datetime.now()

        self.logger.info("metrics_reset")


@contextmanager
def track_latency(operation_name: str, monitor: Optional[PredictionMonitor] = None):
    """
    Context manager for tracking operation latency.

    Args:
        operation_name: Name of the operation being tracked.
        monitor: Optional PredictionMonitor instance.

    Yields:
        Dictionary that will contain the latency after completion.
    """
    start_time = time.time()
    result = {"latency_ms": 0.0}

    try:
        yield result
    finally:
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = latency_ms

        if monitor:
            monitor.logger.debug(
                "operation_complete",
                operation=operation_name,
                latency_ms=round(latency_ms, 2),
            )


class HealthChecker:
    """Health checker for system components."""

    def __init__(self):
        """Initialize health checker."""
        self.logger = structlog.get_logger("health_checker")

    def check_model_health(self, inference_engine) -> dict[str, Any]:
        """
        Check if model is loaded and responsive.

        Args:
            inference_engine: LLMInference instance.

        Returns:
            Health check result.
        """
        try:
            # Try to get model info
            info = inference_engine.get_model_info()
            return {
                "status": "healthy",
                "model_version": info.get("model_version", "unknown"),
                "device": info.get("device", "unknown"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def check_disk_health(self) -> dict[str, Any]:
        """
        Check disk space availability.

        Returns:
            Health check result.
        """
        import shutil

        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            usage_percent = (used / total) * 100

            status = "healthy" if free_gb > 1 else "warning"

            return {
                "status": status,
                "free_gb": free_gb,
                "usage_percent": round(usage_percent, 2),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def check_memory_health(self) -> dict[str, Any]:
        """
        Check memory availability.

        Returns:
            Health check result.
        """
        try:
            import psutil

            memory = psutil.virtual_memory()

            status = "healthy" if memory.percent < 90 else "warning"

            return {
                "status": status,
                "available_gb": round(memory.available / (2**30), 2),
                "usage_percent": memory.percent,
            }
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil not installed",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def full_health_check(self, inference_engine=None) -> dict[str, Any]:
        """
        Run full system health check.

        Args:
            inference_engine: Optional LLMInference instance.

        Returns:
            Complete health check results.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "disk": self.check_disk_health(),
            "memory": self.check_memory_health(),
        }

        if inference_engine:
            results["model"] = self.check_model_health(inference_engine)

        # Determine overall status
        statuses = [v.get("status") for v in results.values() if isinstance(v, dict)]

        if "unhealthy" in statuses or "error" in statuses:
            results["overall_status"] = "unhealthy"
        elif "warning" in statuses:
            results["overall_status"] = "degraded"
        else:
            results["overall_status"] = "healthy"

        return results


# Initialize monitoring on module import
setup_logging()
