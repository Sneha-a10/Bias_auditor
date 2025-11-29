"""
Observability module for Bias Auditor.

Provides:
- Structured logging with context
- Execution tracing for agent flows
- Metrics collection (timing, scores, performance)
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


class Logger:
    """Structured logger with context management."""
    
    def __init__(self, name: str, run_id: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name (e.g., agent name)
            run_id: Optional run ID for context
        """
        self.logger = structlog.get_logger(name)
        self.name = name
        self.run_id = run_id
        
        # Set up file logging
        self.log_dir = Path(__file__).parent.parent / "data" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id:
            self.log_file = self.log_dir / f"{run_id}.jsonl"
        else:
            self.log_file = self.log_dir / "system.jsonl"
    
    def _write_log(self, level: str, message: str, **kwargs):
        """Write log entry to file and stdout."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        if self.run_id:
            log_entry["run_id"] = self.run_id
        
        # Write to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Also log to structlog
        getattr(self.logger, level.lower())(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._write_log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._write_log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._write_log("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._write_log("DEBUG", message, **kwargs)


class Tracer:
    """Execution tracer for tracking agent flows."""
    
    def __init__(self, run_id: str):
        """
        Initialize tracer.
        
        Args:
            run_id: Run identifier
        """
        self.run_id = run_id
        self.traces: List[Dict[str, Any]] = []
        self.current_span: Optional[Dict[str, Any]] = None
        self.trace_dir = Path(__file__).parent.parent / "data" / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.trace_dir / f"{run_id}.json"
    
    @contextmanager
    def span(self, name: str, **attributes):
        """
        Create a trace span.
        
        Args:
            name: Span name (e.g., "data_auditor", "feature_forensics")
            **attributes: Additional span attributes
        """
        span_data = {
            "name": name,
            "start_time": time.time(),
            "start_iso": datetime.utcnow().isoformat(),
            "attributes": attributes,
            "run_id": self.run_id
        }
        
        parent_span = self.current_span
        if parent_span:
            span_data["parent"] = parent_span["name"]
        
        self.current_span = span_data
        
        try:
            yield span_data
        except Exception as e:
            span_data["error"] = str(e)
            span_data["error_type"] = type(e).__name__
            raise
        finally:
            span_data["end_time"] = time.time()
            span_data["end_iso"] = datetime.utcnow().isoformat()
            span_data["duration_seconds"] = span_data["end_time"] - span_data["start_time"]
            
            self.traces.append(span_data)
            self.current_span = parent_span
            
            # Write trace to file
            self._save_traces()
    
    def _save_traces(self):
        """Save traces to file."""
        with open(self.trace_file, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "traces": self.traces
            }, f, indent=2)
    
    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all traces."""
        return self.traces


class MetricsCollector:
    """Metrics collector for performance and bias metrics."""
    
    def __init__(self, run_id: str):
        """
        Initialize metrics collector.
        
        Args:
            run_id: Run identifier
        """
        self.run_id = run_id
        self.metrics: Dict[str, Any] = {
            "run_id": run_id,
            "execution_metrics": {},
            "bias_metrics": {},
            "agent_metrics": {}
        }
        self.metrics_dir = Path(__file__).parent.parent / "data" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{run_id}.json"
    
    def record_execution_time(self, agent_name: str, duration_seconds: float):
        """Record agent execution time."""
        if "execution_times" not in self.metrics["execution_metrics"]:
            self.metrics["execution_metrics"]["execution_times"] = {}
        
        self.metrics["execution_metrics"]["execution_times"][agent_name] = duration_seconds
        self._save_metrics()
    
    def record_bias_score(self, checkpoint: str, score: float):
        """Record bias score for a checkpoint."""
        if "scores" not in self.metrics["bias_metrics"]:
            self.metrics["bias_metrics"]["scores"] = {}
        
        self.metrics["bias_metrics"]["scores"][checkpoint] = score
        self._save_metrics()
    
    def record_agent_metric(self, agent_name: str, metric_name: str, value: Any):
        """Record custom agent metric."""
        if agent_name not in self.metrics["agent_metrics"]:
            self.metrics["agent_metrics"][agent_name] = {}
        
        self.metrics["agent_metrics"][agent_name][metric_name] = value
        self._save_metrics()
    
    def record_total_time(self, duration_seconds: float):
        """Record total audit execution time."""
        self.metrics["execution_metrics"]["total_duration_seconds"] = duration_seconds
        self._save_metrics()
    
    def record_primary_origin(self, origin: str):
        """Record primary bias origin verdict."""
        self.metrics["bias_metrics"]["primary_origin"] = origin
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics


# Global instances (will be initialized per run)
_loggers: Dict[str, Logger] = {}
_tracers: Dict[str, Tracer] = {}
_metrics: Dict[str, MetricsCollector] = {}


def get_logger(name: str, run_id: Optional[str] = None) -> Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        run_id: Optional run ID
    
    Returns:
        Logger instance
    """
    key = f"{name}:{run_id}" if run_id else name
    if key not in _loggers:
        _loggers[key] = Logger(name, run_id)
    return _loggers[key]


def get_tracer(run_id: str) -> Tracer:
    """
    Get or create a tracer for a run.
    
    Args:
        run_id: Run identifier
    
    Returns:
        Tracer instance
    """
    if run_id not in _tracers:
        _tracers[run_id] = Tracer(run_id)
    return _tracers[run_id]


def get_metrics(run_id: str) -> MetricsCollector:
    """
    Get or create a metrics collector for a run.
    
    Args:
        run_id: Run identifier
    
    Returns:
        MetricsCollector instance
    """
    if run_id not in _metrics:
        _metrics[run_id] = MetricsCollector(run_id)
    return _metrics[run_id]
