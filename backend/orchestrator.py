"""
Orchestrator - Coordinates agent execution.

Responsibilities:
- Execute agents in sequence
- Update run status
- Handle errors
- Track execution with observability
"""
import traceback
import time
from database import RunDB, RunStatus
from agents import (
    run_data_auditor,
    run_feature_forensics,
    run_model_behavior,
    run_bias_aggregator
)
from observability import get_logger, get_tracer, get_metrics


def run_audit(run_id: str) -> None:
    """
    Run the complete audit pipeline with observability.
    
    Args:
        run_id: Unique run identifier
    """
    logger = get_logger("orchestrator", run_id)
    tracer = get_tracer(run_id)
    metrics = get_metrics(run_id)
    
    start_time = time.time()
    
    try:
        logger.info("Starting audit pipeline", run_id=run_id)
        
        # Update status to RUNNING
        RunDB.update_status(run_id, RunStatus.RUNNING)
        
        with tracer.span("audit_pipeline", run_id=run_id):
            # Execute agents in sequence with tracing
            with tracer.span("data_auditor"):
                logger.info("Running Data Auditor", agent="data_auditor")
                agent_start = time.time()
                run_data_auditor(run_id)
                metrics.record_execution_time("data_auditor", time.time() - agent_start)
                logger.info("Data Auditor complete", agent="data_auditor")
            
            with tracer.span("feature_forensics"):
                logger.info("Running Feature Forensics", agent="feature_forensics")
                agent_start = time.time()
                run_feature_forensics(run_id)
                metrics.record_execution_time("feature_forensics", time.time() - agent_start)
                logger.info("Feature Forensics complete", agent="feature_forensics")
            
            with tracer.span("model_behavior"):
                logger.info("Running Model Behavior", agent="model_behavior")
                agent_start = time.time()
                run_model_behavior(run_id)
                metrics.record_execution_time("model_behavior", time.time() - agent_start)
                logger.info("Model Behavior complete", agent="model_behavior")
            
            with tracer.span("bias_aggregator"):
                logger.info("Running Bias Aggregator", agent="bias_aggregator")
                agent_start = time.time()
                run_bias_aggregator(run_id)
                metrics.record_execution_time("bias_aggregator", time.time() - agent_start)
                logger.info("Bias Aggregator complete", agent="bias_aggregator")
        
        # Record total execution time
        total_time = time.time() - start_time
        metrics.record_total_time(total_time)
        
        # Mark as complete
        RunDB.update_status(run_id, RunStatus.COMPLETE)
        logger.info("Audit complete", run_id=run_id, total_time_seconds=total_time)
        
    except Exception as e:
        # Mark as failed with error message
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        RunDB.update_status(run_id, RunStatus.FAILED, error_message=error_msg)
        logger.error("Audit failed", run_id=run_id, error=error_msg)
        raise
