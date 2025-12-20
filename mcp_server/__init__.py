"""MCP Server package for Bias Auditor."""
from .server import run_server
from .resources import (
    get_bias_pattern,
    get_fairness_metric,
    get_mitigation_strategy,
    get_all_bias_patterns,
    get_all_fairness_metrics,
    get_all_mitigation_strategies
)

__all__ = [
    'run_server',
    'get_bias_pattern',
    'get_fairness_metric',
    'get_mitigation_strategy',
    'get_all_bias_patterns',
    'get_all_fairness_metrics',
    'get_all_mitigation_strategies'
]
