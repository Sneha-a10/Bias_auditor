"""Tools package for Bias Auditor."""
from .base import Tool, ToolInput, ToolOutput, ToolRegistry
from .bias_detector import (
    SubgroupAnalysisTool,
    ProxyDetectorTool,
    FairnessMetricTool,
    CounterfactualGeneratorTool
)

# Register all tools
SubgroupAnalysisTool()
ProxyDetectorTool()
FairnessMetricTool()
CounterfactualGeneratorTool()

__all__ = [
    'Tool',
    'ToolInput',
    'ToolOutput',
    'ToolRegistry',
    'SubgroupAnalysisTool',
    'ProxyDetectorTool',
    'FairnessMetricTool',
    'CounterfactualGeneratorTool'
]
