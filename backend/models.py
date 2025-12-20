"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SensitiveAttribute(BaseModel):
    """Configuration for a sensitive attribute."""
    name: str
    expected_categories: Optional[List[str]] = None


class FairnessThresholds(BaseModel):
    """Fairness metric thresholds."""
    demographic_parity_diff_threshold: float = 0.1
    equal_opportunity_diff_threshold: float = 0.1
    equalized_odds_diff_threshold: float = 0.1
    min_group_proportion_threshold: float = 0.05
    min_support_for_metrics: int = 30
    proxy_corr_threshold: float = 0.3
    counterfactual_change_threshold: float = 0.1
    label_imbalance_ratio_threshold: float = 4.0


class RunConfig(BaseModel):
    """Configuration for an audit run."""
    target_column: str
    sensitive_attributes: List[SensitiveAttribute]
    grouping_attributes: Optional[List[str]] = []
    fairness_thresholds: FairnessThresholds = Field(default_factory=FairnessThresholds)


class RunResponse(BaseModel):
    """Response for a run."""
    id: str
    status: str
    config: RunConfig
    created_at: str
    updated_at: str
    error_message: Optional[str] = None


class RunListItem(BaseModel):
    """Summary item for run list."""
    id: str
    status: str
    created_at: str
    primary_origin: Optional[str] = None


class CreateRunRequest(BaseModel):
    """Request to create a new run (config only, files via multipart)."""
    config: RunConfig
