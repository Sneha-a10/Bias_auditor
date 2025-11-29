"""
Bias Pattern Resources for MCP Server.

Provides knowledge base of:
- Common bias patterns
- Fairness metric definitions
- Mitigation strategies
"""
from typing import Dict, List, Any


# Common bias patterns and their descriptions
BIAS_PATTERNS = {
    "underrepresented_groups": {
        "name": "Underrepresented Groups",
        "description": "Certain demographic groups are severely underrepresented in the training data (typically <5%).",
        "stage": "DATA",
        "severity": "HIGH",
        "indicators": [
            "Group proportion < 5%",
            "Imbalanced demographic distribution",
            "Missing demographic categories"
        ],
        "consequences": [
            "Poor model performance on minority groups",
            "Amplified bias in predictions",
            "Unfair outcomes for underrepresented populations"
        ]
    },
    "label_imbalance": {
        "name": "Label Imbalance",
        "description": "Severe imbalance in the target variable distribution, often correlated with sensitive attributes.",
        "stage": "DATA",
        "severity": "MEDIUM",
        "indicators": [
            "Label ratio > 4:1",
            "Correlated with demographic attributes",
            "Unequal positive rates across groups"
        ],
        "consequences": [
            "Model bias towards majority class",
            "Disparate impact on minority groups",
            "Reduced fairness metrics"
        ]
    },
    "proxy_features": {
        "name": "Proxy Features",
        "description": "Features that are highly correlated with sensitive attributes and can encode bias.",
        "stage": "FEATURE",
        "severity": "HIGH",
        "indicators": [
            "High correlation with sensitive attributes (>0.3)",
            "Feature names suggesting demographic encoding",
            "One-hot encoded demographic categories"
        ],
        "consequences": [
            "Indirect discrimination",
            "Circumvention of fairness constraints",
            "Legal compliance issues"
        ]
    },
    "demographic_parity_violation": {
        "name": "Demographic Parity Violation",
        "description": "Significant differences in acceptance/positive prediction rates across demographic groups.",
        "stage": "MODEL",
        "severity": "HIGH",
        "indicators": [
            "Acceptance rate gap > 10%",
            "Disparate impact ratio < 0.8",
            "Unequal selection rates"
        ],
        "consequences": [
            "Discriminatory outcomes",
            "Legal liability",
            "Unfair resource allocation"
        ]
    },
    "equal_opportunity_violation": {
        "name": "Equal Opportunity Violation",
        "description": "Differences in true positive rates (recall) across groups for qualified individuals.",
        "stage": "MODEL",
        "severity": "HIGH",
        "indicators": [
            "TPR gap > 10%",
            "Unequal benefit for qualified individuals",
            "Disparate recall across groups"
        ],
        "consequences": [
            "Missed opportunities for minority groups",
            "Perpetuation of inequality",
            "Reduced trust in the system"
        ]
    },
    "counterfactual_sensitivity": {
        "name": "Counterfactual Sensitivity",
        "description": "Model predictions change significantly when only sensitive attributes are modified.",
        "stage": "MODEL",
        "severity": "MEDIUM",
        "indicators": [
            "Prediction flip rate > 10%",
            "High sensitivity to demographic changes",
            "Inconsistent predictions for similar individuals"
        ],
        "consequences": [
            "Direct discrimination",
            "Lack of individual fairness",
            "Ethical concerns"
        ]
    }
}


# Fairness metric definitions
FAIRNESS_METRICS = {
    "demographic_parity": {
        "name": "Demographic Parity",
        "formula": "P(Ŷ=1|A=a) = P(Ŷ=1|A=b)",
        "description": "Equal acceptance rates across all demographic groups.",
        "when_to_use": "When equal representation in outcomes is desired (e.g., hiring, lending)",
        "threshold": "Difference < 10% or ratio > 0.8",
        "limitations": [
            "May conflict with accuracy",
            "Doesn't account for base rate differences",
            "Can be gamed by random selection"
        ]
    },
    "equal_opportunity": {
        "name": "Equal Opportunity",
        "formula": "P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)",
        "description": "Equal true positive rates (recall) for qualified individuals across groups.",
        "when_to_use": "When ensuring qualified individuals have equal chances (e.g., college admissions)",
        "threshold": "Difference < 10%",
        "limitations": [
            "Only considers qualified individuals",
            "Requires ground truth labels",
            "May allow disparate false positive rates"
        ]
    },
    "equalized_odds": {
        "name": "Equalized Odds",
        "formula": "P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for y∈{0,1}",
        "description": "Equal TPR and FPR across all groups.",
        "when_to_use": "When both benefits and harms should be distributed equally",
        "threshold": "Both TPR and FPR differences < 10%",
        "limitations": [
            "Most restrictive fairness criterion",
            "May significantly reduce accuracy",
            "Difficult to achieve in practice"
        ]
    },
    "individual_fairness": {
        "name": "Individual Fairness",
        "formula": "Similar individuals receive similar predictions",
        "description": "Individuals who are similar with respect to a task should receive similar predictions.",
        "when_to_use": "When consistency and individual treatment matter",
        "threshold": "Application-specific similarity metric",
        "limitations": [
            "Requires defining similarity metric",
            "Computationally expensive",
            "May conflict with group fairness"
        ]
    }
}


# Mitigation strategies
MITIGATION_STRATEGIES = {
    "data_collection": {
        "name": "Targeted Data Collection",
        "stage": "DATA",
        "description": "Collect additional samples from underrepresented groups to balance the dataset.",
        "steps": [
            "Identify underrepresented groups (proportion < 5%)",
            "Calculate target sample sizes for balance",
            "Collect or synthesize additional samples",
            "Validate data quality and representativeness"
        ],
        "pros": ["Addresses root cause", "Improves model performance", "Long-term solution"],
        "cons": ["Time-consuming", "May be expensive", "Not always feasible"],
        "effort": "HIGH"
    },
    "reweighting": {
        "name": "Sample Reweighting",
        "stage": "DATA",
        "description": "Assign higher weights to underrepresented samples during training.",
        "steps": [
            "Calculate group proportions",
            "Compute inverse propensity weights",
            "Apply weights during model training",
            "Validate fairness metrics"
        ],
        "pros": ["Easy to implement", "No new data needed", "Flexible"],
        "cons": ["May reduce accuracy", "Doesn't add information", "Can overfit minority groups"],
        "effort": "LOW"
    },
    "feature_removal": {
        "name": "Remove Proxy Features",
        "stage": "FEATURE",
        "description": "Remove or regularize features that act as proxies for sensitive attributes.",
        "steps": [
            "Identify proxy features (correlation > 0.3)",
            "Assess feature importance for predictions",
            "Remove or apply regularization",
            "Retrain and validate model"
        ],
        "pros": ["Reduces indirect discrimination", "Simple approach", "Legally safer"],
        "cons": ["May reduce accuracy", "Proxies can be subtle", "Doesn't guarantee fairness"],
        "effort": "MEDIUM"
    },
    "fairness_constraints": {
        "name": "Fairness-Aware Training",
        "stage": "MODEL",
        "description": "Incorporate fairness constraints or regularization during model training.",
        "steps": [
            "Choose fairness metric to optimize",
            "Add fairness constraint to loss function",
            "Tune fairness-accuracy tradeoff",
            "Validate on held-out data"
        ],
        "pros": ["Directly optimizes fairness", "Flexible", "Principled approach"],
        "cons": ["Requires specialized algorithms", "May reduce accuracy", "Complex to tune"],
        "effort": "HIGH"
    },
    "threshold_calibration": {
        "name": "Group-Specific Thresholds",
        "stage": "MODEL",
        "description": "Use different decision thresholds for different demographic groups.",
        "steps": [
            "Train single model on all data",
            "Calibrate thresholds per group to achieve fairness",
            "Validate fairness and accuracy metrics",
            "Document and monitor thresholds"
        ],
        "pros": ["Post-processing approach", "Preserves model", "Flexible"],
        "cons": ["Explicit group treatment", "May have legal issues", "Requires group labels at inference"],
        "effort": "MEDIUM"
    },
    "adversarial_debiasing": {
        "name": "Adversarial Debiasing",
        "stage": "MODEL",
        "description": "Train model to make predictions while preventing an adversary from predicting sensitive attributes.",
        "steps": [
            "Design adversarial architecture",
            "Train predictor and adversary jointly",
            "Balance prediction accuracy and fairness",
            "Validate on test data"
        ],
        "pros": ["Learns fair representations", "Doesn't require explicit fairness metrics", "Flexible"],
        "cons": ["Complex to implement", "Difficult to tune", "May reduce accuracy"],
        "effort": "HIGH"
    }
}


def get_bias_pattern(pattern_id: str) -> Dict[str, Any]:
    """Get bias pattern by ID."""
    return BIAS_PATTERNS.get(pattern_id, {})


def get_fairness_metric(metric_id: str) -> Dict[str, Any]:
    """Get fairness metric definition by ID."""
    return FAIRNESS_METRICS.get(metric_id, {})


def get_mitigation_strategy(strategy_id: str) -> Dict[str, Any]:
    """Get mitigation strategy by ID."""
    return MITIGATION_STRATEGIES.get(strategy_id, {})


def get_all_bias_patterns() -> Dict[str, Dict[str, Any]]:
    """Get all bias patterns."""
    return BIAS_PATTERNS


def get_all_fairness_metrics() -> Dict[str, Dict[str, Any]]:
    """Get all fairness metrics."""
    return FAIRNESS_METRICS


def get_all_mitigation_strategies() -> Dict[str, Dict[str, Any]]:
    """Get all mitigation strategies."""
    return MITIGATION_STRATEGIES


def get_strategies_for_stage(stage: str) -> List[Dict[str, Any]]:
    """Get mitigation strategies for a specific stage."""
    return [
        {**strategy, "id": strategy_id}
        for strategy_id, strategy in MITIGATION_STRATEGIES.items()
        if strategy["stage"] == stage
    ]


def get_patterns_for_stage(stage: str) -> List[Dict[str, Any]]:
    """Get bias patterns for a specific stage."""
    return [
        {**pattern, "id": pattern_id}
        for pattern_id, pattern in BIAS_PATTERNS.items()
        if pattern["stage"] == stage
    ]
