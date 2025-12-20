"""
Bias Aggregator Agent - Determines primary bias origin.

Responsibilities:
- Read outputs from all three agents
- Normalize scores to status (Pass/Warning/Fail)
- Apply heuristic to determine primary_origin
- Generate evidence references
- Map issues to recommended fixes
- Create bias_origin_report.json
- Use Gemini for enhanced explanations (if available)
"""
from typing import Dict, Any, List
from utils import (
    load_json,
    save_json,
    get_artifact_path,
    now_iso
)
from observability import get_logger, get_metrics
from gemini_agent import get_gemini_agent, is_gemini_available


def run_bias_aggregator(run_id: str) -> None:
    """
    Run bias aggregator agent with Gemini enhancement.
    
    Args:
        run_id: Unique run identifier
    """
    logger = get_logger("bias_aggregator", run_id)
    metrics = get_metrics(run_id)
    
    logger.info("Starting bias aggregation")
    
    # Load agent outputs
    data_bias = load_json(get_artifact_path(run_id, "data_bias.json"))
    feature_bias = load_json(get_artifact_path(run_id, "feature_bias.json"))
    model_bias = load_json(get_artifact_path(run_id, "model_bias.json"))
    
    # Extract scores
    data_score = data_bias["overall"]["data_bias_score"]
    feature_score = feature_bias["flags"]["feature_bias_score"]
    model_score = model_bias["flags"]["model_bias_score"]
    
    # Convert scores to status
    data_status = _score_to_status(data_score)
    feature_status = _score_to_status(feature_score)
    model_status = _score_to_status(model_score)
    
    # Determine primary origin using heuristic
    primary_origin = _determine_primary_origin(
        data_score, feature_score, model_score,
        data_status, feature_status, model_status
    )
    
    # Build checkpoint summary
    checkpoint_summary = [
        {
            "stage": "Data",
            "status": data_status,
            "score": float(data_score),
            "flagged_issues": _extract_data_issues(data_bias)
        },
        {
            "stage": "Features",
            "status": feature_status,
            "score": float(feature_score),
            "flagged_issues": _extract_feature_issues(feature_bias)
        },
        {
            "stage": "Model",
            "status": model_status,
            "score": float(model_score),
            "flagged_issues": _extract_model_issues(model_bias)
        }
    ]
    
    # Generate explanation (use Gemini if available)
    gemini_agent = get_gemini_agent()
    if gemini_agent:
        logger.info("Using Gemini for enhanced explanation")
        try:
            explanation = gemini_agent.generate_bias_explanation(
                primary_origin, checkpoint_summary, data_bias, feature_bias, model_bias
            )
            logger.info("Gemini explanation generated successfully")
        except Exception as e:
            logger.warning(f"Gemini explanation failed, using fallback: {str(e)}")
            explanation = _generate_explanation(
                primary_origin, data_status, feature_status, model_status,
                data_bias, feature_bias, model_bias
            )
    else:
        logger.info("Gemini not available, using standard explanation")
        explanation = _generate_explanation(
            primary_origin, data_status, feature_status, model_status,
            data_bias, feature_bias, model_bias
        )
    
    # Collect evidence references
    evidence = {
        "data_evidence_ids": _get_data_evidence(data_bias),
        "feature_evidence_ids": _get_feature_evidence(feature_bias),
        "model_evidence_ids": _get_model_evidence(model_bias)
    }
    
    # Generate recommended fixes (use Gemini if available)
    if gemini_agent:
        logger.info("Using Gemini for enhanced recommendations")
        try:
            recommended_fixes = gemini_agent.generate_recommendations(
                checkpoint_summary, primary_origin, data_bias, feature_bias, model_bias
            )
            logger.info("Gemini recommendations generated successfully")
        except Exception as e:
            logger.warning(f"Gemini recommendations failed, using fallback: {str(e)}")
            recommended_fixes = _generate_recommended_fixes(
                checkpoint_summary, data_bias, feature_bias, model_bias
            )
    else:
        logger.info("Gemini not available, using standard recommendations")
        recommended_fixes = _generate_recommended_fixes(
            checkpoint_summary, data_bias, feature_bias, model_bias
        )
    
    # Build final report
    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp": now_iso(),
        "checkpoint_summary": checkpoint_summary,
        "bias_origin_verdict": {
            "primary_origin": primary_origin,
            "explanation": explanation
        },
        "evidence": evidence,
        "recommended_fixes": recommended_fixes
    }
    
    # Record metrics
    metrics.record_bias_score("data", data_score)
    metrics.record_bias_score("features", feature_score)
    metrics.record_bias_score("model", model_score)
    metrics.record_primary_origin(primary_origin)
    
    # Save report
    save_json(get_artifact_path(run_id, "bias_origin_report.json"), report)
    logger.info("Bias aggregation complete", primary_origin=primary_origin)


def _score_to_status(score: float) -> str:
    """Convert bias score to status."""
    if score >= 0.7:
        return "Fail"
    elif score >= 0.3:
        return "Warning"
    else:
        return "Pass"


def _determine_primary_origin(
    data_score: float,
    feature_score: float,
    model_score: float,
    data_status: str,
    feature_status: str,
    model_status: str
) -> str:
    """
    Determine primary bias origin using heuristic.
    
    Priority order: Data > Features > Model
    First check for "Fail" status, then "Warning"
    """
    # Check for Fail status first
    if data_status == "Fail":
        return "DATA"
    elif feature_status == "Fail":
        return "FEATURE"
    elif model_status == "Fail":
        return "MODEL"
    
    # Check for Warning status
    elif data_status == "Warning":
        return "DATA"
    elif feature_status == "Warning":
        return "FEATURE"
    elif model_status == "Warning":
        return "MODEL"
    
    # All Pass or multiple failures
    else:
        # If multiple high scores, mark as unclear
        fail_count = sum([
            data_score >= 0.7,
            feature_score >= 0.7,
            model_score >= 0.7
        ])
        if fail_count > 1:
            return "MULTIPLE/UNCLEAR"
        return "MULTIPLE/UNCLEAR"


def _extract_data_issues(data_bias: Dict[str, Any]) -> List[str]:
    """Extract flagged issues from data bias analysis."""
    issues = []
    
    # Check for underrepresentation
    for attr_name, attr_data in data_bias["sensitive_attributes"].items():
        for subgroup in attr_data["subgroups"]:
            if subgroup.get("underrepresented", False):
                issues.append("UNDERREPRESENTED_GROUPS")
                break
        if issues:
            break
    
    # Check for missing groups
    for attr_name, attr_data in data_bias["sensitive_attributes"].items():
        if attr_data.get("missing_groups", []):
            issues.append("MISSING_EXPECTED_GROUPS")
            break
    
    # Check for label imbalance
    ratio = data_bias["overall"]["global_label_counts"]["ratio"]
    if ratio > 4.0:
        issues.append("SEVERE_LABEL_IMBALANCE")
    
    return issues


def _extract_feature_issues(feature_bias: Dict[str, Any]) -> List[str]:
    """Extract flagged issues from feature bias analysis."""
    issues = []
    
    # Check for proxy features
    proxy_features = feature_bias.get("proxy_features", {})
    total_proxies = sum(len(pf) for pf in proxy_features.values())
    if total_proxies > 0:
        issues.append("PROXY_FEATURES")
    
    # Check for encoding risks
    encoding_risks = feature_bias.get("encoding_risks", [])
    for risk in encoding_risks:
        issues.append(risk["type"])
    
    return issues


def _extract_model_issues(model_bias: Dict[str, Any]) -> List[str]:
    """Extract flagged issues from model bias analysis."""
    issues = []
    
    # Check fairness flags
    for attr_name, attr_data in model_bias["by_sensitive_attribute"].items():
        flags = attr_data.get("fairness_flags", {})
        if flags.get("demographic_parity", False):
            issues.append("DEMOGRAPHIC_PARITY_VIOLATION")
        if flags.get("equal_opportunity", False):
            issues.append("EQUAL_OPPORTUNITY_VIOLATION")
        if flags.get("equalized_odds", False):
            issues.append("EQUALIZED_ODDS_VIOLATION")
    
    # Check counterfactual
    if model_bias["counterfactual"].get("counterfactual_flag", False):
        issues.append("COUNTERFACTUAL_SENSITIVITY")
    
    return list(set(issues))  # Remove duplicates


def _generate_explanation(
    primary_origin: str,
    data_status: str,
    feature_status: str,
    model_status: str,
    data_bias: Dict[str, Any],
    feature_bias: Dict[str, Any],
    model_bias: Dict[str, Any]
) -> str:
    """Generate human-readable explanation of the verdict."""
    
    if primary_origin == "DATA":
        explanation = (
            "The earliest and strongest bias signal appears in the data. "
            f"{data_bias['overall']['summary']} "
            "Feature and model issues, if present, are likely downstream consequences of this initial data bias."
        )
    
    elif primary_origin == "FEATURE":
        explanation = (
            "Bias primarily originates in the feature engineering stage. "
            f"{feature_bias['flags']['summary']} "
            "While the raw data may have some issues, the feature transformation process "
            "introduces or amplifies bias signals."
        )
    
    elif primary_origin == "MODEL":
        explanation = (
            "Bias primarily manifests in the model's predictions. "
            f"{model_bias['flags']['summary']} "
            "The data and features may be relatively unbiased, but the model has learned "
            "to make discriminatory predictions."
        )
    
    else:  # MULTIPLE/UNCLEAR
        explanation = (
            "Bias is present at multiple stages of the pipeline, making it difficult to pinpoint "
            "a single primary origin. A comprehensive remediation strategy addressing all stages "
            "is recommended."
        )
    
    return explanation


def _get_data_evidence(data_bias: Dict[str, Any]) -> List[str]:
    """Get evidence IDs for data bias."""
    evidence = []
    
    for attr_name in data_bias["sensitive_attributes"].keys():
        evidence.append(f"sensitive_attributes.{attr_name}.subgroups")
    
    evidence.append("overall.global_label_counts")
    
    return evidence


def _get_feature_evidence(feature_bias: Dict[str, Any]) -> List[str]:
    """Get evidence IDs for feature bias."""
    evidence = []
    
    if feature_bias.get("proxy_features"):
        evidence.append("proxy_features")
    
    if feature_bias.get("encoding_risks"):
        evidence.append("encoding_risks")
    
    evidence.append("heatmap_data")
    
    return evidence


def _get_model_evidence(model_bias: Dict[str, Any]) -> List[str]:
    """Get evidence IDs for model bias."""
    evidence = []
    
    for attr_name in model_bias["by_sensitive_attribute"].keys():
        evidence.append(f"by_sensitive_attribute.{attr_name}.fairness_gaps")
    
    if model_bias["counterfactual"].get("counterfactual_flag"):
        evidence.append("counterfactual")
    
    return evidence


def _generate_recommended_fixes(
    checkpoint_summary: List[Dict[str, Any]],
    data_bias: Dict[str, Any],
    feature_bias: Dict[str, Any],
    model_bias: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Generate recommended fixes based on detected issues."""
    
    fixes = {
        "Data": [],
        "Features": [],
        "Model": []
    }
    
    # Data fixes
    data_issues = checkpoint_summary[0]["flagged_issues"]
    if "UNDERREPRESENTED_GROUPS" in data_issues:
        fixes["Data"].append(
            "Collect additional samples from underrepresented groups to balance the dataset."
        )
    if "MISSING_EXPECTED_GROUPS" in data_issues:
        fixes["Data"].append(
            "Ensure all expected demographic categories are represented in the training data."
        )
    if "SEVERE_LABEL_IMBALANCE" in data_issues:
        fixes["Data"].append(
            "Apply reweighting, stratified sampling, or SMOTE to balance the label distribution."
        )
    
    # Feature fixes
    feature_issues = checkpoint_summary[1]["flagged_issues"]
    if "PROXY_FEATURES" in feature_issues:
        proxy_attrs = feature_bias.get("proxy_features", {})
        for attr, features in proxy_attrs.items():
            if features:
                fixes["Features"].append(
                    f"Remove or regularize proxy features for '{attr}': {', '.join(features[:3])}{'...' if len(features) > 3 else ''}"
                )
    if "TARGET_LEAKAGE" in feature_issues:
        fixes["Features"].append(
            "Remove features that encode the target variable or its transformations."
        )
    if "SPARSE_ONE_HOT" in feature_issues:
        fixes["Features"].append(
            "Review sparse one-hot encoded features; consider grouping rare categories."
        )
    
    # Model fixes
    model_issues = checkpoint_summary[2]["flagged_issues"]
    if "DEMOGRAPHIC_PARITY_VIOLATION" in model_issues:
        fixes["Model"].append(
            "Consider threshold calibration per subgroup to reduce acceptance rate gaps."
        )
    if "EQUAL_OPPORTUNITY_VIOLATION" in model_issues:
        fixes["Model"].append(
            "Apply fairness-aware training methods or constraints to equalize TPR across groups."
        )
    if "EQUALIZED_ODDS_VIOLATION" in model_issues:
        fixes["Model"].append(
            "Use post-processing techniques to equalize both TPR and FPR across groups."
        )
    if "COUNTERFACTUAL_SENSITIVITY" in model_issues:
        fixes["Model"].append(
            "Model predictions are sensitive to sensitive attributes. Consider removing these features or using fairness constraints."
        )
    
    # Remove empty categories
    fixes = {k: v for k, v in fixes.items() if v}
    
    return fixes
