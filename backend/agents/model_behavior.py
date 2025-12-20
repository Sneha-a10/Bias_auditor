"""
Model Behavior Agent - Analyzes model predictions for bias.

Responsibilities:
- Compute global metrics (accuracy, AUC)
- Calculate per-subgroup metrics (acceptance rate, TPR, FPR, precision)
- Compute fairness gaps
- Perform counterfactual testing on binary attributes
- Generate model_bias.json and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, Any, List
import random

from utils import (
    load_run_config,
    load_raw_data,
    load_features,
    load_model,
    save_json,
    get_artifact_path,
    get_plots_dir,
    now_iso
)


def run_model_behavior(run_id: str) -> None:
    """
    Run model behavior agent.
    
    Args:
        run_id: Unique run identifier
    """
    # Load inputs
    config = load_run_config(run_id)
    df_raw = load_raw_data(run_id)
    df_features = load_features(run_id)
    model = load_model(run_id)
    
    target = config["target_column"]
    sens_attrs = config["sensitive_attributes"]
    thresholds = config["fairness_thresholds"]
    
    # Get predictions
    X = df_features.values
    y_true = df_raw[target].values
    y_pred = model.predict(X)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        y_proba = None
    
    # Initialize results
    results = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp": now_iso(),
        "global_metrics": {},
        "by_sensitive_attribute": {},
        "counterfactual": {},
        "flags": {}
    }
    
    # Compute global metrics
    accuracy = accuracy_score(y_true, y_pred)
    global_metrics = {"accuracy": float(accuracy)}
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            global_metrics["auc"] = float(auc)
        except ValueError:
            pass
    
    results["global_metrics"] = global_metrics
    
    # Analyze by sensitive attribute
    fairness_flags_overall = {
        "demographic_parity": False,
        "equal_opportunity": False,
        "equalized_odds": False
    }
    
    for sa in sens_attrs:
        attr_name = sa["name"]
        if attr_name not in df_raw.columns:
            continue
        
        # Compute per-subgroup metrics
        subgroups = []
        attr_values = df_raw[attr_name]
        
        for value in attr_values.unique():
            if pd.isna(value):
                continue
            
            mask = attr_values == value
            count = int(mask.sum())
            
            if count < thresholds["min_support_for_metrics"]:
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Acceptance rate (positive prediction rate)
            acceptance_rate = float(y_pred_group.mean())
            
            # Confusion matrix metrics
            tp = int(((y_true_group == 1) & (y_pred_group == 1)).sum())
            fp = int(((y_true_group == 0) & (y_pred_group == 1)).sum())
            tn = int(((y_true_group == 0) & (y_pred_group == 0)).sum())
            fn = int(((y_true_group == 1) & (y_pred_group == 0)).sum())
            
            tpr = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            precision = tp / max(1, tp + fp)
            base_rate = float(y_true_group.mean())
            
            subgroups.append({
                "value": str(value),
                "count": count,
                "acceptance_rate": float(acceptance_rate),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "precision": float(precision),
                "base_rate": float(base_rate)
            })
        
        # Compute fairness gaps
        if len(subgroups) >= 2:
            acceptance_rates = [s["acceptance_rate"] for s in subgroups]
            tprs = [s["tpr"] for s in subgroups]
            fprs = [s["fpr"] for s in subgroups]
            precisions = [s["precision"] for s in subgroups]
            base_rates = [s["base_rate"] for s in subgroups]
            
            gaps = {
                "acceptance_rate_gap": float(max(acceptance_rates) - min(acceptance_rates)),
                "tpr_gap": float(max(tprs) - min(tprs)),
                "fpr_gap": float(max(fprs) - min(fprs)),
                "precision_gap": float(max(precisions) - min(precisions)),
                "base_rate_gap": float(max(base_rates) - min(base_rates))
            }
            
            # Check fairness flags
            flags = {
                "demographic_parity": gaps["acceptance_rate_gap"] > thresholds["demographic_parity_diff_threshold"],
                "equal_opportunity": gaps["tpr_gap"] > thresholds["equal_opportunity_diff_threshold"],
                "equalized_odds": (
                    gaps["tpr_gap"] > thresholds["equalized_odds_diff_threshold"] and
                    gaps["fpr_gap"] > thresholds["equalized_odds_diff_threshold"]
                )
            }
            
            # Update overall flags
            for key in fairness_flags_overall:
                if flags[key]:
                    fairness_flags_overall[key] = True
        else:
            gaps = {}
            flags = {}
        
        results["by_sensitive_attribute"][attr_name] = {
            "subgroups": subgroups,
            "fairness_gaps": gaps,
            "fairness_flags": flags
        }
    
    # Counterfactual testing
    counterfactual_results = _run_counterfactual_testing(
        run_id, config, df_raw, df_features, model, thresholds
    )
    results["counterfactual"] = counterfactual_results
    
    # Compute overall flags and score
    model_bias_flag = (
        any(fairness_flags_overall.values()) or
        counterfactual_results.get("counterfactual_flag", False)
    )
    
    # Score: weighted combination of fairness gaps and counterfactual changes
    max_dp_gap = 0.0
    max_eo_gap = 0.0
    for attr_data in results["by_sensitive_attribute"].values():
        gaps = attr_data.get("fairness_gaps", {})
        max_dp_gap = max(max_dp_gap, gaps.get("acceptance_rate_gap", 0.0))
        max_eo_gap = max(max_eo_gap, gaps.get("tpr_gap", 0.0))
    
    max_cf_change = max(
        counterfactual_results.get("change_rate_per_attribute", {}).values(),
        default=0.0
    )
    
    # Normalize to [0, 1]
    norm_dp = min(1.0, max_dp_gap / 0.3)  # 0.3 is a severe gap
    norm_eo = min(1.0, max_eo_gap / 0.3)
    norm_cf = min(1.0, max_cf_change / 0.3)
    
    model_bias_score = min(1.0, 0.4 * norm_dp + 0.4 * norm_eo + 0.2 * norm_cf)
    
    # Build summary
    summary_parts = []
    if any(fairness_flags_overall.values()):
        flagged = [k for k, v in fairness_flags_overall.items() if v]
        summary_parts.append(f"Fairness violations: {', '.join(flagged)}")
    if counterfactual_results.get("counterfactual_flag", False):
        summary_parts.append("Counterfactual sensitivity detected")
    if not summary_parts:
        summary_parts.append("No significant model bias detected")
    
    summary = ". ".join(summary_parts) + "."
    
    results["flags"] = {
        "model_bias_flag": model_bias_flag,
        "model_bias_score": float(model_bias_score),
        "summary": summary
    }
    
    # Save JSON output
    save_json(get_artifact_path(run_id, "model_bias.json"), results)
    
    # Generate visualizations
    _generate_plots(run_id, results, config)


def _run_counterfactual_testing(
    run_id: str,
    config: Dict[str, Any],
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    model: Any,
    thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run counterfactual testing by flipping binary sensitive attributes.
    """
    sens_attrs = config["sensitive_attributes"]
    
    # Identify binary sensitive attributes
    binary_attrs = []
    for sa in sens_attrs:
        attr_name = sa["name"]
        if attr_name in df_raw.columns:
            unique_vals = df_raw[attr_name].dropna().unique()
            if len(unique_vals) == 2:
                binary_attrs.append((attr_name, list(unique_vals)))
    
    if not binary_attrs:
        return {
            "attributes_tested": [],
            "change_rate_per_attribute": {},
            "counterfactual_flag": False,
            "summary": "No binary sensitive attributes available for counterfactual testing."
        }
    
    # Sample rows for testing (max 500)
    sample_size = min(500, len(df_raw))
    sample_indices = random.sample(range(len(df_raw)), sample_size)
    
    change_rates = {}
    
    for attr_name, (val1, val2) in binary_attrs:
        changed_count = 0
        
        for idx in sample_indices:
            original_attr_val = df_raw.iloc[idx][attr_name]
            
            if pd.isna(original_attr_val):
                continue
            
            # Flip the attribute
            flipped_val = val2 if original_attr_val == val1 else val1
            
            # Create flipped feature vector
            flipped_features = df_features.iloc[idx].copy()
            
            # Check if attribute is one-hot encoded in features
            onehot_cols = [col for col in df_features.columns if col.startswith(f"{attr_name}__")]
            
            if onehot_cols:
                # Flip one-hot encoding
                for col in onehot_cols:
                    if col == f"{attr_name}__{original_attr_val}":
                        flipped_features[col] = 0
                    elif col == f"{attr_name}__{flipped_val}":
                        flipped_features[col] = 1
            
            # Get predictions
            original_pred = model.predict(df_features.iloc[idx:idx+1].values)[0]
            flipped_pred = model.predict(flipped_features.values.reshape(1, -1))[0]
            
            if original_pred != flipped_pred:
                changed_count += 1
        
        change_rate = changed_count / max(1, sample_size)
        change_rates[attr_name] = float(change_rate)
    
    # Check if any change rate exceeds threshold
    max_change_rate = max(change_rates.values()) if change_rates else 0.0
    counterfactual_flag = max_change_rate > thresholds["counterfactual_change_threshold"]
    
    summary = f"Tested {len(binary_attrs)} binary attributes on {sample_size} samples."
    if counterfactual_flag:
        summary += f" Max change rate: {max_change_rate:.1%}."
    
    return {
        "attributes_tested": [attr for attr, _ in binary_attrs],
        "change_rate_per_attribute": change_rates,
        "counterfactual_flag": counterfactual_flag,
        "summary": summary
    }


def _generate_plots(run_id: str, results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Generate visualization plots for model bias."""
    plots_dir = get_plots_dir(run_id)
    
    sns.set_style("whitegrid")
    
    # Plot fairness gaps for each sensitive attribute
    for attr_name, attr_data in results["by_sensitive_attribute"].items():
        subgroups = attr_data.get("subgroups", [])
        if len(subgroups) < 2:
            continue
        
        values = [s["value"] for s in subgroups]
        acceptance_rates = [s["acceptance_rate"] for s in subgroups]
        tprs = [s["tpr"] for s in subgroups]
        fprs = [s["fpr"] for s in subgroups]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Acceptance rate
        axes[0].bar(values, acceptance_rates, color='steelblue')
        axes[0].set_xlabel(attr_name)
        axes[0].set_ylabel('Acceptance Rate')
        axes[0].set_title(f'Acceptance Rate by {attr_name}')
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        
        # TPR
        axes[1].bar(values, tprs, color='green')
        axes[1].set_xlabel(attr_name)
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'TPR by {attr_name}')
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        
        # FPR
        axes[2].bar(values, fprs, color='orange')
        axes[2].set_xlabel(attr_name)
        axes[2].set_ylabel('False Positive Rate')
        axes[2].set_title(f'FPR by {attr_name}')
        axes[2].set_ylim([0, 1])
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"model_{attr_name}_metrics.png", dpi=100, bbox_inches='tight')
        plt.close()
