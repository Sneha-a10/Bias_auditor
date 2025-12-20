"""
Data Auditor Agent - Analyzes raw data for bias.

Responsibilities:
- Detect underrepresented groups
- Identify missing expected categories
- Detect severe label imbalance
- Generate data_bias.json and visualizations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

from utils import (
    load_run_config,
    load_raw_data,
    save_json,
    get_artifact_path,
    get_plots_dir,
    now_iso
)


def run_data_auditor(run_id: str) -> None:
    """
    Run data auditor agent.
    
    Args:
        run_id: Unique run identifier
    """
    # Load inputs
    config = load_run_config(run_id)
    df = load_raw_data(run_id)
    
    target = config["target_column"]
    sens_attrs = config["sensitive_attributes"]
    thresholds = config["fairness_thresholds"]
    
    # Initialize results
    results = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp": now_iso(),
        "sensitive_attributes": {},
        "overall": {}
    }
    
    # Track flags
    any_underrep = False
    any_missing = False
    any_severe_imbalance = False
    
    # Analyze each sensitive attribute
    for sa in sens_attrs:
        name = sa["name"]
        expected = sa.get("expected_categories")
        
        if name not in df.columns:
            raise ValueError(f"Sensitive attribute '{name}' not found in raw data")
        
        stats = []
        total = len(df)
        
        # Compute per-group statistics
        for value, sub_df in df.groupby(name):
            count = len(sub_df)
            proportion = count / total
            label_rate = float(sub_df[target].mean())
            underrep = proportion < thresholds["min_group_proportion_threshold"]
            
            stats.append({
                "value": str(value),
                "count": int(count),
                "proportion": float(proportion),
                "label_rate": label_rate,
                "underrepresented": underrep
            })
            
            if underrep:
                any_underrep = True
        
        # Check for missing expected categories
        missing = []
        if expected:
            observed = set(df[name].dropna().unique())
            missing = [c for c in expected if c not in observed]
            if missing:
                any_missing = True
        
        results["sensitive_attributes"][name] = {
            "subgroups": stats,
            "missing_groups": missing
        }
    
    # Analyze global label imbalance
    pos = int((df[target] == 1).sum())
    neg = int((df[target] == 0).sum())
    maj = max(pos, neg)
    min_ = max(1, min(pos, neg))
    ratio = maj / min_
    
    if ratio > thresholds["label_imbalance_ratio_threshold"]:
        any_severe_imbalance = True
    
    # Compute overall bias flag and score
    data_bias_flag = any_underrep or any_missing or any_severe_imbalance
    score = (
        (0.3 if any_underrep else 0) +
        (0.3 if any_missing else 0) +
        (0.4 if any_severe_imbalance else 0)
    )
    score = min(score, 1.0)
    
    # Build summary text
    summary_parts = []
    if any_severe_imbalance:
        summary_parts.append(f"Severe class imbalance detected (ratio: {ratio:.2f})")
    if any_underrep:
        summary_parts.append("Underrepresented groups found")
    if any_missing:
        summary_parts.append("Missing expected categories")
    if not summary_parts:
        summary_parts.append("No significant data bias detected")
    
    summary = ". ".join(summary_parts) + "."
    
    results["overall"] = {
        "data_bias_flag": data_bias_flag,
        "data_bias_score": float(score),
        "global_label_counts": {
            "positive": pos,
            "negative": neg,
            "ratio": float(ratio)
        },
        "summary": summary
    }
    
    # Save JSON output
    save_json(get_artifact_path(run_id, "data_bias.json"), results)
    
    # Generate visualizations
    _generate_plots(run_id, df, config, results)


def _generate_plots(
    run_id: str,
    df: pd.DataFrame,
    config: Dict[str, Any],
    results: Dict[str, Any]
) -> None:
    """Generate visualization plots for data bias."""
    plots_dir = get_plots_dir(run_id)
    target = config["target_column"]
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Subgroup proportions for each sensitive attribute
    for sa in config["sensitive_attributes"]:
        name = sa["name"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Group sizes
        stats = results["sensitive_attributes"][name]["subgroups"]
        values = [s["value"] for s in stats]
        counts = [s["count"] for s in stats]
        proportions = [s["proportion"] for s in stats]
        
        axes[0].bar(values, counts, color='steelblue')
        axes[0].set_xlabel(name)
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Group Sizes by {name}')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add proportion labels
        for i, (v, p) in enumerate(zip(values, proportions)):
            axes[0].text(i, counts[i], f'{p:.1%}', ha='center', va='bottom')
        
        # Subplot 2: Label rates
        label_rates = [s["label_rate"] for s in stats]
        colors = ['orange' if s["underrepresented"] else 'steelblue' for s in stats]
        
        axes[1].bar(values, label_rates, color=colors)
        axes[1].set_xlabel(name)
        axes[1].set_ylabel('Positive Label Rate')
        axes[1].set_title(f'Label Rate by {name}')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"data_{name}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Overall label distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    label_counts = results["overall"]["global_label_counts"]
    labels = ['Negative (0)', 'Positive (1)']
    counts = [label_counts["negative"], label_counts["positive"]]
    colors = ['#ff6b6b', '#4ecdc4']
    
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Global Label Distribution')
    
    # Add ratio annotation
    ratio = label_counts["ratio"]
    ax.text(0.5, max(counts) * 0.9, f'Imbalance Ratio: {ratio:.2f}',
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "data_label_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()
