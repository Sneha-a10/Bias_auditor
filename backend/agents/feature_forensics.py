"""
Feature Forensics Agent - Analyzes feature engineering for bias.

Responsibilities:
- Detect feature types (numeric, binary, one-hot)
- Calculate associations between features and sensitive attributes
- Identify proxy features
- Detect encoding risks (target leakage, sparse one-hot)
- Generate feature_bias.json and heatmap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, Any, List, Tuple

from utils import (
    load_run_config,
    load_raw_data,
    load_features,
    save_json,
    get_artifact_path,
    get_plots_dir,
    now_iso
)


def run_feature_forensics(run_id: str) -> None:
    """
    Run feature forensics agent.
    
    Args:
        run_id: Unique run identifier
    """
    # Load inputs
    config = load_run_config(run_id)
    df_raw = load_raw_data(run_id)
    df_features = load_features(run_id)
    
    sens_attrs = config["sensitive_attributes"]
    thresholds = config["fairness_thresholds"]
    target = config["target_column"]
    
    # Initialize results
    results = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp": now_iso(),
        "features": {},
        "proxy_features": {},
        "encoding_risks": [],
        "heatmap_data": {},
        "flags": {}
    }
    
    # Analyze each feature
    proxy_features_by_attr = {sa["name"]: [] for sa in sens_attrs}
    
    for feature_col in df_features.columns:
        feature_data = df_features[feature_col]
        associations = {}
        
        # Calculate association with each sensitive attribute
        for sa in sens_attrs:
            attr_name = sa["name"]
            if attr_name not in df_raw.columns:
                continue
            
            attr_data = df_raw[attr_name]
            
            # Calculate association score
            score = _calculate_association(feature_data, attr_data)
            associations[attr_name] = float(score)
            
            # Check if proxy
            if score > thresholds["proxy_corr_threshold"]:
                proxy_features_by_attr[attr_name].append(feature_col)
        
        # Determine if this feature is a proxy for any attribute
        proxy_flag = any(
            score > thresholds["proxy_corr_threshold"]
            for score in associations.values()
        )
        
        results["features"][feature_col] = {
            "association_to_sensitive_attrs": associations,
            "proxy_flag": proxy_flag
        }
    
    results["proxy_features"] = proxy_features_by_attr
    
    # Check for encoding risks
    encoding_risks = []
    
    # 1. Target leakage - check if any feature name contains target column name
    target_leakage_features = [
        col for col in df_features.columns
        if target.lower() in col.lower() and col.lower() != target.lower()
    ]
    if target_leakage_features:
        encoding_risks.append({
            "type": "TARGET_LEAKAGE",
            "description": f"Features appear to encode the target: {', '.join(target_leakage_features[:5])}"
        })
    
    # 2. Sparse one-hot - binary features with very low/high activation
    sparse_features = []
    for col in df_features.columns:
        if _is_binary(df_features[col]):
            mean_val = df_features[col].mean()
            if mean_val < 0.01 or mean_val > 0.99:
                sparse_features.append(col)
    
    if sparse_features:
        encoding_risks.append({
            "type": "SPARSE_ONE_HOT",
            "description": f"Several one-hot features have very low activation (<1%): {len(sparse_features)} features"
        })
    
    results["encoding_risks"] = encoding_risks
    
    # Prepare heatmap data
    attr_names = [sa["name"] for sa in sens_attrs if sa["name"] in df_raw.columns]
    feature_names = list(df_features.columns)
    
    # Build association matrix
    matrix = []
    for feature_col in feature_names:
        row = []
        for attr_name in attr_names:
            score = results["features"][feature_col]["association_to_sensitive_attrs"].get(attr_name, 0.0)
            row.append(score)
        matrix.append(row)
    
    results["heatmap_data"] = {
        "sensitive_attributes": attr_names,
        "features": feature_names,
        "matrix": matrix
    }
    
    # Compute aggregate flags and score
    total_features = len(df_features.columns)
    total_proxy_features = sum(len(pf) for pf in proxy_features_by_attr.values())
    num_encoding_risks = len(encoding_risks)
    
    feature_bias_flag = total_proxy_features > 0 or num_encoding_risks > 0
    
    # Score: weighted combination of proxy ratio and encoding risks
    proxy_ratio = total_proxy_features / max(1, total_features)
    feature_bias_score = min(1.0, 0.7 * proxy_ratio + 0.3 * min(1.0, num_encoding_risks / 2))
    
    # Build summary
    summary_parts = []
    if total_proxy_features > 0:
        summary_parts.append(f"{total_proxy_features} proxy features detected")
    if num_encoding_risks > 0:
        summary_parts.append(f"{num_encoding_risks} encoding risks found")
    if not summary_parts:
        summary_parts.append("No significant feature bias detected")
    
    summary = ". ".join(summary_parts) + "."
    
    results["flags"] = {
        "feature_bias_flag": feature_bias_flag,
        "feature_bias_score": float(feature_bias_score),
        "summary": summary
    }
    
    # Save JSON output
    save_json(get_artifact_path(run_id, "feature_bias.json"), results)
    
    # Generate heatmap
    _generate_heatmap(run_id, results)


def _calculate_association(feature: pd.Series, attribute: pd.Series) -> float:
    """
    Calculate association between a feature and a sensitive attribute.
    
    Returns a score between 0 and 1.
    """
    # Remove NaN values
    mask = ~(feature.isna() | attribute.isna())
    feature = feature[mask]
    attribute = attribute[mask]
    
    if len(feature) < 10:
        return 0.0
    
    feature_is_binary = _is_binary(feature)
    feature_is_numeric = _is_numeric(feature)
    attr_is_binary = len(attribute.unique()) == 2
    
    try:
        if feature_is_numeric and attr_is_binary:
            # Point-biserial correlation
            # Encode attribute as 0/1
            attr_encoded = pd.factorize(attribute)[0]
            corr, _ = pointbiserialr(attr_encoded, feature)
            return abs(corr)
        
        elif feature_is_binary and attr_is_binary:
            # Phi coefficient (Pearson on two binary variables)
            feature_encoded = pd.factorize(feature)[0]
            attr_encoded = pd.factorize(attribute)[0]
            corr = np.corrcoef(feature_encoded, attr_encoded)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        
        else:
            # Use mutual information for categorical or mixed types
            feature_encoded = pd.factorize(feature)[0].reshape(-1, 1)
            attr_encoded = pd.factorize(attribute)[0]
            mi = mutual_info_classif(feature_encoded, attr_encoded, random_state=42)[0]
            # Normalize MI to [0, 1] range (approximate)
            return min(1.0, mi / 2.0)
    
    except Exception:
        return 0.0


def _is_binary(series: pd.Series) -> bool:
    """Check if a series is binary (only 0 and 1 values)."""
    unique_vals = set(series.dropna().unique())
    return unique_vals.issubset({0, 1, 0.0, 1.0})


def _is_numeric(series: pd.Series) -> bool:
    """Check if a series is numeric with sufficient unique values."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    n_unique = series.nunique()
    return n_unique > 10  # Arbitrary threshold


def _generate_heatmap(run_id: str, results: Dict[str, Any]) -> None:
    """Generate heatmap visualization of feature-attribute associations."""
    plots_dir = get_plots_dir(run_id)
    
    heatmap_data = results["heatmap_data"]
    attrs = heatmap_data["sensitive_attributes"]
    features = heatmap_data["features"]
    matrix = np.array(heatmap_data["matrix"])
    
    if len(features) == 0 or len(attrs) == 0:
        return
    
    # Limit to top features by max association for readability
    max_features_to_show = 30
    if len(features) > max_features_to_show:
        max_associations = matrix.max(axis=1)
        top_indices = np.argsort(max_associations)[-max_features_to_show:]
        matrix = matrix[top_indices]
        features = [features[i] for i in top_indices]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(attrs) * 2), max(10, len(features) * 0.3)))
    
    sns.heatmap(
        matrix,
        xticklabels=attrs,
        yticklabels=features,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Association Score'},
        ax=ax
    )
    
    ax.set_title('Feature-Attribute Association Heatmap', fontsize=14, pad=20)
    ax.set_xlabel('Sensitive Attributes', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_heatmap.png", dpi=100, bbox_inches='tight')
    plt.close()
