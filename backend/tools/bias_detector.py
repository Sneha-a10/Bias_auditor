"""
Bias detection tools for the Bias Auditor.

Provides specialized tools for:
- Subgroup analysis
- Proxy feature detection
- Fairness metric calculation
- Counterfactual generation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import Field
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from .base import Tool, ToolInput, ToolOutput, ToolRegistry


# ============================================================================
# Subgroup Analysis Tool
# ============================================================================

class SubgroupAnalysisInput(ToolInput):
    """Input for subgroup analysis."""
    data: Dict[str, List[Any]] = Field(description="Data as dict of column->values")
    sensitive_attr: str = Field(description="Sensitive attribute column name")
    target_column: str = Field(description="Target column name")
    min_group_proportion: float = Field(0.05, description="Minimum group proportion threshold")


class SubgroupAnalysisOutput(ToolOutput):
    """Output from subgroup analysis."""
    subgroups: List[Dict[str, Any]] = Field(default_factory=list, description="Subgroup statistics")
    underrepresented_groups: List[str] = Field(default_factory=list, description="Underrepresented group names")
    total_samples: int = Field(0, description="Total number of samples")


class SubgroupAnalysisTool(Tool):
    """Tool for analyzing demographic subgroups in data."""
    
    name = "subgroup_analysis"
    description = "Analyze demographic subgroups for representation and label distribution"
    input_schema = SubgroupAnalysisInput
    output_schema = SubgroupAnalysisOutput
    
    def __init__(self):
        super().__init__()
        ToolRegistry.register(self)
    
    def _run(self, data: Dict[str, List[Any]], sensitive_attr: str, 
             target_column: str, min_group_proportion: float) -> Dict[str, Any]:
        """Analyze subgroups."""
        df = pd.DataFrame(data)
        total_samples = len(df)
        
        subgroups = []
        underrepresented = []
        
        # Group by sensitive attribute
        for group_value, group_df in df.groupby(sensitive_attr):
            count = len(group_df)
            proportion = count / total_samples
            
            # Calculate label distribution
            label_counts = group_df[target_column].value_counts().to_dict()
            
            subgroup_info = {
                "value": str(group_value),
                "count": int(count),
                "proportion": float(proportion),
                "label_distribution": {str(k): int(v) for k, v in label_counts.items()}
            }
            
            # Check if underrepresented
            if proportion < min_group_proportion:
                underrepresented.append(str(group_value))
                subgroup_info["underrepresented"] = True
            else:
                subgroup_info["underrepresented"] = False
            
            subgroups.append(subgroup_info)
        
        return {
            "success": True,
            "subgroups": subgroups,
            "underrepresented_groups": underrepresented,
            "total_samples": total_samples
        }


# ============================================================================
# Proxy Detector Tool
# ============================================================================

class ProxyDetectorInput(ToolInput):
    """Input for proxy detection."""
    features: Dict[str, List[Any]] = Field(description="Feature data as dict")
    sensitive_attrs: Dict[str, List[Any]] = Field(description="Sensitive attribute data as dict")
    correlation_threshold: float = Field(0.3, description="Correlation threshold for proxy detection")


class ProxyDetectorOutput(ToolOutput):
    """Output from proxy detection."""
    proxy_features: Dict[str, List[str]] = Field(default_factory=dict, description="Proxy features by attribute")
    associations: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Feature-attribute associations")


class ProxyDetectorTool(Tool):
    """Tool for detecting proxy features."""
    
    name = "proxy_detector"
    description = "Detect features that act as proxies for sensitive attributes"
    input_schema = ProxyDetectorInput
    output_schema = ProxyDetectorOutput
    
    def __init__(self):
        super().__init__()
        ToolRegistry.register(self)
    
    def _run(self, features: Dict[str, List[Any]], sensitive_attrs: Dict[str, List[Any]],
             correlation_threshold: float) -> Dict[str, Any]:
        """Detect proxy features."""
        df_features = pd.DataFrame(features)
        df_attrs = pd.DataFrame(sensitive_attrs)
        
        proxy_features = {attr: [] for attr in df_attrs.columns}
        associations = {}
        
        for feature_col in df_features.columns:
            feature_data = df_features[feature_col]
            associations[feature_col] = {}
            
            for attr_col in df_attrs.columns:
                attr_data = df_attrs[attr_col]
                
                # Calculate association
                score = self._calculate_association(feature_data, attr_data)
                associations[feature_col][attr_col] = float(score)
                
                # Check if proxy
                if score > correlation_threshold:
                    proxy_features[attr_col].append(feature_col)
        
        return {
            "success": True,
            "proxy_features": proxy_features,
            "associations": associations
        }
    
    def _calculate_association(self, feature: pd.Series, attribute: pd.Series) -> float:
        """Calculate association between feature and attribute."""
        # Remove NaN values
        mask = ~(feature.isna() | attribute.isna())
        feature = feature[mask]
        attribute = attribute[mask]
        
        if len(feature) < 10:
            return 0.0
        
        try:
            # Use mutual information for general case
            feature_encoded = pd.factorize(feature)[0].reshape(-1, 1)
            attr_encoded = pd.factorize(attribute)[0]
            mi = mutual_info_classif(feature_encoded, attr_encoded, random_state=42)[0]
            return min(1.0, mi / 2.0)
        except Exception:
            return 0.0


# ============================================================================
# Fairness Metric Tool
# ============================================================================

class FairnessMetricInput(ToolInput):
    """Input for fairness metrics."""
    predictions: List[int] = Field(description="Model predictions")
    labels: List[int] = Field(description="True labels")
    sensitive_attr: List[Any] = Field(description="Sensitive attribute values")


class FairnessMetricOutput(ToolOutput):
    """Output from fairness metrics."""
    demographic_parity: Dict[str, float] = Field(default_factory=dict, description="Demographic parity by group")
    equal_opportunity: Dict[str, float] = Field(default_factory=dict, description="TPR by group")
    equalized_odds: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="TPR and FPR by group")
    overall_metrics: Dict[str, float] = Field(default_factory=dict, description="Overall fairness gaps")


class FairnessMetricTool(Tool):
    """Tool for calculating fairness metrics."""
    
    name = "fairness_metrics"
    description = "Calculate demographic parity, equal opportunity, and equalized odds"
    input_schema = FairnessMetricInput
    output_schema = FairnessMetricOutput
    
    def __init__(self):
        super().__init__()
        ToolRegistry.register(self)
    
    def _run(self, predictions: List[int], labels: List[int], 
             sensitive_attr: List[Any]) -> Dict[str, Any]:
        """Calculate fairness metrics."""
        df = pd.DataFrame({
            'pred': predictions,
            'label': labels,
            'group': sensitive_attr
        })
        
        demographic_parity = {}
        equal_opportunity = {}
        equalized_odds = {}
        
        for group_value, group_df in df.groupby('group'):
            group_str = str(group_value)
            
            # Demographic parity (acceptance rate)
            acceptance_rate = group_df['pred'].mean()
            demographic_parity[group_str] = float(acceptance_rate)
            
            # Equal opportunity (TPR)
            positives = group_df[group_df['label'] == 1]
            if len(positives) > 0:
                tpr = positives['pred'].mean()
                equal_opportunity[group_str] = float(tpr)
            else:
                equal_opportunity[group_str] = 0.0
            
            # Equalized odds (TPR and FPR)
            negatives = group_df[group_df['label'] == 0]
            if len(negatives) > 0:
                fpr = negatives['pred'].mean()
            else:
                fpr = 0.0
            
            equalized_odds[group_str] = {
                "tpr": equal_opportunity[group_str],
                "fpr": float(fpr)
            }
        
        # Calculate overall gaps
        dp_values = list(demographic_parity.values())
        eo_values = list(equal_opportunity.values())
        
        overall_metrics = {
            "demographic_parity_gap": float(max(dp_values) - min(dp_values)) if dp_values else 0.0,
            "equal_opportunity_gap": float(max(eo_values) - min(eo_values)) if eo_values else 0.0
        }
        
        return {
            "success": True,
            "demographic_parity": demographic_parity,
            "equal_opportunity": equal_opportunity,
            "equalized_odds": equalized_odds,
            "overall_metrics": overall_metrics
        }


# ============================================================================
# Counterfactual Generator Tool
# ============================================================================

class CounterfactualGeneratorInput(ToolInput):
    """Input for counterfactual generation."""
    features: Dict[str, List[Any]] = Field(description="Feature data")
    sensitive_attr_col: str = Field(description="Sensitive attribute column to flip")
    sample_size: int = Field(100, description="Number of samples to generate counterfactuals for")


class CounterfactualGeneratorOutput(ToolOutput):
    """Output from counterfactual generation."""
    counterfactual_data: Dict[str, List[Any]] = Field(default_factory=dict, description="Counterfactual features")
    original_indices: List[int] = Field(default_factory=list, description="Original sample indices")
    flipped_values: Dict[int, Any] = Field(default_factory=dict, description="Flipped attribute values")


class CounterfactualGeneratorTool(Tool):
    """Tool for generating counterfactual examples."""
    
    name = "counterfactual_generator"
    description = "Generate counterfactual examples by flipping sensitive attributes"
    input_schema = CounterfactualGeneratorInput
    output_schema = CounterfactualGeneratorOutput
    
    def __init__(self):
        super().__init__()
        ToolRegistry.register(self)
    
    def _run(self, features: Dict[str, List[Any]], sensitive_attr_col: str,
             sample_size: int) -> Dict[str, Any]:
        """Generate counterfactual examples."""
        df = pd.DataFrame(features)
        
        # Sample random rows
        sample_size = min(sample_size, len(df))
        sampled_indices = np.random.choice(len(df), size=sample_size, replace=False)
        df_sample = df.iloc[sampled_indices].copy()
        
        # Get unique values of sensitive attribute
        unique_values = df[sensitive_attr_col].unique()
        
        if len(unique_values) < 2:
            return {
                "success": False,
                "error": "Sensitive attribute must have at least 2 unique values",
                "counterfactual_data": {},
                "original_indices": [],
                "flipped_values": {}
            }
        
        # Flip sensitive attribute to a different value
        flipped_values = {}
        for idx in range(len(df_sample)):
            original_value = df_sample.iloc[idx][sensitive_attr_col]
            # Pick a different value
            other_values = [v for v in unique_values if v != original_value]
            if other_values:
                flipped_value = np.random.choice(other_values)
                df_sample.iloc[idx, df_sample.columns.get_loc(sensitive_attr_col)] = flipped_value
                flipped_values[int(sampled_indices[idx])] = flipped_value
        
        return {
            "success": True,
            "counterfactual_data": df_sample.to_dict(orient='list'),
            "original_indices": sampled_indices.tolist(),
            "flipped_values": flipped_values
        }
