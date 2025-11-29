"""
Shared utility functions for the Bias Auditor.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def get_run_dir(run_id: str) -> Path:
    """Get the directory for a specific run."""
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_plots_dir(run_id: str) -> Path:
    """Get the plots directory for a specific run."""
    plots_dir = get_run_dir(run_id) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def load_run_config(run_id: str) -> Dict[str, Any]:
    """Load run configuration from database."""
    from database import RunDB
    run = RunDB.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")
    return run["config"]


def load_raw_data(run_id: str) -> pd.DataFrame:
    """Load raw data CSV for a run."""
    path = get_run_dir(run_id) / "raw_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found for run {run_id}")
    return pd.read_csv(path)


def load_features(run_id: str) -> pd.DataFrame:
    """Load processed features CSV for a run."""
    path = get_run_dir(run_id) / "processed_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed features not found for run {run_id}")
    return pd.read_csv(path)


def load_model(run_id: str) -> BaseEstimator:
    """Load pickled model for a run."""
    path = get_run_dir(run_id) / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found for run {run_id}")
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    # Validate that it's actually a model with predict method
    if not hasattr(model, 'predict'):
        raise TypeError(
            f"Loaded object is not a valid model. "
            f"Expected an object with 'predict' method, got {type(model).__name__}. "
            f"Please ensure the uploaded file is a pickled scikit-learn model."
        )
    
    return model


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Save object as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def compute_group_stats(
    df: pd.DataFrame,
    attr: str,
    target: str
) -> List[Dict[str, Any]]:
    """
    Compute statistics for each group in a sensitive attribute.
    
    Returns list of dicts with:
    - value: group value
    - count: number of samples
    - proportion: fraction of total
    - label_rate: mean of target variable
    """
    stats = []
    total = len(df)
    
    for value, group_df in df.groupby(attr):
        count = len(group_df)
        proportion = count / total
        label_rate = float(group_df[target].mean())
        
        stats.append({
            "value": str(value),
            "count": int(count),
            "proportion": float(proportion),
            "label_rate": label_rate
        })
    
    return stats


def compute_confusion_by_group(
    df: pd.DataFrame,
    attr: str,
    y_true_col: str,
    y_pred_col: str
) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrix metrics for each group.
    
    Returns dict mapping group value to confusion metrics:
    - tp, fp, tn, fn
    - tpr, fpr, precision
    """
    results = {}
    
    for value, group_df in df.groupby(attr):
        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values
        
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        # Compute rates with zero-division handling
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        precision = tp / max(1, tp + fp)
        
        results[str(value)] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(precision)
        }
    
    return results


def get_artifact_path(run_id: str, artifact_name: str) -> Path:
    """Get path to a specific artifact file."""
    return get_run_dir(run_id) / artifact_name


def now_iso() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
