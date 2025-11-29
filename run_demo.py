"""
Demo script to run the Bias Auditor programmatically.
Demonstrates observability, custom tools, and Gemini integration.
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from database import RunDB, init_db
from orchestrator import run_audit
import uuid
import json
import shutil

# Initialize database
init_db()

print("="*70)
print("BIAS CHECKPOINT AUDITOR - DEMO RUN")
print("="*70)
print()

# Create run configuration
run_id = str(uuid.uuid4())[:8]
print(f"Run ID: {run_id}")
print()

config = {
    "target_column": "label",
    "sensitive_attributes": [
        {"name": "gender", "expected_groups": ["Male", "Female"]},
        {"name": "race", "expected_groups": ["White", "Black", "Hispanic", "Asian"]}
    ],
    "fairness_thresholds": {
        "demographic_parity_diff_threshold": 0.1,
        "equal_opportunity_diff_threshold": 0.1,
        "equalized_odds_diff_threshold": 0.1,
        "min_group_proportion_threshold": 0.05,
        "min_support_for_metrics": 30,
        "proxy_corr_threshold": 0.3,
        "counterfactual_change_threshold": 0.1,
        "label_imbalance_ratio_threshold": 4.0
    }
}

# Create run in database
RunDB.create(run_id, config)
print("✓ Created run in database")

# Copy test data to run directory
test_data_dir = Path(__file__).parent / "test_data"
data_dir = Path(__file__).parent / "data" / run_id  # Changed from data/runs/{run_id}
data_dir.mkdir(parents=True, exist_ok=True)

shutil.copy(test_data_dir / "raw_data.csv", data_dir / "raw_data.csv")
shutil.copy(test_data_dir / "processed_features.csv", data_dir / "processed_features.csv")
shutil.copy(test_data_dir / "model.pkl", data_dir / "model.pkl")
print("✓ Copied test data to run directory")
print()

# Save config
config_path = data_dir / "config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# Create artifacts directory
artifacts_dir = data_dir / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

print("-"*70)
print("RUNNING AUDIT PIPELINE")
print("-"*70)
print()

# Run the audit
try:
    run_audit(run_id)
    print()
    print("="*70)
    print("✓ AUDIT COMPLETE!")
    print("="*70)
    print()
    
    # Show observability outputs
    print("Observability Outputs:")
    print(f"  - Logs: data/logs/{run_id}.jsonl")
    print(f"  - Traces: data/traces/{run_id}.json")
    print(f"  - Metrics: data/metrics/{run_id}.json")
    print()
    
    # Show results
    results_path = artifacts_dir / "bias_origin_report.json"
    if results_path.exists():
        with open(results_path) as f:
            report = json.load(f)
        
        print("-"*70)
        print("BIAS ORIGIN VERDICT")
        print("-"*70)
        print()
        print(f"Primary Origin: {report['bias_origin_verdict']['primary_origin']}")
        print()
        print("Explanation:")
        print(report['bias_origin_verdict']['explanation'])
        print()
        
        print("-"*70)
        print("CHECKPOINT SUMMARY")
        print("-"*70)
        for checkpoint in report['checkpoint_summary']:
            print(f"\n{checkpoint['stage']}: {checkpoint['status']} (score: {checkpoint['score']:.2f})")
            if checkpoint['flagged_issues']:
                print(f"  Issues: {', '.join(checkpoint['flagged_issues'])}")
        print()
        
        print("-"*70)
        print("RECOMMENDED FIXES")
        print("-"*70)
        for stage, fixes in report['recommended_fixes'].items():
            if fixes:
                print(f"\n{stage}:")
                for fix in fixes:
                    print(f"  • {fix}")
        print()
    
    # Show metrics
    metrics_path = Path(__file__).parent / "data" / "metrics" / f"{run_id}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print("-"*70)
        print("EXECUTION METRICS")
        print("-"*70)
        print()
        if "execution_metrics" in metrics and "execution_times" in metrics["execution_metrics"]:
            print("Agent Execution Times:")
            for agent, time_val in metrics["execution_metrics"]["execution_times"].items():
                print(f"  {agent}: {time_val:.3f}s")
            if "total_duration_seconds" in metrics["execution_metrics"]:
                print(f"  TOTAL: {metrics['execution_metrics']['total_duration_seconds']:.3f}s")
        print()
        
        if "bias_metrics" in metrics and "scores" in metrics["bias_metrics"]:
            print("Bias Scores:")
            for checkpoint, score in metrics["bias_metrics"]["scores"].items():
                print(f"  {checkpoint}: {score:.3f}")
        print()
    
    print("="*70)
    print("View full results in:")
    print(f"  {results_path}")
    print("="*70)
    
except Exception as e:
    print()
    print("="*70)
    print("✗ AUDIT FAILED")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
