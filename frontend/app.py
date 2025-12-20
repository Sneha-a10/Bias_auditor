"""
Streamlit Frontend for Bias Auditor.

Features:
- File upload interface
- Configuration panel
- Run list and status
- Detailed run view with tabs
- Visualization rendering
"""
import streamlit as st
import requests
import json
import time
import pandas as pd
from typing import Dict, Any, Optional
from io import BytesIO

# Backend API URL
API_URL = "http://localhost:8001"

# Page config
st.set_page_config(
    page_title="Bias Auditor",
    page_icon="🔍",
    layout="wide"
)


def main():
    """Main application."""
    st.title("🔍 Agentic Bias Checkpoint Auditor")
    st.markdown("Identify where bias originates in your ML pipeline: Data → Features → Model")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["New Audit", "Run History", "About"]
    )
    
    if page == "New Audit":
        new_audit_page()
    elif page == "Run History":
        run_history_page()
    else:
        about_page()


def new_audit_page():
    """Page for creating a new audit."""
    st.header("Start New Audit")
    
    st.markdown("""
    Upload your ML pipeline artifacts and configure the audit parameters.
    The system will analyze bias at three checkpoints: **Data**, **Features**, and **Model**.
    """)
    
    # File uploads
    st.subheader("1. Upload Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        raw_data_file = st.file_uploader(
            "Raw Data (CSV)",
            type=["csv"],
            help="CSV file with raw data including sensitive attributes and target"
        )
    
    with col2:
        features_file = st.file_uploader(
            "Processed Features (CSV)",
            type=["csv"],
            help="CSV file with engineered features (row-aligned with raw data)"
        )
    
    with col3:
        model_file = st.file_uploader(
            "Model (PKL)",
            type=["pkl"],
            help="Pickled scikit-learn compatible model"
        )
    
    # Configuration
    if raw_data_file:
        st.subheader("2. Configure Audit")
        
        # Read CSV to get column names
        raw_df = pd.read_csv(raw_data_file)
        raw_data_file.seek(0)  # Reset file pointer
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Target Column",
                options=raw_df.columns.tolist(),
                help="Binary classification target (0/1)"
            )
        
        with col2:
            sensitive_attrs = st.multiselect(
                "Sensitive Attributes",
                options=[col for col in raw_df.columns if col != target_column],
                help="Attributes to check for bias (e.g., gender, race, age)"
            )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Fairness Thresholds**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dp_threshold = st.slider(
                    "Demographic Parity Threshold",
                    0.0, 0.5, 0.1, 0.01,
                    help="Max acceptable difference in acceptance rates"
                )
                
                eo_threshold = st.slider(
                    "Equal Opportunity Threshold",
                    0.0, 0.5, 0.1, 0.01,
                    help="Max acceptable difference in TPR"
                )
                
                proxy_threshold = st.slider(
                    "Proxy Correlation Threshold",
                    0.0, 1.0, 0.3, 0.05,
                    help="Min correlation to flag as proxy feature"
                )
            
            with col2:
                min_group_prop = st.slider(
                    "Min Group Proportion",
                    0.0, 0.2, 0.05, 0.01,
                    help="Min proportion to avoid underrepresentation flag"
                )
                
                label_imbalance_ratio = st.slider(
                    "Label Imbalance Ratio",
                    1.0, 10.0, 4.0, 0.5,
                    help="Max acceptable ratio of majority to minority class"
                )
                
                cf_threshold = st.slider(
                    "Counterfactual Change Threshold",
                    0.0, 0.5, 0.1, 0.01,
                    help="Min prediction change rate to flag sensitivity"
                )
        
        # Build config
        config = {
            "target_column": target_column,
            "sensitive_attributes": [
                {"name": attr, "expected_categories": None}
                for attr in sensitive_attrs
            ],
            "fairness_thresholds": {
                "demographic_parity_diff_threshold": dp_threshold,
                "equal_opportunity_diff_threshold": eo_threshold,
                "equalized_odds_diff_threshold": eo_threshold,
                "min_group_proportion_threshold": min_group_prop,
                "min_support_for_metrics": 30,
                "proxy_corr_threshold": proxy_threshold,
                "counterfactual_change_threshold": cf_threshold,
                "label_imbalance_ratio_threshold": label_imbalance_ratio
            }
        }
        
        # Submit button
        st.subheader("3. Run Audit")
        
        if st.button("🚀 Start Audit", type="primary", disabled=not all([raw_data_file, features_file, model_file, sensitive_attrs])):
            if not sensitive_attrs:
                st.error("Please select at least one sensitive attribute")
            else:
                with st.spinner("Uploading files and starting audit..."):
                    run_id = create_run(raw_data_file, features_file, model_file, config)
                    
                    if run_id:
                        st.success(f"✅ Audit started! Run ID: {run_id}")
                        st.info("Navigate to 'Run History' to view progress and results.")
                        time.sleep(2)
                        st.rerun()


def run_history_page():
    """Page showing run history and details."""
    st.header("Run History")
    
    # Fetch runs
    try:
        response = requests.get(f"{API_URL}/runs")
        response.raise_for_status()
        runs = response.json()
    except Exception as e:
        st.error(f"Failed to fetch runs: {str(e)}")
        return
    
    if not runs:
        st.info("No audit runs yet. Start a new audit to get started!")
        return
    
    # Display runs table
    runs_df = pd.DataFrame([
        {
            "Run ID": run["id"][:8] + "...",
            "Status": run["status"],
            "Primary Origin": run.get("primary_origin", "N/A"),
            "Created": run["created_at"][:19].replace("T", " ")
        }
        for run in runs
    ])
    
    st.dataframe(runs_df, use_container_width=True)
    
    # Run selector
    st.subheader("Run Details")
    
    run_ids = {f"{run['id'][:8]}... ({run['status']})": run['id'] for run in runs}
    selected_display = st.selectbox("Select Run", options=list(run_ids.keys()))
    selected_run_id = run_ids[selected_display]
    
    # Fetch run details
    try:
        response = requests.get(f"{API_URL}/runs/{selected_run_id}")
        response.raise_for_status()
        run_details = response.json()
    except Exception as e:
        st.error(f"Failed to fetch run details: {str(e)}")
        return
    
    # Display status
    status = run_details["status"]
    
    if status == "PENDING":
        st.info("⏳ Run is pending...")
    elif status == "RUNNING":
        st.warning("🔄 Run is in progress...")
        if st.button("🔄 Refresh"):
            st.rerun()
    elif status == "FAILED":
        st.error("❌ Run failed")
        if run_details.get("error_message"):
            with st.expander("Error Details"):
                st.code(run_details["error_message"])
    elif status == "COMPLETE":
        st.success("✅ Run complete")
        
        # Add PDF download button
        col1, col2 = st.columns([3, 1])
        with col2:
            pdf_url = f"{API_URL}/runs/{selected_run_id}/report/pdf"
            st.markdown(
                f'<a href="{pdf_url}" download><button style="background-color:#28a745;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;width:100%">📄 Download PDF Report</button></a>',
                unsafe_allow_html=True
            )
        
        display_run_results(selected_run_id)
    
    # Auto-refresh for running jobs
    if status in ["PENDING", "RUNNING"]:
        time.sleep(2)
        st.rerun()


def display_run_results(run_id: str):
    """Display detailed results for a completed run."""
    
    # Fetch all artifacts
    try:
        data_bias = fetch_artifact(run_id, "data_bias")
        feature_bias = fetch_artifact(run_id, "feature_bias")
        model_bias = fetch_artifact(run_id, "model_bias")
        report = fetch_artifact(run_id, "bias_origin_report")
    except Exception as e:
        st.error(f"Failed to fetch artifacts: {str(e)}")
        return
    
    # Display verdict at top
    st.markdown("---")
    st.subheader("🎯 Bias Origin Verdict")
    
    verdict = report["bias_origin_verdict"]
    primary_origin = verdict["primary_origin"]
    
    # Color-code origin
    origin_colors = {
        "DATA": "🔴",
        "FEATURE": "🟡",
        "MODEL": "🟢",
        "MULTIPLE/UNCLEAR": "🟣"
    }
    
    st.markdown(f"### {origin_colors.get(primary_origin, '⚪')} Primary Origin: **{primary_origin}**")
    st.info(verdict["explanation"])
    
    # Checkpoint summary
    st.markdown("#### Checkpoint Summary")
    
    checkpoint_df = pd.DataFrame(report["checkpoint_summary"])
    
    # Style the dataframe
    def color_status(val):
        colors = {
            "Pass": "background-color: #d4edda",
            "Warning": "background-color: #fff3cd",
            "Fail": "background-color: #f8d7da"
        }
        return colors.get(val, "")
    
    styled_df = checkpoint_df.style.applymap(color_status, subset=["status"])
    st.dataframe(styled_df, use_container_width=True)
    
    # Tabs for detailed views
    st.markdown("---")
    tabs = st.tabs(["📊 Data", "🔧 Features", "🤖 Model", "💡 Recommendations"])
    
    with tabs[0]:
        display_data_tab(run_id, data_bias)
    
    with tabs[1]:
        display_features_tab(run_id, feature_bias)
    
    with tabs[2]:
        display_model_tab(run_id, model_bias)
    
    with tabs[3]:
        display_recommendations_tab(report)


def display_data_tab(run_id: str, data_bias: Dict[str, Any]):
    """Display data bias analysis."""
    st.subheader("Data Bias Analysis")
    
    overall = data_bias["overall"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Bias Score", f"{overall['data_bias_score']:.2f}")
    
    with col2:
        st.metric("Bias Flag", "🚨 Yes" if overall["data_bias_flag"] else "✅ No")
    
    with col3:
        ratio = overall["global_label_counts"]["ratio"]
        st.metric("Label Imbalance Ratio", f"{ratio:.2f}")
    
    st.markdown(f"**Summary:** {overall['summary']}")
    
    # Subgroup statistics
    st.markdown("#### Subgroup Analysis")
    
    for attr_name, attr_data in data_bias["sensitive_attributes"].items():
        st.markdown(f"**{attr_name}**")
        
        subgroups_df = pd.DataFrame(attr_data["subgroups"])
        st.dataframe(subgroups_df, use_container_width=True)
        
        # Display plot
        plot_name = f"data_{attr_name}.png"
        display_plot(run_id, plot_name)
    
    # Label distribution
    st.markdown("#### Label Distribution")
    display_plot(run_id, "data_label_distribution.png")


def display_features_tab(run_id: str, feature_bias: Dict[str, Any]):
    """Display feature bias analysis."""
    st.subheader("Feature Bias Analysis")
    
    flags = feature_bias["flags"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Bias Score", f"{flags['feature_bias_score']:.2f}")
    
    with col2:
        st.metric("Bias Flag", "🚨 Yes" if flags["feature_bias_flag"] else "✅ No")
    
    st.markdown(f"**Summary:** {flags['summary']}")
    
    # Proxy features
    st.markdown("#### Proxy Features")
    
    proxy_features = feature_bias.get("proxy_features", {})
    if any(proxy_features.values()):
        for attr, features in proxy_features.items():
            if features:
                st.warning(f"**{attr}**: {', '.join(features[:10])}{'...' if len(features) > 10 else ''}")
    else:
        st.success("No proxy features detected")
    
    # Encoding risks
    st.markdown("#### Encoding Risks")
    
    encoding_risks = feature_bias.get("encoding_risks", [])
    if encoding_risks:
        for risk in encoding_risks:
            st.warning(f"**{risk['type']}**: {risk['description']}")
    else:
        st.success("No encoding risks detected")
    
    # Heatmap
    st.markdown("#### Feature-Attribute Association Heatmap")
    display_plot(run_id, "feature_heatmap.png")


def display_model_tab(run_id: str, model_bias: Dict[str, Any]):
    """Display model bias analysis."""
    st.subheader("Model Bias Analysis")
    
    flags = model_bias["flags"]
    global_metrics = model_bias["global_metrics"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Bias Score", f"{flags['model_bias_score']:.2f}")
    
    with col2:
        st.metric("Accuracy", f"{global_metrics['accuracy']:.3f}")
    
    with col3:
        if "auc" in global_metrics:
            st.metric("AUC", f"{global_metrics['auc']:.3f}")
    
    st.markdown(f"**Summary:** {flags['summary']}")
    
    # Per-attribute analysis
    st.markdown("#### Fairness Analysis by Sensitive Attribute")
    
    for attr_name, attr_data in model_bias["by_sensitive_attribute"].items():
        st.markdown(f"**{attr_name}**")
        
        # Subgroup metrics
        subgroups_df = pd.DataFrame(attr_data["subgroups"])
        st.dataframe(subgroups_df, use_container_width=True)
        
        # Fairness gaps
        gaps = attr_data.get("fairness_gaps", {})
        if gaps:
            st.markdown("**Fairness Gaps:**")
            gaps_df = pd.DataFrame([gaps])
            st.dataframe(gaps_df, use_container_width=True)
        
        # Fairness flags
        fairness_flags = attr_data.get("fairness_flags", {})
        if any(fairness_flags.values()):
            flagged = [k for k, v in fairness_flags.items() if v]
            st.error(f"Violations: {', '.join(flagged)}")
        
        # Display plot
        plot_name = f"model_{attr_name}_metrics.png"
        display_plot(run_id, plot_name)
    
    # Counterfactual
    st.markdown("#### Counterfactual Testing")
    
    cf = model_bias["counterfactual"]
    st.markdown(cf["summary"])
    
    if cf.get("change_rate_per_attribute"):
        cf_df = pd.DataFrame([
            {"Attribute": k, "Change Rate": f"{v:.1%}"}
            for k, v in cf["change_rate_per_attribute"].items()
        ])
        st.dataframe(cf_df, use_container_width=True)


def display_recommendations_tab(report: Dict[str, Any]):
    """Display recommended fixes."""
    st.subheader("Recommended Fixes")
    
    fixes = report.get("recommended_fixes", {})
    
    if not fixes:
        st.info("No specific recommendations available.")
        return
    
    for stage, recommendations in fixes.items():
        if recommendations:
            st.markdown(f"### {stage}")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")


def display_plot(run_id: str, plot_name: str):
    """Display a plot image."""
    try:
        response = requests.get(f"{API_URL}/runs/{run_id}/plots/{plot_name}")
        if response.status_code == 200:
            st.image(response.content, use_column_width=True)
        else:
            st.warning(f"Plot not available: {plot_name}")
    except Exception as e:
        st.warning(f"Failed to load plot: {plot_name}")


def fetch_artifact(run_id: str, artifact_type: str) -> Dict[str, Any]:
    """Fetch a JSON artifact from the API."""
    response = requests.get(f"{API_URL}/runs/{run_id}/artifacts/{artifact_type}")
    response.raise_for_status()
    return response.json()


def create_run(raw_data, features, model, config) -> Optional[str]:
    """Create a new audit run."""
    try:
        files = {
            "raw_data": ("raw_data.csv", raw_data, "text/csv"),
            "processed_features": ("processed_features.csv", features, "text/csv"),
            "model": ("model.pkl", model, "application/octet-stream")
        }
        
        data = {
            "config": json.dumps(config)
        }
        
        response = requests.post(f"{API_URL}/runs", files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        return result["id"]
    
    except Exception as e:
        st.error(f"Failed to create run: {str(e)}")
        return None


def about_page():
    """About page."""
    st.header("About Bias Auditor")
    
    st.markdown("""
    ## 🎯 Purpose
    
    The **Agentic Bias Checkpoint Auditor** identifies where bias originates in your ML pipeline
    by analyzing three critical checkpoints:
    
    1. **Data** - Raw training data
    2. **Features** - Engineered features
    3. **Model** - Trained model predictions
    
    ## 🔍 What It Detects
    
    ### Data Checkpoint
    - Underrepresented demographic groups
    - Missing expected categories
    - Severe label imbalance
    
    ### Feature Checkpoint
    - Proxy features (correlated with sensitive attributes)
    - Target leakage
    - Sparse one-hot encoding issues
    
    ### Model Checkpoint
    - Demographic parity violations
    - Equal opportunity violations
    - Equalized odds violations
    - Counterfactual sensitivity
    
    ## 📊 How It Works
    
    1. Upload your pipeline artifacts (data, features, model)
    2. Configure sensitive attributes and thresholds
    3. Four specialized agents analyze each checkpoint
    4. Get a verdict on where bias originates
    5. Receive actionable recommendations
    
    ## 🛠️ Technology Stack
    
    - **Backend**: FastAPI + Python
    - **Frontend**: Streamlit
    - **ML**: scikit-learn
    - **Metrics**: Fairness-aware statistical analysis
    
    ## 📝 Version
    
    v1.0.0 - MVP Release
    """)


if __name__ == "__main__":
    main()
