"""
FastAPI Backend for Bias Auditor.

Endpoints:
- POST /runs - Create new audit run
- GET /runs - List all runs
- GET /runs/{run_id} - Get run details
- GET /runs/{run_id}/artifacts/{artifact_type} - Get JSON artifact
- GET /runs/{run_id}/plots/{plot_name} - Get plot image
"""
import uuid
import json
import threading
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from database import RunDB, RunStatus
from models import RunConfig, RunResponse, RunListItem
from orchestrator import run_audit
from utils import get_run_dir, get_artifact_path, get_plots_dir

app = FastAPI(title="Bias Auditor API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Bias Auditor API", "version": "1.0.0"}


@app.post("/runs", response_model=RunResponse)
async def create_run(
    raw_data: UploadFile = File(...),
    processed_features: UploadFile = File(...),
    model: UploadFile = File(...),
    config: str = Form(...)
):
    """
    Create a new audit run.
    
    Args:
        raw_data: CSV file with raw data
        processed_features: CSV file with processed features
        model: Pickled model file
        config: JSON string with run configuration
    """
    # Parse config
    try:
        config_dict = json.loads(config)
        run_config = RunConfig(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
    
    # Generate run ID
    run_id = str(uuid.uuid4())
    
    # Create run directory
    run_dir = get_run_dir(run_id)
    
    # Save uploaded files
    try:
        # Save raw data
        raw_data_path = run_dir / "raw_data.csv"
        with open(raw_data_path, "wb") as f:
            f.write(await raw_data.read())
        
        # Save processed features
        features_path = run_dir / "processed_features.csv"
        with open(features_path, "wb") as f:
            f.write(await processed_features.read())
        
        # Save model
        model_path = run_dir / "model.pkl"
        with open(model_path, "wb") as f:
            f.write(await model.read())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")
    
    # Create database entry
    try:
        RunDB.create(run_id, run_config.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create run: {str(e)}")
    
    # Start audit in background thread
    thread = threading.Thread(target=run_audit, args=(run_id,), daemon=True)
    thread.start()
    
    # Return run info
    run = RunDB.get(run_id)
    return RunResponse(
        id=run["id"],
        status=run["status"],
        config=RunConfig(**run["config"]),
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        error_message=run["error_message"]
    )


@app.get("/runs", response_model=List[RunListItem])
def list_runs():
    """List all audit runs."""
    runs = RunDB.list_all()
    
    result = []
    for run in runs:
        # Try to get primary origin from report if complete
        primary_origin = None
        if run["status"] == RunStatus.COMPLETE:
            try:
                report_path = get_artifact_path(run["id"], "bias_origin_report.json")
                if report_path.exists():
                    with open(report_path, "r") as f:
                        report = json.load(f)
                        primary_origin = report["bias_origin_verdict"]["primary_origin"]
            except Exception:
                pass
        
        result.append(RunListItem(
            id=run["id"],
            status=run["status"],
            created_at=run["created_at"],
            primary_origin=primary_origin
        ))
    
    return result


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str):
    """Get details for a specific run."""
    run = RunDB.get(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return RunResponse(
        id=run["id"],
        status=run["status"],
        config=RunConfig(**run["config"]),
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        error_message=run["error_message"]
    )


@app.get("/runs/{run_id}/artifacts/{artifact_type}")
def get_artifact(run_id: str, artifact_type: str):
    """
    Get a JSON artifact for a run.
    
    Args:
        run_id: Run identifier
        artifact_type: One of: data_bias, feature_bias, model_bias, bias_origin_report
    """
    # Validate run exists
    run = RunDB.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Map artifact type to filename
    artifact_files = {
        "data_bias": "data_bias.json",
        "feature_bias": "feature_bias.json",
        "model_bias": "model_bias.json",
        "bias_origin_report": "bias_origin_report.json"
    }
    
    if artifact_type not in artifact_files:
        raise HTTPException(status_code=400, detail="Invalid artifact type")
    
    artifact_path = get_artifact_path(run_id, artifact_files[artifact_type])
    
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
    
    return JSONResponse(content=data)


@app.get("/runs/{run_id}/plots/{plot_name}")
def get_plot(run_id: str, plot_name: str):
    """
    Get a plot image for a run.
    
    Args:
        run_id: Run identifier
        plot_name: Plot filename (e.g., data_gender.png)
    """
    # Validate run exists
    run = RunDB.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Get plot path
    plot_path = get_plots_dir(run_id) / plot_name
    
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    return FileResponse(plot_path, media_type="image/png")


@app.get("/runs/{run_id}/plots")
def list_plots(run_id: str):
    """List all available plots for a run."""
    # Validate run exists
    run = RunDB.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    plots_dir = get_plots_dir(run_id)
    
    if not plots_dir.exists():
        return {"plots": []}
    
    plots = [p.name for p in plots_dir.glob("*.png")]
    
    return {"plots": plots}


@app.get("/runs/{run_id}/report/pdf")
def download_pdf_report(run_id: str):
    """
    Generate and download PDF report for a run.
    
    Args:
        run_id: Run identifier
    """
    # Validate run exists
    run = RunDB.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Check if run is complete
    if run["status"] != RunStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Run is not complete yet")
    
    # Generate PDF
    try:
        from pdf_generator import generate_pdf_report
        
        pdf_path = get_run_dir(run_id) / f"bias_audit_report_{run_id}.pdf"
        generate_pdf_report(run_id, pdf_path)
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"bias_audit_report_{run_id}.pdf"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
