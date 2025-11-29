# Quick Setup Guide - Enhanced Bias Auditor

## Prerequisites
- Python 3.9+
- Gemini API key (optional but recommended)

## Installation

```powershell
# Navigate to project
cd C:\Users\Mehul\.gemini\antigravity\scratch\bias-auditor

# Install dependencies
pip install -r requirements.txt

# Configure Gemini API (optional)
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Running the System

### Option 1: Full Setup (with MCP Server)

**Terminal 1 - MCP Server:**
```powershell
cd mcp_server
python server.py
```

**Terminal 2 - Backend:**
```powershell
cd backend
python main.py
```

**Terminal 3 - Frontend:**
```powershell
cd frontend
streamlit run app.py
```

### Option 2: Quick Start (Backend + Frontend only)

**Terminal 1 - Backend:**
```powershell
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
streamlit run app.py
```

## Generate Test Data

```powershell
python scripts\generate_test_data.py
```

## Run Your First Audit

1. Open browser at http://localhost:8501
2. Click "New Audit"
3. Upload files from `test_data/`:
   - raw_data.csv
   - processed_features.csv
   - model.pkl
4. Configure:
   - Target: `label`
   - Sensitive Attributes: `gender`, `race`
5. Click "Start Audit"
6. View results in "Run History"

## View Observability Data

```powershell
# View logs
type data\logs\{run_id}.jsonl

# View traces
type data\traces\{run_id}.json

# View metrics
type data\metrics\{run_id}.json
```

## Key Features

✅ **Observability** - Logs, traces, metrics in `data/`  
✅ **Custom Tools** - 4 specialized bias detection tools  
✅ **MCP Server** - Bias pattern knowledge base (optional)  
✅ **Gemini AI** - Enhanced explanations (requires API key)  

## Troubleshooting

### Gemini Not Working
- Check `.env` file has valid `GEMINI_API_KEY`
- System will fall back to rule-based explanations

### MCP Server Not Starting
- MCP server is optional
- System works without it

### Port Already in Use
```powershell
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill the process or change port in main.py
```

## Next Steps

- Read [README.md](file:///C:/Users/Mehul/.gemini/antigravity/scratch/bias-auditor/README.md) for full documentation
- Review [walkthrough.md](file:///C:/Users/Mehul/.gemini/antigravity/brain/ca53fddf-75ec-4bb5-943a-eec7da6fbff9/walkthrough.md) for architecture details
- Try with your own ML pipeline!
