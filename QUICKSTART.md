# Quick Start Guide - Bias Auditor

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```powershell
cd C:\Users\Mehul\.gemini\antigravity\scratch\bias-auditor
pip install -r requirements.txt
```

### Step 2: Generate Test Data
```powershell
python scripts\generate_test_data.py
```

This creates test files in `test_data/`:
- `raw_data.csv` - 1,000 samples with bias
- `processed_features.csv` - Engineered features
- `model.pkl` - Trained model

### Step 3: Start the Application

**Terminal 1 - Backend:**
```powershell
cd C:\Users\Mehul\.gemini\antigravity\scratch\bias-auditor\backend
python main.py
```
Wait for: `INFO: Uvicorn running on http://0.0.0.0:8000`

**Terminal 2 - Frontend:**
```powershell
cd frontend
streamlit run app.py
```
Browser will open at: `http://localhost:8501`

---

## 📝 Running Your First Audit

1. In the Streamlit UI, click **"New Audit"**
2. Upload the test files:
   - Raw Data: `test_data/raw_data.csv`
   - Processed Features: `test_data/processed_features.csv`
   - Model: `test_data/model.pkl`
3. Configure:
   - Target Column: `label`
   - Sensitive Attributes: `gender`, `race`
4. Click **"🚀 Start Audit"**
5. Navigate to **"Run History"** to view results

---

## 🎯 Expected Results

The test data has intentional bias, so you should see:

**Primary Origin: DATA** 🔴

**Checkpoint Summary:**
- Data: **Fail** (score ~0.9)
  - Underrepresented Black group (5%)
  - Label imbalance ratio 4:1
- Features: **Warning** (score ~0.6)
  - Proxy features detected (gender__, race__)
- Model: **Fail** (score ~0.8)
  - Demographic parity violations
  - Equal opportunity violations

**Recommendations:**
- Collect more samples from underrepresented groups
- Remove proxy features
- Apply fairness constraints

---

## 🔧 Troubleshooting

### Backend won't start
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# If in use, kill the process or change port in main.py
```

### Frontend can't connect
- Ensure backend is running at `http://localhost:8000`
- Check browser console for errors
- Try refreshing the page

### Import errors
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📚 Next Steps

- Read the full [README.md](file:///C:/Users/Mehul/.gemini/antigravity/scratch/bias-auditor/README.md)
- Review the [walkthrough](file:///C:/Users/Mehul/.gemini/antigravity/brain/27661ea7-90ef-4795-8abf-f8e5d4c4276c/walkthrough.md)
- Try with your own ML pipeline!

---

**Ready to audit bias? Let's go!** 🚀
