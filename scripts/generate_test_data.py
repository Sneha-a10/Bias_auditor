"""
Generate synthetic test data with known bias for testing the Bias Auditor.

This script creates:
- raw_data.csv with underrepresented groups and label imbalance
- processed_features.csv with proxy features
- model.pkl that exhibits fairness violations
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "test_data"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Generating synthetic biased dataset...")

# Parameters
N_SAMPLES = 1000

# Generate raw data with intentional bias
data = {
    "id": range(N_SAMPLES),
    "gender": [],
    "race": [],
    "age_bucket": [],
    "income": [],
    "credit_score": [],
    "employment_years": [],
    "region": [],
    "label": []
}

# Gender: 70% Male, 30% Female (imbalanced)
data["gender"] = np.random.choice(["Male", "Female"], N_SAMPLES, p=[0.7, 0.3])

# Race: Underrepresented groups
race_probs = [0.6, 0.05, 0.25, 0.1]  # White, Black (underrepresented), Hispanic, Asian
data["race"] = np.random.choice(["White", "Black", "Hispanic", "Asian"], N_SAMPLES, p=race_probs)

# Age bucket
data["age_bucket"] = np.random.choice(
    ["18-24", "25-34", "35-44", "45-54", "55+"],
    N_SAMPLES,
    p=[0.15, 0.25, 0.25, 0.2, 0.15]
)

# Region
data["region"] = np.random.choice(["Urban", "Suburban", "Rural"], N_SAMPLES, p=[0.5, 0.3, 0.2])

# Generate features with bias
for i in range(N_SAMPLES):
    gender = data["gender"][i]
    race = data["race"][i]
    
    # Income: biased by gender and race
    base_income = 50000
    if gender == "Male":
        base_income += 15000  # Gender pay gap
    if race == "White":
        base_income += 10000  # Racial income gap
    elif race == "Black":
        base_income -= 5000
    
    income = base_income + np.random.normal(0, 10000)
    data["income"].append(max(20000, income))
    
    # Credit score: correlated with income and race
    base_credit = 650
    if race == "White":
        base_credit += 50
    elif race == "Black":
        base_credit -= 30
    
    credit = base_credit + (income - 50000) / 1000 + np.random.normal(0, 50)
    data["credit_score"].append(max(300, min(850, credit)))
    
    # Employment years
    emp_years = max(0, int(np.random.exponential(5)))
    data["employment_years"].append(emp_years)

# Generate biased labels
# Approval is biased towards males and certain races
for i in range(N_SAMPLES):
    gender = data["gender"][i]
    race = data["race"][i]
    income = data["income"][i]
    credit = data["credit_score"][i]
    
    # Base probability from legitimate factors
    prob = 0.3
    prob += (income - 50000) / 100000  # Income effect
    prob += (credit - 650) / 500  # Credit effect
    
    # Bias: higher approval for males
    if gender == "Male":
        prob += 0.2
    
    # Bias: lower approval for Black applicants
    if race == "Black":
        prob -= 0.25
    
    # Generate label
    label = 1 if np.random.random() < prob else 0
    data["label"].append(label)

# Create DataFrame
df_raw = pd.DataFrame(data)

print(f"Generated {N_SAMPLES} samples")
print(f"Gender distribution: {df_raw['gender'].value_counts().to_dict()}")
print(f"Race distribution: {df_raw['race'].value_counts().to_dict()}")
print(f"Label distribution: {df_raw['label'].value_counts().to_dict()}")
print(f"Approval rate by gender: {df_raw.groupby('gender')['label'].mean().to_dict()}")
print(f"Approval rate by race: {df_raw.groupby('race')['label'].mean().to_dict()}")

# Save raw data
raw_data_path = OUTPUT_DIR / "raw_data.csv"
df_raw.to_csv(raw_data_path, index=False)
print(f"\n[OK] Saved raw_data.csv to {raw_data_path}")

# Generate processed features
print("\nGenerating processed features...")

features = pd.DataFrame()

# One-hot encode gender (proxy feature!)
features["gender__Male"] = (df_raw["gender"] == "Male").astype(int)
features["gender__Female"] = (df_raw["gender"] == "Female").astype(int)

# One-hot encode race (proxy features!)
for race in ["White", "Black", "Hispanic", "Asian"]:
    features[f"race__{race}"] = (df_raw["race"] == race).astype(int)

# One-hot encode age bucket
for age in ["18-24", "25-34", "35-44", "45-54", "55+"]:
    features[f"age_bucket__{age}"] = (df_raw["age_bucket"] == age).astype(int)

# One-hot encode region
for region in ["Urban", "Suburban", "Rural"]:
    features[f"region__{region}"] = (df_raw["region"] == region).astype(int)

# Numeric features (scaled)
scaler = StandardScaler()
numeric_cols = ["income", "credit_score", "employment_years"]
scaled_numeric = scaler.fit_transform(df_raw[numeric_cols])

features["income_scaled"] = scaled_numeric[:, 0]
features["credit_score_scaled"] = scaled_numeric[:, 1]
features["employment_years_scaled"] = scaled_numeric[:, 2]

# Interaction features (some will be proxies)
features["income_x_credit"] = features["income_scaled"] * features["credit_score_scaled"]

# Save processed features
features_path = OUTPUT_DIR / "processed_features.csv"
features.to_csv(features_path, index=False)
print(f"[OK] Saved processed_features.csv to {features_path}")

# Train a biased model
print("\nTraining biased model...")

X = features.values
y = df_raw["label"].values

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

train_accuracy = model.score(X, y)
print(f"Model training accuracy: {train_accuracy:.3f}")

# Check model predictions by group
y_pred = model.predict(X)
df_raw["prediction"] = y_pred

print(f"\nModel acceptance rate by gender:")
for gender in ["Male", "Female"]:
    mask = df_raw["gender"] == gender
    rate = y_pred[mask].mean()
    print(f"  {gender}: {rate:.3f}")

print(f"\nModel acceptance rate by race:")
for race in ["White", "Black", "Hispanic", "Asian"]:
    mask = df_raw["race"] == race
    if mask.sum() > 0:
        rate = y_pred[mask].mean()
        print(f"  {race}: {rate:.3f}")

# Save model
model_path = OUTPUT_DIR / "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\n[OK] Saved model.pkl to {model_path}")

print("\n" + "="*60)
print("[SUCCESS] Test data generation complete!")
print("="*60)
print(f"\nFiles created in: {OUTPUT_DIR}")
print("  - raw_data.csv")
print("  - processed_features.csv")
print("  - model.pkl")
print("\nExpected bias signals:")
print("  [DATA] Underrepresented Black group (5%), label imbalance")
print("  [FEATURES] Proxy features (gender__, race__)")
print("  [MODEL] Demographic parity violations by gender and race")
print("\nUpload these files to the Bias Auditor to see it in action!")
