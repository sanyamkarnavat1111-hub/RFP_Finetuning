# inference_with_comparison_FIXED.py

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.stats import pearsonr

# ===================== CONFIG =====================
MODEL_PATH = "./saved_model_regression"
CSV_PATH = "/mnt/d/RFP_Finetuning/Backup_files/Dataset/extracted_file_normalized.csv"
OUTPUT_CSV = "/mnt/d/RFP_Finetuning/Backup_files/Dataset/extracted_file_with_predictions.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ===================== LOAD MODEL + RE-APPLY SIGMOID =====================
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# THIS IS THE KEY FIX — RE-APPLY SIGMOID AFTER LOADING!
model.classifier = nn.Sequential(
    model.classifier,
    nn.Sigmoid()
)
print("Sigmoid re-applied → predictions now guaranteed 0.0 – 1.0")

model.to(DEVICE)
model.eval()

# Load dataset
df = pd.read_csv(CSV_PATH)
assert "score_normalized" in df.columns

# ===================== PREDICTION FUNCTION =====================
def predict_normalized(ea: str, rfp: str) -> float:
    text = f"EA_Requirement: {ea} [SEP] RFP_Coverage: {rfp}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        logit = model(**inputs).logits.squeeze().item()   # Now 0.0 – 1.0 thanks to Sigmoid!
    return logit

# ===================== BATCH INFERENCE =====================
print("Running inference...")
predictions = []
for idx, row in df.iterrows():
    pred = predict_normalized(row["EA_Requirement"], row["RFP_Coverage"])
    predictions.append(pred)
    if (idx + 1) % 50 == 0:
        print(f"   Processed {idx + 1}/{len(df)}")

df["predicted_normalized"] = predictions
df["predicted_score_pct"] = (df["predicted_normalized"] * 100).round(2)
df["true_score_pct"] = (df["score_normalized"] * 100).round(2)

# ===================== METRICS =====================
mae = np.mean(np.abs(df["predicted_normalized"] - df["score_normalized"]))
rmse = np.sqrt(np.mean((df["predicted_normalized"] - df["score_normalized"]) ** 2))
pearson_corr, _ = pearsonr(df["score_normalized"], df["predicted_normalized"])

print("\n" + "="*60)
print("FINAL RESULTS (Sigmoid Fixed!)")
print("="*60)
print(f"   Rows: {len(df)}")
print(f"   MAE : {mae:.4f} ({mae*100:.2f}%)")
print(f"   RMSE: {rmse:.4f}")
print(f"   Pearson: {pearson_corr:.4f}")
print(f"   Min predicted: {df['predicted_score_pct'].min():.2f}%")
print(f"   Max predicted: {df['predicted_score_pct'].max():.2f}%")  # ← Will now be ≤ 100.00
print("="*60)

df.drop(columns=["predicted_normalized"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")