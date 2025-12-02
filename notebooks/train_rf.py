#!/usr/bin/env python3
# Random Forest Price Predictor using "intfloat/e5-large"
# Optimized for 1.2 Lakh (120k) samples on GPU 0

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from tqdm import tqdm
from sklearn.base import clone
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------
# CRITICAL: Pin to GPU 0 as requested
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PREFERRED_MODEL = "intfloat/e5-large"
TARGET_SAMPLES = 120000  # <--- 1,20,000 Samples

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "finetune"

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH   = DATA_DIR / "val.jsonl"

OUTPUT_DIR = BASE_DIR / "src" / "models" / "rf_e5_large_120k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Train: {TRAIN_PATH}")
print(f"📂 Val:   {VAL_PATH}")
print(f"🤖 Model: {PREFERRED_MODEL}")
print(f"🎯 Target Training Size: {TARGET_SAMPLES}")

# -------------------------------------------------------------
# 2. Load & Sample Data
# -------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

print("🔁 Loading datasets...")
train_df = load_jsonl(TRAIN_PATH)
val_df = load_jsonl(VAL_PATH)

# Randomly sample 1.2 Lac rows
if len(train_df) > TARGET_SAMPLES:
    print(f"✂️ Downsampling training data from {len(train_df)} to {TARGET_SAMPLES}...")
    train_df = train_df.sample(n=TARGET_SAMPLES, random_state=42)
else:
    print(f"⚠️ Dataset smaller than {TARGET_SAMPLES}, using all {len(train_df)} rows.")

print(f"✅ Final Train rows: {len(train_df)}")
print(f"✅ Final Val rows:   {len(val_df)}")

# -------------------------------------------------------------
# 3. Generate Embeddings (The "e5-large" way)
# -------------------------------------------------------------
print(f"🧠 Loading embedding model: {PREFERRED_MODEL} on GPU 0...")
embedding_model = SentenceTransformer(PREFERRED_MODEL, device="cuda")

# CRITICAL: E5 models require "query: " prefix
print("⚡ Adding 'query: ' prefix for E5 model compatibility...")
train_texts = ["query: " + str(t) for t in train_df["prompt"].tolist()]
val_texts   = ["query: " + str(t) for t in val_df["prompt"].tolist()]

print("🔢 Encoding training set (Batch size 64)...")
X_train = embedding_model.encode(
    train_texts, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("🔢 Encoding validation set...")
X_val = embedding_model.encode(
    val_texts, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"Embedding Shape: {X_train.shape}") 

# -------------------------------------------------------------
# 4. Train Random Forest (Log-Transformed Target)
# -------------------------------------------------------------
train_labels = train_df["response"].astype(float).values
val_labels   = val_df["response"].astype(float).values

# Use Log-Target to handle price skew
y_train_log = np.log1p(train_labels)

print("🌲 Training Random Forest...")

# Optimization: max_features="sqrt"
# This makes training fast even with 1.2 Lac rows and 1024 features
n_estimators = 1000

base_rf = RandomForestRegressor(
    n_estimators=1,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

estimators = []

for i in tqdm(range(n_estimators), desc="Training Trees"):
    rf_tree = clone(base_rf)
    rf_tree.random_state = 42 + i
    rf_tree.fit(X_train, y_train_log)
    estimators.append(rf_tree.estimators_[0])

# Final Wrapper
rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_features="sqrt", 
    n_jobs=-1,
    random_state=42
)

# Initialize internal attributes
print("🔧 Finalizing model structure...")
rf.fit(X_train[:2], y_train_log[:2])

# Overwrite with actual trained trees
rf.estimators_ = estimators

print("✅ Training Complete.")

# -------------------------------------------------------------
# 5. Predict & Evaluate
# -------------------------------------------------------------
print("🔮 Predicting...")
log_preds = rf.predict(X_val)
preds = np.expm1(log_preds)  # Inverse log transform

mae = mean_absolute_error(val_labels, preds)
mse = mean_squared_error(val_labels, preds)
rmse = np.sqrt(mse)

print("\n📊 Validation Metrics")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}\n")

# -------------------------------------------------------------
# 6. Save Artifacts
# -------------------------------------------------------------
print("💾 Saving model...")
joblib.dump(rf, OUTPUT_DIR / "rf_e5_120k_model.pkl")

results = pd.DataFrame({
    "prompt": val_df["prompt"],
    "actual": val_labels,
    "predicted": preds,
    "error": preds - val_labels
})
results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
print(f"Done! Results saved to {OUTPUT_DIR}")