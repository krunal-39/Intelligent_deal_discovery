
#!/usr/bin/env python3

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dynamic Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "finetune"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE   = DATA_DIR / "val.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "src" / "models" / "lightgbm_svd_3000"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load JSONL with tqdm

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in tqdm(f, desc=f"Loading {path.name}"):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# Main

def main():

    print("\nLoading datasets...")
    train_df = load_jsonl(TRAIN_FILE)
    val_df   = load_jsonl(VAL_FILE)

    train_texts  = train_df["prompt"].astype(str).tolist()
    train_labels = train_df["response"].astype(float).values

    val_texts  = val_df["prompt"].astype(str).tolist()
    val_labels = val_df["response"].astype(float).values

    print(f"Train = {len(train_df)}, Val = {len(val_df)}")

    # TF-IDF

    print("\nBuilding TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=150_000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )

    print("Fitting TF-IDF...")
    X_train = vectorizer.fit_transform(train_texts)

    print("Transforming validation...")
    X_val = vectorizer.transform(val_texts)

    # SVD (Dimensionality Reduction)
    print("\nRunning SVD (reduce TF-IDF → 3000 dims)...")
    svd = TruncatedSVD(
        n_components=3000,
        n_iter=7,
        random_state=42
    )

    X_train_svd = svd.fit_transform(X_train)
    X_val_svd   = svd.transform(X_val)

    print("Reduced shape:", X_train_svd.shape)

    # LightGBM
    print("\nTraining LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=256,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_svd, train_labels)
    print("Model trained.")

    # Evaluation
    print("\nPredicting on validation set...")
    preds = model.predict(X_val_svd)

    mae  = mean_absolute_error(val_labels, preds)
    mse  = mean_squared_error(val_labels, preds)
    rmse = np.sqrt(mse)

    print("\n📊 Validation Metrics")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save Artifacts
    
    joblib.dump(model, OUTPUT_DIR / "lightgbm_model.pkl")
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.pkl")
    # joblib.dump(svd, OUTPUT_DIR / "svd_3000.pkl")

    print("\nSaved model, TF-IDF, and SVD transformer.")
    print("Done.")


if __name__ == "__main__":
    main()
