import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- CONFIGURATION ---
IINPUT_FILE = os.path.join("data", "ensemble_dataset_full.csv")
MODEL_SAVE_PATH = os.path.join("src", "models", "ensemble", "price_ensemble.json")

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(Error: Could not find {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic cleaning
    initial_len = len(df)
    df = df[df['ground_truth'] > 0].dropna()
    print(f"Loaded {initial_len} rows. (Used {len(df)} after cleaning)")
    print("-" * 60)

    # 2. Define the Experiments
    fixed_features = ['llama_pred', 'gemini_pred']
    target = 'ground_truth'

    experiments = [
        ("Base Random Forest (TF-IDF)", 'base_rf_pred'),
        ("LightGBM (TF-IDF)",           'lgbm_pred'), 
        ("RF E5 (Embeddings)",          'rf_e5_pred')
    ]

    # 3. Run the Loop
    for model_name, variable_col in experiments:
        print(f"\n🧪 Experiment: LLaMA + Gemini + [{model_name}]")
        
        # Combine features for this specific run
        feature_cols = fixed_features + [variable_col]
        
        X = df[feature_cols]
        y = df[target]

        # Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize XGBoost
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )

        # Train
        xgb_model.fit(X_train, y_train)

        # Predict
        preds = xgb_model.predict(X_test)

        # Calculate Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)

        # Get Weights
        importances = xgb_model.feature_importances_

        # Print Results
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        
        print(f"Model Weights (Importance %):")
        feat_imp_list = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        for name, val in feat_imp_list:
            print(f"{name:<15} : {val:.4f}")

        # SAVE LOGIC
        if variable_col == 'lgbm_pred':
            print(f"\n SAVING MODEL (LightGBM Variant detected)...")
            xgb_model.fit(X, y) 
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            xgb_model.save_model(MODEL_SAVE_PATH)
            print(f"Saved to: {MODEL_SAVE_PATH}")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
