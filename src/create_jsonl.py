#!/usr/bin/env python3
"""
Generate train.jsonl and val.jsonl files for LLaMA QLoRA fine-tuning
using the cleaned combined_balanced_train.csv dataset.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from prompts import row_to_example


# Paths
DATA_DIR = "/data/home/anjeshnarwal/LLM_price_predictor/data/cleaned"
OUTPUT_DIR = "/data/home/anjeshnarwal/LLM_price_predictor/data/finetune"

CSV_PATH = os.path.join(DATA_DIR, "data_clean.csv")
TRAIN_JSONL = os.path.join(OUTPUT_DIR, "train.jsonl")
VAL_JSONL = os.path.join(OUTPUT_DIR, "val.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print(f"Reading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows and columns: {list(df.columns)}")

# Split into train and validation
RANDOM_SEED = 42
TEST_SIZE = 0.1

train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"Split completed → Train: {len(train_df):,} | Validation: {len(val_df):,}")

# Writer helper
def write_jsonl(df_in, path, mode="train"):
    """Writes DataFrame to JSONL file using row_to_example() from prompts.py"""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df_in.iterrows():
            ex = row_to_example(row, mode=mode)
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n:,} examples → {path}")


# Write the JSONL files
write_jsonl(train_df, TRAIN_JSONL, mode="train")
write_jsonl(val_df, VAL_JSONL, mode="test")

print("\nJSONL generation complete!")
print(f"Train file: {TRAIN_JSONL}")
print(f"Validation file: {VAL_JSONL}")
