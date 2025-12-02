#!/usr/bin/env python3
"""
Create final cleaned CSV from selected Amazon categories.

Enhancements:
- Include store field
- Drop rows where ANY final column is empty
- Limit title to max 100 characters
- Trim description/details/features to first 2 sentences
- Work with CSV or Parquet
- Per-category limit: min(file_size, 60000)
- Save output as: data_clean.csv
"""

import pandas as pd
import os
import numpy as np
import re
from tqdm import tqdm


# Configuration

DATA_DIR = "/data/home/anjeshnarwal/LLM_price_predictor/data/cleaned"
OUTPUT_PATH = os.path.join(DATA_DIR, "data_clean.csv")

dataset_names = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

MAX_PER_CATEGORY = 60000


# Cleaning functions

REMOVALS = [
    '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
    '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
    "By Manufacturer", "Item", "Date First", "Package", ":", "Number of",
    "Best Sellers", "Number", "Product "
]

def scrub_details(details):
    if details is None:
        return ""
    text = str(details)
    for r in REMOVALS:
        text = text.replace(r, "")
    return text

def scrub(text):
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r'[:\[\]"{}【】\'()<>]', " ", s)
    s = s.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    cleaned = [w for w in words if len(w) < 7 or not any(c.isdigit() for c in w)]
    return " ".join(cleaned).strip()

def clean_and_trim_sentences(text, max_sentences=2):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[{}\[\]\"'<>]", " ", text)
    t = re.sub(r"\s+", " ", t.strip())
    sentences = re.split(r'(?<=[.!?])\s+', t)
    return " ".join(sentences[:max_sentences]).strip()

def safe_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return np.nan


# Select files

files_to_process = []

for name in dataset_names:
    csv_path = os.path.join(DATA_DIR, f"{name}_clean.csv")
    pq_path  = os.path.join(DATA_DIR, f"{name}_clean.parquet")

    if os.path.exists(pq_path):
        files_to_process.append(pq_path)
    elif os.path.exists(csv_path):
        files_to_process.append(csv_path)


# Process datasets

final_rows = []

for path in tqdm(files_to_process, desc="Processing categories"):
    filename = os.path.basename(path)
    print(f"\nProcessing {filename}")

    try:
        df = pd.read_csv(path, low_memory=False) if path.endswith(".csv") else pd.read_parquet(path)
    except Exception:
        print(f"Failed to load {filename}")
        continue

    # Required fields
    if not all(x in df.columns for x in ["title", "price_num", "store"]):
        print(f"Skipping {filename} — missing required columns")
        continue

    df["price_num"] = df["price_num"].apply(safe_float)
    df = df.dropna(subset=["title", "price_num", "store"])

    # Clean text
    df["features"] = df.get("features", "").astype(str).apply(scrub)
    df["description"] = df.get("description", "").astype(str).apply(lambda x: clean_and_trim_sentences(x, 2))
    df["details"] = df.get("details", "").astype(str).apply(lambda x: clean_and_trim_sentences(scrub_details(x), 2))

    # Title limit
    df["title"] = df["title"].astype(str).str[:150]

    # Category fallback
    if "category" not in df.columns:
        df["category"] = ""

    # Final structure
    out = pd.DataFrame({
        "title": df["title"].astype(str),
        "features": df["features"].astype(str),
        "description": df["description"].astype(str),
        "details": df["details"].astype(str),
        "category": df["category"].astype(str),
        "store": df["store"].astype(str),
        "price": df["price_num"].astype(float),
    })

    # Remove empties
    out = out[
        (out["title"].str.len() > 0) &
        (out["features"].str.len() > 0) &
        (out["description"].str.len() > 0) &
        (out["details"].str.len() > 0) &
        (out["store"].str.len() > 0) &
        (~out["price"].isna())
    ]

    # Limit per category
    limit = min(len(out), MAX_PER_CATEGORY)
    out = out.sample(limit, random_state=42)

    print(f"✔ Final rows from {filename}: {len(out):,}")
    final_rows.append(out)

# Merge & Save

if not final_rows:
    raise RuntimeError("❌ No valid data found!")

final_df = pd.concat(final_rows).drop_duplicates(subset=["title"]).reset_index(drop=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nDONE!")
print(f"Saved → {OUTPUT_PATH}")
print(f"Final shape: {final_df.shape}")
