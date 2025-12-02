#!/usr/bin/env python3
import os
import sys
import json
import time
import re
import joblib
import requests
import csv
import torch
import faiss
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# Import local RAG modules (Only need metadata/index loaders now)
from rag.query_faiss import load_metadata, load_index, TOP_K

# Load Environment Variables
load_dotenv(os.path.join(project_root, ".env"))

# --- 2. CONFIGURATION ---
INPUT_FILE = os.path.join(project_root, "data", "finetune", "val.jsonl")
OUTPUT_FILE = os.path.join(project_root, "data", "ensemble_dataset_full.csv")
VLLM_URL = "http://localhost:8000/v1/completions"
RAG_DIR = os.path.join(project_root, "rag")

# --- 3. GLOBAL MODEL LOADING ---
print("🚀 Starting Global Pre-load...")

# A. Load SHARED Encoder (intfloat/e5-large)
print("   - [1/5] Loading Shared Encoder (intfloat/e5-large)...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Unified encoder for both RAG and RF_E5
    SHARED_ENCODER = SentenceTransformer("intfloat/e5-large", device=device)
    print("     ✅ Shared Encoder loaded on GPU.")
except Exception as e:
    print(f"     ❌ Error loading Encoder: {e}")
    sys.exit(1)

# B. Load FAISS (RAG System)
print("   - [2/5] Loading FAISS RAG...")
try:
    meta_path = os.path.join(RAG_DIR, "metadata.jsonl")
    RAG_METADATA = load_metadata(meta_path)
    
    index_path = os.path.join(RAG_DIR, "faiss_index.bin")
    RAG_INDEX = load_index(index_path)
    
    # Verify dimensions match E5 (1024 dim)
    if RAG_INDEX.d != 1024:
        print(f"     ⚠️ WARNING: FAISS index dim is {RAG_INDEX.d}, but E5 is 1024. This might fail!")
    else:
        print("     ✅ FAISS Index loaded (matches E5-large).")
except Exception as e:
    print(f"     ❌ Error loading FAISS: {e}")

# C. Load BASE Random Forest (TF-IDF)
print("   - [3/5] Loading Base Random Forest...")
rf_folder = os.path.join(project_root, "src", "models", "random_forest_price_model")
try:
    RF_MODEL = joblib.load(os.path.join(rf_folder, "random_forest_model.pkl"))
    TFIDF_VECTORIZER = joblib.load(os.path.join(rf_folder, "tfidf_vectorizer.pkl"))
    print("     ✅ Base RF loaded.")
except Exception as e:
    print(f"     ❌ Error loading Base RF: {e}")

# D. Load LIGHTGBM (TF-IDF)
print("   - [4/5] Loading LightGBM...")
lgbm_folder = os.path.join(project_root, "src", "models", "lightgbm_price_model_full")
try:
    LGBM_MODEL = joblib.load(os.path.join(lgbm_folder, "lightgbm_model.pkl"))
    LGBM_VECTORIZER = joblib.load(os.path.join(lgbm_folder, "tfidf_vectorizer.pkl"))
    print("     ✅ LightGBM loaded.")
except Exception as e:
    print(f"     ❌ Error loading LightGBM: {e}")

# E. Load NEW RANDOM FOREST (E5-Large)
print("   - [5/5] Loading RF (E5-Large Model)...")
rf_e5_folder = os.path.join(project_root, "src", "models", "rf_e5_large_120k")
try:
    RF_E5_MODEL = joblib.load(os.path.join(rf_e5_folder, "rf_e5_120k_model.pkl"))
    print("     ✅ RF (E5) Model loaded.")
except Exception as e:
    print(f"     ❌ Error loading RF (E5): {e}")

# F. Setup Gemini Keys
API_KEYS = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_3"), os.getenv("GEMINI_API_KEY_2")]
API_KEYS = [k for k in API_KEYS if k]
GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"
print("✅ All Models Loaded Globally.\n")


# --- 4. INFERENCE HELPER FUNCTIONS ---

def get_faiss_results_fast(query):
    """
    Retrieves context using the SHARED E5 Encoder.
    CRITICAL: Adds 'query: ' prefix for correct retrieval dynamics.
    """
    # E5-Large Requirement: Prepend "query: " for retrieval
    formatted_query = "query: " + query
    
    q = SHARED_ENCODER.encode([formatted_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = RAG_INDEX.search(q, TOP_K)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0: continue
        m = RAG_METADATA[idx]
        results.append({
            "prompt": m["prompt"], 
            "response": m["response"], 
            "score": float(score)
        })
    return results

def build_unified_prompt(target_product, retrieved_docs):
    prompt_parts = ["Reference Products (Context):"]
    for i, doc in enumerate(retrieved_docs):
        raw_text = doc['prompt'].strip().replace("\n", " ")
        price = doc['response']
        prompt_parts.append(f"\n--- Product {i+1} ---\n{raw_text}\nPrice: {price}")
    prompt_parts.append("\n\nTarget Product to Estimate:")
    prompt_parts.append(target_product.strip())
    prompt_parts.append("\nPrice:")
    return "\n".join(prompt_parts)

def extract_title(full_prompt):
    match = re.search(r"Product Title:\s*(.*)", full_prompt)
    if match: return match.group(1).strip()
    return ""

# --- PREDICTORS ---

def get_llama_pred(prompt_text):
    payload = {"model": "llama-qlora", "prompt": prompt_text, "max_tokens": 10, "temperature": 0.1}
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=30)
        if response.status_code == 200:
            text = response.json()['choices'][0]['text']
            nums = re.findall(r"\d+\.?\d*", text)
            return float(nums[0]) if nums else 0.0
    except: pass
    return 0.0

def get_gemini_pred(prompt_text, current_api_key):
    try:
        genai.configure(api_key=current_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt_text)
        text = response.text.strip()
        nums = re.findall(r"\d+\.?\d*", text)
        return float(nums[0]) if nums else 0.0
    except: return 0.0

def get_base_rf_pred(full_prompt):
    """
    UPDATED: Uses full prompt text instead of just title.
    Assumes this model was trained on log-prices (keeps expm1).
    """
    try:
        vec = TFIDF_VECTORIZER.transform([full_prompt])
        log_pred = RF_MODEL.predict(vec)
        return round(float(np.expm1(log_pred[0])), 2)
    except: return 0.0

def get_lgbm_pred(full_prompt):
    """
    UPDATED: Uses full prompt text instead of just title.
    Directly returns prediction (no expm1) as per training logic.
    """
    try:
        # Transform the input text (Full Prompt)
        vec = LGBM_VECTORIZER.transform([full_prompt])
        
        # Get the raw prediction
        pred = LGBM_MODEL.predict(vec)
        
        # Return the float value directly without expm1
        return round(float(pred[0]), 2)
        
    except Exception as e: 
        return 0.0

def get_rf_e5_pred(prompt_text):
    try:
        # CRITICAL: E5 models must have "query: " prefix
        e5_input = "query: " + prompt_text
        
        # Use SHARED_ENCODER (already loaded)
        embedding = SHARED_ENCODER.encode([e5_input], convert_to_numpy=True, normalize_embeddings=True)
        
        # Predict & Inverse Log
        log_pred = RF_E5_MODEL.predict(embedding)
        return round(float(np.expm1(log_pred[0])), 2)
    except Exception as e: 
        return 0.0

# --- 5. MAIN EXECUTION ---

def main():
    print(f"Reading dataset from {INPUT_FILE}...")
    data_list = []
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: data_list.append(json.loads(line))
                    except: continue
        
        df = pd.DataFrame(data_list)
        print(f"   -> Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Error: File not found.")
        return

    # Sampling
    SAMPLE_SIZE = 2500
    if len(df) > SAMPLE_SIZE:
        print(f"🎲 Randomly sampling {SAMPLE_SIZE} rows...")
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df

    file_exists = os.path.exists(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Header includes all 5 models
        if not file_exists:
            writer.writerow(["title", "ground_truth", "llama_pred", "gemini_pred", "base_rf_pred", "lgbm_pred", "rf_e5_pred"])
        
        print(f"🚀 Processing {len(df_sample)} rows...")
        print("Format: [Title | GT | LLaMA | Gemini | BaseRF | LGBM | RF_E5]")
        print("-" * 85)
        
        for i, (index, row_data) in enumerate(tqdm(df_sample.iterrows(), total=len(df_sample), unit="row")):
            try:
                full_prompt = row_data['prompt']
                ground_truth = float(str(row_data['response']).strip())
                title = extract_title(full_prompt)
                
                if not title: continue

                # 1. Get Context (RAG with E5)
                docs = get_faiss_results_fast(full_prompt)
                unified_prompt = build_unified_prompt(full_prompt, docs)

                # 2. LLM Predictions
                llama_p = get_llama_pred(unified_prompt)
                
                active_key = API_KEYS[i % len(API_KEYS)]
                gemini_p = get_gemini_pred(unified_prompt, active_key)
                
                # 3. Traditional & Embedding Model Predictions
                # UPDATED: Passing full_prompt to RF and LGBM as requested
                base_rf_p = get_base_rf_pred(full_prompt)
                lgbm_p    = get_lgbm_pred(full_prompt)
                rf_e5_p   = get_rf_e5_pred(full_prompt)
                
                time.sleep(1.5) # Rate limit safety
                
                writer.writerow([title, ground_truth, llama_p, gemini_p, base_rf_p, lgbm_p, rf_e5_p])
                f.flush()
                
                # Logging
                t_short = (title[:15] + '..') if len(title) > 15 else title
                tqdm.write(f"{t_short} | {ground_truth:.1f} | {llama_p:.1f} | {gemini_p:.1f} | {base_rf_p:.1f} | {lgbm_p:.1f} | {rf_e5_p:.1f}")

            except Exception as e:
                tqdm.write(f"⚠️ Error row {i}: {e}")
                continue

    print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()