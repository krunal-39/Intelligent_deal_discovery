#!/usr/bin/env python3
"""
Build FAISS index for RAG using local GPU embeddings.

STRICT MODE: 
- Clears old index files first.
- Only indexes 'train.jsonl' to prevent data leakage during validation.
- Pinned to GPU 0.
"""

import os
import sys

# -------------------------------------------------------------
# CRITICAL: Pin to GPU 0
# -------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# -----------------------
# Paths (dynamic)
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "finetune"

# *** CRITICAL: Only use TRAIN data for the index ***
TRAIN_FILE = DATA_DIR / "train.jsonl" 

RAG_DIR = PROJECT_ROOT / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = RAG_DIR / "faiss_index.bin"
META_PATH = RAG_DIR / "metadata.jsonl"
ENCODER_INFO = RAG_DIR / "encoder_info.txt"
EMB_NPY = RAG_DIR / "embeddings.npy"

# -----------------------
# Config
# -----------------------
PREFERRED_MODEL = "intfloat/e5-large"
FALLBACK_MODEL = "sentence-transformers/all-mpnet-base-v2"

BATCH_SIZE = 256
NORMALIZE = True
USE_FAISS_GPU = True
TOP_K = 5

# -----------------------
# Cleanup Function
# -----------------------
def clear_previous_index():
    print("🧹 Cleaning up old RAG artifacts...")
    files_to_remove = [INDEX_PATH, META_PATH, ENCODER_INFO, EMB_NPY]
    for p in files_to_remove:
        if p.exists():
            try:
                os.remove(p)
                print(f"   - Removed: {p.name}")
            except OSError as e:
                print(f"   ⚠️ Error removing {p.name}: {e}")
    print("   - Cleanup complete.\n")

# -----------------------
# Helper: load jsonl
# -----------------------
def load_jsonl(path):
    out = []
    print(f"Loading data from: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    except FileNotFoundError:
        print(f"❌ Error: File not found at {path}")
        return []

# -----------------------
# Main
# -----------------------
def main():
    # 1. Clear old files
    clear_previous_index()

    # 2. Load Documents (TRAIN ONLY)
    docs = load_jsonl(TRAIN_FILE)
    
    if len(docs) == 0:
        raise RuntimeError("❌ No documents found. Please check data/finetune/train.jsonl")

    print(f"✅ Loaded {len(docs)} documents for indexing.")

    # 3. Extract Texts
    # CRITICAL: For E5 models, documents in the index must start with "passage: "
    # We do NOT add this prefix to the metadata stored in JSON, only to the text we embed.
    print("⚡ Adding 'passage: ' prefix for E5 embedding quality...")
    texts = ["passage: " + str(d.get("prompt", "")) for d in docs]
    
    # Metadata stores the RAW text (for humans/LLMs to read later)
    metadata = [
        {"id": i, "prompt": str(docs[i].get("prompt", "")), "response": docs[i].get("response", None)} 
        for i in range(len(docs))
    ]

    # 4. Load Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for embeddings: {device}")

    model_name = PREFERRED_MODEL
    try:
        print(f"Loading embedder: {model_name}")
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        print(f"⚠️ Preferred model not found. Falling back to {FALLBACK_MODEL}")
        model_name = FALLBACK_MODEL
        model = SentenceTransformer(model_name, device=device)

    # Save encoder info
    with open(ENCODER_INFO, "w") as f:
        f.write(model_name + "\n")

    # 5. Compute Embeddings
    n = len(texts)
    embeddings = np.zeros((n, model.get_sentence_embedding_dimension()), dtype=np.float32)
    
    print(f"Computing embeddings for {n} docs (Batch Size: {BATCH_SIZE})...")
    
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Embedding"):
        end = min(n, start + BATCH_SIZE)
        batch_texts = texts[start:end]
        
        # Encode
        emb = model.encode(
            batch_texts, 
            show_progress_bar=False, 
            convert_to_numpy=True, 
            batch_size=BATCH_SIZE, 
            normalize_embeddings=False
        )
        
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
            
        embeddings[start:end] = emb

    dims = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    if NORMALIZE:
        print("Normalizing embeddings (Cosine Similarity)...")
        faiss.normalize_L2(embeddings)

    # 6. Build FAISS Index (IndexFlatIP)
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dims)

    # GPU Acceleration
    if USE_FAISS_GPU and faiss.get_num_gpus() > 0:
        print("⚡ Using GPU for FAISS construction...")
        try:
            res = faiss.StandardGpuResources()
            # We pinned CUDA_VISIBLE_DEVICES=0, so "0" here refers to the only visible GPU
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embeddings)
            # Move back to CPU for saving
            index = faiss.index_gpu_to_cpu(gpu_index)
        except Exception as e:
            print(f"⚠️ GPU Indexing failed ({e}). Falling back to CPU.")
            index.add(embeddings)
    else:
        index.add(embeddings)

    # 7. Save Artifacts
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✅ Saved Index: {INDEX_PATH}")

    print(f"Saving Metadata to {META_PATH}...")
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Optional: Save numpy array
    np.save(str(EMB_NPY), embeddings)
    
    print(f"\n🎉 RAG Index Rebuilt successfully! ({n} documents)")

if __name__ == "__main__":
    main()