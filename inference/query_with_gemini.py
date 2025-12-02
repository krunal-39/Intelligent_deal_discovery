#!/usr/bin/env python3

import os
import sys
import torch
import re
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add root to sys.path so we can import from the 'rag' folder
if project_root not in sys.path:
    sys.path.insert(0, project_root)

RAG_DIR = os.path.join(project_root, "rag")

# Imports from 'rag.query_faiss'
from rag.query_faiss import (
    load_metadata,
    load_index,
    get_encoder_name,
    TOP_K,
)

def get_faiss_results(query):
    # This ensures it finds the files even if you run from root or inference/
    meta_path = os.path.join(RAG_DIR, "metadata.jsonl")
    index_path = os.path.join(RAG_DIR, "faiss_index.bin")

    print(f"Loading RAG data from: {RAG_DIR}...") # Debug print
    
    metadata = load_metadata(meta_path)
    index = load_index(index_path)

    encoder_name = get_encoder_name() or "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)

    q = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    D, I = index.search(q, TOP_K)

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        results.append({
            "prompt": m["prompt"],
            "response": m["response"],
            "score": float(score)
        })
    return results


def build_prompt(query, docs):
    out = []
    out.append("You estimate product prices using relevant examples.")
    out.append(f"User Query:\n{query}\n")
    out.append("Retrieved Examples:")

    for i, d in enumerate(docs):
        out.append(f"\nExample {i+1}:")
        out.append(d["prompt"])
        out.append(f"Price: {d['response']}")
        out.append(f"Score: {d['score']:.4f}")

    out.append(
        "\nUse ONLY these retrieved examples to guide your prediction. "
        "Return ONE final numeric USD price with no explanation."
    )
    return "\n".join(out)


def rag_predict(query):
    # Load .env from project root
    load_dotenv(os.path.join(project_root, ".env"))
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    docs = get_faiss_results(query)
    prompt = build_prompt(query, docs)

    genai.configure(api_key=key)
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    
    try:
        out = model.generate_content(prompt)
        raw_text = out.text.strip()
        
        # Extract the first numeric value found (handles "$19.99", "19.99", etc.)
        numbers = re.findall(r"\d+\.?\d*", raw_text)
        
        if numbers:
            return float(numbers[0])
        else:
            print(f"Gemini returned no number: '{raw_text}'")
            return 0.0

    except Exception as e:
        print(f"Gemini Error: {e}")
        return 0.0

def main():
    print("--- Inference: RAG + Gemini ---")
    print("Enter product query:")
    query = input().strip()
    if not query:
        print("Empty query.")
        return

    print("\nGemini Output:")
    print(rag_predict(query))


if __name__ == "__main__":
    main()
