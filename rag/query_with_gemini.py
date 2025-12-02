#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import google.generativeai as genai

from query_faiss import (
    load_metadata,
    load_index,
    get_encoder_name,
    INDEX_PATH,
    META_PATH,
    TOP_K,
)

import faiss
import torch
from sentence_transformers import SentenceTransformer


def get_faiss_results(query):
    metadata = load_metadata(META_PATH)
    index = load_index(INDEX_PATH)

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


# ----------------------------------------------------
# NEW FUNCTION: this is what your ensemble script calls
# ----------------------------------------------------

def rag_predict(query):
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    docs = get_faiss_results(query)
    prompt = build_prompt(query, docs)

    genai.configure(api_key=key)
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    out = model.generate_content(prompt)

    return out.text.strip()


# ------------------------------
# Original manual CLI still works
# ------------------------------

def main():
    print("Enter product query:")
    query = input().strip()
    if not query:
        print("Empty query.")
        return

    print("\nGemini Output:\n")
    print(rag_predict(query))


if __name__ == "__main__":
    main()
