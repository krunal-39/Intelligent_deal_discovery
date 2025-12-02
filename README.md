# Intelligent_deal_discovery
Autonomous Price Intelligence Agent 🏷️

Overview

This project implements an advanced Multi-Agent Price Prediction System designed to estimate fair market values for e-commerce products. Unlike traditional price predictors that rely on a single algorithm, this system operates as a "Council of Experts" architecture.

By orchestrating a collaboration between Large Language Models (LLMs), Statistical Models, and Retrieval-Augmented Generation (RAG), the system overcomes the individual limitations of each approach:

LLMs (Llama 3.1 & Gemini): Understand semantic nuance and product details but can struggle with precise arithmetic.

Statistical Models (LightGBM & Random Forest): Excel at regression on structured features but often miss contextual clues.

RAG: Provides real-time market grounding by retrieving comparable products.

The final decision is made by a Meta-Learner (XGBoost), which learns which expert to trust based on the specific characteristics of the product being analyzed.

🏗️ System Architecture

The logic is distributed across a set of specialized autonomous agents, managed by a central planner.

1. The Agent Council (src/agents/)

Planning Agent (planning_agent.py): The project manager. It receives the user query, orchestrates the workflow, and ensures the job gets done.

Scanner Agent: Responsible for fetching raw product data, filtering RSS feeds, and formatting inputs.

Specialist Agent (specialist_agent.py): Wraps a fine-tuned Llama 3.1 8B model, serving as our domain expert for detailed product parsing.

Frontier Agent (frontier_agent.py): Interfaces with Google Gemini 2.5 to provide a second opinion using advanced reasoning capabilities.

Statistical Agents (lightgbm_agent.py, rf_agent.py): Lightweight agents that run fast inference using TF-IDF and Embedding-based regression models.

Ensemble Agent (ensemble_agent.py): The judge. It aggregates predictions from all other agents and uses a trained XGBoost model to weigh them into a final consensus price.

2. The Model Layer

Llama 3.1 8B (QLoRA): Fine-tuned on ~100k Amazon products using Flash Attention 2 and BF16 precision.

LightGBM: Trained on TF-IDF vectors reduced via SVD (Singular Value Decomposition).

Random Forest: Trained on intfloat/e5-large embeddings to capture semantic similarity in product titles.

RAG System: A FAISS vector database containing 120k+ product embeddings for finding "nearest neighbor" pricing examples.

📂 Project Structure

├── data/                       # Raw and processed datasets
├── rag/                        # FAISS index, metadata, and RAG utilities
├── src/
│   ├── agents/                 # Agent definitions (Planner, Specialist, etc.)
│   ├── models/                 # Saved model artifacts (PKL files, LoRA adapters)
│   ├── collator.py             # Custom training collator
│   └── ...
├── train_qlora.py              # Llama 3.1 Fine-tuning script
├── train_lgbm.py               # LightGBM training script
├── train_rf.py                 # Random Forest training script
├── train_ensamble.py           # Meta-learner (XGBoost) training
├── build_faiss_index.py        # RAG Index builder
├── app.py                      # Main entry point (UI/Application)
└── install.sh                  # Setup script


🚀 Installation & Setup

Prerequisites

Python 3.10+

NVIDIA GPU (Required for Llama 3.1 inference and training)

HuggingFace Token (for accessing Llama models)

Google Gemini API Key (for the Frontier agent)

1. Environment Setup

We have provided a script to handle dependencies, including the GPU-specific installation of FAISS.

bash install.sh


2. Configuration

Create a .env file in the root directory to store your credentials:

HUGGINGFACE_HUB_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
# Optional: Additional keys for rotation
GEMINI_API_KEY_2=...


🛠️ Training Pipeline

To reproduce the system from scratch, execute the training scripts in the following order.

Step 1: Data Preparation

Clean raw Amazon metadata and convert it into training formats (JSONL/CSV).

python finetune_dataset.py
python create_jsonl.py


Step 2: Train the "Experts"

Train the individual models that act as the brains for our agents.

# 1. Fine-tune Llama 3.1 (QLoRA) - Optimized for RTX A6000
python train_qlora.py

# 2. Train LightGBM (TF-IDF + SVD)
python train_lgbm.py

# 3. Train Random Forest (E5 Embeddings)
python train_rf.py


Step 3: Build the RAG Knowledge Base

Create the vector index so the RAG agent can perform retrieval.

python build_faiss_index.py


Step 4: Train the Meta-Learner

Generate a dataset of predictions from all models, then train the ensemble to weigh them.

# 1. Generate ensemble dataset (runs all models on validation set)
python generate_ensamble_data.py

# 2. Train the XGBoost weight optimizer
python train_ensamble.py


🖥️ Usage

To launch the full Multi-Agent System:

python app.py


Workflow:

The system accepts a product query (e.g., "Best laptop deals").

The Scanner Agent finds relevant items.

The Planning Agent dispatches each item to the Llama, LightGBM, and Frontier agents.

The RAG Utility injects context into the LLMs.

The Ensemble Agent calculates the final "fair price" and compares it to the actual listed price.

If a significant discount is detected, the Messaging Agent alerts the user.

🔬 Technical Highlights

Optimized QLoRA: We utilize 4-bit quantization (bitsandbytes) to fit Llama 3.1 training on a single GPU, targeting all linear layers (q_proj, k_proj, etc.) for maximum reasoning capability.

Hybrid Retrieval: The system uses intfloat/e5-large embeddings for semantic search. The implementation handles the specific "query:" and "passage:" prefixes required by this model family.

Agent Communication: Agents share a standardized message bus, allowing for modular expansion. Adding a new expert model is as simple as creating a new Agent class and registering it with the Ensemble.
