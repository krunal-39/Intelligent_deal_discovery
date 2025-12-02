# Autonomous Multi-Agent System for Intelligent Deal Discovery

# Overview

This project is a Price Intelligence System that autonomously finds "Great Deals" on e-commerce platforms. It uses a Multi-Agent Architecture where specialized AI agents work together to determine the fair market value of a product.

Instead of relying on a single model, we use a "Council of Experts":

LLMs (Llama 3.1 & Gemini): Understand product specs, brand value, and context.

ML Models (LightGBM & Random Forest): Provide stable, statistical price predictions.

RAG (Retrieval-Augmented Generation): Uses historical data to ground predictions in reality.

The final decision is made by an Ensemble Agent (XGBoost) that weighs these opinions to calculate a precise "Fair Price".


# Project Structure

### Root Directory

app.py: The main entry point of the application. It initializes the Planning Agent and starts the autonomous deal discovery loop.

### Source Code (src/)

### Agents (src/agents/)

planning_agent.py: The central orchestrator (Project Manager). It manages the workflow, dispatches tasks, and coordinates all other agents.

scanner_agent.py: Monitors RSS feeds and uses a lightweight LLM (Gemini Flash) to filter out irrelevant items or accessories.

specialist_agent.py: The "Reasoning Expert." Wraps the fine-tuned Llama 3.1 model to analyze complex product specifications.

frontier_agent.py: The "Second Opinion." Interfaces with Google Gemini 2.5 to provide a logic check on the Llama model's output.

lightgbm_agent.py: A wrapper for the LightGBM model that provides fast, keyword-based price estimates.

rf_agent.py: A wrapper for the Random Forest model that provides embedding-based price estimates.

ensemble_agent.py: The "Judge." Aggregates predictions from all experts and runs the XGBoost model to decide the final price.

evaluator_agent.py: The "Quality Assurance." Compares the AI's consensus price against the real price to assign a verdict (e.g., "Great Deal").

messaging_agent.py: Handles user communication, formatting rich alerts and sending them via Telegram or the Console.

rag_utility.py: A utility agent that performs vector searches on the FAISS index to find historical pricing examples.

base_agent.py: The parent class for all agents, handling standard logging, color-coding, and initialization.

deals.py: Defines the data structures (Deal, ScrapedDeal) and contains the logic for parsing raw RSS feed items.

### Utilities & Processing (src/)

dataset_setup.ipynb:  Jupyter Notebook for Exploratory Data Analysis (EDA), visualizing price distributions and cleaning outliers.

finetune_dataset.py: The initial data cleaning script. It scrubs raw Amazon metadata, handles missing values, and formats text.

create_jsonl.py: Converts the cleaned CSV data into the specific JSONL format required for Llama 3.1 fine-tuning.

collator.py: A custom Data Collator class that handles tokenization and masking for efficient LLM training.

merge_llama_qlora.py: A utility script to merge the trained LoRA adapters back into the base Llama model for easier deployment.

prompts.py: A centralized file containing all system prompts and templates used by the LLMs for consistency.


### rag code (rag/)

install.sh: Automated setup script that handles complex dependencies, specifically installing GPU-optimized FAISS and PyTorch.

build_faiss_index.py: Encodes the training dataset into vectors and builds the FAISS index for the RAG system.

### training code (notebooks/)

train_qlora.py: The primary training script for fine-tuning Llama 3.1 8B using QLoRA (4-bit quantization) on the dataset.

train_lgbm.py: Trains the LightGBM statistical model using TF-IDF vectors reduced via SVD for baseline regression.

train_rf.py: Trains the Random Forest model using intfloat/e5-large embeddings to capture semantic similarity.

train_ensamble.py: Trains the XGBoost meta-learner (The Judge) to learn the optimal weights for combining model predictions.

### inference code (inference/) 

generate_ensamble_data.py: A critical script that runs all models on the validation set to generate the dataset used to train the Ensemble/Meta-learner.

llama_infer_merged.py: A standalone script to test inference speed and accuracy on the merged Llama model.

query_with_gemini.py: A test script to verify the RAG retrieval pipeline and Gemini's response generation.

# Workflow 

We have following Agents for the system.

Planning Agent: It coordinates everyone else and starts the process.

Scanner Agent:  It looks for new product deals on the internet and organizes the data.

Ensemble Agent: It uses multiple AI models to estimate what the fair price of the product should be.

Evaluator Agent: It compares the actual price to the guessed price to decide if it is a "Great Deal".

Messaging Agent: It sends an alert (like a text message) if a good deal is found.

### Simplified WorkFlow


(app.py) → (Planning Agent) →  (Scanner Agent) →  (Planning Agent) → (Ensemble Agent) → (Planning Agent) → (Evaluator Agent) →  (Planning Agent) → (Messaging Agent)


# Installation & Setup

## Prerequisites

Python 3.10+

GPU server (like NVIDIA A100/A6000) for training;  smaller GPUs sufficient for inference.

API Keys: HuggingFace (for Llama) & Google Gemini.

### 1. Install Dependencies

Run the included script to handle complex GPU libraries (FAISS & Torch).

bash install.sh


### 2. Configure Environment

Create a .env file in the root directory:

HUGGINGFACE_HUB_TOKEN=your_hf_token
GEMINI_API_KEY=your_gemini_key

## Run the Agent System

To start the autonomous loop that scans feeds, predicts prices, and alerts

python app.py

To start the vllm server

vllm serve src/models/specialist_llama --served-model-name specialist_llama



