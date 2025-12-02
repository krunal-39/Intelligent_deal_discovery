#!/usr/bin/env python3

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "Missing HUGGINGFACE_HUB_TOKEN"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# Paths relative to project root
ADAPTER_DIR = BASE_DIR / "src" / "models" / "llama31_8b_qlora_qkv"
MERGED_DIR = BASE_DIR / "src" / "models" / "llama31_8b_qlora_qkv_merged"
MERGED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False,
    trust_remote_code=True,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN,
)

print("Loading LoRA adapter from:", ADAPTER_DIR)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print("Saving merged model to:", MERGED_DIR)
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Done. Merged model ready for vLLM or HF inference.")
