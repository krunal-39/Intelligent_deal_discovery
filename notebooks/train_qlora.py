#!/usr/bin/env python3
"""
Clean QLoRA Training Script
Optimized for NVIDIA RTX A6000 + Flash Attention 2.
Upgrades: BF16 Precision + Target All Linear Layers (Better Quality).
"""

import os
import sys

# Set GPU 1 as the only visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    AutoPeftModelForCausalLM
)

# 1. Configuration

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "HUGGINGFACE_HUB_TOKEN missing!"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "finetune"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH   = DATA_DIR / "val.jsonl"

OUTPUT_DIR = BASE_DIR / "src" / "models" / "llama31_flash_bf16_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPECIALIST_DIR = BASE_DIR / "src" / "models" / "specialist_llama"
SPECIALIST_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

# 2. Load Datasets
print(f"Loading datasets...")

raw_datasets = load_dataset(
    "json",
    data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
    keep_in_memory=False
)

# 100k rows
TARGET_ROWS = 100000
print(f"Selecting random {TARGET_ROWS} rows for training...")

raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(TARGET_ROWS))

print(f"Train size: {len(raw_datasets['train'])}")
print(f"Validation size: {len(raw_datasets['validation'])}")


# 3. Tokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_batch(examples):
    texts = [(str(p) + " " + str(r)) for p, r in zip(examples["prompt"], examples["response"])]
    out = tokenizer(texts, truncation=True, padding=False, max_length=512, add_special_tokens=False)
    return {"input_ids": out["input_ids"]}

# print("Pretokenizing...")
# raw_datasets["train"] = raw_datasets["train"].map(tokenize_batch, batched=True, batch_size=2048, num_proc=16)
# raw_datasets["validation"] = raw_datasets["validation"].map(tokenize_batch, batched=True, batch_size=2048, num_proc=8)


# 4. Collator

project_root = BASE_DIR
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.collator import DataCollatorForPricePrediction
collator = DataCollatorForPricePrediction(tokenizer=tokenizer, max_length=512)


# 5. Model (BF16 Upgrade)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading model with Flash Attention 2 & Bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_cfg,
    token=HF_TOKEN,
    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Target ALL Linear Layers 
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.config.use_cache = False
model.print_trainable_parameters()

# 6. Trainer 

train_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    
    # Batch Size Settings
    per_device_train_batch_size=16,   
    gradient_accumulation_steps=4,    
    per_device_eval_batch_size=16,
    
    learning_rate=2e-4,
    warmup_ratio=0.03,
    
    fp16=False,
    bf16=True, 
    
    optim="adamw_bnb_8bit",
    lr_scheduler_type="cosine",
    
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=collator
)

print("Starting training...")
trainer.train()


# 7. Merge & Save

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(OUTPUT_DIR)
del model, trainer
torch.cuda.empty_cache()

print("Merging...")
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16, # Load as BF16
    attn_implementation="flash_attention_2"
)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(SPECIALIST_DIR, safe_serialization=True)
tokenizer.save_pretrained(SPECIALIST_DIR)
print("Done! Ready for serving.")
