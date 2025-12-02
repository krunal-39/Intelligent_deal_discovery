# # #!/usr/bin/env python3
# # # ============================================
# # # QLoRA Training Script (Q/K/V modules only + 90K sample subset)
# # # ============================================

# # import os, time, math, sys
# # from pathlib import Path
# # from dotenv import load_dotenv
# # import torch
# # from datasets import load_dataset

# # # -------------------------------------------------------------
# # # Cell 1: Environment Setup
# # # -------------------------------------------------------------
# # load_dotenv()

# # HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
# # assert HF_TOKEN, "❌ Missing Hugging Face token! Please check your .env file."

# # print("✅ Env ready. CUDA available:", torch.cuda.is_available())
# # if torch.cuda.is_available():
# #     for i in range(torch.cuda.device_count()):
# #         print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | Memory total (GB):",
# #               round(torch.cuda.get_device_properties(i).total_memory/1e9, 1))
# # else:
# #     print("⚠️ No GPUs detected — training will be very slow or fail.")

# # DATA_DIR = Path("/data/home/anjeshnarwal/LLM_price_predictor/data/finetune")
# # TRAIN_PATH = DATA_DIR / "train.jsonl"
# # VAL_PATH = DATA_DIR / "val.jsonl"
# # OUTPUT_DIR = Path("../src/models/llama31_8b_qlora_qkv_90k")
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # seed = 42
# # torch.manual_seed(seed)

# # # -------------------------------------------------------------
# # # Cell 2: Load dataset (sample 90k)
# # # -------------------------------------------------------------
# # if not TRAIN_PATH.exists() or not VAL_PATH.exists():
# #     raise FileNotFoundError(f"❌ Missing dataset!\nExpected:\n - {TRAIN_PATH}\n - {VAL_PATH}")

# # print("🔁 Loading JSONL dataset (may take a minute)...")
# # raw_datasets = load_dataset(
# #     "json",
# #     data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
# #     keep_in_memory=False
# # )

# # # ✅ Subsample train split to 90k randomly
# # full_train_size = len(raw_datasets["train"])
# # subset_size = min(50000, full_train_size)
# # raw_datasets["train"] = raw_datasets["train"].shuffle(seed=seed).select(range(subset_size))

# # print(f"📦 Dataset loaded successfully:")
# # print(f"  • Train rows (sampled): {len(raw_datasets['train']):,} / {full_train_size:,}")
# # print(f"  • Validation rows: {len(raw_datasets['validation']):,}")

# # for i, ex in enumerate(raw_datasets["train"].select(range(3))):
# #     prompt = ex.get("prompt", "")[:200].replace("\n", " ")
# #     response = ex.get("response", "")
# #     print(f"\nSAMPLE {i} prompt (trunc): {prompt}  -> response: {response}")

# # # -------------------------------------------------------------
# # # Cell 3: Tokenizer + Pre-tokenization
# # # -------------------------------------------------------------
# # from transformers import AutoTokenizer

# # MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
# # token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# # tokenizer = AutoTokenizer.from_pretrained(
# #     MODEL_ID,
# #     use_fast=False,
# #     trust_remote_code=True,
# #     **token_kwargs
# # )

# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

# # print("✅ Tokenizer loaded. Vocab size:", len(tokenizer))
# # print(f"Pad token: {tokenizer.pad_token} | eos_token: {tokenizer.eos_token}")

# # DO_PRETOKENIZE = True
# # if DO_PRETOKENIZE:
# #     def tokenize_batch(examples):
# #         texts = [(str(p or "") + " " + str(r or "")) for p, r in zip(examples.get("prompt", []), examples.get("response", []))]
# #         out = tokenizer(texts, truncation=True, padding=False, max_length=512, add_special_tokens=False)
# #         return {"input_ids": out["input_ids"]}

# #     print("🔁 Pre-tokenizing train split (num_proc=8)...")
# #     raw_datasets["train"] = raw_datasets["train"].map(
# #         tokenize_batch,
# #         batched=True,
# #         batch_size=1024,
# #         remove_columns=raw_datasets["train"].column_names,
# #         num_proc=8
# #     )
# #     print("🔁 Pre-tokenizing validation split (num_proc=4)...")
# #     raw_datasets["validation"] = raw_datasets["validation"].map(
# #         tokenize_batch,
# #         batched=True,
# #         batch_size=1024,
# #         remove_columns=raw_datasets["validation"].column_names,
# #         num_proc=4
# #     )
# #     print("✅ Pre-tokenization complete.")

# # # -------------------------------------------------------------
# # # Cell 4: Data Collator
# # # -------------------------------------------------------------
# # try:
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
# # except NameError:
# #     current_dir = os.getcwd()
# # project_root = os.path.abspath(os.path.join(current_dir, '..'))
# # if project_root not in sys.path:
# #     sys.path.insert(0, project_root)

# # from src.collator import DataCollatorForPricePrediction
# # collator = DataCollatorForPricePrediction(tokenizer=tokenizer, max_length=512)
# # print("🧠 Collator initialized successfully")

# # # -------------------------------------------------------------
# # # Cell 5: Load Model + PEFT (QKV only)
# # # -------------------------------------------------------------
# # from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# # from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.float16
# # )

# # print("🔧 Loading model in 4-bit ...")
# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_ID,
# #     device_map="auto",
# #     trust_remote_code=True,
# #     quantization_config=bnb_config,
# #     torch_dtype=torch.float16,
# #     **token_kwargs
# # )
# # print("✅ Model loaded in 4-bit.")

# # if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
# #     model.resize_token_embeddings(len(tokenizer))

# # print("⚙️ Preparing model for k-bit training with gradient checkpointing...")
# # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# # # ✅ Convert FP32 to FP16
# # for p in model.parameters():
# #     if p.dtype == torch.float32:
# #         p.data = p.data.to(torch.float16)

# # # ✅ Train only Q, K, V projections
# # target_modules = ['q_proj', 'k_proj', 'v_proj']

# # lora_config = LoraConfig(
# #     r=16,
# #     lora_alpha=32,
# #     target_modules=target_modules,
# #     lora_dropout=0.05,
# #     bias="none",
# #     task_type="CAUSAL_LM"
# # )

# # model = get_peft_model(model, lora_config)
# # model.config.use_cache = False

# # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # total = sum(p.numel() for p in model.parameters())
# # print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.6f}%)")

# # # -------------------------------------------------------------
# # # Cell 6: TrainingArguments + Trainer
# # # -------------------------------------------------------------
# # from transformers import TrainingArguments, Trainer

# # PER_DEVICE_TRAIN_BATCH = 12
# # GRADIENT_ACCUM_STEPS = 2
# # NUM_EPOCHS = 1
# # LEARNING_RATE = 2e-4
# # EVAL_STEPS = 15000  # faster eval
# # SAVE_STEPS = 15000
# # LOGGING_STEPS = 1000

# # training_args = TrainingArguments(
# #     output_dir=str(OUTPUT_DIR),
# #     num_train_epochs=NUM_EPOCHS,
# #     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
# #     per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH,
# #     gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
# #     learning_rate=LEARNING_RATE,
# #     fp16=True,
# #     bf16=False,
# #     optim="adamw_bnb_8bit",
# #     warmup_ratio=0.03,
# #     logging_steps=LOGGING_STEPS,
# #     evaluation_strategy="steps",
# #     eval_steps=EVAL_STEPS,
# #     save_strategy="steps",
# #     save_steps=SAVE_STEPS,
# #     save_total_limit=3,
# #     dataloader_num_workers=8,
# #     group_by_length=True,
# #     gradient_checkpointing=True,
# #     remove_unused_columns=False,
# #     torch_compile=False,
# #     report_to="none"
# # )

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=raw_datasets["train"],
# #     eval_dataset=raw_datasets["validation"],
# #     tokenizer=tokenizer,
# #     data_collator=collator
# # )

# # num_train = len(trainer.train_dataset)
# # gpus = torch.cuda.device_count() or 1
# # steps_per_epoch = math.ceil(num_train / (PER_DEVICE_TRAIN_BATCH * gpus * GRADIENT_ACCUM_STEPS))
# # total_steps = steps_per_epoch * NUM_EPOCHS
# # print(f"Num train examples: {num_train:,}; GPUs: {gpus}")
# # print(f"Steps/epoch ≈ {steps_per_epoch:,}; Total steps ≈ {total_steps:,}")

# # # -------------------------------------------------------------
# # # Cell 8: Start Training
# # # -------------------------------------------------------------
# # print("🚀 Starting QLoRA fine-tuning (Q/K/V modules only)...")

# # resume_from_checkpoint = None
# # if "--resume_from_checkpoint" in sys.argv:
# #     idx = sys.argv.index("--resume_from_checkpoint")
# #     if idx + 1 < len(sys.argv):
# #         resume_from_checkpoint = sys.argv[idx + 1]
# #         if os.path.exists(resume_from_checkpoint):
# #             print(f"🔁 Resuming from checkpoint: {resume_from_checkpoint}")
# #         else:
# #             print(f"⚠️ Checkpoint not found: {resume_from_checkpoint}, starting fresh.")

# # start_time = time.time()
# # try:
# #     train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
# #     trainer.save_model(str(OUTPUT_DIR))
# #     trainer.save_state()
# #     print(f"✅ Model + state saved to: {OUTPUT_DIR}")
# # except Exception as e:
# #     print(f"❌ Training crashed: {e}")
# #     raise

# # elapsed = time.time() - start_time
# # h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
# # print(f"✅ Training completed in {h}h {m}m {s}s")
# # print("Train result summary:", train_result)


# #!/usr/bin/env python3
# # ============================================
# # QLoRA Training Script (Q/K/V modules only + 90K sample subset)
# # ============================================

# import os, time, math, sys
# from pathlib import Path
# from dotenv import load_dotenv
# import torch
# from datasets import load_dataset

# # -------------------------------------------------------------
# # Cell 1: Environment Setup
# # -------------------------------------------------------------
# load_dotenv()

# HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
# assert HF_TOKEN, "❌ Missing Hugging Face token! Please check your .env file."

# print("✅ Env ready. CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | Memory total (GB):",
#               round(torch.cuda.get_device_properties(i).total_memory/1e9, 1))
# else:
#     print("⚠️ No GPUs detected — training will be very slow or fail.")

# DATA_DIR = Path("/data/home/anjeshnarwal/LLM_price_predictor/data/finetune")
# TRAIN_PATH = DATA_DIR / "train.jsonl"
# VAL_PATH = DATA_DIR / "val.jsonl"
# OUTPUT_DIR = Path("../src/models/llama31_8b_qlora_qkv_90k")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# seed = 42
# torch.manual_seed(seed)

# # -------------------------------------------------------------
# # Cell 2: Load dataset (sample 90k)
# # -------------------------------------------------------------
# if not TRAIN_PATH.exists() or not VAL_PATH.exists():
#     raise FileNotFoundError(f"❌ Missing dataset!\nExpected:\n - {TRAIN_PATH}\n - {VAL_PATH}")

# print("🔁 Loading JSONL dataset (may take a minute)...")
# raw_datasets = load_dataset(
#     "json",
#     data_files={"train": str(TRAIN_PATH), "validation": str(VAL_PATH)},
#     keep_in_memory=False
# )

# # ✅ Subsample train split to 90k randomly
# full_train_size = len(raw_datasets["train"])
# subset_size = min(50000, full_train_size)
# raw_datasets["train"] = raw_datasets["train"].shuffle(seed=seed).select(range(subset_size))

# print(f"📦 Dataset loaded successfully:")
# print(f"  • Train rows (sampled): {len(raw_datasets['train']):,} / {full_train_size:,}")
# print(f"  • Validation rows: {len(raw_datasets['validation']):,}")

# for i, ex in enumerate(raw_datasets["train"].select(range(3))):
#     prompt = ex.get("prompt", "")[:200].replace("\n", " ")
#     response = ex.get("response", "")
#     print(f"\nSAMPLE {i} prompt (trunc): {prompt}  -> response: {response}")

# # -------------------------------------------------------------
# # Cell 3: Tokenizer + Pre-tokenization
# # -------------------------------------------------------------
# from transformers import AutoTokenizer

# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
# token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_ID,
#     use_fast=False,
#     trust_remote_code=True,
#     **token_kwargs
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

# print("✅ Tokenizer loaded. Vocab size:", len(tokenizer))
# print(f"Pad token: {tokenizer.pad_token} | eos_token: {tokenizer.eos_token}")

# DO_PRETOKENIZE = True
# if DO_PRETOKENIZE:
#     def tokenize_batch(examples):
#         texts = [(str(p or "") + " " + str(r or "")) for p, r in zip(examples.get("prompt", []), examples.get("response", []))]
#         out = tokenizer(texts, truncation=True, padding=False, max_length=512, add_special_tokens=False)
#         return {"input_ids": out["input_ids"]}

#     print("🔁 Pre-tokenizing train split (num_proc=8)...")
#     raw_datasets["train"] = raw_datasets["train"].map(
#         tokenize_batch,
#         batched=True,
#         batch_size=1024,
#         remove_columns=raw_datasets["train"].column_names,
#         num_proc=8
#     )
#     print("🔁 Pre-tokenizing validation split (num_proc=4)...")
#     raw_datasets["validation"] = raw_datasets["validation"].map(
#         tokenize_batch,
#         batched=True,
#         batch_size=1024,
#         remove_columns=raw_datasets["validation"].column_names,
#         num_proc=4
#     )
#     print("✅ Pre-tokenization complete.")

# # -------------------------------------------------------------
# # Cell 4: Data Collator
# # -------------------------------------------------------------
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     current_dir = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from src.collator import DataCollatorForPricePrediction
# collator = DataCollatorForPricePrediction(tokenizer=tokenizer, max_length=512)
# print("🧠 Collator initialized successfully")

# # -------------------------------------------------------------
# # Cell 5: Load Model + PEFT (QKV only)
# # -------------------------------------------------------------
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )

# print("🔧 Loading model in 4-bit ...")

# # ✅ FIX: Ensure model loads on the correct device for training
# local_rank = int(os.environ.get("LOCAL_RANK", -1))
# if torch.cuda.is_available():
#     device_for_load = torch.cuda.current_device()
#     device_map = {"": device_for_load}
#     print(f"⚙️ Loading model onto local CUDA device: cuda:{device_for_load} (device_map={{'': cuda:{device_for_load}}})")
# else:
#     device_map = "cpu"
#     print("⚠️ No CUDA available — loading model on CPU (very slow).")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     device_map=device_map,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     torch_dtype=torch.float16,
#     **token_kwargs
# )

# print("✅ Model loaded in 4-bit.")

# if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
#     model.resize_token_embeddings(len(tokenizer))

# print("⚙️ Preparing model for k-bit training with gradient checkpointing...")
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# # ✅ Convert FP32 to FP16
# for p in model.parameters():
#     if p.dtype == torch.float32:
#         p.data = p.data.to(torch.float16)

# # ✅ Train only Q, K, V projections
# target_modules = ['q_proj', 'k_proj', 'v_proj']

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=target_modules,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, lora_config)
# model.config.use_cache = False

# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total = sum(p.numel() for p in model.parameters())
# print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.6f}%)")

# # -------------------------------------------------------------
# # Cell 6: TrainingArguments + Trainer
# # -------------------------------------------------------------
# from transformers import TrainingArguments, Trainer

# PER_DEVICE_TRAIN_BATCH = 12
# GRADIENT_ACCUM_STEPS = 2
# NUM_EPOCHS = 1
# LEARNING_RATE = 2e-4
# EVAL_STEPS = 15000
# SAVE_STEPS = 15000
# LOGGING_STEPS = 1000

# training_args = TrainingArguments(
#     output_dir=str(OUTPUT_DIR),
#     num_train_epochs=NUM_EPOCHS,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
#     per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH,
#     gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
#     learning_rate=LEARNING_RATE,
#     fp16=True,
#     bf16=False,
#     optim="adamw_bnb_8bit",
#     warmup_ratio=0.03,
#     logging_steps=LOGGING_STEPS,
#     evaluation_strategy="steps",
#     eval_steps=EVAL_STEPS,
#     save_strategy="steps",
#     save_steps=SAVE_STEPS,
#     save_total_limit=3,
#     dataloader_num_workers=8,
#     group_by_length=True,
#     gradient_checkpointing=True,
#     remove_unused_columns=False,
#     torch_compile=False,
#     report_to="none"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=raw_datasets["train"],
#     eval_dataset=raw_datasets["validation"],
#     tokenizer=tokenizer,
#     data_collator=collator
# )

# num_train = len(trainer.train_dataset)
# gpus = torch.cuda.device_count() or 1
# steps_per_epoch = math.ceil(num_train / (PER_DEVICE_TRAIN_BATCH * gpus * GRADIENT_ACCUM_STEPS))
# total_steps = steps_per_epoch * NUM_EPOCHS
# print(f"Num train examples: {num_train:,}; GPUs: {gpus}")
# print(f"Steps/epoch ≈ {steps_per_epoch:,}; Total steps ≈ {total_steps:,}")

# # -------------------------------------------------------------
# # Cell 8: Start Training
# # -------------------------------------------------------------
# print("🚀 Starting QLoRA fine-tuning (Q/K/V modules only)...")

# resume_from_checkpoint = None
# if "--resume_from_checkpoint" in sys.argv:
#     idx = sys.argv.index("--resume_from_checkpoint")
#     if idx + 1 < len(sys.argv):
#         resume_from_checkpoint = sys.argv[idx + 1]
#         if os.path.exists(resume_from_checkpoint):
#             print(f"🔁 Resuming from checkpoint: {resume_from_checkpoint}")
#         else:
#             print(f"⚠️ Checkpoint not found: {resume_from_checkpoint}, starting fresh.")

# start_time = time.time()
# try:
#     train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#     trainer.save_model(str(OUTPUT_DIR))
#     trainer.save_state()
#     print(f"✅ Model + state saved to: {OUTPUT_DIR}")
# except Exception as e:
#     print(f"❌ Training crashed: {e}")
#     raise

# elapsed = time.time() - start_time
# h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
# print(f"✅ Training completed in {h}h {m}m {s}s")
# print("Train result summary:", train_result)
