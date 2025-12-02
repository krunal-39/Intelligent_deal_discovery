"""
src/prompts.py
Prompt templates for training, validation, and inference
for the LLaMA-8B QLoRA price prediction fine-tuning task.
"""

# -----------------------------
# 🧠 System prompt (base)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that estimates the price of a product "
    "based on its title, category, store, features, description, and details. "
    "Return only the numeric price in USD, formatted with two decimals (e.g., 19.99)."
)

# -----------------------------
# 📘 Training / Validation prompt
# -----------------------------
TRAIN_PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Product Title: {title}\n"
    "Product Category: {category}\n"
    "Store / Brand: {store}\n"
    "Product Features: {features}\n"
    "Product Description: {description}\n"
    "Additional Details: {details}\n\n"
    "Estimate the price (USD) for this product."
)

# -----------------------------
# 🧪 Inference / Testing prompt
# -----------------------------
TEST_PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Product Title: {title}\n"
    "Product Category: {category}\n"
    "Store / Brand: {store}\n"
    "Product Features: {features}\n"
    "Product Description: {description}\n"
    "Additional Details: {details}\n\n"
    "Predict the price in USD (two decimals)."
)

# -----------------------------
# 🧩 Helper function to build prompt
# -----------------------------
def build_prompt(
    title="",
    category="",
    store="",
    features="",
    description="",
    details="",
    mode="train",
):
    """
    mode = "train" | "test"
    Returns formatted prompt text for given mode.
    """
    template = TRAIN_PROMPT_TEMPLATE if mode == "train" else TEST_PROMPT_TEMPLATE
    return template.format(
        system=SYSTEM_PROMPT,
        title=title or "",
        category=category or "",
        store=store or "",
        features=features or "",
        description=description or "",
        details=details or "",
    )


# -----------------------------
# 🧱 Convert a DataFrame row into a JSONL-ready example
# -----------------------------
def row_to_example(row, mode="train"):
    """
    Converts a single DataFrame row into a prompt-response pair.
    Returns None if price is invalid.
    """
    try:
        prompt = build_prompt(
            title=row.get("title", ""),
            category=row.get("category", ""),
            store=row.get("store", ""),
            features=row.get("features", ""),
            description=row.get("description", ""),
            details=row.get("details", ""),
            mode=mode,
        )
        price = float(row.get("price", 0))
        response = f"{price:.2f}"
        return {"prompt": prompt, "response": response}
    except Exception as e:
        print(f"⚠️ Skipping row due to error: {e}")
        return None
