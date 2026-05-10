import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import os

# --- CONFIGURATION ---
# CRITICAL CHANGE 1: Switched to RoBERTa-base
BASE_MODEL_ID = "roberta-base"
# CRITICAL CHANGE 2: Updated path to the new BitFit adapter location
ADAPTER_PATH = "models/final_roberta_bitfit_model"
# CRITICAL CHANGE 3: Updated output directory name
MERGED_MODEL_OUTPUT_DIR = "models/merged_roberta_bitfit_model"

# Define Labels for Classification (Must match training)
labels = ["Benign", "T1003.002"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 1. LOAD BASE MODEL ---
print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- 2. LOAD TOKENIZER ---
print(f"Loading tokenizer for {BASE_MODEL_ID}...")
# CRITICAL CHANGE 4: Use AutoTokenizer for RoBERTa
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)


# --- 3. LOAD AND MERGE ADAPTER ---
print(f"Loading BitFit adapter from: {ADAPTER_PATH}...")
# Attach the saved BitFit weights to the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter weights into the base model...")
# Merge the adapter weights into the base model weights
# This creates a single, deployable model structure
model = model.merge_and_unload()
print("Merge complete.")


# --- 4. SAVE MERGED MODEL AND TOKENIZER ---
print(f"Saving merged model to: {MERGED_MODEL_OUTPUT_DIR}...")
os.makedirs(MERGED_MODEL_OUTPUT_DIR, exist_ok=True)
model.save_pretrained(MERGED_MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_DIR)

print(f"\n--- ✅ Merged model and tokenizer saved successfully to {MERGED_MODEL_OUTPUT_DIR} ---")