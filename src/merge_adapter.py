import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from peft import PeftModel
import os

# --- CONFIGURATION ---
BASE_MODEL_ID = "roberta-base"
# Ensure this matches your 4-class training output dir
ADAPTER_PATH = "models/multiclass_roberta_lora" 
# New output for the 4-class model
MERGED_MODEL_OUTPUT = "models/merged_4class_roberta"

# --- CRITICAL: LABELS MUST MATCH TRAINING EXACTLY ---
labels = ["Benign", "T1003.002", "T1562", "T1134"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def main():
    print(f"--- Starting Merge Process (4 Classes) ---")
    
    # 1. Load the Tokenizer
    print(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    except:
        print("Adapter tokenizer not found, loading from base...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 2. Load the Base Model
    print(f"Loading base model: {BASE_MODEL_ID}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels), # Now 4
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 3. Load the Peft Model
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 4. Merge
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # 5. Save
    print(f"Saving merged model to: {MERGED_MODEL_OUTPUT}...")
    model.save_pretrained(MERGED_MODEL_OUTPUT)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT)

    print("--- Merge Complete ---")

if __name__ == "__main__":
    main()