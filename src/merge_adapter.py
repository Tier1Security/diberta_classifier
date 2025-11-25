import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from peft import PeftModel
import os

# --- CONFIGURATION ---
BASE_MODEL_ID = "roberta-base"
# This must match the OUTPUT_DIR from your training script
ADAPTER_PATH = "models/multiclass_roberta_lora" 
# This is where the final standalone model will be saved
MERGED_MODEL_OUTPUT = "models/merged_multiclass_roberta"

# --- CRITICAL: LABELS MUST MATCH TRAINING EXACTLY ---
labels = ["Benign", "T1003.002", "T1562"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def main():
    print(f"--- Starting Merge Process ---")
    
    # 1. Load the Tokenizer
    # We load it from the adapter path (best practice) or base if not found
    print(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    except:
        print("Adapter tokenizer not found, loading from base...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 2. Load the Base Model
    # We MUST provide the label mappings here, or the shape will mismatch
    print(f"Loading base model: {BASE_MODEL_ID}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 3. Load the Peft Model (Apply the Adapter)
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 4. Merge
    print("Merging adapter weights into base model...")
    # This physically adds the LoRA weights to the base weights
    model = model.merge_and_unload()

    # 5. Save the Final Model
    print(f"Saving merged model to: {MERGED_MODEL_OUTPUT}...")
    model.save_pretrained(MERGED_MODEL_OUTPUT)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT)

    print("--- Merge Complete ---")
    print(f"You can now load this model directly using:")
    print(f"AutoModelForSequenceClassification.from_pretrained('{MERGED_MODEL_OUTPUT}')")

if __name__ == "__main__":
    main()