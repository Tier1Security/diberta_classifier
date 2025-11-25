import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import PeftModel
import numpy as np
import evaluate
import os
import json
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- CONFIGURATION ---
BASE_MODEL_ID = "roberta-base" 
ADAPTER_PATH = "models/multiclass_roberta_lora" 
TEST_DATA_FILE = "data_3class_security/test.csv" 
RESULT_OUTPUT_DIR = "results/multiclass_test_results" 

set_seed(42)

# --- LABELS (Must match Training) ---
labels = ["Benign", "T1003.002", "T1562"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def main():
    # --- 1. LOAD TOKENIZER ---
    print(f"Loading tokenizer from {ADAPTER_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    except:
        print("Adapter tokenizer not found, loading from base...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # --- 2. LOAD DATASET ---
    print(f"Loading test dataset from {TEST_DATA_FILE}...")
    raw_test_dataset = load_dataset('csv', data_files=TEST_DATA_FILE, split="train")

    # Filter for valid labels only
    raw_test_dataset = raw_test_dataset.filter(lambda example: example['label'] in [0, 1, 2])

    def preprocess_function(examples):
        # Tokenize and keep the label as 'labels' so Trainer can access true labels
        inputs = tokenizer(
            examples["text"], 
            max_length=128, 
            padding=False, 
            truncation=True
        )
        inputs["labels"] = examples["label"]
        return inputs

    # Keep the original 'label' column (so we can map it to 'labels'), remove other columns
    keep_columns = [c for c in raw_test_dataset.column_names if c == 'label']
    processed_test_dataset = raw_test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[c for c in raw_test_dataset.column_names if c not in keep_columns]
    )

    # --- 3. LOAD MODEL ---
    print(f"Loading base model {BASE_MODEL_ID} and adapter from {ADAPTER_PATH}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # --- 4. PREPARE TRAINER ---
    eval_args = TrainingArguments(
        output_dir=RESULT_OUTPUT_DIR,
        per_device_eval_batch_size=64,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # --- 5. RUN PREDICTION ---
    print("\n--- Running Inference on Test Data ---")
    predictions_output = trainer.predict(processed_test_dataset)
    
    # Extract logits and convert to class IDs
    logits = predictions_output.predictions
    predicted_ids = np.argmax(logits, axis=1)
    true_ids = predictions_output.label_ids

    # --- 6. PRINT METRICS ---
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)

    # A. Accuracy
    acc = accuracy_score(true_ids, predicted_ids)
    print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # B. Confusion Matrix
    cm = confusion_matrix(true_ids, predicted_ids)
    print("\nConfusion Matrix:")
    print("-" * 30)
    # Print headers
    print(f"{'':<15} {'Predicted'}")
    print(f"{'Actual':<15} {labels}")
    for i, row in enumerate(cm):
        print(f"{labels[i]:<15} {row}")
    print("-" * 30)

    # C. Detailed Classification Report (Precision, Recall, F1)
    print("\nDetailed Metrics by Class:")
    print("-" * 60)
    print(classification_report(true_ids, predicted_ids, target_names=labels, digits=4))
    print("-" * 60)

    # --- 7. SAVE RESULTS TO FILE ---
    results = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(true_ids, predicted_ids, target_names=labels, output_dict=True)
    }
    
    os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(RESULT_OUTPUT_DIR, "final_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ Results saved to {results_path}")

if __name__ == "__main__":
    main()