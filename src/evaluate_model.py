import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import PeftModel
import numpy as np
import evaluate
import os
import json
import pathlib
from sklearn.metrics import confusion_matrix

# --- 0. VRAM Measurement Setup ---
if torch.cuda.is_available():
    print("CUDA is available. VRAM measurement is enabled.")
    torch.cuda.reset_peak_memory_stats()
    device = torch.cuda.current_device()
else:
    print("CUDA is not available. VRAM measurement is disabled.")
    device = None

# --- 1. CONFIGURATION ---
# CRITICAL FIX: Switch to RoBERTa to avoid DeBERTa tokenizer errors
BASE_MODEL_ID = "roberta-base" 

# Paths must match your training script's output
ADAPTER_PATH = "models/multiclass_roberta_bitfit" 
MERGED_MODEL_PATH = "models/merged_multiclass_roberta"

# Ensure this points to your 3-class test file
TEST_DATA_FILE = "data_3class/test.csv" 
RESULT_OUTPUT_DIR = "results/multiclass_test_results" 

set_seed(42)

# --- CRITICAL: Define 3 Labels (Must match training) ---
labels = ["Benign", "T1003.002", "T1134"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD TOKENIZER ---
# Logic to find the best tokenizer source
adapter_path_candidate = pathlib.Path(ADAPTER_PATH)
merged_candidate = pathlib.Path(MERGED_MODEL_PATH)

if merged_candidate.exists():
    tokenizer_source = str(merged_candidate)
elif adapter_path_candidate.exists():
    tokenizer_source = str(adapter_path_candidate)
else:
    tokenizer_source = BASE_MODEL_ID

print(f"Loading tokenizer from {tokenizer_source}...")
# RoBERTa tokenizer loads easily without the sentencepiece errors
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    inputs["labels"] = examples["label"]
    return inputs

# --- 3. LOAD DATASET ---
print(f"Loading test dataset from {TEST_DATA_FILE}...")
raw_test_dataset = load_dataset('csv', data_files=TEST_DATA_FILE, split="train")

print("Filtering dataset for labels 0, 1, and 2...")
# CRITICAL CHANGE: Filter for all 3 classes
raw_test_dataset = raw_test_dataset.filter(lambda example: example['label'] in [0, 1, 2])

print("Preprocessing test dataset...")
processed_test_dataset = raw_test_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=raw_test_dataset.column_names
)

# --- 4. LOAD MODEL (Logic: Merged -> Adapter -> Base) ---
if merged_candidate.exists():
    print(f"Found merged model at {MERGED_MODEL_PATH}. Loading full model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MERGED_MODEL_PATH,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto"
    )
elif adapter_path_candidate.exists():
    print(f"Found adapter at {ADAPTER_PATH}. Loading base + adapter...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
else:
    print(f"⚠️ No local models found. Loading untrained base model: {BASE_MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto"
    )

# --- 5. METRICS CALCULATION ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # CRITICAL CHANGE: Use 'weighted' for multi-class
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):\n{cm}")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# --- 6. INITIALIZE TRAINER ---
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
    compute_metrics=compute_metrics,
    eval_dataset=processed_test_dataset,
)

# --- 7. RUN EVALUATION ---
print("\n--- Running Final Evaluation on Unseen Test Data ---")
results = trainer.evaluate()

print("\n--- Evaluation Results ---")
print(f"  Accuracy: {results['eval_accuracy']:.4f}")
print(f"  Precision (Weighted): {results['eval_precision']:.4f}")
print(f"  Recall (Weighted): {results['eval_recall']:.4f}")
print(f"  F1-Score (Weighted): {results['eval_f1']:.4f}")

# --- 8. VRAM Measurement ---
if device is not None:
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) 
    print(f"\n--- VRAM Usage ---")
    print(f"  Peak VRAM allocated: {peak_vram:.2f} MB")

# --- 9. SAVE RESULTS ---
os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)
results_file = os.path.join(RESULT_OUTPUT_DIR, "final_evaluation_metrics.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n--- ✅ Results saved to {results_file} ---")