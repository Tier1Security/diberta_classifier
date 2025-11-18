import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    DebertaV2TokenizerFast, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import PeftModel, PeftConfig # Import PeftModel for loading
import numpy as np
import evaluate
import os
import json
from sklearn.metrics import confusion_matrix

# --- 0. VRAM Measurement Setup ---
if torch.cuda.is_available():
    print("CUDA is available. VRAM measurement is enabled.")
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    # Get the current device
    device = torch.cuda.current_device()
else:
    print("CUDA is not available. VRAM measurement is disabled.")
    device = None

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "microsoft/deberta-v3-base" 
ADAPTER_PATH = "models/final_deberta_model" # Directory where trainer.save_model() saved the LoRA adapter
TEST_DATA_FILE = "data/test.csv" # *** CRITICAL: Ensure this is a new, unseen dataset ***
RESULT_OUTPUT_DIR = "results/final_test_results" 

# Set seed for reproducibility
set_seed(42)

# --- Define Labels for Classification (Must match training) ---
labels = ["Benign", "T1003.002"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD TOKENIZER & PREPROCESSING ---
# Try to load tokenizer from the adapter/merged model path if available, else fall back to BASE_MODEL_ID
from transformers import AutoTokenizer, AutoConfig
import pathlib

adapter_path_candidate = pathlib.Path(ADAPTER_PATH)
merged_candidate = pathlib.Path("models/merged_roberta_bitfit_model")

if adapter_path_candidate.exists():
    tokenizer_source = str(adapter_path_candidate)
elif merged_candidate.exists():
    tokenizer_source = str(merged_candidate)
else:
    tokenizer_source = BASE_MODEL_ID

print(f"Loading tokenizer from {tokenizer_source}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

def preprocess_function(examples):
    # Tokenize the text column
    inputs = tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    # The 'label' column is used for evaluation
    inputs["labels"] = examples["label"]
    return inputs

# --- 3. LOAD DATASET ---
print(f"Loading test dataset from {TEST_DATA_FILE}...")
# Load the test set. Use split="train" if the CSV is not split in the file itself.
raw_test_dataset = load_dataset('csv', data_files=TEST_DATA_FILE, split="train")

# --- Filter dataset for binary classification ---
print("Filtering dataset for 'Benign' and 'T1003.002' classes...")
raw_test_dataset = raw_test_dataset.filter(lambda example: example['label'] in [0, 1])

print("Preprocessing test dataset...")
# Remove columns not needed by the model's forward method
processed_test_dataset = raw_test_dataset.map(preprocess_function, batched=True, remove_columns=raw_test_dataset.column_names)


# --- 4. LOAD BASE MODEL AND ADAPTER (PEFT) ---
# --- 4. LOAD MODEL (PEFT adapter or merged full model) ---
selected_model_path = None
if adapter_path_candidate.exists():
    selected_model_path = str(adapter_path_candidate)
elif merged_candidate.exists():
    selected_model_path = str(merged_candidate)

if selected_model_path is None:
    print(f"No local model found at {ADAPTER_PATH} or models/merged_roberta_bitfit_model; loading base model: {BASE_MODEL_ID}...")
    # Load base model from the hub
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )
else:
    print(f"Loading model from: {selected_model_path}...")
    # Attempt to load as a PEFT adapter attached to a base model first
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"Attempting to load PEFT adapter from: {selected_model_path}...")
        model = PeftModel.from_pretrained(base_model, selected_model_path)
        model = model.merge_and_unload()
    except Exception as e:
        print(f"PEFT load failed ({e}). Falling back to loading full model from {selected_model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            selected_model_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )

# --- 5. METRICS CALCULATION (Same as training) ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
mcc_metric = evaluate.load("matthews_correlation")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    # Use 'binary' average for the binary classification
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    mcc = mcc_metric.compute(predictions=predictions, references=labels)
    
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:\n{cm}")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
        "matthews_correlation": mcc["matthews_correlation"],
    }

# --- 6. INITIALIZE TRAINER FOR EVALUATION ---
# We only need minimal arguments for evaluation
eval_args = TrainingArguments(
    output_dir=RESULT_OUTPUT_DIR,
    per_device_eval_batch_size=64, # Can be larger for evaluation
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
print(f"  Precision: {results['eval_precision']:.4f}")
print(f"  Recall: {results['eval_recall']:.4f}")
print(f"  F1-Score: {results['eval_f1']:.4f}")
print(f"  Matthews Correlation: {results['eval_matthews_correlation']:.4f}")

# --- 8. VRAM Measurement Reporting ---
if device is not None:
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) # Convert bytes to MB
    print(f"\n--- VRAM Usage ---")
    print(f"  Peak VRAM allocated during evaluation: {peak_vram:.2f} MB")


# --- 9. SAVE RESULTS ---
os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)
# Save the results to a JSON file
results_file = os.path.join(RESULT_OUTPUT_DIR, "final_evaluation_metrics.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n--- ✅ Results saved to {results_file} ---")