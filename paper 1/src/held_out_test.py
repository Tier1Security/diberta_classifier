import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    DebertaV2TokenizerFast, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    AutoTokenizer,
    AutoConfig,
)
import pathlib
from peft import PeftModel, PeftConfig # Import PeftModel for loading
import numpy as np
import evaluate
import os
import json

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "microsoft/deberta-v3-base" 
ADAPTER_PATH = "models/final_deberta_model" # Directory where trainer.save_model() saved the LoRA adapter
TEST_DATA_FILE = "data/held_out.csv" # *** CRITICAL: Ensure this is a new, unseen dataset ***
RESULT_OUTPUT_DIR = "results/unseen_test_results" 

# Set seed for reproducibility
set_seed(42)

# --- Define Labels for Classification (auto-detect from merged model if available) ---
adapter_path_candidate = pathlib.Path(ADAPTER_PATH)
merged_candidate = pathlib.Path("models/merged_roberta_bitfit_model")

# Default fallback label set (used if we can't detect from model)
default_labels = ["Benign", "T1003.002", "T1087.001", "T1049"]

if merged_candidate.exists():
    try:
        cfg = AutoConfig.from_pretrained(str(merged_candidate))
        if getattr(cfg, "id2label", None):
            # cfg.id2label keys may be strings
            id2label = {int(k): v for k, v in cfg.id2label.items()}
            labels = [id2label[i] for i in sorted(id2label.keys())]
            label2id = {label: i for i, label in enumerate(labels)}
        else:
            labels = default_labels
            id2label = {i: label for i, label in enumerate(labels)}
            label2id = {label: i for i, label in enumerate(labels)}
    except Exception:
        labels = default_labels
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
else:
    labels = default_labels
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD TOKENIZER & PREPROCESSING ---
# Try to load tokenizer from local model directories if present
adapter_path_candidate = pathlib.Path(ADAPTER_PATH)
merged_candidate = pathlib.Path("models/merged_roberta_bitfit_model")

if adapter_path_candidate.exists():
    tokenizer_source = str(adapter_path_candidate)
elif merged_candidate.exists():
    tokenizer_source = str(merged_candidate)
else:
    tokenizer_source = BASE_MODEL_ID

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

print("Preprocessing test dataset...")
# Remove columns not needed by the model's forward method
processed_test_dataset = raw_test_dataset.map(preprocess_function, batched=True, remove_columns=raw_test_dataset.column_names)


# --- 4. LOAD BASE MODEL AND ADAPTER (PEFT) ---
print(f"Loading base model: {BASE_MODEL_ID}...")
# Choose dtype/device options depending on CUDA availability
use_cuda = torch.cuda.is_available()
torch_dtype = (torch.bfloat16 if use_cuda else torch.float32)
device_map = "auto" if use_cuda else None

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch_dtype,
    device_map=device_map,
)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
# Attach the saved LoRA weights to the base model if adapter exists, else try loading full model
if adapter_path_candidate.exists():
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
elif merged_candidate.exists():
    model = AutoModelForSequenceClassification.from_pretrained(
        str(merged_candidate),
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
else:
    # Fall back to attempting to load adapter path (may raise)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()

# --- 5. METRICS CALCULATION (Same as training) ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    # Use 'weighted' average for the multiclass classification
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# --- 6. INITIALIZE TRAINER FOR EVALUATION ---
# We only need minimal arguments for evaluation
eval_args = TrainingArguments(
    output_dir=RESULT_OUTPUT_DIR,
    per_device_eval_batch_size=64, # Can be larger for evaluation
    bf16=True if torch.cuda.is_available() else False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=eval_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 7. EVALUATE ON TEST SET ---
print("Starting final evaluation on held-out test set...")
results = trainer.evaluate(eval_dataset=processed_test_dataset)

# --- 8. SAVE AND DISPLAY RESULTS ---
# Manually add the model preparation time to the results
# results["eval_model_preparation_time"] = model_prep_time

# Save metrics to a JSON file
output_eval_file = os.path.join(RESULT_OUTPUT_DIR, "final_evaluation_metrics.json")
with open(output_eval_file, "w") as writer:
    json.dump(results, writer, indent=4)

print("\n--- ✅ FINAL TEST SET METRICS ---")
for key, value in results.items():
    # Format floats to 4 decimal places for cleaner output
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
print("----------------------------------")

print(f"Evaluation complete. Results saved to {output_eval_file}")
