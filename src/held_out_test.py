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

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "microsoft/deberta-v3-base" 
ADAPTER_PATH = "models/final_deberta_model" # Directory where trainer.save_model() saved the LoRA adapter
TEST_DATA_FILE = "data/held_out.csv" # *** CRITICAL: Ensure this is a new, unseen dataset ***
RESULT_OUTPUT_DIR = "results/unseen_test_results" 

# Set seed for reproducibility
set_seed(42)

# --- Define Labels for Classification (Must match training) ---
labels = ["Benign", "T1003.002", "T1087.001", "T1049"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD TOKENIZER & PREPROCESSING ---
tokenizer = DebertaV2TokenizerFast.from_pretrained(BASE_MODEL_ID)

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
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Loading LoRA adapter from: {ADAPTER_PATH}...")
# Attach the saved LoRA weights to the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
# Merge the adapter weights into the base model weights for final inference/deployment
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
    bf16=True,
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
