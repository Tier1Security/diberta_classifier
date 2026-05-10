import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    # CRITICAL: Use AutoTokenizer as RoBERTa's tokenizer class is different
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
# CRITICAL: Correct import for BitFit (with LoRA fallback)
try:
    from peft import BitFitConfig, get_peft_model
    _PEFT_BACKEND = "bitfit"
except Exception:
    # Fall back to LoRA if BitFit isn't available in this PEFT installation
    from peft import LoraConfig, get_peft_model
    _PEFT_BACKEND = "lora"
import numpy as np
import evaluate
import os

# --- 0. VRAM Measurement SETUP ---
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
# CRITICAL CHANGE: Switched to RoBERTa-base
MODEL_ID = "roberta-base" 
TRAIN_DATA_FILE = "data/train.csv"
VAL_DATA_FILE = "data/validation.csv"
# CRITICAL CHANGE: Updated output directory name
OUTPUT_DIR = "models/final_roberta_bitfit_model" 

# Set seed for reproducibility
set_seed(42)

# --- Define Labels for Classification ---
labels = ["Benign", "T1003.002"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
DELIMITER = " ### MITRE_ID: "

# --- 2. LOAD DATASET ---
print(f"Loading datasets from {TRAIN_DATA_FILE} and {VAL_DATA_FILE}...")
raw_datasets = DatasetDict({
    "train": load_dataset('csv', data_files=TRAIN_DATA_FILE, split="train"),
    "validation": load_dataset('csv', data_files=VAL_DATA_FILE, split="train"),
})

print("Filtering dataset for 'Benign' and 'T1003.002' classes...")
raw_datasets = raw_datasets.filter(lambda example: example['label'] in [0, 1])

# --- 3. TOKENIZER & PREPROCESSING ---
# CRITICAL CHANGE: Using AutoTokenizer for RoBERTa
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    # Tokenize the text column
    inputs = tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    
    # The 'label' column is used for training
    inputs["labels"] = examples["label"]
    
    return inputs

print("Preprocessing datasets...")
processed_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)


# --- 4. MODEL LOADING (Sequence Classification) ---
print(f"Loading model for Sequence Classification: {MODEL_ID}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Gradient checkpointing and input grads enabled for PEFT
model.gradient_checkpointing_enable() 
model.enable_input_require_grads() 

# --- 5. PEFT CONFIG (BitFit preferred, LoRA fallback) ---
if _PEFT_BACKEND == "bitfit":
    peft_config = BitFitConfig(
        bias="all",
        task_type="SEQ_CLS",
    )
else:
    # LoRA fallback: small rank to keep compute reasonable
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=None,
        bias="none",
        task_type="SEQ_CLS",
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 6. METRICS CALCULATION ---
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Use 'binary' average for binary classification
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# --- 7. TRAINING ARGUMENTS (Tuned for BitFit) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10, 
    per_device_train_batch_size=64, 
    # CRITICAL: High LR for BitFit to force strong, confident boundary separation
    learning_rate=2.0e-4, 
    weight_decay=0.01, # Reduced WD for less regularization pressure
    bf16=True,
    optim="adamw_torch", # Use a supported AdamW optimizer
    
    # --- Scheduler and Evaluation ---
    lr_scheduler_type='linear',
    warmup_steps=500,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    
    logging_steps=10,
    report_to="none",
    
    max_grad_norm=1.0, 
)

# --- 8. INITIALIZE TRAINER ---
# Early stopping patience is 6 to allow time to exit the F1=0.0 state
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=6, 
    early_stopping_threshold=0.005 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# --- 9. TRAIN ---
print("Starting training for Sequence Classification (RoBERTa-base with BitFit)...")
trainer.train()

# --- 10. VRAM MEASUREMENT REPORTING ---
if device is not None:
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) # Convert bytes to MB
    print(f"\n--- VRAM Usage ---")
    print(f"  Peak VRAM allocated during training: {peak_vram:.2f} MB")


# --- 11. SAVE ---
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Final model saved to {OUTPUT_DIR}")