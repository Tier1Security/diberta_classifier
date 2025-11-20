import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# PEFT Import Logic
try:
    from peft import BitFitConfig, get_peft_model
    _PEFT_BACKEND = "bitfit"
except Exception:
    from peft import LoraConfig, get_peft_model
    _PEFT_BACKEND = "lora"

import numpy as np
import evaluate
import os

# --- 0. VRAM Measurement SETUP ---
if torch.cuda.is_available():
    print("CUDA is available. VRAM measurement is enabled.")
    torch.cuda.reset_peak_memory_stats()
    device = torch.cuda.current_device()
else:
    print("CUDA is not available. VRAM measurement is disabled.")
    device = None

# --- 1. CONFIGURATION ---
MODEL_ID = "roberta-base" 
# Ensure these point to your new 3-class data files
TRAIN_DATA_FILE = "data_3class/train.csv"
VAL_DATA_FILE = "data_3class/validation.csv"
# CRITICAL CHANGE: New output directory for the multi-class model
OUTPUT_DIR = "models/multiclass_roberta_bitfit" 

set_seed(42)

# --- CRITICAL CHANGE: Define 3 Labels ---
labels = ["Benign", "T1003.002", "T1134"] 
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD DATASET ---
print(f"Loading datasets from {TRAIN_DATA_FILE}...")
raw_datasets = DatasetDict({
    "train": load_dataset('csv', data_files=TRAIN_DATA_FILE, split="train"),
    "validation": load_dataset('csv', data_files=VAL_DATA_FILE, split="train"),
})

print("Filtering dataset for labels 0, 1, and 2...")
# CRITICAL CHANGE: Allow label 2 (T1134) to pass through filter
raw_datasets = raw_datasets.filter(lambda example: example['label'] in [0, 1, 2])

text_column = "text"
label_column = "label"

# --- 3. TOKENIZER & PREPROCESSING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    inputs = tokenizer(
        examples[text_column], 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    inputs["labels"] = examples[label_column]
    return inputs

print("Preprocessing datasets...")
cols_to_remove = [col for col in raw_datasets["train"].column_names if col not in [label_column]]
processed_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True, 
    remove_columns=cols_to_remove
)

# --- 4. MODEL LOADING ---
print(f"Loading model for Multi-Class Classification ({len(labels)} labels)...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels), # This will now be 3
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)

model.gradient_checkpointing_enable() 
model.enable_input_require_grads() 

# --- 5. PEFT CONFIG ---
if _PEFT_BACKEND == "bitfit":
    peft_config = BitFitConfig(bias="all", task_type="SEQ_CLS")
    print("Using BitFit configuration.")
else:
    peft_config = LoraConfig(r=8, lora_alpha=32, bias="none", task_type="SEQ_CLS")
    print("Falling back to LoRA configuration.")

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
    
    # CRITICAL CHANGE: Use 'weighted' average for multi-class
    # 'binary' will crash if there are more than 2 labels.
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# --- 7. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10, 
    per_device_train_batch_size=64, 
    learning_rate=2.0e-4, 
    weight_decay=0.01,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type='linear',
    warmup_steps=500,
    eval_strategy="steps",
    eval_steps=100,       # Increased slightly for larger dataset
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    report_to="none",
    max_grad_norm=1.0, 
)

# --- 8. INITIALIZE TRAINER ---
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5, 
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
print("Starting training for Multi-Class Classification...")
trainer.train()

# --- 10. VRAM REPORT ---
if device is not None:
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"\n--- VRAM Usage ---")
    print(f"  Peak VRAM allocated during training: {peak_vram:.2f} MB")

# --- 11. SAVE ---
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Final model saved to {OUTPUT_DIR}")