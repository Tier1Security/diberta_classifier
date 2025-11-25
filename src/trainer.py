import os
# Keep this to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import evaluate
import shutil

# --- 0. SETUP ---
if torch.cuda.is_available():
    print("CUDA is available.")
    torch.cuda.reset_peak_memory_stats()
else:
    print("CUDA is not available.")

# --- 1. CONFIGURATION ---
MODEL_ID = "roberta-base" 
TRAIN_DATA_FILE = "data/train.csv" # Pointing to your Normalized Data folder
VAL_DATA_FILE = "data/validation.csv"
ADAPTER_DIR = "models/roberta_lora_adapter" 
FINAL_MERGED_DIR = "models/merged_4class_roberta" # MATCHES STRESS TEST PATH

set_seed(42)

labels = ["Benign", "T1003.002", "T1562", "T1134"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD DATASET ---
print(f"Loading datasets from {TRAIN_DATA_FILE}...")
raw_datasets = DatasetDict({
    "train": load_dataset('csv', data_files=TRAIN_DATA_FILE, split="train"),
    "validation": load_dataset('csv', data_files=VAL_DATA_FILE, split="train"),
})

# --- 3. TOKENIZER & PREPROCESSING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    # CRITICAL UPDATE: Enforce lowercase here too for safety
    # This ensures the model never sees a capital letter, even if data gen missed one
    texts = [t.lower() for t in examples["text"]]
    
    return tokenizer(
        texts, 
        max_length=128, 
        padding=False, 
        truncation=True
    )

print("Preprocessing datasets...")
processed_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True, 
    load_from_cache_file=False 
)

# --- 4. MODEL LOADING ---
print(f"Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- 5. PEFT CONFIG ---
# OPTIMIZATION: Added more target modules for better command-line syntax understanding
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,               
    lora_alpha=32,      
    lora_dropout=0.1,
    # Targeting 'key' and 'value' helps with syntax; 
    # 'dense' helps with the classification logic.
    target_modules=["query", "value", "key", "dense"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 6. METRICS ---
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

# --- 7. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=ADAPTER_DIR,
    num_train_epochs=10, # High epochs, let EarlyStopping handle the cut-off
    
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32, 
    gradient_accumulation_steps=2,
    
    learning_rate=2.0e-4, 
    weight_decay=0.01,
    bf16=True, 
    
    optim="adamw_torch", 
    lr_scheduler_type='linear',
    warmup_steps=500, 
    
    eval_strategy="steps",
    eval_steps=100,      
    save_steps=100,
    save_total_limit=2, # Save disk space
    
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 8. TRAINER ---
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5, # Stop if no improvement for 5 evals
    early_stopping_threshold=0.001 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# --- 9. TRAIN ---
print("Starting Training...")
trainer.train()

# --- 10. MERGE & SAVE (CRITICAL STEP) ---
print("Merging LoRA weights with base model...")
# 1. Save Adapter First
model.save_pretrained(ADAPTER_DIR)

# 2. Merge
merged_model = model.merge_and_unload()

# 3. Save Final Model for Stress Testing
print(f"Saving merged model to {FINAL_MERGED_DIR}...")
merged_model.save_pretrained(FINAL_MERGED_DIR)
tokenizer.save_pretrained(FINAL_MERGED_DIR)

print("\n[+] Training and Merging Complete.")
print(f"    You can now run 'python src/stress.py'")