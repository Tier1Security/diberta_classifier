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
# CRITICAL: Point to the folder created in the previous step
TRAIN_DATA_FILE = "data_4class_security/train.csv"
VAL_DATA_FILE = "data_4class_security/validation.csv"
OUTPUT_DIR = "models/multiclass_roberta_lora" 

set_seed(42)

# --- CRITICAL: Define 3 Labels (Must match Data Generation) ---
# Label 0: Benign
# Label 1: T1003.002 (Registry Hive Dumping)
# Label 2: T1562 (Malicious Firewall / Impair Defenses)
labels = ["Benign", "T1003.002", "T1562", "T1134"]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# --- 2. LOAD DATASET ---
print(f"Loading datasets from {TRAIN_DATA_FILE}...")
raw_datasets = DatasetDict({
    "train": load_dataset('csv', data_files=TRAIN_DATA_FILE, split="train"),
    "validation": load_dataset('csv', data_files=VAL_DATA_FILE, split="train"),
})

print("Filtering dataset for labels 0, 1, 2, and 3...")
raw_datasets = raw_datasets.filter(lambda example: example['label'] in [0, 1, 2, 3])

text_column = "text"
label_column = "label"

# --- 3. TOKENIZER & PREPROCESSING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    return tokenizer(
        examples[text_column], 
        max_length=128, 
        padding=False, # We use DataCollator for dynamic padding
        truncation=True
    )

print("Preprocessing datasets...")
cols_to_remove = [col for col in raw_datasets["train"].column_names if col != label_column]
processed_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True, 
    remove_columns=cols_to_remove
)

# --- 4. MODEL LOADING ---
print(f"Loading model for Multi-Class Classification ({len(labels)} labels)...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# --- 5. PEFT CONFIG (LoRA) ---
# LoRA is preferred over BitFit for learning structural differences 
# between XML logs (Firewall) and CLI commands (Registry)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,               # Rank
    lora_alpha=32,      # Alpha
    lora_dropout=0.1,
    target_modules=["query", "value"] # RoBERTa attention modules
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
    
    # Weighted average is required for multiclass
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
    per_device_train_batch_size=32, 
    gradient_accumulation_steps=2,
    learning_rate=2.0e-4, 
    weight_decay=0.01,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type='linear',
    warmup_steps=500,
    eval_strategy="steps",
    
    # UPDATED: Evaluate roughly once per epoch (750 steps total per epoch)
    eval_steps=500,      
    save_steps=500,
    
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",
)

# Use DataCollator for dynamic padding (Saves VRAM)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 8. INITIALIZE TRAINER ---
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10, 
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
print("Starting training for Multi-Class Classification...")
trainer.train()

# --- 10. SAVE ---
# Save the adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete. Adapter saved to {OUTPUT_DIR}")