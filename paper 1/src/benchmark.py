import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "roberta-base"
ADAPTER_PATH = "models/final_roberta_bitfit_model" 
HELD_OUT_FILE = "data/held_out.csv" # Path to the unseen evaluation data
LABELS = ["Benign", "T1003.002"] # Map: 0 -> Benign, 1 -> T1003.002

# --- 2. LOAD MODEL AND TOKENIZER ---
def load_model():
    """Loads the base model and the fine-tuned BitFit adapter."""
    try:
        print(f"Loading base model: {BASE_MODEL_ID}...")
        # Load base model structure
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            num_labels=len(LABELS),
            id2label={i: label for i, label in enumerate(LABELS)},
            label2id={label: i for i, label in enumerate(LABELS)},
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"Loading BitFit adapter from: {ADAPTER_PATH}...")
        # Load the adapter weights onto the base model
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval() 
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        print("Model and Tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or adapter: {e}")
        print("Ensure the adapter directory and files exist at the specified path.")
        return None, None

# --- 3. RUN INFERENCE ON DATASET ---
def run_inference(model, tokenizer, df):
    """Runs prediction on all commands in the provided DataFrame."""
    commands = df['text'].tolist()
    batch_size = 64
    predictions = []

    print(f"Running inference on {len(commands)} commands in the held-out dataset...")
    
    # Process data in batches for efficiency
    for i in range(0, len(commands), batch_size):
        batch = commands[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", max_length=128, truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted IDs
        predicted_ids = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(predicted_ids)
        
        if (i + batch_size) % 1000 == 0:
            print(f"Processed {i + batch_size} samples...")

    # Convert numeric predictions back to string labels
    predicted_labels = [LABELS[id] for id in predictions]
    return predicted_labels

# --- 4. CALCULATE AND REPORT METRICS ---
def benchmark_model():
    """Main function to load data, run inference, and print metrics."""
    # Check if data file exists
    if not os.path.exists(HELD_OUT_FILE):
        print(f"ERROR: Held-out file not found at {HELD_OUT_FILE}. Please generate it using data_generator.py.")
        return

    # Load the model
    model, tokenizer = load_model()
    if model is None:
        return

    # Load the dataset
    df = pd.read_csv(HELD_OUT_FILE)
    df['label_name'] = df['label'].apply(lambda x: LABELS[x])
    true_labels = df['label_name'].tolist()
    print(f"Dataset loaded: {len(df)} samples.")

    # Get predictions
    predicted_labels = run_inference(model, tokenizer, df)
    
    # ----------------------------------------------------------------------
    # 5. METRICS REPORTING
    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print("         T1003.002 CLASSIFICATION BENCHMARK")
    print("="*50)

    # 5.1. Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    # 5.2. Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=[f'Predicted {l}' for l in LABELS])
    print("--- Confusion Matrix ---")
    print(cm_df.to_markdown())
    print("\n")
    
    # Interpretation of Confusion Matrix (Crucial for security)
    # T1003.002 (Malicious) is the POSITIVE class.
    # True Positive (TP): Model correctly identifies Malicious (Top-Right cell in the markdown table)
    # False Negative (FN): Model misses Malicious (Benign prediction on Malicious command) (Bottom-Left cell)
    # False Positive (FP): Model flags Benign as Malicious (Top-Right cell)
    
    # 5.3. Classification Report (Precision, Recall, F1)
    report = classification_report(true_labels, predicted_labels, target_names=LABELS, digits=4)
    print("--- Classification Report (Per-Class Metrics) ---")
    print(report)

    # Key Security Metric Interpretation:
    # 1. Recall (T1003.002): High recall means few False Negatives (FN) - minimum missed attacks. CRITICAL.
    # 2. Precision (T1003.002): High precision means few False Positives (FP) - minimum false alarms. IMPORTANT.
    
    print("="*50)


if __name__ == "__main__":
    benchmark_model()