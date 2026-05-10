import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
BASE_MODEL_ID = "roberta-base"
ADAPTER_PATH = "models/final_roberta_bitfit_model" 
LABELS = ["Benign", "T1003.002"]

# --- 1. HAND-CRAFTED GENERALIZATION TEST CASES ---
# These commands introduce new paths, wrappers, and case variations not directly
# present in the synthetic dataset used for training/validation/held-out sets.
TEST_COMMANDS = [
    # --------------------------------------------------------------------------
    # MALICIOUS (Test new obfuscation and paths) - MUST be T1003.002 (Label 1)
    # --------------------------------------------------------------------------
    # 1. New path, mixed case, and an atypical flag (/e for export)
    "reg save hklm\\sAm c:\\p\\s.hiv /e", 
    # 2. Extreme case obfuscation on executable and wrapper
    "pOwErSheLl.ExE -c \"reg.exe save hkey_local_machine\\system c:\\temp.dat\"",
    # 3. Security hive saved to a user profile directory (testing path generalization)
    "reg save HKLM\\SECURITY C:\\Users\\Public\\sec.dat /y", 

    # --------------------------------------------------------------------------
    # BENIGN (Test deep queries, new benign saves, and custom commands) - MUST be Benign (Label 0)
    # --------------------------------------------------------------------------
    # 4. Deep, complex, but benign HKLM query (should NOT be flagged)
    "reg query HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USB /s /f", 
    # 5. Benign HKCU save command, wrapped
    "cmd /c reg save HKEY_LOCAL_MACHINE\\Software\\Policies C:\\Backup\\pol.reg",
    # 6. Completely non-registry command (testing boundary against general utilities)
    "netsh firewall set opmode enable",
]

# --- 2. LOAD MODEL AND TOKENIZER ---
print(f"Loading base model: {BASE_MODEL_ID}...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    num_labels=len(LABELS),
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)},
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Loading BitFit adapter from: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() 
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# --- 3. RUN INFERENCE ---
results = []

with torch.no_grad():
    for command in TEST_COMMANDS:
        inputs = tokenizer(command, return_tensors="pt", max_length=128, truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).float().cpu().numpy()[0]
        predicted_id = np.argmax(probs)
        predicted_label = LABELS[predicted_id]
        confidence = probs[predicted_id]
        t1003_confidence = probs[LABELS.index("T1003.002")]
        
        results.append({
            "Command": command,
            "Predicted Label": predicted_label,
            "Confidence": f"{confidence:.4f}",
            "T1003.002 Score": f"{t1003_confidence:.4f}"
        })

# --- 4. DISPLAY RESULTS ---
print("\n--- Final Generalization Prediction Results ---")
df = pd.DataFrame(results)
print(df.to_markdown(index=False))