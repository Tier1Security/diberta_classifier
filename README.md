🛡️ MITRE ATT&CK T1003.002 Classifier (RoBERTa-Base + BitFit)

This model is a fine-tuned RoBERTa-base classifier designed to detect the high-risk MITRE ATT&CK technique T1003.002 — OS Credential Dumping from system log or command-line snippets.
It predicts whether an input is Benign or T1003.002 (Malicious).

🌟 Model Performance Summary

The model achieved an overall Accuracy: 0.9995 on a held-out test set, demonstrating strong generalization on both benign and malicious samples.

Confusion Matrix
	Predicted Benign	Predicted T1003.002
Actual Benign	1923 (TN)	2 (FP)
Actual T1003.002	0 (FN)	1997 (TP)

Perfect recall (0 False Negatives) — every malicious sample was successfully detected.

Classification Report
Class	Precision	Recall	F1-Score	Support
Benign	1.0000	0.9990	0.9995	1925
T1003.002 (Malicious)	0.9990	1.0000	0.9995	1997
Macro Average	0.9995	0.9995	0.9995	3922
Weighted Average	0.9995	0.9995	0.9995	3922
🎯 Classification Task and Labels

A binary sequence classification task assigning one of two labels:

MITRE Technique	MITRE ID	Description
OS Credential Dumping	T1003.002	Accessing Registry hives (SAM/SECURITY) often using reg.exe or similar.
💻 Technical Implementation
Base Model & Fine-Tuning Details
Component	Value	Notes
Base Model	roberta-base	Robust transformer architecture by Facebook/Meta.
PEFT Method	BitFit	Only bias terms are trainable — fast training, tiny checkpoint size.
Precision	BF16	Reduced memory usage + increased training speed.
Training Data Distribution

Dataset included 20,000 high-quality generated samples, with injected hard negatives to limit false positives.

Training: 80%

Validation: 10%

Test (Held-Out): 10%

🚀 Usage & Deployment

The merged, production-ready model is located in:

final_roberta_model/

1. Requirements
pip install -r requirements.txt

2. Python Inference Example
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load the merged model and tokenizer
MODEL_PATH = "./final_roberta_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Input Example (T1003.002 attempt)
log_entry = "cmd.exe /c reg query HKLM\\SECURITY\\SAM\\Domains\\Account"

# 3. Predict
inputs = tokenizer(log_entry, return_tensors="pt", max_length=128, truncation=True).to(device)

with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.softmax(logits, dim=1)
confidence_score = torch.max(probabilities).item()
predicted_class_id = torch.argmax(probabilities, dim=1).item()
predicted_label = model.config.id2label[predicted_class_id]

print(f"Log: {log_entry}")
print(f"Predicted MITRE ID: {predicted_label} (Confidence: {confidence_score:.4f})")


Example output:

Predicted MITRE ID: T1003.002 (Confidence: 0.9990)

3. API Deployment

The model is served through a lightweight Flask API:

POST https://site.com/api/predict


JSON Body:

{
    "text": "The user ran a new command prompt that queried the C:\\Users directory."
}


GPU acceleration is automatically used when available.

📚 Project Structure
├── .venv/                      # Ignored virtual environment
├── final_roberta_model/        # Production-ready merged model
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv                # Held-out test data
├── app.py                      # Flask API for inference
├── requirements.txt
├── test_evaluation.py          # Evaluation + metrics generation
└── train_classifier.py         # BitFit fine-tuning script
