-----

# 🛡️ MITRE ATT&CK classifier built on RoBERTa BitFit

This model is a fine-tuned DeBERTa-v3 text classifier designed to detect cybersecurity threats. It analyzes text to categorize it as either benign or a specific MITRE ATT&CK technique, such as credential dumping or process discovery.

## 🌟 Model Performance Summary

This model achieved **perfect classification** on a dedicated, held-out test set, demonstrating robust generalization across the defined MITRE techniques.

--- Confusion Matrix ---
|           |   Predicted Benign |   Predicted T1003.002 |
|:----------|-------------------:|----------------------:|
| Benign    |               1923 |                     2 |
| T1003.002 |                  0 |                  1997 |

--- Classification Report (Per-Class Metrics) ---
              precision    recall  f1-score   support

      Benign     1.0000    0.9990    0.9995      1925
   T1003.002     0.9990    1.0000    0.9995      1997

    accuracy                         0.9995      3922
   macro avg     0.9995    0.9995    0.9995      3922
weighted avg     0.9995    0.9995    0.9995      3922

==================================================

### Inference Performance

The model is highly efficient, utilizing $\text{DeBERTa-v3-base}$ with LoRA for fast inference.

| Metric | Value |
| :--- | :--- |
| **Samples per Second** | $1520.6$ |
| **Runtime** | $1.31 \text{ s}$ |

-----

## 🎯 Classification Task and Labels

The model is trained as a multi-class sequence classifier to assign one of four labels to an input log snippet.

| MITRE Technique | MITRE ID | Description |
| :--- | :--- | :--- |
| Credential Dumping | **T1003.002** |	Focuses on accessing sensitive data stored in Registry Hives (specifically the SAM and SECURITY hives) or the LSA secrets. |
| Account Discovery: Local Account | **T1087.001** | Adversaries may attempt to get a listing of local system accounts. This information can help adversaries determine which local accounts exist on a system to aid in follow-on behavior. |
| System Network Connections Discovery | **T1049** | Adversaries may attempt to get a listing of network connections to or from the compromised system they are currently accessing or from remote systems by querying for information over the network. |

-----

## 💻 Technical Implementation

### Base Model

  * **Model:** `microsoft/deberta-v3-base`
  * **Method:** LoRA (Low-Rank Adaptation) for efficient fine-tuning.
  * **Precision:** $\text{BF}16$ (Bfloat16)

### Training Data Distribution

The model was trained on a dataset of **20,000 generated samples** split as follows:

  * **Training Set:** $80\%$
  * **Validation Set:** $10\%$
  * **Test Set (Held-Out):** $10\%$

-----

## 🚀 Usage and Deployment

The final, merged model is available in the `final_deberta_model` directory and is ready for inference.

### 1\. Requirements

Install the necessary libraries:

```bash
pip install -r requirements.txt
```

### 2\. Python Inference Example

You can load and use the model directly via the Hugging Face `transformers` library:

```python
import torch
from transformers import AutoModelForSequenceClassification, DebertaV2TokenizerFast

# 1. Load the merged model and tokenizer
MODEL_PATH = "./final_deberta_model" 
tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Input Data
log_entry = "cmd.exe /c reg query HKLM\SECURITY\SAM\Domains\Account"

# 3. Predict
inputs = tokenizer(log_entry, return_tensors="pt", max_length=128, truncation=True).to(device)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = torch.argmax(logits, dim=1).item()
predicted_label = model.config.id2label[predicted_class_id]

# Output
print(f"Log: {log_entry}")
print(f"Predicted MITRE ID: {predicted_label}") 
# Example output: Predicted MITRE ID: T1003.002
```

### 3\. API Deployment

For production, the model is served via a lightweight Flask API (see `app.py`):

```bash
POST site.com/api

{
    "text": "The user ran a new command prompt that queried the C:\\Users directory."
}
```

The model automatically manages the `torch_dtype` and runs on the available GPU for low-latency responses (average latency below $2 \text{ ms}$).

-----

## 📚 Project Structure

```
├── .venv/                         # Ignored by .gitignore
├── final_deberta_model/           # **Production-ready merged model weights**
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv                   # Held-out test data
├── app.py                         # Flask API for inference
├── requirements.txt
├── test_evaluation.py             # Script for generating final metrics
└── train_classifier.py            # LoRA fine-tuning script
```
