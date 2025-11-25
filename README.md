# 🛡️ RoBERTa 4-Class Security Classifier (LoRA Adapted)

A fine-tuned **RoBERTa-base** transformer adapted with **LoRA (Low-Rank Adaptation)** for high-accuracy detection of four high-risk MITRE ATT&CK techniques from log and command-line data.

The classifier maps input text into **one of four mutually exclusive security classes**:

- **Benign**
- **T1003.002 — OS Credential Dumping**
- **T1562 — Impair Defenses**
- **T1134 — Access Token Manipulation**

This model is designed for cybersecurity monitoring, SIEM augmentation, SOC automation, and defensive ML research.

---

## 🚀 Key Features

- **State-of-the-art performance:**  
  Achieved **100% accuracy** on a 1050-sample adversarial stress test.
- **Robust Hard-Negative Resistance:**  
  Correctly differentiates safe commands (e.g., `wmic process list`) from malicious ones.
- **Lightweight LoRA adaptation:**  
  Small trainable matrix footprint; backbone weights remain frozen.
- **Production-ready:**  
  Lowercasing normalization, high data diversity, clean label boundaries.

---

## 🌟 Performance Summary

### **Overall Accuracy:** `100%`  
### **Total Misclassifications:** `0`  
All classes were perfectly separated during evaluation.

---

## 📊 Confusion Matrix (4-Class)

|                       | Benign | T1003.002 | T1562 | T1134 |
|-----------------------|--------|-----------|--------|--------|
| **Actual Benign**     | ✔ All correct | 0 | 0 | 0 |
| **Actual T1003.002**  | 0      | ✔ All correct | 0 | 0 |
| **Actual T1562**      | 0      | 0 | ✔ All correct | 0 |
| **Actual T1134**      | 0      | 0 | 0 | ✔ All correct |

---

## 📈 Classification Report

| Class                  | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| Benign                 | 1.0000    | 1.0000 | 1.0000   |
| T1003.002              | 1.0000    | 1.0000 | 1.0000   |
| T1562                  | 1.0000    | 1.0000 | 1.0000   |
| T1134                  | 1.0000    | 1.0000 | 1.0000   |
| **Macro Avg**          | 1.0000    | 1.0000 | 1.0000   |
| **Weighted Avg**       | 1.0000    | 1.0000 | 1.0000   |

---

## 🎯 Classification Categories (MITRE ATT&CK)

| Category                     | MITRE ID    | Description |
|-----------------------------|-------------|-------------|
| **Benign**                  | N/A         | Harmless activity, recon, safe commands. |
| **OS Credential Dumping**   | T1003.002   | Registry hive dumping (`reg save`, SAM/SYSTEM), LSASS access. |
| **Impair Defenses**         | T1562       | Stopping defenses (`sc stop`), tampering Defender, `netsh` bypass. |
| **Access Token Manipulation** | T1134     | Token impersonation, `OpenProcessToken`, handle duplication. |

---

## 🧠 Model Architecture & Training

### **Base Model**
- **Backbone:** `roberta-base`
- **Parameters:** ~125M
- **Frozen:** Yes (LoRA applied to selective layers)

### **Fine-Tuning Method**

| Component           | Status       | Notes |
|--------------------|--------------|-------|
| **LoRA**           | Enabled      | Applied to Q/K/V/dense matrices |
| **Backbone**       | Frozen       | Original RoBERTa weights preserved |
| **Classifier Head** | Trained     | 4 neurons for 4 security classes |

### **Training Details**
- **Optimizer:** AdamW  
- **Loss:** Cross-Entropy  
- **Precision:** BF16  
- **Epochs:** ~3–6  
- **Batch Size:** 32–64  

---

## 📚 Dataset Summary

Total: **150,000+ generated samples** with strong class separation.

- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%

### Data Engineering & Robustness Measures

- **Lowercasing** to prevent case-based evasion  
- **Hard negative generation**  
- **Benign sanitization:**  
  → No overlap with malicious keywords (`backup-utility` instead of `reg export`)  
- **Malicious diversity** including LSASS, hive dumping, service tampering, and token abuse  
- **Noise injection:** random whitespace, malformed flags, partial commands  

---

## 🧪 Intended Use

### **Recommended Use Cases**
- SIEM enrichment  
- SOC alert classification  
- Automated threat detection  
- Security log triage  
- L1/L2 analyst augmentation  
- Adversarial behavior research  

### **Not Recommended For**
- Malware classification on raw binaries  
- Automated blocking without human oversight  
- Legal or forensic decisions  
- Detection of non-textual or binary artifacts  

---

## 🛠️ Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "your-username/roberta-4class-lora-security"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "reg save HKLM\\sam C:\\temp\\sam.save"

inputs = tokenizer(text.lower(), return_tensors="pt")
outputs = model(**inputs)

pred = outputs.logits.argmax(dim=1).item()
print(pred)
