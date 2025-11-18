# 🛡️ MITRE ATT&CK T1003.002 Classifier (RoBERTa-Base + BitFit)

This model is a fine-tuned **RoBERTa-base** classifier designed to detect the high-risk MITRE ATT&CK technique **T1003.002 — OS Credential Dumping** from system log or command-line snippets.  
It predicts whether an input is **Benign** or **T1003.002 (Malicious)**.

---

## 🌟 Model Performance Summary

The model achieved an overall **Accuracy: 0.9995** on a held-out test set.

### Confusion Matrix

|                       | **Predicted Benign** | **Predicted T1003.002** |
|-----------------------|----------------------|---------------------------|
| **Actual Benign**     | 1923 (TN)            | 2 (FP)                    |
| **Actual T1003.002**  | 0 (FN)               | 1997 (TP)                 |

**Perfect recall (0 False Negatives)** — every malicious sample was detected.

---

## Classification Report

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| Benign                 | 1.0000    | 0.9990 | 0.9995   | 1925    |
| T1003.002 (Malicious)  | 0.9990    | 1.0000 | 0.9995   | 1997    |
| **Macro Average**      | 0.9995    | 0.9995 | 0.9995   | 3922    |
| **Weighted Average**   | 0.9995    | 0.9995 | 0.9995   | 3922    |

---

## 🎯 Classification Task and Labels

| MITRE Technique        | MITRE ID     | Description                                                                 |
|------------------------|--------------|-----------------------------------------------------------------------------|
| OS Credential Dumping  | T1003.002    | Accessing Registry hives (SAM/SECURITY), typically via reg.exe or similar. |

---

## 💻 Technical Implementation

### Base Model & Fine-Tuning Details

| Component       | Value            | Notes                                                                 |
|----------------|------------------|------------------------------------------------------------------------|
| **Base Model** | `roberta-base`   | Robust transformer architecture by Meta.                               |
| **PEFT Method**| BitFit           | Only bias terms are trainable — fast training, tiny checkpoints.       |
| **Precision**  | BF16             | Reduced memory usage + higher training throughput.                      |

### Training Data Distribution

Dataset included **20,000 generated command samples**, including hard negatives.

- **Training:** 80%  
- **Validation:** 10%  
- **Test:** 10%  

---

