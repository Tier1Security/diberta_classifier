import torch
import torch.nn as nn
import numpy as np
import faiss
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from torch.cuda.amp import autocast

# ==========================================
# 1. Unified Model Architecture (Shared Encoder)
# ==========================================

class SecurityAgentX_Model(nn.Module):
    """
    Unified Architecture: Shared RoBERTa Encoder with 
    dual heads for Classification (Phase 1) and Reconstruction (Phase 2).
    """
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 3):
        super(SecurityAgentX_Model, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        
        # Phase 1 Head: 3 MITRE TTPs (T1003.002, T1562, T1134)
        self.classifier_head = nn.Linear(768, num_labels)
        
        # Phase 2 Head: Autoencoder Decoder
        self.decoder_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :] # CLS token
        
        # Dual Head Output
        logits = self.classifier_head(embedding)
        reconstructed = self.decoder_head(embedding)
        
        return embedding, reconstructed, logits

# ==========================================
# 2. Hybrid Inference Logic
# ==========================================

class HybridSecurityEngine:
    def __init__(self, 
                 model_path="anomaly_engine.pt", # Combined model file
                 atlas_index_path="mitre_atlas.index",
                 metadata_path="mitre_atlas_metadata.csv",
                 device="cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load the Unified Model (3 labels + Anomaly Threshold)
        print("[*] Initializing Unified Security Agent X Model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize with 3 labels as per your Phase 1 training
        self.model = SecurityAgentX_Model(num_labels=3).to(self.device)
        
        # Handle state dict (In case of LoRA or standard training)
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'], strict=False)
            self.threshold = checkpoint.get('threshold', 0.001) # Fallback if not set
        else:
            self.model.load_state_dict(checkpoint)
            self.threshold = 0.001 # Manual default for PoC

        self.model.eval()
        
        # Load Attribution Atlas (FAISS) for Zero-Day mapping
        print("[*] Loading MITRE Attribution Atlas...")
        self.index = faiss.read_index(atlas_index_path)
        self.metadata = pd.read_csv(metadata_path)
        
        # Your specific 3 MITRE labels
        self.labels = ["T1003.002", "T1562", "T1134"]
        print(f"[+] Hybrid Engine Online. Anomaly Threshold: {self.threshold:.8f}")

    def analyze_command(self, command):
        """
        Inference Logic:
        1. Calculate Reconstruction Error (Heuristic Anomaly Check).
        2. If Normal (Error < Threshold) -> Label as Benign.
        3. If Anomaly (Error > Threshold) -> Classify into one of 3 TTPs OR attribute via Atlas.
        """
        cmd_clean = str(command).lower()
        inputs = self.tokenizer(cmd_clean, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            with autocast(dtype=self.dtype):
                emb, rec, logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                
                # Step 1: Anomaly Verification
                error = torch.mean((emb - rec)**2).item()
                is_anomalous = error > self.threshold

                if not is_anomalous:
                    return {
                        "verdict": "BENIGN",
                        "detection_type": "HEURISTIC_MATCH",
                        "reconstruction_error": f"{error:.8f}",
                        "command": command
                    }

                # Step 2: TTP Classification (For the 3 known labels)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                conf, label_idx = torch.max(probs, dim=-1)
                
                label = self.labels[label_idx.item()]
                confidence = conf.item()

                # Step 3: Global Attribution (Is it a Zero-Day?)
                # If classifier is unsure, we use the Vector Atlas (FAISS)
                if confidence < 0.70:
                    query_vec = emb.cpu().float().numpy().astype('float32')
                    distances, indices = self.index.search(query_vec, 1)
                    match_idx = indices[0][0]
                    mitre_match = self.metadata.iloc[match_idx]
                    
                    return {
                        "verdict": "MALICIOUS",
                        "detection_type": "ZERO_DAY_ANOMALY",
                        "mitre_id": mitre_match['mitre_id'],
                        "technique_name": mitre_match['technique_name'],
                        "similarity": f"{(1 - distances[0][0]):.4f}",
                        "command": command
                    }

                return {
                    "verdict": "MALICIOUS",
                    "detection_type": "KNOWN_TTP_MATCH",
                    "mitre_id": label,
                    "confidence": f"{confidence:.2%}",
                    "command": command
                }

if __name__ == "__main__":
    # Note: Use your actual paths from the local directory
    # engine = HybridSecurityEngine(model_path="anomaly_engine.pt")
    pass